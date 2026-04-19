"""Toxicology in the 21st Century (Tox21) Neural Network Model.

This module implements a fully-connected neural network with dropout regularization
to predict toxicity outcomes from molecular features. It loads the Tox21 dataset,
computes Morgan circular fingerprints (ECFP) using RDKit, trains a Keras model
on the processed molecular features, and evaluates performance on validation data.

The pipeline:
    1. Downloads and loads the Tox21 dataset from S3
    2. Converts SMILES strings to 1024-bit Morgan fingerprints
    3. Splits data into train/validation/test sets
    4. Trains a 2-layer neural network with dropout
    5. Evaluates and visualizes training curves

Typical Usage:
    python main.py
"""

# from ast import Add
import os
import logging


# Suppress noisy RDKit warnings for deprecated Morgan internals.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for consistent performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import csv
from datetime import datetime
import gzip
import urllib.request
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

RDLogger.DisableLog('rdApp.*')

# Reproducibility
SEED = 456
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

DATA_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
DATA_FILE = './tox21.csv.gz'
TASK_INDEX = 0
INPUT_DIM = 1024


def download_tox21(filename=DATA_FILE, url=DATA_URL):
    """Download Tox21 dataset if not already present.
    
    Parameters
    ----------
    filename : str, optional
        Path where the dataset file will be saved. Default is DATA_FILE.
    url : str, optional
        URL to download the dataset from. Default is DATA_URL.
    
    Returns
    -------
    str
        Path to the downloaded dataset file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    return filename


def smiles_to_ecfp(smiles, radius=2, n_bits=INPUT_DIM):
    """Convert SMILES string to extended connectivity circular fingerprint (ECFP).
    
    Uses RDKit to parse SMILES and compute Morgan fingerprints, which are
    converted to dense numpy arrays.
    
    Parameters
    ----------
    smiles : str
        SMILES representation of a molecule.
    radius : int, optional
        Radius (number of hops) for Morgan fingerprint. Default is 2.
    n_bits : int, optional
        Number of bits in the fingerprint. Default is INPUT_DIM (1024).
    
    Returns
    -------
    np.ndarray or None
        Dense fingerprint array of shape (n_bits,) and dtype float32.
        Returns None if SMILES parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bitvect = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    ConvertToNumpyArray(bitvect, arr)
    return arr


def load_tox21_features(data_file=None):
    """Load and featurize the Tox21 dataset.
    
    Downloads the gzip-compressed Tox21 CSV file from S3, parses SMILES strings,
    computes Morgan fingerprints, and extracts multi-task labels and weights.
    Skips molecules that fail to parse.
    
    Parameters
    ----------
    data_file : str, optional
        Path to dataset file. If None, uses DATA_FILE constant.
    
    Returns
    -------
    tuple of (X, y, w)
        X : np.ndarray of shape (n_samples, 1024) and dtype float32
            Morgan fingerprints for each molecule.
        y : np.ndarray of shape (n_samples, n_tasks) and dtype float32
            Binary labels for each task (0-1).
        w : np.ndarray of shape (n_samples, n_tasks) and dtype float32
            Weights indicating label presence (0=missing, 1=present).
    """
    if data_file is None:
        data_file = DATA_FILE
    data_file = download_tox21(data_file)
    with gzip.open(data_file, 'rt') as f:
        reader = csv.DictReader(f)
        task_names = [name for name in reader.fieldnames if name not in ('mol_id', 'smiles')]
        X, y, w = [], [], []
        for row in reader:
            fingerprint = smiles_to_ecfp(row['smiles'])
            if fingerprint is None:
                continue

            labels = []
            weights = []
            for target in task_names:
                value = row[target].strip()
                if value == '':
                    labels.append(0)
                    weights.append(0)
                else:
                    labels.append(int(float(value)))
                    weights.append(1)

            X.append(fingerprint)
            y.append(labels)
            w.append(weights)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(w, dtype=np.float32)


def split_dataset(X, y, w, valid_fraction=0.1, test_fraction=0.1):
    """Split dataset into train, validation, and test sets.
    
    Shuffles data with reproducible random seed and partitions into
    disjoint train/validation/test splits.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples, n_tasks)
        Label matrix.
    w : np.ndarray of shape (n_samples, n_tasks)
        Weight matrix.
    valid_fraction : float, optional
        Fraction of data to allocate to validation. Default is 0.1.
    test_fraction : float, optional
        Fraction of data to allocate to test. Default is 0.1.
    
    Returns
    -------
    tuple
        (train_X, train_y, train_w, valid_X, valid_y, valid_w, test_X, test_y, test_w)
    """
    n = len(X)
    indices = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)

    n_valid = int(np.floor(n * valid_fraction))
    n_test = int(np.floor(n * test_fraction))
    n_train = n - n_valid - n_test

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]

    return (
        X[train_idx], y[train_idx], w[train_idx],
        X[valid_idx], y[valid_idx], w[valid_idx],
        X[test_idx], y[test_idx], w[test_idx],
    )


def prepare_data():
    """Load, featurize, and prepare the Tox21 dataset.
    
    Orchestrates the full data preparation pipeline: loads features and labels,
    splits into train/validation/test sets, extracts the first toxicity task,
    and removes samples with missing labels.
    
    Returns
    -------
    tuple of (train_X, train_y, valid_X, valid_y, test_X, test_y)
        Each element is a numpy array with samples filtered to have valid labels.
        X arrays have shape (n_samples, 1024).
        y arrays have shape (n_samples,).
    """
    X, y, w = load_tox21_features()
    train_X, train_y, train_w, valid_X, valid_y, valid_w, test_X, test_y, test_w = split_dataset(X, y, w)

    train_y = train_y[:, TASK_INDEX]
    valid_y = valid_y[:, TASK_INDEX]
    test_y = test_y[:, TASK_INDEX]
    train_w = train_w[:, TASK_INDEX]
    valid_w = valid_w[:, TASK_INDEX]
    test_w = test_w[:, TASK_INDEX]

    train_mask = train_w != 0
    valid_mask = valid_w != 0
    test_mask = test_w != 0

    train_X, train_y, train_w = train_X[train_mask], train_y[train_mask], train_w[train_mask]
    valid_X, valid_y, valid_w = valid_X[valid_mask], valid_y[valid_mask], valid_w[valid_mask]
    test_X, test_y, test_w = test_X[test_mask], test_y[test_mask], test_w[test_mask]

    print(f"Train X shape: {train_X.shape}, Train y shape: {train_y.shape}")
    print(f"Validation X shape: {valid_X.shape}, Validation y shape: {valid_y.shape}")
    print(f"Test X shape: {test_X.shape}, Test y shape: {test_y.shape}")

    return train_X, train_y, valid_X, valid_y, test_X, test_y, train_w, valid_w, test_w


def build_random_forest_model(n_estimators=50, random_state=SEED):
    """Build a random forest classifier.
    
    Parameters
    ----------
    n_estimators : int, optional
        Number of trees in the forest. Default is 50.
    max_depth : int, optional
        Maximum depth of each tree. Default is None (nodes are expanded until all leaves are pure).
    random_state : int, optional
        Random seed for reproducibility. Default is SEED.
    
    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Configured random forest classifier.
    """
    return RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=random_state)


def train_random_forest_model(model, train_X, train_y):
    """Train the random forest model on the training data.
    
    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        Random forest classifier to train.
    train_X : np.ndarray of shape (n_train, 1024)
        Training feature matrix.
    train_y : np.ndarray of shape (n_train,)
        Training labels.
    
    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Trained random forest model.
    """
    model.fit(train_X, train_y)
    return model


def evaluate_random_forest_model(model, train_X, valid_X, test_X, train_y, valid_y, test_y, train_w, valid_w, test_w):
    """Evaluate the random forest model on validation and test data.
    
    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        Trained random forest model to evaluate.
    train_X : np.ndarray of shape (n_train, 1024)
        Training feature matrix.
    valid_X : np.ndarray of shape (n_valid, 1024)
        Validation feature matrix.
    test_X : np.ndarray of shape (n_test, 1024)
        Test feature matrix.
    train_y, valid_y, test_y : np.ndarray of shape (n_split,)
        True labels for each split.
    train_w, valid_w, test_w : np.ndarray of shape (n_split,)
        Sample weights for each split.
    
    Returns
    -------
    tuple
        (train_accuracy, valid_accuracy, test_accuracy, feature_importances)
    """
    train_y_pred = model.predict(train_X)
    valid_y_pred = model.predict(valid_X)
    test_y_pred = model.predict(test_X)
    train_accuracy = accuracy_score(train_y, train_y_pred, sample_weight=train_w)
    valid_accuracy = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
    test_accuracy = accuracy_score(test_y, test_y_pred, sample_weight=test_w)
    feature_importances = model.feature_importances_

    return train_accuracy, valid_accuracy, test_accuracy, feature_importances


def build_model(hidden_units=50, layers=1, learning_rate=0.001, dropout_rate=0.5):
    """Build and compile a fully-connected neural network with dropout.
    
    Architecture:
        Input (1024) -> Dense (hidden_units, relu) -> Dropout -> Dense (1, sigmoid)
    
    Parameters
    ----------
    hidden_units : int, optional
        Number of units in the hidden layer. Default is 50.
    layers : int, optional
        Number of hidden layers. Default is 1.
    learning_rate : float, optional
        Adam optimizer learning rate. Default is 0.001.
    dropout_rate : float, optional
        Dropout regularization rate. Default is 0.5.
    
    Returns
    -------
    tf.keras.Sequential
        Compiled model ready for training.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_DIM,)),
    ])

    for _ in range(layers):
        model.add(tf.keras.layers.Dense(hidden_units, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model

def train_model(model, train_X, train_y, valid_X, valid_y, train_w, valid_w, n_epochs=10, batch_size=100):
    """Train the neural network model.
    
    Fits the model on training data with validation monitoring.
    
    Parameters
    ----------
    model : tf.keras.Sequential
        Compiled model to train.
    train_X : np.ndarray of shape (n_train, 1024)
        Training feature matrix.
    train_y : np.ndarray of shape (n_train,)
        Training labels.
    valid_X : np.ndarray of shape (n_valid, 1024)
        Validation feature matrix.
    valid_y : np.ndarray of shape (n_valid,)
        Validation labels.
    n_epochs : int, optional
        Number of training epochs. Default is 10.
    batch_size : int, optional
        Batch size for mini-batch training. Default is 100.
    
    Returns
    -------
    tf.keras.callbacks.History
        Training history containing loss and accuracy metrics.
    """
    log_dir = "logs/tox21/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,        # Enables weight histograms
        write_graph=True,        # Logs model graph
        write_images=True,
        update_freq='epoch'      # Log every epoch
    )
    
    history = model.fit(
        train_X,
        train_y,
        sample_weight=train_w,
        validation_data=(valid_X, valid_y, valid_w),
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[tensorboard_callback]
    )

    print(f"\nTensorBoard logs saved to: {log_dir}")

    return history


def evaluate_model(model, valid_X, valid_y, valid_w, threshold=0.5):
    """Evaluate binary classification accuracy on validation data.
    
    Parameters
    ----------
    model : tf.keras.Sequential
        Trained model to evaluate.
    valid_X : np.ndarray of shape (n_valid, 1024)
        Validation feature matrix.
    valid_y : np.ndarray of shape (n_valid,)
        Validation labels.
    
    Returns
    -------
    float
        Classification accuracy (0-1).
    """
    valid_y_pred = (model.predict(valid_X) > threshold).astype(int)
    accuracy = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)

    return accuracy


def plot_loss_curve(history):
    """Plot training and validation loss curves.
    
    Parameters
    ----------
    history : tf.keras.callbacks.History
        Training history object containing loss metrics.
    
    Returns
    -------
    None
        Displays plot via matplotlib.pyplot.show().
    """
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def evalhyperparams(train_X, train_y, valid_X, valid_y, train_w, valid_w, nodes_hidden, layers, learning_rate, dropout_rate, epochs, batch_size, repeats=3):
    """Evaluate different hyperparameter configurations for the neural network model.
    
    This function can be used to systematically explore the impact of various
    hyperparameters (e.g., hidden_units, learning_rate, dropout_rate) on model
    performance. It trains and evaluates models with different settings and
    summarizes results in a table or plot.
    
    Returns
    -------
    float
        Average weighted validation accuracy over all repeats.
    """
    scores = []

    for r in range(repeats):
        print(f"Repeat {r+1}/{repeats}")

        model = build_model(hidden_units=nodes_hidden, layers=layers, learning_rate=learning_rate, dropout_rate=dropout_rate)

        train_model(model, train_X, train_y, valid_X, valid_y, train_w, valid_w, n_epochs=epochs, batch_size=batch_size)

        score = evaluate_model(model, valid_X, valid_y, valid_w)
        scores.append(score)
        print(f"Validation Accuracy: {score:.4f}")

    avg_score = np.mean(scores)
    print(f"Average Validation Accuracy over {repeats} repeats: {avg_score:.4f}")

    return avg_score



def main():
    """Main entry point for the Tox21 prediction pipeline.
    
    Orchestrates the full pipeline:
        1. Prepare data
        2. Build model
        3. Train model
        4. Evaluate on validation set
        5. Visualize training curves
    
    Returns
    -------
    None
    """
    train_X, train_y, valid_X, valid_y, test_X, test_y, train_w, valid_w, test_w = prepare_data()

    # Build Forest model
    rf_model = build_random_forest_model()
    rf_model = train_random_forest_model(rf_model, train_X, train_y)
    rf_train_acc, rf_valid_acc, rf_test_acc, rf_feature_importances = evaluate_random_forest_model(rf_model, train_X, valid_X, test_X, train_y, valid_y, test_y, train_w, valid_w, test_w)
    print(f"Random Forest - Weighted Train Accuracy: {rf_train_acc:.4f}\n Weighted Validation Accuracy: {rf_valid_acc:.4f}\n Weighted Test Accuracy: {rf_test_acc:.4f}\n")


    nodes_hidden_units = [50, 100]
    layers = [1, 2]
    learning_rates = [0.001, 0.01]
    dropout_rates = [0.3, 0.5]
    epochs = [10, 20]
    batch_sizes = [32, 64]

    best_accuracy = 0
    best_params = None

    for h in nodes_hidden_units:
        for l in layers:
            for lr in learning_rates:
                for dr in dropout_rates:
                    for e in epochs:
                        for bs in batch_sizes:
                            print(f"Evaluating: hidden_units={h}, layers={l}, learning_rate={lr}, dropout_rate={dr}, epochs={e}, batch_size={bs}")
                            avg_acc = evalhyperparams(train_X, train_y, valid_X, valid_y, train_w, valid_w, h, l, lr, dr, e, bs)
                            if avg_acc > best_accuracy:
                                best_accuracy = avg_acc
                                best_params = (h, l, lr, dr, e, bs)

    print(f"Best Hyperparameters: hidden_units={best_params[0]}, layers={best_params[1]}, learning_rate={best_params[2]}, dropout_rate={best_params[3]}, epochs={best_params[4]}, batch_size={best_params[5]} with Validation Accuracy: {best_accuracy:.4f}")


    print("\nEvaluating best hyperparameters on test set...")
    best_model = build_model(hidden_units=best_params[0], layers=best_params[1], learning_rate=best_params[2], dropout_rate=best_params[3])
    history = train_model(best_model, train_X, train_y, valid_X, valid_y, train_w, valid_w, n_epochs=best_params[4], batch_size=best_params[5])
    test_accuracy = evaluate_model(best_model, test_X, test_y, test_w)
    print(f"Test Accuracy with best hyperparameters: {test_accuracy:.4f}")

    plot_loss_curve(history)

if __name__ == '__main__':
    main()
