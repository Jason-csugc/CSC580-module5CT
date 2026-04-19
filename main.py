"""Tox21 training and evaluation pipeline.

This module trains and evaluates two models for the first Tox21 task:
an RF baseline and a fully connected neural network. Data loading and
featurization are delegated to DeepChem's MolNet loader with ECFP features.

Workflow summary:
    1. Load train/validation/test splits from DeepChem Tox21.
    2. Select task index 0 labels and corresponding sample weights.
    3. Train and evaluate a weighted random forest baseline.
    4. Run a grid search over neural network hyperparameters.
    5. Retrain the best neural model and evaluate on test data.
    6. Plot train/validation loss curves.

Run:
    python3 main.py
"""

import os
import logging

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for consistent performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from datetime import datetime
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import deepchem as dc

# Reproducibility
SEED = 456
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# Dataset configuration
TASK_INDEX = 0
INPUT_DIM = 1024


def prepare_data():
    """Load Tox21 splits and extract arrays for task index 0.

    Uses DeepChem MolNet's ``load_tox21`` with ECFP featurization, then
    selects the first task from the multitask label and weight matrices.

    Returns
    -------
    tuple
        (train_X, train_y, valid_X, valid_y, test_X, test_y,
         train_w, valid_w, test_w), where each y/w array is 1-D for task 0.
    """
    _, (train, valid, test), _ = dc.molnet.load_tox21(featurizer='ECFP')

    train_X, train_y, train_w = train.X, train.y, train.w
    valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
    test_X, test_y, test_w = test.X, test.y, test.w

    train_y = train_y[:, 0]
    valid_y = valid_y[:, 0]
    test_y = test_y[:, 0]

    train_w = train_w[:, 0]
    valid_w = valid_w[:, 0]
    test_w = test_w[:, 0]

    return train_X, train_y, valid_X, valid_y, test_X, test_y, train_w, valid_w, test_w


def build_random_forest_model(n_estimators=50, random_state=SEED):
    """Build a weighted random forest classifier.
    
    Parameters
    ----------
    n_estimators : int, optional
        Number of trees in the forest. Default is 50.
    random_state : int, optional
        Random seed for reproducibility. Default is SEED.
    
    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Configured random forest classifier.
    """
    return RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=random_state)


def train_random_forest_model(model, train_X, train_y, train_w):
    """Train the random forest model on weighted training data.
    
    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        Random forest classifier to train.
    train_X : np.ndarray of shape (n_train, 1024)
        Training feature matrix.
    train_y : np.ndarray of shape (n_train,)
        Training labels.
    train_w : np.ndarray of shape (n_train,)
        Per-sample training weights.
    
    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Trained random forest model.
    """
    model.fit(train_X, train_y, sample_weight=train_w)
    return model


def evaluate_random_forest_model(model, train_X, valid_X, test_X, train_y, valid_y, test_y, train_w, valid_w, test_w):
    """Evaluate weighted random forest accuracy on all splits.
    
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
        (train_accuracy, valid_accuracy, test_accuracy)
        using weighted ``accuracy_score`` for each split.
    """
    train_y_pred = model.predict(train_X)
    valid_y_pred = model.predict(valid_X)
    test_y_pred = model.predict(test_X)
    train_accuracy = accuracy_score(train_y, train_y_pred, sample_weight=train_w)
    valid_accuracy = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
    test_accuracy = accuracy_score(test_y, test_y_pred, sample_weight=test_w)

    return train_accuracy, valid_accuracy, test_accuracy


def build_model(hidden_units=50, layers=1, learning_rate=0.001, dropout_rate=0.5):
    """Build and compile a feed-forward binary classifier.
    
    Architecture:
        Input(INPUT_DIM) -> [Dense(hidden_units, relu) + Dropout] x layers
        -> Dense(1, sigmoid)
    
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
    model = keras.Sequential([
        keras.layers.Input(shape=(INPUT_DIM,)),
    ])

    for _ in range(layers):
        model.add(keras.layers.Dense(hidden_units, activation='relu'))
        model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model

def train_model(model, train_X, train_y, valid_X, valid_y, train_w, valid_w, n_epochs=10, batch_size=100):
    """Train the neural network with weighted training and validation.
    
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
    train_w : np.ndarray of shape (n_train,)
        Per-sample weights for training examples.
    valid_w : np.ndarray of shape (n_valid,)
        Per-sample weights for validation examples.
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

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,        
        write_graph=True,        
        write_images=True,
        update_freq='epoch'      
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
    """Compute weighted binary accuracy for model predictions.
    
    Parameters
    ----------
    model : tf.keras.Sequential
        Trained model to evaluate.
    valid_X : np.ndarray of shape (n_valid, 1024)
        Validation feature matrix.
    valid_y : np.ndarray of shape (n_valid,)
        Validation labels.
    valid_w : np.ndarray of shape (n_valid,)
        Validation sample weights.
    threshold : float, optional
        Probability cutoff for converting sigmoid output to class label.
        Default is 0.5.
    
    Returns
    -------
    float
        Classification accuracy (0-1).
    """
    valid_y_pred = (model.predict(valid_X) > threshold).astype(int)
    accuracy = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)

    return accuracy


def plot_loss_curve(history):
    """Plot train and validation loss from a Keras History object.
    
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

def eval_tox21_hyperparams(train_X, train_y, valid_X, valid_y, train_w, valid_w, nodes_hidden, layers, learning_rate, dropout_rate, epochs, batch_size, repeats=3):
    """Score one neural hyperparameter configuration over repeated runs.

    Each repeat trains a fresh model with the provided hyperparameters and
    reports weighted validation accuracy. The function returns the mean score
    across repeats for use in grid search.

    Parameters
    ----------
    train_X, valid_X : np.ndarray
        Feature matrices for training and validation.
    train_y, valid_y : np.ndarray
        Binary labels for training and validation.
    train_w, valid_w : np.ndarray
        Per-sample weights for training and validation.
    nodes_hidden : int
        Units per hidden dense layer.
    layers : int
        Number of hidden layers.
    learning_rate : float
        Adam learning rate.
    dropout_rate : float
        Dropout probability after each hidden layer.
    epochs : int
        Number of training epochs per repeat.
    batch_size : int
        Mini-batch size per optimizer step.
    repeats : int, optional
        Number of repeated trainings for averaging. Default is 3.

    Returns
    -------
    float
        Mean weighted validation accuracy across repeats.
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
    """Run the end-to-end Tox21 experiment.

    The function loads data, evaluates a random forest baseline, performs
    neural hyperparameter grid search on validation data, retrains the best
    neural model, evaluates on the test split, and plots loss curves.
    """
    train_X, train_y, valid_X, valid_y, test_X, test_y, train_w, valid_w, test_w = prepare_data()

    rf_model = build_random_forest_model()
    rf_model = train_random_forest_model(rf_model, train_X, train_y, train_w)
    rf_train_acc, rf_valid_acc, rf_test_acc = evaluate_random_forest_model(rf_model, train_X, valid_X, test_X, train_y, valid_y, test_y, train_w, valid_w, test_w)
    print(f"Random Forest -\n Weighted Train Accuracy: {rf_train_acc:.4f}\n Weighted Validation Accuracy: {rf_valid_acc:.4f}\n Weighted Test Accuracy: {rf_test_acc:.4f}\n")

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
                            avg_acc = eval_tox21_hyperparams(train_X, train_y, valid_X, valid_y, train_w, valid_w, h, l, lr, dr, e, bs)
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
