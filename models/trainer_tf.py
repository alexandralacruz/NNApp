# app/models/trainer_tf.py
import tensorflow as tf
from .tensorflow_arch import build_tabular_model
from pathlib import Path

def train_tabular(X_train, y_train, X_test, y_test, epochs=20):
    model = build_tabular_model(input_dim=X_train.shape[1])
    model.fit(X_train, y_train, epochs=epochs,
              validation_data=(X_test, y_test))
    save_dir = Path("models/saved/tf_tabular.keras")
    model.save(save_dir)
    return save_dir
