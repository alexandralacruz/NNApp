import tensorflow as tf

_models = {}

def get_model(data_type):
    if data_type not in _models:
        if data_type == "tabular":
            model = tf.keras.models.load_model("models/saved/tf_tabular.keras")
        elif data_type == "image":
            model = tf.keras.models.load_model("models/saved/tf_image.keras")
        else:
            model = tf.keras.models.load_model("models/saved/tf_audio.keras")
        _models[data_type] = model
    return _models[data_type]
