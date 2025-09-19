import tensorflow as tf

_models = {}

def get_model(data_type):
    if data_type not in _models:
        if data_type == "tabular":
            model = tf.keras.models.load_model("saved_models/tf_tabular")
        elif data_type == "image":
            model = tf.keras.models.load_model("saved_models/tf_image")
        else:
            model = tf.keras.models.load_model("saved_models/tf_audio")
        _models[data_type] = model
    return _models[data_type]
