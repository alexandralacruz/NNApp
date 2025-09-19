import torch

_models = {}

def get_model(data_type):
    if data_type not in _models:
        # Importar tu clase de modelo según el tipo
        if data_type == "tabular":
            from .pytorch_arch import TabularNet
            model = TabularNet()
            model.load_state_dict(torch.load("saved_models/pt_tabular.pt"))
        elif data_type == "image":
            from .pytorch_arch import ImageCNN
            model = ImageCNN()
            model.load_state_dict(torch.load("saved_models/pt_image.pt"))
        else:
            from .pytorch_arch import AudioCNN
            model = AudioCNN()
            model.load_state_dict(torch.load("saved_models/pt_audio.pt"))
        model.eval()
        _models[data_type] = model
    return _models[data_type]
