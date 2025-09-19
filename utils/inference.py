

def run_inference(x, framework, data_type):
    if framework == "tensorflow":
        from models.tensorflow_models import get_model
        model = get_model(data_type)
        preds = model.predict(x)
    else:
        from models.pytorch_models import get_model
        model = get_model(data_type)
        import torch
        with torch.no_grad():
            preds = model(x).cpu().numpy()
    return preds.tolist()
