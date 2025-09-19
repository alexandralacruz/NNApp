from fastapi import FastAPI, UploadFile, File, Form
from utils.inference import run_inference
from utils.preprocess import preprocess_tabular, preprocess_image, preprocess_audio
from pathlib import Path
from utils.data import load_and_split
from models import trainer_tf, trainer_pt
app = FastAPI()

@app.post("/predict/")
async def predict(
    data_type: str = Form(...),          # 'tabular' | 'image' | 'audio'
    framework: str = Form(...),          # 'tensorflow' | 'pytorch'
    tabular_data: str = Form(None),      # CSV o JSON
    file: UploadFile = File(None)        # Imagen o audio
):
    # Preprocesar según tipo
    if data_type == "tabular":
        x = preprocess_tabular(tabular_data)
    elif data_type == "image":
        x = preprocess_image(await file.read())
    elif data_type == "audio":
        x = preprocess_audio(await file.read())
    else:
        return {"error": "Tipo de dato no soportado"}

    result = run_inference(x, framework, data_type)
    return {"prediction": result}

@app.post("/train")
async def train_model(
    csv_file: UploadFile,
    framework: str = Form(...),
    epochs: int = Form(20)
):
    temp_path = Path("uploads") / csv_file.filename
    with open(temp_path, "wb") as f:
        f.write(await csv_file.read())

    X_train, X_test, y_train, y_test = load_and_split(temp_path)

    if framework.lower() == "pytorch":
        model_path = trainer_pt.train_tabular(X_train, y_train, X_test, y_test, epochs)
    else:
        model_path = trainer_tf.train_tabular(X_train, y_train, X_test, y_test, epochs)

    return {"status": "ok", "saved_model": str(model_path)}
