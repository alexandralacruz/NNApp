import streamlit as st
import requests
import pandas as pd

FASTAPI_URL = "http://localhost:8000"

st.title("Demo Multimodal con TensorFlow / PyTorch")

data_type = st.selectbox("Tipo de dato", ["tabular", "image", "audio"])
framework = st.selectbox("Framework", ["tensorflow", "pytorch"])
mode = st.radio("Modo de operación", ["Entrenar", "Predecir"])   

if mode == "Entrenar" and data_type == "tabular":
    st.subheader("Entrenamiento")
    csv_file = st.file_uploader("Sube CSV de entrenamiento", type="csv")
    epochs = st.slider("Épocas", 1, 100, 20)
    if csv_file and st.button("Entrenar"):
        files = {"csv_file": (csv_file.name, csv_file.getvalue(), "text/csv")}
        data = {"framework": framework, "data_type": data_type, "epochs": str(epochs)}
        with st.spinner("Entrenando el modelo..."):
            res = requests.post(f"{FASTAPI_URL}/train/", files=files, data=data)
        st.write("Status:", res.status_code)
        st.write("Raw response:", res.text)

        st.write(res.json())

elif mode == "Predecir":
    if data_type == "tabular":
        st.write("Sube un CSV limpio:")
        file = st.file_uploader("CSV", type="csv")
        if file and st.button("Predecir"):
            df = pd.read_csv(file)
            payload = {
                "data_type": "tabular",
                "framework": framework,
                "tabular_data": df.to_json(orient="records")
            }
            r = requests.post(f"{FASTAPI_URL}/predict/", data=payload)
            st.write(r.json())
    else:
        file = st.file_uploader("Sube archivo", type=["png","jpg","wav","mp3"])
        if file and st.button("Predecir"):
            data = {"data_type": data_type, "framework": framework}
            files = {"file": (file.name, file.getvalue())}
            r = requests.post(f"{FASTAPI_URL}/predict/", data=data, files=files)
            st.write(r.json())
