
# Install the viraual environment
> python -m venv .venv

# Activate the venv
> .\.venv\Scripts\activate

# install requirements packages
> pip intall -r .\requirements.txt

# Abre un terminal y ejecutas el Backend:
> uvicorn app.main:app --reload --port 8000

# abre otro terminal y ejecutas el frontned
> streamlit run ui/app.py