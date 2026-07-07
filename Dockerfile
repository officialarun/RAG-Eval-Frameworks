FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY raglens/ ./raglens/
COPY dashboard/ ./dashboard/
COPY README.md ./

RUN pip install --no-cache-dir -e .

# Hugging Face Spaces (Docker SDK) expects the app on port 7860.
EXPOSE 7860

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
