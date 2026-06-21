FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 user
USER user

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
