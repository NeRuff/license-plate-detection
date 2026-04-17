FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 wget && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model_impl.py .
COPY cli.py .
COPY runs/ ./runs/
RUN mkdir -p data
ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]
