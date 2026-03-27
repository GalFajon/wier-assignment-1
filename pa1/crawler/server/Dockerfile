FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --default-timeout=100 -r requirements.txt
RUN apt-get update
RUN apt-get install make
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
