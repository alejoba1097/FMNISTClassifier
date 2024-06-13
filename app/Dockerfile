FROM python:3.10-slim

WORKDIR /src

COPY src/ .

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "app:fmnist_app", "--host", "0.0.0.0", "--port", "8080"]