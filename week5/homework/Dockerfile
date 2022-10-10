FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system

COPY ["credit-service.py", "model_C=1.0.bin", "./"]

EXPOSE 9898

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9898", "credit-service:app"]