FROM apache/airflow:2.10.0-python3.8

# Set environment variables for Airflow
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False

# Install system dependencies as root
USER root
RUN apt-get update \
    && apt-get install -y \
    gcc \
    libmysqlclient-dev \
    libpq-dev \
    && apt-get clean

# Switch back to airflow user and install Python packages
USER airflow
RUN pip install --no-cache-dir \
    apache-airflow-providers-redis==2.0.0 \
    apache-airflow-providers-mysql==2.0.0 \
    apache-airflow-providers-postgres==2.0.0 \
    psycopg2-binary==2.9.3 \
    mlflow==2.4.0 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    scikit-learn==1.0.2 \
    PyMySQL==1.1.2 \
    great-expectations==0.18.12 \
    sqlalchemy==1.4.53 \
    apache-airflow==2.10.0 \
    xgboost==1.7.6 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    redis==5.0.0 \
    pydantic==1.10.2 \
    evidently==0.4.15

