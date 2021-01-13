FROM python:3.7

WORKDIR ~

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY tests.py .

COPY coordinate_descent_svc coordinate_descent_svc