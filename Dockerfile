FROM python:3.7

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY coordinate_descent_svc coordinate_descent_svc

COPY test.py .

RUN features=10; samples=10000; seed=123; \
acc=$(./test.py -m coordinate-descent -f $features -s $samples -r $seed); \
echo $features, $samples, $acc >> results.csv
