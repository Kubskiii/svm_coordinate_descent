FROM python:3.7

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY coordinate_descent_svc coordinate_descent_svc

COPY *.py ./

RUN mkdir results

ARG SEED=1234

RUN ./test.py -f 10 -s 1000 -r $SEED -m lbfgsb -o results/lbfgsb_10_1000.csv

RUN ./test.py -f 10 -s 1000 -r $SEED -m coordinate-descent -o results/coord_desc_10_1000.csv

RUN ./test.py -f 10 -s 100000 -r $SEED -m lbfgsb -o results/lbfgsb_10_100000.csv

RUN ./test.py -f 10 -s 100000 -r $SEED -m coordinate-descent -o results/coord_desc_10_100000.csv

RUN ./test.py -f 1000 -s 1000 -r $SEED -m lbfgsb -o results/lbfgsb_1000_1000.csv

RUN ./test.py -f 1000 -s 1000 -r $SEED -m coordinate-descent -o results/coord_desc_1000_1000.csv

RUN ./test.py -f 1000 -s 100000 -r $SEED -m lbfgsb -o results/lbfgsb_1000_100000.csv

RUN ./test.py -f 1000 -s 100000 -r $SEED -m coordinate-descent -o results/coord_desc_1000_100000.csv

RUN ./generate_plots.py -d results
