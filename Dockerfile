FROM python:3.7

# copy project
COPY . .

# build package
RUN python setup.py sdist bdist_wheel

# install builded package
RUN pip install dist/*.whl

# install test dependencies
RUN pip install -r requirements-test.txt

# create directory for results
RUN mkdir results

# define arguments for all test cases
ARG SEED=123456789
ARG LITTLE_FEATURES=50
ARG MANY_FEATURES=1000
ARG LITTLE_SAMPLES=10000
ARG MANY_SAMPLES=100000

# run test case for data set with little features and little samples using L-BFGS-B
RUN ./test.py \
--features $LITTLE_FEATURES \
--samples $LITTLE_SAMPLES \
--random-seed $SEED \
--method lbfgsb \
--output results/lbfgsb_${LITTLE_FEATURES}_${LITTLE_SAMPLES}.csv

# run test case for data set with little features and little samples using Coordinate Descent
RUN ./test.py \
--features $LITTLE_FEATURES \
--samples $LITTLE_SAMPLES \
--random-seed $SEED \
--method coordinate-descent \
--output results/coord_desc_${LITTLE_FEATURES}_${LITTLE_SAMPLES}.csv

# run test case for data set with little features and many samples using L-BFGS-B
RUN ./test.py \
--features $LITTLE_FEATURES \
--samples $MANY_SAMPLES \
--random-seed $SEED \
--method lbfgsb \
--output results/lbfgsb_${LITTLE_FEATURES}_${MANY_SAMPLES}.csv

# run test case for data set with little features and many samples using Coordinate Descent
RUN ./test.py \
--features $LITTLE_FEATURES \
--samples $MANY_SAMPLES \
--random-seed $SEED \
--method coordinate-descent \
--output results/coord_desc_${LITTLE_FEATURES}_${MANY_SAMPLES}.csv

# run test case for data set with many features and little samples using L-BFGS-B
RUN ./test.py \
--features $MANY_FEATURES \
--samples $LITTLE_SAMPLES \
--random-seed $SEED \
--method lbfgsb \
--output results/lbfgsb_${MANY_FEATURES}_${LITTLE_SAMPLES}.csv

# run test case for data set with many features and little samples using Coordinate Descent
RUN ./test.py \
--features $MANY_FEATURES \
--samples $LITTLE_SAMPLES \
--random-seed $SEED \
--method coordinate-descent \
--output results/coord_desc_${MANY_FEATURES}_${LITTLE_SAMPLES}.csv

# run test case for data set with many features and many samples using L-BFGS-B
RUN ./test.py \
--features $MANY_FEATURES \
--samples $MANY_SAMPLES \
--random-seed $SEED \
--method lbfgsb \
--output results/lbfgsb_${MANY_FEATURES}_${MANY_SAMPLES}.csv

# run test case for data set with many features and many samples using Coordinate Descent
RUN ./test.py \
--features $MANY_FEATURES \
--samples $MANY_SAMPLES \
--random-seed $SEED \
--method coordinate-descent \
--output results/coord_desc_${MANY_FEATURES}_${MANY_SAMPLES}.csv

# generate plots for performed tests
RUN ./generate_plots.py --directory results
