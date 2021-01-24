# Coordicate Descent SVM Classifier

This package is an implementation of SVM Classifier (SVC) with coordinate descent optimization method.
It also implements SVM with L-BFGS-B optimizer for test purposes.

To build package, run:

    python setup.py bdist wheel

Also scripts for testing are available:
- [test.py](test.py) - calculates accuracy on test/train dataset and value of loss function (and it's prime) 
after each iteration and dumps it into CSV file. It also measures time of training and saves it to .time file,
- [generate_plots.py](generate_plots.py) - generates plots basing results from `test.py`
   
There is also Docker image prepared running multiple tests. To build image, run

    docker build --tag <image-tag> <path-to-directory-with-dockerfile>
    
All results are saved to `/results` directory inside image. To receive them, firstly create container on builded image

    docker run <image-tag>
    
copy created container id and copy results directory

    docker cp <container-id>:/results <destination-directory>
    
# Testing
Implementation of SVM with L-BFGS-B and coordinate descent were tested on several generated datasets:
- with 50 features and 10000 samples,
- with 50 features and 100000 samples,
- with 1000 features and 10000 samples,
- with 1000 features and 100000 samples.

Tests were run in Docker container defined in [Dockerfile](Dockerfile). 
To see results, check [results](results) directory.
