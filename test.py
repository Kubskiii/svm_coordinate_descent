#!/usr/local/bin/python
import sys
import getopt
import sklearn.datasets as skd
import sklearn.model_selection as skms
import sklearn.metrics as skm

from coordinate_descent_svc import CoordinateDescentSVC, lbfgsbSVM


methods_dict = {
    "coordinate-descent": CoordinateDescentSVC(),
    "lbfgsb": lbfgsbSVM()
}


def test_method(method_name, samples, features, seed=None):
    assert method_name in methods_dict.keys(), "unknown method"
    assert type(samples) == int and samples > 0, "samples must be positive integer"
    assert type(features) == int and features > 0, "features must be positive integer"

    X, y = skd.make_classification(
        n_classes=2,
        n_samples=samples,
        n_features=features,
        random_state=seed
    )
    X_train, X_test, y_train, y_test = skms.train_test_split(X, y, train_size=0.8)
    method = methods_dict.get(method_name)
    model = method.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return skm.accuracy_score(y_test, y_pred)


def usage():
    msg = """
Usage: test [OPTION]
Test performance of selected method over selected data set.
    -f,    --features=NUMBER    Number of test data features
    -s,    --samples=NUMBER     Number of test data samples
    -m,    --method=NAME        Name of the method
"""
    print(msg)


def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hf:s:m:r:",
            ["help", "features=", "samples=", "method=", "random-seed="]
        )
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    method_name = features = samples = seed = None
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-f", "--features"):
            features = int(a)
        elif o in ("-s", "--samples"):
            samples = int(a)
        elif o in ("-m", "--method"):
            method_name = a
        elif o in ("-r", "--random-seed"):
            seed = int(a)
        else:
            assert False, "unhandled option"
    assert features, "features must be provided"
    assert samples, "samples must be provided"
    assert method_name, "method must be provided"
    print(test_method(method_name, samples, features, seed))


if __name__ == "__main__":
    main()
