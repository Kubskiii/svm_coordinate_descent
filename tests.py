from coordinate_descent_svc import CoordinateDescentSVC
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import accuracy_score


def test_sample_coordinate_decent_binary_classification():
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data
    y = bunch.target
    model = CoordinateDescentSVC().fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc}")
    assert acc > 0.5


def test_sample_coordinate_decent_nonbinary_classification():
    bunch = load_iris(as_frame=True)
    X = bunch.data
    y = bunch.target
    model = CoordinateDescentSVC().fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc}")
    assert acc > 0.5