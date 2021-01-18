from coordinate_descent_svc import CoordinateDescentSVC
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def test_sample_coordinate_decent_binary_classification():
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data
    y = bunch.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = CoordinateDescentSVC().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    assert acc > 0.5