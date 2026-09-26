import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.smo import SMOBinary, SMOOneVsOne, SMOOneVsRest


@pytest.mark.parametrize(
    "kernel,kwargs",
    [
        ("linear", {}),
        ("poly", {"gamma": 0.2, "degree": 2, "coef0": 1.0}),
        ("rbf", {"gamma": 0.2}),
    ],
)
def test_binary_agrees_with_libsvm(kernel, kwargs):
    X, y0 = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        class_sep=1.5,
        random_state=7,
    )
    X = StandardScaler().fit_transform(X)
    y = np.where(y0 == 1, 1.0, -1.0)
    own = SMOBinary(C=1.0, kernel=kernel, tol=1e-4, **kwargs).fit(X, y)
    ref = SVC(C=1.0, kernel=kernel, tol=1e-6, **kwargs).fit(X, y)

    assert own.equality_residual_ < 1e-8
    assert own.kkt_gap_ <= 2e-3
    assert np.mean(own.predict(X) == ref.predict(X)) >= 0.98


def test_explicit_multiclass_wrappers():
    X, y = make_blobs(n_samples=90, centers=3, cluster_std=0.7, random_state=12)
    X = StandardScaler().fit_transform(X)
    params = dict(C=2.0, kernel="rbf", gamma=0.5, tol=1e-3)
    for model in (SMOOneVsRest(**params), SMOOneVsOne(**params)):
        pred = model.fit(X, y).predict(X)
        assert np.mean(pred == y) > 0.95
