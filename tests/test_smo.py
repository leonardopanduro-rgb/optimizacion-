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


def test_degenerate_pair_with_zero_curvature():
    # Dos puntos idénticos con etiquetas opuestas dan eta = 0. El paso debe ir
    # al extremo del intervalo en vez de descartar el par.
    X = np.array([[0.0], [0.0], [2.0]])
    y = np.array([-1.0, 1.0, -1.0])
    own = SMOBinary(C=1.0, kernel="linear").fit(X, y)
    ref = SVC(C=1.0, kernel="linear", tol=1e-8).fit(X, y)
    alpha_ref = np.zeros(len(y))
    alpha_ref[ref.support_] = np.abs(ref.dual_coef_[0])

    assert own.converged_
    assert own.kkt_gap_ <= own.tol
    assert own.equality_residual_ < 1e-12
    assert np.allclose(own.alpha_, alpha_ref, atol=1e-6)
    assert own.dual_objective_ == pytest.approx(2.0)
    assert own.termination_reason_ == "converged"


@pytest.mark.parametrize("C", [1.0, 1e17])
def test_small_positive_curvature_keeps_quadratic_solution(C):
    X = np.array([[0.0], [1e-8]])
    y = np.array([-1.0, 1.0])
    # eta=1e-16. D(a)=2*a-eta*a**2/2: máximo en min(C, 2/eta).
    eta = float((X[1, 0] - X[0, 0]) ** 2)
    expected_alpha = min(C, 2.0 / eta)
    expected_dual = 2.0 * expected_alpha - 0.5 * eta * expected_alpha**2
    own = SMOBinary(C=C, kernel="linear", max_iter=1).fit(X, y)

    np.testing.assert_allclose(own.alpha_, expected_alpha, rtol=1e-12, atol=0)
    assert own.dual_objective_ == pytest.approx(expected_dual, rel=1e-12)
    assert own.dual_objective_ > 0.0  # El valor inicial era cero.
    assert own.equality_residual_ == 0.0
    assert own.kkt_gap_ <= own.tol
    assert own.converged_  # Un solo paso puede bastar aun con max_iter=1.
    assert own.termination_reason_ == "converged"


def test_skipped_small_step_does_not_certify_convergence():
    X = np.array([[-1e8], [0.0], [1e-8]])
    y = np.array([-1.0, 1.0, -1.0])
    own = SMOBinary(C=1.0, kernel="linear").fit(X, y)

    # El umbral de pasos deja una violación sin resolver. Un candidato
    # posterior sin violación no debe ocultarla con converged_=True.
    assert own.kkt_gap_ == pytest.approx(1.0)
    assert own.kkt_gap_ > own.tol
    assert not own.converged_
    assert own.termination_reason_ == "stalled"


def test_iteration_limit_does_not_certify_convergence():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    own = SMOBinary(C=1.0, kernel="linear", max_iter=1).fit(X, y)

    assert own.n_iter_ == 1
    assert own.kkt_gap_ > own.tol
    assert not own.converged_
    assert own.termination_reason_ == "max_iter"
