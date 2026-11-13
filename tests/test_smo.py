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


# --- Ejemplos resueltos a mano en el informe (Hito II) ---------------------


def test_hand_example_hard_margin_four_points():
    # Sección III-B: w=(1,0), b=-1, margen 1, P=D=1/2.
    X = np.array([[2.0, 0.0], [2.0, 1.0], [0.0, 0.0], [0.0, 1.0]])
    y = np.array([1.0, 1.0, -1.0, -1.0])
    own = SMOBinary(C=1e6, kernel="linear", tol=1e-9).fit(X, y)
    w = (own.alpha_ * y) @ X
    np.testing.assert_allclose(w, [1.0, 0.0], atol=1e-8)
    assert own.b_ == pytest.approx(-1.0, abs=1e-8)
    assert own.dual_objective_ == pytest.approx(0.5, abs=1e-10)
    assert own.converged_


def test_hand_example_smo_two_points():
    # Sección V-G: una iteración da alpha=(1/2,1/2), w=1, b=0, D=1/2.
    X = np.array([[1.0], [-1.0]])
    y = np.array([1.0, -1.0])
    own = SMOBinary(C=1.0, kernel="linear", tol=1e-12).fit(X, y)
    np.testing.assert_allclose(own.alpha_, [0.5, 0.5], atol=1e-14)
    assert own.b_ == pytest.approx(0.0, abs=1e-14)
    assert own.dual_objective_ == pytest.approx(0.5)
    assert own.n_iter_ <= 2
    # La historia global empieza en m-M=2 y termina en 0.
    assert own.gap_history_[0] == pytest.approx(2.0)
    assert own.gap_history_[-1] == pytest.approx(0.0, abs=1e-14)
    assert own.pointwise_kkt_violation() < 1e-12


@pytest.mark.parametrize(
    "C, w_expected, b_expected",
    [
        (0.2, 0.5, 0.0),       # tramo C < 2/5: w = 5C/2, b = 1 - 5C
        (1.0, 1.0, -1.0),      # tramo 2/5 <= C <= 2: candidato A es óptimo
        (4.0, 2.0, -1.0),      # tramo 2 <= C < 8: w = C/2; frontera en u = 1/2
        (6.0, 3.0, -1.0),      # frontera en u = 1/3; x5 bien clasificado
        (20.0, 4.0, -1.0),     # C >= 8: margen duro
    ],
)
def test_hand_example_soft_margin_path_in_C(C, w_expected, b_expected):
    # Sección IV-C (reemplazo): cinco puntos y camino de la solución en C.
    X = np.array([[2.0, 0.0], [2.0, 1.0], [0.0, 0.0], [0.0, 1.0], [0.5, 0.0]])
    y = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    own = SMOBinary(C=C, kernel="linear", tol=1e-10).fit(X, y)
    ref = SVC(C=C, kernel="linear", tol=1e-10).fit(X, y)
    w = (own.alpha_ * y) @ X
    np.testing.assert_allclose(w, [w_expected, 0.0], atol=1e-6)
    np.testing.assert_allclose(ref.coef_[0], [w_expected, 0.0], atol=1e-5)
    assert own.b_ == pytest.approx(b_expected, abs=1e-6)
    assert ref.intercept_[0] == pytest.approx(b_expected, abs=1e-5)


def test_ovo_lexicographic_tie_break_table7():
    # Tabla 7: votos (1,1,1), márgenes (0.2,-0.3,0.1) -> gana A.
    votes = np.array([[1, 1, 1], [1, 1, 1], [2, 1, 0]])
    margins = np.array([[0.2, -0.3, 0.1], [0.1, 0.1, 0.1], [-1e12, 1e12, 0.0]])
    choice = SMOOneVsOne._lexicographic_choice(votes, margins)
    # Fila 2: empate total -> primera clase. Fila 3: los votos dominan
    # aunque el margen sea enorme (la regla v + 1e-9 S elegiría B).
    assert choice.tolist() == [0, 0, 0]


def test_global_history_ends_at_certified_residual():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = np.where(X[:, 0] + 0.3 * rng.normal(size=60) > 0, 1.0, -1.0)
    own = SMOBinary(C=2.0, kernel="rbf", gamma=0.5, tol=1e-4).fit(X, y)
    assert own.gap_history_[-1] == pytest.approx(own.kkt_gap_)
    assert own.gap_history_[-1] <= own.tol
    # El dual no decrece: SMO solo acepta pasos de ganancia no negativa.
    assert np.all(np.diff(own.dual_history_) >= -1e-10)
