"""Unit tests for the mathematical core of QMI-CFS."""
import numpy as np
import pytest

from qmi_cfs.quantum_encoding import AngleEncoder
from qmi_cfs.qmi_computation import QMICalculator, VonNeumannEntropy
from qmi_cfs.causal_estimation import AIPWEstimator
from qmi_cfs.evaluation import compute_f1, compute_recall, compute_precision


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestDensityMatrices:
    @pytest.mark.parametrize("entangle", [False, True])
    def test_trace_one_symmetric_psd(self, rng, entangle):
        enc = AngleEncoder(3, use_entanglement=entangle)
        a, b, c = (rng.uniform(0.05, 3.09, 200) for _ in range(3))
        for rho in (enc.encode_single_feature(a),
                    enc.encode_feature_pair(a, b),
                    enc.encode_feature_triple(a, b, c)):
            assert np.isclose(np.trace(rho), 1.0)
            assert np.allclose(rho, rho.T)
            assert np.linalg.eigvalsh(rho).min() > -1e-10

    def test_invalid_n_features(self):
        with pytest.raises(ValueError):
            AngleEncoder(0)


class TestEntropyAndQMI:
    def test_pure_state_zero_entropy(self):
        assert VonNeumannEntropy.compute(np.diag([1.0, 0.0])) == pytest.approx(0.0, abs=1e-9)

    def test_maximally_mixed_one_bit(self):
        assert VonNeumannEntropy.compute(np.eye(2) / 2) == pytest.approx(1.0)

    def test_qmi_nonnegative(self, rng):
        calc = QMICalculator(AngleEncoder(1, use_entanglement=False))
        x = rng.uniform(0.05, 3.09, 300)
        y = rng.uniform(0.05, 3.09, 300)
        assert calc.compute_qmi(x, y) >= 0.0
        assert calc.compute_conditional_qmi(x, y, rng.uniform(0.05, 3.09, 300)) >= 0.0

    def test_dependent_exceeds_independent(self, rng):
        calc = QMICalculator(AngleEncoder(1, use_entanglement=False))
        x = rng.uniform(0.3, 2.8, 2000)
        dep = np.clip(x + rng.normal(0, 0.05, 2000), 0.05, 3.09)
        ind = rng.uniform(0.3, 2.8, 2000)
        assert calc.compute_qmi(x, dep) > calc.compute_qmi(x, ind)

    def test_pairwise_matrix_symmetric_zero_diag(self, rng):
        calc = QMICalculator(AngleEncoder(4, use_entanglement=False))
        M = calc.compute_pairwise_qmi_matrix(rng.uniform(0.05, 3.09, (100, 4)))
        assert np.allclose(M, M.T)
        assert np.allclose(np.diag(M), 0.0)


class TestAIPW:
    def test_recovers_known_ate(self):
        rng = np.random.default_rng(1)
        n = 3000
        X = rng.normal(size=(n, 2))
        T = rng.binomial(1, 1 / (1 + np.exp(-X[:, 0])))
        Y = 2.0 * T + X[:, 0] + X[:, 1] + rng.normal(0, 0.5, n)
        est = AIPWEstimator(random_state=0).fit(X, T, Y)
        assert est.estimate_ate(X, T, Y) == pytest.approx(2.0, abs=0.2)

    def test_ci_contains_estimate_and_scales_with_alpha(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(500, 2))
        T = rng.binomial(1, 0.5, 500)
        Y = T + X[:, 0] + rng.normal(size=500)
        est = AIPWEstimator(random_state=0).fit(X, T, Y)
        lo95, hi95 = est.compute_confidence_interval(X, T, Y, alpha=0.05)
        lo90, hi90 = est.compute_confidence_interval(X, T, Y, alpha=0.10)
        assert lo95 < est.estimate_ate(X, T, Y) < hi95
        assert (hi90 - lo90) < (hi95 - lo95)


class TestMetrics:
    def test_f1_precision_recall(self):
        sel, truth = [0, 1, 5], [0, 1, 2]
        assert compute_precision(sel, truth) == pytest.approx(2 / 3)
        assert compute_recall(sel, truth) == pytest.approx(2 / 3)
        assert compute_f1(sel, truth) == pytest.approx(2 / 3)


class TestReproducibility:
    def test_selector_deterministic_given_seed(self):
        from qmi_cfs.data_generation import generate_lld
        from qmi_cfs.feature_selection import QMIFeatureSelector
        X, T, Y, _, _ = generate_lld(n_samples=150, n_features=10, seed=3)
        runs = [QMIFeatureSelector(n_bootstrap=10, random_state=7).fit(X, T, Y)
                .get_selected_features() for _ in range(2)]
        assert runs[0] == runs[1]
