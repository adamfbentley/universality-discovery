import importlib.util
import json
from pathlib import Path

import numpy as np

from src.analysis.clustering import ClusterResult, characterize_clusters


ROOT = Path(__file__).resolve().parents[1]


def load_experiment_62():
    path = ROOT / "experiments" / "62_feature_space_clustering.py"
    spec = importlib.util.spec_from_file_location("exp62_feature_space_clustering", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exp62_feature_vector_is_finite():
    exp62 = load_experiment_62()
    x = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    height_profile = np.sin(x) + 0.25 * np.sin(3 * x)

    features = exp62.compute_features_single(height_profile)

    assert features.shape == (6,)
    assert np.all(np.isfinite(features))
    assert features[0] > 0


def test_exp62_tiny_ew_simulation_is_centered_and_finite():
    exp62 = load_experiment_62()

    trajectory = exp62.simulate_ew(L=16, T=4, seed=123)

    assert trajectory.shape == (4, 16)
    assert np.all(np.isfinite(trajectory))
    assert abs(float(trajectory[-1].mean())) < 1e-10


def test_cluster_characterization_returns_basic_surface_statistics():
    surfaces = np.array(
        [
            [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
            [[0.0, 0.2, 0.4], [0.5, 0.7, 0.9], [1.0, 1.2, 1.4]],
            [[1.0, 1.0, 1.0], [1.5, 1.5, 1.5], [2.0, 2.0, 2.0]],
        ]
    )
    result = ClusterResult(
        labels=np.array([0, 0, 1]),
        n_clusters=2,
        cluster_centers=np.array([[0.1, 0.2], [0.8, 0.9]]),
        cluster_sizes=np.array([2, 1]),
        silhouette_score=0.5,
    )

    summaries = characterize_clusters(result, surfaces)

    assert [summary["size"] for summary in summaries] == [2, 1]
    assert all("mean_abs_gradient" in summary for summary in summaries)
    assert all("final_width" in summary for summary in summaries)


def test_exp62_result_snapshot_records_partial_clustering():
    result_path = ROOT / "results_exp62" / "results.json"
    data = json.loads(result_path.read_text())

    assert data["hdbscan"]["n_clusters"] == 4
    assert 0.45 <= data["hdbscan"]["ari"] <= 0.55
    assert data["knn_3"]["mean_accuracy"] > 0.75
