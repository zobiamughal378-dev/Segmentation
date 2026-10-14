"""Tests for segmentation.py. They build a small synthetic dataset, so no download is needed."""

import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import segmentation as seg  # noqa: E402


def make_data(n_per_group: int = 60) -> pd.DataFrame:
    """Three clearly different groups of customers with skewed (log-normal) spending."""
    rng = np.random.default_rng(0)
    centers = [[9.5, 6.5, 6.5, 7.0], [6.5, 9.5, 9.0, 6.0], [7.5, 7.5, 6.0, 9.5]]
    rows = np.vstack([np.exp(rng.normal(c, 0.4, size=(n_per_group, 4))) for c in centers]).round().astype(int)
    df = pd.DataFrame(rows, columns=["Fresh", "Milk", "Grocery", "Frozen"])
    df.insert(0, "Region", rng.integers(1, 4, len(df)))
    df.insert(0, "Channel", rng.integers(1, 3, len(df)))
    return df


def test_select_features_drops_codes_ids_and_constant_columns():
    df = make_data().assign(CustomerID=range(180), Constant=7, Name="x")
    features, dropped = seg.select_features(df, exclude=[])
    assert list(features.columns) == ["Fresh", "Milk", "Grocery", "Frozen"]
    assert {"Channel", "Region", "CustomerID", "Constant", "Name"} <= set(dropped)


def test_log_transform_only_touches_skewed_non_negative_columns():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"skewed": np.exp(rng.normal(5, 1.2, 500)), "symmetric": rng.normal(50, 5, 500)})
    out, logged = seg.log_transform_skewed(df)
    assert logged == ["skewed"]
    assert out["skewed"].skew() < df["skewed"].skew()
    assert out["symmetric"].equals(df["symmetric"])


def test_pca_keeps_enough_components_for_requested_variance():
    x = seg.StandardScaler().fit_transform(np.log1p(make_data()[["Fresh", "Milk", "Grocery", "Frozen"]]))
    full, reduced, n = seg.run_pca(x, 0.90)
    assert reduced.shape == (len(x), n) and n >= 2
    assert full.explained_variance_ratio_[:n].sum() >= 0.90


def test_find_elbow_detects_the_bend():
    ks = np.arange(1, 9, dtype=float)
    inertia = np.array([1000, 520, 180, 150, 135, 125, 118, 112], dtype=float)  # clear bend at k = 3
    assert seg.find_elbow(ks, inertia) == 3


def test_name_segments_describes_high_and_low_groups():
    z = pd.DataFrame({"Milk": [1.4, -0.6, -0.5], "Fresh": [0.2, 1.1, -0.7], "Frozen": [0.1, 0.9, -0.6]}, index=[0, 1, 2])
    names = seg.name_segments(z)
    assert names[0].endswith("High Milk")
    assert "Fresh" in names[1] and "Frozen" in names[1]
    assert names[2].endswith("Low spenders")


def test_full_run_creates_every_output_and_finds_three_groups():
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "customers.csv"
        make_data().to_csv(data_path, index=False)
        out = Path(tmp) / "out"
        args = Namespace(data=str(data_path), out=str(out), k=None, k_max=8, variance=0.90, exclude=None, no_log=False)
        result = seg.run(args)
        for name in ["pca_variance.png", "elbow_silhouette.png", "clusters_pca.png", "segment_profiles.png",
                     "customers_segmented.csv", "cluster_summary.csv", "pca_loadings.csv", "report.md"]:
            assert (out / name).exists(), name
        assert result["k"] == 3 and result["silhouette"] > 0.4
        segmented = pd.read_csv(out / "customers_segmented.csv")
        assert len(segmented) == 180 and set(segmented["Cluster"]) == {1, 2, 3}
        assert "Silhouette" in (out / "report.md").read_text(encoding="utf-8")
