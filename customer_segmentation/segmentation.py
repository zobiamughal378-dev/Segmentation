"""Task 3: Unsupervised customer segmentation.

Pipeline: load data -> clean -> (log-transform skewed spend) -> standardise -> PCA
          -> Elbow Method to choose k -> K-Means -> profile and visualise the segments.

Run:
    python segmentation.py                       # downloads the UCI Wholesale Customers dataset
    python segmentation.py --data my_file.csv    # use your own CSV (numeric customer metrics)
    python segmentation.py --k 4                 # force a number of clusters
"""

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # save figures to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DATASET_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv",
]
DEFAULT_DATA = Path("data") / "Wholesale customers data.csv"
# Columns that are identifiers or category codes, not measurable behaviour: never clustered on.
NON_FEATURE_NAMES = {"channel", "region", "customerid", "customer id", "id", "genre", "gender"}
PALETTE = ["#5B45C9", "#22A6B3", "#E1558B", "#F2A33A", "#2FBF8F", "#4D7CFE", "#E5575B", "#8E6BD8", "#7A869A", "#B5651D"]


# ---------------------------------------------------------------------------
# 1. Load and prepare
# ---------------------------------------------------------------------------
def load_data(path: Optional[Path]) -> pd.DataFrame:
    """Read the CSV at ``path``; if it is missing, try the default location and then download it."""
    candidate = Path(path) if path else DEFAULT_DATA
    if candidate.exists():
        return pd.read_csv(candidate)
    if path:
        sys.exit(f"File not found: {candidate}")
    for url in DATASET_URLS:
        try:
            print(f"Downloading dataset from {url} ...")
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                DEFAULT_DATA.parent.mkdir(parents=True, exist_ok=True)
                DEFAULT_DATA.write_bytes(response.read())
            return pd.read_csv(DEFAULT_DATA)
        except Exception as exc:  # noqa: BLE001 - any network failure ends in the same instructions
            print(f"  download failed: {exc}")
    sys.exit(
        "Could not download the dataset automatically.\n"
        "Download 'Wholesale customers data.csv' from the UCI repository "
        "(https://archive.ics.uci.edu/dataset/292/wholesale+customers), save it as\n"
        f"  {DEFAULT_DATA}\nand run the script again (or pass your own file with --data)."
    )


def select_features(df: pd.DataFrame, exclude: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Keep numeric behaviour columns; drop ids, category codes and constant columns."""
    skip = NON_FEATURE_NAMES | {name.lower() for name in exclude}
    numeric = df.select_dtypes(include="number")
    dropped = [col for col in df.columns if col not in numeric.columns or col.lower() in skip]
    features = numeric[[col for col in numeric.columns if col.lower() not in skip]]
    constant = [col for col in features.columns if features[col].nunique() <= 1]
    features = features.drop(columns=constant)
    dropped += constant
    if features.shape[1] < 2:
        sys.exit("Need at least two numeric feature columns to cluster.")
    return features, dropped


def log_transform_skewed(features: pd.DataFrame, threshold: float = 1.0) -> tuple[pd.DataFrame, list[str]]:
    """Apply log(1+x) to strongly right-skewed, non-negative columns (typical for spending data)."""
    out = features.copy()
    logged = [c for c in out.columns if out[c].min() >= 0 and out[c].skew() > threshold]
    for col in logged:
        out[col] = np.log1p(out[col])
    return out, logged


# ---------------------------------------------------------------------------
# 2. PCA, Elbow, K-Means
# ---------------------------------------------------------------------------
def run_pca(x_scaled: np.ndarray, variance: float) -> tuple[PCA, np.ndarray, int]:
    """Fit PCA and keep the fewest components (at least 2) that explain ``variance`` of the data."""
    full = PCA(random_state=RANDOM_STATE).fit(x_scaled)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    n_components = max(2, int(np.searchsorted(cumulative, variance) + 1))
    n_components = min(n_components, x_scaled.shape[1])
    reduced = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return full, reduced.fit_transform(x_scaled), n_components


def elbow_curve(x: np.ndarray, k_max: int) -> pd.DataFrame:
    """Inertia (k=1..k_max) and silhouette score (k=2..k_max) for K-Means."""
    rows = []
    for k in range(1, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(x)
        silhouette = silhouette_score(x, model.labels_) if k > 1 else np.nan
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette})
    return pd.DataFrame(rows)


def find_elbow(ks: np.ndarray, inertias: np.ndarray) -> int:
    """Elbow = the point farthest from the straight line joining the first and last points."""
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (inertias - inertias[-1]) / (inertias[0] - inertias[-1])
    distance = np.abs(x + y - 1) / np.sqrt(2)
    return max(2, int(ks[int(np.argmax(distance))]))


# ---------------------------------------------------------------------------
# 3. Interpretation
# ---------------------------------------------------------------------------
def build_profiles(features: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster means in original units, and the same means as z-scores versus all customers."""
    means = features.groupby(labels).mean()
    z = (means - features.mean()) / features.std(ddof=0)
    return means, z


def name_segments(z: pd.DataFrame) -> dict[int, str]:
    """Describe each cluster by the features where it is clearly above (or below) average."""
    names = {}
    for cluster, row in z.iterrows():
        high = row[row >= 0.5].sort_values(ascending=False).index[:2].tolist()
        if high:
            label = "High " + " & ".join(high)
        elif (row <= -0.5).sum() >= max(1, len(row) // 2):
            label = "Low spenders"
        else:
            label = "Average / balanced"
        names[int(cluster)] = f"Segment {int(cluster) + 1}: {label}"
    return names


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)


def plot_pca_variance(full: PCA, n_keep: int, path: Path) -> None:
    ratio = full.explained_variance_ratio_ * 100
    cumulative = np.cumsum(ratio)
    idx = np.arange(1, len(ratio) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    ax.bar(idx, ratio, color=PALETTE[0], alpha=0.85, label="Variance per component")
    ax.plot(idx, cumulative, color=PALETTE[2], marker="o", lw=2, label="Cumulative variance")
    ax.axvline(n_keep, color="#555", ls="--", lw=1.2)
    ax.text(n_keep + 0.08, 8, f"keep {n_keep} components\n({cumulative[n_keep - 1]:.0f}% of variance)", fontsize=9, color="#333")
    ax.set_xticks(idx); ax.set_ylim(0, 105); ax.legend(frameon=False, loc="center right")
    _style(ax, "PCA: variance explained", "Principal component", "% of variance")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_elbow(curve: pd.DataFrame, chosen_k: int, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    ax1.plot(curve["k"], curve["inertia"], marker="o", lw=2, color=PALETTE[0])
    pick = curve[curve["k"] == chosen_k].iloc[0]
    ax1.scatter([chosen_k], [pick["inertia"]], s=160, color=PALETTE[2], zorder=5, label=f"elbow: k = {chosen_k}")
    ax1.set_xticks(curve["k"]); ax1.legend(frameon=False)
    _style(ax1, "Elbow Method", "Number of clusters (k)", "Inertia (within-cluster sum of squares)")
    sil = curve.dropna(subset=["silhouette"])
    ax2.plot(sil["k"], sil["silhouette"], marker="o", lw=2, color=PALETTE[1])
    ax2.axvline(chosen_k, color=PALETTE[2], ls="--", lw=1.4)
    ax2.set_xticks(sil["k"])
    _style(ax2, "Silhouette score (higher is better)", "Number of clusters (k)", "Silhouette")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_clusters(x_pca: np.ndarray, labels: np.ndarray, names: dict[int, str], pca: PCA, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6), dpi=150)
    for cluster in sorted(names):
        mask = labels == cluster
        ax.scatter(x_pca[mask, 0], x_pca[mask, 1], s=34, alpha=0.75, color=PALETTE[cluster % len(PALETTE)],
                   edgecolor="white", linewidth=0.4, label=f"{names[cluster]} (n={mask.sum()})")
        ax.scatter(*x_pca[mask, :2].mean(axis=0), marker="X", s=260, color=PALETTE[cluster % len(PALETTE)],
                   edgecolor="black", linewidth=1.2, zorder=6)
    ratio = pca.explained_variance_ratio_ * 100
    _style(ax, "Customer segments in PCA space (X = cluster centre)",
           f"PC1 ({ratio[0]:.0f}% of variance)", f"PC2 ({ratio[1]:.0f}% of variance)")
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_profiles(z: pd.DataFrame, names: dict[int, str], sizes: pd.Series, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 0.9 * len(z) + 3.2), dpi=150,
                                   gridspec_kw={"width_ratios": [3.2, 1.2]})
    im = ax1.imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax1.set_xticks(range(z.shape[1])); ax1.set_xticklabels(z.columns, rotation=30, ha="right")
    ax1.set_yticks(range(len(z))); ax1.set_yticklabels([names[int(i)] for i in z.index])
    for r in range(z.shape[0]):
        for c in range(z.shape[1]):
            ax1.text(c, r, f"{z.values[r, c]:+.1f}", ha="center", va="center", fontsize=9,
                     color="white" if abs(z.values[r, c]) > 1.1 else "#222")
    ax1.set_title("Segment profile (z-score vs all customers)", fontsize=13, fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    ax2.barh(range(len(sizes)), sizes.values, color=[PALETTE[int(i) % len(PALETTE)] for i in sizes.index])
    for i, value in enumerate(sizes.values):
        ax2.text(value, i, f"  {value} ({value / sizes.sum() * 100:.0f}%)", va="center", fontsize=9)
    ax2.set_yticks([]); ax2.invert_yaxis(); ax2.set_xlim(0, sizes.max() * 1.45)
    _style(ax2, "Segment size", "Customers")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
def md_table(df: pd.DataFrame, floatfmt: str = "{:,.1f}") -> str:
    """Render a DataFrame as a Markdown table (integers stay integers, NaN becomes '-')."""
    def cell(value) -> str:
        if isinstance(value, (float, np.floating)):
            return "-" if np.isnan(value) else floatfmt.format(value)
        return str(value)

    head = "| " + " | ".join(str(c) for c in [df.index.name or ""] + list(df.columns)) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + [cell(v) for v in r]) + " |"
            for i, r in zip(df.index, df.astype(object).values)]
    return "\n".join([head, sep] + rows)


def write_report(path: Path, ctx: dict) -> None:
    curve = ctx["curve"].copy()
    curve["k"] = curve["k"].astype(int)
    curve_table = curve.set_index("k").rename(columns={"inertia": "Inertia", "silhouette": "Silhouette"})
    summary = ctx["summary"]
    lines = [
        "# Customer Segmentation Report (Task 3)", "",
        f"**Customers analysed:** {ctx['n_rows']}  |  **Features used:** {', '.join(ctx['features'])}", "",
        "## 1. Data preparation",
        f"- Columns not used for clustering (ids, category codes, text): {', '.join(ctx['dropped']) or 'none'}.",
        f"- Rows removed because of missing values: {ctx['n_missing']}.",
        f"- Log transform `log(1 + x)` applied to skewed columns: {', '.join(ctx['logged']) or 'none'}.",
        "- All features standardised with `StandardScaler` (mean 0, standard deviation 1).", "",
        "## 2. PCA (dimensionality reduction)",
        f"- {ctx['n_components']} principal components kept, explaining {ctx['kept_variance']:.1f}% of the variance "
        f"(target: {ctx['variance_target'] * 100:.0f}%).",
        "- Figure: `pca_variance.png`.", "",
        md_table(ctx["variance_table"], "{:.1f}"), "",
        "## 3. Choosing the number of clusters (Elbow Method)",
        f"- The elbow of the inertia curve is at **k = {ctx['k']}**" + (" (set manually with --k)." if ctx["k_manual"] else "."),
        "- Figure: `elbow_silhouette.png`.", "", md_table(curve_table, "{:,.3f}"), "",
        "## 4. Final K-Means model",
        f"- k = {ctx['k']}, silhouette score = **{ctx['silhouette']:.3f}**, Davies-Bouldin index = **{ctx['dbi']:.3f}** (lower is better).",
        "- Figures: `clusters_pca.png`, `segment_profiles.png`.", "",
        "## 5. Segment summary (average value per customer, original units)", "", md_table(summary, "{:,.1f}"), "",
        "## 6. Segment descriptions",
    ]
    for cluster, name in ctx["names"].items():
        size = int(ctx["sizes"][cluster])
        row = ctx["z"].loc[cluster].sort_values(ascending=False)
        above = ", ".join(f"{k} ({v:+.1f})" for k, v in row[row >= 0.5].items())
        below = ", ".join(f"{k} ({v:+.1f})" for k, v in row[row <= -0.5].sort_values().items())
        parts = [f"above average in {above}" if above else "", f"below average in {below}" if below else ""]
        detail = "; ".join(p for p in parts if p) or "close to the average on every feature"
        lines.append(f"- **{name}**: {size} customers ({size / ctx['n_rows'] * 100:.0f}%), {detail}.")
    lines += ["", "(Numbers in brackets are z-scores: how many standard deviations the segment average is above or below all customers.)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1/6  Loading data")
    raw = load_data(Path(args.data) if args.data else None)
    print(f"          {raw.shape[0]} rows x {raw.shape[1]} columns")

    print("Step 2/6  Selecting features and cleaning")
    features, dropped = select_features(raw, args.exclude or [])
    n_before = len(features)
    features = features.dropna()
    n_missing = n_before - len(features)
    transformed, logged = (features, []) if args.no_log else log_transform_skewed(features)
    print(f"          features: {list(features.columns)}; log-transformed: {logged or 'none'}")

    print("Step 3/6  Standardising (StandardScaler) and running PCA")
    x_scaled = StandardScaler().fit_transform(transformed)
    full_pca, x_pca, n_components = run_pca(x_scaled, args.variance)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(x_scaled)
    kept = float(pca.explained_variance_ratio_.sum() * 100)
    print(f"          {n_components} components keep {kept:.1f}% of the variance")

    print("Step 4/6  Elbow Method")
    k_max = min(args.k_max, len(features) - 1)
    curve = elbow_curve(x_pca, k_max)
    elbow_k = find_elbow(curve["k"].to_numpy(dtype=float), curve["inertia"].to_numpy())
    k = args.k or elbow_k
    print(f"          elbow at k = {elbow_k}" + (f"; using k = {k} (manual)" if args.k else ""))

    print(f"Step 5/6  Fitting K-Means with k = {k}")
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(x_pca)
    labels = model.labels_
    silhouette, dbi = silhouette_score(x_pca, labels), davies_bouldin_score(x_pca, labels)
    print(f"          silhouette = {silhouette:.3f}, Davies-Bouldin = {dbi:.3f}")

    print("Step 6/6  Profiling segments and saving results")
    means, z = build_profiles(features, labels)
    names = name_segments(z)
    sizes = pd.Series(labels).value_counts().sort_index()
    summary = means.copy()
    summary.insert(0, "Customers", sizes.values)
    summary.insert(1, "Share %", (sizes.values / len(labels) * 100).round(1))
    summary.index = [names[int(i)] for i in summary.index]
    summary.index.name = "Segment"

    plot_pca_variance(full_pca, n_components, out_dir / "pca_variance.png")
    plot_elbow(curve, elbow_k if not args.k else k, out_dir / "elbow_silhouette.png")
    plot_clusters(x_pca, labels, names, pca, out_dir / "clusters_pca.png")
    plot_profiles(z, names, sizes, out_dir / "segment_profiles.png")

    result = raw.loc[features.index].copy()
    result["Cluster"] = labels + 1
    result["Segment"] = [names[int(i)] for i in labels]
    result["PC1"], result["PC2"] = x_pca[:, 0], x_pca[:, 1]
    result.to_csv(out_dir / "customers_segmented.csv", index=False)
    summary.to_csv(out_dir / "cluster_summary.csv")
    pd.DataFrame(pca.components_.T, index=features.columns,
                 columns=[f"PC{i + 1}" for i in range(n_components)]).to_csv(out_dir / "pca_loadings.csv")

    variance_table = pd.DataFrame({
        "Variance %": full_pca.explained_variance_ratio_ * 100,
        "Cumulative %": np.cumsum(full_pca.explained_variance_ratio_) * 100,
    }, index=[f"PC{i + 1}" for i in range(len(full_pca.explained_variance_ratio_))])
    variance_table.index.name = "Component"
    write_report(out_dir / "report.md", dict(
        n_rows=len(features), features=list(features.columns), dropped=dropped, n_missing=n_missing, logged=logged,
        n_components=n_components, kept_variance=kept, variance_target=args.variance, variance_table=variance_table,
        k=k, k_manual=bool(args.k), curve=curve, silhouette=silhouette, dbi=dbi, summary=summary, names=names,
        sizes=sizes, z=z))
    print(f"\nDone. Results saved in: {out_dir.resolve()}")
    return {"k": k, "elbow_k": elbow_k, "silhouette": silhouette, "labels": labels, "names": names}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer segmentation: StandardScaler + PCA + Elbow + K-Means")
    parser.add_argument("--data", help="path to a CSV file (default: UCI Wholesale Customers, downloaded automatically)")
    parser.add_argument("--out", default="outputs", help="folder for figures, CSVs and the report")
    parser.add_argument("--k", type=int, help="force the number of clusters instead of using the elbow")
    parser.add_argument("--k-max", type=int, default=10, help="largest k tried by the elbow method")
    parser.add_argument("--variance", type=float, default=0.90, help="share of variance PCA must keep (0-1)")
    parser.add_argument("--exclude", nargs="+", help="extra columns to leave out of the clustering")
    parser.add_argument("--no-log", action="store_true", help="skip the log transform of skewed columns")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
