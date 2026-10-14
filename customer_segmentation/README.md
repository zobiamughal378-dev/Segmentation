# Customer Segmentation with K-Means and PCA

Progree Data Science Internship, **Task 3: Unsupervised Machine Learning Customer Segmentation (Clustering)**.

The script groups customers by their buying behaviour. It standardises the data, reduces it with PCA, picks the number of clusters with the Elbow Method, fits K-Means and draws the segments.

## Task 3 requirements and where they are met

| Requirement | What the code does | In `segmentation.py` |
|---|---|---|
| Ingest transaction or consumer metric data with Scikit-Learn / Python | Loads a customer spending CSV with pandas (UCI *Wholesale Customers* by default) | `load_data`, `select_features` |
| Data standardisation | `StandardScaler` (after a log transform of skewed spending columns) | `log_transform_skewed`, `run` |
| Principal Component Analysis (PCA) | Keeps the fewest components that explain 90 percent of the variance | `run_pca` |
| Optimise the cluster count with the Elbow Method | Inertia for k = 1 to 10, elbow found as the point farthest from the straight line between the ends; silhouette score shown as a second check | `elbow_curve`, `find_elbow` |
| Fit a K-Means model | `KMeans` with the chosen k | `run` |
| Visualise the segment distributions | PCA scatter plot, segment profile heatmap, segment sizes, elbow and PCA variance charts | `plot_*` functions |

## Dataset

Default: **Wholesale Customers** from the UCI Machine Learning Repository (440 customers; yearly spending on Fresh, Milk, Grocery, Frozen, Detergents_Paper and Delicassen products).
Page: https://archive.ics.uci.edu/dataset/292/wholesale+customers

The script downloads it automatically on the first run. If your internet is blocked, download `Wholesale customers data.csv` from that page and save it as `data/Wholesale customers data.csv`.

Other customer datasets also work, for example *Mall Customers* (`--data Mall_Customers.csv`). Columns such as `CustomerID`, `Genre`, `Channel` and `Region` are ignored automatically, because they are identifiers or category codes and not behaviour.

## How to run

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python segmentation.py
```

Useful options:

```powershell
python segmentation.py --data Mall_Customers.csv   # your own file
python segmentation.py --k 4                        # force the number of clusters
python segmentation.py --variance 0.95              # keep more PCA variance
python segmentation.py --exclude Age                # leave a column out of the clustering
```

Tests: `python -m pytest` (they use a small synthetic dataset, no download needed).

## What you get in `outputs/`

| File | Content |
|---|---|
| `report.md` | Full written report: preprocessing, PCA table, elbow table, metrics, segment summary and descriptions |
| `elbow_silhouette.png` | Elbow Method chart and silhouette scores |
| `pca_variance.png` | Variance explained by each principal component |
| `clusters_pca.png` | Customers plotted on the first two principal components, coloured by segment |
| `segment_profiles.png` | Heatmap of what makes each segment different, plus segment sizes |
| `customers_segmented.csv` | Every customer with their `Cluster`, `Segment` name and PC1 / PC2 |
| `cluster_summary.csv` | Average spending per segment |
| `pca_loadings.csv` | How much each original feature contributes to each component |

The numbers (best k, silhouette score, segment names) depend on the data. Run the script and use what `report.md` says; do not copy numbers from anywhere else.

## Method in simple words

1. **Clean:** keep only numeric behaviour columns; drop rows with missing values.
2. **Log transform:** spending data has a few very large values. `log(1 + x)` stops those from dominating.
3. **Standardise:** put every feature on the same scale (mean 0, standard deviation 1), so no feature wins just because its numbers are bigger.
4. **PCA:** merge correlated features into a few new axes (components) that keep most of the information. This removes noise and makes a 2D picture possible.
5. **Elbow Method:** try k = 1 to 10 clusters. Inertia (how spread out each cluster is) always falls as k grows; the best k is where the fall suddenly slows down, like an elbow.
6. **K-Means:** assign each customer to the nearest of k cluster centres and move the centres until they stop changing.
7. **Profile:** compare each segment's average spending with all customers (z-score) and name it by what stands out.

## Reading the results

- **Silhouette score** (-1 to 1): how well customers fit their own cluster compared with the nearest other one. Higher is better; above about 0.5 is a clear structure, 0.25 to 0.5 is moderate.
- **Davies-Bouldin index:** lower is better.
- **Segment names** are generated automatically from the strongest features. Read the heatmap and rename them for a business audience (for example "Cafe and restaurant buyers").

## Limitations

- K-Means prefers round, similar-sized clusters and is sensitive to outliers.
- The elbow can be unclear. If it is, compare with the silhouette chart or set `--k`.
- Segments describe patterns in this data only; they are not proof of cause.

## Files

```
customer_segmentation/
├── segmentation.py        # the whole pipeline
├── requirements.txt
├── data/                  # dataset is saved here
├── outputs/               # figures, CSVs and report.md are written here
└── tests/test_segmentation.py
```
