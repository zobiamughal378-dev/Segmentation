# Customer Segmentation Report (Task 3)

**Customers analysed:** 440  |  **Features used:** Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen

## 1. Data preparation
- Columns not used for clustering (ids, category codes, text): Channel, Region.
- Rows removed because of missing values: 0.
- Log transform `log(1 + x)` applied to skewed columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen.
- All features standardised with `StandardScaler` (mean 0, standard deviation 1).

## 2. PCA (dimensionality reduction)
- 4 principal components kept, explaining 92.1% of the variance (target: 90%).
- Figure: `pca_variance.png`.

| Component | Variance % | Cumulative % |
|---|---|---|
| PC1 | 44.1 | 44.1 |
| PC2 | 27.2 | 71.3 |
| PC3 | 10.7 | 82.0 |
| PC4 | 10.1 | 92.1 |
| PC5 | 4.9 | 97.0 |
| PC6 | 3.0 | 100.0 |

## 3. Choosing the number of clusters (Elbow Method)
- The elbow of the inertia curve is at **k = 2** (set manually with --k).
- Figure: `elbow_silhouette.png`.

| k | Inertia | Silhouette |
|---|---|---|
| 1 | 2,430.764 | - |
| 2 | 1,635.121 | 0.316 |
| 3 | 1,344.497 | 0.288 |
| 4 | 1,179.338 | 0.210 |
| 5 | 1,061.112 | 0.222 |
| 6 | 969.019 | 0.224 |
| 7 | 886.722 | 0.226 |
| 8 | 825.534 | 0.216 |
| 9 | 777.796 | 0.227 |
| 10 | 746.217 | 0.199 |

## 4. Final K-Means model
- k = 2, silhouette score = **0.316**, Davies-Bouldin index = **1.263** (lower is better).
- Figures: `clusters_pca.png`, `segment_profiles.png`.

## 5. Segment summary (average value per customer, original units)

| Segment | Customers | Share % | Fresh | Milk | Grocery | Frozen | Detergents_Paper | Delicassen |
|---|---|---|---|---|---|---|---|---|
| Segment 1: High Grocery & Detergents_Paper | 188 | 42.7 | 9,355.9 | 10,346.4 | 14,697.1 | 2,222.5 | 6,084.5 | 2,177.4 |
| Segment 2: Average / balanced | 252 | 57.3 | 13,973.1 | 2,401.8 | 2,918.7 | 3,705.7 | 491.9 | 1,038.0 |

## 6. Segment descriptions
- **Segment 1: High Grocery & Detergents_Paper**: 188 customers (43%), above average in Grocery (+0.7), Detergents_Paper (+0.7), Milk (+0.6).
- **Segment 2: Average / balanced**: 252 customers (57%), below average in Grocery (-0.5), Detergents_Paper (-0.5).

(Numbers in brackets are z-scores: how many standard deviations the segment average is above or below all customers.)
