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
- The elbow of the inertia curve is at **k = 3**.
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
- k = 3, silhouette score = **0.288**, Davies-Bouldin index = **1.263** (lower is better).
- Figures: `clusters_pca.png`, `segment_profiles.png`.

## 5. Segment summary (average value per customer, original units)

| Segment | Customers | Share % | Fresh | Milk | Grocery | Frozen | Detergents_Paper | Delicassen |
|---|---|---|---|---|---|---|---|---|
| Segment 1: High Detergents_Paper | 82 | 18.6 | 2,996.8 | 7,045.4 | 12,504.6 | 607.7 | 5,511.2 | 796.7 |
| Segment 2: Low spenders | 215 | 48.9 | 12,031.9 | 2,068.5 | 2,526.4 | 3,237.4 | 421.9 | 905.1 |
| Segment 3: High Milk & Grocery | 143 | 32.5 | 17,115.6 | 10,684.7 | 13,496.6 | 4,236.3 | 5,071.5 | 2,874.3 |

## 6. Segment descriptions
- **Segment 1: High Detergents_Paper**: 82 customers (19%), above average in Detergents_Paper (+0.6); below average in Fresh (-0.7), Frozen (-0.5).
- **Segment 2: Low spenders**: 215 customers (49%), below average in Grocery (-0.6), Detergents_Paper (-0.5), Milk (-0.5).
- **Segment 3: High Milk & Grocery**: 143 customers (32%), above average in Milk (+0.7), Grocery (+0.6).

(Numbers in brackets are z-scores: how many standard deviations the segment average is above or below all customers.)
