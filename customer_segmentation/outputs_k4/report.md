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
- The elbow of the inertia curve is at **k = 4** (set manually with --k).
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
- k = 4, silhouette score = **0.210**, Davies-Bouldin index = **1.423** (lower is better).
- Figures: `clusters_pca.png`, `segment_profiles.png`.

## 5. Segment summary (average value per customer, original units)

| Segment | Customers | Share % | Fresh | Milk | Grocery | Frozen | Detergents_Paper | Delicassen |
|---|---|---|---|---|---|---|---|---|
| Segment 1: High Grocery & Milk | 117 | 26.6 | 11,793.4 | 12,143.0 | 16,403.7 | 2,330.0 | 6,853.8 | 2,681.7 |
| Segment 2: Low spenders | 127 | 28.9 | 9,527.0 | 1,511.4 | 2,019.8 | 2,199.0 | 339.8 | 578.7 |
| Segment 3: Average / balanced | 65 | 14.8 | 2,911.7 | 6,828.7 | 11,970.7 | 557.3 | 5,120.9 | 662.9 |
| Segment 4: High Frozen & Fresh | 131 | 29.8 | 19,092.4 | 3,769.6 | 4,158.2 | 5,828.6 | 686.7 | 1,836.7 |

## 6. Segment descriptions
- **Segment 1: High Grocery & Milk**: 117 customers (27%), above average in Grocery (+0.9), Milk (+0.9), Detergents_Paper (+0.8).
- **Segment 2: Low spenders**: 127 customers (29%), below average in Grocery (-0.6), Milk (-0.6), Detergents_Paper (-0.5).
- **Segment 3: Average / balanced**: 65 customers (15%), below average in Fresh (-0.7), Frozen (-0.5).
- **Segment 4: High Frozen & Fresh**: 131 customers (30%), above average in Frozen (+0.6), Fresh (+0.6).

(Numbers in brackets are z-scores: how many standard deviations the segment average is above or below all customers.)
