# Simulation Methodology for User-Item Interaction Data

## Abstract

No publicly available dataset provides user-level perfume ratings that map cleanly to the Fragrantica catalog used in this project. This section documents the design, parameter selection, validation, and limitations of the synthetic interaction data used to evaluate the collaborative filtering component of this system. All evaluation metrics derived from this data should be interpreted as internal consistency measures, not estimates of real-world recommendation quality.

---

## 1. Motivation

Collaborative filtering requires a user-item rating matrix $R \in \mathbb{R}^{U \times I}$ where $R_{ui}$ represents the rating given by user $u$ to item $i$. The Fragrantica dataset provides only aggregate statistics per perfume (mean rating, rating count) — no individual user identifiers or per-user ratings are available. Alternatives were investigated and rejected:

- **Amazon Beauty Reviews (McAuley Lab, 2023):** 701,528 reviews across 112,590 products. After filtering to fragrance-specific products, 4,667 candidates remained. Fuzzy name matching against the Fragrantica catalog yielded an estimated overlap of 10–15% (~500–700 perfumes), representing 2% of the catalog. Coverage was deemed insufficient for meaningful collaborative filtering evaluation.

- **VADER Sentiment on Fragrantica Reviews:** Deriving ratings from review text via sentiment analysis (Hutto & Gilbert, 2014) introduces two sequential approximation layers (preference → language → sentiment score) and would require scraping Fragrantica at scale, which violates their terms of service.

Simulation was therefore adopted as the only viable path, with parameters calibrated against the Amazon Beauty dataset as an empirical reference distribution.

---

## 2. Generative Model

### 2.1 User Preference

Each simulated user $u \in \{1, \ldots, U\}$ is associated with a latent preference vector $\boldsymbol{\theta}_u \in \mathbb{R}^V$, where $V = 1{,}671$ is the note vocabulary size. The preference vector is generated as:

$$\boldsymbol{\theta}_u = \mathbf{x}_{s(u)} + \boldsymbol{\epsilon}_u, \quad \boldsymbol{\epsilon}_u \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}), \quad \boldsymbol{\theta}_u \leftarrow \text{clip}(\boldsymbol{\theta}_u, 0, 1)$$

where $s(u)$ is a seed perfume drawn with probability proportional to rating count:

$$P(s(u) = i) = \frac{\text{RatingCount}_i}{\sum_{j=1}^{I} \text{RatingCount}_j}$$

This reflects the empirical observation that popular perfumes have higher real-world exposure — users are more likely to have encountered and formed preferences around widely available fragrances.

### 2.2 Affinity Scoring

The affinity of user $u$ for perfume $i$ is modelled as noisy cosine similarity:

$$\tilde{s}_{ui} = \text{cos}(\boldsymbol{\theta}_u, \mathbf{x}_i) + \delta_{ui}, \quad \delta_{ui} \sim \mathcal{N}(0, \tau^2)$$

where $\mathbf{x}_i \in \{0,1\}^V$ is the multi-hot note vector of perfume $i$. The noise term $\delta_{ui}$ captures idiosyncratic reactions to perfumes that cannot be explained by note composition alone.

### 2.3 Observation Model

An interaction $(u, i)$ is observed if:

$$\tilde{s}_{ui} > \rho \quad \text{and} \quad \text{Bernoulli}(\text{sparsity}) = 1$$

The threshold $\rho$ reflects the fact that users predominantly rate perfumes they expect to enjoy. The Bernoulli draw with probability $p = \text{sparsity}$ simulates that users rate only a fraction of perfumes they encounter.

### 2.4 Rating Assignment

Raw cosine similarities in high-dimensional sparse binary space cluster tightly in the range $[0.35, 0.55]$. Direct linear scaling to a 1–5 rating scale produces a severely compressed distribution (mean $\approx 1.97$, std $\approx 0.22$) inconsistent with any known review platform. A rank-based assignment was adopted instead, mapping each user's within-user similarity rank to a rating according to the empirical J-curve distribution of Amazon Beauty ratings:

| Rank Percentile | Rating | Amazon Beauty Frequency |
|---|---|---|
| > 45% | 5 | ~55% |
| 20–45% | 4 | ~25% |
| 10–20% | 3 | ~10% |
| 5–10% | 2 | ~5% |
| < 5% | 1 | ~5% |

This guarantees the marginal rating distribution matches the target regardless of the absolute similarity values, which are an artifact of the high-dimensional sparse feature space rather than a meaningful signal.

---

## 3. Parameter Selection

Parameters were tuned iteratively to minimise divergence between the simulated and Amazon Beauty distributions across five metrics: rating mean, rating standard deviation, ratings per user (median and mean), ratings per item (median and mean), Gini coefficient of item ratings, and sparsity.

| Parameter | Symbol | Final Value | Role |
|---|---|---|---|
| Users | $U$ | 5,000 | Total simulated users |
| Preference noise | $\sigma$ | 0.1 | Deviation of user taste from seed perfume |
| Rating noise | $\tau$ | 0.05 | Idiosyncratic reaction noise |
| Observation threshold | $\rho$ | 0.40 | Minimum affinity to generate a rating |
| Observation rate | sparsity | 0.05 | Fraction of above-threshold pairs rated |

### 3.1 Validation Against Amazon Beauty

| Metric | Amazon Beauty | Simulation | Notes |
|---|---|---|---|
| Rating mean | 3.96 | 3.42 | Acceptable |
| Rating std | 1.49 | 1.70 | Acceptable |
| Ratings/user median | 1 | 2 | Acceptable |
| Ratings/user mean | 1.1 | 3.4 | Slightly too high |
| Ratings/item median | 2 | 1 | Acceptable |
| Ratings/item mean | 6.1 | 1.5 | Too low — see §4 |
| Gini coefficient | 0.692 | 0.269 | Structurally limited — see §4 |
| Sparsity | 0.0010% | 0.0751% | 75× denser than target |

---

## 4. Limitations

### 4.1 Circularity
The simulation generates ratings from note-space cosine similarity. The content-based recommender being evaluated also uses note-space cosine similarity. Evaluating the recommender against these ratings therefore partially measures whether the model recovers its own generating process. Precision@K and Recall@K numbers should be treated as **consistency metrics** — they confirm internal coherence, not real-world performance.

### 4.2 Gini Coefficient Gap
The simulated item rating distribution has Gini coefficient 0.269, compared to 0.692 for Amazon Beauty. This reflects a fundamental structural constraint: achieving Gini $\approx 0.69$ with $U = 5{,}000$ users requires the top 1% of items (240 perfumes) to hold ~50% of all ratings. Under popularity-weighted seed sampling, interaction concentration improves but remains far below the target. Reproducing Amazon-level Gini would require $U \approx 600{,}000$ users, which implies a $(600k \times 24k)$ similarity matrix of approximately 228 GB — computationally infeasible without distributed infrastructure. The practical consequence is that the severity of the cold-start problem is **underestimated** in this evaluation: niche perfumes receive proportionally more simulated interactions than they would in a real deployment.

### 4.3 Single-Seed User Model
Each user is modelled as a Gaussian perturbation around a single seed perfume. Real users may have multimodal preferences — enjoying both clean aquatics and heavy orientals, for example — which a unimodal Gaussian in note space cannot represent.

### 4.4 Missing Quality Signal
The item's aggregate Rating Value (mean score across all real raters) was not incorporated into the generative model. A perfume rated 4.8 by thousands of users is not treated differently from one rated 3.1. This omits a real signal: some perfumes are more broadly well-executed independent of note composition.

### 4.5 Static Preferences
The model assumes time-invariant user preferences. Real preference evolution — seasonal shifts, growing sophistication, trend effects — is not captured.

---

## 5. Reproducibility

The simulation is fully deterministic given the random seed:

```python
np.random.seed(42)
```

The output file `data/simulated_ratings.csv` contains columns `user_id`, `perfume_id`, and `rating`. All downstream evaluation notebooks load this file directly.

---

## 6. References

- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *ICWSM*.
- Hou, Y. et al. (2024). Bridging Language and Items for Retrieval and Recommendation. *arXiv:2403.03952*. [Amazon Reviews 2023 Dataset]
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer*, 42(8), 30–37.
