# Data Pipeline & User Interaction Survey

## Overview

This document describes the complete data pipeline for the perfume recommendation system, including dataset sourcing, merging decisions, the survey of available user rating data across platforms, and the design and validation of the synthetic interaction dataset used for collaborative filtering evaluation.

---

## 1. Perfume Catalog Data

### 1.1 Source Datasets

Two complementary datasets were sourced from Kaggle, both scraped from Fragrantica.com:

**Dataset A — Raw Fragrantica Scrape**
- 70,103 perfumes
- Columns: `Name`, `Gender`, `Rating Value`, `Rating Count`, `Main Accords`, `Perfumers`, `Description`, `url`
- Main Accords are community-voted broad scent families (e.g. floral, woody, citrus)
- These reflect perceived olfactory character, not ingredient composition

**Dataset B — Cleaned Fragrantica Scrape**
- 24,063 perfumes
- Columns: `url`, `Perfume`, `Brand`, `Country`, `Gender`, `Rating Value`, `Rating Count`, `Year`, `Top`, `Middle`, `Base`, `Perfumer1`, `Perfumer2`, `mainaccord1–5`
- Top/Middle/Base note pyramid scraped from individual perfume pages
- These reflect brand-declared ingredient composition

### 1.2 Key Distinction

Dataset A and Dataset B are **not** raw vs. cleaned versions of the same scrape. They capture fundamentally different information from the same Fragrantica pages:

- **Accords** (Dataset A) = community perception — how users collectively describe the scent
- **Notes** (Dataset B) = brand declaration — what ingredients are in the bottle

A perfume with notes of `mandarin orange, rose, musk` might have accords of `floral, citrus, musky` — the same perfume described at two levels of abstraction. This distinction has real modeling implications: accords capture perceived similarity, notes capture ingredient similarity. These are not the same thing.

### 1.3 Merging

The datasets were merged on URL, normalized to lowercase and stripped of whitespace:

```python
df['url_clean'] = df['url'].str.lower().str.strip()
cleaned['url_clean'] = cleaned['url'].str.lower().str.strip()
merged = pd.merge(df, cleaned[['url_clean', 'Top', 'Middle', 'Base']], 
                  on='url_clean', how='inner')
```

Result: 24,063 perfumes with both accord and note pyramid data. Dataset B is entirely contained within Dataset A — every URL in the cleaned dataset exists in the raw dataset.

### 1.4 Cleaning Steps

| Issue | Resolution |
|---|---|
| `Rating Count` stored as string with comma separators (`"6,865"`) | `.str.replace(',', '').astype(int)` |
| 41 duplicate URLs | `drop_duplicates(subset='url')` |
| Encoding errors (accented characters) | `encoding='latin-1'` on CSV read |
| Notes stored as comma-separated strings | Parsed to lists via `str.split(',')` |
| Soliflore perfumes (1 note) | Verified as legitimate, not missing data |

### 1.5 EDA Findings

**Rating distribution:** Right-skewed, concentrated between 3.5–4.5. Reflects selection bias — users predominantly rate perfumes they expected to like.

**Note frequency:** Highly power-law distributed. Musk appears in ~45% of all perfumes, bergamot in ~35%. This directly motivates TF-IDF weighting — ubiquitous notes carry minimal discriminative signal.

**Rating count distribution:** Long-tail power law. A small number of iconic perfumes (Chanel No. 5, Black Opium) have 25,000+ ratings; the median perfume has fewer than 200. This motivates popularity-weighted seed sampling in the simulation.

**Note vocabulary:** 1,671 unique notes across the catalog.

**Gender distribution:** 47% women's, 32% unisex, 21% men's.

---

## 2. Survey of Available User Rating Data

Collaborative filtering requires individual user-item ratings. An extensive survey was conducted to determine whether real user-level perfume ratings were obtainable.

### 2.1 Fragrantica

**Status: Unavailable**

Fragrantica allows logged-in users to rate perfumes, save favorites, and write reviews. However:

- No public API exists. Multiple requests from developers and researchers on their forum have gone unanswered since at least 2019.
- Individual user ratings are not exposed in any public endpoint.
- Scraping is actively rate-limited and blocked.
- Fragrantica has explicitly refused API access to researchers (including for master's thesis work).

The data exists — it is simply not accessible.

### 2.2 Amazon Beauty Reviews (McAuley Lab, 2023)

**Status: Available but insufficient**

The McAuley Lab Amazon Reviews 2023 dataset contains 701,528 reviews across 112,590 beauty products. After filtering to fragrance-specific products using keyword matching on product titles:

```
Perfume products identified:    3,684
Perfume reviews:               16,677
Unique users:                  16,232
Users with 3+ perfume ratings:     54   (0.33% of perfume reviewers)
Maximum ratings per user:           8
```

**Critical finding:** Only 54 users in the entire Amazon Beauty dataset rated 3 or more perfumes. The maximum any single user rated was 8 perfumes. This makes held-out collaborative filtering evaluation structurally impossible — there is insufficient interaction density at the user level.

**Rating distribution (perfume subset):**

| Stars | Proportion |
|---|---|
| 5★ | 66.3% |
| 4★ | 9.8% |
| 3★ | 7.0% |
| 2★ | 5.1% |
| 1★ | 11.8% |

The bimodal distribution (high 5★ and 1★, low middle ratings) likely reflects fulfilment complaints (broken bottles, counterfeits, wrong sizes) rather than pure olfactory preference. This makes Amazon ratings a noisy proxy for genuine fragrance preference.

**Fuzzy matching feasibility:** An attempt to join Amazon ASINs to Fragrantica perfume names was evaluated. After filtering to perfume products, 3,684 Amazon products remain against 24,063 Fragrantica perfumes. Estimated fuzzy match rate: 10–15% (~500–700 perfumes). Catalog coverage of 2% was deemed insufficient for meaningful CF evaluation.

### 2.3 Parfumo

**Status: Unavailable**

Parfumo is an alternative fragrance community with user ratings. The Parfumo Fragrance Dataset (Kaggle, Olga G.) covers 59,325 perfumes but contains only aggregate ratings — no individual user identifiers or per-user ratings.

### 2.4 GoldenScent (Saudi retailer)

**Status: Available but incompatible**

One GitHub project (rawanalqarni/Perfumes_Recommender) scraped GoldenScent with a reviews dataset containing: `User nickname, Perfume_name, Brand, Overall_rating, User_rating, Reviews, Date`. However, GoldenScent's catalog overlaps minimally with the Fragrantica catalog, making joining impossible without extensive data engineering.

### 2.5 Chinese Fragrance Forums

**Status: Historically available, no longer accessible**

Kelly Peng (2017) scraped the largest Chinese fragrance forum using 6 AWS EC2 instances, obtaining a user-item rating matrix with ratings on a 2–10 scale. This dataset was never publicly released and the platform has since tightened scraping restrictions. Cultural and demographic differences also limit generalizability to Western fragrance preferences.

### 2.6 Fragrantica Reviews via VADER Sentiment

**Status: Technically possible, rejected**

Fragrantica does display text reviews publicly (without login). One approach involves scraping review text and deriving ratings via VADER sentiment analysis. This was rejected for two reasons:

1. Two sequential approximation layers (preference → text → sentiment score) introduce compounding noise
2. Scraping Fragrantica at scale violates their Terms of Service and is actively blocked

### 2.7 Conclusion

Individual user-level perfume ratings do not exist in any publicly accessible, catalog-compatible dataset. This is a structural property of the fragrance domain — perfume is an extremely low-interaction category. On the world's largest product review platform (Amazon), fewer than 0.33% of perfume reviewers rate 3 or more perfumes. This makes collaborative filtering evaluation on real data impossible without either proprietary data access or scraping at scale.

---

## 3. Synthetic Interaction Data

Given the absence of real user ratings, synthetic interactions were generated to enable collaborative filtering evaluation.

### 3.1 Design Philosophy

The simulation is designed to model the **dense user segment** — the small fraction of users who interact with multiple perfumes. This is the only segment for which CF evaluation is meaningful. Real Amazon data confirms this segment has median 3 ratings per user and mean 4.3 ratings per user, which our simulation matches (median 3, mean 3.4).

The simulation is explicitly **not** attempting to reproduce the full population of perfume consumers, which would require modeling the 99.7% of users who rate 1–2 perfumes and contribute nothing to collaborative filtering.

### 3.2 Generative Model

**Step 1 — Seed Selection**

Each user is seeded from a real perfume, weighted by popularity:

$$P(\text{seed} = i) \propto \text{RatingCount}_i$$

Rationale: popular perfumes have higher real-world exposure. A random perfume consumer is more likely to have encountered Chanel No. 5 than an obscure niche fragrance.

**Step 2 — Preference Vector**

The user's latent preference vector is a noisy perturbation of their seed:

$$\boldsymbol{\theta}_u = \mathbf{x}_{\text{seed}} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 I)$$

$$\boldsymbol{\theta}_u \leftarrow \text{clip}(\boldsymbol{\theta}_u, 0, 1)$$

**Step 3 — Affinity Scoring**

Noisy cosine similarity between user profile and all perfumes:

$$\tilde{s}_{ui} = \cos(\boldsymbol{\theta}_u, \mathbf{x}_i) + \delta_{ui}, \quad \delta_{ui} \sim \mathcal{N}(0, \tau^2)$$

**Step 4 — Observation Model**

An interaction is observed if:

$$\tilde{s}_{ui} > \rho \quad \text{and} \quad \text{Bernoulli}(\text{sparsity}) = 1$$

**Step 5 — Rating Assignment**

Rank-based mapping to 1–5 stars using Amazon Beauty's J-curve:

| Rank Percentile | Rating |
|---|---|
| > 45% | 5★ |
| 20–45% | 4★ |
| 10–20% | 3★ |
| 5–10% | 2★ |
| < 5% | 1★ |

Direct linear scaling from cosine similarity to ratings was rejected because cosine similarities in 1,671-dimensional sparse binary space cluster tightly around 0.35–0.55, producing ratings with mean ~1.97 and std ~0.22. Rank-based assignment produces a distribution comparable to real review platforms regardless of absolute similarity values.

### 3.3 Final Parameters

| Parameter | Value | Justification |
|---|---|---|
| n_users | 5,000 | Practical RAM limit for (U × I) similarity matrix |
| σ (preference noise) | 0.10 | Moderate taste deviation from seed |
| τ (rating noise) | 0.05 | Idiosyncratic reaction noise |
| ρ (observation threshold) | 0.40 | Minimum affinity to generate interaction |
| sparsity | 0.05 | 5% of above-threshold pairs observed |

### 3.4 Simulation Output vs. Targets

| Metric | Amazon Perfume Dense | Amazon Beauty Dense | Simulation |
|---|---|---|---|
| Users | 54 | 9,159 | 2,060 active |
| Median ratings/user | 3 | 3 | 2 |
| Mean ratings/user | 3.5 | 4.3 | 3.4 |
| Max ratings/user | 8 | 165 | 30 |
| Rating mean | — | 3.96 | 3.42 |
| Rating std | — | 1.49 | 1.70 |
| Gini coefficient | — | 0.692 | 0.269 |

### 3.5 Validation Statement

Simulated dense users (3+ ratings, mean 3.4 ratings/user) are statistically comparable to real Amazon Beauty dense users (3+ ratings, mean 4.3 ratings/user) in interaction density. This provides empirical grounding for using the simulation to evaluate model behavior under realistic dense-user conditions.

### 3.6 Known Limitations

**Circularity:** The simulation generates ratings from note-space cosine similarity. The content-based recommender being evaluated also uses note-space cosine similarity. Evaluation metrics therefore measure internal consistency, not real-world performance. Relative model rankings are more meaningful than absolute Precision@K values.

**Gini gap:** Simulated item rating distribution has Gini=0.269 vs. Amazon Beauty's 0.692. Achieving Amazon-level concentration requires ~600,000 users — a 228GB similarity matrix. The practical consequence is that cold-start severity for niche items is underestimated.

**Single-seed preferences:** Each user has a unimodal Gaussian preference around one seed perfume. Real users may have multimodal preferences (e.g. loving both fresh aquatics and heavy orientals simultaneously).

**Missing quality signal:** Aggregate Rating Value is not incorporated into the generative model. A perfume with a 4.8 average from thousands of reviewers is treated identically to one with a 3.1 average.

**Perfume-category sparsity:** On real platforms, only 0.33% of perfume reviewers rate 3+ perfumes (54 users on Amazon). Held-out CF evaluation on real perfume data is structurally impossible at this catalog scale. The simulation explicitly models a denser regime to enable evaluation.

---

## 4. Reproducibility

All data pipeline steps are fully reproducible:

```
notebooks/01_eda.ipynb          — dataset loading, merging, cleaning, EDA
notebooks/03_simulate_users.ipynb — synthetic interaction generation
data/processed.csv              — merged, cleaned perfume catalog
data/simulated_ratings.csv      — synthetic user-item interactions
```

Random seed: `np.random.seed(42)` in all notebooks.

---

## 5. References

- Hou, Y. et al. (2024). Bridging Language and Items for Retrieval and Recommendation. *arXiv:2403.03952*. [Amazon Reviews 2023]
- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *ICWSM*.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer*, 42(8), 30–37.
- Peng, K. (2017). How I Built a Recommendation System for Fragrance. *Medium / Towards Data Science*.
