# The Common Nose
### A Data-Driven Perfume Recommendation System

A content-based recommendation system built on 68,000 fragrances from Fragrantica. Tell it three perfumes you love. It will find you similar ones using accord-based vector similarity filtered by a learned accord compatibility matrix, with no ratings and no popularity bias.

**Live demo:** [the-common-nose.streamlit.app](https://the-common-nose.streamlit.app/)

---

## Background: Notes vs Accords

Perfumes are described in two distinct vocabularies that are easy to conflate.

**Notes** are the raw ingredients listed by the fragrance house and sometimes perfume reviewers: rose, oud, bergamot, sandalwood. They are company-supplied, ingredient-level descriptors. Two perfumes can share many notes and smell completely different due to differences in the proportions, blending, and concentration all determine the final character. The dataset came with a whopping 1671 notes across 24000 perfumes.

**Accords** are the emergent olfactory character of a finished perfume: floral, woody, oriental, fresh. They are crowd-sourced on Fragrantica by users describing how a perfume actually smells, not what went into it. Where notes describe composition, accords describe perception. Among 68000 perfumes, we totalled at 90 accords.

This distinction turns out to be statistically significant and was the backbone of our recommender system.

Upon spritzing a perfume, we attempt to find 'anchors' in our minds of what the perfume smells like. These different anchors are then usually formalised as 'accords', which, broadly speaking, is our attempt of grouping up synergistic and similar notes into one coherent olfactory experience. For example, most fruity notes, regardless of the specific kind of fruits, get labelled as 'fruity'. A perfume that smells like predominantly strawberries or lychees both end up being 'fruity'. Gardenia, jasmine and angelica usually become 'white florals'. Once an anchor (accord) is figured out, we move onto the next ('this perfume smells fruity *and* powdery'), attempting to locate the next grouping of note that produces a different olfactory experience while 'skipping over' all the notes that make up the accord we just finished processing. 

---

## What the System Does

Each of the 68,000 perfumes in the catalog is represented as a 90-dimensional vector over the accord space, weighted by the positional prominence of each accord within that perfume. A learned transition probability matrix filters the candidate pool to accords that are empirically compatible with the input before ranking by cosine similarity. Recommendations are generated per input seed rather than from a centroid, preserving the distinct character of each liked perfume.

---

## Process

### 1. Notes are the wrong feature space (r = 0.262)

The starting, most naive assumption (that shared ingredients imply similar smell) was tested directly. Pairwise note similarity and pairwise accord similarity were computed across a 24k catalog with both note and accord data, and their correlation measured.

**Result: r = 0.262, R² = 0.069.** Note similarity explains only 6.9% of accord variance. Why? 

For example, *Shalimar* by Guerlain and *Flowerbomb* by Viktor & Rolf share several prominent notes, including vanilla, iris, rose. However, their accord profiles are starkly different (Shalimar: oriental, powdery, vanilla; Flowerbomb: floral, sweet, powdery). The same ingredients, combined differently and in different quantities, then mixed with different supporting ingredients, ended up producing two different olfactory experience.

This implication is structural: note-based features cannot reliably predict olfactory similarity. The accord space (how a perfume is perceived) is the correct feature space.


### 2. TF-IDF improves MRR for the wrong reason

TF-IDF weighting on notes improved MRR from 0.0009 to 0.0076, an 8× improvement that looked promising. Before accepting it, the mechanism of improvement was investigated.

TF-IDF down-weights common notes (musk, woody, amber, which appears in thousands of perfumes) and up-weights rare ones (a specific spice or unusual botanical). This means TF-IDF-weighted vectors are dominated by the most idiosyncratic ingredients,  pulling recommendations toward niche, rare perfumes rather than popular ones. In a catalog where the popularity baseline returns only the most-rated perfumes, suggesting rare perfumes mechanically improves Recall@100, but not because the recommendations are more similar, simply because they are more spread out.

Note→accord correlation *dropped* from r = 0.262 to r = 0.240 under TF-IDF weighting. A method that improves an evaluation metric while degrading the underlying signal is not a real improvement, which led us to reject TF-IDF.

### 3. Position encodes importance in accord list

Accord lists on Fragrantica are ordered by prominence. Early accords (citrus, fruity, floral) (accords that define what a scent is, these are usually what the nose picks up first) have a mean position of ~3; late accords (balsamic, animalic, soapy) have a mean position of ~6–7. Position is informative and should be encoded in the representation.

A global position weighting scheme (weight = 1 / (mean_position + 1), derived from the corpus mean position of each accord) was initially implemented. Analysis of the resulting vectors revealed an oversight: **7 perfumes with different primary accords but identical accord sets received identical vectors**. Because global weights are fixed per accord, the same 10 accords in a different internal order produce the same weighted sum.

The fix was within-perfume positional weighting:

```
weight = 1 - (position / n_accords)
```

Weights are now relative to each perfume's own accord list. Position 0 always receives weight 1.0; the last position always receives weight approaching 0. This preserves ordinal importance without collapsing distinct profiles.

### 4. K-Means cannot cluster accord space, but top-2 accord can!

To test whether the accord manifold contains discrete olfactory families, K-Means was run across k = 5–34 on multiple representations: all accords, top-4, top-2, SVD-compressed vectors, and agglomerative clustering with Ward linkage. Best silhouette score: **0.337 on top-2 accords only**. The full accord space returned silhouette = 0.075, which was basically random.

This has two implications: hard family assignments (floral, oriental, woody) would be arbitrary. The accord manifold is continuous, not discretely clusterable. A hard filter ("only recommend florals to floral lovers") would be unjustified by the data.

Second, and more importantly: the only subspace with meaningful structure is the **top-2 accords**. Silhouette rising from 0.075 to 0.337 when restricting to top-2 suggests that the primary and secondary accord together are the main character-defining features of a perfume. A floral-woody and a floral-citrus are more similar to each other than either is to an oud-leather, even though all four share at least one accord. This finding directly motivates using the top-2 accords as the basis for the compatibility filter. This can also be seen by the fact that the top two accord on most perfumes on Fragrantica have overwhelmingly more votes than any other accord of the same perfume (this quantity was not available in the datasets)

### 5. The compatibility filter: learning accord co-occurrence

Rather than a hard filter by family, a **transition probability matrix** was learned from co-occurrence across the full 68k catalog:

```
P(accord_b in top-2 | accord_a in top-2)
```

This gives soft, empirically grounded compatibility scores between any pair of accords:

| Input accords | Candidate accords | Compatibility |
|---|---|---|
| floral, aromatic | floral, citrus | 0.503 |
| floral, aromatic | oud, leather | 0.014 |

A 36× difference between compatible and incompatible families. Candidates must pass a minimum compatibility threshold (0.05) against the input's top-2 accords before being ranked by cosine similarity. This ensures olfactory coherence without imposing arbitrary boundaries.

### 6. Word2Vec captures semantic note relationships (r = 0.460)

For note-based constraint search, multi-hot note vectors were a poor proxy: fruity-fruity pair similarity was only 0.13, meaning mango and peach were treated as essentially unrelated. Word2Vec trained on the note co-occurrence corpus (notes appearing together in the same perfume) improved note→accord correlation from r = 0.262 to **r = 0.460** (75% improvement) and raised fruity-fruity similarity to 0.82, correctly capturing that mango ≈ pineapple ≈ peach in olfactory space, since they co-occur with the same surrounding notes.

### 7. Collaborative filtering is structurally impossible on this data

Amazon Beauty reviews were analysed as a potential source of real user preference data. However, there were only **54 users with 3+ perfume ratings** (0.33% of reviewers), maximum 8 ratings per user. CF evaluation on this data would be circular. Content-based recommendation with session-based Bayesian personalisation via Dirichlet posteriors over the accord space is the principled choice given the data constraints.

---

## Evaluation

Evaluated against accord-based simulated ratings (k=100, MRR metric):

| Model | MRR | Notes |
|---|---|---|
| Popularity baseline | 0.0009 | Floor |
| Multi-hot notes | 0.0009 | No signal over baseline |
| TF-IDF notes | 0.0076 | Diversity gain, not precision |
| Note SVD (n=10) | 0.0014 | Worse than TF-IDF |
| **Accord vectors (final)** | **0.0180** | **20× over baseline** |

All evaluation is on simulated data generated from accord similarity. Results should be treated as internal consistency metrics. Primary qualitative validation: 10/10 recommendations for a floral-aromatic seed had either floral or aromatic in their top-2 accords.

---

## Stack

```
pandas · numpy · scikit-learn · gensim (Word2Vec) · rapidfuzz · streamlit · huggingface_hub · joblib
```

---

## Running Locally

```bash
git clone https://github.com/anvo2-git/perfume-rec
cd perfume-rec
uv sync
uv run streamlit run app.py
```

Models and data (~500MB) are downloaded automatically from Hugging Face on first run.

## Contact

Built by Ian Vo. Feedback and questions welcome at [avo2@uchicago.edu](mailto:avo2@uchicago.edu). Let me know if you need a perfume rec! :)