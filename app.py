import ast
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from rapidfuzz import process
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="The Common Nose", page_icon="👃", layout="centered")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=Source+Sans+3:wght@300;400&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: #f7f5f0;
    color: #1c1c1c;
}

.site-title {
    font-family: 'EB Garamond', serif;
    font-size: 3rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    margin-bottom: 0;
    line-height: 1.1;
}
.site-subtitle {
    font-family: 'EB Garamond', serif;
    font-style: italic;
    font-size: 1.15rem;
    color: #888;
    margin-top: 0.2rem;
    margin-bottom: 0.5rem;
}
.intro-text {
    font-size: 0.95rem;
    color: #555;
    line-height: 1.6;
    margin-bottom: 2rem;
    max-width: 560px;
}
.section-label {
    font-family: 'EB Garamond', serif;
    font-size: 1.1rem;
    font-weight: 500;
    color: #333;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    letter-spacing: 0.03em;
}
.because-label {
    font-family: 'EB Garamond', serif;
    font-style: italic;
    font-size: 1rem;
    color: #888;
    margin-top: 1.2rem;
    margin-bottom: 0.4rem;
}
.card {
    background: #fff;
    border: 1px solid #e4dfd6;
    border-radius: 6px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.5rem;
}
.card-name {
    font-size: 0.97rem;
    font-weight: 500;
    margin-bottom: 0.4rem;
}
.pill {
    display: inline-block;
    border-radius: 20px;
    padding: 2px 9px;
    font-size: 0.7rem;
    margin-right: 4px;
    margin-top: 3px;
    font-weight: 400;
}
.pill-lg {
    padding: 3px 11px;
    font-size: 0.78rem;
    font-weight: 600;
}
.fav-item {
    background: #fff;
    border: 1px solid #d6cfc4;
    border-left: 3px solid #a89880;
    border-radius: 5px;
    padding: 0.55rem 0.9rem;
    margin-bottom: 0.35rem;
    font-size: 0.88rem;
}
.search-result {
    background: #fff;
    border: 1px solid #e4dfd6;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.35rem;
    cursor: pointer;
    font-size: 0.9rem;
}
.divider {
    border: none;
    border-top: 1px solid #e4dfd6;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Accord family colours ─────────────────────────────────────────────────────
ACCORD_COLOURS = {
    # Fresh: teal
    'fresh': ('#e0f4f7', '#0e7490'),
    'aquatic': ('#e0f4f7', '#0e7490'),
    'ozonic': ('#e0f4f7', '#0e7490'),
    'marine': ('#e0f4f7', '#0e7490'),
    'salty': ('#e0f4f7', '#0e7490'),
    'clean': ('#e0f4f7', '#0e7490'),
    # Floral: mauve
    'floral': ('#fce7f3', '#9d174d'),
    'rose': ('#fce7f3', '#9d174d'),
    'white floral': ('#fce7f3', '#9d174d'),
    'iris': ('#fce7f3', '#9d174d'),
    'violet': ('#fce7f3', '#9d174d'),
    'jasmine': ('#fce7f3', '#9d174d'),
    'orange blossom': ('#fce7f3', '#9d174d'),
    'lily': ('#fce7f3', '#9d174d'),
    'peony': ('#fce7f3', '#9d174d'),
    # Woody: warm brown
    'woody': ('#fdf2e4', '#92400e'),
    'cedar': ('#fdf2e4', '#92400e'),
    'sandalwood': ('#fdf2e4', '#92400e'),
    'vetiver': ('#fdf2e4', '#92400e'),
    'oud': ('#f5e6d8', '#7c2d12'),
    # Citrus: yellow
    'citrus': ('#fef9c3', '#854d0e'),
    'fruity': ('#fef3c7', '#b45309'),
    'tropical': ('#fef3c7', '#b45309'),
    # Oriental/Warm: clay warhammer brown
    'vanilla': ('#fde8c8', '#92400e'),
    'amber': ('#fde8c8', '#92400e'),
    'balsamic': ('#fde8c8', '#92400e'),
    'sweet': ('#fde8c8', '#92400e'),
    'gourmand': ('#fde8c8', '#92400e'),
    'caramel': ('#fde8c8', '#92400e'),
    'chocolate': ('#fde8c8', '#92400e'),
    'honey': ('#fde8c8', '#92400e'),
    # Spicy: maroon
    'warm spicy': ('#fee2e2', '#991b1b'),
    'fresh spicy': ('#fee2e2', '#991b1b'),
    'spicy': ('#fee2e2', '#991b1b'),
    # Aromatic/Fougère: green
    'aromatic': ('#dcfce7', '#166534'),
    'lavender': ('#dcfce7', '#166534'),
    'herbal': ('#dcfce7', '#166534'),
    'fougere': ('#dcfce7', '#166534'),
    # Earthy/Chypre: olive
    'earthy': ('#ecfccb', '#3f6212'),
    'mossy': ('#ecfccb', '#3f6212'),
    'green': ('#ecfccb', '#3f6212'),
    'patchouli': ('#ecfccb', '#3f6212'),
    # Leather: dark tan
    'leather': ('#f3e8d0', '#78350f'),
    'suede': ('#f3e8d0', '#78350f'),
    # Musky/Powdery: gray
    'musky': ('#f3f4f6', '#4b5563'),
    'powdery': ('#f3f4f6', '#6b7280'),
    'animalic': ('#f3f4f6', '#4b5563'),
    'musk': ('#f3f4f6', '#4b5563'),
    'aldehydic': ('#f3f4f6', '#6b7280'),
}
DEFAULT_COLOUR = ('#f3f4f6', '#6b7280')

def accord_pill(accord, large=False):
    bg, fg = ACCORD_COLOURS.get(accord.lower(), DEFAULT_COLOUR)
    cls = 'pill pill-lg' if large else 'pill'
    return f'<span class="{cls}" style="background:{bg};color:{fg}">{accord}</span>'

# ── Assets ────────────────────────────────────────────────────────────────────
HF_REPO = "anvo2/perfume-rec-assets"

@st.cache_resource
def load_assets():
    df = pd.read_csv(hf_hub_download(HF_REPO, "data/catalog_70k.csv", repo_type="dataset"))
    accord_matrix = np.load(hf_hub_download(HF_REPO, "models/accord_matrix_within_pos.npy", repo_type="dataset"))
    candidate_lookup = joblib.load(hf_hub_download(HF_REPO, "models/accord_candidate_lookup.pkl", repo_type="dataset"))
    return df, accord_matrix, candidate_lookup

df, accord_matrix, candidate_lookup = load_assets()

# ── Helpers ───────────────────────────────────────────────────────────────────
GENDER_SYMBOL = {"for women": "♀", "for men": "♂", "for women and men": "⚥"}

def parse_accords(accords):
    if isinstance(accords, list):
        return accords
    if isinstance(accords, str):
        try:
            return ast.literal_eval(accords)
        except Exception:
            return []
    return []

def clean_name(row):
    name = row['Name']
    gender = row['Gender'] if pd.notna(row['Gender']) else ''
    return name.replace(gender, '').strip()

def gender_sym(row):
    g = row['Gender'] if pd.notna(row['Gender']) else ''
    return GENDER_SYMBOL.get(g.strip(), '')

def find_perfumes(query, threshold=45, limit=6):
    names = df['Name'].tolist()
    results = process.extract(query, names, limit=limit)
    return [(r[2], r[0]) for r in results if r[1] >= threshold]

def render_card(idx):
    row = df.iloc[idx]
    name = clean_name(row)
    sym = gender_sym(row)
    accords = parse_accords(row['accords'])
    pills = ''.join([
        accord_pill(a, large=(i < 2)) for i, a in enumerate(accords[:6])
    ])
    st.markdown(f"""
    <div class="card">
        <div class="card-name">{name} <span style="color:#aaa;font-weight:300">{sym}</span></div>
        <div>{pills}</div>
    </div>
    """, unsafe_allow_html=True)

def recommend_grouped(liked_ids, k=10):
    seen = set(liked_ids)
    grouped = {}
    recs_per_input = max(3, k // len(liked_ids))
    for liked_id in liked_ids:
        input_accords = parse_accords(df['accords'].iloc[liked_id])[:2]
        candidate_sets = [
            set(candidate_lookup.get(acc, np.array([])).tolist())
            for acc in input_accords
        ]
        candidates = list(candidate_sets[0].union(*candidate_sets[1:]) - seen)
        if not candidates:
            grouped[liked_id] = []
            continue
        candidates = np.array(candidates)
        sims = cosine_similarity([accord_matrix[liked_id]], accord_matrix[candidates])[0]
        top = [candidates[i] for i in np.argsort(sims)[::-1][:recs_per_input]]
        grouped[liked_id] = top
        seen.update(top)
    return grouped

# ── Session state ─────────────────────────────────────────────────────────────
if "favourites" not in st.session_state:
    st.session_state.favourites = []
if "recs" not in st.session_state:
    st.session_state.recs = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="site-title">The Common Nose</div>', unsafe_allow_html=True)
st.markdown('<div class="site-subtitle">Tell us your 3 favourite perfumes.</div>', unsafe_allow_html=True)
st.markdown("""
<p class="intro-text">
Search for up to three perfumes you love. We'll analyse their accord profiles —
the olfactory building blocks that define how a fragrance smells — and find you
similar fragrances from a catalogue of 68,000 perfumes.
No ratings. No popularity bias. Just scent.
</p>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Search ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Find a perfume</div>', unsafe_allow_html=True)
query = st.text_input("", placeholder="e.g. Black Orchid, Sauvage, Flowerbomb…", key="search", label_visibility="collapsed")

if query:
    results = find_perfumes(query)
    if results:
        names_list = [r[1].replace(df.iloc[r[0]]['Gender'] if pd.notna(df.iloc[r[0]]['Gender']) else '', '').strip()
                      for r in results]
        choice = st.radio("Select the perfume you mean:", names_list, key="radio_choice", label_visibility="visible")
        chosen_idx = results[names_list.index(choice)][0]

        render_card(chosen_idx)

        already = any(f[0] == chosen_idx for f in st.session_state.favourites)
        if already:
            st.caption("Already in your favourites.")
        elif len(st.session_state.favourites) >= 3:
            st.caption("You've added 3 perfumes. Please clear one to add another.")
        else:
            if st.button("Add to favourites"):
                row = df.iloc[chosen_idx]
                st.session_state.favourites.append((chosen_idx, clean_name(row), parse_accords(row['accords'])))
                st.session_state.recs = None
                st.rerun()
    else:
        st.caption("No matches found. Try a shorter or different name.")

# ── Favourites ────────────────────────────────────────────────────────────────
if st.session_state.favourites:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Your favourites</div>', unsafe_allow_html=True)
    for _, fav_name, fav_accords in st.session_state.favourites:
        pills = ''.join([accord_pill(a, large=(i < 2)) for i, a in enumerate(fav_accords[:4])])
        st.markdown(f'<div class="fav-item"><strong>{fav_name}</strong><br>{pills}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Clear all"):
            st.session_state.favourites = []
            st.session_state.recs = None
            st.rerun()
    with col2:
        if st.button("Find similar perfumes →", type="primary"):
            liked_ids = [f[0] for f in st.session_state.favourites]
            st.session_state.recs = recommend_grouped(liked_ids, k=10)
            st.rerun()

# ── Recommendations ───────────────────────────────────────────────────────────
if st.session_state.recs:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Recommendations</div>', unsafe_allow_html=True)
    fav_names = {f[0]: f[1] for f in st.session_state.favourites}
    for seed_id, rec_list in st.session_state.recs.items():
        if not rec_list:
            continue
        seed_name = fav_names.get(seed_id, "your selection")
        st.markdown(f'<div class="because-label">Because you liked <em>{seed_name}</em></div>', unsafe_allow_html=True)
        for idx in rec_list:
            render_card(idx)