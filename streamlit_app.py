import re
import numpy as np
import pandas as pd
import nltk
import streamlit as st
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Malaysian News Sentiment Analyser | TNL6323",
    page_icon="🇲🇾",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .hero {
    background: linear-gradient(135deg, #003479, #1a56a8);
    border-radius: 1rem;
    padding: 2rem 1.75rem;
    color: white;
    margin-bottom: 1.5rem;
  }
  .hero h1 { font-size: 1.7rem; font-weight: 800; margin-bottom: .4rem; }
  .hero p  { opacity: .85; font-size: .95rem; margin: 0; }

  .model-card {
    background: white;
    border: 1.5px solid #e2e8f0;
    border-radius: .9rem;
    padding: 1.2rem;
    text-align: center;
    height: 100%;
  }
  .model-title {
    font-size: .72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .7px;
    color: #64748b;
    margin-bottom: .5rem;
  }
  .badge-positive { background:#dcfce7; color:#16a34a; border-radius:.5rem;
                    padding:.45rem 1rem; font-weight:700; font-size:1.05rem;
                    display:inline-block; margin:.4rem 0; }
  .badge-neutral  { background:#dbeafe; color:#2563eb; border-radius:.5rem;
                    padding:.45rem 1rem; font-weight:700; font-size:1.05rem;
                    display:inline-block; margin:.4rem 0; }
  .badge-negative { background:#fee2e2; color:#dc2626; border-radius:.5rem;
                    padding:.45rem 1rem; font-weight:700; font-size:1.05rem;
                    display:inline-block; margin:.4rem 0; }
  .metric-small { font-size:.72rem; color:#94a3b8; margin-top:.3rem; }
  .best-tag { background:#FFD100; color:#78350f; border-radius:.35rem;
              font-size:.65rem; font-weight:700; padding:.1rem .4rem; }
  .consensus-box {
    border-radius: .9rem;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .processed-box {
    background: #f8fafc;
    border: 1.5px dashed #cbd5e1;
    border-radius: .7rem;
    padding: .8rem 1rem;
    font-family: monospace;
    font-size: .82rem;
    color: #475569;
    word-break: break-word;
  }
  /* hide default streamlit header menu */
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── NLTK downloads (cached — runs only once per session) ──────────────────────
@st.cache_resource(show_spinner=False)
def _download_nltk():
    for res in [
        "punkt", "punkt_tab", "stopwords", "wordnet",
        "omw-1.4", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    ]:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass


_download_nltk()

# ── Preprocessing ──────────────────────────────────────────────────────────────
NEGATION_WORDS = {
    "not", "no", "nor", "never", "neither",
    "without", "against", "hardly", "barely", "scarcely",
    "nothing", "nowhere", "nobody",
}
_STOPWORDS  = set(stopwords.words("english")) - NEGATION_WORDS
_lemmatizer = WordNetLemmatizer()


def _penn_to_wn(tag: str) -> str:
    return {"J": wordnet.ADJ, "V": wordnet.VERB,
            "N": wordnet.NOUN, "R": wordnet.ADV}.get(tag[0], wordnet.NOUN)


def preprocess(raw: str) -> str:
    text = str(raw).lower()
    text = re.sub(r"<[^>]+>",           " ", text)
    text = re.sub(r"&[a-z]+;",          " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b([emi])-(\w+)",   r"\1\2", text)
    text = re.sub(r"-",                 " ", text)
    text = re.sub(r"[^a-z0-9\s]",       "", text)
    text = re.sub(r"\s+",               " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOPWORDS]
    if tokens:
        tagged = pos_tag(tokens)
        tokens = [_lemmatizer.lemmatize(w, _penn_to_wn(t)) for w, t in tagged]
    return " ".join(tokens)


# ── Model training (cached — trains only once) ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    import pathlib
    candidates = [
        "cleaned_news_data_v2_out.csv",
        "Training/Cleaned Dataset/cleaned_news_data_v2_out.csv",
    ]
    path = next((p for p in candidates if pathlib.Path(p).exists()), None)
    if path is None:
        st.error("Dataset CSV not found. Make sure cleaned_news_data_v2_out.csv is in the repo root.")
        st.stop()

    df = pd.read_csv(path, encoding="utf-8-sig")
    col = next((c for c in ["TEXT_PROCESSED", "TEXT_CLEANED", "TEXT"] if c in df.columns), None)
    df = df.dropna(subset=[col, "SENTIMENT"]).copy()
    df["SENTIMENT"] = df["SENTIMENT"].str.strip().str.upper()
    df = df[df["SENTIMENT"].isin({"POSITIVE", "NEUTRAL", "NEGATIVE"})]

    x, y = df[col], df["SENTIMENT"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    specs = {
        "naive_bayes":         MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "linear_svm":          LinearSVC(random_state=42),
    }

    models, metrics = {}, {}
    for name, est in specs.items():
        pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))), ("clf", est)])
        pipe.fit(x_train, y_train)
        preds = pipe.predict(x_test)
        acc = accuracy_score(y_test, preds)
        _, _, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0
        )
        models[name]  = pipe
        metrics[name] = {"accuracy": round(float(acc), 4), "macro_f1": round(float(f1), 4)}

    return models, metrics


# ── Confidence helper ──────────────────────────────────────────────────────────
def get_confidence(pipe, processed: str) -> dict | None:
    clf, tfidf = pipe.named_steps["clf"], pipe.named_steps["tfidf"]
    try:
        if hasattr(clf, "predict_proba"):
            proba   = pipe.predict_proba([processed])[0]
            classes = clf.classes_
        else:
            X_vec  = tfidf.transform([processed])
            scores = clf.decision_function(X_vec)[0]
            exp    = np.exp(scores - np.max(scores))
            proba  = exp / exp.sum()
            classes = clf.classes_
        return {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}
    except Exception:
        return None


# ── Sentiment config ───────────────────────────────────────────────────────────
SCFG = {
    "POSITIVE": {"icon": "😊", "badge": "badge-positive", "label": "Positive", "color": "#16a34a"},
    "NEUTRAL":  {"icon": "😐", "badge": "badge-neutral",  "label": "Neutral",  "color": "#2563eb"},
    "NEGATIVE": {"icon": "😟", "badge": "badge-negative", "label": "Negative", "color": "#dc2626"},
}

MODEL_META = {
    "naive_bayes":         {"name": "Naive Bayes",        "best": False},
    "logistic_regression": {"name": "Logistic Regression","best": False},
    "linear_svm":          {"name": "Linear SVM",         "best": True},
}

EXAMPLES = [
    ("📈 Economy Growth",
     "Malaysia's economy surged by 5.2% in the second quarter, outpacing regional peers as export revenues and consumer spending reached record highs, boosting investor confidence across all major sectors."),
    ("📉 Cost of Living",
     "Rising cost of living continues to burden Malaysians as inflation hits a 14-year high, with food prices surging 8.3% and housing affordability declining sharply in Kuala Lumpur and Selangor."),
    ("📰 Policy Update",
     "The Malaysian government announced new digital economy policies focusing on AI infrastructure investment, broadband expansion and technology talent development in a ministerial press briefing today."),
    ("⛽ Petronas Record",
     "Petronas recorded its highest-ever quarterly profit of RM 21 billion, with the national oil company attributing the record results to disciplined cost management and strong global energy demand."),
    ("🌊 Flood Crisis",
     "Flash floods in Johor have displaced over 15,000 residents, with rescue operations ongoing as authorities warn of continued heavy rainfall throughout the week, causing severe disruption to businesses and schools."),
]

# ── Session state ──────────────────────────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

def _set_example(text: str):
    st.session_state["input_text"] = text

# ==============================================================================
# UI
# ==============================================================================
st.markdown("""
<div class="hero">
  <h1>Malaysian News Sentiment Analyser</h1>
  <p>Paste a Malaysian news headline or article — three ML models will classify its
  sentiment as <strong>Positive</strong>, <strong>Neutral</strong>, or <strong>Negative</strong>.</p>
</div>
""", unsafe_allow_html=True)

# ── Example buttons ────────────────────────────────────────────────────────────
st.markdown("**Try an example:**")
cols_ex = st.columns(len(EXAMPLES))
for i, (label, text) in enumerate(EXAMPLES):
    cols_ex[i].button(
        label,
        use_container_width=True,
        on_click=_set_example,
        args=(text,),
    )

# ── Text input  (key binds widget ↔ session_state automatically) ───────────────
user_text = st.text_area(
    "Enter news text",
    key="input_text",
    height=140,
    placeholder="e.g. Malaysia's GDP grew by 4.3% in Q2, driven by strong manufacturing exports…",
    label_visibility="collapsed",
)

analyse = st.button(
    "🔍  Analyse Sentiment",
    type="primary",
    disabled=len(user_text.strip()) < 5,
    use_container_width=True,
)

# ── Run prediction ─────────────────────────────────────────────────────────────
if analyse and len(user_text.strip()) >= 5:
    with st.spinner("Loading models and analysing…"):
        models, metrics = load_models()
        processed = preprocess(user_text)

    results = {}
    for name, pipe in models.items():
        pred = pipe.predict([processed])[0]
        conf = get_confidence(pipe, processed)
        results[name] = {"pred": pred, "conf": conf}

    votes = [r["pred"] for r in results.values()]
    consensus = max(set(votes), key=votes.count)
    cfg = SCFG.get(consensus, {})

    st.divider()

    # Consensus banner
    st.markdown(f"""
    <div class="consensus-box" style="background:{cfg.get('color','#94a3b8')}18;
         border:2px solid {cfg.get('color','#94a3b8')}55;">
      <span style="font-size:2.4rem">{cfg.get('icon','🤔')}</span>
      <div>
        <div style="font-size:.75rem;font-weight:700;text-transform:uppercase;
             letter-spacing:.6px;color:#64748b;">Overall Consensus</div>
        <div style="font-size:1.5rem;font-weight:800;color:{cfg.get('color','#1e293b')}">
          {cfg.get('label', consensus)}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Model cards
    st.markdown("#### Model Predictions")
    col1, col2, col3 = st.columns(3)
    card_cols = {"naive_bayes": col1, "logistic_regression": col2, "linear_svm": col3}

    for name, col in card_cols.items():
        r    = results[name]
        m    = metrics[name]
        meta = MODEL_META[name]
        scfg = SCFG.get(r["pred"], {})
        best_tag = '<span class="best-tag">★ BEST</span>' if meta["best"] else ""

        with col:
            st.markdown(f"""
            <div class="model-card">
              <div class="model-title">{meta['name']} {best_tag}</div>
              <div class="{scfg.get('badge','')}">
                {scfg.get('icon','')} {r['pred']}
              </div>
              <div class="metric-small">
                Acc: {m['accuracy']*100:.1f}% &nbsp;|&nbsp; F1: {m['macro_f1']:.3f}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if r["conf"]:
                st.markdown("")
                for sent in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
                    pct = r["conf"].get(sent, 0)
                    s   = SCFG[sent]
                    st.caption(f"{s['label']}  **{pct:.1f}%**")
                    st.progress(int(pct))

    # Preprocessed text
    st.divider()
    st.markdown("#### 🔧 Preprocessed Input")
    st.caption("Text after cleaning → tokenisation → stopword removal → lemmatisation:")
    st.markdown(f'<div class="processed-box">{processed or "(empty after preprocessing)"}</div>',
                unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin-top:3rem;border-color:#e2e8f0"/>
<p style="text-align:center;font-size:.78rem;color:#94a3b8;">
  TNL6323 Natural Language Processing · Trimester March/April 2026<br>
  Malaysian News Articles Sentiment Analysis · Naive Bayes · Logistic Regression · Linear SVM
</p>
""", unsafe_allow_html=True)
