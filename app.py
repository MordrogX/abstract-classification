"""Streamlit demo: classify research abstracts as Science, Business or Social Science."""
import re
import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Abstract Classifier", page_icon="📚", layout="wide")


# ---------- Setup (cached so it only runs once) ----------
@st.cache_resource
def load_nlp():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()


@st.cache_resource
def load_artifacts():
    return joblib.load("model/artifacts.joblib")


STOP_WORDS, LEMMATIZER = load_nlp()
art = load_artifacts()


def clean_text(text):
    """Exactly the same cleaning steps as in the notebook."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in STOP_WORDS]
    words = [LEMMATIZER.lemmatize(w) for w in words]
    return " ".join(w for w in words if len(w) > 2)


def predict(texts, model_name):
    """Return predicted labels and class probabilities for a list of abstracts."""
    X = art["vectorizer"].transform([clean_text(t) for t in texts])
    model = art["models"][model_name]
    return model.predict(X), model.predict_proba(X), model.classes_


# ---------- Sidebar ----------
st.sidebar.header("Settings")
model_names = list(art["models"].keys())
best = art["best_model"]
model_name = st.sidebar.selectbox("Choose a model", model_names, index=model_names.index(best))
if model_name == best:
    st.sidebar.success("This is the best model from the notebook.")

st.sidebar.subheader("Test-set performance")
st.sidebar.dataframe(art["metrics"].round(3).set_index("Model"), use_container_width=True)

# ---------- Main page ----------
st.title("📚 Research Abstract Classifier")
st.write("Predict whether a paper belongs to **Science**, **Business** or **Social Science** from its abstract.")

tab_text, tab_file = st.tabs(["✍️ Type an abstract", "📄 Upload a dataset"])

# --- Tab 1: single abstract ---
with tab_text:
    text = st.text_area("Paste an abstract here", height=220)
    if st.button("Classify", type="primary"):
        if len(text.split()) < 10:
            st.warning("Please enter a longer text (at least 10 words).")
        else:
            labels, probs, classes = predict([text], model_name)
            st.subheader(f"Predicted field: {labels[0]}")
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.barh(classes, probs[0], color="#4C72B0")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            st.pyplot(fig)

# --- Tab 2: CSV upload ---
with tab_file:
    st.write("Upload a CSV with a column of abstracts. If it also has a column of true labels, "
             "the app will compute accuracy and a confusion matrix.")
    uploaded = st.file_uploader("CSV file", type="csv")
    use_demo = st.checkbox("Use the bundled demo_test_data.csv instead")

    data = None
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif use_demo:
        data = pd.read_csv("demo_test_data.csv")

    if data is not None:
        text_col = st.selectbox("Column with abstracts", data.columns,
                                index=list(data.columns).index("abstract") if "abstract" in data.columns else 0)
        label_options = ["(none)"] + list(data.columns)
        label_col = st.selectbox("Column with true labels (optional)", label_options,
                                 index=label_options.index("label") if "label" in data.columns else 0)

        if st.button("Run predictions"):
            data = data.dropna(subset=[text_col]).reset_index(drop=True)
            labels, probs, classes = predict(data[text_col].tolist(), model_name)
            out = data.copy()
            out["predicted_label"] = labels
            out["confidence"] = probs.max(axis=1).round(3)

            st.subheader("Predictions")
            st.dataframe(out, use_container_width=True)
            st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")

            st.subheader("Predicted label counts")
            counts = out["predicted_label"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(counts.index, counts.values, color="#55A868")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            if label_col != "(none)":
                y_true = data[label_col].astype(str)
                st.subheader("Evaluation against your labels")
                st.metric("Accuracy", f"{accuracy_score(y_true, labels):.3f}")
                st.text(classification_report(y_true, labels, zero_division=0))

                names = sorted(set(y_true) | set(labels))
                cm = confusion_matrix(y_true, labels, labels=names)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.imshow(cm, cmap="Blues")
                ax.set_xticks(range(len(names)))
                ax.set_yticks(range(len(names)))
                ax.set_xticklabels(names, rotation=30, ha="right")
                ax.set_yticklabels(names)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                for i in range(len(names)):
                    for j in range(len(names)):
                        ax.text(j, i, cm[i, j], ha="center", va="center",
                                color="white" if cm[i, j] > cm.max() / 2 else "black")
                st.pyplot(fig)
