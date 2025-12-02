import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


st.title("📰 Fake News Detection – Simple ML Project")

# -----------------------------
# WARNING MESSAGE (Requested)
# -----------------------------
st.warning("⚠️ Note: This model may occasionally provide inaccurate answers.")

st.write("""
Upload a CSV containing:
- **text** (news content)
- **label** → 1 = real, 0 = fake
""")


# ----------------------------------
# Simple text cleaning
# ----------------------------------
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns or "label" not in df.columns:
        st.error("CSV must contain 'text' and 'label' columns!")
        df = None
    else:
        st.success("Dataset loaded successfully!")
        df["clean_text"] = df["text"].apply(clean_text)

        # ----------------------------------
        # SHOW PREVIEW ONLY WHEN CLICKED
        # ----------------------------------
        if st.button("Show Dataset Preview"):
            st.dataframe(df.head())

        # ----------------------------------
        # TRAIN MODEL BUTTON
        # ----------------------------------
        if st.button("Train Model"):
            X = df["clean_text"]
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            vectorizer = TfidfVectorizer()
            X_train_tfidf = vectorizer.fit_transform(X_train)

            model = MultinomialNB()
            model.fit(X_train_tfidf, y_train)

            X_test_tfidf = vectorizer.transform(X_test)
            preds = model.predict(X_test_tfidf)

            acc = accuracy_score(y_test, preds)

            st.success(f"Model Trained Successfully! Accuracy: {acc*100:.2f}%")

            # Save for predictions
            st.session_state["model"] = model
            st.session_state["vectorizer"] = vectorizer


# ----------------------------------
# MANUAL PREDICTION
# ----------------------------------
st.subheader("🔍 Test a news headline")

user_input = st.text_area("Enter news text:")

if st.button("Predict"):
    if "model" not in st.session_state:
        st.error("Train the model first!")
    else:
        cleaned = clean_text(user_input)
        vect = st.session_state["vectorizer"].transform([cleaned])
        pred = st.session_state["model"].predict(vect)[0]

        if pred == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")


st.write("---")
st.caption("Simple Fake News Project • Streamlit • Naive Bayes")
