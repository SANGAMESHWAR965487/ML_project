import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------------------------------------------
# 1️⃣ Check current folder and files
# -------------------------------------------------------------------
print("📁 Current directory:", os.getcwd())
print("📄 Files in directory:", os.listdir())

# -------------------------------------------------------------------
# 2️⃣ Load dataset
# -------------------------------------------------------------------
try:
    data = pd.read_csv("news.csv")
    print("✅ news.csv loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: news.csv not found!")
    print("➡️ Please place 'news.csv' in this folder:")
    print(os.getcwd())
    exit()

# -------------------------------------------------------------------
# 3️⃣ Clean and prepare data
# -------------------------------------------------------------------
# Drop rows that have missing 'text' or 'label'
data = data.dropna(subset=["text", "label"])

# Convert text to string and label to integer
data["text"] = data["text"].astype(str)
data["label"] = data["label"].astype(int)

# -------------------------------------------------------------------
# 4️⃣ Split features and labels
# -------------------------------------------------------------------
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------------------
# 5️⃣ Convert text to TF-IDF features
# -------------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -------------------------------------------------------------------
# 6️⃣ Train model
# -------------------------------------------------------------------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# -------------------------------------------------------------------
# 7️⃣ Evaluate model
# -------------------------------------------------------------------
y_pred = model.predict(X_test_tfidf)

print("\n🎯 Model Evaluation Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------------------------------------------
# 8️⃣ Interactive prediction
# -------------------------------------------------------------------
print("\n📰 Fake News Detection — Type a news headline to test.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter a news headline: ").strip()
    if user_input.lower() == "exit":
        print("👋 Exiting program. Stay smart, verify before you share!")
        break

    user_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(user_tfidf)[0]

    if prediction == 1:
        print("✅ This looks like **REAL NEWS**.\n")
    else:
        print("❌ This looks like **FAKE NEWS**.\n")
