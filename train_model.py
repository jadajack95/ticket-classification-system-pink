import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv("data/helpdesk_tickets_dataset_v2.csv")

# Feature column
X = df["text"]

# -----------------------------
# CATEGORY MODEL
# -----------------------------
y_category = df["category"]

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    X, y_category, test_size=0.2, random_state=42
)

category_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

category_model.fit(X_train_cat, y_train_cat)
y_pred_cat = category_model.predict(X_test_cat)

print("CATEGORY MODEL")
print("Accuracy:", accuracy_score(y_test_cat, y_pred_cat))
print("\nClassification Report:\n")
print(classification_report(y_test_cat, y_pred_cat))

joblib.dump(category_model, "models/ticket_category_model.pkl")
print("\nSaved category model to models/ticket_category_model.pkl")


# -----------------------------
# PRIORITY MODEL
# -----------------------------
y_priority = df["priority"]

X_train_pri, X_test_pri, y_train_pri, y_test_pri = train_test_split(
    X, y_priority, test_size=0.2, random_state=42
)

priority_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

priority_model.fit(X_train_pri, y_train_pri)
y_pred_pri = priority_model.predict(X_test_pri)

print("\n\nPRIORITY MODEL")
print("Accuracy:", accuracy_score(y_test_pri, y_pred_pri))
print("\nClassification Report:\n")
print(classification_report(y_test_pri, y_pred_pri))

joblib.dump(priority_model, "models/ticket_priority_model.pkl")
print("\nSaved priority model to models/ticket_priority_model.pkl")