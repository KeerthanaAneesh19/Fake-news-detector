"""
Fake News Detection using Machine Learning
Author: Keerthana Aneesh
"""

import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

df_fake["class"] = 0
df_true["class"] = 1

df = pd.concat([df_fake, df_true], axis=0)
df = df.drop(["title", "subject", "date"], axis=1)
df = df.sample(frac=1).reset_index(drop=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    return text

df["text"] = df["text"].apply(clean_text)

x = df["text"]
y = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(xv_train, y_train)
    pred = model.predict(xv_test)
    print(f"\n{name} Results:")
    print(classification_report(y_test, pred))

def predict_news(news):
    news = clean_text(news)
    vec = vectorizer.transform([news])
    for name, model in models.items():
        pred = model.predict(vec)[0]
        label = "Fake News" if pred == 0 else "Real News"
        print(f"{name}: {label}")

news = input("Enter news text: ")
predict_news(news)
