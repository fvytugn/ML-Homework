import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
df = pd.read_csv("machine_failure_dataset.csv")
print("\n=== Statistics Summary (Original Raw Data) ===\n")
print(df.describe(include='all'))
os.makedirs("data/original_data", exist_ok=True)
os.makedirs("data/preprocessing", exist_ok=True)
os.makedirs("data/result", exist_ok=True)
df.to_csv("data/original_data/original_dataset.csv", index=False)
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    X = df.drop(columns=['Failure_Risk'])
    y = df['Failure_Risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    pd.DataFrame(X_train_resampled, columns=X.columns).to_csv("data/preprocessing/X_train.csv", index=False)
    pd.DataFrame(X_test, columns=X.columns).to_csv("data/preprocessing/X_test.csv", index=False)
    pd.DataFrame(y_train_resampled, columns=["Failure_Risk"]).to_csv("data/preprocessing/Y_train.csv", index=False)
    pd.DataFrame(y_test, columns=["Failure_Risk"]).to_csv("data/preprocessing/Y_test.csv", index=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    models = { 'SVM': SVC(probability=True),
              'DecisionTree': DecisionTreeClassifier(),
              'RandomForest': RandomForestClassifier(),
              'KNN': KNeighborsClassifier(),
              'NaiveBayes': GaussianNB(),
              'ANN': MLPClassifier(max_iter=1000),
              'LogisticRegression': LogisticRegression(max_iter=1000) }
    scores = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train_resampled)
        y_pred = model.predict(X_test_scaled)
        pd.DataFrame(y_pred).to_csv(f"data/result/predictions_{name}.csv", index=False)
        print(f"\n{name} Classification Report:\n")
        print(classification_report(y_test, y_pred))
        scores[name] = accuracy_score(y_test, y_pred)
        plt.figure(figsize=(10, 6))
        plt.bar(scores.keys(), scores.values())
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("data/result/accuracy_comparison.png")
        plt.close()
        plt.figure(figsize=(15, 10))
        df.hist(bins=20, figsize=(15, 10))
        plt.suptitle("Feature Distributions", fontsize=16)
        plt.tight_layout()
        plt.savefig("data/result/visualization.png")
        plt.close()
