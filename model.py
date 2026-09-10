import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("Cancer_Data.csv")
df.drop("Unnamed: 32", axis=1, inplace=True)
df["diagnosis"] = (df["diagnosis"] == "M").astype(int)

X = df[df.columns[2:]].values
y = df["diagnosis"].values
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=1, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=1, stratify=y_temp)

def make_pipe(k):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ros", RandomOverSampler(random_state=1)),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])

best_k, best_recall = None, -1
for k in range(1, 31, 2):          # k ímpar
    pipe = make_pipe(k).fit(X_train, y_train)
    r = recall_score(y_valid, pipe.predict(X_valid))
    if r > best_recall:
        best_recall, best_k = r, k
print(f"Melhor k = {best_k} (recall no valid = {best_recall:.3f})")

final = make_pipe(best_k).fit(X_train, y_train)
print(classification_report(y_test, final.predict(X_test)))

joblib.dump(final, "knn_pipeline.joblib")