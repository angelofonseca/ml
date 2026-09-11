import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("Cancer_Data.csv")
df.drop("Unnamed: 32", axis=1, inplace=True)
df["diagnosis"] = (df["diagnosis"] == "M").astype(int)


X = df[df.columns[2:]].values
y = df["diagnosis"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ros = RandomOverSampler(random_state=1)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train_resampled, y_train_resampled)

print(classification_report(y_test, model.predict(X_test_scaled)))

joblib.dump({"scaler": scaler, "model": model}, "knn_model.joblib")
