# hw3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector
import warnings
warnings.filterwarnings("ignore")

SEED = 17342

# load
df = pd.read_csv("optical_interconnection_network.csv", sep=";", decimal=",")
df = df.dropna(axis=1, how="all")  # some cols are empty bc of trailing ;
df.columns = [c.strip() for c in df.columns]
df = df.dropna()

y = df["Spatial Distribution"]
X = df.drop(columns=["Spatial Distribution"])

# encode catagorical cols
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])
if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)

feature_names = list(X.columns)
X_scaled = StandardScaler().fit_transform(X)
print("Shape:", X_scaled.shape, "Classes:", set(y))

# part 2
models = {
    "Linear Classifier": Perceptron(max_iter=1000, random_state=SEED),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gaussian NB": GaussianNB(),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=SEED),
}
rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=SEED)

print("\nPart 2:")
part2_rows = []
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=rkf, scoring="accuracy", n_jobs=-1)
    print(f"{name:20s} mean={scores.mean():.4f} std={scores.std():.4f}")
    part2_rows.append({"Algorithm": name, "Mean Accuracy": scores.mean(), "Std": scores.std()})
pd.DataFrame(part2_rows).to_csv("part2_results.csv", index=False)

# part 3 - forward selection (faster than exhuastive)
search_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=SEED)
print("\nPart 3:")
part3_rows = []
for name, model in models.items():
    sfs = SequentialFeatureSelector(estimator=model, n_features_to_select="auto",
        direction="forward", scoring="accuracy", cv=search_cv, n_jobs=-1)
    sfs.fit(X_scaled, y)
    mask = sfs.get_support()
    selected = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
    X_sel = X_scaled[:, mask]
    scores = cross_val_score(model, X_sel, y, cv=rkf, scoring="accuracy", n_jobs=-1)
    print(f"{name:20s} mean={scores.mean():.4f} std={scores.std():.4f}  feats={selected}")
    part3_rows.append({"Algorithm": name, "# Features": int(mask.sum()),
        "Best Feature Subset": ", ".join(selected),
        "Mean Accuracy": scores.mean(), "Std": scores.std()})
pd.DataFrame(part3_rows).to_csv("part3_results.csv", index=False)