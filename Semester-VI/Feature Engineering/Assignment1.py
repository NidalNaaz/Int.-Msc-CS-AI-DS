from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import LeaveOneOut, LeavePOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np

data = load_breast_cancer()
X = data.data
y = data.target

model = LogisticRegression(max_iter=5000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
holdout_score = model.score(X_test, y_test)
print("Hold-Out Accuracy:", holdout_score)

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

kf_scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("K-Fold Accuracy Scores:", kf_scores)
print("Average K-Fold Accuracy:", kf_scores.mean())

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

skf_scores = cross_val_score(
    model,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)

print("Stratified K-Fold Accuracy Scores:", skf_scores)
print("Average Stratified K-Fold Accuracy:", skf_scores.mean())

loo = LeaveOneOut()

loo_scores = cross_val_score(
    model,
    X,
    y,
    cv=loo,
    scoring="accuracy"
)

print("LOOCV Mean Accuracy:", loo_scores.mean())

lpo = LeavePOut(p=2)

lpo_scores = cross_val_score(
    model,
    X,
    y,
    cv=lpo,
    scoring="accuracy"
)

print("LPO Mean Accuracy:", lpo_scores.mean())
