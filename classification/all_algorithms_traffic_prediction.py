import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
import sys
import os

FEATURE_COLS = [2, 3, 4, 5]   # 0-indexed
TARGET_COL = 6                # 0-indexed

def load_data(data_dir="data"):
    train_path = os.path.join(data_dir, "train_set.csv")
    test_path  = os.path.join(data_dir, "test_set.csv")

    # read
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    X_train = train.iloc[:, FEATURE_COLS]
    y_train = train.iloc[:, TARGET_COL]
    X_test  = test.iloc[:, FEATURE_COLS]
    y_test  = test.iloc[:, TARGET_COL]
    return X_train, y_train, X_test, y_test

def build_preprocessor(X):
    # Treat all provided feature columns as categorical (matches original LabelEncoder+OneHotEncoder usage)
    cat_features = list(range(X.shape[1]))
    pre = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)],
        remainder='drop',
        sparse_threshold=0.0,
    )
    return pre

ALGO_MAP = {
    'naive':  GaussianNB(),
    'knn':    KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2),
    'dt':     DecisionTreeClassifier(criterion='gini', random_state=1000, max_depth=None, min_samples_leaf=5),
    'lr':     LogisticRegression(max_iter=1000, random_state=20),
    'rf':     RandomForestClassifier(n_estimators=25, criterion='entropy', random_state=0),
}

def run(algo_key):
    if algo_key not in ALGO_MAP:
        print(f"Unknown algo '{algo_key}'. Choose from: {', '.join(ALGO_MAP)}")
        sys.exit(1)

    X_train, y_train, X_test, y_test = load_data()
    pre = build_preprocessor(X_train)
    model = ALGO_MAP[algo_key]
    pipe = Pipeline([('pre', pre), ('clf', model)])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"=== {algo_key.upper()} ===")
    print(f"Accuracy = {acc:.4f}")
    print("Cohen's Kappa =", cohen_kappa_score(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

def menu():
    while True:
        print("\nChoose an algorithm:")
        print(" 1) Naive Bayes")
        print(" 2) KNN")
        print(" 3) Decision Tree")
        print(" 4) Logistic Regression")
        print(" 5) Random Forest")
        print(" 0) Exit")
        choice = input("Enter choice: ").strip()
        mapping = {'1':'naive','2':'knn','3':'dt','4':'lr','5':'rf','0':'exit'}
        algo = mapping.get(choice)
        if algo is None:
            print("Invalid choice. Try again.")
            continue
        if algo == 'exit':
            break
        run(algo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=list(ALGO_MAP.keys()), help='Run a single algorithm and exit')
    args = parser.parse_args()
    if args.algo:
        run(args.algo)
    else:
        menu()