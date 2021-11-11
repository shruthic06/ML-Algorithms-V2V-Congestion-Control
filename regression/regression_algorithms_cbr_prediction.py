import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def load_data(csv_path, target_col=None, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)

    # Auto-detect target if not given
    if target_col is None:
        for cand in ['CBR', 'cbr', 'cbr_value', 'target']:
            if cand in df.columns:
                target_col = cand
                break
        else:
            target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split into train/test
    return train_test_split(X, y, test_size=test_size, random_state=random_state), target_col

def build_preprocessor(X):
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols),
        ]
    )

REG_MAP = {
    'ridge': Ridge(alpha=1.0),
    'lasso': Lasso(alpha=0.001, max_iter=10000),
    'enet': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
    'linear': LinearRegression(),
    'rfr': RandomForestRegressor(n_estimators=200, random_state=0),
}

def run(algo, csv_path, target_col=None, test_size=0.2, random_state=42):
    (X_train, X_test, y_train, y_test), target_col = load_data(csv_path, target_col, test_size, random_state)
    pre = build_preprocessor(X_train)
    model = REG_MAP[algo]
    pipe = Pipeline([('pre', pre), ('reg', model)])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"=== {algo.upper()} regression on target '{target_col}' ===")
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    print(f"MAE  : {mean_absolute_error(y_test, y_pred):.6f}")
    print(f"RMSE : {mean_squared_error(y_test, y_pred, squared=False):.6f}")
    print(f"R^2  : {r2_score(y_test, y_pred):.6f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='CBR regression (Ridge/Lasso/ElasticNet/Linear/RF)')
    ap.add_argument('--algo', choices=list(REG_MAP.keys()), default='ridge')
    ap.add_argument('--csv', required=True, help='Path to single dataset (e.g., data/cbrdataset.csv)')
    ap.add_argument('--target', default=None, help='Target column (default: auto-detect)')
    ap.add_argument('--test-size', type=float, default=0.2, help='Fraction for test split (default: 0.2)')
    ap.add_argument('--random-state', type=int, default=42, help='Random seed (default: 42)')
    args = ap.parse_args()

    run(args.algo, args.csv, args.target, args.test_size, args.random_state)
