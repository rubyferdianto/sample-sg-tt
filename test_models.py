"""Test advanced prediction models with rich feature engineering."""
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, BaggingRegressor, HistGradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

# --- Load data ---
df = pd.read_excel("ToTo.xlsx")
df.columns = df.columns.str.strip()
df['Draw'] = df['Draw'].astype(str).str.strip()
valid = df[df['Draw'].apply(lambda x: x.isdigit())].copy()
valid['Draw'] = valid['Draw'].astype(int)
valid = valid.sort_values('Draw').reset_index(drop=True)

num_cols = ['Winning Number 1', '2', '3', '4', '5', '6']
all_numbers = valid[num_cols].values


def build_rich_features(all_numbers, idx, window=5):
    """Build rich feature vector for a given draw index using history."""
    features = []

    # 1. Raw numbers from last `window` draws (flattened)
    start = max(0, idx - window)
    window_data = all_numbers[start:idx]
    for row in window_data:
        features.extend(row.tolist())
    # Pad if not enough history
    while len(features) < window * 6:
        features.insert(0, 25)  # neutral padding

    # 2. Gap features: draws since each number 1-49 last appeared
    recent_nums = all_numbers[max(0, idx - 50):idx]
    for num in range(1, 50):
        found = False
        for lookback in range(len(recent_nums) - 1, -1, -1):
            if num in recent_nums[lookback]:
                gap = len(recent_nums) - 1 - lookback
                features.append(gap)
                found = True
                break
        if not found:
            features.append(50)

    # 3. Frequency features: how often each number appeared in last 20 draws
    recent_20 = all_numbers[max(0, idx - 20):idx]
    freq_counter = Counter()
    for row in recent_20:
        freq_counter.update(row.tolist())
    for num in range(1, 50):
        features.append(freq_counter.get(num, 0))

    # 4. Per-position frequency in last 20 draws
    for pos in range(6):
        pos_counter = Counter()
        for row in recent_20:
            pos_counter[row[pos]] += 1
        # Top 3 most frequent in this position
        top3 = pos_counter.most_common(3)
        for j in range(3):
            features.append(top3[j][0] if j < len(top3) else 0)

    # 5. Statistics of last 3 draws
    last3 = all_numbers[max(0, idx - 3):idx]
    for row in last3:
        features.extend([
            np.mean(row), np.std(row),
            np.max(row) - np.min(row),
            np.sum(row % 2),          # odd count
            np.sum(row <= 25),         # low count
            np.sum(row),              # sum
        ])
    while len(features) < window * 6 + 49 + 49 + 18 + 18:
        features.append(0)

    return features


def deduplicate_prediction(pred_row):
    """Remove duplicate numbers by nudging to nearest unused value."""
    used = set()
    result = []
    for val in pred_row:
        v = int(np.clip(np.round(val), 1, 49))
        if v not in used:
            used.add(v)
            result.append(v)
        else:
            for offset in range(1, 49):
                for candidate in [v + offset, v - offset]:
                    if 1 <= candidate <= 49 and candidate not in used:
                        used.add(candidate)
                        result.append(candidate)
                        break
                else:
                    continue
                break
    return result


# --- Build training data using ALL history up to draw 4157 ---
# Use draws with enough history (at least window=5)
window = 5
cutoff_train = valid[valid['Draw'] == 4157].index[0]
cutoff_test_end = valid[valid['Draw'] == 4162].index[0]

# Training: predict draws from index window..cutoff_train
X_train, y_train = [], []
for idx in range(window, cutoff_train + 1):
    X_train.append(build_rich_features(all_numbers, idx, window))
    y_train.append(all_numbers[idx])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Test: draws 4158-4163 (4163 actual = [7,13,14,17,40,44])
# We need to add 4163 to the data for testing
actual_4163 = [7, 13, 14, 17, 40, 44]

test_draws = []
test_actuals = []
for draw_num in range(4158, 4163):
    idx = valid[valid['Draw'] == draw_num].index[0]
    test_draws.append(draw_num)
    test_actuals.append(all_numbers[idx])
test_draws.append(4163)
test_actuals.append(actual_4163)
test_actuals = np.array(test_actuals)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print(f"Training size: {len(X_train)} samples, {X_train.shape[1]} features")
print(f"Test draws: {test_draws}")
print()

# --- Define models ---
model_defs = {
    'KNN (k=3)': KNeighborsRegressor(n_neighbors=3),
    'KNN (k=7)': KNeighborsRegressor(n_neighbors=7),
    'KNN (k=11)': KNeighborsRegressor(n_neighbors=11),
    'Linear Regression': LinearRegression(),
    'Ridge (a=10)': Ridge(alpha=10),
    'Ridge (a=50)': Ridge(alpha=50),
    'Lasso (a=0.1)': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Bayesian Ridge': BayesianRidge(),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=200, random_state=42, max_depth=15),
    'GradBoost': None,  # per-column
    'HistGradBoost': None,  # per-column
    'Bagging-Ridge': BaggingRegressor(estimator=Ridge(alpha=10), n_estimators=50, random_state=42),
    'SVR-RBF': None,  # per-column with MultiOutput
}

# --- Evaluate each model ---
best_model_name = None
best_score = -1

for name in model_defs:
    preds_all = []
    # Build test features sequentially (using rolling predictions)
    # But for evaluation, use ACTUAL history to build features
    for ti, draw_num in enumerate(test_draws):
        if draw_num <= 4162:
            idx = valid[valid['Draw'] == draw_num].index[0]
            x_feat = build_rich_features(all_numbers, idx, window)
        else:
            # For 4163, use all_numbers up to 4162 index
            x_feat = build_rich_features(all_numbers, cutoff_test_end + 1, window)

        x_scaled = scaler.transform([x_feat])

        if name in ('GradBoost', 'HistGradBoost', 'SVR-RBF'):
            pred_row = []
            for col_idx in range(6):
                if name == 'GradBoost':
                    m = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
                elif name == 'HistGradBoost':
                    m = HistGradientBoostingRegressor(max_iter=200, max_depth=5, random_state=42)
                else:
                    m = SVR(kernel='rbf', C=10)
                m.fit(X_train_scaled, y_train[:, col_idx])
                p = m.predict(x_scaled)[0]
                pred_row.append(p)
            pred_row = deduplicate_prediction(pred_row)
        elif name in ('Lasso (a=0.1)', 'ElasticNet', 'Bayesian Ridge'):
            pred_row = []
            for col_idx in range(6):
                if name == 'Lasso (a=0.1)':
                    m = Lasso(alpha=0.1)
                elif name == 'ElasticNet':
                    m = ElasticNet(alpha=0.1, l1_ratio=0.5)
                else:
                    m = BayesianRidge()
                m.fit(X_train_scaled, y_train[:, col_idx])
                p = m.predict(x_scaled)[0]
                pred_row.append(p)
            pred_row = deduplicate_prediction(pred_row)
        else:
            model = model_defs[name]
            if name == 'Bagging-Ridge':
                model = MultiOutputRegressor(BaggingRegressor(
                    estimator=Ridge(alpha=10), n_estimators=50, random_state=42))
            model.fit(X_train_scaled, y_train)
            pred = model.predict(x_scaled)[0]
            pred_row = deduplicate_prediction(pred)

        preds_all.append(pred_row)

    # Score
    mae_sum = 0
    match_sum = 0
    match_detail = []
    for i in range(len(test_draws)):
        pred_set = set(preds_all[i])
        actual_set = set(test_actuals[i].tolist())
        matches = pred_set & actual_set
        match_sum += len(matches)
        mae_sum += np.mean(np.abs(np.array(sorted(preds_all[i])) - np.array(sorted(test_actuals[i].tolist()))))
        match_detail.append((test_draws[i], sorted(preds_all[i]), sorted(test_actuals[i].tolist()), sorted(matches), len(matches)))

    avg_mae = mae_sum / len(test_draws)

    print(f"=== {name} | Total Matches: {match_sum} | Avg MAE: {avg_mae:.1f} ===")
    for draw, pred, actual, matches, mc in match_detail:
        print(f"  {draw}: Pred={pred} Actual={actual} Match={matches} ({mc}/6)")
    print()

    if match_sum > best_score:
        best_score = match_sum
        best_model_name = name

print(f"\n*** BEST MODEL: {best_model_name} with {best_score} total matches ***")
