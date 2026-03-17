import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import warnings
import random
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ToTo Number Analysis", page_icon="🎲", layout="wide")

@st.cache_data
def load_data():
    """Load and prepare the ToTo data"""
    try:
        # Read the Excel file
        df = pd.read_excel("ToTo.xlsx")
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def analyze_numbers_by_period(df, year, month=None):
    """Analyze number frequency for a specific year and optionally month"""
    if df is None:
        return None, None
    
    # Filter data by year
    filtered_data = df[df['Date'].dt.year == year]
    
    # Filter by month if specified (None means all months)
    if month is not None:
        filtered_data = filtered_data[filtered_data['Date'].dt.month == month]
    
    # Extract all winning numbers (columns 'Winning Number 1', '2', '3', '4', '5', '6')
    winning_number_columns = ['Winning Number 1', '2', '3', '4', '5', '6']
    all_numbers = []
    
    for col in winning_number_columns:
        if col in filtered_data.columns:
            all_numbers.extend(filtered_data[col].dropna().tolist())
    
    # Count frequency of each number
    number_counts = Counter(all_numbers)
    
    return number_counts, filtered_data

def predict_numbers(df, n_predictions=6, n_sets=5, filtered_df=None):
    """Predict multiple sets of numbers based on historical data (6 winning numbers + 1 supplementary)
    Set 1 uses filtered_df (year/month filtered) if provided, Sets 2-5 use ALL data."""
    if df is None or df.empty:
        return []
    
    latest_data = df
    
    # Extract winning numbers for analysis
    winning_number_columns = ['Winning Number 1', '2', '3', '4', '5', '6']
    
    # ALL data frequencies (used by Sets 2-5)
    all_numbers = []
    for col in winning_number_columns:
        all_numbers.extend(latest_data[col].dropna().tolist())
    
    number_counts = Counter(all_numbers)
    most_frequent = [num for num, count in number_counts.most_common(20)]
    
    # Filtered data frequencies for Set 1
    filtered_data = filtered_df if filtered_df is not None and len(filtered_df) > 0 else latest_data
    filtered_numbers = []
    for col in winning_number_columns:
        filtered_numbers.extend(filtered_data[col].dropna().tolist())
    filtered_counts = Counter(filtered_numbers)
    most_frequent_filtered = [num for num, count in filtered_counts.most_common(20)]
    
    # Get least frequent numbers (numbers that appear less often)
    all_possible_numbers = list(range(1, 50))
    least_frequent = [num for num in all_possible_numbers if num not in most_frequent[:15]]
    medium_frequent = [num for num, count in number_counts.most_common(30)[15:]]  # Numbers 16-30 in frequency
    
    # Analyze supplementary numbers separately (ALL data for Sets 2-5)
    supplementary_numbers = latest_data['Additional Number'].dropna().tolist()
    supplementary_counts = Counter(supplementary_numbers)
    most_frequent_supplementary = [num for num, count in supplementary_counts.most_common(10)]
    least_frequent_supplementary = [num for num in all_possible_numbers if num not in most_frequent_supplementary[:5]]
    
    # Filtered supplementary for Set 1
    filtered_supp_numbers = filtered_data['Additional Number'].dropna().tolist()
    filtered_supp_counts = Counter(filtered_supp_numbers)
    most_frequent_supp_filtered = [num for num, count in filtered_supp_counts.most_common(10)]
    
    # Create features for machine learning
    features = []
    targets = []
    
    # Use historical patterns to create training data
    for i in range(len(latest_data) - 1):
        current_draw = latest_data.iloc[i]
        next_draw = latest_data.iloc[i + 1]
        
        # Features: current draw numbers, supplementary number, and some statistics
        row_features = [
            current_draw['Winning Number 1'],
            current_draw['2'],
            current_draw['3'],
            current_draw['4'],
            current_draw['5'],
            current_draw['6'],
            current_draw['Additional Number'],
            current_draw['Low'],
            current_draw['High'],
            current_draw['Odd'],
            current_draw['Even']
        ]
        
        # Target: next draw's first winning number
        target = next_draw['Winning Number 1']
        
        features.append(row_features)
        targets.append(target)
    
    if len(features) < 3:
        # If insufficient data, use frequency-based prediction
        # Generate multiple sets of predictions
        all_prediction_sets = []
        available_numbers = list(range(1, 50))  # ToTo numbers are 1-49
        
        for set_num in range(n_sets):
            predictions = set()
            
            # Apply distinct probability strategy for each set
            while len(predictions) < n_predictions:
                if set_num == 0:  # Set 1: Most frequent from selected year
                    if most_frequent_filtered:
                        pred = np.random.choice(most_frequent_filtered[:15])
                    else:
                        pred = np.random.randint(1, 50)
                elif set_num == 1:  # Set 2: 80% from top 20, 20% from others
                    if most_frequent and np.random.random() < 0.8:
                        pred = np.random.choice(most_frequent[:20])
                    else:
                        others_pool = [n for n in range(1, 50) if n not in most_frequent[:15]]
                        pred = np.random.choice(others_pool if others_pool else list(range(1, 50)))
                elif set_num == 2:  # Set 3: 60% frequent, 40% medium/less frequent
                    if most_frequent and np.random.random() < 0.6:
                        pred = np.random.choice(most_frequent[:25])
                    else:
                        medium_pool = [n for n in range(1, 50) if n not in most_frequent[:20]]
                        pred = np.random.choice(medium_pool if medium_pool else list(range(1, 50)))
                elif set_num == 3:  # Set 4: 30% frequent, 70% less frequent
                    if most_frequent and np.random.random() < 0.3:
                        pred = np.random.choice(most_frequent[:30])
                    else:
                        less_frequent_pool = [n for n in range(1, 50) if n not in most_frequent[:25]]
                        pred = np.random.choice(less_frequent_pool if less_frequent_pool else list(range(1, 50)))
                else:  # Set 5: 10% frequent, 90% least frequent
                    if most_frequent and np.random.random() < 0.1:
                        pred = np.random.choice(most_frequent)
                    else:
                        least_frequent_pool = [n for n in range(1, 50) if n not in most_frequent[:30]]
                        pred = np.random.choice(least_frequent_pool if least_frequent_pool else list(range(1, 50)))
                
                predictions.add(pred)
            
            # Generate supplementary number with distinct strategy
            if set_num == 0:  # Set 1: Most frequent supplementary from selected year
                if most_frequent_supp_filtered:
                    supplementary = most_frequent_supp_filtered[0]
                else:
                    supplementary = np.random.randint(1, 50)
            elif set_num == 1:  # Set 2: Top frequent supplementary (2nd-4th)
                if most_frequent_supplementary and len(most_frequent_supplementary) >= 4:
                    supplementary = np.random.choice(most_frequent_supplementary[1:4])
                else:
                    supplementary = np.random.choice(supplementary_numbers if supplementary_numbers else list(range(1, 50)))
            elif set_num == 2:  # Set 3: Medium frequent supplementary
                if most_frequent_supplementary and np.random.random() < 0.6:
                    supplementary = np.random.choice(most_frequent_supplementary[3:8])
                else:
                    medium_supp = [n for n in supplementary_numbers if n not in most_frequent_supplementary[:5]] if supplementary_numbers and most_frequent_supplementary else list(range(1, 50))
                    supplementary = np.random.choice(medium_supp if medium_supp else list(range(1, 50)))
            elif set_num == 3:  # Set 4: Less frequent supplementary
                if most_frequent_supplementary and np.random.random() < 0.3:
                    supplementary = np.random.choice(most_frequent_supplementary[-3:])
                else:
                    less_freq_supp = [n for n in range(1, 50) if n not in most_frequent_supplementary[:8]] if most_frequent_supplementary else list(range(1, 50))
                    supplementary = np.random.choice(less_freq_supp if less_freq_supp else list(range(1, 50)))
            else:  # Set 5: Least frequent supplementary
                least_freq_supp = [n for n in range(1, 50) if n not in most_frequent_supplementary[:10]] if most_frequent_supplementary else list(range(1, 50))
                supplementary = np.random.choice(least_freq_supp if least_freq_supp else list(range(1, 50)))
            
            # Ensure supplementary is not in winning numbers
            attempts = 0
            while supplementary in predictions and attempts < 20:
                # Retry with same strategy
                if set_num == 0 and most_frequent_supplementary:
                    supplementary = np.random.choice(most_frequent_supplementary[:5])
                elif set_num == 4 and least_frequent_supplementary:
                    supplementary = np.random.choice(least_frequent_supplementary)
                else:
                    supplementary = np.random.choice(supplementary_numbers if supplementary_numbers else all_possible_numbers)
                attempts += 1
            
            # Create final set: 6 winning numbers + 1 supplementary
            final_set = sorted(list(predictions)) + [supplementary]
            all_prediction_sets.append(final_set)
        
        return all_prediction_sets
    
    # Train a model for prediction
    try:
        X = np.array(features)
        y = np.array(targets)
        
        # Use Random Forest for prediction
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        
        # Generate multiple sets of predictions
        all_prediction_sets = []
        last_features = features[-1]
        
        for set_num in range(n_sets):
            predictions = set()
            
            # Use frequency-based approach instead of ML for more distinct sets
            # Only use a small portion from ML, majority from frequency strategy
            ml_predictions = 0
            
            # Generate 1-2 numbers from ML for variety (only for sets 1-3)
            if set_num <= 2:
                ml_noise_factor = 0.05 + (set_num * 0.05)
                attempts = 0
                while ml_predictions < 2 and attempts < 20:
                    varied_features = [f + np.random.normal(0, ml_noise_factor) for f in last_features]
                    pred = model.predict([varied_features])[0]
                    pred_int = int(np.clip(pred, 1, 49))
                    if pred_int not in predictions:
                        predictions.add(pred_int)
                        ml_predictions += 1
                    attempts += 1
            
            # Fill remaining positions with frequency-based strategy
            while len(predictions) < n_predictions:
                if set_num == 0:  # Set 1: Most frequent from selected year
                    if most_frequent_filtered:
                        pred = np.random.choice(most_frequent_filtered[:15])
                    else:
                        pred = np.random.randint(1, 50)
                elif set_num == 1:  # Set 2: 80% from top 20, 20% from medium
                    if most_frequent and np.random.random() < 0.8:
                        pred = np.random.choice(most_frequent[:20])
                    else:
                        medium_pool = [n for n in range(1, 50) if n not in most_frequent[:15]]
                        pred = np.random.choice(medium_pool if medium_pool else list(range(1, 50)))
                elif set_num == 2:  # Set 3: 60% frequent, 40% medium/less frequent
                    if most_frequent and np.random.random() < 0.6:
                        pred = np.random.choice(most_frequent[:25])
                    else:
                        less_frequent_pool = [n for n in range(1, 50) if n not in most_frequent[:20]]
                        pred = np.random.choice(less_frequent_pool if less_frequent_pool else list(range(1, 50)))
                elif set_num == 3:  # Set 4: 30% frequent, 70% less frequent
                    if most_frequent and np.random.random() < 0.3:
                        pred = np.random.choice(most_frequent[:30])
                    else:
                        less_frequent_pool = [n for n in range(1, 50) if n not in most_frequent[:25]]
                        pred = np.random.choice(less_frequent_pool if less_frequent_pool else list(range(1, 50)))
                else:  # Set 5: 10% frequent, 90% least frequent
                    if most_frequent and np.random.random() < 0.1:
                        pred = np.random.choice(most_frequent)
                    else:
                        least_frequent_pool = [n for n in range(1, 50) if n not in most_frequent[:30]]
                        pred = np.random.choice(least_frequent_pool if least_frequent_pool else list(range(1, 50)))
                
                predictions.add(pred)
            
            # If we don't have enough unique predictions, add some based on set strategy
            while len(predictions) < n_predictions:
                if set_num == 0:  # Set 1: Most frequent from selected year
                    if most_frequent_filtered:
                        for num in most_frequent_filtered[:15]:
                            if num not in predictions:
                                predictions.add(num)
                                break
                elif set_num == 1:  # Set 2: 75% frequent
                    if most_frequent and np.random.random() < 0.75:
                        for num in most_frequent[:12]:
                            if num not in predictions:
                                predictions.add(num)
                                break
                    else:
                        predictions.add(np.random.randint(1, 50))
                elif set_num == 2:  # Set 3: 50% frequent
                    if most_frequent and np.random.random() < 0.5:
                        for num in most_frequent[:10]:
                            if num not in predictions:
                                predictions.add(num)
                                break
                    else:
                        predictions.add(np.random.randint(1, 50))
                elif set_num == 3:  # Set 4: 25% frequent
                    if most_frequent and np.random.random() < 0.25:
                        for num in most_frequent[:8]:
                            if num not in predictions:
                                predictions.add(num)
                                break
                    else:
                        # Prefer less frequent numbers
                        available_less_frequent = [n for n in range(1, 50) if n not in most_frequent[:10]]
                        if available_less_frequent:
                            predictions.add(np.random.choice(available_less_frequent))
                        else:
                            predictions.add(np.random.randint(1, 50))
                else:  # Set 5: Least frequent
                    # Prefer numbers not in most frequent
                    available_less_frequent = [n for n in range(1, 50) if n not in most_frequent[:15]]
                    if available_less_frequent:
                        predictions.add(np.random.choice(available_less_frequent))
                    else:
                        predictions.add(np.random.randint(1, 50))
                
                # Safety break
                if len(predictions) >= n_predictions:
                    break
            
            # Generate supplementary number with distinct strategy for each set
            if set_num == 0:  # Set 1: Most frequent supplementary from selected year
                if most_frequent_supp_filtered:
                    supplementary = most_frequent_supp_filtered[0]
                else:
                    supplementary = np.random.randint(1, 50)
            elif set_num == 1:  # Set 2: Top 5 most frequent supplementary
                if most_frequent_supplementary and len(most_frequent_supplementary) >= 5:
                    supplementary = np.random.choice(most_frequent_supplementary[1:4])  # Pick from 2nd-4th most frequent
                else:
                    supplementary = np.random.choice(supplementary_numbers if supplementary_numbers else list(range(1, 50)))
            elif set_num == 2:  # Set 3: Mix of frequent and medium supplementary
                if most_frequent_supplementary and np.random.random() < 0.6:
                    supplementary = np.random.choice(most_frequent_supplementary[:7])
                else:
                    medium_supp_pool = [n for n in supplementary_numbers if n not in most_frequent_supplementary[:5]] if supplementary_numbers and most_frequent_supplementary else list(range(1, 50))
                    supplementary = np.random.choice(medium_supp_pool if medium_supp_pool else list(range(1, 50)))
            elif set_num == 3:  # Set 4: Prefer less frequent supplementary
                if most_frequent_supplementary and np.random.random() < 0.3:
                    supplementary = np.random.choice(most_frequent_supplementary[-3:])  # Least frequent from the frequent list
                else:
                    less_frequent_supp = [n for n in range(1, 50) if n not in most_frequent_supplementary[:7]] if most_frequent_supplementary else list(range(1, 50))
                    supplementary = np.random.choice(less_frequent_supp if less_frequent_supp else list(range(1, 50)))
            else:  # Set 5: Least frequent supplementary numbers
                less_frequent_supp = [n for n in range(1, 50) if n not in most_frequent_supplementary[:10]] if most_frequent_supplementary else list(range(1, 50))
                supplementary = np.random.choice(less_frequent_supp if less_frequent_supp else list(range(1, 50)))
            
            # Ensure supplementary is not in winning numbers
            attempts = 0
            while supplementary in predictions and attempts < 20:
                if most_frequent_supplementary:
                    supplementary = np.random.choice(supplementary_numbers)
                else:
                    supplementary = np.random.randint(1, 50)
                attempts += 1
            
            # Create final set: 6 winning numbers + 1 supplementary
            final_set = sorted(list(predictions))[:n_predictions] + [supplementary]
            all_prediction_sets.append(final_set)
        
        return all_prediction_sets
        
    except Exception as e:
        st.warning(f"Prediction error: {e}. Using frequency-based prediction.")
        # Fallback to frequency-based prediction
        all_prediction_sets = []
        
        for set_num in range(n_sets):
            predictions = set()
            
            # Generate unique predictions with distinct strategy for each set
            while len(predictions) < n_predictions:
                if set_num == 0:  # Set 1: Most frequent from selected year
                    if most_frequent_filtered:
                        pred = np.random.choice(most_frequent_filtered[:15])
                    else:
                        pred = np.random.randint(1, 50)
                elif set_num == 1:  # Set 2: 80% from top 20, 20% from others
                    if most_frequent and np.random.random() < 0.8:
                        pred = np.random.choice(most_frequent[:20])
                    else:
                        others_pool = [n for n in range(1, 50) if n not in most_frequent[:15]]
                        pred = np.random.choice(others_pool if others_pool else list(range(1, 50)))
                elif set_num == 2:  # Set 3: 60% frequent, 40% medium/less frequent
                    if most_frequent and np.random.random() < 0.6:
                        pred = np.random.choice(most_frequent[:25])
                    else:
                        medium_pool = [n for n in range(1, 50) if n not in most_frequent[:20]]
                        pred = np.random.choice(medium_pool if medium_pool else list(range(1, 50)))
                elif set_num == 3:  # Set 4: 30% frequent, 70% less frequent  
                    if most_frequent and np.random.random() < 0.3:
                        pred = np.random.choice(most_frequent[:30])
                    else:
                        less_frequent_pool = [n for n in range(1, 50) if n not in most_frequent[:25]]
                        pred = np.random.choice(less_frequent_pool if less_frequent_pool else list(range(1, 50)))
                else:  # Set 5: 10% frequent, 90% least frequent
                    if most_frequent and np.random.random() < 0.1:
                        pred = np.random.choice(most_frequent)
                    else:
                        least_frequent_pool = [n for n in range(1, 50) if n not in most_frequent[:30]]
                        pred = np.random.choice(least_frequent_pool if least_frequent_pool else list(range(1, 50)))
                
                predictions.add(pred)
            
            # Generate supplementary number with same probability strategy
            if set_num == 0:  # Set 1: Most frequent supplementary from selected year
                if most_frequent_supp_filtered:
                    supplementary = np.random.choice(most_frequent_supp_filtered[:3])
                else:
                    supplementary = np.random.randint(1, 50)
            elif set_num == 1:  # Set 2: 2nd-4th most frequent supplementary
                if most_frequent_supplementary and len(most_frequent_supplementary) >= 4:
                    supplementary = np.random.choice(most_frequent_supplementary[1:4])
                else:
                    supplementary = np.random.choice(supplementary_numbers if supplementary_numbers else list(range(1, 50)))
            elif set_num == 2:  # Set 3: Medium frequent supplementary
                if most_frequent_supplementary and np.random.random() < 0.6:
                    supplementary = np.random.choice(most_frequent_supplementary[3:8])
                else:
                    medium_supp = [n for n in supplementary_numbers if n not in most_frequent_supplementary[:5]] if supplementary_numbers and most_frequent_supplementary else list(range(1, 50))
                    supplementary = np.random.choice(medium_supp if medium_supp else list(range(1, 50)))
            elif set_num == 3:  # Set 4: Less frequent supplementary
                if most_frequent_supplementary and np.random.random() < 0.3:
                    supplementary = np.random.choice(most_frequent_supplementary[-3:])
                else:
                    less_freq_supp = [n for n in range(1, 50) if n not in most_frequent_supplementary[:8]] if most_frequent_supplementary else list(range(1, 50))
                    supplementary = np.random.choice(less_freq_supp if less_freq_supp else list(range(1, 50)))
            else:  # Set 5: Least frequent supplementary
                least_freq_supp = [n for n in range(1, 50) if n not in most_frequent_supplementary[:10]] if most_frequent_supplementary else list(range(1, 50))
                supplementary = np.random.choice(least_freq_supp if least_freq_supp else list(range(1, 50)))
            
            # Ensure supplementary is not in winning numbers
            attempts = 0
            while supplementary in predictions and attempts < 20:
                if most_frequent_supplementary:
                    supplementary = np.random.choice(supplementary_numbers)
                else:
                    supplementary = np.random.randint(1, 50)
                attempts += 1
            
            # Create final set: 6 winning numbers + 1 supplementary
            final_set = sorted(list(predictions))[:n_predictions] + [supplementary]
            all_prediction_sets.append(final_set)
        
        return all_prediction_sets

def deduplicate_prediction(pred_row):
    """Remove duplicate numbers from a prediction row by nudging duplicates to nearest unused value."""
    used = set()
    result = []
    for val in pred_row:
        v = int(np.clip(np.round(val), 1, 49))
        if v not in used:
            used.add(v)
            result.append(v)
        else:
            # Find nearest unused number
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


def build_rich_features(all_numbers, idx, window=5):
    """Build rich feature vector using gap analysis, frequency, position stats, and draw statistics."""
    features = []

    # 1. Raw numbers from last `window` draws (flattened)
    start = max(0, idx - window)
    window_data = all_numbers[start:idx]
    for row in window_data:
        features.extend(row.tolist())
    while len(features) < window * 6:
        features.insert(0, 25)

    # 2. Gap features: draws since each number 1-49 last appeared
    recent_nums = all_numbers[max(0, idx - 50):idx]
    for num in range(1, 50):
        found = False
        for lookback in range(len(recent_nums) - 1, -1, -1):
            if num in recent_nums[lookback]:
                features.append(len(recent_nums) - 1 - lookback)
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

    # 4. Per-position frequency in last 20 draws (top 3 per position)
    for pos in range(6):
        pos_counter = Counter()
        for row in recent_20:
            pos_counter[row[pos]] += 1
        top3 = pos_counter.most_common(3)
        for j in range(3):
            features.append(top3[j][0] if j < len(top3) else 0)

    # 5. Statistics of last 3 draws
    last3 = all_numbers[max(0, idx - 3):idx]
    for row in last3:
        features.extend([
            np.mean(row), np.std(row),
            np.max(row) - np.min(row),
            np.sum(row % 2),
            np.sum(row <= 25),
            np.sum(row),
        ])
    # Pad to consistent size
    target_len = window * 6 + 49 + 49 + 18 + 18
    while len(features) < target_len:
        features.append(0)
    return features


@st.cache_data(show_spinner=False)
def run_ml_model_analysis(_df, test_start=4158, test_end=4162, window=5, future_draws=5):
    """Run multiple ML models using ALL historical data with rich features."""
    from sklearn.linear_model import BayesianRidge, ElasticNet
    from sklearn.ensemble import (
        ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
    )

    df_copy = _df.copy()
    df_copy['Draw'] = df_copy['Draw'].astype(str).str.strip()
    valid = df_copy[df_copy['Draw'].apply(lambda x: x.isdigit())].copy()
    valid['Draw'] = valid['Draw'].astype(int)
    valid = valid.sort_values('Draw').reset_index(drop=True)

    num_cols = ['Winning Number 1', '2', '3', '4', '5', '6']
    all_numbers = valid[num_cols].values

    # Training: all draws up to test_start - 1
    train_end_idx = valid[valid['Draw'] == test_start - 1].index[0]

    X_train, y_train = [], []
    for idx in range(window, train_end_idx + 1):
        X_train.append(build_rich_features(all_numbers, idx, window))
        y_train.append(all_numbers[idx])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Test data
    test = valid[valid['Draw'].between(test_start, test_end)]
    test_nums = test[num_cols].values
    train_display = valid[valid['Draw'].between(test_start - 14, test_start - 1)]

    # Define models: (mode, factory)
    # 'multi' = native multi-output, 'percol' = train per column
    model_defs = {
        'KNN (k=3)': ('multi', lambda: KNeighborsRegressor(n_neighbors=3)),
        'KNN (k=7)': ('multi', lambda: KNeighborsRegressor(n_neighbors=7)),
        'KNN (k=11)': ('multi', lambda: KNeighborsRegressor(n_neighbors=11)),
        'Linear Regression': ('multi', lambda: LinearRegression()),
        'Ridge': ('multi', lambda: Ridge(alpha=50)),
        'Lasso': ('percol', lambda: Lasso(alpha=0.1)),
        'Bayesian Ridge': ('percol', lambda: BayesianRidge()),
        'Random Forest': ('multi', lambda: RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1)),
        'ExtraTrees': ('multi', lambda: ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1)),
        'GradBoost': ('percol', lambda: GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
        'HistGradBoost': ('percol', lambda: HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42)),
    }

    # Pre-build test features (same for all models)
    test_features = []
    for ti in range(len(test_nums)):
        draw_num = test_start + ti
        idx = valid[valid['Draw'] == draw_num].index[0]
        test_features.append(build_rich_features(all_numbers, idx, window))
    X_test_scaled = scaler.transform(np.array(test_features))

    # Evaluate each model: train ONCE, predict all test draws
    results = {}
    for name, (mode, model_fn) in model_defs.items():
        if mode == 'percol':
            # Train 6 models (one per column), predict all test draws at once
            trained = []
            for col_idx in range(6):
                m = model_fn()
                m.fit(X_train_scaled, y_train[:, col_idx])
                trained.append(m)
            preds_all = []
            for ti in range(len(test_nums)):
                pred_row = [trained[c].predict(X_test_scaled[ti:ti+1])[0] for c in range(6)]
                preds_all.append(deduplicate_prediction(pred_row))
        else:
            model = model_fn()
            model.fit(X_train_scaled, y_train)
            preds_all = []
            for ti in range(len(test_nums)):
                pred = model.predict(X_test_scaled[ti:ti+1])[0]
                preds_all.append(deduplicate_prediction(pred))

        preds_arr = np.array(preds_all)
        mae = np.mean(np.abs(preds_arr - test_nums))
        match_counts = [len(set(preds_all[i]) & set(test_nums[i].tolist())) for i in range(len(test_nums))]

        results[name] = {
            'predictions': preds_all,
            'mae': mae,
            'match_counts': match_counts,
            'total_matches': sum(match_counts),
            'avg_matches': np.mean(match_counts),
        }

    # --- Generate future predictions using ALL data up to test_end ---
    full_end_idx = valid[valid['Draw'] == test_end].index[0]
    X_full, y_full = [], []
    for idx in range(window, full_end_idx + 1):
        X_full.append(build_rich_features(all_numbers, idx, window))
        y_full.append(all_numbers[idx])
    X_full = np.array(X_full)
    y_full = np.array(y_full)
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full)

    # Pre-build future features using rolling predictions
    # Each model predicts one draw at a time, appends prediction to history,
    # then uses updated history for the next draw's features.
    future_preds = {}
    raw_preds_for_ensemble = {}

    for name, (mode, model_fn) in model_defs.items():
        raw = []
        preds = []
        # Work on a copy of all_numbers so each model gets fresh history
        extended_numbers = all_numbers.copy()

        if mode == 'percol':
            trained = []
            for col_idx in range(6):
                m = model_fn()
                m.fit(X_full_scaled, y_full[:, col_idx])
                trained.append(m)
            for draw_idx in range(future_draws):
                x_feat = build_rich_features(extended_numbers, len(extended_numbers), window)
                x_scaled = scaler_full.transform([x_feat])
                raw_row = [np.clip(trained[c].predict(x_scaled)[0], 1, 49) for c in range(6)]
                raw.append(raw_row)
                pred_row = sorted(deduplicate_prediction(raw_row))
                preds.append(pred_row)
                # Append prediction to history so next draw has different features
                extended_numbers = np.vstack([extended_numbers, [pred_row]])
        else:
            model = model_fn()
            model.fit(X_full_scaled, y_full)
            for draw_idx in range(future_draws):
                x_feat = build_rich_features(extended_numbers, len(extended_numbers), window)
                x_scaled = scaler_full.transform([x_feat])
                pred = model.predict(x_scaled)[0]
                raw_row = [np.clip(p, 1, 49) for p in pred]
                raw.append(raw_row)
                pred_row = sorted(deduplicate_prediction(pred))
                preds.append(pred_row)
                # Append prediction to history so next draw has different features
                extended_numbers = np.vstack([extended_numbers, [pred_row]])

        future_preds[name] = preds
        raw_preds_for_ensemble[name] = np.array(raw)

    # Voting Ensemble: count number frequency across ALL models
    vote_preds = []
    for draw_idx in range(future_draws):
        vote_counter = Counter()
        for name in future_preds:
            for n in future_preds[name][draw_idx]:
                vote_counter[n] += 1
        top6 = [n for n, _ in vote_counter.most_common(6)]
        vote_preds.append(sorted(top6))
    future_preds['Voting Ensemble'] = vote_preds

    # Average Ensemble
    ensemble_raw = np.mean(list(raw_preds_for_ensemble.values()), axis=0)
    ensemble_preds = []
    for i in range(future_draws):
        row = deduplicate_prediction(ensemble_raw[i])
        ensemble_preds.append(sorted(row))
    future_preds['Average Ensemble'] = ensemble_preds

    return {
        'train': train_display,
        'test': test,
        'test_nums': test_nums,
        'num_cols': num_cols,
        'results': results,
        'future_preds': future_preds,
        'train_start': test_start - 14,
        'train_end': test_start - 1,
        'test_start': test_start,
        'test_end': test_end,
        'total_training_samples': len(X_train),
        'num_features': X_train.shape[1],
    }


def main():
    st.title("🎲 ToTo Number Analysis Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Could not load ToTo.xlsx file. Please ensure the file exists in the current directory.")
        st.info("Expected file structure: ToTo.xlsx with date and number columns")
        return
    
    # Parameters in main page
    st.subheader("📅 Analysis Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        available_years = sorted(df['Date'].dt.year.unique(), reverse=True)
        selected_year = st.selectbox("Select Year", available_years)
    
    with col2:
        # Month selection with "All Months" option
        month_options = ["All Months"] + [f"{i:02d} - {pd.Timestamp(2024, i, 1).strftime('%B')}" for i in range(1, 13)]
        selected_month_str = st.selectbox("Select Month", month_options)
    
    st.markdown("---")
    
    # Convert month selection to numeric value (None for "All Months")
    if selected_month_str == "All Months":
        selected_month = None
        period_text = f"{selected_year}"
    else:
        selected_month = int(selected_month_str.split(" - ")[0])
        month_name = selected_month_str.split(" - ")[1]
        period_text = f"{month_name} {selected_year}"
    
    # Get filtered data for the selected period
    if selected_month is None:
        filtered_df = df[df['Date'].dt.year == selected_year]
    else:
        filtered_df = df[(df['Date'].dt.year == selected_year) & (df['Date'].dt.month == selected_month)]
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dataset Overview", "📈 Frequency Analysis", "🔮 Number Predictions", "📊 Distribution Analysis", "🤖 ML Model Analysis"])
    
    # Tab 1: Dataset Overview
    with tab1:
        st.header(f"📊 Dataset Overview - {period_text}")
        
        if len(filtered_df) > 0:
            # Basic metrics for the selected period
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Draws", len(filtered_df))
            
            with col2:
                date_span = (filtered_df['Date'].max() - filtered_df['Date'].min()).days + 1
                st.metric("Date Span (Days)", date_span)
            
            with col3:
                latest_draw = filtered_df['Draw'].max()
                st.metric("Latest Draw", latest_draw)
            
            with col4:
                earliest_draw = filtered_df['Draw'].min()
                st.metric("Earliest Draw", earliest_draw)
            
            # Additional period-specific metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_gap = filtered_df['Draw'].diff().mean()
                st.metric("Avg Draw Gap", f"{avg_gap:.1f}" if not pd.isna(avg_gap) else "N/A")
            
            with col2:
                # Calculate total winning numbers drawn
                winning_cols = ['Winning Number 1', '2', '3', '4', '5', '6']
                total_numbers = len(filtered_df) * len(winning_cols)
                st.metric("Total Numbers Drawn", total_numbers)
            
            with col3:
                unique_dates = filtered_df['Date'].nunique()
                st.metric("Unique Draw Dates", unique_dates)
            
            with col4:
                # Average supplementary number
                avg_supp = filtered_df['Additional Number'].mean()
                st.metric("Avg Supplementary #", f"{avg_supp:.1f}")
            
            # Data preview for selected period with pagination
            st.subheader(f"Recent Draws in {period_text}")
            display_columns = ['Draw', 'Date', 'Winning Number 1', '2', '3', '4', '5', '6', 'Additional Number']
            
            # Calculate pagination
            total_rows = len(filtered_df)
            rows_per_page = 10
            total_pages = (total_rows + rows_per_page - 1) // rows_per_page  # Ceiling division
            
            if total_pages > 1:
                # Add pagination controls
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    page_number = st.selectbox(
                        f"Page (Total: {total_pages} pages, {total_rows} draws)",
                        range(1, total_pages + 1),
                        key="pagination_selectbox"
                    )
            else:
                page_number = 1
            
            # Calculate start and end indices for current page
            start_idx = (page_number - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, total_rows)
            
            # Display current page data
            current_page_data = filtered_df.iloc[start_idx:end_idx].copy()
            
            # Format the Date column to show only date without time
            current_page_data['Date'] = current_page_data['Date'].dt.strftime('%Y-%m-%d')
            
            # Add page info
            if total_pages > 1:
                st.caption(f"Showing draws {start_idx + 1} to {end_idx} of {total_rows} total draws")
            
            st.dataframe(current_page_data[display_columns], use_container_width=True, hide_index=True)
        
        else:
            st.warning(f"No data available for {period_text}")
    
    # Tab 2: Frequency Analysis
    with tab2:
        st.header("📈 Number Frequency Analysis")
        
        # Analyze numbers for selected period
        number_counts, period_data = analyze_numbers_by_period(df, selected_year, selected_month)
        
        if number_counts and len(number_counts) > 0:
            # Display most frequent numbers
            st.subheader(f"Most Frequent Numbers in {period_text}")
            
            # Show some stats about the period
            total_draws = len(period_data)
            st.info(f"Total draws in {period_text}: {total_draws}")
            
            # Top 15 most frequent numbers
            top_numbers = dict(number_counts.most_common(15))
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig_bar = px.bar(
                    x=list(top_numbers.keys()),
                    y=list(top_numbers.values()),
                    title=f"Top 15 Most Frequent Numbers in {period_text}",
                    labels={'x': 'Number', 'y': 'Frequency'},
                    color=list(top_numbers.values()),
                    color_continuous_scale='viridis'
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Display as table
                freq_df = pd.DataFrame([
                    {
                        'Rank': i+1,
                        'Number': num, 
                        'Frequency': count, 
                        'Percentage': f"{(count/sum(number_counts.values())*100):.1f}%"
                    }
                    for i, (num, count) in enumerate(top_numbers.items())
                ])
                st.dataframe(freq_df, use_container_width=True, hide_index=True)
            
            # Statistics
            all_nums = list(number_counts.keys())
            col1, col2 = st.columns(2)
            
            with col1:
                most_freq_num = max(number_counts, key=number_counts.get)
                st.metric("Most Frequent", f"{most_freq_num} ({max(number_counts.values())}x)")
            with col2:
                avg_freq = sum(number_counts.values())/len(number_counts)
                st.metric("Average Frequency", f"{avg_freq:.1f}")
        
        else:
            st.warning(f"No data available for {period_text}.")
    
    # Tab 3: Number Predictions
    with tab3:
        st.header("🔮 Number Predictions")
        st.info("Generating 5 prediction sets with **distinct probability strategies**:\n"
                f"• **Set 1**: 100% most frequent numbers from **selected year** ({selected_year})\n"
                "• **Set 2**: 80% frequent + 20% others (ALL historical data)\n" 
                "• **Set 3**: 60% frequent + 40% medium/less frequent (ALL data)\n"
                "• **Set 4**: 30% frequent + 70% less frequent (ALL data)\n"
                "• **Set 5**: 10% frequent + 90% least frequent (ALL data)")
        
        # Initialize prediction counter and sets in session state
        if 'prediction_counter' not in st.session_state:
            st.session_state.prediction_counter = 0
        
        # Re-generate button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🎲 Re-generate Numbers", type="primary", use_container_width=True):
                # Increment counter to trigger regeneration
                st.session_state.prediction_counter += 1
        
        # Generate predictions based on counter and selected year (regenerates when either changes)
        prediction_key = f"predictions_{st.session_state.prediction_counter}_{selected_year}_{selected_month}"
        if prediction_key not in st.session_state:
            with st.spinner("Generating predictions..."):
                # Reset random seed for different results
                import random
                import time
                random.seed(int(time.time()) + st.session_state.prediction_counter)
                st.session_state[prediction_key] = predict_numbers(df, filtered_df=filtered_df)
        
        # Use current predictions
        prediction_sets = st.session_state[prediction_key]
        
        if prediction_sets:
            # Display multiple prediction sets
            st.subheader("🎯 Predicted Number Sets")
            
            # Create a table to display all prediction sets
            prediction_data = []
            probability_labels = ["Highest", "High", "Medium", "Low", "Lowest"]
            for i, pred_set in enumerate(prediction_sets):
                prediction_data.append({
                    'Set': f"Set {i+1}",
                    'Probability': probability_labels[i],
                    'Number 1': pred_set[0],
                    'Number 2': pred_set[1],
                    'Number 3': pred_set[2],
                    'Number 4': pred_set[3],
                    'Number 5': pred_set[4],
                    'Number 6': pred_set[5],
                    'Supplementary': pred_set[6]
                })
            
            pred_df = pd.DataFrame(prediction_data)
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Overall analysis based on historical year data
            st.subheader(f"📈 Overall Analysis — {period_text}")
            
            # Use historical data from the selected year/month for analysis
            winning_cols_analysis = ['Winning Number 1', '2', '3', '4', '5', '6']
            hist_numbers = []
            for col in winning_cols_analysis:
                hist_numbers.extend(filtered_df[col].dropna().astype(int).tolist())
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Frequency chart of historical numbers
                from collections import Counter
                hist_counts = Counter(hist_numbers)
                
                if hist_counts:
                    sorted_counts = dict(sorted(hist_counts.items()))
                    fig_freq = px.bar(
                        x=list(sorted_counts.keys()),
                        y=list(sorted_counts.values()),
                        title=f"Number Frequency in {period_text} ({len(filtered_df)} draws)",
                        labels={'x': 'Number', 'y': 'Frequency'},
                        color=list(sorted_counts.values()),
                        color_continuous_scale='viridis'
                    )
                    fig_freq.update_layout(showlegend=False, xaxis_type='category')
                    st.plotly_chart(fig_freq, use_container_width=True)
            
            with col2:
                # Overall statistics from historical data
                st.write("**Overall Statistics:**")
                st.write(f"- Total Draws: {len(filtered_df)}")
                st.write(f"- Unique Numbers Drawn: {len(set(hist_numbers))}")
                if hist_counts:
                    most_common_num = max(hist_counts, key=hist_counts.get)
                    st.write(f"- Most Common: {most_common_num} ({hist_counts[most_common_num]}x)")
                avg_sum = filtered_df[winning_cols_analysis].sum(axis=1).mean()
                st.write(f"- Average Sum: {avg_sum:.1f}")
                
                # Range distribution from historical data
                ranges = {
                    '1-10': sum(1 for p in hist_numbers if 1 <= p <= 10),
                    '11-20': sum(1 for p in hist_numbers if 11 <= p <= 20),
                    '21-30': sum(1 for p in hist_numbers if 21 <= p <= 30),
                    '31-40': sum(1 for p in hist_numbers if 31 <= p <= 40),
                    '41-49': sum(1 for p in hist_numbers if 41 <= p <= 49)
                }
                
                st.write("**Range Distribution:**")
                for range_name, count in ranges.items():
                    if count > 0:
                        st.write(f"- {range_name}: {count} numbers")
    
    # Tab 4: Distribution Analysis
    with tab4:
        st.header("📊 Number Distribution Analysis")
        
        # Get data for the selected period
        number_counts, period_data = analyze_numbers_by_period(df, selected_year, selected_month)
        
        if number_counts and len(number_counts) > 0:
            st.subheader(f"Number Distribution for {period_text}")
            
            # Range analysis
            range_analysis = {
                '1-10': sum(count for num, count in number_counts.items() if 1 <= num <= 10),
                '11-20': sum(count for num, count in number_counts.items() if 11 <= num <= 20),
                '21-30': sum(count for num, count in number_counts.items() if 21 <= num <= 30),
                '31-40': sum(count for num, count in number_counts.items() if 31 <= num <= 40),
                '41-49': sum(count for num, count in number_counts.items() if 41 <= num <= 49)
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart for range distribution
                fig_pie = px.pie(
                    values=list(range_analysis.values()),
                    names=list(range_analysis.keys()),
                    title=f"Number Range Distribution ({period_text})"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Range statistics table
                range_df = pd.DataFrame([
                    {
                        'Range': range_name,
                        'Total Occurrences': count,
                        'Percentage': f"{(count/sum(range_analysis.values())*100):.1f}%"
                    }
                    for range_name, count in range_analysis.items()
                ])
                st.dataframe(range_df, use_container_width=True, hide_index=True)
            
            # Odd/Even analysis
            st.subheader("Odd/Even Distribution")
            odd_count = sum(count for num, count in number_counts.items() if num % 2 == 1)
            even_count = sum(count for num, count in number_counts.items() if num % 2 == 0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_odd_even = px.pie(
                    values=[odd_count, even_count],
                    names=['Odd', 'Even'],
                    title=f"Odd vs Even Numbers ({period_text})"
                )
                st.plotly_chart(fig_odd_even, use_container_width=True)
            
            with col2:
                st.metric("Odd Numbers", f"{odd_count} ({odd_count/(odd_count+even_count)*100:.1f}%)")
                st.metric("Even Numbers", f"{even_count} ({even_count/(odd_count+even_count)*100:.1f}%)")
        
        else:
            st.warning(f"No data available for {period_text}.")
    
    # Tab 5: ML Model Analysis
    with tab5:
        st.header("🤖 ML Model Analysis")
        st.info(
            "Train **11 ML models** on **ALL historical data** (~1800+ draws), validate against draws **4158–4162**, "
            "then generate future predictions using voting and average ensembles.\n\n"
            "**Models:** KNN (k=3/7/11), Linear Regression, Ridge, Lasso, Bayesian Ridge, "
            "Random Forest, ExtraTrees, GradientBoosting, HistGradientBoosting\n\n"
            "**Features (164):** Last 5 draws, gap analysis (draws since each number appeared), "
            "frequency in last 20 draws, per-position frequency, draw statistics"
        )

        with st.spinner("Running ML models (this may take a moment)..."):
            ml_results = run_ml_model_analysis(df)

        st.caption(f"Training: **{ml_results['total_training_samples']}** samples, **{ml_results['num_features']}** features per sample")

        # --- Training Data (show recent draws for context) ---
        st.subheader(f"📋 Recent Training Draws (last 14 before test set)")
        train_display = ml_results['train'][['Draw'] + ml_results['num_cols']].copy()
        train_display = train_display.reset_index(drop=True)
        st.dataframe(train_display, use_container_width=True, hide_index=True)

        # --- Validation Data ---
        st.subheader(f"✅ Validation Data (Draws {ml_results['test_start']}–{ml_results['test_end']})")
        test_display = ml_results['test'][['Draw'] + ml_results['num_cols']].copy()
        test_display = test_display.reset_index(drop=True)
        st.dataframe(test_display, use_container_width=True, hide_index=True)

        st.markdown("---")

        # --- Model Comparison ---
        st.subheader("📊 Model Comparison")

        comparison_data = []
        for name, res in ml_results['results'].items():
            comparison_data.append({
                'Model': name,
                'MAE': round(res['mae'], 2),
                'Avg Matches/Draw': round(res['avg_matches'], 2),
                'Total Matches': res['total_matches'],
            })
        comp_df = pd.DataFrame(comparison_data).sort_values('Total Matches', ascending=False)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Bar chart of total matches
        fig_comp = px.bar(
            comp_df, x='Model', y='Total Matches',
            color='MAE', color_continuous_scale='RdYlGn_r',
            title='Model Performance: Total Number Matches on Validation Set (higher is better)',
            text='Total Matches'
        )
        fig_comp.update_layout(yaxis_title='Total Exact Matches', xaxis_title='')
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")

        # --- Detailed Validation Results ---
        st.subheader("🔍 Detailed Validation: Predicted vs Actual")
        selected_model = st.selectbox("Select Model", list(ml_results['results'].keys()))

        res = ml_results['results'][selected_model]
        test_nums = ml_results['test_nums']

        detail_rows = []
        for i in range(len(test_nums)):
            draw_num = ml_results['test_start'] + i
            pred = sorted(res['predictions'][i])
            actual = sorted(test_nums[i].tolist())
            matched = sorted(set(pred) & set(actual))
            detail_rows.append({
                'Draw': draw_num,
                'Predicted': ', '.join(map(str, pred)),
                'Actual': ', '.join(map(str, actual)),
                'Matches': ', '.join(map(str, matched)) if matched else '—',
                'Match Count': f"{len(matched)}/6",
            })
        detail_df = pd.DataFrame(detail_rows)
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # --- Future Predictions ---
        last_draw = ml_results['test_end']
        st.subheader(f"🔮 Future Predictions (Draws {last_draw+1}–{last_draw+5})")
        st.caption(f"Models trained on ALL historical data ({ml_results['total_training_samples']} samples), then predict the next 5 draws.")

        future_rows = []
        for name, preds in ml_results['future_preds'].items():
            for i, pred in enumerate(preds):
                future_rows.append({
                    'Model': name,
                    'Draw': last_draw + 1 + i,
                    'Num 1': pred[0], 'Num 2': pred[1], 'Num 3': pred[2],
                    'Num 4': pred[3], 'Num 5': pred[4], 'Num 6': pred[5],
                })

        future_df = pd.DataFrame(future_rows)

        # Show ensemble first, then individual models
        model_tabs = list(ml_results['future_preds'].keys())
        for ens_name in ['Voting Ensemble', 'Average Ensemble']:
            if ens_name in model_tabs:
                model_tabs.remove(ens_name)
                model_tabs.insert(0, ens_name)

        ftabs = st.tabs(model_tabs)
        for tab_obj, model_name in zip(ftabs, model_tabs):
            with tab_obj:
                model_future = future_df[future_df['Model'] == model_name].drop(columns=['Model'])
                st.dataframe(model_future, use_container_width=True, hide_index=True)

        # Combined heatmap of all model future predictions
        st.subheader("🗺️ Number Frequency Across All Models' Predictions")
        all_future_nums = []
        for preds in ml_results['future_preds'].values():
            for row in preds:
                all_future_nums.extend(row)
        future_counts = Counter(all_future_nums)
        freq_items = sorted(future_counts.items(), key=lambda x: -x[1])

        fig_future = px.bar(
            x=[str(n) for n, _ in freq_items],
            y=[c for _, c in freq_items],
            title='How Often Each Number Appears Across All Models\' Future Predictions',
            labels={'x': 'Number', 'y': 'Frequency'},
            color=[c for _, c in freq_items],
            color_continuous_scale='viridis'
        )
        fig_future.update_layout(showlegend=False, xaxis_type='category')
        st.plotly_chart(fig_future, use_container_width=True)

        # Quick reference table
        st.subheader("📌 Quick Reference: Top Predicted Numbers")
        top_n = min(15, len(freq_items))
        ref_df = pd.DataFrame([
            {'Rank': i+1, 'Number': n, 'Appearances': c,
             'Confidence': f"{c / len(list(ml_results['future_preds'].keys())) / 5 * 100:.0f}%"}
            for i, (n, c) in enumerate(freq_items[:top_n])
        ])
        st.dataframe(ref_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.info("💡 Note: Predictions are based on historical patterns and should be used for entertainment purposes only.")

if __name__ == "__main__":
    main()
