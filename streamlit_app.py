import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
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

def predict_numbers(df, n_predictions=6, n_sets=5):
    """Predict multiple sets of numbers based on historical data (6 winning numbers + 1 supplementary)"""
    if df is None or df.empty:
        return []
    
    # Use ALL historical data for predictions (ignore year/month filters)
    latest_data = df
    
    # Extract winning numbers for analysis
    winning_number_columns = ['Winning Number 1', '2', '3', '4', '5', '6']
    
    # Get frequency analysis from the latest year
    all_numbers = []
    for col in winning_number_columns:
        all_numbers.extend(latest_data[col].dropna().tolist())
    
    number_counts = Counter(all_numbers)
    
    # Get the most frequent numbers from the latest year
    most_frequent = [num for num, count in number_counts.most_common(20)]
    
    # Get least frequent numbers (numbers that appear less often)
    all_possible_numbers = list(range(1, 50))
    least_frequent = [num for num in all_possible_numbers if num not in most_frequent[:15]]
    medium_frequent = [num for num, count in number_counts.most_common(30)[15:]]  # Numbers 16-30 in frequency
    
    # Analyze supplementary numbers separately
    supplementary_numbers = latest_data['Supplementary Number'].dropna().tolist()
    supplementary_counts = Counter(supplementary_numbers)
    most_frequent_supplementary = [num for num, count in supplementary_counts.most_common(10)]
    least_frequent_supplementary = [num for num in all_possible_numbers if num not in most_frequent_supplementary[:5]]
    
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
            current_draw['Supplementary Number'],
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
                if set_num == 0:  # Set 1: Only most frequent (top 15)
                    if most_frequent:
                        pred = np.random.choice(most_frequent[:15])
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
            if set_num == 0:  # Set 1: Most frequent supplementary
                if most_frequent_supplementary:
                    supplementary = most_frequent_supplementary[0]  # Always the most frequent
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
                if set_num == 0:  # Set 1: Only most frequent (top 15)
                    if most_frequent:
                        pred = np.random.choice(most_frequent[:15])
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
                if set_num == 0:  # Set 1: Only most frequent
                    if most_frequent:
                        for num in most_frequent[:15]:
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
            if set_num == 0:  # Set 1: Top 3 most frequent supplementary
                if most_frequent_supplementary and len(most_frequent_supplementary) >= 3:
                    supplementary = most_frequent_supplementary[0]  # Always pick #1 most frequent
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
                if set_num == 0:  # Set 1: Only most frequent (top 15)
                    if most_frequent:
                        pred = np.random.choice(most_frequent[:15])
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
            if set_num == 0:  # Set 1: Most frequent supplementary
                if most_frequent_supplementary:
                    supplementary = np.random.choice(most_frequent_supplementary[:3])
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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "📈 Frequency Analysis", "🔮 Number Predictions", "📊 Distribution Analysis"])
    
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
                avg_supp = filtered_df['Supplementary Number'].mean()
                st.metric("Avg Supplementary #", f"{avg_supp:.1f}")
            
            # Data preview for selected period with pagination
            st.subheader(f"Recent Draws in {period_text}")
            display_columns = ['Draw', 'Date', 'Winning Number 1', '2', '3', '4', '5', '6', 'Supplementary Number']
            
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
        st.info("Generating 5 prediction sets with **distinct probability strategies** based on **ALL historical data**:\n"
                "• **Set 1**: 100% most frequent numbers (Top 15 historical winners)\n"
                "• **Set 2**: 80% frequent + 20% others (Top 20 + alternatives)\n" 
                "• **Set 3**: 60% frequent + 40% medium/less frequent\n"
                "• **Set 4**: 30% frequent + 70% less frequent numbers\n"
                "• **Set 5**: 10% frequent + 90% least frequent numbers\n\n"
                "**Note:** Predictions use complete dataset regardless of year/month filter to optimize performance.")
        
        # Initialize prediction counter and sets in session state
        if 'prediction_counter' not in st.session_state:
            st.session_state.prediction_counter = 0
        
        # Re-generate button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🎲 Re-generate Numbers", type="primary", use_container_width=True):
                # Increment counter to trigger regeneration
                st.session_state.prediction_counter += 1
        
        # Generate predictions based on counter (regenerates when counter changes)
        # Use ALL data for predictions (independent of year/month filters)
        prediction_key = f"predictions_{st.session_state.prediction_counter}"
        if prediction_key not in st.session_state:
            with st.spinner("Generating predictions..."):
                # Reset random seed for different results
                import random
                import time
                random.seed(int(time.time()) + st.session_state.prediction_counter)
                st.session_state[prediction_key] = predict_numbers(df)
        
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
            
            # Overall analysis
            st.subheader("📈 Overall Analysis")
            
            # Flatten all predictions for analysis
            all_predictions = [num for pred_set in prediction_sets for num in pred_set]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Frequency chart of all predicted numbers
                from collections import Counter
                pred_counts = Counter(all_predictions)
                
                if pred_counts:
                    fig_freq = px.bar(
                        x=list(pred_counts.keys()),
                        y=list(pred_counts.values()),
                        title="Frequency of Predicted Numbers Across All Sets",
                        labels={'x': 'Number', 'y': 'Frequency'},
                        color=list(pred_counts.values()),
                        color_continuous_scale='viridis'
                    )
                    fig_freq.update_layout(showlegend=False)
                    st.plotly_chart(fig_freq, use_container_width=True)
            
            with col2:
                # Overall statistics
                st.write("**Overall Statistics:**")
                st.write(f"- Total Sets: {len(prediction_sets)}")
                st.write(f"- Unique Numbers: {len(set(all_predictions))}")
                st.write(f"- Most Common: {max(pred_counts, key=pred_counts.get)} ({max(pred_counts.values())}x)")
                st.write(f"- Average Sum: {np.mean([sum(s) for s in prediction_sets]):.1f}")
                
                # Range distribution across all sets
                ranges = {
                    '1-10': sum(1 for p in all_predictions if 1 <= p <= 10),
                    '11-20': sum(1 for p in all_predictions if 11 <= p <= 20),
                    '21-30': sum(1 for p in all_predictions if 21 <= p <= 30),
                    '31-40': sum(1 for p in all_predictions if 31 <= p <= 40),
                    '41-49': sum(1 for p in all_predictions if 41 <= p <= 49)
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
    
    st.markdown("---")
    st.info("💡 Note: Predictions are based on historical patterns and should be used for entertainment purposes only.")

if __name__ == "__main__":
    main()
