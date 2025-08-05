# ToTo Analysis Dashboard

A Python Streamlit application to analyze Singapore ToTo lottery data and predict numbers based on historical patterns.

## Features

1. **Tabbed Interface**: Organized analysis in separate tabs
2. **Dataset Overview**: Responsive overview based on selected year/month
3. **Year & Month Filtering**: Parameter selection with "All Months" option
4. **Frequency Analysis**: Find the most frequently appearing numbers
5. **Number Predictions**: Predict 5 series of numbers based on latest year consolidated data
6. **Distribution Analysis**: Number range and odd/even distribution analysis
7. **Visual Analytics**: Interactive charts and statistics

## Requirements

- Python 3.8+
- ToTo.xlsx file in the project directory

## Installation

1. Clone or download this repository
2. Ensure ToTo.xlsx is in the project folder
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Option 1: Using the run script
```bash
./run_app.sh
```

### Option 2: Manual execution
```bash
streamlit run streamlit_app.py
```

Then open your browser and go to: http://localhost:8501

## Data Structure

The application expects ToTo.xlsx with the following columns:
- Draw: Draw number
- Date: Draw date
- Winning Number 1, 2, 3, 4, 5, 6: The six winning numbers
- Supplementary Number: The supplementary number
- Additional statistical columns

## Features Details

### 📊 Dataset Overview (Tab 1)
- Responsive to selected year and month parameters
- Total draws, date span, latest/earliest draw numbers
- Average draw gap, total numbers drawn, unique draw dates
- Recent draws preview for the selected period
- Statistical summary with winning number statistics
- Draw frequency by day of week
- Timeline visualization of monthly draw frequency

### 📈 Frequency Analysis (Tab 2)
- Select any year and month (or "All Months") to analyze
- View most frequent numbers for the selected period
- Number range distribution (1-10, 11-20, etc.)
- Statistical insights and ranking tables

### 🔮 Number Predictions (Tab 3)
- Generates 5 different sets using **probability-based strategy**:
  - **Set 1**: Highest probability (most frequent numbers from historical data)
  - **Set 2**: High probability (75% frequent + 25% medium frequent)
  - **Set 3**: Medium probability (50% frequent + 50% less frequent)
  - **Set 4**: Low probability (25% frequent + 75% less frequent)
  - **Set 5**: Lowest probability (least frequent numbers from historical data)
- Each set contains 6 winning numbers + 1 supplementary number (complete ToTo format)
- Machine learning-based predictions using Random Forest with frequency weighting
- Supplementary numbers follow same probability strategy as winning numbers
- Overall statistical analysis across all prediction sets

## Prediction Algorithm - Detailed Logic

The prediction system uses a sophisticated multi-layered approach to generate 5 different sets of numbers:

### 🧠 **Core Algorithm Steps:**

#### 1. **Data Preparation**
- Uses **latest year's data only** (2025) for maximum relevance
- Extracts all winning numbers from the 6 main positions
- **Analyzes supplementary numbers separately** from historical supplementary data
- Analyzes frequency patterns of numbers 1-49

#### 2. **Frequency Analysis Foundation**
```python
# Extract numbers from latest year
all_numbers = [all winning numbers from latest year draws]
number_counts = Counter(all_numbers)
most_frequent = [top 20 most frequent numbers]

# Analyze supplementary numbers separately
supplementary_numbers = [all supplementary numbers from latest year]
supplementary_counts = Counter(supplementary_numbers)
most_frequent_supplementary = [top 10 most frequent supplementary numbers]
```

#### 3. **Machine Learning Features (Primary Method)**
- **Training Data**: Uses sequential draw patterns from latest year
- **Features**: Current draw's 6 numbers + supplementary + statistics (Low, High, Odd, Even)
- **Target**: Next draw's first winning number
- **Model**: Random Forest Regressor (100 trees, max depth 10)

#### 4. **Prediction Generation Logic**
For each of the 5 sets, the algorithm:

**Winning Numbers Generation:**
```python
for set_num in range(5):  # Generate 5 different sets
    noise_factor = 0.1 + (set_num * 0.05)  # Different noise: 0.1, 0.15, 0.2, 0.25, 0.3
    varied_features = [f + random_noise(noise_factor) for f in last_draw_features]
    prediction = model.predict(varied_features)
    # Generate 6 unique winning numbers (1-49)
```

**Supplementary Number Generation (Separate Logic):**
```python
# Choose from historical supplementary number patterns
if most_frequent_supplementary:
    if random.random() < 0.7:  # 70% chance
        supplementary = choice(top_5_frequent_supplementary)
    else:
        supplementary = choice(all_historical_supplementary)
else:
    supplementary = random(1-49)  # Fallback

# Ensure supplementary differs from winning numbers
while supplementary in winning_numbers:
    supplementary = choice(supplementary_pool)
```

**Uniqueness Guarantee:**
- Uses `set()` data structure to ensure no duplicate numbers within each set
- Maximum 50 attempts per set to generate unique numbers
- Clips all predictions to valid ToTo range (1-49)
- Each set contains exactly 7 numbers (6 winning + 1 supplementary, matching complete ToTo format)

#### 5. **Fallback Mechanisms (Hierarchical)**

**Level 1 - ML Insufficient Data:**
```python
if len(training_data) < 3:
    # Use frequency-based prediction
```

**Level 2 - ML Prediction Gaps:**
```python
while len(predictions) < 7:
    # Fill gaps with most frequent numbers
    add_from_most_frequent_numbers()
```

**Level 3 - Final Fallback:**
```python
while len(predictions) < 7:
    # Add random valid numbers (1-49)
    add_random_number()
```

### 🎯 **Weighting System:**
- **70% probability**: Select from historically frequent numbers
- **30% probability**: Select completely random numbers (1-49)
- **Smart Balancing**: Combines data-driven insights with randomness

### 🔄 **Probability-Based Set Strategy:**
Each of the 5 sets uses **distinct probability approaches** for maximum diversity:
- **Set 1 (Highest)**: 100% from top 15 most frequent numbers + #1 most frequent supplementary
- **Set 2 (High)**: 80% from top 20 frequent + 20% others + 2nd-4th frequent supplementary  
- **Set 3 (Medium)**: 60% from top 25 frequent + 40% medium/less frequent + medium supplementary
- **Set 4 (Low)**: 30% from top 30 frequent + 70% less frequent + less frequent supplementary
- **Set 5 (Lowest)**: 10% from frequent + 90% least frequent numbers + least frequent supplementary

**Algorithm Emphasis:**
- **Frequency-Based Logic**: Primary method ensuring distinct sets
- **Machine Learning**: Secondary (1-2 numbers per set for Sets 1-3 only)
- **Supplementary Strategy**: Position-specific selection ensuring each set uses different supplementary pools

### 📊 **Quality Assurance:**
- **Range Validation**: All numbers guaranteed 1-49
- **Uniqueness**: Each set contains exactly 7 unique numbers (6 winning + 1 supplementary)
- **Variety**: Different algorithms ensure set diversity
- **Robustness**: Multiple fallback layers prevent failures

### 🧮 **Statistical Foundation:**
- **Winning Number Frequency**: Numbers that appeared often recently in winning positions
- **Supplementary Number Frequency**: Numbers that appeared often recently in supplementary position  
- **Pattern Recognition**: ML learns from sequential draw relationships
- **Controlled Randomness**: Prevents over-fitting to historical data
- **Ensemble Approach**: Combines multiple prediction methodologies
- **Position-Aware Logic**: Treats supplementary numbers differently from winning numbers

This creates a sophisticated prediction system that balances historical analysis, machine learning insights, and controlled randomness to generate diverse yet statistically-informed complete ToTo sets (6 winning + 1 supplementary number).

### 📊 Distribution Analysis (Tab 4)
- Number range distribution analysis for selected period
- Pie charts and statistics tables
- Odd vs Even number distribution
- Range-based frequency analysis

### 🎛️ Parameter Controls (Sidebar)
- **Year Selection**: Choose from available years in the dataset
- **Month Selection**: "All Months" or specific month (01-January, 02-February, etc.)
- **Responsive Design**: All tabs update based on selected parameters

## Previous Prediction Algorithm Summary

The application uses a hybrid approach:
1. **Machine Learning**: Random Forest model trained on historical patterns
2. **Frequency Analysis**: Incorporates most frequent numbers from recent data  
3. **Statistical Methods**: Ensures number uniqueness and valid ranges
4. **Multi-Set Generation**: Creates 5 diverse prediction sets using different noise factors
5. **Robust Fallbacks**: Multiple layers ensure reliable prediction generation

## Data Source

Historical data from: https://en.lottolyzer.com/history/singapore/toto/page/1/per-page/50/summary-view

## Disclaimer

This application is for educational and entertainment purposes only. Lottery numbers are random, and past performance does not guarantee future results. Please gamble responsibly.