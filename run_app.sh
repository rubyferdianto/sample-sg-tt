#!/bin/bash

# Navigate to the project directory
cd "/Applications/RF/NTU/SCTP in DSAI/sample-sg-tt"

# Activate virtual environment and run Streamlit
echo "Starting ToTo Analysis Dashboard..."
echo "Open your browser and go to: http://localhost:8501"
echo "Press Ctrl+C to stop the application"

"/Applications/RF/NTU/SCTP in DSAI/sample-sg-tt/.venv/bin/python" -m streamlit run streamlit_app.py
