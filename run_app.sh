#!/bin/bash

# Navigate to the project directory
cd "C:\Ruby\GitClone\sample-sg-tt"

# Activate virtual environment and run Streamlit
echo "Starting ToTo Analysis Dashboard..."
echo "Open your browser and go to: http://localhost:8501"
echo "Press Ctrl+C to stop the application"

"C:\Ruby\GitClone\sample-sg-tt\.venv\Scripts\python.exe" -m streamlit run streamlit_app.py
