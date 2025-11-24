# Stock-trend-Prediction
TrendMaster: Stock Trend Prediction Web App

1. Project Title

TrendMaster: AI-Powered Stock Trend Analysis & Prediction System

2. Overview

TrendMaster is an educational web application designed to demonstrate the application of machine learning concepts in finance. It allows users to visualize historical stock data and generates trend predictions using Linear Regression algorithms. The system features a secure login mechanism, real-time data fetching from Yahoo Finance, and interactive visualizations.

3. Features

User Authentication: Secure login simulation to protect access.

Real-time Data: Fetches live stock market data using the yfinance API.

Interactive Visualization: Zoomable and interactive charts using Plotly.

Machine Learning Integration: Uses Linear Regression (Scikit-Learn) to predict future price trends based on historical performance.

Performance Analytics: Displays R2 Score and Mean Squared Error (MSE) to evaluate model accuracy.

Responsive Design: Built with Streamlit for a seamless mobile and desktop experience.

4. Technologies & Tools

Language: Python 3.9+

Framework: Streamlit (Web UI)

Data Source: yfinance (Yahoo Finance API)

Data Processing: Pandas, NumPy

Machine Learning: Scikit-Learn

Visualization: Plotly

Version Control: Git

5. Installation & Execution

Prerequisites

Ensure you have Python installed. You can check via:

python --version


Steps to Install

Clone the Repository

git clone [https://github.com/yourusername/trendmaster.git](https://github.com/yourusername/trendmaster.git)
cd trendmaster


Install Dependencies

pip install streamlit yfinance pandas numpy scikit-learn plotly


Steps to Run

Execute the following command in your terminal:

streamlit run stock_prediction_app.py


The application will open in your default browser at http://localhost:8501.

6. Instructions for Testing

Login: Use the credentials -> Username: student, Password: project2024.

Select Ticker: Enter a valid stock symbol (e.g., MSFT, TSLA, NVDA) in the sidebar.

Adjust Parameters: Change the "Years of History" slider and observe the data reload.

View Analysis: Click the "Trend Analysis" tab to see the prediction lines (Orange = Trend, Green = Future).

Check Metrics: Click "Model Metrics" to see the mathematical accuracy of the prediction.

7. Screenshots

(Placeholder for actual screenshots of the running app)

Login Screen

Main Dashboard

Prediction Chart
