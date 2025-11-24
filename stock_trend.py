import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go
from datetime import date, timedelta
import logging

# --- CONFIGURATION & LOGGING ---
st.set_page_config(page_title="TrendMaster Stock AI", layout="wide", page_icon="📈")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MODULE 1: USER MANAGEMENT (Simulated) ---
class UserManager:
    """
    Handles user authentication and session management.
    In a real-world scenario, this would connect to a SQL/NoSQL database.
    """
    def __init__(self):
        # Mock database of users
        self.users = {
            "admin": "password123",
            "student": "project2024"
        }

    def login(self, username, password):
        if username in self.users and self.users[username] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            logger.info(f"User {username} logged in successfully.")
            return True
        logger.warning(f"Failed login attempt for {username}.")
        return False

    def logout(self):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        logger.info("User logged out.")

# --- MODULE 2: DATA INPUT & PROCESSING ---
class DataProcessor:
    """
    Handles fetching data from APIs and cleaning/preparing it for the model.
    """
    @staticmethod
    @st.cache_data(ttl=3600) # Caching for Performance (Non-functional req)
    def load_data(ticker, start_date, end_date):
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                return None
            df.reset_index(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None

    @staticmethod
    def prepare_data_for_model(df):
        """
        Prepares dataframe for Linear Regression.
        Converts dates to ordinal numbers for regression.
        """
        df['Date_Ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
        X = df[['Date_Ordinal']]
        y = df['Close']
        return X, y

# --- MODULE 3: PREDICTION & ANALYTICS ---
class TrendPredictor:
    """
    Handles the Machine Learning logic.
    """
    def __init__(self):
        self.model = LinearRegression()

    def train_and_predict(self, X, y, future_days=30):
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Predict Future
        last_date = X.iloc[-1, 0]
        future_dates = np.array([last_date + i for i in range(1, future_days + 1)]).reshape(-1, 1)
        future_prices = self.model.predict(future_dates)
        
        return {
            'model': self.model,
            'mse': mse,
            'r2': r2,
            'future_dates': future_dates,
            'future_prices': future_prices,
            'test_predictions': predictions,
            'X_test': X_test,
            'y_test': y_test
        }

# --- MODULE 4: UI & VISUALIZATION ---
class Dashboard:
    """
    Handles the rendering of the Streamlit UI.
    """
    def render_login(self, user_manager):
        st.title("🔐 Login to TrendMaster")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if user_manager.login(username, password):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try 'student' / 'project2024'")

    def render_main(self, user_manager, data_processor, predictor):
        # Sidebar
        with st.sidebar:
            st.title(f"Welcome, {st.session_state['username']}")
            if st.button("Logout"):
                user_manager.logout()
                st.rerun()
            
            st.header("Configuration")
            ticker = st.text_input("Stock Ticker (e.g., AAPL, GOOG)", "AAPL")
            years = st.slider("Years of History", 1, 10, 3)
            future_days = st.slider("Days to Predict", 7, 90, 30)

        # Main Content
        st.title("📈 Stock Trend Prediction Engine")
        
        start_date = date.today() - timedelta(days=years*365)
        end_date = date.today()

        with st.spinner('Fetching market data...'):
            data = data_processor.load_data(ticker, start_date, end_date)

        if data is not None:
            # Tab Structure for clean UI
            tab1, tab2, tab3 = st.tabs(["Historical Data", "Trend Analysis", "Model Metrics"])
            
            with tab1:
                st.subheader(f"{ticker} Historical Data")
                st.dataframe(data.tail())
                
                # Simple Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price"))
                fig.layout.update(title_text='Time Series Data with Rangeslider', xaxis_rangeslider_visible=True)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Price Prediction (Linear Trend)")
                
                # Prepare and Predict
                X, y = data_processor.prepare_data_for_model(data)
                results = predictor.train_and_predict(X, y, future_days)
                
                # Convert ordinal back to datetime for plotting
                future_dt = [date.fromordinal(int(d[0])) for d in results['future_dates']]
                
                # Advanced Plotting
                fig_pred = go.Figure()
                
                # Actual Data
                fig_pred.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Actual Price", line=dict(color='blue')))
                
                # Regression Line on Historical
                full_pred = results['model'].predict(X)
                fig_pred.add_trace(go.Scatter(x=data['Date'], y=full_pred, name="Trend Line", line=dict(color='orange', dash='dash')))
                
                # Future Prediction
                fig_pred.add_trace(go.Scatter(x=future_dt, y=results['future_prices'], name="Forecast", line=dict(color='green')))
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                st.info(f"Predicted price in {future_days} days: ${results['future_prices'][-1]:.2f}")

            with tab3:
                st.subheader("Model Evaluation")
                col1, col2 = st.columns(2)
                col1.metric("R2 Score", f"{results['r2']:.4f}", help="Close to 1 is better")
                col2.metric("Mean Squared Error", f"{results['mse']:.4f}", help="Lower is better")
                
                st.markdown("""
                **Methodology:**
                We use a **Linear Regression** algorithm to identify the macro trend of the stock over the selected time period.
                *Note: Stock markets are non-linear; this model assumes a linear trend and should be used for educational purposes only.*
                """)

        else:
            st.error("Could not load data. Please check the Ticker symbol.")

# --- MAIN APP ENTRY POINT ---
def main():
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # Instantiate Modules
    user_manager = UserManager()
    data_processor = DataProcessor()
    predictor = TrendPredictor()
    dashboard = Dashboard()

    # Router Logic
    if not st.session_state['logged_in']:
        dashboard.render_login(user_manager)
    else:
        dashboard.render_main(user_manager, data_processor, predictor)

if __name__ == "__main__":
    main()
