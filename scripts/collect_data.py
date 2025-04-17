from datetime import datetime
from src.data.api_client import MT5APIClient
from src.data.db_manager import DBManager
from src.data.preprocessor import DataPreprocessor

def collect_and_store_data():
    """Fetch, clean, and store EURUSD 15-min data."""
    # Initialize components
    api_client = MT5APIClient()
    db_manager = DBManager()
    preprocessor = DataPreprocessor()

    # Define time range (5 years)
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)

    # Fetch data
    candles = api_client.fetch_historical_data(start_date, end_date)

    # Clean data
    cleaned_candles = preprocessor.clean_data(candles)

    # Store in PostgreSQL
    db_manager.create_table()
    db_manager.insert_candles(cleaned_candles)

    print(f"Data collection complete: {len(cleaned_candles)} candles stored")

if __name__ == "__main__":
    collect_and_store_data()