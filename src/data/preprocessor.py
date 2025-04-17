import pandas as pd
from src.utils.logger import setup_logger

class DataPreprocessor:
    def __init__(self):
        self.logger = setup_logger("preprocessor", "logs/data_preprocessing.log")

    def clean_data(self, candles):
        """Clean and preprocess candlestick data."""
        try:
            if not candles:
                self.logger.warning("No candles to clean")
                return []

            # Convert to DataFrame
            df = pd.DataFrame(candles)

            # Ensure correct data types
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
            df["timestamp_eet"] = pd.to_datetime(df["timestamp_eet"])
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(int)

            # Remove duplicates based on timestamp_utc
            df = df.drop_duplicates(subset=["timestamp_utc"], keep="last")
            self.logger.info(f"Removed {len(candles) - len(df)} duplicate candles")

            # Check for missing values
            if df.isnull().any().any():
                self.logger.warning("Missing values detected, filling with forward fill")
                df = df.fillna(method="ffill")

            # Sort by timestamp_utc
            df = df.sort_values("timestamp_utc")

            # Convert back to list of dicts
            cleaned_candles = df.to_dict("records")
            self.logger.info(f"Cleaned {len(cleaned_candles)} candles")
            return cleaned_candles

        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            return []

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    # Test with sample data
    sample_candles = [
        {"timestamp_utc": "2020-01-01 00:00:00", "timestamp_eet": "2020-01-01 02:00:00", "open": 1.12345, "high": 1.12350, "low": 1.12340, "close": 1.12348, "volume": 100}
    ]
    cleaned = preprocessor.clean_data(sample_candles)
    print(cleaned)