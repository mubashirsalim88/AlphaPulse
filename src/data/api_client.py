import MetaTrader5 as mt5
import yaml
import os
from datetime import datetime, timedelta
import logging
import time
import pytz
from src.utils.logger import setup_logger

class MT5APIClient:
    def __init__(self, config_path="configs/api_config.yaml"):
        """Initialize MT5 API client with config."""
        self.logger = setup_logger("mt5_api", "logs/data_collection.log")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.login = config.get("mt5_login")
        self.password = config.get("mt5_password")
        self.server = config.get("mt5_server")
        if not all([self.login, self.password, self.server]):
            self.logger.error("MT5 credentials missing in config")
            print("Error: MT5 credentials missing in config")
            raise ValueError("MT5 credentials missing")
        self.symbol = "EURUSD"
        self.timeframe = mt5.TIMEFRAME_M15
        self.max_retries = 3
        self.retry_delay = 5
        self.eet_tz = pytz.timezone("EET")

    def initialize(self):
        """Initialize MT5 connection."""
        for attempt in range(1, self.max_retries + 1):
            try:
                if mt5.initialize(login=self.login, password=self.password, server=self.server):
                    self.logger.info("MT5 initialized successfully")
                    print("MT5 initialized successfully")
                    return True
                else:
                    self.logger.error(f"MT5 initialize failed on attempt {attempt}: {mt5.last_error()}")
                    print(f"MT5 initialize failed on attempt {attempt}: {mt5.last_error()}")
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt}: {e}")
                print(f"Unexpected error on attempt {attempt}: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        return False

    def fetch_candles(self, start_date, end_date):
        """Fetch candlestick data for a date range."""
        if not self.initialize():
            self.logger.error("Cannot fetch data: MT5 initialization failed")
            print("Cannot fetch data: MT5 initialization failed")
            return []

        try:
            utc_from = start_date
            utc_to = end_date
            self.logger.info(f"Fetching data from {utc_from} to {utc_to}")
            print(f"Fetching data from {utc_from} to {utc_to}")
            rates = mt5.copy_rates_range(self.symbol, self.timeframe, utc_from, utc_to)
            
            if rates is None or len(rates) == 0:
                self.logger.warning(f"No data returned for {utc_from} to {utc_to}: {mt5.last_error()}")
                print(f"No data returned for {utc_from} to {utc_to}: {mt5.last_error()}")
                return []

            candles = []
            for rate in rates:
                # Convert Unix timestamp to EET
                eet_time = datetime.fromtimestamp(rate['time'], tz=pytz.UTC).astimezone(self.eet_tz)
                utc_time = eet_time.astimezone(pytz.UTC)
                candles.append({
                    "timestamp_utc": utc_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "timestamp_eet": eet_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "open": float(rate['open']),
                    "high": float(rate['high']),
                    "low": float(rate['low']),
                    "close": float(rate['close']),
                    "volume": int(rate['tick_volume'])
                })
            self.logger.info(f"Fetched {len(candles)} candles for {utc_from} to {utc_to}")
            print(f"Fetched {len(candles)} candles for {utc_from} to {utc_to}")
            return candles

        except Exception as e:
            self.logger.error(f"Error fetching candles: {e}")
            print(f"Error fetching candles: {e}")
            return []
        finally:
            mt5.shutdown()
            self.logger.info("MT5 connection closed")
            print("MT5 connection closed")

    def fetch_historical_data(self, start_date, end_date):
        """Fetch historical data in yearly batches."""
        all_candles = []
        current_date = start_date.replace(day=1, month=1)
        
        while current_date <= end_date:
            year_end = min(datetime(current_date.year, 12, 31), end_date)
            candles = self.fetch_candles(current_date, year_end)
            all_candles.extend(candles)
            current_date = datetime(current_date.year + 1, 1, 1)
            self.logger.info(f"Processed batch for {current_date.year - 1}")
            print(f"Processed batch for {current_date.year - 1}")
            time.sleep(2)

        # Sort and deduplicate
        all_candles = sorted(all_candles, key=lambda x: x["timestamp_utc"])
        unique_candles = {candle["timestamp_utc"]: candle for candle in all_candles}.values()
        self.logger.info(f"Total unique candles fetched: {len(unique_candles)}")
        print(f"Total unique candles fetched: {len(unique_candles)}")
        return list(unique_candles)

if __name__ == "__main__":
    client = MT5APIClient()
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 31)
    candles = client.fetch_historical_data(start, end)
    print(f"Candles fetched: {len(candles)}")