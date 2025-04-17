import psycopg2
import yaml
from src.utils.logger import setup_logger

class DBManager:
    def __init__(self, config_path="configs/db_config.yaml"):
        self.logger = setup_logger("db_manager", "logs/db_manager.log")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.conn_params = {
            "host": config["host"],
            "port": config["port"],
            "database": config["database"],
            "user": config["user"],
            "password": config["password"]
        }
        self.conn = None
        self.cursor = None

    def connect(self):
        """Connect to PostgreSQL."""
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            self.cursor = self.conn.cursor()
            self.logger.info("Connected to PostgreSQL")
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            raise

    def create_table(self):
        """Create or update eurusd_15min table."""
        try:
            self.connect()
            # Create table with new timestamp_eet column
            create_table_query = """
            CREATE TABLE IF NOT EXISTS eurusd_15min (
                timestamp_utc TIMESTAMP NOT NULL,
                timestamp_eet TIMESTAMP NOT NULL,
                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                volume BIGINT NOT NULL,
                PRIMARY KEY (timestamp_utc)
            );
            """
            self.cursor.execute(create_table_query)
            self.conn.commit()
            self.logger.info("Table eurusd_15min created or verified")
        except Exception as e:
            self.logger.error(f"Error creating table: {e}")
            raise
        finally:
            self.close()

    def insert_candles(self, candles):
        """Insert candles into eurusd_15min table."""
        try:
            self.connect()
            insert_query = """
            INSERT INTO eurusd_15min (timestamp_utc, timestamp_eet, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (timestamp_utc) DO NOTHING;
            """
            data = [
                (
                    candle["timestamp_utc"],
                    candle["timestamp_eet"],
                    candle["open"],
                    candle["high"],
                    candle["low"],
                    candle["close"],
                    candle["volume"]
                )
                for candle in candles
            ]
            self.cursor.executemany(insert_query, data)
            self.conn.commit()
            self.logger.info(f"Inserted {self.cursor.rowcount} candles")
        except Exception as e:
            self.logger.error(f"Error inserting candles: {e}")
            raise
        finally:
            self.close()

    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        self.logger.info("Database connection closed")

if __name__ == "__main__":
    db = DBManager()
    db.create_table()