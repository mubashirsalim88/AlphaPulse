import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from src.utils.logger import setup_logger

class TradingEnv(gym.Env):
    def __init__(self, config_path="configs/db_config.yaml", episode_length=2880):
        super(TradingEnv, self).__init__()
        self.logger = setup_logger("trading_env", "logs/trading_env.log")
        
        # Load data from PostgreSQL
        self.df = self._load_data(config_path)
        if self.df.empty:
            self.logger.error("No data loaded from PostgreSQL")
            raise ValueError("No data loaded from PostgreSQL")
        
        # Add technical indicators
        self.df = self._add_indicators()
        
        # Environment parameters
        self.episode_length = min(episode_length, len(self.df))
        self.initial_balance = 10000  # Match $10,000 deposit
        self.lot_size = 0.1  # Micro lot (10,000 units, ~$0.10/pip for EURUSD)
        self.spread = 0.00015  # 1.5 pips (IC Markets Raw Spread avg.)
        self.daily_drawdown_limit = 0.05  # 5%
        self.overall_drawdown_limit = 0.10  # 10%
        self.profit_target = 0.08  # 8%
        
        # State and action spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )  # [open, high, low, close, volume, sma_20, rsi_14]
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        
        # Trading state
        self.current_step = 0
        self.start_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0  # 1 (long), -1 (short), 0 (none)
        self.entry_price = 0
        self.daily_loss = 0
        self.max_equity = self.initial_balance
        self.done = False
        
    def _load_data(self, config_path):
        """Load EURUSD 15-min data from PostgreSQL using SQLAlchemy."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            connection_string = (
                f"postgresql+psycopg2://{config['user']}:{config['password']}@"
                f"{config['host']}:{config['port']}/{config['database']}"
            )
            engine = create_engine(connection_string)
            query = """
            SELECT timestamp_eet, open, high, low, close, volume
            FROM eurusd_15min
            ORDER BY timestamp_eet;
            """
            df = pd.read_sql(query, engine)
            engine.dispose()
            self.logger.info(f"Loaded {len(df)} candles from PostgreSQL")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return pd.DataFrame()

    def _add_indicators(self):
        """Add SMA and RSI indicators."""
        try:
            df = self.df.copy()
            df["sma_20"] = SMAIndicator(df["close"], window=20).sma_indicator()
            df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
            df = df.dropna().reset_index(drop=True)
            self.logger.info(f"Added indicators, {len(df)} candles after dropping NaN")
            return df
        except Exception as e:
            self.logger.error(f"Error adding indicators: {e}")
            return self.df

    def reset(self, **kwargs):
        """Reset environment for a new episode."""
        self.start_step = np.random.randint(0, len(self.df) - self.episode_length)
        self.current_step = self.start_step
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.daily_loss = 0
        self.max_equity = self.initial_balance
        self.done = False
        self.logger.info(f"Reset episode at step {self.current_step}")
        return self._get_obs(), {}

    def _get_obs(self):
        """Get current observation."""
        row = self.df.iloc[self.current_step]
        return np.array([
            row["open"],
            row["high"],
            row["low"],
            row["close"],
            row["volume"],
            row["sma_20"],
            row["rsi_14"]
        ], dtype=np.float32)

    def step(self, action):
        """Execute one step in the environment."""
        if self.done:
            return self._get_obs(), 0, True, False, {}

        self.current_step += 1
        current_price = self.df.iloc[self.current_step]["close"]
        reward = 0
        info = {}
        truncated = False

        # Update position and equity
        if self.position != 0:
            price_diff = (current_price - self.entry_price) * (1 if self.position == 1 else -1)
            profit = price_diff * self.lot_size * 100000  # Pips to USD
            self.equity = self.balance + profit
        else:
            self.equity = self.balance

        # Calculate drawdowns
        daily_drawdown = (self.max_equity - self.equity) / self.max_equity
        overall_drawdown = (self.initial_balance - self.equity) / self.initial_balance
        self.max_equity = max(self.max_equity, self.equity)

        # Execute action
        if action == 0 and self.position != 1:  # Buy
            if self.position == -1:  # Close short
                self.balance = self.equity
            self.position = 1
            self.entry_price = current_price + self.spread
            self.logger.info(f"Buy at {self.entry_price}")
        elif action == 1 and self.position != -1:  # Sell
            if self.position == 1:  # Close long
                self.balance = self.equity
            self.position = -1
            self.entry_price = current_price - self.spread
            self.logger.info(f"Sell at {self.entry_price}")
        elif action == 2:  # Hold
            pass

        # Reward function
        if self.position != 0:
            price_diff = (current_price - self.entry_price) * (1 if self.position == 1 else -1)
            reward = price_diff * self.lot_size * 100000  # Profit in USD
            # Penalize overbought/oversold RSI
            rsi = self.df.iloc[self.current_step]["rsi_14"]
            if rsi > 70 and self.position == 1:  # Overbought, long
                reward -= 50
            elif rsi < 30 and self.position == -1:  # Oversold, short
                reward -= 50
            # Reward trend alignment
            sma = self.df.iloc[self.current_step]["sma_20"]
            if current_price > sma and self.position == 1:  # Bullish, long
                reward += 10
            elif current_price < sma and self.position == -1:  # Bearish, short
                reward += 10

        # Check drawdown limits
        if daily_drawdown > self.daily_drawdown_limit:
            reward -= 1000  # Heavy penalty
            self.done = True
            self.logger.warning(f"Daily drawdown exceeded: {daily_drawdown*100:.2f}%")
        if overall_drawdown > self.overall_drawdown_limit:
            reward -= 1000
            self.done = True
            self.logger.warning(f"Overall drawdown exceeded: {overall_drawdown*100:.2f}%")

        # Check profit target
        profit = (self.equity - self.initial_balance) / self.initial_balance
        if profit >= self.profit_target:
            reward += 1000  # Bonus
            self.done = True
            self.logger.info(f"Profit target reached: {profit*100:.2f}%")

        # Check episode end
        if self.current_step >= self.start_step + self.episode_length or self.current_step >= len(self.df) - 1:
            self.done = True
            truncated = True
            self.logger.info("Episode ended")

        return self._get_obs(), reward, self.done, truncated, info

    def render(self):
        """Render current state (for debugging)."""
        row = self.df.iloc[self.current_step]
        self.logger.info(
            f"Step: {self.current_step}, Price: {row['close']}, "
            f"Balance: {self.balance:.2f}, Equity: {self.equity:.2f}, "
            f"Position: {self.position}, SMA: {row['sma_20']:.5f}, RSI: {row['rsi_14']:.2f}"
        )

if __name__ == "__main__":
    env = TradingEnv()
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Random action for testing
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if truncated:
            break