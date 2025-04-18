from src.environment.trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            ep_rew = self.locals["infos"][0].get("episode", {}).get("r", 0)
            ep_len = self.locals["infos"][0].get("episode", {}).get("l", 0)
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(ep_len)
            print(f"Episode {len(self.episode_rewards)}: Reward = {ep_rew:.2f}, Length = {ep_len}")
        return True

def train_ppo():
    # Initialize environment
    env = TradingEnv()
    
    # Check environment
    check_env(env)
    
    # Initialize PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0005,  # Increased from 0.0003
        n_steps=4096,          # Increased from 2048
        batch_size=128,        # Increased from 64
        n_epochs=15,           # Increased from 10
        gamma=0.99,
        tensorboard_log="./tensorboard/"
    )
    
    # Train with callback
    callback = LoggingCallback()
    model.learn(total_timesteps=500000, progress_bar=True, callback=callback, tb_log_name="ppo_alphapulse_v2")
    
    # Save model
    model.save("models/ppo_alphapulse_v2")
    print("Model saved to models/ppo_alphapulse_v2.zip")

if __name__ == "__main__":
    train_ppo()