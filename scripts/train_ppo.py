from src.environment.trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

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
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99
    )
    
    # Train
    model.learn(total_timesteps=100000, progress_bar=True)
    
    # Save model
    model.save("models/ppo_alphapulse")
    print("Model saved to models/ppo_alphapulse.zip")

if __name__ == "__main__":
    train_ppo()