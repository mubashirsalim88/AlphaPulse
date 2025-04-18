from src.environment.trading_env import TradingEnv
from stable_baselines3 import PPO
import pandas as pd

def test_ppo(model_path="models/ppo_alphapulse.zip", episodes=10):
    # Initialize environment
    env = TradingEnv()
    
    # Load model
    model = PPO.load(model_path)
    
    # Test
    results = []
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        profits = []
        drawdowns = []
        
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Track profit and drawdown
            equity = env.equity
            profit = (equity - env.initial_balance) / env.initial_balance
            daily_drawdown = (env.max_equity - equity) / env.max_equity
            overall_drawdown = (env.initial_balance - equity) / env.initial_balance
            profits.append(profit)
            drawdowns.append(max(daily_drawdown, overall_drawdown))
            
            env.render()
        
        results.append({
            "episode": episode + 1,
            "reward": episode_reward,
            "length": episode_length,
            "profit": max(profits) if profits else 0,
            "max_drawdown": max(drawdowns) if drawdowns else 0
        })
    
    # Summarize
    df = pd.DataFrame(results)
    print("\nTest Results:")
    print(df)
    print("\nSummary:")
    print(f"Average Reward: {df['reward'].mean():.2f}")
    print(f"Average Length: {df['length'].mean():.2f}")
    print(f"Average Profit: {df['profit'].mean()*100:.2f}%")
    print(f"Average Max Drawdown: {df['max_drawdown'].mean()*100:.2f}%")
    
    return df

if __name__ == "__main__":
    test_ppo()