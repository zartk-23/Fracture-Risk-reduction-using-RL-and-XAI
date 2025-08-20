#%%
import numpy as np
import matplotlib.pyplot as plt
from environment import BoneHealthEnv
from stable_baselines3 import PPO

# Load model and env
model = PPO.load("ppo_bone_health")
env = BoneHealthEnv(csv_path="bone_health.csv")

# Simulate all 120 patients
all_rewards = []
all_micro_traumas = []
all_actions = []
for _ in range(len(env.data)):
    state, _ = env.reset()
    episode_rewards = []
    episode_micro_traumas = []
    episode_actions = []
    for _ in range(50):  # 50 weeks
        action, _ = model.predict(state)
        state, reward, done, truncated, _ = env.step(action)
        episode_rewards.append(reward)
        episode_micro_traumas.append(state[4])
        episode_actions.append(action)
        if done or truncated:
            break
    all_rewards.append(episode_rewards)
    all_micro_traumas.append(episode_micro_traumas)
    all_actions.append(episode_actions)

# Average across patients
avg_rewards = np.mean(all_rewards, axis=0)
avg_micro_traumas = np.mean(all_micro_traumas, axis=0)
avg_actions = np.mean(all_actions, axis=0)
#%%
# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(avg_rewards, label="Average Reward")
plt.title("Average Reward Over 50 Weeks")
plt.xlabel("Week")
plt.ylabel("Reward")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(avg_micro_traumas, label="Average MicroTrauma", color="orange")
plt.title("Average MicroTrauma Over 50 Weeks")
plt.xlabel("Week")
plt.ylabel("Events/Month")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(avg_actions, label="Average Action", color="green")
plt.title("Average Actions Over 50 Weeks")
plt.xlabel("Week")
plt.ylabel("Action (0-3)")
plt.legend()

plt.tight_layout()
plt.savefig("results.png", dpi=300, bbox_inches="tight")
print("Results plot saved as results.png")
plt.close()
# %%
