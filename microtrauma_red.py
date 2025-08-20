#%%
import numpy as np
import matplotlib.pyplot as plt
from environment import BoneHealthEnv
from stable_baselines3 import PPO

# Load model and env
model = PPO.load("ppo_bone_health")
env = BoneHealthEnv(csv_path="bone_health.csv")

# Simulate all 120 patients
all_micro_traumas = []
for _ in range(len(env.data)):
    state, _ = env.reset()
    episode_micro_traumas = [state[4]]  # Initial micro-trauma
    for _ in range(50):
        action, _ = model.predict(state)
        state, _, done, truncated, _ = env.step(action)
        episode_micro_traumas.append(state[4])
        if done or truncated:
            break
    all_micro_traumas.append(episode_micro_traumas)

# Average across patients
avg_micro_traumas = np.mean(all_micro_traumas, axis=0)

plt.figure(figsize=(8, 4))
plt.plot(range(51), avg_micro_traumas, label="Average MicroTrauma", color="orange")
plt.axhline(y=0, color='r', linestyle='--', label="Target: 0")
plt.title('Average MicroTrauma Reduction Over 50 Weeks')
plt.xlabel('Week')
plt.ylabel('MicroTrauma Events/Month')
plt.legend()
plt.savefig('microtrauma_reduction.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved microtrauma_reduction.png")
# %%
