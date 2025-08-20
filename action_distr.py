#%%
import numpy as np
import matplotlib.pyplot as plt
from environment import BoneHealthEnv
from stable_baselines3 import PPO

model = PPO.load("ppo_bone_health")
env = BoneHealthEnv(csv_path="bone_health.csv")

actions = np.zeros(4)  # 0=Activity, 1=Calcium, 2=Sleep, 3=Nothing
for _ in range(len(env.data)):
    state, _ = env.reset()
    for _ in range(50):
        action, _ = model.predict(state)
        actions[action] += 1
        state, _, done, truncated, _ = env.step(action)
        if done or truncated:
            break

plt.figure(figsize=(6, 4))
plt.bar(['Activity', 'Calcium', 'Sleep', 'Nothing'], actions / actions.sum() * 100, color=['blue', 'green', 'red', 'gray'])
plt.title('Percentage of Actions Chosen Over 50 Weeks')
plt.ylabel('Percentage (%)')
plt.savefig('action_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved action_distribution.png")
# %%
