#%%
import numpy as np
from stable_baselines3 import PPO
from environment import BoneHealthEnv

# Initialize environment and model
env = BoneHealthEnv(csv_path="bone_health.csv")
model = PPO.load("ppo_bone_health")  # Load your trained model

# Select 20 random participants
np.random.seed(42)  # For reproducibility
sample_indices = np.random.choice(len(env.data), 20, replace=False)
env.set_patient_indices(sample_indices)

# Simulate and collect results
results = []
for idx in sample_indices:
    state, _ = env.reset()
    initial_mt = state[4]
    total_reward = 0
    steps = 0
    dominant_action = None
    action_counts = np.zeros(4)  # 0=Activity, 1=Calcium, 2=Sleep, 3=Nothing

    for step in range(50):
        action, _ = model.predict(state)
        action_counts[action] += 1
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if done or truncated or state[4] == 0:
            break

    final_mt = state[4]
    avg_reward = total_reward / steps if steps > 0 else total_reward / 50
    dominant_action = np.argmax(action_counts) if np.max(action_counts) > 0 else 3  # 3=None
    dominant_action = ["Activity", "Calcium", "Sleep", "None"][dominant_action]
    results.append((idx + 1, initial_mt, final_mt, avg_reward, dominant_action, steps))

# Print table data
print("Table 3: Sample Simulation Results (20 of 120 Participants)")
print("| Participant ID | Initial Micro-Trauma | Final Micro-Trauma | Reward | Dominant Action | Steps to Zero |")
print("|----------------|----------------------|--------------------|--------|-----------------|---------------|")
for pid, init_mt, final_mt, reward, action, steps in results:
    print(f"| {pid} | {init_mt:.1f} | {final_mt:.1f} | {reward:.1f} | {action} | {steps} |")

# Optionally save to a file for manual copying
with open("simulation_table.txt", "w") as f:
    f.write("Table 3: Sample Simulation Results (20 of 120 Participants)\n")
    f.write("| Participant ID | Initial Micro-Trauma | Final Micro-Trauma | Reward | Dominant Action | Steps to Zero |\n")
    f.write("|----------------|----------------------|--------------------|--------|-----------------|---------------|\n")
    for pid, init_mt, final_mt, reward, action, steps in results:
        f.write(f"| {pid} | {init_mt:.1f} | {final_mt:.1f} | {reward:.1f} | {action} | {steps} |\n")
print("Table data saved to simulation_table.txt")
# %%
