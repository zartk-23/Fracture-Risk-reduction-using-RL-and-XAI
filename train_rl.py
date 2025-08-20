#%%
import numpy as np
from stable_baselines3 import PPO
from environment import BoneHealthEnv

# Initialize environment
env = BoneHealthEnv(csv_path=r"C:\Users\zarth\OneDrive\Desktop\fracture_rl_xai\new\bone_health.csv")
train_indices, val_indices = env.split_data(train_ratio=0.8)

# Debug prints to verify split
print(f"Total patients: {len(env.data)}")  # Should be 120
print(f"Training indices length: {len(train_indices)}")  # Should be 96
print(f"Validation indices length: {len(val_indices)}")  # Should be 24

# Train on 80% of 120 (96 patients)
env.set_patient_indices(train_indices)  # Set training subset
model = PPO("MlpPolicy", env, verbose=1, seed=42, ent_coef=0.2)
model.learn(total_timesteps=20000)  # Consistent with manuscript
model.save("ppo_bone_health")

# Validate on 20% (24 patients)
env.set_patient_indices(val_indices)  # Set validation subset
state, _ = env.reset()
for _ in range(50):
    action, _states = model.predict(state)
    state, reward, done, truncated, _ = env.step(action)
    print(f"Validation - Action: {action}, Reward: {reward}, MicroTrauma: {state[4]}")
    if done or truncated:
        break
# %%
