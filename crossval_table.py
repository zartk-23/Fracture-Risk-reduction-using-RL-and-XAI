#%%
import numpy as np
from stable_baselines3 import PPO
from environment import BoneHealthEnv
from sklearn.model_selection import KFold

# Initialize the base environment
env = BoneHealthEnv(csv_path="bone_health.csv")
model = PPO.load("ppo_bone_health")  # Load your pre-trained model

# Set up 5-fold cross-validation
data_indices = np.arange(len(env.data))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

# Perform cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(data_indices)):
    # Create a new environment instance for this fold
    fold_env = BoneHealthEnv(csv_path="bone_health.csv")
    fold_env.set_patient_indices(train_idx)
    
    # Optionally re-train the model on the training subset
    # Note: If you want to re-train, initialize a new model or fine-tune the loaded one
    # For now, we'll use the pre-trained model for validation only
    model.set_env(fold_env)  # Associate the environment with the model
    model.learn(total_timesteps=20000, reset_num_timesteps=False)  # Fine-tune if desired
    
    # Switch to validation subset for reward calculation
    fold_env.set_patient_indices(val_idx)
    
    total_reward = 0
    for _ in range(len(val_idx)):
        state, _ = fold_env.reset()
        for step in range(50):
            action, _ = model.predict(state)
            state, reward, done, truncated, _ = fold_env.step(action)
            total_reward += reward
            if done or truncated or state[4] == 0:  # Stop if micro-trauma reaches 0
                break
        avg_reward = total_reward / 50  # Average reward per step
    fold_results.append((fold + 1, train_idx, val_idx, avg_reward))

# Print table data
print("Table 2: Cross-Validation Results")
print("| Fold | Training Indices | Validation Indices | Average Reward |")
print("|------|------------------|--------------------|----------------|")
for fold, train_idx, val_idx, avg_reward in fold_results:
    print(f"| {fold} | {train_idx} | {val_idx} | {avg_reward:.2f} |")

# Optionally save to a file for manual copying
with open("cross_validation_table.txt", "w") as f:
    f.write("Table 2: Cross-Validation Results\n")
    f.write("| Fold | Training Indices | Validation Indices | Average Reward |\n")
    f.write("|------|------------------|--------------------|----------------|\n")
    for fold, train_idx, val_idx, avg_reward in fold_results:
        f.write(f"| {fold} | {train_idx} | {val_idx} | {avg_reward:.2f} |\n")
print("Table data saved to cross_validation_table.txt")
# %%
