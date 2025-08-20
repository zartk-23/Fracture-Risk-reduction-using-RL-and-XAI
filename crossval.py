#%%
from sklearn.model_selection import KFold
import numpy as np
from stable_baselines3 import PPO
from environment import BoneHealthEnv

env = BoneHealthEnv(csv_path="bone_health.csv")
data_indices = np.arange(len(env.data))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kf.split(data_indices):
    env.set_patient_indices(train_idx)
    model = PPO("MlpPolicy", env, verbose=0, seed=42, ent_coef=0.2)
    model.learn(total_timesteps=20000)
    env.set_patient_indices(val_idx)
    state, _ = env.reset()
    total_reward = 0
    for _ in range(50):
        action, _ = model.predict(state)
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if done or truncated:
            break
    scores.append(total_reward / 50)
print(f"Average validation reward: {np.mean(scores)} ± {np.std(scores)}")
# %%
