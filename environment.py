#%%
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class BoneHealthEnv(gym.Env):
    def __init__(self, csv_path=r"C:\Users\zarth\OneDrive\Desktop\fracture_rl_xai\new\bone_health.csv"):
        super(BoneHealthEnv, self).__init__()
        self.data = pd.read_csv(csv_path)  # Load data, header is excluded by default
        print(f"Total patients loaded: {len(self.data)}")  # Debug: Should be 120
        self.all_patient_indices = np.arange(len(self.data))  # Indices for all data rows
        np.random.shuffle(self.all_patient_indices)  # Randomize once
        self.patient_indices = self.all_patient_indices  # Default to all indices
        self.patient_idx = 0  # Initialize patient index
        self.action_space = spaces.Discrete(4)  # 0=Activity, 1=Calcium, 2=Sleep, 3=Nothing
        self.observation_space = spaces.Box(low=0, high=100, shape=(7,), dtype=np.float32)
        self.state = None
        self.steps = 0
        self.max_steps = 50  # Updated to 50 weeks

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Ensure patient_idx stays within the assigned indices
        if self.patient_idx >= len(self.patient_indices):
            self.patient_idx = 0  # Reset to 0 when exceeding subset
        idx = self.patient_indices[self.patient_idx]
        self.state = np.array([
            self.data.iloc[idx]["Age"],
            self.data.iloc[idx]["Activity"],
            self.data.iloc[idx]["Calcium"],
            self.data.iloc[idx]["Sleep"],
            self.data.iloc[idx]["MicroTrauma"],
            self.data.iloc[idx]["Gender"],
            self.data.iloc[idx]["Smoking"]
        ], dtype=np.float32)
        self.steps = 0
        self.patient_idx = (self.patient_idx + 1) % len(self.patient_indices)  # Cycle within subset
        return self.state, {}

    def step(self, action):
        if action == 0:  # Increase Activity
            self.state[1] = min(self.state[1] + 1, 20)
            self.state[4] = max(self.state[4] - 0.25, 0)  # Updated reduction rate
        elif action == 1:  # Increase Calcium
            self.state[2] = min(self.state[2] + 1, 10)
            self.state[4] = max(self.state[4] - 0.125, 0)  # Updated reduction rate
        elif action == 2:  # Increase Sleep
            self.state[3] = min(self.state[3] + 1, 12)
            self.state[4] = max(self.state[4] - 0.125, 0)  # Updated reduction rate
        # Action 3 = Do nothing, no change

        risk = 0.1 * self.state[0] + 0.8 * self.state[4] - 0.1 * (self.state[1] + self.state[2] + self.state[3])
        reward = 10.0 if self.state[4] == 0 else -10.0 * self.state[4] - 0.1 * (self.state[1] + self.state[2] + self.state[3])
        self.steps += 1
        done = self.steps >= self.max_steps
        truncated = False
        return self.state, reward, done, truncated, {}

    def set_patient_indices(self, indices):
        """Set the subset of patient indices to use (e.g., train or val)."""
        self.patient_indices = indices
        self.patient_idx = 0  # Reset index when changing subsets

    def split_data(self, train_ratio=0.8):
        """Split data indices for training and validation."""
        n = len(self.data)
        indices = np.arange(n)  # Use arange to ensure correct size
        np.random.shuffle(indices)  # Shuffle in-place
        train_size = int(train_ratio * n)
        return indices[:train_size], indices[train_size:]
# %%r"C:\Users\zarth\OneDrive\Desktop\fracture_rl_xai\new\bone_health.csv"):
