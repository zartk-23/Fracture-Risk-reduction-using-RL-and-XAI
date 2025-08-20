#%%
import numpy as np
import pandas as pd
import shap
from stable_baselines3 import PPO
from environment import BoneHealthEnv
import matplotlib.pyplot as plt

# Load trained model
model = PPO.load("ppo_bone_health")

# Load data
data = pd.read_csv("bone_health.csv")  # Update to 120-response CSV
feature_names = ["Age", "Activity", "Calcium", "Sleep", "MicroTrauma", "Gender", "Smoking"]

# Function to predict actions
def predict_fn(states):
    if len(states.shape) == 1:
        states = states.reshape(1, -1)
    actions, _ = model.predict(states, deterministic=True)
    return actions

# Use all 120 states
states = data[feature_names].values

# Compute SHAP values
explainer = shap.KernelExplainer(predict_fn, states)
shap_values = explainer.shap_values(states)
mean_shap = np.mean(np.abs(shap_values), axis=0)  # Mean absolute SHAP
print(f"Mean Absolute SHAP Values: {mean_shap}")

# Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, states, feature_names=feature_names, plot_type="bar")
plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
print("SHAP plot saved as shap_summary.png")
plt.close()
# %%
