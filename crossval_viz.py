#%%
import numpy as np
import matplotlib.pyplot as plt

# Simulated fold rewards (replace with actual values if saved)
fold_rewards = [2.1, 2.5, 2.0, 2.8, 2.3]  # Example; use your actual 5-fold results
avg_reward = np.mean(fold_rewards)
std_reward = np.std(fold_rewards)

plt.figure(figsize=(6, 4))
plt.boxplot(fold_rewards, vert=True, patch_artist=True)
plt.axhline(y=avg_reward, color='r', linestyle='--', label=f'Average: {avg_reward:.2f} ± {std_reward:.2f}')
plt.title('Cross-Validation Reward Distribution (5 Folds)')
plt.ylabel('Average Reward per Step')
plt.xlabel('Folds')
plt.legend()
plt.savefig('cross_validation_reward.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved cross_validation_reward.png")
# %%
