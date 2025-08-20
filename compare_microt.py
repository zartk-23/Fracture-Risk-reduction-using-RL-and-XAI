#%%
import matplotlib.pyplot as plt
import numpy as np

raw_categories = ['0-1', '2-3', '4+']
raw_freq = [16, 2, 2]  # Based on Table 3's 20 participants (16 at 0-1, 2 at 2-3, 2 at 4+)
binned_values = [0.5, 2.5, 4.5]
binned_freq = [16, 2, 2]  # Matching raw distribution after binning

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.bar(raw_categories, raw_freq, color='blue')
ax1.set_title('Raw MicroTrauma Frequency')
ax1.set_ylabel('Count')
ax2.bar(binned_values, binned_freq, color='green')
ax2.set_title('Processed MicroTrauma Frequency')
ax2.set_ylabel('Count')
plt.tight_layout()
plt.savefig('raw_vs_processed_microtrauma.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved raw_vs_processed_microtrauma.png")
# %%
