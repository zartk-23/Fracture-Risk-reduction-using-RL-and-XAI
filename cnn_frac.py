
# %%
import matplotlib.pyplot as plt

enhancements = ['Expansion (200+)', 'Clinical Trial', 'NLP Integration', 'CNN Integration', 'Mobile Deployment']
impact = [30, 25, 20, 20, 15]  # Hypothetical impact percentages
plt.figure(figsize=(10, 6))
plt.bar(enhancements, impact, color=['blue', 'green', 'orange', 'red', 'purple'])
plt.title('Future Work Enhancements and Expected Impact')
plt.ylabel('Priority/Impact (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('future_work_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved future_work_visualization.png")
# %%
