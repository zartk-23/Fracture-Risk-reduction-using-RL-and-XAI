#%%
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, AnnotationBbox
import matplotlib.patches as patches

# Data to visualize
nlp_result = {
    'text': 'Patient reports frequent stumbles; prescribe calcium.',
    'microtrauma': True,
    'matched_keywords': ['stumble']
}

# Create figure
plt.figure(figsize=(10, 5), dpi=300)

# Create a rectangle for the text box
ax = plt.gca()
rect = patches.Rectangle((0.15, 0.25), 0.7, 0.4, linewidth=1, edgecolor='black', facecolor='lightgray', transform=ax.transAxes)
ax.add_patch(rect)

# Add text with manual positioning and coloring
plt.text(0.25, 0.55, "'text': ", color='black', fontsize=10, transform=ax.transAxes)
plt.text(0.31, 0.55, nlp_result['text'], color='black', fontsize=10, transform=ax.transAxes, wrap=True)
plt.text(0.25, 0.45, "'microtrauma': ", color='black', fontsize=10, transform=ax.transAxes)
plt.text(0.45, 0.45, str(nlp_result['microtrauma']), color='green', fontsize=10, transform=ax.transAxes)
plt.text(0.25, 0.35, "'matched_keywords': ", color='black', fontsize=10, transform=ax.transAxes)
plt.text(0.45, 0.35, str(nlp_result['matched_keywords']), color='blue', fontsize=10, transform=ax.transAxes)

# Add title
plt.title('NLP Analysis Output for Prescription Text', fontsize=12, pad=10)

# Hide axes
plt.gca().set_axis_off()

# Save the figure
plt.savefig('nlp_output_visualization_final.png', dpi=300, bbox_inches='tight')
plt.show()
n # %%
