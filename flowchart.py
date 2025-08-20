#%%
import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph()
G.add_edges_from([("Data Collection", "Preprocessing"), 
                 ("Preprocessing", "PPO Training"), 
                 ("PPO Training", "Cross-Validation"), 
                 ("Cross-Validation", "SHAP Analysis"), 
                 ("SHAP Analysis", "MicroTrauma Reduction")])
pos = nx.planar_layout(G)
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, arrows=True)
plt.title("End-to-End Workflow of RL/XAI Framework")
plt.savefig("workflow.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved workflow.png")
# %%
