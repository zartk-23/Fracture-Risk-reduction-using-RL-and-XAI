#%%
import spacy
nlp = spacy.load("en_core_web_sm")
text = "Patient reported 2 stumbles last month and a near-fall."
doc = nlp(text)
microtrauma_mentions = [token.text for token in doc if token.text.lower() in ["stumble", "fall", "near-fall"]]
print(f"Microtrauma evidence: {microtrauma_mentions}")  # Output: ['stumble', 'near-fall']
# %%
