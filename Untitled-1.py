#%%
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow_hub as hub  # For a lightweight NLP alternative
import re

# 1. CNN Model for Fracture Image Feature Extraction
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),  # Grayscale X-ray input
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # Binary: Fracture vs. No Fracture
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Example: Load and preprocess a sample X-ray image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(64, 64), color_mode='grayscale')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

# Simulate CNN prediction
cnn_model = create_cnn_model()
try:
    sample_image = preprocess_image('sample_xray.png')  # Placeholder for X-ray image
    prediction = cnn_model.predict(sample_image)
    print("CNN Prediction (Fracture Probability):", prediction[0])
except FileNotFoundError:
    print("Warning: 'sample_xray.png' not found. Using random data for demo.")
    sample_image = np.random.rand(1, 64, 64, 1)
    prediction = cnn_model.predict(sample_image)
    print("CNN Demo Prediction (Random Data):", prediction[0])

# 2. NLP Model for Prescription Text Analysis (TensorFlow-based keyword extraction)
def analyze_prescription(text):
    microtrauma_keywords = ['stumble', 'fall', 'balance', 'trauma']
    evidence = any(re.search(rf'\b{word}\b', text.lower()) for word in microtrauma_keywords)
    return {"text": text, "microtrauma_evidence": evidence, "matched_keywords": [word for word in microtrauma_keywords if word in text.lower()]}

# Load a simple TF-Hub text embedding (optional for future scaling)
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Example prescription text
prescription_text = "Patient reports frequent stumbles; prescribe calcium and physical therapy."
nlp_result = analyze_prescription(prescription_text)
embedding = embed([prescription_text])[0]  # Get embedding for future use
print("NLP Analysis:", nlp_result)
print("Text Embedding (for future ML):", embedding[:5])  # Show first 5 dimensions

# 3. Future Integration Concept (Pseudo-code)
"""
Future Work:
- CNN: Train on 1000+ X-ray images to classify fracture severity, extracting features (e.g., bone density) for RL state.
- NLP: Scale keyword extraction to 500+ prescriptions, using TF-Hub embeddings to quantify micro-trauma impact, feeding into RL rewards.
- Integrate with PPO via updated environment.py:
  state = [cnn_features, nlp_embedding, survey_data]
  reward += nlp_microtrauma_score
"""

# Run the models (for demonstration)
if __name__ == "__main__":
    print("CNN Model Summary:")
    cnn_model.summary()
    print("\nNLP Prescription Analysis:")
    print(nlp_result)
# %%
