import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import cv2
import time

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Smart Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ==============================
# CONSTANTS
# ==============================

MODEL_PATH = "model/plant_disease_model_final.h5"
CLASS_PATH = "model/class_names.json"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60

# ==============================
# FAST MODEL LOADING
# ==============================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_class_names():
    with open(CLASS_PATH, "r") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

# ==============================
# SEVERITY ESTIMATION
# ==============================

def estimate_severity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    infected_pixels = np.sum(gray < 100)
    total_pixels = gray.size
    ratio = infected_pixels / total_pixels

    if ratio < 0.15:
        return "Low"
    elif ratio < 0.35:
        return "Moderate"
    else:
        return "High"

# ==============================
# DYNAMIC PESTICIDE ENGINE
# ==============================

def generate_remedy(disease, severity):
    disease_parts = disease.split("___")
    crop = disease_parts[0]
    disease_name = disease_parts[1].replace("_", " ")
    disease_lower = disease_name.lower()

    if "healthy" in disease_lower:
        return (
            "No chemical treatment required. Maintain proper irrigation and soil nutrition.",
            "Use compost and neem oil preventive spray."
        )

    if "blight" in disease_lower:
        base = "Apply Mancozeb or Chlorothalonil fungicide"
        organic = "Neem oil spray every 7 days"

    elif "rust" in disease_lower:
        base = "Apply Propiconazole or Azoxystrobin fungicide"
        organic = "Sulfur-based organic spray"

    elif "mildew" in disease_lower:
        base = "Use Sulfur or Potassium bicarbonate spray"
        organic = "Milk spray solution (1:10 ratio)"

    elif "spot" in disease_lower:
        base = "Apply Copper-based fungicide"
        organic = "Garlic extract spray"

    elif "rot" in disease_lower:
        base = "Apply Carbendazim fungicide"
        organic = "Trichoderma bio-control treatment"

    else:
        base = "Use broad-spectrum Mancozeb-based fungicide"
        organic = "Neem oil preventive spray"

    if severity == "Low":
        dosage = "Use standard dosage once per week."
    elif severity == "Moderate":
        dosage = "Apply twice per week with monitoring."
    else:
        dosage = "Apply every 4–5 days until infection reduces."

    chemical = f"{base}. {dosage}"

    return chemical, organic

# ==============================
# CORRECT PREDICTION FUNCTION
# ==============================

def predict_disease(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)

    # 🔥 IMPORTANT FIX (Correct preprocessing)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    top_indices = predictions[0].argsort()[-3:][::-1]

    results = []
    for i in top_indices:
        results.append((class_names[i], float(predictions[0][i])))

    return results

# ==============================
# UI
# ==============================

st.title("🌿 Smart Plant Disease Detection System")
st.markdown("### AI-Powered Crop Health Monitoring & Remedy Recommendation")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload a clear image of the plant leaf",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        time.sleep(1)
        predictions = predict_disease(image)

    top_disease, confidence = predictions[0]

    if confidence < CONFIDENCE_THRESHOLD:
        with col2:
            st.error("Low confidence prediction. Please upload a clearer image.")
    else:
        severity = estimate_severity(np.array(image))
        chemical, organic = generate_remedy(top_disease, severity)

        with col2:
            st.success(f"Predicted Disease: {top_disease}")
            st.write(f"Confidence: {round(confidence * 100, 2)}%")
            st.write(f"Estimated Severity: {severity}")

            st.markdown("### Recommended Chemical Treatment")
            st.info(chemical)

            st.markdown("### Organic Alternative")
            st.warning(organic)

        st.markdown("---")
        st.markdown("### Top 3 Predictions")

        for disease, conf in predictions:
            st.write(f"{disease} — {round(conf * 100, 2)}%")

st.markdown("---")
st.caption("Production-grade AI system for real-world agricultural disease detection.")
