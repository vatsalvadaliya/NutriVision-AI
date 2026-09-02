import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NutriVision AI",
    page_icon="🥗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# CLEAN / MINIMAL STYLING
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main {
        background-color: #fafafa;
    }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 720px;
    }
    h1 {
        font-weight: 600;
        letter-spacing: -0.5px;
        color: #1a1a1a;
    }
    .subtitle {
        color: #6b6b6b;
        font-size: 1.05rem;
        margin-top: -0.8rem;
        margin-bottom: 2.5rem;
    }
    div[data-testid="stFileUploader"] {
        border: 1.5px dashed #d0d0d0;
        border-radius: 12px;
        padding: 1.5rem;
        background-color: #ffffff;
    }
    .result-card {
        background-color: #ffffff;
        border: 1px solid #ececec;
        border-radius: 16px;
        padding: 2rem;
        margin-top: 2rem;
    }
    .dish-name {
        font-size: 1.6rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.2rem;
    }
    .confidence {
        color: #8a8a8a;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .nutrient-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }
    .nutrient-box {
        background-color: #f6f8f6;
        border-radius: 10px;
        padding: 1rem 0.5rem;
        text-align: center;
    }
    .nutrient-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a1a;
    }
    .nutrient-label {
        font-size: 0.78rem;
        color: #8a8a8a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# ⚠️ CONFIGURE THESE FOR YOUR MODEL
# ──────────────────────────────────────────────────────────────

# TODO: Replace with the exact class names your model was trained on,
# IN THE SAME ORDER as your training generator's class_indices.
CLASS_NAMES = [
    "pizza",
    "burger",
    "salad",
    "pasta",
    "sushi",
    # ... add the rest of your actual classes here
]

# TODO: Set this to whatever input size your model expects (check your
# training code, e.g. img_height, img_width used with ImageDataGenerator).
IMAGE_SIZE = (224, 224)

# TODO: Fill in real nutrition data for each class above.
# Values are per typical serving — adjust to your needs.
NUTRITION_DB = {
    "pizza":  {"calories": 285, "protein": 12, "carbs": 36, "fat": 10},
    "burger": {"calories": 354, "protein": 17, "carbs": 29, "fat": 20},
    "salad":  {"calories": 152, "protein": 5,  "carbs": 12, "fat": 10},
    "pasta":  {"calories": 221, "protein": 8,  "carbs": 43, "fat": 1.3},
    "sushi":  {"calories": 200, "protein": 9,  "carbs": 38, "fat": 0.7},
}

MODEL_PATH = "model/food_model.keras"


# ──────────────────────────────────────────────────────────────
# MODEL LOADING (cached so it only loads once, not on every rerun)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def predict(image: Image.Image):
    model = load_model()

    img = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100

    dish = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else "Unknown"
    return dish, confidence


# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────
st.markdown("# NutriVision AI")
st.markdown('<p class="subtitle">Upload a photo of your food to see what\'s in it.</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# UPLOAD
# ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a food photo",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    with st.spinner("Analyzing..."):
        dish, confidence = predict(image)

    nutrition = NUTRITION_DB.get(dish)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="dish-name">{dish.title()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence">{confidence:.1f}% confidence</div>', unsafe_allow_html=True)

    if nutrition:
        st.markdown(f"""
        <div class="nutrient-grid">
            <div class="nutrient-box">
                <div class="nutrient-value">{nutrition['calories']}</div>
                <div class="nutrient-label">Calories</div>
            </div>
            <div class="nutrient-box">
                <div class="nutrient-value">{nutrition['protein']}g</div>
                <div class="nutrient-label">Protein</div>
            </div>
            <div class="nutrient-box">
                <div class="nutrient-value">{nutrition['carbs']}g</div>
                <div class="nutrient-label">Carbs</div>
            </div>
            <div class="nutrient-box">
                <div class="nutrient-value">{nutrition['fat']}g</div>
                <div class="nutrient-label">Fat</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No nutrition data available for this dish yet.")

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<p style="color:#b0b0b0; text-align:center; margin-top:1rem;">'
        "No image uploaded yet</p>",
        unsafe_allow_html=True,
    )