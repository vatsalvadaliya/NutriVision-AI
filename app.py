import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np
import base64
from io import BytesIO

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
    #MainMenu, footer {visibility: hidden;}

    .app-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 4px;
    }
    .app-header-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: #e6f4ea;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }
    h1 {
        font-weight: 600;
        letter-spacing: -0.5px;
        color: #1a1a1a;
        margin: 0;
    }
    .subtitle {
        color: #6b6b6b;
        font-size: 0.95rem;
        margin: 0 0 2rem 46px;
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
        overflow: hidden;
        margin-top: 1.5rem;
    }
    .result-photo {
        width: 100%;
        height: 220px;
        object-fit: cover;
        display: block;
    }
    .result-body {
        padding: 1.5rem;
    }
    .result-top-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
    }
    .dish-name {
        font-size: 1.35rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0;
    }
    .confidence-note {
        color: #8a8a8a;
        font-size: 0.85rem;
        margin: 0 0 1.5rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
    }
    .badge-light   { background: #e6f4ea; color: #1e7b3d; }
    .badge-moderate{ background: #fdf1dc; color: #a3690f; }
    .badge-heavy   { background: #fbe7e7; color: #b23c3c; }

    .nutrient-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 1.5rem;
    }
    .nutrient-box {
        background-color: #f6f8f6;
        border-radius: 10px;
        padding: 0.75rem 0.4rem;
        text-align: center;
    }
    .nutrient-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a1a;
    }
    .nutrient-label {
        font-size: 0.7rem;
        color: #8a8a8a;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-top: 2px;
    }

    .macro-title {
        font-size: 0.72rem;
        color: #8a8a8a;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin: 0 0 10px;
    }
    .macro-row { margin-bottom: 10px; }
    .macro-row-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        margin-bottom: 4px;
    }
    .macro-row-name { color: #4a4a4a; }
    .macro-row-pct { color: #8a8a8a; }
    .macro-track {
        height: 6px;
        background: #f0f0f0;
        border-radius: 4px;
        overflow: hidden;
    }
    .macro-fill { height: 100%; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# ⚠️ CONFIGURE THESE FOR YOUR MODEL
# ──────────────────────────────────────────────────────────────

# These are your model's actual trained classes, in alphabetical order
# (matches Keras' default folder-based class indexing).
CLASS_NAMES = [
    "aloo gobi",
    "aloo matar",
    "aloo tikki",
    "biryani",
    "butter chicken",
    "chana masala",
    "chicken tikka",
    "chicken tikka masala",
    "dal makhani",
    "dal tadka",
    "gulab jamun",
    "jalebi",
    "kadai paneer",
    "naan",
    "palak paneer",
    "paneer butter masala",
    "poha",
    "rasgulla",
    "shrikhand",
]

# TODO: Set this to whatever input size your model expects (check your
# training code, e.g. img_height, img_width used with ImageDataGenerator).
IMAGE_SIZE = (160, 160)

# TODO: These are rough per-serving estimates — replace with more precise
# values if you have a specific source or serving size in mind.
NUTRITION_DB = {
    "aloo gobi":            {"calories": 180, "protein": 4,  "carbs": 22, "fat": 9},
    "aloo matar":           {"calories": 190, "protein": 5,  "carbs": 24, "fat": 8},
    "aloo tikki":           {"calories": 210, "protein": 4,  "carbs": 28, "fat": 10},
    "biryani":              {"calories": 350, "protein": 12, "carbs": 45, "fat": 12},
    "butter chicken":       {"calories": 490, "protein": 27, "carbs": 12, "fat": 36},
    "chana masala":         {"calories": 210, "protein": 9,  "carbs": 30, "fat": 6},
    "chicken tikka":        {"calories": 230, "protein": 28, "carbs": 4,  "fat": 11},
    "chicken tikka masala": {"calories": 440, "protein": 26, "carbs": 14, "fat": 30},
    "dal makhani":          {"calories": 280, "protein": 11, "carbs": 26, "fat": 15},
    "dal tadka":            {"calories": 180, "protein": 9,  "carbs": 24, "fat": 6},
    "gulab jamun":          {"calories": 300, "protein": 4,  "carbs": 45, "fat": 12},
    "jalebi":               {"calories": 310, "protein": 2,  "carbs": 55, "fat": 9},
    "kadai paneer":         {"calories": 320, "protein": 14, "carbs": 12, "fat": 25},
    "naan":                 {"calories": 260, "protein": 8,  "carbs": 45, "fat": 5},
    "palak paneer":         {"calories": 290, "protein": 13, "carbs": 10, "fat": 22},
    "paneer butter masala": {"calories": 400, "protein": 15, "carbs": 14, "fat": 31},
    "poha":                 {"calories": 180, "protein": 4,  "carbs": 30, "fat": 5},
    "rasgulla":             {"calories": 186, "protein": 4,  "carbs": 33, "fat": 4},
    "shrikhand":            {"calories": 220, "protein": 6,  "carbs": 30, "fat": 8},
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
st.markdown(
    '<div class="app-header"><div class="app-header-icon">🥗</div><h1>NutriVision AI</h1></div>'
    '<p class="subtitle">Upload a photo of your food to see what\'s in it.</p>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# UPLOAD
# ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a food photo",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

def image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def health_badge(calories: float):
    """Simple calorie-based indicator. Tune thresholds to your data."""
    if calories < 250:
        return "light", "Light"
    elif calories < 450:
        return "moderate", "Moderate"
    else:
        return "heavy", "Heavy"


if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Analyzing..."):
        dish, confidence = predict(image)

    nutrition = NUTRITION_DB.get(dish)
    img_b64 = image_to_base64(image)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(
        f'<img class="result-photo" src="data:image/jpeg;base64,{img_b64}" />',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="result-body">', unsafe_allow_html=True)

    if nutrition:
        badge_class, badge_label = health_badge(nutrition["calories"])
        st.markdown(
            f'<div class="result-top-row">'
            f'<p class="dish-name">{dish.title()}</p>'
            f'<span class="badge badge-{badge_class}">{badge_label}</span>'
            f'</div>'
            f'<p class="confidence-note">{confidence:.1f}% confidence</p>'
            f'<div class="nutrient-grid">'
            f'<div class="nutrient-box"><div class="nutrient-value">{nutrition["calories"]}</div><div class="nutrient-label">Kcal</div></div>'
            f'<div class="nutrient-box"><div class="nutrient-value">{nutrition["protein"]}g</div><div class="nutrient-label">Protein</div></div>'
            f'<div class="nutrient-box"><div class="nutrient-value">{nutrition["carbs"]}g</div><div class="nutrient-label">Carbs</div></div>'
            f'<div class="nutrient-box"><div class="nutrient-value">{nutrition["fat"]}g</div><div class="nutrient-label">Fat</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Macro breakdown as % of calories (protein/carbs = 4 kcal/g, fat = 9 kcal/g)
        p_cal = nutrition["protein"] * 4
        c_cal = nutrition["carbs"] * 4
        f_cal = nutrition["fat"] * 9
        total = max(p_cal + c_cal + f_cal, 1)

        macros = [
            ("Protein", p_cal / total * 100, "#1e9e5a"),
            ("Carbs", c_cal / total * 100, "#3a7bd5"),
            ("Fat", f_cal / total * 100, "#e0a52c"),
        ]

        rows_html = '<p class="macro-title">Macro breakdown</p>'
        for name, pct, color in macros:
            rows_html += (
                '<div class="macro-row">'
                '<div class="macro-row-labels">'
                f'<span class="macro-row-name">{name}</span>'
                f'<span class="macro-row-pct">{pct:.0f}%</span>'
                '</div>'
                '<div class="macro-track">'
                f'<div class="macro-fill" style="width:{pct:.0f}%; background:{color};"></div>'
                '</div>'
                '</div>'
            )
        st.markdown(rows_html, unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="dish-name">{dish.title()}</p>', unsafe_allow_html=True)
        st.info("No nutrition data available for this dish yet.")

    st.markdown('</div>', unsafe_allow_html=True)  # .result-body
    st.markdown('</div>', unsafe_allow_html=True)  # .result-card
else:
    st.markdown(
        '<p style="color:#b0b0b0; text-align:center; margin-top:1rem;">'
        "No image uploaded yet</p>",
        unsafe_allow_html=True,
    )