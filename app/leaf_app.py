# import streamlit as st
# import cv2
# import numpy as np
# import joblib
# from skimage.feature import hog

# # --- Load models ---
# clf = joblib.load("../models/leaf_classifier_svm_rbf.pkl")
# scaler = joblib.load("../models/scaler.pkl")
# pca = joblib.load("../models/pca.pkl")

# # --- Feature extraction ---
# def extract_features(img):
#     img = cv2.resize(img, (128,128))
#     _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     hu = cv2.HuMoments(cv2.moments(thresh)).flatten()
#     hog_features = hog(img, pixels_per_cell=(16,16), cells_per_block=(2,2),
#                        orientations=9, block_norm='L2-Hys')
#     return np.hstack([hu, hog_features])

# # --- Streamlit App ---
# st.title("Leaf Classification App 🌿")
# st.write("Upload a leaf image to predict its species.")

# uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg","png","jpeg"])

# if uploaded_file is not None:
#     # Convert to numpy array
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
#     st.image(img, caption='Uploaded Leaf', use_column_width=True)
    
#     # Extract features & predict
#     features = extract_features(img).reshape(1, -1)
#     features_scaled = scaler.transform(features)
#     features_pca = pca.transform(features_scaled)
    
#     prediction = clf.predict(features_pca)[0]
    
#     st.success(f"Predicted Species: {prediction}")



# import streamlit as st
# import cv2
# import numpy as np
# import joblib
# from skimage.feature import hog
# import base64

# # =========================
# # --- Custom Page Config ---
# # =========================
# st.set_page_config(
#     page_title="Leaf Classification 🌿",
#     page_icon="🍃",
#     layout="centered",
# )

# # =========================
# # --- Background Styling ---
# # =========================
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as f:
#         encoded = base64.b64encode(f.read()).decode()
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("data:image/png;base64,{encoded}");
#             background-size: cover;
#             background-position: center;
#             background-attachment: fixed;
#             opacity: 0.95;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # ضع أي صورة خلفية شفافة هنا (PNG)
# add_bg_from_local("background.png")  

# # =========================
# # --- Load Models ---
# # =========================
# clf = joblib.load("../models/leaf_classifier_svm_rbf.pkl")
# scaler = joblib.load("../models/scaler.pkl")
# pca = joblib.load("../models/pca.pkl")

# # =========================
# # --- Feature Extraction ---
# # =========================
# def extract_features(img):
#     img = cv2.resize(img, (128,128))
#     _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     hu = cv2.HuMoments(cv2.moments(thresh)).flatten()
#     hog_features = hog(img, pixels_per_cell=(16,16), cells_per_block=(2,2),
#                        orientations=9, block_norm='L2-Hys')
#     return np.hstack([hu, hog_features])

# # =========================
# # --- App Layout ---
# # =========================
# st.markdown(
#     "<h1 style='text-align: center; color: white;'>🍃 Leaf Classification App 🌿</h1>",
#     unsafe_allow_html=True
# )
# st.write("<p style='text-align: center; color: #f0f0f0;'>Upload a leaf image and let AI predict its species!</p>", unsafe_allow_html=True)

# uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg","png","jpeg"])

# if uploaded_file is not None:
#     # Convert to numpy array
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
#     st.image(img, caption="🌱 Uploaded Leaf", use_column_width=True)

#     # Extract features & predict
#     features = extract_features(img).reshape(1, -1)
#     features_scaled = scaler.transform(features)
#     features_pca = pca.transform(features_scaled)

#     # Get probabilities
#     probs = clf.decision_function(features_pca)
#     top3_idx = np.argsort(probs[0])[-3:][::-1]
#     top3_labels = clf.classes_[top3_idx]
#     top3_scores = probs[0][top3_idx]

#     # Display Predictions
#     st.subheader("🔮 Prediction Results")
#     for label, score in zip(top3_labels, top3_scores):
#         st.markdown(
#             f"""
#             <div style="background: rgba(0, 0, 0, 0.6); padding: 10px; 
#                         border-radius: 10px; margin-bottom: 10px; color: white;">
#                 🌿 <b>{label}</b> — {score*100:.2f}%
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
import base64

# =========================
# --- Page Config ---
# =========================
st.set_page_config(
    page_title="Leaf Classification 🌿",
    page_icon="🍃",
    layout="wide",
)

# =========================
# --- Background Styling ---
# =========================
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            background: rgba(0,0,0,0.55);
            padding: 2rem;
            border-radius: 20px;
        }}
        h1, h2, h3, p {{
            color: #f5f5f5;
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# استبدل الخلفية بصورة مناسبة (PNG بخلفية شفافة أفضل)
add_bg_from_local("background.png")

# =========================
# --- Load Models ---
# =========================
clf = joblib.load("../models/leaf_classifier_svm_rbf.pkl")
scaler = joblib.load("../models/scaler.pkl")
pca = joblib.load("../models/pca.pkl")

# =========================
# --- Feature Extraction ---
# =========================
def extract_features(img):
    img = cv2.resize(img, (128,128))
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hu = cv2.HuMoments(cv2.moments(thresh)).flatten()
    hog_features = hog(img, pixels_per_cell=(16,16), cells_per_block=(2,2),
                       orientations=9, block_norm='L2-Hys')
    return np.hstack([hu, hog_features])

# =========================
# --- App Layout ---
# =========================
st.markdown("<h1>🍃 Leaf Classification App 🌿</h1>", unsafe_allow_html=True)
st.write("<p>Upload a leaf image and let AI predict its species with style!</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1,1])

with col1:
    uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.image(img, caption="🌱 Uploaded Leaf", use_container_width=True)


with col2:
    if uploaded_file is not None:
        features = extract_features(img).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        probs = clf.decision_function(features_pca)
        top3_idx = np.argsort(probs[0])[-3:][::-1]
        top3_labels = clf.classes_[top3_idx]
        top3_scores = probs[0][top3_idx]

        st.subheader("🔮 Prediction Results")
        for label, score in zip(top3_labels, top3_scores):
            st.markdown(
                f"""
                <div style="background: linear-gradient(90deg,#006400,#228B22); 
                            padding: 12px; border-radius: 12px; margin-bottom: 10px; 
                            color: white; font-size: 18px; text-align: center;">
                    🌿 <b>{label}</b> — {score*100:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )
