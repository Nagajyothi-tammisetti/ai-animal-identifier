
import streamlit as st
from transformers import pipeline
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="AI Animal Identifier",
    page_icon="🐾",
    layout="centered"
)

st.title("🐾 AI Animal Identification")
st.write("Upload an animal image and let AI identify it!")

# Load the image classification model
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )

# Load model
with st.spinner("Loading AI model..."):
    classifier = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Choose an animal image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Identify Animal"):
        with st.spinner("Analyzing image..."):
            predictions = classifier(image, top_k=5)

        st.subheader("AI Predictions")

        for prediction in predictions:
            label = prediction["label"]
            score = prediction["score"]

            st.write(f"**{label.replace('_', ' ').title()}**")
            st.progress(float(score))
            st.write(f"Confidence: {score * 100:.2f}%")

        best_prediction = predictions[0]

        st.success(
            f"Top prediction: "
            f"{best_prediction['label'].replace('_', ' ').title()}"
        )
else:
    st.info("Upload an image to get started.")
