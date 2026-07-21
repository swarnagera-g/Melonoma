import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tempfile

from gradcam import make_gradcam_heatmap
from gradcam import overlay_heatmap

#############################################

st.set_page_config(
    page_title="Melanoma Detection",
    layout="centered"
)

st.title("🩺 Melanoma Detection")

st.write(
"""
Upload a dermoscopic image to evaluate the trained research model.
"""
)

#############################################

MODEL_PATH = "best_model.keras"

LAST_CONV_LAYER = "top_conv"

model = tf.keras.models.load_model(MODEL_PATH)

#############################################

uploaded_file = st.file_uploader(

    "Upload Dermoscopic Image",

    type=["jpg","jpeg","png"]

)

#############################################

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    #########################################

    img = np.array(image)

    img224 = cv2.resize(img,(224,224))

    x = img224.astype(np.float32)/255.0

    x = np.expand_dims(x,0)

    #########################################

    probability = float(model.predict(x, verbose=0)[0][0])

    prediction = "Melanoma" if probability >= 0.5 else "Non-Melanoma"

    confidence = probability if prediction=="Melanoma" else 1-probability

    #########################################

    st.subheader("Prediction")

    st.success(prediction)

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

    #########################################

    heatmap = make_gradcam_heatmap(

        x,

        model,

        LAST_CONV_LAYER

    )

    heatmap_img, overlay = overlay_heatmap(

        img,

        heatmap

    )

    #########################################

    st.subheader("Grad-CAM")

    col1,col2 = st.columns(2)

    with col1:

        st.image(
            img,
            caption="Original Image",
            use_container_width=True
        )

    with col2:

        st.image(
            overlay,
            caption="Grad-CAM Overlay",
            use_container_width=True
        )

    #########################################

    st.subheader("AI Model Explanation")

    st.info(

        """
        The highlighted regions contributed most strongly
        to the model prediction.

        Grad-CAM provides a visual explanation of the areas
        influencing the model's decision but does not verify
        that the highlighted regions are medically correct.
        """

    )

#############################################

st.warning(
"""
⚠ **Research Prototype**

This application is developed solely for academic research and educational purposes.

It is **not** a medical diagnostic tool and must **not** be used for clinical decision-making, diagnosis, or treatment.
"""
)
