import io
import streamlit as st
import tensorflow as tf
from PIL import Image
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForImageClassification


def load_image():
    uploadedFile = st.file_uploader('Upload image here')

    if uploadedFile is not None:
        st.write("Filename:", uploadedFile.name)
        image_data = uploadedFile.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None
    
def preprocess_image(img):
    img = img.resize((100, 100))
    x = tf.keras.utils.img_to_array(img)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    return x

@st.cache
def load_model():
    processor = AutoImageProcessor.from_pretrained("JuanMa360/room-classification")
    model = AutoModelForImageClassification.from_pretrained("JuanMa360/room-classification")
    return pipeline("image-classification", model=model, image_processor=processor)

# Project Title
st.title("Room Classification Project")

# Team Member
st.write("""
         House & Apartaments Classification model
         #### TEAM MEMBER
         - Рахарди Сандикха РИМ-130908
         - Мухин Виктор Александрович РИМ-130908
         - Шлёгин Лев Русланович РИМ-130908
         - Сидоркин Георгий Владимирович РИМ-130908
         """)

st.write("""#### Our Project""")

# Initial function
loadedImage = load_image()
model = load_model()

result = st.button('Submit')

if result:
    x = preprocess_image(loadedImage)
    prediction = model.predict(loadedImage)
    st.write("""#### Output""")
    st.write(prediction)