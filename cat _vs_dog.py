import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, GlobalMaxPool2D, Dense
from tensorflow.keras.models import Model
from PIL import Image
import requests

CONFIG = {
    "BATCH_SIZE": 16,
    "IMG_SIZE": (300, 300),
    "STRIDE_1": 1,
    "STRIDE_2": 2,
    "REGUL": 0.001,
    "DROP_OUT": 0.2,
    "DENSE_L1": 1024,
    "DENSE_L2": 512,
    "DENSE_L3": 256,
    "DENSE_L4": 128,
    "KERNEL_SIZE_3": (3, 3),
    "KERNEL_SIZE_2": (5, 5),
    "KERNEL_SIZE_1": (7, 7),
    "LR": 0.001,
    "POOL_SIZE": (2, 2),
    "train_path": "/kaggle/input/cat-and-dog/training_set/training_set",
    "FILTERS": 32
}

# Load the model weights
github_raw_url = "https://github.com/JacksonDivakarProjects/Cat-and-Dog-Classification/raw/refs/heads/main/Final%20Model.h5"

model_temp = None
try:
    response = requests.get(github_raw_url)
    if response.status_code == 200:
        with open("model.h5", "wb") as f:
            f.write(response.content)
        model_temp = tf.keras.models.load_model("model.h5")
    else:
        st.error(f"Failed to retrieve model. Status code: {response.status_code}")
except Exception as e:
    st.error(f"Error loading model: {e}")

def get_model(input_shape=(300, 300, 3), no_of_labels=1):
    tf.keras.backend.clear_session()
    i = Input(shape=input_shape)
    # First
    x = Conv2D(filters=32, kernel_size=3, activation='relu', strides=CONFIG['STRIDE_1'], padding='valid')(i)
    x = MaxPool2D(pool_size=CONFIG['POOL_SIZE'], strides=CONFIG['STRIDE_2'])(x)

    # Second
    x = Conv2D(filters=32, kernel_size=3, activation='relu', strides=CONFIG['STRIDE_1'], padding='valid')(x)
    x = MaxPool2D(pool_size=CONFIG['POOL_SIZE'], strides=CONFIG['STRIDE_2'])(x)

    x = GlobalMaxPool2D()(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dense(no_of_labels, activation='sigmoid', name='last_layer')(x)

    return Model(i, x)

def preprocess(image):
    image = image.resize((300, 300))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

st.title('Dog Vs Cat Classification')
model = get_model()
if model_temp:
    model.set_weights(model_temp.get_weights())

def final(output):
    return 'Dog' if output else 'Cat'

file_upload = st.file_uploader("Choose the file to upload ...", type=['jpg', 'jpeg', 'png'])
if file_upload is not None:
    image = Image.open(file_upload)
    preprocessed = preprocess(image)
    st.image(image=image, caption='Uploaded Image', use_column_width=True)
    output = np.round(model.predict(preprocessed)[0])
    st.write('Classifying..')
    st.write("The Predicted output:", final(output[0]))
