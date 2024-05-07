import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision
import numpy as np
from reg import EfficientNetV2S # model need to be defined in reg.py

LABELS = ['Bottle', 'Can']

@st.cache_data
def load_model():
    model = torch.load('model.pt')
    model.eval()
    return model

img_file_buffer = st.camera_input("Take a picture")
model = load_model()

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # To Tensor
    transform = torchvision.transforms.Compose([
        # resize to 224x224
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    img_tensor = transform(img).unsqueeze(0)

    # To predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    st.write(LABELS[predicted.item()])