'''
Author: hibana2077 hibana2077@gmail.com
Date: 2024-05-05 11:49:31
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2024-05-14 15:05:52
FilePath: \srcc:\Users\USER\Bottles-and-Cans-classifier\src\web\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import streamlit as st
from PIL import Image
import requests
import numpy as np
import torch
import torchvision
import numpy as np
from reg import EfficientNetV2S # model need to be defined in reg.py

THINGSPEAK_URL = 'https://api.thingspeak.com/'
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

    ## Send data to ThingSpeak
    ### TBW