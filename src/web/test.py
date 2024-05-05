'''
Author: hibana2077 hibana2077@gmail.com
Date: 2024-05-05 12:05:02
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2024-05-05 18:40:53
FilePath: \Bottles-and-Cans-classifier\src\web\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import streamlit as st
import cv2
import numpy as np
import torch
from torchvision import transforms
import pandas as pd
import os
from reg import RegNetY32GF
import time

def load_model():
    model = RegNetY32GF()
    device = torch.device('cpu')
    model.load_state_dict(torch.load('../train/model_state_dict_2.pth', map_location=device))
    model.eval()
    return model

def predict(model, image_array):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_array).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

model = load_model()
csv_file = 'records.csv'
labels = {0: 'Bottle', 1: 'Can'}
category_count = {label: 0 for label in labels.values()}  # Initialize category count

def main():
    st.title("Bottles and Cans Classifier")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    last_time = time.time()
    data_frame = pd.DataFrame()
    data_display = st.empty()  # Use an empty placeholder
    count_display = st.empty()  # Placeholder for category count

    def update_csv(new_data):
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            df = new_data
        df.to_csv(csv_file, index=False)
        return df

    while run:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        
        current_time = time.time()
        if current_time - last_time >= 8:  # Check if 8 seconds have passed
            last_time = current_time
            result = predict(model, frame)
            
            if result in labels:
                label = labels[result]
                new_data = pd.DataFrame({
                    'Prediction': [label],
                    'Timestamp': [pd.Timestamp.now()]
                })
                data_frame = update_csv(new_data)
                category_count[label] += 1  # Increment category count
                st.write('Prediction: ', label)
            else:
                st.write('No target detected.')
        
        data_display.dataframe(data_frame)  # Update the display
        count_display.json(category_count)  # Update the category count display

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    st.write('Stopped')
    cap.release()

if __name__ == '__main__':
    main()
