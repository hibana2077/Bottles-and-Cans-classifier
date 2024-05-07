import streamlit as st
import cv2
import numpy as np
import torch
from torchvision import transforms
import pandas as pd
import os
from reg import RegNetY32GF
import time

@st.cache_data
def load_model():
    model = RegNetY32GF()
    device = torch.device('cpu')
    model.load_state_dict(torch.load('../train/model_state_dict_2.pth', map_location=device))
    model.eval()
    return model
model = load_model()

def predict(model, image_array):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # Ensure input size is consistent
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_array).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

csv_file = 'records.csv'
labels = {0: 'Can', 1: 'Bottle'}
category_count = {label: 0 for label in labels.values()}

def main():
    st.title("Bottles and Cans Classifier")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    last_time = time.time()
    data_frame = pd.DataFrame()
    data_display = st.empty()
    chart_placeholder = st.empty()

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
        if ret:
            frame = cv2.resize(frame, (224, 224))  # Resize the image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
            
            current_time = time.time()
            if current_time - last_time >= 8:
                last_time = current_time
                result = predict(model, frame_rgb)
                
                if result in labels:
                    label = labels[result]
                    new_data = pd.DataFrame({
                        'Prediction': [label],
                        'Timestamp': [pd.Timestamp.now()]
                    })
                    data_frame = update_csv(new_data)
                    category_count[label] += 1
                    st.write('Prediction: ', label)
                else:
                    st.write('No target detected.')
            
            data_display.dataframe(data_frame)
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                plot_data = df['Prediction'].value_counts().reindex(labels.values(), fill_value=0)
                chart_placeholder.bar_chart(plot_data)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    st.write('Stopped')
    cap.release()

if __name__ == '__main__':
    main()
