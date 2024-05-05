import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import pandas as pd
import os
from reg import RegNetY32GF 

def load_model():
    # 確保模型加載到 CPU
    model = RegNetY32GF()
    device = torch.device('cpu')
    model.load_state_dict(torch.load('../train/model_state_dict.pth', map_location=device))
    model.eval()
    return model

# 預測函數
def predict(model, image_array):
    # 將numpy圖像轉換為PyTorch張量
    transform = transforms.Compose([
        transforms.ToTensor(),  # 轉換為Tensor
        transforms.Resize((224, 224)),  # 調整圖像大小以適應模型
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化
    ])
    image_tensor = transform(image_array).unsqueeze(0)  # 增加批次維度
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# 加載模型
model = load_model()

# CSV檔案設定
csv_file = 'records.csv'

# 使用 Streamlit 的相機輸入
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # 讀取圖像文件緩衝區為PIL圖像
    img = Image.open(img_file_buffer).convert('RGB')

    # 將PIL圖像轉換為numpy數組
    img_array = np.array(img)

    # 使用模型進行預測
    result = predict(model, img_array)

    # 更新CSV文件
    labels = {0: 'Bottle', 1: 'Can'}
    new_data = pd.DataFrame({
        'Prediction': [labels[result]],
        'Timestamp': [pd.Timestamp.now()]
    })

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data

    df.to_csv(csv_file, index=False)

    st.write('Prediction: ', labels[result])
    st.dataframe(df)
    st.write('Data type of image array:', type(img_array))
    st.write('Shape of image array:', img_array.shape)