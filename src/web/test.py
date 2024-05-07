import streamlit as st
import cv2
import numpy as np
import torch
from torchvision import transforms
import sqlite3
import pandas as pd
import plotly.express as px
from reg import RegNetY32GF
import time

# 数据库初始化
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            prediction TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# 记录预测结果到数据库
def log_prediction(prediction):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('INSERT INTO predictions (prediction) VALUES (?)', (prediction,))
    conn.commit()
    conn.close()

# 从数据库加载数据
def load_data():
    conn = sqlite3.connect('predictions.db')
    df = pd.read_sql('SELECT * FROM predictions', conn, parse_dates=['timestamp'])
    conn.close()
    return df

# 模型加载
@st.cache_data
def load_model():
    model = RegNetY32GF()
    device = torch.device('cpu')
    model.load_state_dict(torch.load('model_state_dict_2.pth', map_location=device))
    model.eval()
    return model

# 图像预处理并进行预测
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

# 主函数
def main():
    st.title("Bottles and Cans Classifier with Real-time Database Visualization")
    model = load_model()
    labels = {0: 'Bottle', 1: 'Can'}
    init_db()

    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    last_time = time.time()

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb)

                if time.time() - last_time >= 8:
                    last_time = time.time()
                    frame_resized = cv2.resize(frame_rgb, (224, 224))
                    result = predict(model, frame_resized)
                    if result in labels:
                        label = labels[result]
                        log_prediction(label)
                        st.write('Prediction: ', label)

                    # 加载数据并绘制柱状图
                    df = load_data()
                    value_counts = df['prediction'].value_counts().reset_index()
                    value_counts.columns = ['Prediction', 'Count']  # 显式设置列名为 'Prediction' 和 'Count'

                    # 使用更新后的 DataFrame 绘制柱状图
                    fig = px.bar(value_counts, x='Prediction', y='Count', labels={'Prediction': 'Prediction', 'Count': 'Count'})
                    st.plotly_chart(fig)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

if __name__ == '__main__':
    main()
