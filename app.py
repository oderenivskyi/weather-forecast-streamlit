import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Прогноз дощу в Австралії", page_icon="Pf", layout="wide")

# Завантаження моделі

@st.cache_resource
def load_model():
    model = joblib.load('model/rain_prediction.joblib')
    return model

pipeline = load_model()

# Інтерфейс

st.title("Прогноз дощу в Австралії")
st.markdown("""
Цей додаток використовує модель машинного навчання для прогнозування, чи піде дощ завтра використовуючи метеорологічні дані сьогодні.
""")

locations = ['Adelaide', 'Albany', 'Albury', 'AliceSprings', 'BadgerysCreek', 'Ballarat', 'Bendigo', 'Brisbane', 'Cairns', 'Canberra', 'Cobar', 'CoffsHarbour', 'Dartmoor', 'Darwin', 'GoldCoast', 'Hobart', 'Katherine', 'Launceston', 'Melbourne', 'MelbourneAirport', 'Mildura', 'Moree', 'MountGambier', 'MountGinini', 'Newcastle', 'Nhil', 'NorahHead', 'NorfolkIsland', 'Nuriootpa', 'PearceRAAF', 'Penrith', 'Perth', 'PerthAirport', 'Portland', 'Richmond', 'Sale', 'SalmonGums', 'Sydney', 'SydneyAirport', 'Townsville', 'Tuggeranong', 'Uluru', 'WaggaWagga', 'Walpole', 'Watsonia', 'Williamtown', 'Witchcliffe', 'Wollongong', 'Woomera']
wind_directions = ['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW', 'SW', 'W', 'WNW', 'WSW']

st.header("Введіть вхідні дані")

st.subheader("Локація та опади")
c1_1, c1_2, c1_3, c1_4, c1_5 = st.columns(5)

with c1_1:
    location = st.selectbox("Location", locations)
with c1_2:
    rain_today = st.selectbox("Rain Today", ["No", "Yes"])
with c1_3:
    rainfall = st.number_input("Rainfall", min_value=0.0, value=0.0, step=0.1)
with c1_4:
    evaporation = st.number_input("Evaporation", min_value=0.0, value=5.0, step=0.1)
with c1_5:
    sunshine = st.number_input("Sunshine", min_value=0.0, max_value=24.0, value=8.0, step=0.1)

st.markdown("---")

st.subheader("Температура")
c2_1, c2_2, c2_3, c2_4 = st.columns(4)

with c2_1:
    min_temp = st.number_input("Min Temp (°C)", value=15.0, step=0.1)
with c2_2:
    max_temp = st.number_input("Max Temp (°C)", value=25.0, step=0.1)
with c2_3:
    temp_9am = st.number_input("Temp 9am (°C)", value=18.0, step=0.1)
with c2_4:
    temp_3pm = st.number_input("Temp 3pm (°C)", value=24.0, step=0.1)
    
st.markdown("---")

st.subheader("Вітер")
c3_1, c3_2, c3_3, c3_4, c3_5, c3_6 = st.columns(6)

with c3_1:
    wind_gust_dir = st.selectbox("Gust Dir", wind_directions)
with c3_2:
    wind_gust_speed = st.number_input("Gust Speed", min_value=0, value=40, step=1)
with c3_3:
    wind_dir_9am = st.selectbox("Dir 9am", wind_directions)
with c3_4:
    wind_speed_9am = st.number_input("Speed 9am", min_value=0, value=15, step=1)
with c3_5:
    wind_dir_3pm = st.selectbox("Dir 3pm", wind_directions)
with c3_6:
    wind_speed_3pm = st.number_input("Speed 3pm", min_value=0, value=20, step=1)

st.markdown("---")

st.subheader("Вологість та тиск")
c4_1, c4_2, c4_3, c4_4 = st.columns(4)

with c4_1:
    humidity_9am = st.slider("Humidity 9am", 0, 100, 70)
with c4_2:
    humidity_3pm = st.slider("Humidity 3pm", 0, 100, 50)
with c4_3:
    pressure_9am = st.number_input("Pressure 9am", value=1015.0, step=0.1)
with c4_4:
    pressure_3pm = st.number_input("Pressure 3pm", value=1012.0, step=0.1)
    
st.subheader("Хмарність")

c5_1, c5_2 = st.columns([1, 1])

with c5_1:
    cloud_9am = st.slider("Cloud 9am", 0, 8, 4)
with c5_2:
    cloud_3pm = st.slider("Cloud 3pm", 0, 8, 4)

st.markdown("---")

submit_button = st.button('Отримати прогноз', type="primary", use_container_width=True)

# Обробка та Прогнозування

if submit_button:
    input_data = pd.DataFrame({
        'Location': [location],
        'MinTemp': [min_temp],
        'MaxTemp': [max_temp],
        'Rainfall': [rainfall],
        'Evaporation': [evaporation],
        'Sunshine': [sunshine],
        'WindGustDir': [wind_gust_dir],
        'WindGustSpeed': [wind_gust_speed],
        'WindDir9am': [wind_dir_9am],
        'WindDir3pm': [wind_dir_3pm],
        'WindSpeed9am': [wind_speed_9am],
        'WindSpeed3pm': [wind_speed_3pm],
        'Humidity9am': [humidity_9am],
        'Humidity3pm': [humidity_3pm],
        'Pressure9am': [pressure_9am],
        'Pressure3pm': [pressure_3pm],
        'Cloud9am': [cloud_9am],
        'Cloud3pm': [cloud_3pm],
        'Temp9am': [temp_9am],
        'Temp3pm': [temp_3pm],
        'RainToday': [rain_today]
    })

    st.subheader("Результат прогнозу:")

    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0]

    col_res1, col_res2 = st.columns(2)

    with col_res1:
        if prediction == 'Yes':
            st.error(f"Чи піде дощ завтра? **ТАК**")
        else:
            st.success(f"Чи піде дощ завтра? **НІ**")

    with col_res2:
        st.info(f"Ймовірність дощу: **{probability[1]*100:.2f}%**")
        st.progress(int(probability[1]*100))
            
  