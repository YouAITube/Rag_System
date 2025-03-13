

import pandas as pd
import streamlit as st
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('Data/cars.jpg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Demo Cars Moldova",
        page_icon=image,

    )

    st.write(
        """
        # Classification type of car.
        """
    )

    st.image(image)


def write_user_data(df):
    st.write("## Your data")
    st.write(df)


def write_prediction(prediction, prediction_probas):
    st.write("## Prediction")
    st.write(prediction)

    st.write("## Probability of prediction")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('User-defined parameters')
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():
    make = st.sidebar.selectbox("Company", ('Toyota', 'Renault', 'Opel', 'Mercedes', 'Volkswagen', 'BMW', 'Volvo', 'Nissan', 'Hyundai',
                                           'Audi', 'GAZ', 'Rare', 'Lincoln', 'Lexus', 'Dodge', 'Porsche', 'Dacia', 'Peugeot', 'Ford',
                                           'Honda', 'Skoda', 'Mazda', 'Chevrolet', 'Citroen', 'Jaguar', 'Infiniti', 'Land Rover',
                                           'KIA', 'Seat', 'Mitsubishi', 'Fiat', 'Suzuki', 'Daewoo', 'Subaru', 'Mini',
                                           'Rover', 'Chrysler', 'Jeep', 'Smart', 'Alfa Romeo', 'Daihatsu'))

    fuel = st.sidebar.selectbox("Fuel_type", ("Diesel", "Hybrid",'Petrol','Electric'))

    engine = st.sidebar.slider("Engine_capacity(cm3)", min_value=200, max_value=5000, value=200,
                            step=1)

    age = st.sidebar.slider(
        "Age",
        min_value=1, max_value=51, value=1, step=1)

    price = st.sidebar.slider(
        "Price",
        min_value=200, max_value=100000, value=200, step=1)

    class_ = st.sidebar.slider("class",
                               min_value=1, max_value=5, value=1, step=1)


    data = {
        "Make": make,
        "Engine_capacity(cm3)": engine,
        "class": class_,
        "Age": age,
        "Fuel_type": fuel,
        "Price(euro)": price,
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()
