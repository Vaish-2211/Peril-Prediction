import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import re
import json
import pickle
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv(r"D:\peril prediction\animal.csv")


def predict_population(animal, year):
    bd = df.loc[df['Animal'] == animal]
    bd.drop(['Animal', 'Scientific Name', 'Class'], axis=1, inplace=True)
    bd = bd.T
    bd.dropna(inplace=True)
    bd = bd.reset_index().rename(columns={2: 'population', 'index': 'year'})
    x = bd.iloc[:, 0].values.reshape(-1, 1)
    y = bd.iloc[:, 1].values.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    y_pred = model.predict([[year]])
    return int(y_pred)


def main():
    st.title("PERIL PREDICTION");
    animal = st.selectbox("Animal", (
    "Select Animal", "Red Panda", "Mountain Gorilla", "Irrwwaddy dolphin", "Asian Elephant", "Giant Panda",
    "Sunda Tiger", "Leopard", "African Wild dogs", "Bonobo", "Javan Rhino", "Green Turtle", "Asiatic Lion", "Sea lion",
    "Black Rhino", "Vaquita", "Horn Shark",
    "Gaur " , "Philippine Pangolin", "Sloth Bear", "White Steenbras", "Bog Turtle", "Grey Wolf", "Arabian oryx",
    "Black Footed Ferret", "Red wolf", "Cross River Gorilla", "Hawalian Crow",
    "Malyan Tiger", "Snow leopard", "Lar Gibbon"))

    year = st.text_input("year", "Type here")

    count_html = """
    <div padding:10px>
    <h3 style:"color:green;text-align:left;>Count"""

    safe_html = """ 
    <div padding:10px>
    <h2 style="color:blue;text-align:center;">VULNERABLE SPECIES</h2>
    </div>
    """
    danger_html = """
    <div padding:10px>
    <h2 style="color:Red;text-align:center;">ENDANGERED SPECIES</h2>
    """
    if st.button("Predict"):
        output = predict_population(animal, year)
        str = "{} population in {} will be {}"
        st.success(str.format(animal, year, output))

        if output < 5000:
            st.markdown(danger_html, unsafe_allow_html=True)
        else:
            st.markdown(safe_html, unsafe_allow_html=True)


if __name__ == '__main__':
    main()

