import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import os
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image




def main():
    img = Image.open('ship.jpg')

    df = pd.read_csv('island_data.csv')

    #encode the sec column
    df.loc[df['Sex']=='male', 'Sex'] = 1
    df.loc[df['Sex']=='female', 'Sex'] = 0

    #split the data in to independent x and y variables
    X = df.drop('Survived', axis =1)
    y=df['Survived'].values.astype(np.float32)
    X = X.values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=1)

if __name__ == '__main__':
    main()