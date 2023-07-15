import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import os
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import pickle



img = Image.open('ship.jpg')

df = pd.read_csv('island_data.csv')

#encode the sec column
df.loc[df['Sex']=='male', 'Sex'] = 1
df.loc[df['Sex']=='female', 'Sex'] = 0