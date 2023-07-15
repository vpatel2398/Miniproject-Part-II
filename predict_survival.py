import pandas as pd
import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import os
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image




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


model_filename = "island_survival.h5"

def train_model(X_train,y_train):
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(7,)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=25, batch_size=1,validation_data=(X_val, y_val))
    
    return model

 # Check if the file already exists
if os.path.exists(model_filename):
    # Load the existing model
    model = load_model(model_filename)
    print("Existing model loaded.")
else:
    # Train and save the model
    model = train_model(X_train, y_train)
    model.save(model_filename)
    print("New model trained and saved.")

def survival():

    st.image(img,width=300, channels='RGB',caption=None)
    st.title('PREDICT YOUR SURVIVAL ON THE ISLAND')
    sex = st.selectbox('Sex', ('Male','Female'))
    age = st.slider('Age',0,80)
    Health_condition = st.slider('Health Condition',1,3)
    Shelter_Built = st.selectbox('Is Shelter Available?',[0,1])
    Food_Available = st.selectbox('Food_Available',[0,1])
    Water_Available = st.selectbox('Water_Available',[0,1])
    Days_Survived = st.slider('Days_Survived',5,10)

    
    if sex == 'Male':
        sex = 1
    else:
        sex = 0

    data = [[Health_condition, sex, age, Shelter_Built,Food_Available,Water_Available,Days_Survived]]
    data = tf.constant(data)

    if st.button('Predict'):
        prediction = model.predict(data, steps=1)
        pred = [round(x[0]) for x in prediction]

        if pred ==[0]:
            st.write('You did not survive')
        else:
            st.write('You survived')


if __name__ == '__main__':
    survival()
