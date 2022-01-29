import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as datas
from tensorflow.keras.models import load_model
import streamlit as st
##
from sklearn import metrics
import plotly.express as px
##

def app():
    st.title('Model - LSTM')

    #start = '2004-08-18'
    #end = '2022-01-20'
    start = st.date_input('Start' , value=pd.to_datetime('2004-08-18'))
    end = st.date_input('End' , value=pd.to_datetime('today'))

    st.title('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil' , 'GOOG')

    df = datas.DataReader(user_input, 'yahoo', start, end)

    # Describiendo los datos

    st.subheader('Datos del 2004 al 2022') 
    st.write(df.describe())

    #Visualizaciones 

  
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    

    st.subheader('Closing Price vs Time chart con 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close)
    plt.plot(ma100,'r', label='ma100')
    plt.legend()
    st.pyplot(fig)







    st.subheader('Closing Price vs Time chart con 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig5 = plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    plt.plot(ma100,'r', label='ma100')
    plt.plot(ma200,'g', label='ma200')
    plt.legend()
    st.pyplot(fig5)




    # Splitting data into training and testing 

    # Cree un nuevo marco de datos con solo la columna 'Close'
    data = df.filter(['Close'])

    # Convierte el marco de datos en una matriz numpy
    dataset = data.values

    # Obtenga el número de filas para entrenar el modelo
    training_data_len = int(np.ceil( len(dataset) * .95 ))

    # Escalando la data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)


    # Cargar mi modelo

    model = load_model('keras_model.h5')


    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002 
    # Crear el conjunto de datos de prueba
    # Crear una nueva matriz que contenga valores escalados del índice 1543 al 2002
    test_data = scaled_data[training_data_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
            
    # Convert the data to a numpy array
    # Convierte los datos en una matriz numpy
    x_test = np.array(x_test)

    # Reshape the data
    # Reforma los datos
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Get the models predicted price values 
    # Obtenga los valores de precios predichos de los modelos
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    # Obtener la raíz del error cuadrático medio (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    rmse

    # Graficos Finales
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Visualizando la data
    st.subheader('Comparacion de entrenamiento y validacion ')
    fig2=plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Tiempo', fontsize=18)
    plt.ylabel('Precio dolar ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
    st.pyplot(fig2)
    
    # Visualizando la data 2
    st.subheader('Comparacion de entrenamiento, validacion y prediccion')
    fig3=plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Tiempo', fontsize=18)
    plt.ylabel('Precio dolar ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
    st.pyplot(fig3)

    # Visualizando la data 3
    st.subheader('Precio predecido vs Precio Original')
    fig4=plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label = 'Precio Original')
    plt.plot(predictions, 'r', label= 'Precio Predecido')
    plt.xlabel('Tiempo')
    plt.ylabel('Precio')
    plt.legend()
    st.pyplot(fig4)
    
    # Visualizando datos
    st.subheader('Mostrar los datos originales y predecidos') 
    st.write(valid)



##########PLANTILLA####################
    # Evaluación del modelo
    
    st.title('Evaluación del Modelo LSTM')
    ## Métricas
    MAE=metrics.mean_absolute_error(y_test, predictions)
    MSE=metrics.mean_squared_error(y_test, predictions)
    RMSE=np.sqrt(metrics.mean_squared_error(y_test, predictions))
    
    metricas = {
        'metrica' : ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'],
        'valor': [MAE, MSE, RMSE]
    }
    metricas = pd.DataFrame(metricas)  
    ### Gráfica de las métricas
    st.subheader('Métricas de rendimiento') 
    fig = px.bar(        
        metricas,
        x = "metrica",
        y = "valor",
        title = "Métricas del Modelo LSTM",
        color="metrica"
    )
    st.plotly_chart(fig)



   









