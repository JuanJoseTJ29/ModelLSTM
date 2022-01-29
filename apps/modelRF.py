import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow.keras.models import load_model
import streamlit as st
import talib
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, plot_roc_curve
from sklearn.metrics import roc_curve, auc
import plotly.express as px


def app():
    st.title('Model - Random Forest')

    start = st.date_input('Start' , value=pd.to_datetime('2004-08-18'))
    end = st.date_input('End' , value=pd.to_datetime('today'))


    st.title('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil', 'GOOG')

    stock_data = data.DataReader(user_input, 'yahoo', start, end)

    # Describiendo los datos

    st.subheader('Datos del 2004 al 2022')
    st.write(stock_data.describe())

    # Visualizaciones
    st.subheader('Precio de cierre ajustado')
    fig = px.line(stock_data,y='Adj Close')
    st.plotly_chart(fig)

    st.subheader('Cambio porcentual de cierre ajustado de 1 día')
    fig = plt.figure(figsize=(12, 6))
    plt.hist(stock_data['Adj Close'].pct_change(), bins=50)
    plt.ylabel("Frecuencia")
    plt.xlabel("Cambio porcentual de cierre ajustado de 1 día")
    st.pyplot(fig)

    feature_names = []
    for n in [14, 30, 50, 200]:
        stock_data['ma' +
                   str(n)] = talib.SMA(stock_data['Adj Close'].values, timeperiod=n)
        stock_data['rsi' +
                   str(n)] = talib.RSI(stock_data['Adj Close'].values, timeperiod=n)

        feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]

    stock_data['Volume_1d_change'] = stock_data['Volume'].pct_change()

    volume_features = ['Volume_1d_change']
    feature_names.extend(volume_features)

    stock_data['5d_future_close'] = stock_data['Adj Close'].shift(-5)

    stock_data['5d_close_future_pct'] = stock_data['5d_future_close'].pct_change(
        5)
    stock_data.dropna(inplace=True)

    X = stock_data[feature_names]
    y = stock_data['5d_close_future_pct']

    train_size = int(0.85 * y.shape[0])
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    grid = {'n_estimators': [200], 'max_depth': [3],
            'max_features': [4, 8], 'random_state': [42]}

    test_scores = []

    rf_model = RandomForestRegressor()

    for g in ParameterGrid(grid):
        rf_model.set_params(**g)
        rf_model.fit(X_train, y_train)
        test_scores.append(rf_model.score(X_test, y_test))

    best_index = np.argmax(test_scores)

    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=3, max_features=4, random_state=42)

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    st.subheader('Porcentaje de cambio de precio de cierre previsto de 5 días')
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    fig = px.line(y_pred_series)
    st.plotly_chart(fig)
    
    
    # Evaluación del modelo
    
    st.title('Evaluación del Modelo RFR')
    ## Métricas
    MAE=metrics.mean_absolute_error(y_test, y_pred)
    MSE=metrics.mean_squared_error(y_test, y_pred)
    RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
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
        title = "Métricas del Modelo Random Forest Regressor",
        color="metrica"
    )
    st.plotly_chart(fig)
    
    
    ## Curva ROC
    
    #ax = plt.gca()
    #rfc_disp = plot_roc_curve(rf_model, X_test, y_test, ax=ax, alpha=0.8)
    #plt.show()
    #st.pyplot(fig)
    
    