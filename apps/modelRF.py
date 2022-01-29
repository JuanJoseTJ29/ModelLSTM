from tensorflow.keras.models import load_model
import streamlit as st
#import talib
from sklearn import metrics
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, plot_roc_curve
from sklearn.metrics import roc_curve, auc
import plotly.express as px


def app():
    st.title('Model - Random Forest')

    #start = st.date_input('Start' , value=pd.to_datetime('2004-08-18'))
    #end = st.date_input('End' , value=pd.to_datetime('today'))



    ticker= "googl"
    stock_data = yf.download(ticker, start="2008-01-04", end="2022-01-27")


    fig = plt.figure(figsize = (12,6))
    stock_data['Adj Close'].plot()
    plt.ylabel("Precio de cierre ajustado")
    st.pyplot(fig)

    st.title('Predicción de tendencia de acciones')

	

    # Visualizaciones
    st.subheader('Precio de cierre ajustado')
    fig = px.line(stock_data,y='Adj Close')
    st.plotly_chart(fig)

    st.subheader('Cambio porcentual de cierre ajustado de 1 día')
    fig = plt.figure(figsize=(12, 6))
	
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
