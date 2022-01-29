import yfinance as yf
import talib
import plotly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
%matplotlib inline
import streamlit as st


def app():
    st.title('Model - Random Forest')


    ticker= "googl"
    stock_data = yf.download(ticker, start="2008-01-04", end="2022-01-27")


    fig = plt.figure(figsize = (12,6))
    stock_data['Adj Close'].plot()
    plt.ylabel("Precio de cierre ajustado")
    st.pyplot(fig)
    
    fig2 = plt.figure(figsize = (12,6))
    stock_data['Adj Close'].pct_change().plot.hist(bins=50)
    plt.xlabel("Cambio porcentual de cierre ajustado de 1 día")
    st.pyplot(fig2) 
    
    feature_names = []
    for n in [14, 30, 50, 200]:
        stock_data['ma' + str(n)] = talib.SMA(stock_data['Adj Close'].values, timeperiod=n)
        stock_data['rsi' + str(n)] = talib.RSI(stock_data['Adj Close'].values, timeperiod=n)

        feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]

    stock_data['Volume_1d_change'] = stock_data['Volume'].pct_change()

    volume_features = ['Volume_1d_change']
    feature_names.extend(volume_features)
    
    stock_data['5d_future_close'] = stock_data['Adj Close'].shift(-5)
    stock_data['5d_close_future_pct'] = stock_data['5d_future_close'].pct_change(5)
    
    # Visualizando datos
    st.subheader('Mostrar los datos') 
    st.write(stock_data.head())

    stock_data.dropna(inplace=True)

    X = stock_data[feature_names]
    y = stock_data['5d_close_future_pct']

    train_size = int(0.85 * y.shape[0])
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    grid = {'n_estimators': [200], 'max_depth': [3], 'max_features': [4, 8], 'random_state': [42]}
    test_scores = []

    rf_model = RandomForestRegressor()

    for g in ParameterGrid(grid):
        rf_model.set_params(**g) 
        rf_model.fit(X_train, y_train)
        test_scores.append(rf_model.score(X_test, y_test))

    best_index = np.argmax(test_scores)
    
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=3, max_features=4, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)


    y_pred_series = pd.Series(y_pred, index=y_test.index)   


    st.subheader('Grafica de cambio de precio')
    fig3 = plt.figure(figsize = (12,6))
    y_pred_series.plot()
    plt.ylabel("Porcentaje de cambio de precio de cierre previsto de 5 días")
    st.pyplot(fig3)


    from sklearn import metrics
    MAE=metrics.mean_absolute_error(y_test, y_pred)
    MSE=metrics.mean_squared_error(y_test, y_pred)
    RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    #print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    #print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    #print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


    metrics = {
    'metrics' : ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'],
    'valor': [MAE, MSE, RMSE]
    }
    metrics = pd.DataFrame(metrics)

    import plotly.express as px

    fig = px.bar(metrics, x="metrics", y="valor", color="metrics",text_auto=True,title="Métricas del modelo Random Forest ")

    
    importances = rf_model.feature_importances_
    sorted_index = np.argsort(importances)[::-1]
    x_values = range(len(importances))
    labels = np.array(feature_names)[sorted_index]
    
    st.subheader('Grafica ma')
    fig5 = plt.figure(figsize = (12,6))
    plt.bar(x_values, importances[sorted_index], tick_label=labels)
    plt.xticks(rotation=90)
    plt.legend()
    st.pyplot(fig)
    
