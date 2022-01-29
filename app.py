import streamlit as st
from multiapp import MultiApp
from apps import home, model, model3 , modelRF  # import your app modules here model2

app = MultiApp()

st.markdown("""
#  Inteligencia de Negocios - Grupo 6

""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Modelo LSTM", model.app)
#app.add_app("Modelo SVR", model2.app)
app.add_app("Modelo Random Forest", modelRF.app)
app.add_app("Modelo SVC", model3.app)
# The main app
app.run()



