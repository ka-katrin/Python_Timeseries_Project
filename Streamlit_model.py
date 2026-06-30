#%writefile app.py

import streamlit as st
import mlflow
import pandas as pd

from Projekt.statistical_models import model_evaluation

# zum laufen lassen im Terminal: streamlit run Streamlit_model.py

st.title('🚀 Searching the best Model!')
st.write('Here will be some metrics to see! But first, I want you to guess the best model.')
st.write('')


# Add some interactive elements
name = st.text_input('Enter your name:')
if name:
    st.write(f'Hi there, {name}. Tell me your favorite model!')



number = int(st.text_input('Enter your guess (number of the model:'))
if number:
    st.write(f'{name} That was my guess too!')
    st.write(f'{name} - I really liked that model, it was easy to implement!')
else:
    st.write('Sorry, - I was not able to understand :((')

#click to show?
client = mlflow.tracking.MlflowClient()
#experiment = client.get_experiment_by_name("timeseries_project_forecasting")
#runs = client.search_runs(experiment.0)
df_runs = mlflow.search_runs(experiment_names=["timeseries_project_forecasting"])
print(df_runs[['run_id', 'metrics.mae', 'metrics.mse', 'metrics.r2']])
model_evaluation(df_runs)
#metrics=
#st.write(metrics)
#st.line_chart(metrics)


'''# Add a slider
number = st.slider('Pick a number', 0, 100, 50)
st.write(f'You selected: {number}')'''

# Add a chart
#st.write(model_evaluation)
#st.line_chart(model_evaluation)

#!streamlit run app.py --server.port 8501 --server.headless true >/dev/null 2>&1 &
#!./cloudflared tunnel --url http://localhost:8501 --no-autoupdate