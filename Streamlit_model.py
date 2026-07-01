#%writefile app.py

import streamlit as st
import mlflow
import pandas as pd
from darts.models import ARIMA

# to run enter in terminal: streamlit run Projekt/Streamlit_model.py   #your path here or just the filename

st.title('🚀 Searching the best Model!')
st.write('Here will be some metrics to see! But first, I want you to guess the **:rainbow[best model]**.')
st.write('I tested the following Models:')

#Connecting to the mlflow file/experiment
mlflow.set_tracking_uri("sqlite:///D:/Katrin/Bildung und Ausbildung/5. Data Analyst/Python Timeseries/Projekt/mlflow.db")
mlflow.set_experiment("timeseries_project_forecasting")

client = mlflow.tracking.MlflowClient()
df_runs = mlflow.search_runs(experiment_names=["timeseries_project_forecasting"], run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY) #Dataframe of saved metrics etc

####   sort and delete double entries
df_sorted = df_runs.sort_values("start_time", ascending=False)
# Per runName only keep the newest, delete the rest (with group by runName)
to_delete = df_sorted.groupby("tags.mlflow.runName").apply(lambda x: x.iloc[1:]).reset_index(drop=True)

for run_id in to_delete["run_id"]:
    client.delete_run(run_id)
####


st.write(df_runs[['tags.mlflow.runName']].sort_values('tags.mlflow.runName', ascending=True)) #here are the models I used
st.write('This is the data, that I want to predict:')

#### loading data, and visualize
data = pd.read_csv('D:/Katrin/Bildung und Ausbildung/5. Data Analyst/Python Timeseries/Projekt/data_cleaned.csv')
data['date']= pd.to_datetime(data['Unnamed: 0'])
train = data[data['date'] < '2014-01-01']
test = data[data['date'] >= '2014-01-01']

holidays=pd.read_csv('D:/Katrin/Bildung und Ausbildung/5. Data Analyst/Python Timeseries/Projekt/holidays_cleaned.csv')
holidays['date']= pd.to_datetime(holidays['date'])

oil=pd.read_csv('D:/Katrin/Bildung und Ausbildung/5. Data Analyst/Python Timeseries/Projekt/oil_cleaned.csv')
oil['date']= pd.to_datetime(oil['Unnamed: 0'])

#visualize
st.line_chart(data, x='date',y='unit_sales',x_label='Date', y_label='Unit Sales')

####


st.write('Which one do you think is the winner?')
model_names = df_runs['tags.mlflow.runName'].sort_values().tolist()
best_model='' # create variable for later use
for name in model_names: #for every model one checkbox
    if st.checkbox(name): #for the checked checkbox:
        st.write(f"You chose {name}.. let's see if it's right")
        best_model=name #save for later to answer the given question

st.write("We've got four metrics: Mean absolute error, Mean squared error, R squared and max error.")
st.write("I'll give you a hint: The forecast of the models look like this:")

#### loading models, creating forecasts and visualize
#linear regression
run_id = df_runs[['run_id','metrics.mae']].sort_values('metrics.mae', ascending=True).iloc[0]['run_id'] #best model for mae
# Load the model
model_uri = f"runs:/{run_id}/model_linearr" #I know this is the best model. if its another -> one change the name
loaded_linear = mlflow.sklearn.load_model(model_uri)

# prophet
run_id = df_runs[df_runs['tags.mlflow.runName'] == 'Prophet']['run_id'].iloc[0]
model_uri = f"runs:/{run_id}/model_prophet"
loaded_prophet = mlflow.prophet.load_model(model_uri)

# Auto S/ARIMA, different, because arima/sarima is not normal saveable with mlflow
run_id = df_runs[df_runs['tags.mlflow.runName'] == 'Auto (S)ARIMA optimized']['run_id'].iloc[0]
local_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="model_sarima_opt"
)
loaded_sarima_opt = ARIMA.load(local_path)

# sARIMA self optimzied
run_id = df_runs[df_runs['tags.mlflow.runName'] == 'SARIMA self optimized']['run_id'].iloc[0]
local_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="model_sarima"
)
loaded_sarima = ARIMA.load(local_path)

# ARIMA self optimized
run_id = df_runs[df_runs['tags.mlflow.runName'] == 'ARIMA self optimized']['run_id'].iloc[0]
local_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="model_arima"
)
loaded_arima = ARIMA.load(local_path)

# Random Forest Opt
run_id = df_runs[df_runs['tags.mlflow.runName'] == 'Random Forest Opt']['run_id'].iloc[0]
model_uri = f"runs:/{run_id}/model_forest_opt"
loaded_forest_opt = mlflow.sklearn.load_model(model_uri)

# Random Forest
run_id = df_runs[df_runs['tags.mlflow.runName'] == 'Random Forest']['run_id'].iloc[0]
model_uri = f"runs:/{run_id}/model_forest"
loaded_forest = mlflow.sklearn.load_model(model_uri)

# XGBoost Opt
run_id = df_runs[df_runs['tags.mlflow.runName'] == 'XGBoost Opt']['run_id'].iloc[0]
model_uri = f"runs:/{run_id}/model_xgb_opt"
loaded_xgb_opt = mlflow.xgboost.load_model(model_uri)

# XGBoost
run_id = df_runs[df_runs['tags.mlflow.runName'] == 'XGBoost']['run_id'].iloc[0]
model_uri = f"runs:/{run_id}/model_xgboost"
loaded_xgb = mlflow.xgboost.load_model(model_uri)


#for feature engineering models:
def iterative_forecast(model, train_df, n_future_days):
    train_df = train_df.set_index("date")
    # Work on a copy so we don't mutate the original
    history = train_df["unit_sales"].copy()

    forecasts = []
    last_date = history.index[-1]

    for i in range(1, n_future_days + 1):
        next_date = last_date + pd.Timedelta(days=i)

        #
        # --- Lag features ---
        lag_1  = history.iloc[-1]
        lag_7  = history.iloc[-7]  if len(history) >= 7  else history.iloc[0]
        lag_30 = history.iloc[-30] if len(history) >= 30 else history.iloc[0]

        # --- Rolling average (last 2 values) ---
        rolling_avg_3 = history.iloc[-3:].mean()

        # --- Calendar features ---
        day_of_week = next_date.dayofweek          # 0 = Monday … 6 = Sunday
        is_weekend  = int(day_of_week >= 5)
        if next_date in set(holidays['date']):
            is_holiday= 1
        else:
            is_holiday = 0
        dcoilwtico  = oil[oil['date']==next_date]['dcoilwtico'].values[0]


        # --- Build the feature row (column order must match training) ---
        row = pd.DataFrame({
            "lag_1":          [lag_1],
            "lag_7":          [lag_7],
            "lag_30":         [lag_30],
            "rollings_avg_3": [rolling_avg_3],
            "day_of_week":    [day_of_week],
            "is_weekend":     [is_weekend],
            'is_holiday':     [is_holiday],
            "dcoilwtico":     [dcoilwtico]
        })

        #
        # --- Predict and store ---
        yhat = model.predict(row)[0]
        forecasts.append({"date": next_date, "unit_sales": yhat})

        # --- Append prediction to history so next iteration can use it ---
        history.loc[next_date] = yhat

    return pd.DataFrame(forecasts)

#predictions
pred_linear = iterative_forecast(loaded_linear, train, len(test))
pred_prophet = loaded_prophet.make_future_dataframe(periods=len(test))
pred_prophet = pd.DataFrame(loaded_prophet.predict(pred_prophet))
pred_auto_sarima=loaded_sarima_opt.predict(len(test))
pred_sarima=loaded_sarima.predict(len(test))
pred_arima=loaded_sarima.predict(len(test))
pred_forest_opt = iterative_forecast(loaded_forest_opt, train, len(test))
pred_forest = iterative_forecast(loaded_forest, train, len(test))
pred_xgb_opt = iterative_forecast(loaded_xgb_opt, train, len(test))
pred_xgb = iterative_forecast(loaded_xgb, train, len(test))

# toggle and things to click at, choosing the model to show
all_models = df_runs['tags.mlflow.runName'].tolist()
with st.container(border=True):
    Models = st.multiselect("Models", all_models, default='Random Forest') #only one pre-selected, otherwise its a mess

#DF with all data - name of the columns = name of the model, to later visualize only the selected
predictions = pd.concat([
    test['date'].reset_index(drop=True),
    pred_prophet['yhat'].reset_index(drop=True),
    pd.Series(pred_auto_sarima['unit_sales'].values().flatten()),
    pd.Series(pred_sarima['unit_sales'].values().flatten()),
    pd.Series(pred_arima['unit_sales'].values().flatten()),
    pred_linear['unit_sales'].reset_index(drop=True),
    pred_forest_opt['unit_sales'].reset_index(drop=True),
    pred_forest['unit_sales'].reset_index(drop=True),
    pred_xgb_opt['unit_sales'].reset_index(drop=True),
    pred_xgb['unit_sales'].reset_index(drop=True)
], axis=1)
predictions.columns = ['date', 'Prophet', 'Auto (S)ARIMA optimized', 'SARIMA self optimized',
                        'ARIMA self optimized', 'Linear Regression', 'Random Forest Opt',
                        'Random Forest', 'XGBoost Opt', 'XGBoost'] #names are exactly like in tags.mlflow.runName


# visualize: in predictions is all the predictions from every model, y=Model ensures that the selected Models are shown
st.line_chart(predictions, x='date',y=Models,x_label='Date', y_label='Unit Sales')
####



st.title("Make your Guess, don't scroll down and peek!")
st.divider()
st.divider()
st.divider()# three, so accidentally peeking is harder...
st.write("Here are the Results!")
#always show the three best at every score
st.write("Best Mean absolute Error")
st.write(df_runs[['tags.mlflow.runName', 'metrics.mae', 'metrics.mse', 'metrics.r2','metrics.max_error']].sort_values('metrics.mae', ascending=True).head(3))
st.write("Best Mean squared Error")
st.write(df_runs[['tags.mlflow.runName', 'metrics.mae', 'metrics.mse', 'metrics.r2','metrics.max_error']].sort_values('metrics.mse', ascending=True).head(3))
st.write("Best R squared")
st.write(df_runs[['tags.mlflow.runName', 'metrics.mae', 'metrics.mse', 'metrics.r2','metrics.max_error']].sort_values('metrics.r2', ascending=False).head(3))
st.write("Best Max Error")
st.write(df_runs[['tags.mlflow.runName', 'metrics.mae', 'metrics.mse', 'metrics.r2','metrics.max_error']].sort_values('metrics.max_error', ascending=True).head(3))

st.title("The best model in nearly all metrics is **:rainbow[Linear Regression]**") #make it fancy
st.write("See how it matches the Data:")

pred_linear['Prediction']=pred_linear['unit_sales'].reset_index(drop=True)
data_forecast=pd.concat([test.reset_index(drop=True), pred_linear['Prediction']], axis=1) #need all data in one DF
st.line_chart(data_forecast,x= 'date', x_label='Date', y=['Prediction','unit_sales'], y_label='Unit Sales')


#results of checked box
st.write('Your Result:')
st.write(f'You chose {best_model}')
if best_model == 'Linear Regression':
    st.write('Congratulations!')
    if st.button("Send balloons!"):
        st.balloons()           #a little bit fun!
    else:
        st.write('Better luck next time!')
