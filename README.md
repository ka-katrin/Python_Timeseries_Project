# Hello #
This is my ML-project with a timeseries sales data to forecast, compare ML-algorithms, save metrics with mlflow 
and visualize the result via streamlit on a local website.

How to use: Run the following steps with the notebooks
_______________________________________________________________________________________________________________
## 1. Getting the Data 
The original data is from https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data
My data was preedited and from 2013-2014 (one year to train, 3 months to evaluate). So it might be different.
uploaded the cleaned version of the files, so if you want to try with the exact same data, there it is.
_______________________________________________________________________________________________________________
## 2. Cleaning the data -- Cleaner.py
I wrote a short script: Cleaner.py to clean the data, e.g. make sure all dates are there, no NaN Values etc.
Same with oil and holidays. The stores data was given, but I didn't need it. If you want to use my uploaded data,
skip steps 1 and 2, download the cleaned versions and start with step 3. 
_______________________________________________________________________________________________________________
## 3. EDA and statistical models -- statistical_models.ipynb
After loading the data and looking at it, splitting in train and test sets (at a specific time - change it if you 
must), three statistical models were used to forecast: ARIMA, SARIMA and Prophet. With S/ARIMA I first programmed 
a function (similar to GridSearch) to find the best hyperparameters before I used AUTO ARIMA to compare both. If 
you use different data, I advise you to try out different hyperparameters. Prophet has no Hyperparameters to edit 
- it finds the best for the data itself, or changing would change the model type.
I tracked the metrics inside the notebook via the variable 'model_evaluation' and outside with mlflow.
_______________________________________________________________________________________________________________
## 4. feature engineering -- feature_engineering_models.ipynb
Here I first engineered some features (lags, rolling average, holidays, weekdays, oil stock,..) to then use them
in the three models XGBoost, Random Forest and Linear Regression. The function iterative_forecast is to avoid data
leakage. XGBoost and Random Forest were optimized with HyperparameterOpt and everything again tracked with
model_evaluation and mlflow.
_______________________________________________________________________________________________________________
## 5. Local Website -- Streamlit_model.py
Run this only after all the other steps, or the mlflow experiment might be empty or not there at all. To avoid
having data doubled, there is a section to delete any double entries, so don't worry about running a script twice
and having the entries doubled in the mlflow experiment.
You might have to change the paths, though. In the beginning is a code for the command terminal to run, so the
website is built and opened. 
_______________________________________________________________________________________________________________
Have fun :)
