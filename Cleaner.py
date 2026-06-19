import pandas as pd

#Daten einlesen
data=pd.read_csv('timeseries.csv')
holidays = pd.read_csv('holidays.csv')
oil= pd.read_csv('oil.csv')
stores=pd.read_csv('stores.csv')

#data.head() and
# data.info() zur Übersicht

data['date'] = pd.to_datetime(data['date'])  #before it was str, now its a date objekt
data.set_index('date', inplace=True)   # Set date as the index
#data.isnull().sum()            #checked for missing values, there are none


date_range = pd.date_range(start=data.index.min(), end=data.index.max())  #all the dates, there should be
#date_range.difference(data.index)      #which ones are missing? Two are missing

#set index to date range -> missing dates are now in it
data = data.reindex(date_range)
#forward fill: fills nan values with the last entry
data['unit_sales'] = data['unit_sales'].ffill()

#### For holidays
holidays['date'] = pd.to_datetime(holidays['date'])
holidays.set_index('date', inplace=True)
#checked the other entries with .unique() and for uncleaned data. None was found, so its done.
#No NaN-values
#no filling up, there have to be dates missing

#### For oil
oil['date'] = pd.to_datetime(oil['date'])
oil.set_index('date', inplace=True)
date_range_o = pd.date_range(start=oil.index.min(), end=oil.index.max())

oil = oil.reindex(date_range_o)
oil['dcoilwtico'] = oil['dcoilwtico'].ffill()
#more values are missing -> logically always the weekends.
# filled them nonetheless, because it may make a difference to have the information of the day before

#### stores
#no cleaning to be done. Checked with unique() for uncleaned strings and like before for NaN-Values

### Export the cleaned files
data.to_csv('data_cleaned.csv', index=True, header=True )
holidays.to_csv('holidays_cleaned.csv', index=True, header=True)
oil.to_csv('oil_cleaned.csv', index=True, header=True)

