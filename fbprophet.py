import pandas as pd 
from fbprophet import Prophet
import matplotlib.pyplot as plt

# ensure input file, linked below, contains two columns only for your historic data.. 'ds' for date and 'y' for your values.
df = pd.read_csv(r'C:\replace\with\your\input\file\path.csv')
df.head()
## uncomment the line below if you want to specify a change point in your data..eg a new website launch 
#model = Prophet(changepoints=['2020-04-14'])
model.fit(df,algorithm='Newton')
## use the line below to specify the number of days you'd like fbprophet to predict
future = model.make_future_dataframe(periods=60)
future.tail()
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
plot1 = model.plot(forecast)
plot2 = model.plot_components(forecast)
forecast.to_csv(r'C:\replace\with\your\output\file\path.csv')
plt.show()