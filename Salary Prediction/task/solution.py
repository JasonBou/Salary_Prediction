import os
import requests
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('../Data/data.csv')

# find correlation between variables
#corr = data.corr()

#create heatmap graph to display correlation between variables
#plt.imshow(corr)
#plt.colorbar()
#plt.gcf().set_size_inches(7, 7)
#ticks = ['rating','salary','draft_round','age','experience','bmi']
#plt.xticks(range(len(corr.columns)), ticks, fontsize=12, rotation=90)
#plt.yticks(range(len(corr.columns)), ticks, fontsize=12)
#plt.title('salary and other elements correlation', fontsize=16, pad=20)

#labels = corr.values
#for a in range(labels.shape[0]):
#    for b in range(labels.shape[1]):
#        plt.text(a, b, '{:.3f}'.format(labels[b, a]), ha='center', va='center', color='black')
#print(data.corr)
#plt.show()

#determine predictors as x and y as target
y = data[['salary']]
#X= data[['rating','draft_round','age','experience','bmi']]
X= data[['rating','draft_round','bmi']]

#create a instance of linear regression
model = LinearRegression()

# determine train and test data split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=100)

#train the data with linear regression
model.fit(X_train, y_train)

#predict based on the test data
y_hat = model.predict(X_test)

# replace negative predictions with 0s
y_hat[y_hat<0] = 0

# replace negative predictions with median values
#median_y_hat = y_train.median()
#y_hat[y_hat<0] = median_y_hat

# calculate mape and print
mape_predict = mape(y_test, y_hat)
print(round(mape_predict,5))

#coeffs = []

#print different coefficients
#print(f'{model.coef_.flatten()[0]}, {model.coef_.flatten()[1]}, {model.coef_.flatten()[2]}, {model.coef_.flatten()[3]}, {model.coef_.flatten()[4]}')


#used previously to determine the best fit model and mappe
#z = 2
#list_name = ['mape_exp2','mape_exp3','mape_exp3']
#best_mappe = 100000

#for name in list_name:
#    new_x = X ** z

#    #determine train and test data split
#    X_train, X_test, y_train, y_test = train_test_split( new_x, y, test_size=0.3, random_state=100)
   
    #train the data with linear regression
#    model.fit(X_train, y_train)

    #Predict data based on test data
#    y_hat = model.predict(X_test)

    #calculate MAPE
#    name= mape(y_test, y_hat)
#    if name < best_mappe:
#        best_mappe = name
#    z +=1

#print(round(best_mappe,5))
#print(f'{round(float(model.intercept_),5)} {round(float(model.coef_),5)} {round(mape_predict,5)}')
