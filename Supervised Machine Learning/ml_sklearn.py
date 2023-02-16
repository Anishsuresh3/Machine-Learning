import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Datasets/kc_house_data.csv')
# print(df.head())
x_train = []
x_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','lat','long','sqft_living15','sqft_lot15']
fd = pd.DataFrame(df,columns=x_features)
# print(fd.head(15))
for i in range(301):
    x_train.append(fd.loc[i].tolist())
y_train = df['price'].tolist()[:301]

scaler = StandardScaler()
x_norm = scaler.fit_transform(x_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(x_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(x_norm,axis=0)}")

sgdr = SGDRegressor(max_iter=10000)
sgdr.fit(x_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters: w: {w_norm}, b:{b_norm}")

# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(x_norm)
print(f"Prediction on training set:\n{y_pred_sgd[4]}" )
print(len(y_pred_sgd))
print(f"Target values \n{y_train[4]}")
