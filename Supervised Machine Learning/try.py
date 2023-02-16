# v= round((2.546356456)**2,3)
# print(v)
import numpy as np
import pandas as pd
# x_train = np.array([1.1,1.3,1.5,2.0,2.2,2.9,3.0,3.2,3.2,3.7,3.9,4.0,4.0,4.1,4.5,4.9,5.1,5.3,5.9,6.0])
# y_train = np.array([39343,46205,37731,43525,39891,56642,60150,54445,64445,57189,63218,55794,56957,57081,61111,67938,66029,83088,81363,93940])
# print(x_train.shape[0],y_train.shape[0])
# x=np.array([[2323,34,3,4554435],[344,334,3,2,3,2],[32,34,2,223,32]])
# print(x.shape[0])
# y=np.zeros(10)
# print(y)
# ff=[[2,3,434],[34,341,222],[767,5464,8]]
# m,n=len(ff),len(ff[0])
# print(m,n)
# n=3
# dj_dw=np.zeros((n,))
# print(dj_dw)
df = pd.read_csv('Datasets/kc_house_data.csv')
print(df.head())
# print(df.head())
x_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','lat','long','sqft_living15','sqft_lot15']
fd = pd.DataFrame(df,columns=x_features)
# print(fd.loc[1].tolist())
# print(df['X2'].tolist())
# print(df['X3'].tolist())
# print(df['X4'].tolist())
x_train=[]
i=0
for i in range(300):
        x_train.append(fd.loc[i].tolist())
# mu=np.mean(x_train)
# s=np.std(x_train)
# x_norm = (x_train-mu)/s
# print(len(x_norm),len(x_norm[0]))
# print(len(x_train),len(x_train[0]))
# print(fd.head())
print(fd.loc[120].tolist())
