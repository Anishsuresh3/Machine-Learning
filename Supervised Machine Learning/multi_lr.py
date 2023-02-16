import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict(x,w,b):
    f_wb = np.dot(x,w)+b
    return f_wb

def compute_cost(x,y,w,b):
    m = len(x)
    cost=0.0
    for i in range(m):
        f_wb = np.dot(x[i],w)+b
        cost+=(f_wb-y[i])**2
    cost=cost/(2*m)
    return cost

def compute_gradient(x,y,w,b):
    m,n=len(x),len(x[0]) #300 #4
    dj_dw=np.zeros((n,))
    dj_db=0.
    for i in range(m):
        err =(np.dot(x[i],w)+b)-y[i]
        # print(err)
        for j in range(n):
            dj_dw[j] += err * x[i][j]
        dj_db += err
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw,dj_db

def gradient_descent(x,y,w_in,b_in,alpha,iter):
    w=w_in
    b=b_in
    j_his=[]
    for i in range(iter):
        dj_dw,dj_db=compute_gradient(x,y,w,b)

        w = w-alpha*dj_dw
        b = b-alpha*dj_db
        
        if i%100==0:
            j_his.append(compute_cost(x,y,w,b))
    
    return w,b,j_his

def z_score_normalization(x):
    mu=np.mean(x)
    sigma = np.std(x)
    x_norm =(x-mu)/sigma
    return x_norm

def main():
    df = pd.read_csv('Datasets/kc_house_data.csv')
    # print(df.head())
    x_train = []
    x_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','lat','long','sqft_living15','sqft_lot15']
    fd = pd.DataFrame(df,columns=x_features)
    for i in range(301):
        x_train.append(fd.loc[i].tolist())
    y_train = df['price'].tolist()[:301]
    # print(x_train)
    # print(y_train)
    x_norm = z_score_normalization(x_train)
    # print(len(x_norm),len(x_norm[0]))
    alpha=0.0001
    iter=10000
    w_initial=np.zeros(15)
    b_initial=0
    w_final,b_final,j_his=gradient_descent(x_norm,y_train,w_initial,b_initial,alpha,iter)
    print(w_final)
    print(b_final)
    x_test=[3.0, 2.5, 2400.0, 6474.0, 1.0, 0.0, 2.0, 3.0, 8.0, 1560.0, 840.0, 47.7728, -122.386, 2340.0, 10856.0]
    x_test_norm=z_score_normalization(x_test)
    # print(x_test_norm)
    print(predict(x_test_norm,w_final,b_final))
    fig,ax=plt.subplots()
    iter_arr=np.arange(100,10001,100)
    ax.plot(iter_arr,j_his)
    plt.show()

main()
