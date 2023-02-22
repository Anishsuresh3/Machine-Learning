import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy, math

def sigmoid(z):
    g = 1/(1+ np.exp(-z))
    return g

def compute_cost(x,y,w,b):
    m = len(x)
    cost=0.0
    for i in range(m):
        f_wb = sigmoid(np.dot(x[i],w)+b)
        cost += -y[i]*np.log(f_wb)-(1-y[i])*np.log(1-f_wb)
    cost = cost/m
    return cost

def compute_graident(x,y,w,b):
    m,n = len(x),len(x[0])
    dj_dw=np.zeros((n,))
    dj_db=0.
    for i in range(m):
        err = sigmoid(np.dot(x[i],w)+b) - y[i]
        for j in range(n):
            dj_dw[j]+=err*x[i][j]
        dj_db+=err
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db


def gradient_descent(x,y,w_in,b_in,alpha,iter):
    j_his=[]
    w=w_in
    b=b_in
    for i in range(iter):
        dj_dw,dj_db=compute_graident(x,y,w,b)
        
        w=w-alpha*dj_dw
        b=b-alpha*dj_db

        if i%100==0:
            j_his.append(compute_cost(x,y,w,b))

    return w,b,j_his
# 148 - 200 

def main():
    df = pd.read_csv('./Datasets/habermann.csv')
    df.columns=['age','year','axillary','survival']
    df['survival'] = df['survival']-1
    # No of 0 - 224
    # No of 1 - 81
    # fd=pd.DataFrame(df,columns=['age','axillary'])
    # x_train = []
    # for i in range(241):
    #     x_train.append(fd.loc[i].tolist())
    # y_train = df['survival'].tolist()[:241]
    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    # print(x_train)
    # print(y_train)
    # w_in=np.zeros(2)
    # b_in=0
    # alpha = 0.0001
    # iter=10000
    # w,b,j_his = gradient_descent(x_train,y_train,w_in,b_in,alpha,iter)
    # print(f"w: {w} ,b: {b}")
    # fig,ax=plt.subplots()
    # iter_arr=np.arange(100,10001,100)
    # ax.plot(iter_arr,j_his)
    # plt.show()
    # w: [-0.02715896  0.08194569] ,b: -0.014439518556218094



main()
