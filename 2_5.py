import numpy as np
import matplotlib.pyplot as plt

def fun(x):
    y=np.sin(abs(x))-abs(x)/3+np.cos(6*abs(x));
    return y


def getdata():
    x=np.arange(-2.5,10.7,0.2)
    train_x=x[::3]
    train_y = fun(train_x)
    validation_x=x[1::3]
    validation_y = fun(validation_x)
    test_x=x[2::3]
    test_y = fun(test_x)
    return train_x,validation_x,test_x,train_y,validation_y,test_y

#.......................................................................
def fun1(x):
    y=np.sin(abs(x))-abs(x)/3+np.cos(5*abs(x));
    return y


def getdata1():
    x=np.arange(-2.5,2.7,0.15)
    train_x=x[::3]
    train_y = fun(train_x)
    validation_x=x[1::3]
    validation_y = fun(validation_x)
    test_x=x[2::3]
    test_y = fun(test_x)
    return train_x, validation_x, test_x, train_y, validation_y, test_y



trainx,validationx, testx, trainy, validationy, testy  = getdata()

def mmtpolyfit(X,Y,degree):
    poly = np.polyfit(X,Y,degree)
    Polyid = np.poly1d(poly)
    plt.plot(X, Y, 'o', mfc='r', mec='r')
    plt.plot(X, Polyid(X))
    plt.legend(["data"])
    plt.show()
    return Polyid

#mmtpolyfit(trainx,trainy,1)
mmtpolyfit(trainx,trainy,1)