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


def costfunc(x, y, weight):
    inner = np.power((np.dot(np.matrix(x).T, np.matrix(weight)).T - y), 2)
    return np.sqrt( np.sum(inner) / (len(y)))

trainx,validationx, testx, trainy, validationy, testy  = getdata1()

def LSR_NonIterative(xtrain, ytrain):
    weight = np.array([0,0],np.float64)
    weight[1] = np.sum( (ytrain - np.mean(ytrain)) * (xtrain - (np.mean(xtrain))))/np.sum(np.power( xtrain - np.full(xtrain.shape, np.mean(xtrain)),2))
    weight[0] = np.mean(ytrain - weight[1]*xtrain)
    print(weight)
    plt.clf()
    plt.plot(trainx, trainy, 'o', mfc='r', mec='r')
    axes = plt.gca()
    yline = []
    for k in axes.get_xlim():
        yline.append(weight[0] + weight[1] * k)
    plt.plot(axes.get_xlim(), yline)
    plt.xlabel('train X')
    plt.ylabel('train Y')
    plt.show()

LSR_NonIterative(trainx,trainy)

