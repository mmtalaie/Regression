import numpy as np
import matplotlib.pyplot as plt


def fun(x):
    y = np.sin(abs(x)) - abs(x) / 3 + np.cos(6 * abs(x));
    return y


def getdata():
    x = np.arange(-2.5, 10.7, 0.2)
    train_x = x[::3]
    train_y = fun(train_x)
    validation_x = x[1::3]
    validation_y = fun(validation_x)
    test_x = x[2::3]
    test_y = fun(test_x)
    return train_x, validation_x, test_x, train_y, validation_y, test_y


# .......................................................................
def fun1(x):
    y = np.sin(abs(x)) - abs(x) / 3 + np.cos(5 * abs(x));
    return y


def getdata1():
    x = np.arange(-2.5, 2.7, 0.15)
    train_x = x[::3]
    train_y = fun(train_x)
    validation_x = x[1::3]
    validation_y = fun(validation_x)
    test_x = x[2::3]
    test_y = fun(test_x)
    return train_x, validation_x, test_x, train_y, validation_y, test_y


def costfunc(x, y, poly):
    emrs = []
    for i in range(len(poly)):
        Polyid = np.poly1d(poly[i])
        inner = np.power(Polyid(x) - y, 2)
        smm = np.sum(inner)
        emrs.append(np.sqrt(np.sum(smm) / 2*(len(y))))
    plt.plot(range(len(poly)), emrs)
    plt.legend(["EMRS"])
    plt.show()


trainx, validationx, testx, trainy, validationy, testy = getdata1()


def mmtpolyfit(X, Y, degree):
    poly = np.polyfit(X, Y, degree)
    Polyid = np.poly1d(poly)
    return poly


ply = []
for i in range(20):
    ply.append(mmtpolyfit(trainx, trainy, i))
costfunc(validationx, validationy, ply)
