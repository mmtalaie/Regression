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
        emrs.append(np.sqrt(np.sum(smm) / 2 * (len(y))))
    plt.plot(range(len(poly)), emrs)
    plt.annotate("minimom", xy=(np.argmin(emrs), np.min(emrs)), xytext=(np.argmin(emrs), np.min(emrs)),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    plt.legend(["EMRS"])
    plt.show()
    print("EMRS for train Data : %e" , np.min(emrs))
    return np.argmin(emrs)


def costfunc2(x, y, poly):
    emrs = 0
    Polyid = np.poly1d(poly)
    inner = np.power(Polyid(x) - y, 2)
    smm = np.sum(inner)
    emrs = np.sqrt(np.sum(smm) / 2 * (len(y)))

    print("EMRS for test Data : %e", emrs)


trainx, validationx, testx, trainy, validationy, testy = getdata1()


def mmtpolyfit(X, Y, degree):
    poly = np.polyfit(X, Y, degree)
    Polyid = np.poly1d(poly)
    return poly


def mmtpolyfit2(X, Y, degree):
    poly = np.polyfit(X, Y, degree)
    Polyid = np.poly1d(poly)
    plt.clf()
    plt.plot(X, Y, 'o', mfc='r', mec='r')
    plt.plot(trainx, Polyid(trainx))
    plt.legend(["data"])
    plt.show()
    return Polyid


ply = []
for i in range(20):
    ply.append(mmtpolyfit(trainx, trainy, i))
# costfunc(trainx,trainy, ply)
deg = costfunc(validationx, validationy, ply)
print(deg)

x = np.append(trainx, validationx, axis=0)
y = np.append(trainy, validationy, axis=0)

f = np.column_stack((x, y))
f[np.argsort(f[:, 0])]
x = f[:, 0]
y = f[:, 1]

plf = mmtpolyfit2(x, y, deg)
costfunc2(testx, testy, plf)
