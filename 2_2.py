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
#plt.plot(trainx, trainy ,'o', mfc='r', mec='r', )
#plt.xlabel('train X')
#plt.ylabel('train Y')

#plt.show()
weight = np.array([0, 0],float)

def costfunc(x, y, weight):
    inner = np.power((np.dot(np.matrix(x).T, np.matrix(weight)).T - y), 2)
    return np.sqrt( np.sum(inner) / (len(y)))

def LSR_Iterative(landa,xtrain, ytrain, epoch):
    ERMS = []
    cost = costfunc(xtrain, ytrain, weight)
    ERMS.append(cost)
    for i in range(epoch):
        for j in range(len(xtrain)):
            y = weight[1]*xtrain[j]+weight[0]
            weight[0] = weight[0] + landa * (ytrain[j]-y)
            weight[1] = weight[1] + landa * (ytrain[j]-y) *xtrain[j]
        cost = costfunc(xtrain, ytrain, weight)
        print(cost)
        ERMS.append(cost)
        plt.clf()
        plt.plot(trainx, trainy, 'o', mfc='r', mec='r' )
        axes = plt.gca()
        yline=[]
        for k in axes.get_xlim():
            yline.append(weight[0]+weight[1]*k)
        plt.plot(axes.get_xlim(), yline)
        plt.xlabel('train X')
        plt.ylabel('train Y')
        plt.show()

    plt.title('EMRS vs Iteration')
    plt.xlabel("Iteration")
    plt.ylabel('ERMS')
    plt.plot(ERMS)
    plt.show()
def LSR_Iterativeemrs(landa,xtrain, ytrain, emrs):
    ERMS = []
    Cost = 100
    it =0
    while (it < 100) & (Cost > emrs):
        it += 1

        for j in range(len(xtrain)):
            y = weight[1]*xtrain[j]+weight[0]
            weight[0] = weight[0] + landa * (ytrain[j]-y)
            weight[1] = weight[1] + landa * (ytrain[j]-y) *xtrain[j]
        Cost = costfunc(xtrain, ytrain, weight)
        print(Cost)
        ERMS.append(Cost)
        plt.clf()
        plt.plot(trainx, trainy, 'o', mfc='r', mec='r' )
        axes = plt.gca()
        yline=[]
        for k in axes.get_xlim():
            yline.append(weight[0]+weight[1]*k)
        plt.plot(axes.get_xlim(), yline)
        plt.xlabel('train X')
        plt.ylabel('train Y')
        plt.show()

    plt.title('ERMS vs Iteration')
    plt.xlabel("Iteration")
    plt.ylabel('ERMS')
    plt.plot(ERMS)
    plt.show()
LSR_Iterative(0.0001,xtrain =trainx,ytrain =trainy,epoch = 20 )
#LSR_Iterativeemrs(0.0001,xtrain =trainx,ytrain =trainy,emrs=2.3)

