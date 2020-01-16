import theano
import theano.tensor as T
import numpy


m = numpy.random.randint(5,size=(5,3))
x = T.dmatrix('x')
s = 1/(1+T.exp(-x))
logestic = theano.function([x],s)
print(logestic(m))



