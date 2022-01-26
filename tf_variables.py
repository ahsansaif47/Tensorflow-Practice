from re import X
from numpy import gradient
import tensorflow as tf

# variable comprising of zeros..
var = tf.Variable([0, 0, 0])

# assigning it some value..
var.assign([1, 2, 3])


# differenciation..
# y = f(x)
# calculating gradients of model loss/error with model weights.

x = tf.Variable(1.0)


def f(x):
    y = x**2 + 2*x - 5
    return y


# now calculating derivative of function y wrt x
with tf.GradientTape() as tape:
    y = f(x)
    print("Value of y is: {}".format(y))

der_y = tape.gradient(y, x)
print("First derivative of y is: {}".format(der_y))
