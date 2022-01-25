import tensorflow as tf

x = tf.constant([[1., 2., 3.], [4., 5., 6.]])

print(x)
print(x.shape)
print(x.dtype)

# tf constant operations

# tensors addition..
print("Addition operation..")
print(x+x)

# tensor and scaler multiplication..
print("Multiplication operation..")
print(5*x)

# transpose calculates the transpose of the tensor..
print("Transpose operation..")
print(tf.transpose(x))

# concatination can be done based on axis
# axis zero refers to row based concatination and vice versa..
print("Concatination operation..")
print(tf.concat([x, x, x], axis=0))

# reduced sum calculates the sum of all the elements combined..
print("Reduced sum..")
print(tf.reduce_sum(x))

# exponential operation
# y = e**(element)
print("Exponential operation..")
print(tf.math.exp(x))

# softmax is computed using tf.exp(x) / tf.reduced_sum(tf.exp(x), axis)
print("Softmax..")
print(tf.nn.softmax(x))
