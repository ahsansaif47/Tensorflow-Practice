from multiprocessing.sharedctypes import Value
import tensorflow as tf


class Module(tf.Module):
    def __init__(self, val):
        self.weight = tf.Variable(val)

    @tf.function
    def fun(self, x):
        return x * self.weight


mod = Module(3)
mod.fun(tf.constant([1, 2, 3]))

model_path = "./Saved Models/"
tf.saved_model.save(mod, model_path)


loaded_model = tf.saved_model.load(model_path)
print(loaded_model.fun(tf.constant([4, 5, 6])))
