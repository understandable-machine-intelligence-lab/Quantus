import tensorflow as tf

# There's no need to compile small models used for tests
tf.config.run_functions_eagerly(True)
