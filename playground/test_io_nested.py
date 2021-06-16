import tensorflow as tf
import numpy as np

@tf.keras.utils.register_keras_serializable(package="test")
class InnerLayer(tf.keras.layers.Layer):

    def __init__(self, x: float, **kwargs):
        super(InnerLayer, self).__init__(**kwargs)

        self.x = x

    def get_config(self):
        config = super(InnerLayer, self).get_config()
        config.update({
            "x": self.x
        })

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs):
        return 2.0 * inputs
    
@tf.keras.utils.register_keras_serializable(package="test")
class NestedLayer(tf.keras.layers.Layer):

    def __init__(self, 
        x_outer: float, 
        inner_layer: InnerLayer,
        **kwargs
        ):
        super(NestedLayer, self).__init__(**kwargs)

        self.x_outer = x_outer
        self.inner_layer = inner_layer

    @classmethod
    def construct(cls, x_inner: float, x_outer: float):
        return cls(
            x_outer=x_outer,
            inner_layer=InnerLayer(x_inner)
            )

    def get_config(self):
        config = super(NestedLayer, self).get_config()
        config.update({
            "x_outer": self.x_outer,
            "inner_layer": self.inner_layer
        })

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs):
        return self.inner_layer(inputs)

# Save a model with an inner layer
il = InnerLayer(x=0.5)
model = tf.keras.Sequential([il])

# Call it once to build it
test_input = np.random.rand(2,3)
test_output = model(test_input)
print("Test input:", test_input)
print("Test output:", test_output)

# Save
model.save("saved_models/test_save_inner") # Should work!

# Load back
model = tf.keras.models.load_model("test_save_inner")

# Call it again
test_input = np.random.rand(2,3)
test_output = model(test_input)
print("Test loaded input:", test_input)
print("Test loaded output:", test_output)

# Next: save a model with a nested layer
nl = NestedLayer.construct(x_inner=0.5, x_outer=0.2)
model = tf.keras.Sequential([nl])

# Call it once to build it
test_input = np.random.rand(2,3)
test_output = model(test_input)
print("Test input:", test_input)
print("Test output:", test_output)

# Save
# New in TensoFlow 2.4 The argument save_traces has been added to model.save, which 
# allows you to toggle SavedModel function tracing. Functions are saved to allow the 
# Keras to re-load custom objects without the original class definitons, so when save_traces=False, 
# all custom objects must have defined get_config/from_config methods. When loading, the custom 
# objects must be passed to the custom_objects argument. save_traces=False reduces the disk space 
# used by the SavedModel and saving time.
# https://www.tensorflow.org/guide/keras/save_and_serialize/
#
model.save("saved_models/test_save_nested", save_traces=True) # Warns: 'Found untraced functions such as...'
#
# Instead:
#
# model.save("test_save_nested", save_traces=False) # Works fine

# Load back
model = tf.keras.models.load_model("test_save_nested")

# Call it again
test_input = np.random.rand(2,3)
test_output = model(test_input)
print("Test loaded input:", test_input)
print("Test loaded output:", test_output)

