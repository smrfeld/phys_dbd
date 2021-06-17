import tensorflow as tf
import numpy as np

@tf.keras.utils.register_keras_serializable(package="test")
class TestLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(TestLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super(TestLayer, self).get_config()
        print("Called layer")
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, input_tensor, training=False):
        return 2.0 * input_tensor

class TestModel(tf.keras.Model):

    def __init__(self, 
        x : float,
        layer = TestLayer(),
        **kwargs):
        super(TestModel, self).__init__(**kwargs)

        self.x = x
        # self.layer = TestLayer()
        self.layer = layer

    def get_config(self):
        print("get_config")
        return {
            "x": self.x,
            "layer": self.layer # For whatever reason, do NOT put this!
        }

    @classmethod
    def from_config(cls, config):
        print("from_config")
        return cls(**config)

    def call(self, input_tensor, training=False):
        return self.layer(input_tensor) + self.x

model = TestModel(x=0.5)

# Put in some input to build the model
test_input = np.random.rand(2,3)
test_output = model(test_input)
print("Test input: ", test_input)
print("Test output: ", test_output)

# Save
model.save("saved_models/model", save_traces=True)

print(model)

# Load
model = tf.keras.models.load_model(
    "saved_models/model", 
    custom_objects={"TestModel": TestModel}
    )

print(model)
print(model.x)

# Call it again
test_input = np.random.rand(2,3)
test_output = model(test_input)
print("Test input after loading: ", test_input)
print("Test output after loading: ", test_output)