import tensorflow as tf
from tensorflow import keras

class CustomModel(keras.Model):
    def __init__(self, hidden_units):
        super(CustomModel, self).__init__()
        self.hidden_units = hidden_units
        self.dense_layers = [keras.layers.Dense(u) for u in hidden_units]

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        return {"hidden_units": self.hidden_units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


model = CustomModel([16, 16, 10])
print(model)

# Build the model by calling it
input_arr = tf.random.uniform((1, 5))
outputs = model(input_arr)
model.save("saved_models/my_model")

# Option 1: Load with the custom_object argument.
loaded_1 = keras.models.load_model(
    "saved_models/my_model", custom_objects={"CustomModel": CustomModel}
)
print(loaded_1)