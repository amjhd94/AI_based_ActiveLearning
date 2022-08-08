from tensorflow import keras

class MyModel(keras.Model):
    # {'activation':'swish', 'kernel_initializer':None} #activation = "swish", "elu", "softplus", "tanh"; All work fine in this case.
    def model(self, activation='swish', ker_initializer=None):
        inputs = keras.Input(shape=(1,))
        x = keras.layers.Dense(8, activation=activation, kernel_initializer=ker_initializer)(inputs)
        x = keras.layers.Dense(32, activation=activation, kernel_initializer=ker_initializer)(x)
        x = keras.layers.Dense(16, activation=activation, kernel_initializer=ker_initializer)(x)
        x = keras.layers.Dense(8, activation=activation, kernel_initializer=ker_initializer)(x)
        outputs = keras.layers.Dense(1)(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model      