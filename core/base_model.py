from tensorflow import keras

class MyModel(keras.Model):
    def model(self, activation='swish', ker_initializer=None):
        """

        Parameters
        ----------
        activation : (optional: The default is 'swish') Activation function.
            
        ker_initializer : (optional: The default is None) Determines the selection 
                            process of the initial weights of each layer.

        Returns
        -------
        The individual NN model of the NN ensemble

        """
        inputs = keras.Input(shape=(1,))
        x = keras.layers.Dense(8, activation=activation, kernel_initializer=ker_initializer)(inputs)
        x = keras.layers.Dense(32, activation=activation, kernel_initializer=ker_initializer)(x)
        x = keras.layers.Dense(16, activation=activation, kernel_initializer=ker_initializer)(x)
        x = keras.layers.Dense(8, activation=activation, kernel_initializer=ker_initializer)(x)
        outputs = keras.layers.Dense(1)(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        return self.model      
