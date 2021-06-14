from .net import RxnInputsLayer

import tensorflow as tf

class RxnModel(tf.keras.Model):
    def __init__(self, nv:int, nh: int, rxn_lyr: RxnInputsLayer, subnet: tf.keras.Model):
        super(RxnModel, self).__init__(name='')

        self.nv = nv
        self.nh = nh
        self.size_wt = nv*nh
        self.size_b = nv
        self.size_sig2 = 1
        self.no_outputs = self.size_wt + self.size_b + self.size_sig2

        self.rxn_lyr = rxn_lyr
        self.subnet = subnet
        self.output_lyr = tf.keras.layers.Dense(self.no_outputs, activation='relu')

    def call(self, input_tensor, training=False):
        x = self.rxn_lyr(input_tensor)
        x = self.subnet(x)
        x = self.output_lyr(x)

        # Reshape outputs into dictionary
        batch_size = tf.shape(x)[0]

        s = 0
        e = s + self.size_wt
        wt_TE = tf.reshape(x[:,s:e],shape=(batch_size,self.nh,self.nv))
        s = e
        e = s + self.size_b
        b_TE = x[:,s:e]
        s = e
        e = s + self.size_sig2
        sig2_TE = x[:,s:e]

        return {
            "wt_TE": wt_TE,
            "b_TE": b_TE,
            "sig2_TE": sig2_TE
        }