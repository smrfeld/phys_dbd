from ..net_common import unit_mat, BirthRxnLayer, DeathRxnLayer, \
    EatRxnLayer, ConvertMomentsToNMomentsLayer, ConvertNMomentsTEtoMomentsTE

import tensorflow as tf

from typing import List, Tuple, Union

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParamsToMomentsGaussLayerFV(tf.keras.layers.Layer):

    def __init__(self,
        nv : int,
        **kwargs
        ):
        # Super
        super(ConvertParamsToMomentsGaussLayerFV, self).__init__(**kwargs)
        self.nv = nv 
    
    def get_config(self):
        config = super(ConvertParamsToMomentsGaussLayerFV, self).get_config()
        config.update({
            "nv": self.nv
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        
        mu = inputs["mu"]
        chol = inputs["chol"]

        cholt = tf.transpose(chol, perm=[0,2,1])

        prec = tf.matmul(chol,cholt)
        cov = tf.linalg.inv(prec)

        return {
            "prec": prec,
            "mu": mu,
            "cov": cov
        }

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertParamsToNMomentsGaussLayerFV(tf.keras.layers.Layer):

    def __init__(self, 
        nv: int,
        **kwargs
        ):
        """Convert params to NMoments layer in one step

        Args:
            nv (int): No. visible species
        """
        # Super
        super(ConvertParamsToNMomentsGaussLayerFV, self).__init__(**kwargs)

        self.nv = nv

        self.paramsToMomentsLayer = ConvertParamsToMomentsGaussLayerFV(nv)
        self.momentsToNMomentsLayer = ConvertMomentsToNMomentsLayer()

    def get_config(self):
        config = super(ConvertParamsToNMomentsGaussLayerFV, self).get_config()
        config.update({
            "nv": self.nv
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        moments = self.paramsToMomentsLayer(inputs)
        nmoments = self.momentsToNMomentsLayer(moments)

        # Collect all parts
        dall = {}
        for d in [inputs,moments,nmoments]:
            dall.update(d)
        
        return dall

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertMomentsTEtoParamsTEGaussLayerFV(tf.keras.layers.Layer):

    def __init__(self, nv: int, **kwargs):
        # Super
        super(ConvertMomentsTEtoParamsTEGaussLayerFV, self).__init__(**kwargs)
    
        self.nv = nv

    def get_config(self):
        config = super(ConvertMomentsTEtoParamsTEGaussLayerFV, self).get_config()
        config.update({
            "nv": self.nv
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        muTE = inputs["muTE"]
        covTE = inputs["covTE"]

        chol = inputs["chol"]
        cholT = tf.transpose(chol,perm=[0,2,1])

        batch_size = tf.shape(inputs["muTE"])[0]

        phi_mat = tf.matmul(cholT,tf.matmul(covTE,chol))

        cholTE = -1.0 * tf.matmul(chol, apply_phi_mat(phi_mat, self.nv, batch_size))

        return {
            "muTE": muTE,
            "cholTE": cholTE
        }

def apply_phi_mat(arg_mat, n: int, batch_size: int):
    res = tf.zeros((batch_size,n,n))
    for i in range(0,n):
        for j in range(0,n):
            if i > j:
                res += tf.map_fn(
                    lambda arg_matL: unit_mat(n,i,j)*arg_matL[i,j],
                    arg_mat)
            elif i == j:
                res += tf.map_fn(
                    lambda arg_matL: 0.5*unit_mat(n,i,j)*arg_matL[i,j],
                    arg_mat)
    return res

@tf.keras.utils.register_keras_serializable(package="physDBD")
class ConvertNMomentsTEtoParamsTEGaussLayerFV(tf.keras.layers.Layer):

    def __init__(self, nv: int, **kwargs):

        # Super
        super(ConvertNMomentsTEtoParamsTEGaussLayerFV, self).__init__(**kwargs)
        
        self.nv = nv

        self.nMomentsTEtoMomentsTE = ConvertNMomentsTEtoMomentsTE()
        self.momentsTEtoParamsTE = ConvertMomentsTEtoParamsTEGaussLayerFV(nv)

    def get_config(self):
        config = super(ConvertNMomentsTEtoParamsTEGaussLayerFV, self).get_config()
        config.update({
            "nv": self.nv
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        outputs0 = self.nMomentsTEtoMomentsTE(inputs)

        outputs0.update(inputs)
        outputs1 = self.momentsTEtoParamsTE(outputs0)

        return outputs1

@tf.keras.utils.register_keras_serializable(package="physDBD")
class RxnInputsGaussLayerFV(tf.keras.layers.Layer):

    def __init__(self, 
        nv: int, 
        rxn_specs : List[Union[Tuple[str,int],Tuple[str,int,int]]],
        **kwargs
        ):
        """Calculate inputs from different reaction approximations in a single layer. 

        Args:
            nv (int): No. visible species
            rxn_specs (List[Union[Tuple[str,int],Tuple[str,int,int]]]): List of reaction specifications of tuples.
                First arguments is one of "BIRTH", "DEATH" or "EAT"
                Second/third argument are the index of the species undergoing the reaction
                e.g. ("EAT",3,2) is the Predator-Prey reaction for species #3 being the huner, species #2 the prey

        Raises:
            ValueError: If reaaction string is not recognized
        """
        # Super
        super(RxnInputsGaussLayerFV, self).__init__(**kwargs)
        
        self.nv = nv

        self.paramstoNMoments = ConvertParamsToNMomentsGaussLayerFV(nv=nv)
        self.nmomentsTEtoParamsTE = ConvertNMomentsTEtoParamsTEGaussLayerFV(nv=nv)

        self.rxn_specs = rxn_specs
        self.rxns = []
        for spec in rxn_specs:
            if spec[0] == "EAT":
                rtype, i_hunter, i_prey = spec
            else:
                rtype, i_sp = spec

            if rtype == "BIRTH":
                self.rxns.append(BirthRxnLayer(nv, 0, i_sp))
            elif rtype == "DEATH":
                self.rxns.append(DeathRxnLayer(nv, 0, i_sp))
            elif rtype == "EAT":
                self.rxns.append(EatRxnLayer(nv, 0, i_hunter, i_prey))
            else:
                raise ValueError("Rxn type: %s not recognized" % rtype)

    def get_config(self):
        config = super(RxnInputsGaussLayerFV, self).get_config()

        config.update({
            "nv": self.nv,
            "rxn_specs": self.rxn_specs
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def make_tril_bool_mask(self, batch_size):
        ones = tf.ones((self.nv, self.nv), dtype=tf.dtypes.int32)
        mask = tf.experimental.numpy.tril(ones)
        mask = tf.dtypes.cast(mask, tf.dtypes.bool)
        # Repeat mask into batch size
        # Size then is (batch_size,nv,nv)
        mask = tf.repeat(mask[tf.newaxis, :, :], batch_size, axis=0)
        return mask

    def call(self, inputs):
        nmoments = self.paramstoNMoments(inputs)
        batch_size = tf.shape(nmoments["cov"])[0]
        tril_mask = self.make_tril_bool_mask(batch_size)

        # Compute reactions
        paramsTEforRxns = []
        for i in range(0,len(self.rxns)):
            # print("--- Rxn idx: %d ---" % i)

            nmomentsTE = self.rxns[i](nmoments)

            # Convert nMomentsTE to params0TE
            nmomentsTE.update(nmoments)
            paramsTE = self.nmomentsTEtoParamsTE(nmomentsTE)

            # Flatten
            # Reshape (batch_size, a, b) into (batch_size, a*b) for each thing in the dict
            cholTE_vec = tf.boolean_mask(paramsTE["cholTE"], tril_mask)
            paramsTEarr = [
                tf.reshape(cholTE_vec, (batch_size, int(self.nv*(self.nv+1)/2))),
                tf.reshape(paramsTE["muTE"], (batch_size, self.nv))
                ]

            # Combine different tensors of size (batch_size, a), (batch_size, b), ... 
            # into one of (batch_size, a+b+...)
            paramsTEflat = tf.concat(paramsTEarr, 1)

            # Store
            paramsTEforRxns.append(paramsTEflat)

        # Flatten all reactions
        # Combine different tensors of size (batch_size, a), (batch_size, b), ... 
        # into one of (batch_size, a+b+...)
        paramsTE = tf.concat(paramsTEforRxns, 1)

        return paramsTE