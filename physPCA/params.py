import numpy as np

from dataclasses import dataclass

def normalize(vec: np.array) -> np.array:
    return vec / np.sqrt(np.sum(vec**2))

@dataclass
class Params:

    wt: np.array
    varh_diag: np.array
    b: np.array
    muh: np.array
    sig2: float

    @classmethod
    def fromPCA(cls, data: np.array, muh: np.array, varh_diag: np.array):

        if len(muh) != len(varh_diag):
            raise ValueError("muh and varh_diag must be of the same length")

        # Dimensionality of data
        d = data.shape[1]

        # Number of latent parameters q < d
        q = len(muh)
        if q >= d:
            raise ValueError("Number of latent parameters: %d muss be less than dimensionality of the data: %d" % (q,d))

        # Mean
        b_ml = np.mean(data,axis=0)

        # Cov
        data_cov = np.cov(np.transpose(data))

        # Eigenvals/vecs, normalized by np
        eigenvals, eigenvecs = np.linalg.eig(data_cov)

        # Adjust sign
        direction_goal = normalize(np.ones(d))
        for i in np.arange(0,len(eigenvecs)):
            angle = np.arccos(np.dot(direction_goal,eigenvecs[:,i]))
            if abs(angle) < 0.5 * np.pi:
                eigenvecs[:,i] *= -1
        
        var_ml = (1.0 / (d-q)) * np.sum(eigenvals[q+1:d])
        uq = eigenvecs[:,:q]
        eigenvalsq = np.diag(eigenvals[:q])
        weight_ml = np.linalg.multi_dot([uq, np.sqrt(eigenvalsq - var_ml * np.eye(q))])

        # Make params in standardized space
        muh_0 = np.full(q, 0.0)
        varh_diag_0 = np.ones(q)
        params = Params(
            wt=np.transpose(weight_ml),
            b=b_ml,
            varh_diag=varh_diag_0,
            muh=muh_0,
            sig2=var_ml
            )

        # Convert params in standardized space
        params.convert_latent_space(muh, varh_diag)

        return params

    def convert_latent_space(self, muh_new: np.array, varh_diag_new: np.array):

        b1 = self.b
        wt1 = self.wt
        muh1 = self.muh
        varh1 = np.diag(self.varh_diag)
        varh1_sqrt = np.sqrt(varh1)

        muh2 = muh_new
        varh2 = np.diag(varh_diag_new)

        w1 = np.transpose(wt1)
        varh2_inv_sqrt = np.linalg.inv(np.sqrt(varh2))

        b2 = b1 + np.dot(w1,muh1) - np.linalg.multi_dot([w1,varh1_sqrt,varh2_inv_sqrt,muh2])
        wt2 = np.linalg.multi_dot([varh2_inv_sqrt,varh1_sqrt,wt1])

        self.b = b2
        self.wt = wt2