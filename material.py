import numpy as np 

class medium():

    def __init__(self, n0=1, z0=2, n1=2, z1=4):

        self.size = z0 + z1
        # refractive indexes
        self.z0 = z0
        self.z1 = z1
        self.n0 = n0
        self.n1 = n1

        self.z_array = np.ones(1000*self.size)

      

    









