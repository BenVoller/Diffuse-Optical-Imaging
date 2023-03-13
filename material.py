import numpy as np 

class medium():

    def __init__(self, n0=2, z1=2, n1=4, z2=4):

        self.size = z1 + z2
        # refractive indexes
        self.z0 = 0 
        self.z1 = z1
        self.z2 = z2
 
        self.n0 = n0
        self.n1 = n1
        
   


        self.layers = {0: self.n0,
                       1: self.n1}
        
        self.distances = {0: 0,
                       1: self.z1,
                       2:self.z2,}


        self.z_array = np.ones(1000*self.size)

    
    
       
            

    









