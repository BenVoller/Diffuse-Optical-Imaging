import numpy as np 

class medium():

    def __init__(self, n1=1, z1=0.01, n2=1, z2=0.02):

        self.size = z1 + z2
        # refractive indexes
        self.z0 = 0 
        self.z1 = z1
        self.z2 = z2
        
        self.n0 = 1
        self.n1 = n1
        self.n2 = n2


        self.layers = {0: self.n0,
                       1: self.n1,
                       2:self.n2,
                       3: self.n0}
        
        self.distances = {0: self.z0,
                       1: self.z1,
                       2:self.z2,
                       3:99999}


     

    
    
       
            

    









