import numpy as np 

class medium():

    def __init__(self):

        
        # refractive indexes
        self.NumberPhotons = 1000

        # [depth, refractive_index(n), u_a, u_t, g]
        layer_null = [-999.9, 1, 1, 1, 0]
        layer0 = [float(0), 1, 1, 1, 0]
        layer1 = [0.5, 1.37, 0.1, 100, 0.9]
        layer2 = [1, 1.37, 0.1, 100, 0.9]
        layer3 = [999.9, 1, 1, 1, 0]

        self.layers = {-1:layer_null,
                       0:layer0,
                       1:layer1,
                       2:layer2,
                       3:layer3}
        
        # Re packages the layers for working out other quantities
        # The above dictionary has extra layers to fix boundary searching problems
        self.layers_important = {0:layer1,
                                 1:layer2}
        
        self.depth = 1


     

    
    
       
            

    









