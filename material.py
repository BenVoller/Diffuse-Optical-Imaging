import numpy as np 

class medium():

    def __init__(self):

        
        # refractive indexes
     

         # [depth, refractive_index(n), u_a, u_t, g]
        layer_null = [-999.9, 1, 1, 1, 0]
        layer0 = [float(0), 1, 1, 1, 0]
        layer1 = [0.01, 1, 10, 90, 0.75]
        layer2 = [0.02, 1, 10, 90, 0.75]
        layer3 = [999.9, 1, 1, 1, 0]

        self.layers = {-1:layer_null,
                       0:layer0,
                       1:layer1,
                       2:layer2,
                       3:layer3}
        
        self.depth = 0.02


     

    
    
       
            

    









