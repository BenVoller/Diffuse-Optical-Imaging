import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd


from material import medium
from Photon_Transport import photons

def run(number):
    
    if number % 100 == 0:
        #time.sleep(3)
        print (number)
    
    material = medium()
    photon = photons(material, weight=1)
    

    # Runs the photon trasnport for Monte Carlo photon transport 
    while photon.alive:
        
        photon.stepSize()
        photon.Refractive_index()

        if not photon.is_scattered:
            # Only true if the photon hasnt moved yet and also 
            photon.fresnelReflection() 

        while photon.hit_boundary():
    
            photon.transmission()
            photon.Refractive_index()
       
        if photon.W == 0:
            
            return photon.final
        
        photon.move()
        photon.absorb()
        photon.scatter()
        photon.roulette()
       
        
    #final_pos = np.concatenate((photon.pos, photon.vel))
    
    
    
    return photon.final