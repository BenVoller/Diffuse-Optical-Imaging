import numpy as np
import matplotlib.pyplot as plt 
import multiprocessing as mp
import time


np.random.seed(1234)

class photons():

    def __init__(self):
        # Defines the initial x,y,z coordinates to be 000 an the cosine 
        self.pos = [0,0,0]
        self.vel = [0,0,1]

        # Psuedo Random Number for the step size of the photon movement
        eta = np.random.random
    def stepSize(self):
        pass

    def boundary_distance(self):
        # This to me looks very computationally intensive I think it would be better to move the photon and if 
        # boundary 
        pass 

    def move(self):
        pass

    def reflect(self):
        pass

    def absorb(self):
        pass

    def scatter(self):
        pass

    def roulette(self):
        pass

    def photon_dead(self):
        pass
    

def run(numberPhotons):
    
    
        # Runs the photon trasnport for Monte Carlo photon trasnport 
        
        # Define Photon
        photon = photons()
        print (photon.pos)
        # Set step size of photon according to -ln(eta) where eta is a psuedo random number

        # Find Boundary distance or change in medium

        # If step > distance to boundary d_b * u_t move to boundary and test for reflect or trasmit otherwise move.

        # If not at boundary - Transmit, Absorb or Scatter

        # photon dead test, i,e absorbed or out of bounds 

        # weight check 

        # Roulette 

        # Repeat if photon is still alive

        # Last Photon? Then End.

    


if __name__ == '__main__':
    t0 = time.time()
    
    n_cpu = mp.cpu_count()  # = 8 
    numberPhotons = 100000 # Number of photons

    pool = mp.Pool(processes=n_cpu) 
    results = [pool.map(run, range(numberPhotons))]

    t1 = time.time()
    
    print ('parallel time: ', t1 - t0)



