import numpy as np
import matplotlib.pyplot as plt 
import multiprocessing as mp


np.random.seed(1234)

class photon():

    def init(self):
        # Defines the initial x,y,z coordinates to be 000 an the cosine 
        self.init_pos = [0,0,0]
        self.init_vel = [0,0,1]

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
    

def run():
    # Runs the photon trasnport for Monte Carlo photon trasnport 
    
    # Define Photon


    # Set step size of photon according to -ln(eta) where eta is a psuedo random number

    # Find Boundary distance or change in medium

    # If step > distance to boundary d_b * u_t move to boundary and test for reflect or trasmit otherwise move.

    # If not at boundary - Transmit, Absorb or Scatter

    # photon dead test, i,e absorbed or out of bounds 

    # weight check 

    # Roulette 

    # Repeat if photon is still alive

    # Last Photon? Then End.

    print ('Run')


print(f"Number of cpu: {mp.cpu_count()}")