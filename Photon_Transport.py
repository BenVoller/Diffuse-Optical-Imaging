import numpy as np
import matplotlib.pyplot as plt 
import multiprocessing as mp
import time


np.random.seed(1234)

class photons():

    def __init__(self, weight):
        # Defines the initial x,y,z coordinates to be 000 an the cosine 
        self.pos = np.array([0,0,0])
        self.vel = np.array([0,0,1])

        # Extinction coefficient
        self.mu_t = 1
        self.s_ = 0
        
        self.mu_a = 0.25
        self.mu_t = 0.75

        self.W = weight 

        # Psuedo Random Number for the step size of the photon movement
        

    def eta(self):
        return np.random.random()
        

    def fresnelReflection(self, n0, n1):
        
        # Takes the two refractive indices of the ambient medium and the first medium and conmputes the 
        # proportion that is refelcted according to Specualar Fresnel Reflectance 

        Rsp = ((n0 - n1) / (n0 + n1))**2

        self.W += -Rsp

    def stepSize(self):
        
        self.s_ = -np.log(self.eta())
    



    def boundary_distance(self):
        # This to me looks very computationally intensive I think it would be better to move the photon and if 
        # boundary 
        pass 

    def move(self):
        self.pos = self.pos + self.vel*self.s_

    def reflect(self):
        pass

    def absorb(self):
        # Once a photon packet reaches an interaction site a fraction of it is absorbed 
        delW = self.mu_a / self.mu_t * self.W

        # Insert some call to adding the weigh to relative absorption

        self.W -= delW

    def scatter(self):
        g = 0.9 # Scattering Anisotropy for most biological tissue 

        if g != 0:
            costheta = (1/2*g)*(1 + g**2 - ((1-g**2)/(1-g+2*g*self.eta()))**2)

        else:
            costheta = 2*self.eta() -1

        

    def roulette(self):
        pass

    def photon_dead(self):
        pass




class mediums():

    def __init__(self, refractiveIndex):

        # For now this is a homogeneous medium 
        self.n0 = 1
        self.n1 = refractiveIndex

        
        self.grid = np.zeros([1000,1000,1000])



def run(medium):
    
    
        # Runs the photon trasnport for Monte Carlo photon trasnport 
        
        # Define Photon
        photon = photons(weight=1)
        print (photon.pos)
        print (photon.W)
        #print (photon.pos)

        photon.fresnelReflection(medium.n0, medium.n1)
        photon.stepSize()
        photon.move()
        photon.absorb()
        print(photon.pos)
        print(photon.W)
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
    numberPhotons = 1000 # Number of photons

    medium1 = mediums(refractiveIndex=2)

    #pool = mp.Pool(processes=n_cpu) 
    #results = [pool.map(run, range(numberPhotons , medium1))]

    run(medium1)


    t1 = time.time()
    
    print ('parallel time: ', t1 - t0)



