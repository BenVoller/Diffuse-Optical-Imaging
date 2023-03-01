import numpy as np
import matplotlib.pyplot as plt 
import multiprocessing as mp
import pandas as pd
import time

from material import *


np.random.seed(1234)

class photons():

    def __init__(self, weight):
        # Defines the initial x,y,z coordinates to be 000 an the cosine 

        self.alive = True 
        
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
    



    def boundary_distance(self, material):
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
            theta = np.arccos(1/2*g)*(1 + g**2 - ((1-g**2)/(1-g+2*g*self.eta()))**2)

        else:
            theta = np.arccos(2*self.eta() -1)

        phi = 2*np.pi*self.eta()

        # Now define new velocity angles 
        u_x, u_y, u_z  = self.vel


        if abs(u_z) > 0.9999:
            u_x = np.sin(theta)*np.cos(phi)
            u_y = np.sin(theta)*np.sin(phi)
            u_z = np.sign(u_z)*np.cos(theta) # Sgn function returns one when the u_z is positive and -1 when negative

        else:
            u_x = (np.sin(theta)*(u_x*u_z*np.cos(phi) - u_y*np.sin(phi)))/np.sqrt(1-u_z**2) + u_z*np.cos(theta)
            u_y = (np.sin(theta)*(u_y*u_z*np.cos(phi) + u_x*np.sin(phi)))/np.sqrt(1-u_z**2) + u_y*np.cos(theta)
            u_z = np.sqrt(1-u_z**2)*(np.sin(theta)*np.cos(phi)) + u_z*np.cos(theta)

        self.vel = np.array([u_x, u_y, u_z])

    

    def roulette(self):
        # defines the chance that a photon is terminated
        m = 10 

        if self.W < 0.0001:
            eta = self.eta()

            if eta <= 1/m:
                self.W = m*self.W
            else:
                self.W = 0
                self.alive = False



    def photon_dead(self):
        # Currently the only mechanism for photon termination is roulette, include boundaries soon.
        pass




class mediums():

    def __init__(self, size, refractiveIndex1):

        # For now this is a homogeneous medium 
        self.n0 = 1
        self.n1 = refractiveIndex1



        self.grid = np.zeros([1000,1000,1000])
        material_z  = np.ones(1000)

        



def run(medium):
    
    names = ['x','y','z','vx','vy','vz']
    data = np.empty(len(names))
    df = pd.DataFrame(columns=names)
    
    two_layer = material(l1depth=1, l1n=1, l2depth=1, l2n=2)
    photon = photons(weight=1)


    
    # Runs the photon trasnport for Monte Carlo photon trasnport 
    while photon.alive:
        # Define Photon
        
        #(photon.pos)
        #print (photon.W)
       
        photon.boundary_distance(two_layer.z_array)
        photon.fresnelReflection(two_layer.n0, two_layer.n1)
        photon.stepSize()
        photon.move()
        photon.absorb()
        photon.scatter()
        photon.roulette()
        
        #print('position:{}, velocity: {}, weight: {}'.format(photon.pos, photon.vel, photon.W))
        
        # Set step size of photon according to -ln(eta) where eta is a psuedo random number

        # Find Boundary distance or change in medium


        # If step > distance to boundary d_b * u_t move to boundary and test for reflect or trasmit otherwise move.

        # If not at boundary - Transmit, Absorb or Scatter

        # photon dead test, i,e absorbed or out of bounds 

        # weight check 

        # Roulette 

        # Repeat if photon is still alive

        # Last Photon? Then End.

    # Eventually include wavelength in this.
      
    final_pos = np.concatenate((photon.pos, photon.vel))

    data = np.vstack([data, final_pos])
    

    
    print (data)
    return data
    

if __name__ == '__main__':
    t0 = time.time()

    n_cpu = mp.cpu_count()  # = 8 
    numberPhotons = 1000 # Number of photons

    medium1 = mediums(4, refractiveIndex1=2)

    #pool = mp.Pool(processes=n_cpu) 
    #results = [pool.map(run, range(numberPhotons , medium1))]

    run(medium1)


    t1 = time.time()
    
    print ('parallel time: ', t1 - t0)



