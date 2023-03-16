import numpy as np
import matplotlib.pyplot as plt 
import multiprocessing as mp
import pandas as pd
import time

from material import *


np.random.seed(1234)

class photons():

    def __init__(self, medium, weight):
        # Defines the initial x,y,z coordinates to be 000 an the cosine 

        self.reflectance = np.empty(4)
        self.transmittance = np.empty(4)

        self.z0 = 1
        self.z1 = medium.z1

        self.alive = True 
        
        self.pos = np.array([0,0,0])
        self.vel = np.array([0,0,1])

        # Extinction coefficient
        self.mu_a = 0.25
        self.mu_t = 0.75

        self.W = weight 

        # These are the refractive indices and the current one is set to the first in the array 
        self.indices = medium.layers
        self.n_current = self.indices[0]
        self.layer_no = 0 # Defines which layer we are starting in

        self.distances = medium.distances
        self.z_current = self.distances[1] 

        self.upper_bound = self.distances[2]

        self.ni = 1
        self.nt = self.n_current
     



        # Psuedo Random Number for the step size of the photon movement
    def eta(self):
        return np.random.random()
    
    
    def stepSize(self):
        
        self.s_ = -np.log(self.eta())


    def Refractive_index(self):

        # Returns a postive or negative number based on the direction of the photon
        direction = np.sign(self.vel[-1])
        z = self.pos[-1]
        
        # Sets the next boundary to psuedo infinity
        if direction == 0: 
            db = 99999999

        if z < self.distances[0]: 
            
            pass

        elif z < self.distances[1]:
            # current refractive index
            ni = self.indices[1]
            
        
            if direction == 1:
                
                
                # Distance to next layer, next refractive index
                zt = self.distances[1]
                
                nt = self.indices[2]
                

                
            elif direction == -1:
                zt = self.distances[0]
                nt = self.indices[0]

        elif z == self.distances[1]: 
            
            if direction == 1:
                ni = self.indices[2]
                nt = self.indices[3]
                zt = self.distances[2]

            elif direction == -1:
                 ni = self.indices[1]
                 nt = self.indices[0]
                 zt = self.distances[0]

        elif z < self.distances[2]: 
            # current refractive index
            
            
            ni = self.indices[2]

            if direction == 1:
                # Distance to next layer, next refractive index
                zt = self.distances[2]
                nt = self.indices[3]

            elif direction == -1:
                zt = self.distances[1]
                nt = self.indices[1]

        elif z == self.distances[2]:

            ni = self.indices[2]

            if direction == 1:
                zt = self.distances[3]
                nt = self.indices[3]

            if direction == -1: 
                zt = self.distances[1]
                nt = self.indices[1]

    
        db = (zt - z) / self.vel[-1]
        # Returns the current refractive layer and then the next layer which the photon is incident upon 
        self.ni = ni    #current n
        self.nt  = nt   # new n of next layer
        self.db = db    # distance to boundary 
        self.zt = zt    # depth of next boundary

        
    def hit_boundary(self):
        
        # Calls the Refractive index function to find the position and location of the next boundary

        if abs(self.db*self.mu_t) < abs(self.s_):
            #  Photon is moved to the boundary and the step size is updated
            self.s_ -= self.db*self.mu_t
            #self.layer_no += np.sign(self.vel[-1])
            self.pos[-1] = self.zt # moves the photon to the boundary.
            
    
           
            return True
        
        else:
            return False

    def fresnelReflection(self, n0, n1):
        
        # Takes the two refractive indices of the ambient medium and the first medium and conmputes the 
        # proportion that is refelcted according to Specualar Fresnel Reflectance 

        Rsp = ((n0 - n1) / (n0 + n1))**2


        self.W += -Rsp


    def move(self):
        
        self.pos = self.pos + self.vel*self.s_
        self.s_ = 0

    
        # Finds the refractive index of the initial layer and that of the new layer

    def transmission(self):
        # specular reflection 
        alpha_i = np.arccos(abs(self.vel[-1]))

        # Gathers the refractive indices for the iniital and new medium
        
        alpha_t = np.arcsin(self.ni*np.sin(alpha_i)/self.nt)

        # Check if the photon is reflected if alpha_i is greater than the critical angle

        if self.ni > self.nt and alpha_i > np.arcsin(self.nt/self.ni):
            Ri = 1

        else: 
            # Average if the reflectance for two orthogonal linear poloarisation states because light is assumed to 
            # be randomly polarised
            Ri = 0.5*( (np.sin(alpha_i - alpha_t)**2)/(np.sin(alpha_i + alpha_t)**2) + (np.tan(alpha_i - alpha_t)**2)/(np.tan(alpha_i + alpha_t)**2) )

        # Now check is the photon packet is reflected or transmitted. 
        if self.eta() <= Ri:
            # Reverses the z direction of the photon packet.
            self.vel[-1] = -self.vel[-1]
            
            

        elif self.nt == 1: # i.e the photon is leaving the material.
            # Calls the photon exit function looking to record refletivity, Transmission and unscattered emmission. 
            
            self.photon_exit()
            
        else:
            # The photon is refracted according to Snells Law
            u_x = self.vel[0] * self.ni / self.nt
            u_y = self.vel[1] * self.ni / self.nt
            u_z = np.sign(self.vel[-1]) * np.cos(alpha_t)

            self.vel = np.array([u_x, u_y, u_z])
            

            

        

    def photon_exit(self):
        #print('exiting')
        exit_data = np.hstack([self.pos, self.W])
        #print(exit_data)

        if self.pos[-1] == 0:
            
            self.reflectance = np.vstack([self.reflectance, exit_data])
            #print('reflection', self.reflectance)

        if self.pos[-1] == self.upper_bound:
            
            self.transmittance = np.vstack([self.transmittance, exit_data])
            #print('transmittance', self.transmittance)

        self.W = 0 

        # Unalives photon but the weight and energy is recorded for within th reflection and transmission    
        self.alive = False    



    def absorb(self):
        # Once a photon packet reaches an interaction site a fraction of it is absorbed 
        delW = self.mu_a / self.mu_t * self.W

        # Insert some call to adding the weigh to relative absorption

        self.W -= delW

    def scatter(self):

        
        g = 0.9 # Scattering Anisotropy for most biological tissue 

        if g != 0:
            theta = np.arccos((0.5*g)*(1 + g**2 - ((1 - g**2)/(1 - g + 2*g*self.eta()))**2))

        else:
            theta = np.arccos(2*self.eta()-1)

        phi = 2*np.pi*self.eta()
        
        # Now define new velocity angles 
        u_x, u_y, u_z  = self.vel


        if abs(u_z) > 0.99999:
            u_x_new = np.sin(theta)*np.cos(phi)
            u_y_new = np.sin(theta)*np.sin(phi)
            u_z_new = np.sign(u_z)*np.cos(theta) # Sgn function returns one when the u_z is positive and -1 when negative
            

    

        else:
            
            u_x_new = (np.sin(theta)*(u_x*u_z*np.cos(phi) - u_y*np.sin(phi)))/np.sqrt(1-u_z**2) + (u_x*np.cos(theta))
            u_y_new = (np.sin(theta)*(u_y*u_z*np.cos(phi) + u_x*np.sin(phi)))/np.sqrt(1-u_z**2) + (u_y*np.cos(theta))
            u_z_new = -np.sqrt(1-u_z**2)*(np.sin(theta)*np.cos(phi)) + u_z*np.cos(theta)
            
        self.vel = np.array([u_x_new, u_y_new, u_z_new])
        

 
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





def run(number):
    
    if number % 100 == 0:
        print (number)
    
    two_layer = medium()
    photon = photons(two_layer, weight=1)
    

    # Runs the photon trasnport for Monte Carlo photon trasnport 
    while photon.alive:
        
        photon.stepSize()
        photon.Refractive_index()

        while photon.hit_boundary():
            
            #time.sleep(1)
            #photon.fresnelReflection(two_layer.n0, two_layer.n1)
            photon.transmission()
            photon.Refractive_index()
            
            if photon.W == 0:
                break
            # 
            #if photon.exiting_media():
                #break
        
        photon.move()
        photon.absorb()
        photon.scatter()
        photon.roulette()
       
        
    final_pos = np.concatenate((photon.pos, photon.vel))
    

    return final_pos
    

if __name__ == '__main__':
    t0 = time.time()

    n_cpu = mp.cpu_count()  # = 8 
    numberPhotons = 1000 # Number of photons

    names = ['x','y','z','vx','vy','vz']
    photon_data = np.empty(len(names))

    for i in range(numberPhotons):
        photon_data = np.vstack([photon_data,run(i)])
    
    '''
    # create and configure the process pool
    with mp.Pool(processes=n_cpu) as pool:
        # execute tasks in order
        for result in pool.map(run, range(numberPhotons)):
            photon_data = np.vstack([photon_data, result])
    '''
    
    # process pool is closed automatically

    t1 = time.time()
    
    print ('parallel time: ', t1 - t0)

    df = pd.DataFrame(data=photon_data, columns=names)
    df.drop(0, inplace=True)
    print(df.head())
    print(df.describe())
    
    
    
    plt.figure()
    plt.hist(df['z'], bins=100)
    plt.show()






