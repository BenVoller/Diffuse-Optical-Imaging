import numpy as np
import matplotlib.pyplot as plt 
import multiprocessing as mp
import pandas as pd
import time

from material import *


np.random.seed(1234)

class photons():

    def __init__(self, medium,inclusion_size, inclusion_centre_depth, weight):
        # Defines the initial x,y,z coordinates to be 000 an the cosine 

        # These will record the values used to analyse the validity of the solver
        # Will soon need to also include wavelength.
        self.unsc_reflectance = 0 

        # Wavelength of input light that may become an array when run in practise
        self.wavelength = 1000 #Hz

        # Will trigger once one none boundary scattering event occurs
        self.is_scattered = False

        self.alive = True 
        
        self.pos = np.array([0,0,0], dtype=float)
        self.vel = np.array([0,0,1], dtype=float)

        # Extinction coefficient cm^-1

        self.W = weight 
        
    
        self.layers = medium.layers
        
        
        self.upper_bound = medium.depth
        self.lower_bound = 0

        
        # Defines the planes of the inclusion as a list of dictionaries
        self.inclusion, self.inclusion_layer = medium.inclusion(inclusion_size, inclusion_centre_depth)
        self.inclusion_properties = medium.inclusion_properties
        # Denotes if the photon is within the inclusion
        self.in_inclusion = False
        


        # Psuedo Random Number for the step size of the photon movement
    def eta(self):
        return np.random.random()
    
    
    def stepSize(self):
        
        self.s_ = -np.log(self.eta())

    def Coefficient_check(self):
        # Defines whether the photon packet is within the inclusion
        # saves time in checking for boundaries.
        
        # Calls the original Refractive index function
        
        self.Refractive_index()
        

        if self.in_inclusion:
            self.ni = self.inclusion_properties[1]
            self.mu_a = self.inclusion_properties[2]
            self.mu_a = self.inclusion_properties[3]
            self.g = self.inclusion_properties[4]
        

        inclusion_dist, self.face = medium.find_collision_distance(self,planes=self.inclusion, position=self.pos, velocity=self.vel)
        if inclusion_dist < self.db and not self.exiting:
            '''
            print (self.pos, self.vel, self.W)
            print (self.current_coeffs)
            print (self.exiting)
            print (inclusion_dist, '<', self.db)
            print ('----', self.inclusion_properties)
            '''
            self.db = inclusion_dist
            self.nt = self.inclusion_properties[1]
            
            
        
        
            
                


    def axis_rotation(self):

        pos= self.pos
        vel= self.vel

        if self.face == 'left' or self.face == 'right':
            # Changes the coordinate system for transmission then returns it
            # rotates to zxy
            self.pos = np.array([-pos[2],pos[1],pos[0]], dtype=float)
            self.vel = np.array([-vel[2],vel[1],vel[0]], dtype=float)

            self.transmission()

            self.pos = np.array([pos[2],pos[1],-pos[0]], dtype=float)
            self.vel = np.array([vel[2],vel[1],-vel[0]], dtype=float)


            
        elif self.face == 'back' or self.face == 'front':
            print ('before', self.pos, self.vel)
            # Changes the coordinate system for transmission then returns it
            self.pos = np.array([pos[0],pos[2],-pos[1]], dtype=float)
            self.vel = np.array([vel[0],vel[2],-vel[1]], dtype=float)
            
            self.transmission()

            self.pos = np.array([pos[0],-pos[2],pos[1]], dtype=float)
            self.vel = np.array([vel[0],-vel[2],vel[1]], dtype=float)
            print('after', self.pos, self.vel)
        else:
            print ('It is getting here')
            self.transmission()


        
        



        

    def Refractive_index(self):

        
        # Returns a postive or negative number based on the direction of the photon
        direction = np.sign(self.vel[-1])
        z = self.pos[-1]
        self.exiting = False

        if direction == 0: 
            self.db = 99999999

        
        

        for i in range(-1,len(self.layers)):

            self.current_coeffs = self.layers[i]
            self.layer_index =  i
            self.mu_a = self.layers[i][2]
            self.mu_s = self.layers[i][3]
            self.g = self.layers[i][4]
            self.mu_t = self.mu_a + self.mu_s

            if z < self.layers[i][0] and direction == 1:
               
                
                if z == self.upper_bound:
                    #print ('Exciting here')
                    self.exiting = True
                    
                    

                # Sets the layers based on direction = 1
                
                
                self.ni = self.layers[i-1][1]
                self.nt = self.layers[i][1]
                self.zt = self.layers[i][0]
                self.db = (self.zt - z) / self.vel[-1]
                break

            if z < self.layers[i][0] and direction == -1:
                
                # Sets the layers based on dir = -1
                self.ni = self.layers[i][1]
                self.nt = self.layers[i-1][1]
                self.zt = self.layers[i-1][0]

                # Checking if the photon is exciting
                if z == self.lower_bound:
                    # print('This should have happened')
                    self.exiting = True
                    
                    

                # Checking if the nearest boundary is current position
                # in which case it is set to a the one lower
                # print(self.pos,self.vel, direction,self.db)

                if self.zt == z:
                    # print ('is this triggering')
                    self.ni = self.layers[i-1][1]
                    self.nt = self.layers[i-2][1]
                    self.zt = self.layers[i-2][0]
                    # Accounts that this is immediately moving out of the layer 
                    self.current_layer = self.layers[i-1]

                    self.mu_a = self.layers[i-1][2]
                    self.mu_s = self.layers[i-1][3]
                    self.g = self.layers[i-1][4]
                    self.mu_t = self.mu_a + self.mu_s
                    

                # Sets the distance to the boundary
                self.db = (self.zt - z) / self.vel[-1]
                break

            
    


    def hit_boundary(self):
        
        #print(self.pos, self.vel,(self.s_/self.mu_t),self.zt, self.is_scattered)

        # Calls the Refractive index function to find the position and location of the next boundary

        if abs(self.db*self.mu_t) < abs(self.s_):
            
            #  Photon is moved to the boundary and the step size is updated
            self.s_ -= self.db*self.mu_t
            #self.layer_no += np.sign(self.vel[-1])
            self.pos[-1] = self.zt # moves the photon to the boundary.
            
            return True
        
        elif self.exiting:
            self.transmission()
            return False
        
        else:
            #print ('not hitting')
            return False

    def fresnelReflection(self):
        
        # Takes the two refractive indices of the ambient medium and the first medium and conmputes the 
        # proportion that is refelcted according to Specualar Fresnel Reflectance 
        n0 = self.ni
        n1 = self.nt
        Rsp = ((n0 - n1) / (n0 + n1))**2


        self.W += -Rsp
        self.unsc_reflectance += Rsp


    def move(self):
        
        self.pos = self.pos + self.vel*(self.s_/self.mu_t)
        self.s_ = 0

    
        # Finds the refractive index of the initial layer and that of the new layer

    def transmission(self):

        #print('Transmitting')
        # specular reflection 
        alpha_i = np.arccos(abs(self.vel[-1]))

        # Gathers the refractive indices for the iniital and new medium
        
        alpha_t = np.arcsin(self.ni*np.sin(alpha_i)/self.nt)

        

        # Check if the photon is reflected if alpha_i is greater than the critical angle

        if self.ni > self.nt and alpha_i > np.arcsin(self.nt/self.ni):
            Ri = 1

        elif alpha_i == 0 and alpha_t == 0:
            # Fixes a Runtime error in the Reflection amount
            Ri = 0

        else: 
            # Average if the reflectance for two orthogonal linear poloarisation states because light is assumed to 
            # be randomly polarised
            Ri = 0.5*( (np.sin(alpha_i - alpha_t)**2)/(np.sin(alpha_i + alpha_t)**2) + (np.tan(alpha_i - alpha_t)**2)/(np.tan(alpha_i + alpha_t)**2) )

        # Now check is the photon packet is reflected or transmitted. 
        if self.eta() <= Ri:
            # Reverses the z direction of the photon packet.
            self.vel[-1] = -self.vel[-1]
            
            
        #####I think this may be redundant
        elif self.exiting: # i.e the photon is leaving the material.
            # Calls the photon exit function looking to record refletivity, Transmission and unscattered emmission. 
            #print ('HERE')
            self.photon_exit()
            
          
        else:
            # The photon is refracted according to Snells Law
            u_x = np.float(self.vel[0] * self.ni / self.nt)
            u_y = np.float(self.vel[1] * self.ni / self.nt)
            u_z = np.float(np.sign(self.vel[-1]) * np.cos(alpha_t))

            self.vel = np.array([u_x, u_y, u_z])
            

    def transmission_y_plane(self):
        alpha_i = np.arccos(abs(self.vel[0]))
        alpha_t = np.arcsin(self.ni*np.sin(alpha_i)/self.nt)

        # Check if the photon is reflected if alpha_i is greater than the critical angle
        if self.ni > self.nt and alpha_i > np.arcsin(self.nt/self.ni):
            Ri = 1
        elif alpha_i == 0 and alpha_t == 0:
            # Fixes a Runtime error in the Reflection amount
            Ri = 0

        else: 
            # Average if the reflectance for two orthogonal linear poloarisation states because light is assumed to 
            # be randomly polarised
            Ri = 0.5*( (np.sin(alpha_i - alpha_t)**2)/(np.sin(alpha_i + alpha_t)**2) + (np.tan(alpha_i - alpha_t)**2)/(np.tan(alpha_i + alpha_t)**2) )

        # Now check is the photon packet is reflected or transmitted. 
        if self.eta() <= Ri:
            # Reverses the z direction of the photon packet.
            self.vel[0] = -self.vel[0]
          
        else:
            # The photon is refracted according to Snells Law
            u_x = np.float(np.sign(self.vel[0]) * np.cos(alpha_t))
            u_y = np.float(self.vel[1] * self.ni / self.nt)
            u_z = np.float(self.vel[2] * self.ni / self.nt)

            self.vel = np.array([u_x, u_y, u_z])


    def transmission_x_plane(self):
        alpha_i = np.arccos(abs(self.vel[1]))
        alpha_t = np.arcsin(self.ni*np.sin(alpha_i)/self.nt)

        # Check if the photon is reflected if alpha_i is greater than the critical angle
        if self.ni > self.nt and alpha_i > np.arcsin(self.nt/self.ni):
            Ri = 1
        elif alpha_i == 0 and alpha_t == 0:
            # Fixes a Runtime error in the Reflection amount
            Ri = 0

        else: 
            # Average if the reflectance for two orthogonal linear poloarisation states because light is assumed to 
            # be randomly polarised
            Ri = 0.5*( (np.sin(alpha_i - alpha_t)**2)/(np.sin(alpha_i + alpha_t)**2) + (np.tan(alpha_i - alpha_t)**2)/(np.tan(alpha_i + alpha_t)**2) )

        # Now check is the photon packet is reflected or transmitted. 
        if self.eta() <= Ri:
            # Reverses the z direction of the photon packet.
            self.vel[1] = -self.vel[1]
          
        else:
            # The photon is refracted according to Snells Law
            u_x =  np.float(self.vel[0] * self.ni / self.nt)
            u_y = np.float(np.sign(self.vel[1]) * np.cos(alpha_t))
            u_z = np.float(self.vel[2] * self.ni / self.nt)

            self.vel = np.array([u_x, u_y, u_z])



        

    def photon_exit(self):
        #print('exiting')
        #exit_data = np.hstack([self.pos, self.W])
        #print(exit_data)

        if self.pos[-1] == 0 and not self.is_scattered:
            exit_type = 1 #Ru
            print ('Here')


        elif self.pos[-1] == 0 and self.is_scattered:   
            exit_type = 2 # Rd'
            #print('reflection', self.reflectance)

        elif self.pos[-1] == self.upper_bound and not self.is_scattered:
            
            exit_type = 3, # Tu
            


        elif self.pos[-1] == self.upper_bound and self.is_scattered:
            exit_type = 4 # Td
            #print('transmittance', self.transmittance)
        
        
        try:
            
            self.pos.astype(float)
            self.W = np.float(self.W)
            r = np.sqrt(self.pos[0]**2 + self.pos[1]**2)
            angle = np.arccos(abs(self.vel[-1]) / np.sqrt(self.vel[0]**2 + self.vel[1]**2 + self.vel[2]**2))
            #self.final = np.hstack((self.pos[0], r, angle,  self.W, exit_type))
            self.final = {'z':self.pos[2],
                        'r':r,
                        'angle':angle,
                        'W':self.W,
                        'exit_type':exit_type}
            self.W = 0 

        except:
            print('Error', self.pos, self.vel, self.W)



        # Unalives photon but the weight and energy is recorded for within th reflection and transmission    
        self.alive = False    



    def absorb(self):
        # Once a photon packet reaches an interaction site a fraction of it is absorbed 
        delW = (self.mu_a / self.mu_t) * self.W
        
        
        # Insert some call to adding the weigh to relative absorption
        if self.is_scattered:
            # 2 refers to scattered absorption
            self.absorbed_type = 2

        elif not self.is_scattered:
            # 1 refers to unscattered absoption
            self.absorbed_type = 1
        
        self.absorbed = delW
        self.W -= delW



    def scatter(self):

        self.is_scattered = True
        
        g = self.g # Scattering Anisotropy for most biological tissue 

        if g != 0:
            theta = np.arccos((1/(2*g))*(1 + g**2 - ((1 - g**2)/(1 - g + 2*g*self.eta()))**2))

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
                self.W += m*self.W
            else:
                self.final = {'z':self.pos[2],
                              'r':0,
                              'angle':0,
                              'W':self.W,
                              'exit_type': 5} # 4 corresponds to Absorbed Ab
                self.W = 0
                self.alive = False
                



    def photon_dead(self):
        # Currently the only mechanism for photon termination is roulette, include boundaries soon.
        pass


    def raman_shift(self):
        # Raman scatters the wavelength of the tissue based on a low random chance 
        # and also a sampling of the makeup of the tissue being investigated.

        self.wavelength 



