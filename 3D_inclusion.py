import numpy as np 
import pandas as pd 

class Material_3D():

    def __init__(self, medium, inclusion=0, inclusion_bounds=0, no_pixels=1000):

        '''
        medium: defines the [n, mu_a, mu_t, g]
        inclusion bounds:  should include a max and min x,y,z
        to define the position but also the 3 above coefficients
        [n, mu_a, mu_t, g]
        '''
        
        x_size, y_size, z_size = medium['size'][-3:]

        #inclusion size
        x0, y0, z0, x1, y1, z1 = (inclusion['size'] * no_pixels).astype(int)

        
        # Defines the refractive index of the ambient medium
        self.n_amb = 1

        # Grid elements for the medium, each one of these will be a key to a dictionary
        # holding all the information of the material.
        self.grid = np.ones([int(no_pixels*x_size), int(no_pixels*y_size), int(no_pixels*z_size)])
        print (np.shape(self.grid))
        self.inclusion = np.ones([x1-x0, y1-y0, z1-z0]) * 2
        print (np.shape(self.inclusion))

        self.grid[x0:x1, y0:y1, z0:z1] = self.inclusion

        print(self.grid[253][253][103])

        self.unit_sq = no_pixels

        self.mediums = {1:medium,
                        2:inclusion}


    def access_refractive_index(self, pos):
        '''Rounds the position down and turns it to a n integer so that it can index the
        material for its refractive index key'''
        pos1 = self.unit_sq * pos
        rounded_pos = np.floor(pos1).astype(int)

        # Key for the material information 
        key = self.grid[rounded_pos[0],rounded_pos[1], rounded_pos[2]]
        print (key)

        print (self.mediums[key]['a'])
        

    def get_refractive_index(self, pos, vel, step):
        '''Takes the position, velocity and unitless step size of the photon
        returns the current refractive index and the refractive index'''

        




    def distance_to_boundary(self):
        pass

    def hit_boundary(self):
        pass


# Ultimate bounds of the material
x_min = 0 
y_min = 0
z_min = 0
x_depth = 0.5
y_depth = 0.5
z_depth = 0.2

# n = refractive index, a = absorbtion coefficient, s = scattering coefficient cm^-1
medium = {'size':np.array([x_min, y_min, z_min, x_depth, y_depth, z_depth]),
          'n':1,
          'a':10,
          's':90} 

inclusion = {'size':np.array([0.1, 0.1, 0.1, 0.15, 0.15, 0.15]),
             'n':1.5, 
             'a':10,
             's':90}


tester = Material_3D(medium, inclusion, no_pixels=1000)
Material_3D.access_refractive_index(tester, pos=np.array([0.11,0.12, 0.13]))
