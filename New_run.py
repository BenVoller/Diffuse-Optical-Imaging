import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import time 
import multiprocessing as mp


from material import medium
from Photon_Transport import photons

def run(number):
    
    if number % 100 == 0:
        #time.sleep(3)
        print (number)
    
    material = medium()
    photon = photons(material, weight=1)

    absorption = np.zeros(5)
    

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
            
            return photon.final, absorption
        
        photon.move()
        photon.absorb()

        temp_list = np.hstack((photon.pos, photon.absorbed, photon.absorbed_type))
        absorption = np.vstack([absorption, temp_list])

        photon.scatter()
        photon.roulette()
       
        
    #final_pos = np.concatenate((photon.pos, photon.vel))
    
    
    
    return photon.final, absorption


def SORS(df, xmin=0, width=0.001, r=0):
    '''
    df takes the data from a monte carlo simulation
    
    This funciton subsections the output data by limiting the area 
    in which light can be absorbed hence acting like a emmiter and detector 
    with xmin and ymin determining the distance from the detector and r denoting
    a circular detector'''

    df = df[(df['x'] > xmin) & (df['x'] < (xmin + width) )]
    df = df[(df['y'] > (0 - width/2)) & (df['y'] < 0 + width/2)]

    return df




# ---------------------------------------
# Running code Below
# ---------------------------------------



if __name__ == '__main__':

    # Begining time for the simulation
    t0 = time.time()
    n_cpu = mp.cpu_count()  # = 8 
    numberPhotons = 1000 # Number of photons

    
    # Number of grid elements set at 5 - 10% such that it minimises relative error while 
    # maintaining good resolution.
    N_grid = 20

    material = medium()

    # size of Grid elements
    delta_z = material.depth / N_grid
    delta_r = material.depth * 2 / N_grid
    delta_alpha = np.pi / (2*N_grid)



    # Optimal coordinates of the simulated quantities
    R_ir = np.arange(N_grid)
    alpha_ia = np.arange(N_grid)
    Z_i = np.arange(N_grid)

    # Define Histogram bins 
    bins_z = Z_i * delta_z
    bins_r = R_ir *delta_r
    bins_alpha = alpha_ia * delta_alpha



    # See extra terms from Taylor exapansion to improve accuracy
    R_ir_vals = (R_ir + 0.5)*delta_r
    alpha_ia_vals = (alpha_ia + 0.5)*delta_alpha # extra term (1 - 0.5*delta_a*np.cot(delta_a/2))*(np.cot(i+0.5)*delta_a)
    Z_i_vals = (Z_i + 0.5)*delta_z

    # area and solid angle
    delta_a = 2*np.pi*(R_ir_vals) * delta_r # cm^2
    delta_omega = 4*np.pi*np.sin(alpha_ia_vals)*np.sin(delta_alpha/2) # sr

    # Imports the layers that are in the material in question avoiding exta 
    # layers used to preserve the code. 
    layers = material.layers_important

    layer_depths = []
    # finding the bins that each layer represents
    for index in layers:
        layer = layers[index]
        depth = layer[0]
        layer_depths.append(depth)
        

    u_a_bins = np.digitize(bins_z, layer_depths)
    print (u_a_bins)

    u_a_vals = []
    for i in range(len(u_a_bins)):
        u_a_vals.append(layers[u_a_bins[i]][2])

    


    # empty_listsfor storing weights into 
    diffuse_reflectance = np.zeros([N_grid, N_grid])
    diffuse_transmittance = np.zeros([N_grid, N_grid])
    unscat_reflectance = 0
    unscat_transmittance = 0

    Absorbtion = np.zeros(N_grid)


    

    names = ['z','r','angle', 'weight','type']
    photon_data = np.empty(len(names))

    
    #  Linear computation for bugfixing
    for i in range(numberPhotons):
        # The data is in the form  ['x','y','z','vx','vy', 'vz', 'weight','type']
        data, absorbtion = run(i)

        # Assigns a bin number to the data so that the weight can be stored

        
        z_bin = np.digitize(data['z'], bins_z)
        r_bin = np.digitize(data['r'], bins_r)
        angle_bin = np.digitize(data['r'], bins_alpha)


        if data['exit_type'] == 1:
            # Unscattered Refelctance
            unscat_reflectance += data['W']


        elif data['exit_type'] == 2:
            # Diffuse Reflectance
            diffuse_reflectance[r_bin-1][angle_bin-1] += data['W']

        elif data['exit_type'] == 3:
            # Unscattered Transmission
            unscat_transmittance += data['W']


        elif data['exit_type'] == 4:
            # Diffuse Transmittance
            diffuse_transmittance[r_bin-1][angle_bin-1] += data['W']
        


        # photon_data = np.vstack([photon_data, data])

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

    r_transmittance = np.sum(diffuse_transmittance, axis=0)
    angle_transmittance = np.sum(diffuse_transmittance, axis=1)

    r_reflectance = np.sum(diffuse_reflectance, axis=0)
    angle_reflectance = np.sum(diffuse_reflectance, axis=1)

    T_tot = np.sum(diffuse_transmittance)
    R_tot = np.sum(diffuse_reflectance)

    print (T_tot/numberPhotons, R_tot/numberPhotons)

    

