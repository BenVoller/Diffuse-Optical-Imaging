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
    
    photon = photons(material,
                     inclusion_size=material.inclusion_size, 
                     inclusion_center=material.inclusion_center, 
                     weight=1)

    absorption = np.zeros(4)
    

    # Runs the photon trasnport for Monte Carlo photon transport 
    #print ('NEW PHOTON')
    while photon.alive:
        
        photon.stepSize()
        
        photon.Coefficient_check()
        

        if not photon.is_scattered:
            # Only true if the photon hasnt moved yet and also 
            photon.fresnelReflection() 

        while photon.hit_boundary():
            
            try:
                if photon.faces == 'front' or photon.faces == 'back':
                    photon.transmission_x_plane()
                    print ('x_plane')

                elif photon.faces == 'left' or photon.faces == 'right':
                    photon.transmission_y_plane()
                    print('yplane')
            
            except:
                photon.transmission()


            photon.Coefficient_check()
        
        if photon.W == 0:
            
            return photon.final, absorption
        #print ('*weight',photon.W)
        photon.move()
        
        photon.absorb()
        

        # [z, W, type]
        temp_list = np.hstack((photon.pos[0], photon.pos[-1], photon.absorbed, photon.absorbed_type))
        absorption = np.vstack([absorption, temp_list])

       # print ('weight&', photon.W)
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

    material = medium()
    # Begining time for the simulation
    t0 = time.time()
    n_cpu = mp.cpu_count()  # = 8 
    numberPhotons = material.NumberPhotons # Number of photons

    
    # Number of grid elements set at 5 - 10% such that it minimises relative error while 
    # maintaining good resolution.
    N_grid = 200

    

    # size of Grid elements
    delta_z = material.depth / N_grid
    delta_r = material.depth * 2 / N_grid
    delta_alpha = np.pi / (2*N_grid)



    # Optimal coordinates of the simulated quantities
    R_ir = np.arange(N_grid)
    alpha_ia = np.arange(N_grid)
    Z_i = np.arange(N_grid)
    X_i = np.arange(N_grid) - N_grid/2

    # Define Histogram bins 
    bins_x = X_i * delta_z
    bins_z = Z_i * delta_z
    bins_r = R_ir *delta_r
    bins_alpha = alpha_ia * delta_alpha



    # See extra terms from Taylor exapansion to improve accuracy
    R_ir_vals = (R_ir + 0.5)*delta_r
    alpha_ia_vals = (alpha_ia + 0.5)*delta_alpha # extra term (1 - 0.5*delta_a*np.cot(delta_a/2))*(np.cot(i+0.5)*delta_a)
    Z_i_vals = (Z_i + 0.5)*delta_z
    X_i_vals = (X_i + 0.5)*delta_z

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

    u_a_vals_array = np.ones([len(u_a_vals), len(u_a_vals)])
    for row in range(len(u_a_vals_array)):
        u_a_vals_array[row] *= u_a_vals[row]

    print (u_a_vals_array)


    # empty_listsfor storing weights into 
    diffuse_reflectance = np.zeros([N_grid, N_grid])
    diffuse_transmittance = np.zeros([N_grid, N_grid])
    unscat_reflectance = 0
    unscat_transmittance = 0


    
    
    scattered_absorption = np.zeros([N_grid, N_grid])
    unscattered_absorption = np.zeros([N_grid, N_grid])
    
    absorption_weights = np.zeros([N_grid, N_grid])


    names = ['z','r','angle', 'weight','type']
    photon_data = np.empty(len(names))

   

        # create and configure the process pool
    with mp.Pool(processes=n_cpu) as pool:
        # execute tasks in order
        #for data, absorption in pool.map(run, range(numberPhotons)):
        
         
        #  Linear computation for bugfixing
        for i in range(numberPhotons):
            # The data is in the form  ['x','y','z','vx','vy', 'vz', 'weight','type']
            data, absorption = run(i)
        

            # Assigns a bin number to the data so that the weight can be stored
            x_bin = np.digitize(data['x'], bins_z)
            z_bin = np.digitize(data['z'], bins_z)
            r_bin = np.digitize(data['r'], bins_r)
            angle_bin = np.digitize(data['angle'], bins_alpha)

               
            # Absorption data
            # Assigning bin values to the absorption data corresponding with absorbed z vals
            


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
            


            # # # This splits the absorption into specific scattered or unscattered bins
            
            if absorption.ndim != 1:

                # Absorption scattered or unscattered
                for i in range(len(absorption)):    
                    
                    absorption_x_bin = np.digitize(absorption[i][0], bins_z)
                    absorption_z_bin = np.digitize(absorption[i][1], bins_z)
                    
                    if absorption[i][-1] == 1:
                        # Unscattered Absorption
                        unscattered_absorption[absorption_z_bin-1] += absorption[i][2]
                        # absorption_weights[absorption_z_bin-1][absorption_x_bin-1] += absorption[i][2]
                
                    elif absorption[i][-1] == 2:
                        # Scattered absorption
                        scattered_absorption[absorption_z_bin-1][absorption_x_bin-1] += absorption[i][2]
                        # absorption_weights[absorption_z_bin-1][absorption_x_bin-1] += absorption[i][2]

                
            
            #absorption_weights += (scattered_absorption + unscattered_absor[tion)
            absorption_weights += scattered_absorption
        
            # photon_data = np.vstack([photon_data, data])

        


        
        # process pool is closed automatically

        t1 = time.time()
        
        print ('parallel time: ', t1 - t0)

        r_transmittance = np.sum(diffuse_transmittance, axis=1)
        angle_transmittance = np.sum(diffuse_transmittance, axis=0)

        r_reflectance = np.sum(diffuse_reflectance, axis=1)
        angle_reflectance = np.sum(diffuse_reflectance, axis=0)

        T_tot = np.sum(diffuse_transmittance)
        R_tot = np.sum(diffuse_reflectance)

        print (T_tot/numberPhotons, R_tot/numberPhotons)

        # Raw R_dr and T_dr are converted to probablities of reimission per unit unit area 
        R_dr = r_reflectance / (numberPhotons*delta_a)
        T_dr = r_transmittance / (numberPhotons*delta_a)


        # Raw R_da and T_da are converted to reimission per solid angle
        R_da = angle_reflectance / (numberPhotons*delta_omega)
        T_da = angle_transmittance / (numberPhotons*delta_omega)

        # Convert raw absorption data to physical quantity
        A_z = absorption_weights / numberPhotons * delta_z 
        Total_absorption = np.sum(A_z)


        ### Fluence
        Fluence = A_z / u_a_vals

        np.save('Fluence_data', Fluence)

        print (Fluence)

        Fluence_z = np.sum(Fluence, axis=0)

    np.save('Fluence_data_z', Fluence_z)

    images = True
    if images == True:

        #plt.figure()
        #plt.pcolormesh([Z_i_vals, X_i_vals,], Fluence)
        
        plt.figure()
        plt.ylabel('Diffuse Reflectance $sr^{-1}$')
        plt.xlabel('Exit angle (rad)')
        plt.xticks(np.arange(0, np.pi/2+1, step=(np.pi/10)), ['0','0.1π','0.2π','0.3π','0.4π', '0.5π'])
        plt.plot(alpha_ia_vals, R_da, 'x')

        plt.figure()
        plt.ylabel('Diffuse Transmission $sr^{-1}$')
        plt.xlabel('Exit angle (rad)')
        plt.xticks(np.arange(0, np.pi/2+1, step=(np.pi/10)), ['0','0.1π','0.2π','0.3π','0.4π', '0.5π'])
        plt.plot(alpha_ia_vals, T_da, 'x')
        
        plt.figure()
        plt.plot(Z_i_vals, Fluence_z, 'x')
        plt.xlabel('z depth')
        plt.ylabel('Fluence')


        #print (df.head())
        #plt.hist(d_transmittance['weight'], bins=d_transmittance['bins'])
        plt.show()
        
    

