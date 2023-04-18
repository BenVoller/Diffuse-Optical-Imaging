from material import *

from Photon_Transport import *

print (np.__version__)
print (pd.__version__)
import seaborn as sns
np.random.seed(1234)


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
        #print ('Lets gooo')
        #print (photon.db)
        #print (photon.s_)

        if not photon.is_scattered:
            # Only true if the photon hasnt moved yet and also 
            photon.fresnelReflection() 

        while photon.hit_boundary():
            #time.sleep(2)
            
            #time.sleep(1)
            
            photon.transmission()
            photon.Refractive_index()
            
            #print (photon.pos, photon.vel)
            
            #print (photon.exiting)

            
            
        if photon.W == 0:
            
            return photon.final, absorption
        
        photon.move()
        photon.absorb()

        # Recording absorbtion for plotting Fluence [x,y,z,W,type]
        temp_list = np.hstack((photon.pos, photon.absorbed, photon.absorbed_type))
        absorption = np.vstack([absorption, temp_list])
        photon.scatter()
        photon.roulette()
       



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






if __name__ == '__main__':
    t0 = time.time()

    n_cpu = mp.cpu_count()  # = 8 
    numberPhotons = 50000 # Number of photons

    names = ['x','y','z','vx','vy', 'vz', 'weight','type'] # names for the final pos of a photon 
    absorbed_names = ['x', 'y', 'z', 'weight', 'absorbed_type'] # absorbed data names
    photon_data = np.empty(len(names))
    absorbed_data = np.empty(len(absorbed_names))


    
    #  Linear computation for bugfixing
    for i in range(numberPhotons):

        data, absorbed = run(i)
        

        photon_data = np.vstack([photon_data, data])
        absorbed_data = np.vstack([absorbed_data, absorbed])

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

#    Define dataframes for better file storage
    df = pd.DataFrame(data=photon_data, columns=names)
    df_abs = pd.DataFrame(data=absorbed_data, columns=absorbed_names)

    # sets the type to an integer
    df['type'] = df['type'].astype(int)
    df_abs['absorbed_type'] = df_abs['absorbed_type'].astype(int)

    # Drops the fist row of zeros used to initialise the array
    df.drop(0, inplace=True)
    df_abs.drop(0, inplace=True)


    # drop all the 0 vaued absorbed types as they are relics of initialisation
    df_abs = df_abs[df_abs.absorbed_type != 0]
    
 
    # Separates the unscattered trnasmission from the model. Tu:0, td:1, Ru:2, Rd:3, Ab:4
    

    # Create a new column r that denotes the combined XY distance of the photons
    df['r'] = np.sqrt (df['x']**2 + df['y']**2)
    df['angle'] = np.arccos(abs(df['vz']) / np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2))
    df['solid_angle'] = 2*np.pi*(1 - np.cos(df['angle'])) / (df['vx']**2 + df['vy']**2 + df['vz']**2)

    # Defines the r value useful for fluence and also drops the x and y coords as they are no longer useful.
    df_abs['r'] = np.sqrt(df_abs['x']**2 + df_abs['y']**2)
    df_abs.drop('x', axis=1, inplace=True)
    df_abs.drop('y', axis=1, inplace=True)
    df_abs = df_abs.reindex(columns=['z', 'r', 'weight', 'absorbed_type']) # reorders the dataframe for neatness
    df_abs = df_abs.reset_index(drop=True)  # resets index after dropping the 0 values

    print ('_____')
    print(df_abs.head())
    print ('_____')


    # Saving the Dataframes for use in a different file
    df.to_csv('testing_data.csv')

    df_abs.to_csv('testing_absorbtion.csv')

    print(df.head())
    print(df.describe())



    #%%
     
    reflectance =  df[(df['type'] == 1) | (df['type'] == 2)]
    transmittance = df[(df['type'] == 3) | (df['type'] == 4)]
    #print(transmittance.head())
    #print(reflectance.head())
    
    transmittance_sors = SORS(transmittance, xmin=0, width=0.005)
    print ('trasnmittance_sors')
    print (transmittance_sors.head())
    print (transmittance_sors.describe())



    # Sum of all reflectance and transmission values 
    # Reflectance
    R_tot = np.sum(reflectance['weight'])
    # Transmission 
    T_tot = np.sum(transmittance['weight'])

    # Unscattered transmission
    T_unscattered = df[(df['type'] == 3)]
    T_tot_unscattered = np.sum(T_unscattered['weight'])


    d_transmittance = df[df['type'] == 4]
    

    print('Transmittance:', T_tot/numberPhotons)
    print('Reflectance:', R_tot/numberPhotons)
    print('Unscattered transmittance:', T_tot_unscattered/numberPhotons)








    images = False
    if images == True:
        # Sort the Graphs based on their solid angle. 
        plt.figure()
        plt.title('Reflectance vs Angle')
        plt.xlabel('angle')
        plt.ylabel('weight')
        reflectance = reflectance.sort_values('angle')
        plt.hist(reflectance['angle']/np.pi, weights=reflectance['weight'], bins=20)

        plt.figure()
        plt.title('diffuse Transmission against exit angle ')
        plt.xlabel('angle')
        plt.ylabel('weight')
        d_transmittance = d_transmittance.sort_values('angle')
        plt.hist(d_transmittance['angle']/np.pi, weights=d_transmittance['weight'], bins=20)

        plt.figure()
        plt.title('Transmission against exit angle ')
        plt.xlabel('angle')
        plt.ylabel('weight')
        transmittance = transmittance.sort_values('angle')
        plt.hist(transmittance['angle']/np.pi, weights=transmittance['weight'], bins=20)
        
        plt.figure()
        plt.title('Reflectance vs radius')
        plt.xlabel('radius')
        plt.ylabel('weight')
        reflectance = reflectance.sort_values('r')
        plt.hist(reflectance['r']/np.pi, weights=reflectance['weight'], bins=20)


        plt.figure()
        plt.title(' Transmission against exit radius ')
        plt.xlabel('r')
        plt.ylabel('weight')
        transmittance = transmittance.sort_values('r')
        plt.hist(transmittance['r']/np.pi, weights=transmittance['weight'], bins=20)

        plt.figure()
        plt.title('diffuse Transmission against exit radius ')
        plt.xlabel('r')
        plt.ylabel('weight')
        d_transmittance = d_transmittance.sort_values('r')
        plt.hist(d_transmittance['r']/np.pi, weights=d_transmittance['weight'], bins=20)

        #print (df.head())
        #plt.hist(d_transmittance['weight'], bins=d_transmittance['bins'])
        plt.show()
        
    
        








