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
    
    two_layer = medium()
    photon = photons(two_layer, weight=1)
    

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
            
            return photon.final
        
        photon.move()
        photon.absorb()
        photon.scatter()
        photon.roulette()
       
        
    #final_pos = np.concatenate((photon.pos, photon.vel))
    
    
    
    return photon.final
    

if __name__ == '__main__':
    t0 = time.time()

    n_cpu = mp.cpu_count()  # = 8 
    numberPhotons = 1000 # Number of photons

    names = ['x','y','z','vx','vy', 'vz', 'weight','type']
    photon_data = np.empty(len(names))

    
    #  Linear computation for bugfixing
    for i in range(numberPhotons):
        photon_data = np.vstack([photon_data, run(i)])

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
    df['type'] = df['type'].astype(int)
    df.drop(0, inplace=True)
    #df.drop(columns=0, inplace=True)
    
    print(df.head())
    print(df.describe())
    

    with open('0.2_Nrel=1.csv', 'wb') as f:
            np.save(f, df)

    

    # Separates the unscattered trnasmission from the model. Tu:0, td:1, Ru:2, Rd:3, Ab:4
    

    # Create a new column r that denotes the combined XY distance of the photons
    df['r'] = np.sqrt (df['x']**2 + df['y']**2)
    df['angle'] = np.arccos(df['vz'] / np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2))
     
    reflectance =  df[(df['type'] == 1) | (df['type'] == 2)]
    transmittance = df[(df['type'] == 3) | (df['type'] == 4)]

    #print(transmittance.head())
    #print(reflectance.head())

    # Sum of all reflectance and transmission values 
    # Refelctance
    R_tot = np.sum(reflectance['weight'])
    # Transission 

    T_unscattered = df[(df['type']==3)]
    T_tot_unscattered = np.sum(T_unscattered['weight'])

    T_tot = np.sum(transmittance['weight'])

    print('Transmittance:', T_tot/numberPhotons)
    print('Reflectance:', R_tot/numberPhotons)
    print('Unscattered transmittance:', T_tot_unscattered/numberPhotons)

    d_transmittance = df[df['type'] == 4]
    #d_transmittance['bins'] = pd.cut(d_transmittance['r'], 10)
    #print(d_transmittance.head())
    
    #print (df.head())
    #plt.hist(d_transmittance['weight'], bins=d_transmittance['bins'])
    plt.show()
    
 
    








