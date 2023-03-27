from material import *
from Photon_Transport import *
import seaborn as sns
np.random.seed(1234)


def run(number):
    
    if number % 100 == 0:
        print (number)
    
    two_layer = medium()
    photon = photons(two_layer, weight=1)
    

    # Runs the photon trasnport for Monte Carlo photon trasnport 
    while photon.alive:
        
        photon.stepSize()
        photon.Refractive_index()

        if not photon.is_scattered:
            # Only true if the photon hasnt moved yet and also 
            photon.fresnelReflection() 

        while photon.hit_boundary():
            
            #time.sleep(1)
            
            photon.transmission()
            photon.Refractive_index()
            
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

    names = ['x','y','z','weight','type']
    photon_data = np.empty(len(names))

    '''
    #  Linear computation for bugfixing
    for i in range(numberPhotons):
        photon_data = np.vstack([photon_data, run(i)])

    '''
    # create and configure the process pool
    with mp.Pool(processes=n_cpu) as pool:
        # execute tasks in order
        for result in pool.map(run, range(numberPhotons)):
            photon_data = np.vstack([photon_data, result])
    
    
    # process pool is closed automatically

    t1 = time.time()
    
    print ('parallel time: ', t1 - t0)

    df = pd.DataFrame(data=photon_data, columns=names)
    df['type'] = df['type'].astype(int)
    df.drop(0, inplace=True)
    #df.drop(columns=0, inplace=True)
    print(df.head(20))
    #print(df.info())
    print(df.describe())

    # Separates the unscattered trnasmission from the model. Tu:0, td:1, Ru:2, Rd:3, Ab:4
    

    # Create a new column r that denotes the combined XY distance of the photons
    df['r'] = np.sqrt (df['x']**2 + df['y']**2)
    
    reflectance =  df[(df['type'] == 1) | (df['type'] == 2)]
    transmittance = df[(df['type'] == 3) | (df['type'] == 4)]

    print(transmittance.head())
    print(reflectance.head())

    # Sum of all reflectance and transmission values 
    # Refelctance
    R_tot = np.sum(reflectance['weight'])
    # Transission 

    T_tot = np.sum(transmittance['weight'])

    print('Transmittance:', T_tot/numberPhotons)
    print('Reflectance:', R_tot/numberPhotons)
    
    #print (df.head())
    plt.hist(df['z'], bins=100)
    plt.show()
    
 
    








