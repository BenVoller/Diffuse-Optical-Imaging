from material import *
from Photon_Transport import *
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

        while photon.hit_boundary():
            
            #time.sleep(1)
            #photon.fresnelReflection(two_layer.n0, two_layer.n1)
            photon.transmission()
            photon.Refractive_index()
            
        if photon.W == 0:
            break 
            
        photon.move()
        photon.absorb()
        photon.scatter()
        photon.roulette()
       
        
    final_pos = np.concatenate((photon.pos, photon.vel))
    
    

    return final_pos
    

if __name__ == '__main__':
    t0 = time.time()

    n_cpu = mp.cpu_count()  # = 8 
    numberPhotons = 10000 # Number of photons

    names = ['x','y','z','vx','vy','vz']
    photon_data = np.empty(len(names))

    '''
    # Linear computation for bugfixing
    for i in range(numberPhotons):
        photon_data = np.vstack([photon_data,run(i)])
    '''
    #'''
    # create and configure the process pool
    with mp.Pool(processes=n_cpu) as pool:
        # execute tasks in order
        for result in pool.map(run, range(numberPhotons)):
            photon_data = np.vstack([photon_data, result])
    #'''
    
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




