import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

df = pd.read_csv('testing_data.csv', engine='python')

print (df.head())
print (df.describe())

#
#
# Defining Reflectance and Transmission as arrays 
#
#

# Number of grid elements set at 5 - 10% such that it minimises relative error while 
# maintaining good resolution.
N_grid = 20 

# size of Grid elements
delta_z = df['z'].max() / N_grid
delta_r = df['r'].max() / N_grid
delta_a = np.pi / (2*N_grid)

# Optimal coordinates of the simulated quantities
R_ir = np.arange(N_grid)
alpha_ia = np.arange(N_grid)


print (delta_r *20)

# See extra terms from Taylor exapansion to improve accuracy
R_ir = (R_ir + 0.5)*delta_r
alpha_ia = (alpha_ia + 0.5)*delta_a # extra term (1 - 0.5*delta_a*np.cot(delta_a/2))*(np.cot(i+0.5)*delta_a)


Reflectance = np.zeros(shape=(N_grid,N_grid))
Transmittance = np.zeros(shape=(N_grid, N_grid))













