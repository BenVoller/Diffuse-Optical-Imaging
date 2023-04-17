import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

df = pd.read_csv('testing_data.csv', engine='python')

#print (df.head())
#print (df.describe())

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
delta_alpha = np.pi / (2*N_grid)

# Optimal coordinates of the simulated quantities
R_ir = np.arange(N_grid)
alpha_ia = np.arange(N_grid)

bins_r = R_ir *delta_r
bins_alpha = alpha_ia * delta_alpha

print (delta_r *20)

# See extra terms from Taylor exapansion to improve accuracy
R_ir_vals = (R_ir + 0.5)*delta_r
alpha_ia_vals = (alpha_ia + 0.5)*delta_alpha # extra term (1 - 0.5*delta_a*np.cot(delta_a/2))*(np.cot(i+0.5)*delta_a)

d_reflectance = df[df['type'] == 2]
d_transmittance = df[df['type'] == 4]


# Converts the values 
d_reflectance_angle = d_reflectance['angle'].values # converts to numpy array
d_reflectance_r = d_reflectance['r'].values

d_transmittance_angle = d_transmittance['angle'].values
d_transmittance_r = d_transmittance['r'].values

d_refl_angle_bins = np.digitize(d_reflectance_angle, bins_alpha)


print (bins_alpha)
print (d_reflectance_angle)
print (d_refl_angle_bins)




    

    












