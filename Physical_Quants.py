import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


df = pd.read_csv('testing_data.csv', index_col=0, engine='python')

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

d_reflectance = df[df['type'] == 2].reset_index(drop=True)
d_transmittance = df[df['type'] == 4].reset_index(drop=True)
#d_reflectance.reset_index(inplace=True)
#d_reflectance.drop()



# Converts the values 
d_reflectance_angle = d_reflectance['angle'].values # converts to numpy array
d_reflectance_r = d_reflectance['r'].values

d_transmittance_angle = d_transmittance['angle'].values
d_transmittance_r = d_transmittance['r'].values

# Finds which bin each element is in 
d_refl_angle_bins = np.digitize(d_reflectance_angle, bins_alpha)
d_refl_r_bins = np.digitize(d_reflectance_r, bins_r)
d_trans_angle_bins = np.digitize(d_transmittance_angle, bins_alpha)
d_trans_r_bins = np.digitize(d_transmittance_r, bins_r)

print (bins_r)
print (d_reflectance_r)
print (type(d_refl_r_bins))

d_transmittance['angle_bins'] = d_trans_angle_bins
d_transmittance['r_bins'] = d_trans_r_bins
d_reflectance['angle_bins'] = d_refl_angle_bins
d_reflectance['r_bins'] = d_refl_r_bins

print (d_transmittance.head(10))
print (d_reflectance.head(10))


d_transmittance_angle_array = np.zeros(N_grid)
d_transmittance_r_array = np.zeros(N_grid)
d_reflectance_angle_array = np.zeros(N_grid)
d_reflectance_r_array = np.zeros(N_grid)

weight = d_transmittance['angle_bins'][4]
print ('________________________________________')
print (weight)

# Diffuse Transmittance with angle
for i in range(len(d_transmittance)):
    weight = d_transmittance.loc[i, 'weight']
    index = d_transmittance.loc[i,'angle_bins']
    d_transmittance_angle_array[index -1] += weight

# Diffuse Transmittance with radius
for i in range(len(d_transmittance)):
    weight = d_transmittance.loc[i, 'weight']
    index = d_transmittance.loc[i,'r_bins']
    d_transmittance_r_array[index -1] += weight

# Diffuse Reflection with Angle
for i in range(len(d_reflectance)):
    weight = d_reflectance.loc[i, 'weight']
    index = d_reflectance.loc[i,'angle_bins']
    d_reflectance_angle_array[index -1] += weight

# Diffuse Reflection with radius
for i in range(len(d_reflectance)):
    weight = d_reflectance.loc[i, 'weight']
    index = d_reflectance.loc[i,'r_bins']
    d_reflectance_r_array[index -1] += weight


plt.figure()
plt.plot(alpha_ia_vals, d_reflectance_angle_array)
plt.show()







    





    

    












