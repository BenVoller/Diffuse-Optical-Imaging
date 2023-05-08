import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from material import medium
pd.options.mode.chained_assignment = None  # default='warn'

material = medium()

df = pd.read_csv('testing_data.csv', index_col=0, engine='python')
df_abs = pd.read_csv('testing_absorbtion.csv', index_col=0, engine='python')





#print (df.head())
#print (df.describe())
# Number of photons 
N = material.NumberPhotons
#
#
# Defining Reflectance and Transmission as arrays 
#
#

# Number of grid elements set at 5 - 10% such that it minimises relative error while 
# maintaining good resolution.
N_grid = 200

# size of Grid elements
delta_z = material.depth / N_grid
delta_r = df['r'].max() / N_grid
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


#############
# Defining the coefficient of absorption at each depth for the values of z. 
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


print (u_a_vals)






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



T_da = np.zeros(N_grid)
T_dr = np.zeros(N_grid)
R_da = np.zeros(N_grid)
R_dr = np.zeros(N_grid)



# Diffuse Transmittance with angle
for i in range(len(d_transmittance)):
    weight = d_transmittance.loc[i, 'weight']
    index = d_transmittance.loc[i,'angle_bins']
    T_da[index -1] += weight

# Diffuse Transmittance with radius
for i in range(len(d_transmittance)):
    weight = d_transmittance.loc[i, 'weight']
    index = d_transmittance.loc[i,'r_bins']
    T_dr[index -1] += weight

# Diffuse Reflection with Angle
for i in range(len(d_reflectance)):
    weight = d_reflectance.loc[i, 'weight']
    index = d_reflectance.loc[i,'angle_bins']
    R_da[index -1] += weight

# Diffuse Reflection with radius
for i in range(len(d_reflectance)):
    weight = d_reflectance.loc[i, 'weight']
    index = d_reflectance.loc[i,'r_bins']
    R_dr[index -1] += weight


# Raw R_dr and T_dr are converted to probablities of reimission per unit unit area 
R_dr = R_dr / (N*delta_a)
T_dr = T_dr / (N*delta_a)


# Raw R_da and T_da are converted to reimission per solid angle
R_da = R_da / (N*delta_omega)
T_da = T_da / (N*delta_omega)



#### ________ 
####
# Depth Resolved Fluence

print (df_abs.head())

# Setting an array to count the weights into
A_z = np.zeros(N_grid)

absorbed_zvals = df_abs['z'].values
absorbtion_z_bins = np.digitize(absorbed_zvals, bins_z)

# Finds the Raw values for absorbtion split into bins according to the size of the bin specified by delta_z
for i in range(len(df_abs)):
    weight = df_abs.loc[i, 'weight']
    index = absorbtion_z_bins[i]
    A_z[index - 1] += weight

# Convert into physical quantities by dividing by N del Z
A_z = A_z / N*delta_z # cm^-1
# finds the total weight of the absorption
Total_absorption = np.sum(A_z)
print (Total_absorption)


######### Fluence ######
Fluence_z  = A_z / u_a_vals # dimensionless
print (Fluence_z)





plt.figure()
plt.plot(Z_i_vals, Fluence_z, 'x')
plt.xlabel('z depth')
plt.ylabel('Fluence')

'''
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
'''

plt.show()






    





    

    











    





    

    












