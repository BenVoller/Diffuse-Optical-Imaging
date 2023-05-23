import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
from material import medium
np.seterr(divide = 'ignore')

data1 = np.load('Fluence_data_10k1.npy')
data2 = np.load('Fluence_data_10k2.npy')
data3= np.load('Fluence_data_50k1.npz')
data1_5= np.load('Fluence_data_50kdepth1_5.npz')

z_data = np.load('Fluence_data_z.npy')
inclusion_data = np.load('inclusion_data.npz')
inclusion1_5 = np.load('inclusion_data_1.5.npz')
inclusion_07 = np.load('inclusion_data_0_7.npz')
raman_data = np.load('raman_data.npz')
u_a_array = np.load('ua_vals.npy')

fluence_data = data3['a']
absorption_data = data3['b']

# Inclusion data
alpha_vals = inclusion_data['a'] 
inclusion_da = inclusion_data['b']
R_vals = inclusion_data['c']
inclusion_dr = inclusion_data['d']
inclusion_dr_15 = inclusion1_5['d']
inclusion_dr_07 = inclusion_07['d']

raman_a = raman_data['b']
raman_r = raman_data['d']


data = fluence_data
inclusion = inclusion_data
raman = raman_data

keep = True
if keep == True:
    np.savez('Fluence_keep', a=data, b=inclusion, c=raman)

material = medium()

# Number of grid elements set at 5 - 10% such that it minimises relative error while 
    # maintaining good resolution.
N_grid = 100

# size of Grid elements
delta_z = material.depth / N_grid
delta_r = material.depth * 2 / N_grid
delta_alpha = np.pi / (2*N_grid)



# Optimal coordinates of the simulated quantities
R_ir = np.arange(N_grid)
alpha_ia = np.arange(N_grid)
Z_i = np.arange(N_grid)
X_i = np.arange(N_grid) - N_grid/2



# See extra terms from Taylor exapansion to improve accuracy
R_ir_vals = (R_ir + 0.5)*delta_r
alpha_ia_vals = (alpha_ia + 0.5)*delta_alpha # extra term (1 - 0.5*delta_a*np.cot(delta_a/2))*(np.cot(i+0.5)*delta_a)
Z_i_vals = (Z_i + 0.5)*delta_z

X_i_vals = (X_i + 0.5)*delta_z
print (data)

X, Y = np.meshgrid(X_i_vals, Z_i_vals)




fig, ax = plt.subplots()
cax = ax.pcolormesh(X,Y, np.log(data))
ax.set_title('Cube inclusion fluence')
cbar = fig.colorbar(cax)
cbar.ax.set_xticklabels(['Low', 'High'])  # horizontal colorbar

# Absorption plot
fig, ax = plt.subplots()
cax = ax.pcolormesh(X,Y, np.log(absorption_data))
#ax.set_title('Cube inclusion fluence')
cbar = fig.colorbar(cax)
ax.set_xlabel('x direction')
ax.set_ylabel('z direction')
cbar.ax.set_xticklabels(['Low', 'High'])  # horizontal colorbar
'''
plt.figure()
plt.plot(Z_i_vals[:10], np.log(z_data[:10]), 'x')
plt.plot(Z_i_vals[:10],(Z_i_vals[:10]*-1.74)+ 5.45)
plt.xlabel('z depth')
plt.ylabel('Fluence')
'''

plt.figure()
plt.plot(alpha_vals, inclusion_da, 'x')
plt.ylabel('R_alpha')

n = 40
plt.figure()
#plt.plot(R_vals[0:n], inclusion_dr[0:n]/np.sum(inclusion_dr), 'b')
plt.plot(R_vals[0:n], inclusion_dr_15[0:n]/np.sum(inclusion_dr_15), 'g')
plt.plot(R_vals[0:n], inclusion_dr_07[0:n]/np.sum(inclusion_dr_07), 'r')
plt.ylabel('Normalised Raidally resolved reflection')
plt.xlabel('radius [cm]')
plt.legend(['depth=1.5', 'depth=0.7'])



plt.show()