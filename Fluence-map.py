import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
from material import medium

data1 = np.load('Fluence_data_10k1.npy')
data2 = np.load('Fluence_data_10k2.npy')
z_data = np.load('Fluence_data_z.npy')
u_a_array = np.load('ua_vals.npy')

data = data1 + data2

keep = False
if keep == True:
    np.save('Fluence_keep', data)

material = medium()

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

cbar = fig.colorbar(cax, ticks=[0, 1])

cbar.ax.set_xticklabels(['Low', 'High'])  # horizontal colorbar



plt.figure()
plt.plot(Z_i_vals[:10], np.log(z_data[:10]), 'x')
plt.plot(Z_i_vals[:10],(Z_i_vals[:10]*-1.74)+ 5.45)
plt.xlabel('z depth')
plt.ylabel('Fluence')
plt.show()


