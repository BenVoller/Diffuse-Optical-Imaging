import numpy as np
import matplotlib.pyplot as plt 

data_forward= np.load('Reflectance_data_forward.npz')
data_iso = np.load('Reflectance_data_iso.npz')
data_fluence = np.load('Fluence_data_z.npz')
data_fluence1_37 = np.load('Fluence_data_z_1.37.npz')



R_ir_vals = data_forward['c'][:50]
R_dr_forward= data_forward['d'][:50]
R_dr_iso = data_iso['d'][:50]

Z_i_vals = data_fluence['a']
fluence_z = data_fluence['b']
fluence_z137 = data_fluence1_37['b']


plt.figure()
plt.plot(R_ir_vals, np.log(R_dr_forward), 'xb')
plt.plot(R_ir_vals, np.log(R_dr_iso), 'xr')



plt.figure()
plt.plot(R_ir_vals, ((R_dr_iso - R_dr_forward)/R_dr_forward), 'x')

n=40
plt.figure()
plt.plot(Z_i_vals[:n], np.log(fluence_z[:n]), 'xb')
plt.plot(Z_i_vals[:n], np.log(fluence_z137[:n]), 'xr')
plt.xlabel('Z Depth [cm]')
plt.ylabel('Fluence [-]')




plt.show()


