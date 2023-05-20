import numpy as np
import matplotlib.pyplot as plt 


alpha_ia_vals, R_da_forward, R_ir_vals, R_dr_forward = np.load('Reflectance_data')
alpha_ia_vals, R_da_iso, R_ir_vals, R_dr_iso = np.load('Reflectance_data_iso')

plt.figure()
plt.plot(R_ir_vals, R_dr_forward)
plt.plot(R_ir_vals, R_dr_iso)


plt.figure()
plt.plot(R_ir_vals, ((R_dr_iso - R_dr_forward)/R_dr_forward))


