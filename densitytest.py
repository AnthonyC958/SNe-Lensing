import pickle
import Convergence
import matplotlib.pyplot as plt
import itertools as itt
from astropy.io import fits
import numpy as np

RADII = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
         0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0,
         2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
         4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5,
         5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25,
         7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0,
         9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75,
         11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5,
         12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.5,
         15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0,
         18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0,
         24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]

# with fits.open('MICEsim5.fits') as hdul1:
#     z = hdul1[1].data['z']
#     ID = hdul1[1].data['id']
#
# ID = set(np.array(ID)[[z >= 0.01]])
# z = np.array(z)[[z >= 0.01]]

pickle_in = open(f"MICEkappa_weighted.pickle", "rb")
kappa = pickle.load(pickle_in)
print(kappa["Radius30.0"].keys())

max_z = 1.5
bins = 111
chi_bin_widths, chi_bins, z_bins, z_bin_widths = Convergence.create_z_bins(0.01, max_z, bins, OM=0.27, OL=0.73, h=0.738)
z_limits = np.cumsum(z_bin_widths) + z_bins[0]
chi_limits = np.zeros(len(z_limits))
Vs = {}
for n, lim in enumerate(z_limits):
    # Comoving distance to bin limit (height of cone)
    chi_limits[n] = Convergence.b_comoving(0, lim, OM=0.25, OL=0.75, h=0.7)[-1]*1000
for cone_radius in RADII:
    Rs = np.zeros(len(z_limits))
    Vs[f"Radius{cone_radius}"] = np.zeros(len(z_limits))
    for n, lim in enumerate(chi_limits):
        theta_rad = cone_radius/60 * np.pi/180.0
        # Perpendicular distance (base radius of cone)
        Rs[n] = theta_rad * lim
        if n == 0:
            Vs[f"Radius{cone_radius}"][0] = np.pi / 3.0 * Rs[0] ** 2 * chi_limits[0]
        else:
            Vs[f"Radius{cone_radius}"][n] = np.pi / 3.0 * (Rs[n] ** 2 * chi_limits[n] - Rs[n-1] ** 2 * chi_limits[n-1])

pickle_in = open(f"MICEexpected.pickle", "rb")
exp = pickle.load(pickle_in)

for cone_radius in RADII:
    plt.plot(chi_limits, exp[f"Radius{cone_radius}"]/Vs[f"Radius{cone_radius}"])
plt.xlabel("$\chi$ (Mpc)")
plt.ylabel("$\\bar{n}$ (Mpc$^{-3}$)")
plt.show()

data = {}
for rad in RADII:
    data[f'Radius{rad}'] = {}
    for j, (key, SN) in enumerate(kappa[f"Radius{rad}"].items()):
        if j < 1500:
            cone_IDs = np.array([], dtype=np.int16)
            for r in RADII[0:np.argmin(np.abs(RADII - np.array(rad)))]:
                cone_IDs = np.append(cone_IDs, lens_data[f"Radius{r}"][f"Shell{j+1}"])
            gal_zs[key] = alldata['z'][cone_IDs]
            data[f'Radius{rad}'][key] = {"Zs": gal_zs[key], "SNZ": SN_z[j], "SNMU": SN_mu[j],
                                         "SNMU_ERR": SN_mu_err[j], "WEIGHT": lens_data[f"Radius{rad}"]['WEIGHT'][j]}
