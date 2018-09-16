from astropy.io import fits
import Convergence
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp

c = 2.998E5  # km/s

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
colours = [[0, 150/255, 100/255], [225/255, 149/255, 0], [207/255, 0, 48/255], 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5']


def b_comoving_integrand(a_val, OM=0.27, OL=0.73):
    """Numerical integration of get_h_inv to create an array of comoving values.

    Inputs:
     a_val -- scalefactor value.
     OM -- matter density parameter. Defaults to 0.27.
     OL -- dark energy density parameter. Defaults to 0.73.
    """
    OK = 1 - OM - OL
    return 1 / np.sqrt(a_val * OM + a_val ** 2 * OK + a_val ** 4 * OL)


def b_comoving(z_lo, z_hi, OM=0.27, OL=0.73, n=1001, h=0.738):
    """Numerical integration of b_comoving_integrand to create an array of comoving values. Uses start and end redshift
    as opposed to an array of z values.

    Inputs:
     z_lo -- start redshift.
     z_hi -- end redshift.
     OM -- matter density parameter. Defaults to 0.27.
     OL -- dark energy density parameter. Defaults to 0.73.
     n -- number of integration steps. Defaults to 1001.
    """
    vecIntegrand = np.vectorize(b_comoving_integrand)
    a1 = 1 / (1 + z_hi)  # backwards in a
    a2 = 1 / (1 + z_lo)
    a_arr = np.linspace(a1, a2, n)
    integrands = vecIntegrand(a_arr, OM, OL)
    comoving_coord = sp.cumtrapz(integrands, x=a_arr, initial=0)
    H0 = 1000 * 100 * h  # km/s/Gpc
    return comoving_coord * c / H0


def create_z_bins(z_lo, z_hi, num_bins, plot=False, OM=0.27, OL=0.73, h=0.738):
    """Takes a line sight from z_lo to z_hi and divides it into bins even in redshift.

    Inputs:
     z_lo -- beginning redshift.
     z_hi -- end redshift.
     num_bins -- number of bins to create.
     plot -- boolean to create plot of chi versus z with bins. Defaults to False.
    """
    z_values = np.linspace(z_lo, z_hi, num_bins * 2 - 1)
    z_bin_edges = z_values[0::2]
    z_widths = z_bin_edges[1:] - z_bin_edges[:-1]
    zs = z_values[1::2]

    chi_values = np.linspace(0, 0, len(z_values))
    for k in range(len(z_values)):
        chi = b_comoving(z_lo, z_values[k], OM=OM, OL=OL, h=h)
        chi_values[k] = chi[-1]

    chi_bin_edges = chi_values[0::2]
    chis = chi_values[1::2]
    chi_widths = chi_bin_edges[1:] - chi_bin_edges[:-1]

    if plot:
        plt.plot([z_bin_edges, z_bin_edges], [chi_bin_edges[0], chi_bin_edges[-1]], color=[0.75, 0.75, 0.75],
                 linestyle='-', linewidth=0.8)
        plt.plot([z_lo, z_hi], [chi_bin_edges, chi_bin_edges], color=[0.75, 0.75, 0.75], linestyle='-', linewidth=0.8)
        plt.plot(np.linspace(z_lo, z_hi, 1001), b_comoving(z_lo, z_hi, OM=OM, OL=OL, h=h))
        plt.plot(zs, chis, linestyle='', marker='o', markersize=3)
        plt.xlabel(' $z$')

        plt.ylabel('$R_0\chi$ (Gpc)')
        plt.axis([0, z_hi, 0, chis[-1] + chi_widths[-1] / 2])
        plt.show()

    return chi_widths, chis, zs, z_widths


max_z = 1.41
bins = 111
chi_bin_widths, chi_bins, z_bins, z_bin_widths = create_z_bins(0.01, max_z, bins, OM=0.25, OL=0.75, h=0.7)
z_limits = np.cumsum(z_bin_widths) + z_bins[0]
chi_limits = np.zeros(len(z_limits))
Vs = {}
for n, lim in enumerate(z_limits):
    # Comoving distance to bin limit (height of cone)
    chi_limits[n] = b_comoving(0, lim, OM=0.25, OL=0.75, h=0.7)[-1]*1000  #[-1] selects final chi value from
                                                                                      #integral, *1000 to get in Mpc
for cone_radius in RADII:
    Rs = np.zeros(len(z_limits))
    Vs[f"Radius{cone_radius}"] = np.zeros(len(z_limits))
    for n, lim in enumerate(chi_limits):
        theta_rad = cone_radius/60 * np.pi/180.0
        # Perpendicular distance (base radius of cone)
        Rs[n] = theta_rad * lim
        if n == 0:
            Vs[f"Radius{cone_radius}"][0] = np.pi / 3.0 * Rs[0] ** 2 * chi_limits[0]  # Actual cone
        else:
            Vs[f"Radius{cone_radius}"][n] = np.pi / 3.0 * (Rs[n] ** 2 * chi_limits[n] - Rs[n-1] ** 2 * chi_limits[n-1])
                                                                                      # Conical segment

pickle_in = open(f"MICEexpected.pickle", "rb")
expected_counts = pickle.load(pickle_in)

for cone_radius in RADII:
    plt.plot(chi_limits, expected_counts[f"Radius{cone_radius}"] / Vs[f"Radius{cone_radius}"])
plt.xlabel("$\chi$ (Mpc)")
plt.ylabel("$\\bar{n}$ (Mpc$^{-3}$)")
plt.show()

pickle_in = open(f"big_cone.pickle", "rb")
big_cone = pickle.load(pickle_in)
z_limits = np.insert(z_limits, 0, 0)
expected_counts = {f"Radius108.0": []}
for num1 in range(len(z_limits) - 1):
    expected_counts[f"Radius108.0"].append((np.count_nonzero(np.logical_and(big_cone['Zs'] > z_limits[num1],
                                                                            big_cone['Zs'] < z_limits[num1 + 1]))
                                            / 5.0))  # Made 5 cones, so take average

Rs = np.zeros(len(z_limits) - 1)
Vs[f"Radius108.0"] = np.zeros(len(Rs))
for n, lim in enumerate(chi_limits):
    theta_rad = 1.8/60.0 * np.pi / 180.0
    # Perpendicular distance (base radius of cone)
    Rs[n] = theta_rad * lim
    if n == 0:
        Vs[f"Radius108.0"][0] = np.pi / 3.0 * Rs[0] ** 2 * chi_limits[0]  # Actual cone
    else:
        Vs[f"Radius108.0"][n] = np.pi / 3.0 * (Rs[n] ** 2 * chi_limits[n] - Rs[n - 1] ** 2 * chi_limits[n - 1])
        # Conical segment


plt.plot(z_limits[1:], expected_counts[f"Radius108.0"] / Vs[f"Radius108.0"], color=colours[0])
plt.xlabel("$z$")
# plt.ylim([0.001186, 0.0089])
plt.ylabel("$\\bar{n}$ (Mpc$^{-3}$)")
plt.show()

# with fits.open('MICEsim5.fits') as hdul1:
#     z = hdul1[1].data['z']
#     ID = hdul1[1].data['id']
#
# ID = set(np.array(ID)[[z >= 0.01]])
# z = np.array(z)[[z >= 0.01]]

# data = {}
# for rad in RADII:
#     data[f'Radius{rad}'] = {}
#     for j, (key, SN) in enumerate(kappa[f"Radius{rad}"].items()):
#         if j < 1500:
#             cone_IDs = np.array([], dtype=np.int16)
#             for r in RADII[0:np.argmin(np.abs(RADII - np.array(rad)))]:
#                 cone_IDs = np.append(cone_IDs, lens_data[f"Radius{r}"][f"Shell{j+1}"])
#             gal_zs[key] = alldata['z'][cone_IDs]
#             data[f'Radius{rad}'][key] = {"Zs": gal_zs[key], "SNZ": SN_z[j], "SNMU": SN_mu[j],
#                                          "SNMU_ERR": SN_mu_err[j], "WEIGHT": lens_data[f"Radius{rad}"]['WEIGHT'][j]}
