import MICE
import cones
import pickle
import numpy as np
import Convergence
import matplotlib.pyplot as plt

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
colours = [[0, 150 / 255, 100 / 255], [225 / 255, 149 / 255, 0], [207 / 255, 0, 48 / 255],
           [30 / 255, 10 / 255, 171 / 255],
           'C4', 'C9', 'C6', 'C7', 'C8', 'C5']


def find_expected_weights(data, bins, redo=False, plot=False):
    all_zs = data['z']
    all_RAs = data['RA']
    all_DECs = data['DEC']
    max_z = 1.41
    chi_bin_widths, chi_bins, z_bins, z_bin_widths = Convergence.create_z_bins(0.01, max_z, bins, OM=0.25, OL=0.75,
                                                                               h=0.7)
    limits = np.cumsum(z_bin_widths) + z_bins[0]
    limits = np.insert(limits, 0, 0)
    expected = {}
    pickle_in = open("MICE_SN_data_fis.pickle", "rb")
    SN_data = pickle.load(pickle_in)
    pickle_in = open("random_cones_new_fis.pickle", "rb")
    lens_data = pickle.load(pickle_in)

    if redo:
        for cone_radius in RADII:
            cone_zs = {}
            cone_RAs = {}
            cone_DECs = {}
            for SN_num, key in enumerate(lens_data[f"Radius{cone_radius}"].keys()):
                if key != 'WEIGHT':
                    cone_indices = np.array([], dtype=np.int16)
                    # Get shells from all previous RADII
                    for r in RADII[0:np.argmin(np.abs(RADII - np.array(cone_radius))) + 1]:
                        cone_indices = np.append(cone_indices, lens_data[f"Radius{r}"][key])
                    # Get redshifts, RAs and DECs of all galaxies in each SN cone
                    cone_zs[key] = all_zs[cone_indices]
                    cone_RAs[key] = all_RAs[cone_indices]
                    cone_DECs[key] = all_DECs[cone_indices]
            print(len(cone_zs))

            cumul_tot = np.zeros((len(limits), len(lens_data[f"Radius{cone_radius}"])))
            for num1, lim in enumerate(limits):
                for num2, (key2, cone) in enumerate(cone_zs.items()):
                    # print(SN_num+1)
                    thetas = (((cone_RAs[key2] - SN_data[f"Radius{cone_radius}"]["SNRA"][num2]) ** 2 +
                               (cone_DECs[key2] - SN_data[f"Radius{cone_radius}"]["SNDEC"][num2]) **
                               2) ** 0.5 * np.pi / 180)
                    Dparas = thetas * np.interp(cone_zs[key2], fine_z, Dpara_fine) * 1000.0 / (
                            1 + np.array(cone_zs[key2]))
                    all_IPs = 1 / Dparas
                    cumul_tot[num1][num2] = np.sum(all_IPs[cone_zs[key2] < lim])
            expected[f"Radius{str(cone_radius)}"] = np.diff(np.mean(cumul_tot, 1))

            print(f"Finished radius {str(cone_radius)}'")
        pickle_out = open("MICEexpected_IPs.pickle", "wb")
        pickle.dump(expected, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("MICEexpected_IPs.pickle", "rb")
        expected = pickle.load(pickle_in)

    if plot:
        for cone_radius in RADII:
            plt.plot([0, 0.6], [0, 0], color=[0.75, 0.75, 0.75], linestyle='--')
            plt.plot((limits[1:] + limits[:-1]) / 2.0, expected[f"Radius{str(cone_radius)}"], marker='o',
                     markersize=2.5, color=colours[0])
            plt.xlabel('$z$')
            plt.ylabel('Expected Count')
            plt.xlim([0, 0.6])
            plt.show()

    return [limits, expected, chi_bin_widths, chi_bins, z_bins]


fine_z = np.linspace(0, 1.5, 1001)
Dpara_fine = Convergence.comoving(fine_z)
data = MICE.get_data()
all_zs = data['z']
all_RAs = data['RA']
all_DECs = data['DEC']
exp_data = find_expected_weights(data, 51, redo=False)
pickle_in = open("random_cones_new_fis.pickle", "rb")
lenses = pickle.load(pickle_in)
pickle_in = open("MICE_SN_data_fis.pickle", "rb")
SN_data = pickle.load(pickle_in)
zs = []
perps = []
ws = []
lenses_IP = {}
redo_IP = True
if redo_IP:
    for radius in RADII:
        print(radius)
        cone_zs = {}
        cone_RAs = {}
        cone_DECs = {}
        for SN_num, key in enumerate(lenses[f"Radius{radius}"].keys()):
            if key != 'WEIGHT':
                cone_indices = np.array([], dtype=np.int16)
                # Get shells from all previous RADII
                for r in RADII[0:np.argmin(np.abs(RADII - np.array(radius))) + 1]:
                    cone_indices = np.append(cone_indices, lenses[f"Radius{r}"][key])
                # Get redshifts, RAs and DECs of all galaxies in each SN cone
                cone_zs[key] = all_zs[cone_indices]
                cone_RAs[key] = all_RAs[cone_indices]
                cone_DECs[key] = all_DECs[cone_indices]
        lenses_IP[f'Radius{radius}'] = {}
        average_per_bin = []
        for num, (key, _) in enumerate(cone_zs.items()):
            lenses_IP[f'Radius{radius}'][key] = []
            for z, ra, dec in zip(cone_zs[key], cone_RAs[key], cone_DECs[key]):
                theta = (((ra - SN_data[f"Radius{radius}"]["SNRA"][num]) ** 2 + (dec -
                                                                                 SN_data[f"Radius{radius}"]["SNDEC"][
                                                                                     num]) ** 2) ** 0.5 * np.pi / 180)
                Dpara = np.interp(z, fine_z, Dpara_fine) * 1000.0
                Dperp = theta * Dpara / (1 + z)
                lenses_IP[f'Radius{radius}'][key].append(1.0 / Dperp)

    pickle_out = open(f"MICElenses_IP.pickle", "wb")
    pickle.dump(lenses_IP, pickle_out)
    pickle_out.close()
else:
    pickle_in = open("MICElenses_IP.pickle", "rb")
    lenses_IP = pickle.load(pickle_in)

kappa_impact = cones.find_convergence(lenses_IP, exp_data, redo=False, plot_scatter=True, impact=True)
conv_total_impact = []
for cone_radius in RADII:
    conv_total_impact.append(kappa_impact[f"Radius{str(cone_radius)}"]["Total"])
plt.plot(RADII, conv_total_impact, marker='o', markersize=2, color=colours[3])
plt.plot(RADII, np.zeros(len(RADII)), color=[0.75, 0.75, 0.75], linestyle='--')
plt.xlabel("Cone Radius (arcmin)")
plt.ylabel("Total $\kappa$")
plt.show()
impact = cones.find_correlation(kappa_impact, lenses_IP, plot_radii=True, impact=True)
