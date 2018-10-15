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
colours = [[0, 150/255, 100/255], [225/255, 149/255, 0], [207/255, 0, 48/255], [30/255, 10/255, 171/255],
           'C4', 'C9', 'C6', 'C7', 'C8', 'C5']


def find_avg_counts(cut_data, exp_data):
    RA1 = np.array(cut_data['RA1'])
    DEC1 = np.array(cut_data['DEC1'])
    RA2 = cut_data['RA2']
    DEC2 = np.array(cut_data['DEC2'])
    z1 = np.array(cut_data['z1'])
    z2 = cut_data['z2']
    lens_data = {}
    for cone_radius in RADII:
        lens_data[f"Radius{str(cone_radius)}"] = {}
        for num, (SRA, SDE, SZ) in enumerate(zip(RA2, DEC2, z2)):
            lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}'] = {'Zs': [], 'SNZ': SZ, 'SNRA': SRA,
                                                                         'SNDEC': SDE, 'RAs': [], 'DECs': []}
            if SDE > 1.28 - cone_radius / 60.0:
                h = SDE - (1.28 - cone_radius / 60.0)
            elif SDE < -(1.28 - cone_radius / 60.0):
                h = -(1.28 - cone_radius / 60.0) - SDE
            else:
                h = 0
            theta = 2 * np.arccos(1 - h / (cone_radius / 60.0))
            fraction_outside = 1 / (2 * np.pi) * (theta - np.sin(theta))
            lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['WEIGHT'] = 1.0 - fraction_outside
            indices = [(RA1 - SRA) ** 2 + (DEC1 - SDE) ** 2 <= (cone_radius / 60.0) ** 2]
            lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['Zs'] = z1[indices]
            lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['RAs'] = RA1[indices]
            lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['DECs'] = DEC1[indices]
        print(f"Finished sorting radius {str(cone_radius)}'")
    # with open(f"lenses_weighted.pickle", "rb") as pickle_in:
    #     lens_data = pickle.load(pickle_in)
    limits = exp_data[0]
    avg_counts = {}
    avg_IPs = {}
    for cone_radius in RADII:
        avg_counts[f"Radius{cone_radius}"] = []
        avg_IPs[f"Radius{cone_radius}"] = []
        lenses = lens_data[f"Radius{cone_radius}"]
        print(cone_radius)
        counts = {}
        IPs = {}
        for key, lens in lenses.items():
            counts[key] = np.zeros(51-2)
            IPs[key] = np.zeros(51-2)
            thetas = (((lens['RAs'] - lens["SNRA"]) ** 2 + (lens['DECs'] - lens["SNDEC"]) ** 2) ** 0.5 * np.pi / 180)
            Dparas = thetas * np.interp(lens['Zs'], fine_z, Dpara_fine) * 1000.0 / lens['Zs']
            all_IPs = 1 / Dparas
            for num2 in range(51-2):
                indices2 = [np.logical_and(limits[num2] < lens['Zs'], (lens['Zs'] <= limits[num2 + 1]))]
                counts[key][num2] = np.count_nonzero(indices2)
                IPs[key][num2] = np.sum(all_IPs[indices2])
        for bin in range(51-2):
            total_per_bin = []
            total_IPs = []
            for key in lenses.keys():
                total_per_bin.append(counts[key][bin])
                total_IPs.append(IPs[key][bin])
            avg_counts[f"Radius{cone_radius}"].append(np.mean(total_per_bin))
            avg_IPs[f"Radius{cone_radius}"].append(np.mean(total_IPs))
            for num, _ in enumerate(avg_counts[f"Radius{cone_radius}"]):
                if avg_counts[f"Radius{cone_radius}"][num] == 0:
                    avg_counts[f"Radius{cone_radius}"][num] = np.max(avg_counts[f"Radius{cone_radius}"]) / 100.0
        print(avg_IPs[f"Radius{cone_radius}"])

    pickle_out = open(f"avgs_per_bin.pickle", "wb")
    pickle.dump([avg_counts, avg_IPs], pickle_out)
    pickle_out.close()
    return avg_counts, avg_IPs


def find_expected_weights(cut_data, bins, redo=False, plot=False):
    """Uses the test cones to find the expected number of galaxies per bin, for bins of even redshift.

    Inputs:
     test_cones -- dictionary of data to obtain expected counts from for a variety of cone widths.
     bins -- number of bins along the line of sight to maximum SN comoving distance.
     redo -- boolean that determines whether expected counts are calculated or loaded. Default false.
     plot -- boolean that determines whether expected counts are plotted. Default false.
    """
    max_z = 0.6
    chi_bin_widths, chi_bins, z_bins, z_bin_widths = Convergence.create_z_bins(0.01, max_z, bins)
    limits = np.cumsum(z_bin_widths)

    if redo:
        RA1 = np.array(cut_data['RA1'])
        DEC1 = np.array(cut_data['DEC1'])
        RA2 = cut_data['RA2']
        DEC2 = np.array(cut_data['DEC2'])
        z1 = np.array(cut_data['z1'])
        z2 = cut_data['z2']
        lens_data = {}
        for cone_radius in RADII:

            lens_data[f"Radius{str(cone_radius)}"] = {}
            for num, (SRA, SDE, SZ) in enumerate(zip(RA2, DEC2, z2)):
                lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}'] = {'Zs': [], 'SNZ': SZ, 'SNRA': SRA,
                                                                             'SNDEC': SDE, 'RAs': [], 'DECs': []}

                if SDE > 1.28 - cone_radius / 60.0:
                    h = SDE - (1.28 - cone_radius / 60.0)
                elif SDE < -(1.28 - cone_radius / 60.0):
                    h = -(1.28 - cone_radius / 60.0) - SDE
                else:
                    h = 0

                if h == 0:
                    lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['WEIGHT'] = 1.0
                    indices = [(RA1 - SRA) ** 2 + (DEC1 - SDE) ** 2 <= (cone_radius / 60.0) ** 2]
                    lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['Zs'] = z1[indices]
                    lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['RAs'] = RA1[indices]
                    lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['DECs'] = DEC1[indices]
            print(f"Finished sorting radius {str(cone_radius)}'")

        expected = {}
        for cone_radius in RADII[25:]:
            # IPs = {}
            lens = lens_data[f"Radius{str(cone_radius)}"]
            cumul_tot = np.zeros((len(limits), len(lens)))
            for num1, lim in enumerate(limits):
                for num2, (key, item) in enumerate(lens.items()):
                    # IPs[key] = np.zeros(51 - 2)
                    thetas = (((item['RAs'] - item["SNRA"]) ** 2 + (
                               item['DECs'] - item["SNDEC"]) ** 2) ** 0.5 * np.pi / 180)
                    Dparas = thetas * np.interp(item['Zs'], fine_z, Dpara_fine) * 1000.0 / (1 + np.array(item['Zs']))
                    all_IPs = 1 / Dparas
                    cumul_tot[num1][num2] = np.sum(all_IPs[item['Zs'] < lim])
            expected[f"Radius{str(cone_radius)}"] = np.diff([np.mean(cumul_tot[i][cumul_tot[i] != 0]) for i in range(np.size(cumul_tot, 0))])
            # for index, count in enumerate(expected[f"Radius{str(cone_radius)}"]):
            #     if count == 0:
            #         try:
            #             expected[f"Radius{str(cone_radius)}"][index] = 0.5 * (expected[f"Radius{str(cone_radius)}"]
            #                                                                   [index+1] + expected[
            #                                                                    f"Radius{str(cone_radius)}"][index-1])
            #         except IndexError:
            #             expected[f"Radius{str(cone_radius)}"][index] = 0.5 * (expected[f"Radius{str(cone_radius)}"]
            #                                                                   [index] + expected[
            #                                                                    f"Radius{str(cone_radius)}"][index - 1])

            print(f"Finished radius {str(cone_radius)}'")
        pickle_out = open("expected_IPs_mean.pickle", "wb")
        pickle.dump(expected, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("expected_IPs_mean.pickle", "rb")
        expected = pickle.load(pickle_in)

    if plot:
        for cone_radius in RADII:
            plt.plot([0, 0.6], [0, 0], color=[0.75, 0.75, 0.75], linestyle='--')
            plt.plot((limits[1:]+limits[:-1])/2.0, expected[f"Radius{str(cone_radius)}"], marker='o',
                     markersize=2.5, color=colours[0])
            plt.xlabel('$z$')
            plt.ylabel('Expected Count')
            plt.xlim([0, 0.6])
            plt.show()

    return [limits, expected, chi_bin_widths, chi_bins, z_bins]

# for key, item in lenses.items():
#     print(key)
#     FIS_indices = np.where(item["WEIGHT"] == 1.0)
#     print(len(FIS_indices[0]))
#     for num, w in enumerate(item["WEIGHT"]):
#         if w != 1.0:
#             lenses[key].pop(f"Shell{num+1}")
#     SN_data_fis[key] = {"mu_diff": SN_data["mu_diff"][FIS_indices],
#                         "SNZ": SN_data["SNZ"][FIS_indices],
#                         "SNkappa": SN_data["SNkappa"][FIS_indices],
#                         "SNRA": SN_data["SNRA"][FIS_indices],
#                         "SNDEC": SN_data["SNDEC"][FIS_indices],
#                         "SNMU": SN_data["SNMU"][FIS_indices],
#                         "SNMU_ERR": SN_data["SNMU_ERR"][FIS_indices]}
#
# pickle_out = open(f"random_cones_new_fis.pickle", "wb")
# pickle.dump(lenses, pickle_out)
# pickle_out.close()
#
# print(SN_data_fis["Radius20.0"]['mu_diff'])
# pickle_out = open(f"MICE_SN_data_fis.pickle", "wb")
# pickle.dump(SN_data_fis, pickle_out)
# pickle_out.close()

crit_weight = 0.1
crit_angles = [3.0, 6.0, 12.0, 24.0]
crit_dists = [2.5, 5.0, 7.5, 10.0]
fine_z = np.linspace(0, 0.7, 1001)
Dpara_fine = Convergence.comoving(fine_z)
data, _ = cones.get_data(new_data=False)
lenses = cones.sort_SN_gals(data, redo=False, weighted=True)
exp = cones.find_expected_counts(_, 51)
exp_data = find_expected_weights(data, 51, redo=True, plot=False)

# average_counts = find_avg_counts(data, exp)
with open(f"avgs_per_bin.pickle", "rb") as pickle_in:
    average_counts = pickle.load(pickle_in)
# for r in [30.0]:
#     plt.plot((exp[0][1:] + exp[0][:-1])/2.0, average_counts[1][f"Radius{r}"])
#     plt.show()
with open(f"kappa_weighted.pickle", "rb") as pickle_in:
    kappa = pickle.load(pickle_in)
# print(kappa["Radius30.0"]['Counts'])
zs = []
perps = []
ws = []
lenses_IP = {}
# lenses_IP[crit_dist] = {}
redo_IP = True
if redo_IP:
    for radius in RADII:
        print(radius)
        lenses_IP[f'Radius{radius}'] = {}
        average_per_bin = []
        for key, item in lenses[f'Radius{radius}'].items():
            if lenses[f'Radius{radius}'][key]["WEIGHT"] == 1.0:
                lenses_IP[f'Radius{radius}'][key] = {"SNZ": lenses[f'Radius{radius}'][key]["SNZ"],
                                                     "Zs": lenses[f'Radius{radius}'][key]["Zs"],
                                                     "SNMU": lenses[f'Radius{radius}'][key]["SNMU"],
                                                     "SNMU_ERR": lenses[f'Radius{radius}'][key]["SNMU_ERR"],
                                                     "IPWEIGHT": []}
                exp[0].put(0, 0)
                for z, ra, dec in zip(item['Zs'], item["RAs"], item["DECs"]):
                    bin_num = np.where([np.logical_and(exp[0][i] < z, exp[0][i + 1] > z) for i in range(len(exp[0]) - 1)])[0]
                    # print(z, exp[0][bin_num[0]], exp[0][bin_num[0]+1], bin_num[0])
                    theta = (((ra - item["SNRA"])**2 + (dec - item["SNDEC"])**2)**0.5*np.pi/180)
                    Dpara = np.interp(z, fine_z, Dpara_fine) * 1000.0
                    # limperp = crit_angle/60.0*np.pi/180 * Dpara / (1+z)
                    Dperp = theta * Dpara / (1 + z)
                    # print(theta*180/np.pi*60, (1/Dperp + 0.1 - 1/limperp))
                    # print(z, bin_num[0], exp[0][bin_num[0]])
                    # lenses_IP[f'Radius{radius}'][key]['IPWEIGHT'].append(average_counts[0][f"Radius{radius}"][bin_num[0]]/
                    #                                                      average_counts[1][f"Radius{radius}"][bin_num[0]]/
                    #                                                      Dperp)
                    lenses_IP[f'Radius{radius}'][key]['IPWEIGHT'].append(1.0 / Dperp)
                    # zs.append(z)
                    # perps.append(Dperp)
                    # ws.append(crit_weight*limperp/Dperp)
        print(len(lenses_IP[f'Radius{radius}']))
    # print(lenses_IP.keys())
    # print(lenses_IP[3.0].keys())
    # print(lenses_IP[3.0]["Radius3.0"].keys())
    # print(lenses_IP[3.0]["Radius3.0"]["SN342"].keys())
    # print(lenses_IP[2.5]["Radius3.0"]["SN342"]["IPWEIGHT"], lenses_IP[2.5]["Radius3.0"]["SN343"]["IPWEIGHT"],
    #       lenses_IP[7.5]["Radius3.0"]["SN342"]["IPWEIGHT"], lenses_IP[10.0]["Radius3.0"]["SN342"]["IPWEIGHT"])

    # plt.plot(fine_z, 0.05*np.pi/180.0 * Dpara0)")
    # plt.legend(frameon=0)

    s = plt.scatter(perps, ws, c=zs, cmap='coolwarm')
    # plt.gca().set_yscale('log')
    # plt.gca().set_xscale('log')
    # cbar = plt.colorbar(s)
    # cbar.set_label('$z$')
    # plt.xlim(0.00038, 15)
    # plt.show()

    pickle_out = open(f"lenses_IP_min.pickle", "wb")
    pickle.dump(lenses_IP, pickle_out)
    pickle_out.close()
else:
    pickle_in = open("lenses_IP_min.pickle", "rb")
    lenses_IP = pickle.load(pickle_in)

kappa_impact = cones.find_convergence(lenses_IP, exp_data, redo=True, plot_scatter=False, impact=True)
conv_total_impact = []
for cone_radius in RADII:
    conv_total_impact.append(kappa_impact[f"Radius{str(cone_radius)}"]["Total"])
plt.plot(RADII, conv_total_impact, marker='o', markersize=2, color=colours[3])
plt.plot(RADII, np.zeros(len(RADII)), color=[0.75, 0.75, 0.75], linestyle='--')
plt.xlabel("Cone Radius (arcmin)")
plt.ylabel("Total $\kappa$")
plt.show()
impact = cones.find_correlation(kappa_impact, lenses_IP, plot_radii=True, impact=True)
