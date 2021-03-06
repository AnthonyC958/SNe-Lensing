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


def find_expected_weights(cut_data, bins, redo=False, plot=False):
    max_z = 1.41
    chi_bin_widths, chi_bins, z_bins, z_bin_widths = Convergence.create_z_bins(0.01, max_z, bins, OM=0.25, OL=0.75,
                                                                               h=0.7)
    limits = np.cumsum(z_bin_widths) + z_bins[0]
    limits = np.insert(limits, 0, 0)

    if redo:
        RA1 = np.array(cut_data['RA1'])
        DEC1 = np.array(cut_data['DEC1'])
        RA2 = cut_data['RA2']
        DEC2 = np.array(cut_data['DEC2'])
        z1 = np.array(cut_data['z1'])
        z2 = cut_data['z2']
        lens_data = {}
        for cone_radius in RADII[29::2]:
            lens_data[f"Radius{str(cone_radius)}"] = {}
            for num, (SRA, SDE, SZ) in enumerate(zip(RA2, DEC2, z2)):
                lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}'] = {'Zs': [], 'SNZ': SZ, 'SNRA': SRA,
                                                                             'SNDEC': SDE, 'RAs': [], 'DECs': []}
                if SDE > 3.6 - cone_radius / 60.0:
                    h = SDE - (3.6 - cone_radius / 60.0)
                elif SDE < -(0.0 - cone_radius / 60.0):
                    h = -(0.0 - cone_radius / 60.0) - SDE
                else:
                    h = 0.0

                if h == 0:
                    lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['WEIGHT'] = 1.0
                    indices = [(RA1 - SRA) ** 2 + (DEC1 - SDE) ** 2 <= (cone_radius / 60.0) ** 2]
                    lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['Zs'] = z1[indices]
                    lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['RAs'] = RA1[indices]
                    lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['DECs'] = DEC1[indices]
            print(f"Finished sorting radius {str(cone_radius)}'")
            # lens_data[f"Radius{str(cone_radius)}"] = {}
            # for num, (SRA, SDE, SZ) in enumerate(zip(RA2, DEC2, z2)):
            #     lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}'] = {'Zs': [], 'SNZ': SZ, 'SNRA': SRA,
            #                                                                  'SNDEC': SDE, 'RAs': [], 'DECs': []}
            #
            #     if SDE > 1.28 - cone_radius / 60.0:
            #         h = SDE - (1.28 - cone_radius / 60.0)
            #     elif SDE < -(1.28 - cone_radius / 60.0):
            #         h = -(1.28 - cone_radius / 60.0) - SDE
            #     else:
            #         h = 0
            #
            #     if h == 0:
            #         lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['WEIGHT'] = 1.0
            #         indices = [(RA1 - SRA) ** 2 + (DEC1 - SDE) ** 2 <= (cone_radius / 60.0) ** 2]
            #         lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['Zs'] = z1[indices]
            #         lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['RAs'] = RA1[indices]
            #         lens_data[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['DECs'] = DEC1[indices]
            # print(f"Finished sorting radius {str(cone_radius)}'")

        expected = {}
        avg_ws = []
        stds_ws = []
        for cone_radius in RADII[29::2]:
            # IPs = {}
            lens = lens_data[f"Radius{str(cone_radius)}"]
            cumul_tot = np.zeros((len(limits), len(lens)))
            for num1, lim in enumerate(limits):
                for num2, (key, item) in enumerate(lens.items()):
                    # IPs[key] = np.zeros(51 - 2)
                    Zs = np.array(item['Zs'])
                    thetas = (((item['RAs'] - item["SNRA"]) ** 2 + (
                            item['DECs'] - item["SNDEC"]) ** 2) ** 0.5 * np.pi / 180)
                    # print(thetas)

                    # print(thetas[thetas!=0])
                    Dperps = thetas[thetas != 0] * np.interp(Zs[thetas != 0], fine_z, Dpara_fine) * \
                             1000.0 / (1 + Zs[thetas != 0])
                    all_IPs = 1.0 / Dperps
                    # print(all_IPs)
                    cumul_tot[num1][num2] = np.sum(all_IPs[Zs[thetas != 0] < lim])
            expected[f"Radius{str(cone_radius)}"] = np.diff([np.mean(cumul_tot, 1)])
            avg_ws.append(np.mean(expected[f"Radius{str(cone_radius)}"]))
            stds_ws.append(np.std(expected[f"Radius{str(cone_radius)}"]) / np.sqrt(len(expected[f"Radius{str(cone_radius)}"])))
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
        plt.errorbar(RADII[29::2], avg_ws, stds_ws, elinewidth=0.8, marker='o', ms=5, ls='', color=colours[3])
        plt.show()
        exit()
        pickle_out = open("sparse2_expected_IPs.pickle", "wb")
        pickle.dump(expected, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("sparse2_expected_IPs.pickle", "rb")
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


def redo_lenses():
    pickle_in = open("sparse_lenses.pickle", "rb")
    lenses = pickle.load(pickle_in)
    Gal_data = MICE.get_data()
    pickle_in = open("sparse2MICE_SN_data.pickle", "rb")
    SN_data = pickle.load(pickle_in)
    print(min(SN_data["SNkappa"]))
    # plt.plot(SN_data["SNZ"], SN_data["SNMU"], '.', ls='')
    # plt.show()
    pickle_out = open("sparse2_cut_data.pickle", "wb")
    cut_data = {"RA1": Gal_data["RA"], "DEC1": Gal_data["DEC"], "RA2": SN_data["SNRA"], "DEC2": SN_data["SNDEC"],
                "z1": Gal_data["z"], "z2": SN_data["SNZ"], "mu": SN_data["SNMU"], "mu_err": SN_data["SNMU_ERR"],
                "SNkappa": SN_data["SNkappa"]}
    print(lenses.keys(), cut_data.keys())
    pickle.dump(cut_data, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    # redo_lenses()
    # exit()
    # crit_weight = 0.1
    # crit_angles = [3.0, 6.0, 12.0, 24.0]
    # crit_dists = [2.5, 5.0, 7.5, 10.0]
    fine_z = np.linspace(0, 1.5, 1001)
    Dpara_fine = Convergence.comoving(fine_z, OM=0.25, OL=0.75, h=0.7)
    # data = MICE.get_data()
    # lenses = MICE.get_random(data, redo=False)

    # data, _ = cones.get_data(new_data=False)
    # lenses = cones.sort_SN_gals(data, redo=False, weighted=True)
    # exp_data = cones.find_expected_counts(_, 51)
    pickle_in = open("sparse2_cut_data.pickle", "rb")
    data = pickle.load(pickle_in)
    pickle_in = open("sparse_lenses.pickle", "rb")
    lenses = pickle.load(pickle_in)
    pickle_in = open("sparse2_test_cones.pickle", "rb")
    cone_array = pickle.load(pickle_in)
    exp_data = cones.find_expected_counts(cone_array, 111, redo=False, plot=False)
    # exp_data = find_expected_weights(data, 111, redo=False, plot=False)
    zs = []
    perps = []
    ws = []
    # lenses_IP = {}
    # lenses_IP[crit_dist] = {}
    redo_IP = 0
    # pickle_in = open("lenses_IP2_gal.pickle", "rb")
    # lenses_IP = pickle.load(pickle_in)
    # print(lenses_IP.keys(), lenses_IP[0.05].keys())
    lenses_IP = {}
    if redo_IP:
        for radius in RADII[29::2]:
            print(radius)
            lenses_IP[f'Radius{radius}'] = {}
            # average_per_bin = []
            for key, item in lenses[f'Radius{radius}'].items():
                if lenses[f'Radius{radius}'][key]["WEIGHT"] == 1.0:
                    lenses_IP[f'Radius{radius}'][key] = {"SNZ": lenses[f'Radius{radius}'][key]["SNZ"],
                                                         "Zs": lenses[f'Radius{radius}'][key]["Zs"],
                                                         "SNMU": lenses[f'Radius{radius}'][key]["SNMU"],
                                                         "SNMU_ERR": lenses[f'Radius{radius}'][key]["SNMU_ERR"],
                                                         "IPWEIGHT": []}
                    exp_data[0].put(0, 0)
                    for z, ra, dec in zip(item['Zs'], item["RAs"], item["DECs"]):
                        # bin_num = np.where([np.logical_and(exp_data[0][i] < z, exp_data[0][i + 1] > z) for i in
                        #                     range(len(exp_data[0]) - 1)])[0]
                        # print(z, exp[0][bin_num[0]], exp[0][bin_num[0]+1], bin_num[0])
                        theta = (((ra - item["SNRA"]) ** 2 + (dec - item["SNDEC"]) ** 2) ** 0.5 * np.pi / 180)
                        theta = theta[theta!=0]
                        # print(theta)
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
            # print(len(lenses_IP[0.05][f'Radius{radius}']))
        # print(lenses_IP.keys())
        # print(lenses_IP[3.0].keys())
        # print(lenses_IP[3.0]["Radius3.0"].keys())
        # print(lenses_IP[3.0]["Radius3.0"]["SN342"].keys())
        # print(lenses_IP[2.5]["Radius3.0"]["SN342"]["IPWEIGHT"], lenses_IP[2.5]["Radius3.0"]["SN343"]["IPWEIGHT"],
        #       lenses_IP[7.5]["Radius3.0"]["SN342"]["IPWEIGHT"], lenses_IP[10.0]["Radius3.0"]["SN342"]["IPWEIGHT"])

        # plt.plot(fine_z, 0.05*np.pi/180.0 * Dpara0)")
        # plt.legend(frameon=0)

        # s = plt.scatter(perps, ws, c=zs, cmap='coolwarm')
        # plt.gca().set_yscale('log')
        # plt.gca().set_xscale('log')
        # cbar = plt.colorbar(s)
        # cbar.set_label('$z$')
        # plt.xlim(0.00038, 15)
        # plt.show()

        pickle_out = open(f"sparse2_lenses_IP.pickle", "wb")
        pickle.dump(lenses_IP, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("sparse2_lenses_IP.pickle", "rb")
        lenses_IP = pickle.load(pickle_in)
    #
    # plt.plot([0, 0.6], [2.5, 2.5], c=colours[1], lw=2, ls='-')
    # plt.plot([0, 0.6], [5.0, 5.0], c=colours[1], lw=2, ls='--')
    # plt.plot([0, 0.6], [7.5, 7.5], c=colours[1], lw=2, ls='-.')
    # plt.plot([0, 0.6], [10.0, 10.0], c=colours[1], lw=2, ls=':')
    # plt.plot(np.linspace(0, 0.6, 101),
    #          1000.0 * np.deg2rad(3.0 / 60.0) / (1 + np.linspace(0, 0.6, 101)) * Convergence.comoving(
    #              np.linspace(0, 0.6, 101)), c=colours[0], lw=2, ls='-')
    # plt.plot(np.linspace(0, 0.6, 101),
    #          1000.0 * np.deg2rad(6.0 / 60.0) / (1 + np.linspace(0, 0.6, 101)) * Convergence.comoving(
    #              np.linspace(0, 0.6, 101)), c=colours[0], lw=2, ls='--')
    # plt.plot(np.linspace(0, 0.6, 101),
    #          1000.0 * np.deg2rad(12.0 / 60.0) / (1 + np.linspace(0, 0.6, 101)) * Convergence.comoving(
    #              np.linspace(0, 0.6, 101)), c=colours[0], lw=2, ls='-.')
    # plt.plot(np.linspace(0, 0.6, 101),
    #          1000.0 * np.deg2rad(24.0 / 60.0) / (1 + np.linspace(0, 0.6, 101)) * Convergence.comoving(
    #              np.linspace(0, 0.6, 101)), c=colours[0], lw=2, ls=':')
    # plt.xlabel('$z$')
    # plt.ylabel('$\u03BE_0$ (Mpc)')
    # plt.axis([0, 0.6, 0, 11])
    # plt.tight_layout()
    # plt.show()
    # exit()
    exp_wts = find_expected_weights(data, 111, redo=True)
    kappa_impact = cones.find_convergence(lenses_IP, exp_wts, redo=False, plot_scatter=True, impact=True, max_z=1.41)
    conv_total_impact = []
    for cone_radius in RADII[29::2]:
        conv_total_impact.append(kappa_impact[f"Radius{str(cone_radius)}"]["Total"])
    plt.plot(RADII[29::2], conv_total_impact, marker='o', markersize=2, color=colours[3])
    plt.plot(RADII[29::2], np.zeros(len(RADII[29::2])), color=[0.75, 0.75, 0.75], linestyle='--')
    plt.xlabel("Cone Radius (arcmin)")
    plt.ylabel("Total $\kappa$")
    plt.show()
    impact = cones.find_correlation(kappa_impact, lenses_IP, plot_radii=True, impact=True)
