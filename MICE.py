import Convergence
import cones
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import random
from scipy.stats import rankdata
import pickle
from scipy.signal import savgol_filter
import collections

colours = [[0, 150/255, 100/255], [225/255, 149/255, 0], [207/255, 0, 48/255], [145/255, 4/255, 180/255],
           'C4', 'C9', 'C6', 'C7', 'C8', 'C5']
blue = [23/255, 114/255, 183/255, 0.75]
orange = [255/255, 119/255, 15/255, 0.75]
green = [0, 150/255, 100/255, 0.75]
yellow = [253/255, 170/255, 0, 0.75]
grey = [0.75, 0.75, 0.75]
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

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Stixgeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True


def deep_update(old_dict, update_to_dict):
    for key, value in update_to_dict.items():
        if isinstance(value, collections.Mapping):
            old_dict[key] = deep_update(old_dict.get(key, {}), value)
        else:
            old_dict[key] = value
    return old_dict


def get_data():
    with fits.open('MICEsim5.fits') as hdul1:
        RA = hdul1[1].data['ra']
        DEC = hdul1[1].data['dec']
        kap = hdul1[1].data['kappa']
        z = hdul1[1].data['z_v']
        ID = hdul1[1].data['id']

    RA = np.array(RA)[[z >= 0.01]]
    DEC = np.array(DEC)[[z >= 0.01]]
    kap = np.array(kap)[[z >= 0.01]]
    ID = np.array(ID)[[z >= 0.01]]
    z = np.array(z)[[z >= 0.01]]
    cut_data = {'RA': RA, 'DEC': DEC, 'z': z, 'kappa': kap, 'id': ID}

    return cut_data


def make_big_cone(data, redo=False):
    if redo:
        RAs = data['RA']
        DECs = data['DEC']
        zs = data['z']
        kappas = data['kappa']
        centre = [(min(RAs) + max(RAs)) / 2, (min(DECs) + max(DECs)) / 2]
        radius = round(min(max(RAs) - centre[0], centre[0] - min(RAs), max(DECs) - centre[1], centre[1] - min(DECs)), 2)
        # radius = 3.5
        big_cone = {'Zs': zs[(RAs - centre[0]) ** 2 + (DECs - centre[1]) ** 2 <= radius ** 2],
                    'kappa': kappas[(RAs - centre[0]) ** 2 + (DECs - centre[1]) ** 2 <= radius ** 2]}
        for i in [1, 2]:
            big_cone['Zs'] = np.append(big_cone['Zs'], zs[(RAs - centre[0] + 2 * i * radius) ** 2 +
                                                          (DECs - centre[1]) ** 2 <= radius ** 2])
            big_cone['kappa'] = np.append(big_cone['kappa'], kappas[(RAs - centre[0] + 2 * i * radius) ** 2 +
                                                                    (DECs - centre[1]) ** 2 <= radius ** 2])
            big_cone['Zs'] = np.append(big_cone['Zs'], zs[(RAs - centre[0] - 2 * i * radius) ** 2 +
                                                          (DECs - centre[1]) ** 2 <= radius ** 2])
            big_cone['kappa'] = np.append(big_cone['kappa'], kappas[(RAs - centre[0] - 2 * i * radius) ** 2 +
                                                                    (DECs - centre[1]) ** 2 <= radius ** 2])

        pickle_out = open(f"big_cone.pickle", "wb")
        pickle.dump(big_cone, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open(f"big_cone.pickle", "rb")
        big_cone = pickle.load(pickle_in)
        pass

    return big_cone


def find_expected(big_cone, r_big, bins, redo=False, plot=False):
    max_z = 1.41
    chi_bin_widths, chi_bins, z_bins, z_bin_widths = Convergence.create_z_bins(0.01, max_z, bins, OM=0.25, OL=0.75,
                                                                               h=0.7)
    limits = np.cumsum(z_bin_widths) + z_bins[0]
    limits = np.insert(limits, 0, 0)
    expected = {}
    if redo:
        expected_big = []
        for num1 in range(len(limits) - 1):
            expected_big.append(np.count_nonzero(np.logical_and(big_cone['Zs'] > limits[num1], big_cone['Zs'] <
                                                                limits[num1 + 1])) / 5.0)
            # Made 5 cones, so take average
        plt.plot(limits[1:], [np.cumsum(expected_big)[i] * (12.0 / r_big / 60.0) ** 2 for i in range(len(expected_big))],
                 marker='o', markersize=2.5, color=colours[0])
        plt.xlabel('$z$')
        plt.ylabel('Cumulative Count')
        plt.show()

        for cone_radius in RADII:
            expected[f"Radius{str(cone_radius)}"] = [expected_big[i] * (cone_radius / r_big / 60.0) ** 2
                                                     for i in range(len(expected_big))]

        pickle_out = open("MICEexpected.pickle", "wb")
        pickle.dump(expected, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("MICEexpected.pickle", "rb")
        expected = pickle.load(pickle_in)

    if plot:
        for cone_radius in [12.0]:
            plt.plot([0, 5], [0, 0], color=grey, linestyle='--')
            plt.plot((limits[1:]+limits[:-1])/2.0, expected[f"Radius{str(cone_radius)}"], marker='o', markersize=2.5,
                     color=colours[0])
            plt.xlabel('$z$')
            plt.ylabel('Expected Count')
            plt.xlim([0, 1.5])
            plt.show()

    return [limits, expected, chi_bin_widths, chi_bins, z_bins]


def get_random(data, redo=False):
    RAs = data['RA']
    DECs = data['DEC']
    zs = data['z']
    kappas = data['kappa']

    # Don't want to deal with up to 30' (0.5 degrees) cones that have any portion outside left and right bounds.
    SN_DECs = DECs[np.logical_and(RAs < max(RAs) - 0.5, RAs > min(RAs) + 0.5)]
    SN_zs = zs[np.logical_and(RAs < max(RAs) - 0.5, RAs > min(RAs) + 0.5)]
    SN_kappas = kappas[np.logical_and(RAs < max(RAs) - 0.5, RAs > min(RAs) + 0.5)]
    SN_RAs = RAs[np.logical_and(RAs < max(RAs) - 0.5, RAs > min(RAs) + 0.5)]

    # SN_RAs = SN_RAs[np.logical_and(SN_DECs < max(DECs) - 0.5, SN_DECs > min(DECs) + 0.5)]
    # SN_kappas = SN_kappas[np.logical_and(SN_DECs < max(DECs) - 0.5, SN_DECs > min(DECs) + 0.5)]
    # SN_zs = SN_zs[np.logical_and(SN_DECs < max(DECs) - 0.5, SN_DECs > min(DECs) + 0.5)]
    # SN_DECs = SN_DECs[np.logical_and(SN_DECs < max(DECs) - 0.5, SN_DECs > min(DECs) + 0.5)]

    if redo:
        # Pick random sample
        random.seed(1337)
        rand_samp_size = 1500
        indices = random.sample(range(len(SN_zs)), rand_samp_size)
        rand_zs = SN_zs[indices]
        rand_RAs = SN_RAs[indices]
        rand_DECs = SN_DECs[indices]
        rand_kappas = SN_kappas[indices]

        # Add scatter to distance moduli
        dists = []
        rand_chis = []
        for z in rand_zs:
            chi_to_z = Convergence.comoving(np.linspace(0, z, 1001), OM=0.25, OL=0.75, h=0.7)
            dists.append(chi_to_z[-1] * (1 + z))
            rand_chis.append(chi_to_z[-1])
        mus = 5 * np.log10(np.array(dists) / 10 * 1E9)
        mu_diff = - (5.0 / np.log(10) * rand_kappas)
        for i in range(len(mu_diff)):
            mu_diff[i] += random.gauss(0.0, rand_zs[i] * 0.1 / 1.5 + 0.15)  # Width is 0.15 at z=0, 0.25 at z=1.5
        rand_mus = mus + mu_diff
        rand_errs = np.array([abs(random.uniform(0.05+0.1*rand_zs[i], 0.1+0.45*rand_zs[i]))
                              for i in range(rand_samp_size)])
        SN_data = {'mu_diff': mu_diff, 'SNZ': rand_zs, 'SNkappa': rand_kappas,
                   'SNRA': rand_RAs, 'SNDEC': rand_DECs, 'SNMU': rand_mus, 'SNMU_ERR': rand_errs}
        pickle_out = open("MICE_SN_data.pickle", "wb")
        pickle.dump(SN_data, pickle_out)
        pickle_out.close()
        print("Finished SN_data")

        lenses = {}
        prev_rad = 0.0
        for cone_radius in RADII:
            lenses[f"Radius{str(cone_radius)}"] = {}
            for num, (RA, DEC) in enumerate(zip(rand_RAs, rand_DECs)):
                cone_indices = [np.logical_and((RAs - RA) ** 2 + (DECs - DEC) ** 2 >= (prev_rad / 60.0) ** 2,
                                               (RAs - RA) ** 2 + (DECs - DEC) ** 2 <= (cone_radius / 60.0) ** 2)]
                lenses[f"Radius{str(cone_radius)}"][f"Shell{str(num+1)}"] = np.where(cone_indices[0] == 1)
                print(f"Sorted {num+1}/{rand_samp_size} for radius {cone_radius}'")
            heights = np.zeros(rand_samp_size)
            outsides_u = [rand_DECs > 3.6 - cone_radius / 60.0]
            heights[outsides_u] = rand_DECs[outsides_u] - (3.6 - cone_radius / 60.0)
            outsides_d = [rand_DECs < cone_radius / 60.0]
            heights[outsides_d] = cone_radius / 60.0 - rand_DECs[outsides_d]
            thetas = 2 * np.arccos(1 - heights / (cone_radius / 60.0))
            fraction_outside = 1 / (2 * np.pi) * (thetas - np.sin(thetas))
            weights = 1.0 - fraction_outside
            lenses[f"Radius{str(cone_radius)}"]['WEIGHT'] = weights
            print(f"Sorted radius {cone_radius}'")
            prev_rad = cone_radius
        pickle_out = open(f"random_cones_new.pickle", "wb")
        pickle.dump(lenses, pickle_out)
        pickle_out.close()


def plot_cones(data, plot_hist=False, cone_radius=12.0):
    """Plots all galaxies and SNe along with visualisation of cones and galaxies contributing to lensing.

    Input:
     cut_data -- dictionary that contains all data (RA, DEC, z, etc.) of galaxies.
     sorted_data -- dictionary that contains all information for every SN sorted into cones.
     plot_hist -- boolean that determines if a histogram of the galaxy and SNe distribution is plotted. Defaults to
                  False.
     cone_radius -- the radius of the cones. Defaults to 12'.
    """
    pickle_in = open("MICE_SN_data.pickle", "rb")
    SN_data = pickle.load(pickle_in)
    # lenses = sorted_data[f"Radius{str(cone_radius)}"]
    # # Go through all SNe
    # for SN_num, key in enumerate(lenses.keys()):
    #     if key != 'WEIGHT':
    #         cone_indices = np.array([], dtype=np.int16)
    #         # Get shells from all previous RADII
    #         for r in RADII[0:np.argmin(np.abs(RADII - np.array(cone_radius))) + 1]:
    #             cone_indices = np.append(cone_indices, sorted_data[f"Radius{r}"][key])
    #         # Get redshifts of all galaxies in each SN cone
    #         cone_zs[key] = all_zs[cone_indices]
    # print(lenses.keys())
    patches = []
    SNRA = SN_data['SNRA']
    SNDEC = SN_data['SNDEC']
    SNz = SN_data['SNZ']
    for x, y in zip(SNRA, SNDEC):
        circle = Circle((x, y), cone_radius/60.0)
        patches.append(circle)

    RA_gal = data['RA']
    DEC_gal = data['DEC']
    z_gal = data['z']
    fig, ax = plt.subplots()
    ax.plot(RA_gal[1::50], DEC_gal[1::50], marker='o', linestyle='', markersize=1, color=[0.5, 0.5, 0.5])
    contRAs = []
    contDECs = []
    for ra, dec, z in zip(SNRA, SNDEC, SNz):
        indices1 = np.logical_and(z_gal[::50] <= z, (RA_gal[::50] - ra) ** 2 + (DEC_gal[::50] - dec) ** 2 <=
                                  (cone_radius / 60.0) ** 2)
        contRAs = np.append(contRAs, RA_gal[::50][indices1])
        contDECs = np.append(contDECs, DEC_gal[::50][indices1])
    ax.plot(contRAs, contDECs, marker='o', linestyle='', markersize=3, color=colours[1])
    p = PatchCollection(patches, alpha=0.4, color=colours[1])
    ax.add_collection(p)

    ax.plot(SNRA, SNDEC, marker='o', linestyle='', markersize=3, label='Supernova', color=colours[3])
    plt.xlabel('$\\alpha$')
    plt.ylabel('$\delta$')
    plt.text(27, -0.8, f"{cone_radius}' radius")
    # plt.legend(loc='lower right')
    plt.axis('equal')
    plt.xlim([10.0, 11.5])
    plt.ylim([1.5, 2.5])
    plt.show()

    if plot_hist:
        labels = ['Galaxies', 'Supernovae']
        cols = [green, yellow]
        for num, z in enumerate([z_gal, SNz]):
            plt.hist([i for i in z if i <= 1.5], bins=np.arange(0, 1.5 + 0.025, 0.025), normed='max', linewidth=1,
                     fc=cols[num], label=f'{labels[num]}', edgecolor=colours[num])
        plt.xlabel('$z$')
        plt.ylabel('Normalised Count')
        plt.legend(frameon=0)

        plt.show()


def find_convergence(gal_data, exp_data, redo=False, plot_scatter=False, plot_total=False, weighted=False, fis=False):
    """Finds the convergence along each line of sight to a SN for a variety of cone_widths.

    Inputs:
     exp_data -- dictionary containing all expected counts per bin per cone width.
     SNz -- redshifts of each SN.
     redo -- boolean that determines whether convergence is calculated or loaded. Dafault false.
     plot_scatter -- boolean that determined whether scatter plot of convergence per SN redshift is plotted.
                     Default false.
     plot_total -- boolean that determines whether total convergence per cone radius is plotted. Default false.
    """
    all_zs = gal_data['z']
    limits = exp_data[0]
    chi_widths = exp_data[2]
    chi_bis = exp_data[3]
    z_bins = exp_data[4]
    if redo:
        kappa = {}
        if fis:
            pickle_in = open("MICE_SN_data_fis.pickle", "rb")
            SN_data = pickle.load(pickle_in)
            pickle_in = open("random_cones_new_fis.pickle", "rb")
        else:
            pickle_in = open("MICE_SN_data.pickle", "rb")
            SN_data = pickle.load(pickle_in)
            pickle_in = open("random_cones_new.pickle", "rb")
        lens_data = pickle.load(pickle_in)

        for cone_radius in RADII:
            if fis:
                SN_zs = SN_data[f"Radius{cone_radius}"]["SNZ"]
            else:
                SN_zs = SN_data["SNZ"]
            cone_zs = {}
            if weighted:
                SN_weights = lens_data[f"Radius{cone_radius}"]["WEIGHT"]
            # Go through all SNe
            for SN_num, key in enumerate(lens_data[f"Radius{cone_radius}"].keys()):
                if key != 'WEIGHT':
                    cone_indices = np.array([], dtype=np.int16)
                    # Get shells from all previous RADII
                    for r in RADII[0:np.argmin(np.abs(RADII - np.array(cone_radius))) + 1]:
                        cone_indices = np.append(cone_indices, lens_data[f"Radius{r}"][key])
                    # Get redshifts of all galaxies in each SN cone
                    cone_zs[key] = all_zs[cone_indices]
            expected_counts = exp_data[1][f"Radius{str(cone_radius)}"]
            kappa[f"Radius{str(cone_radius)}"] = {"SNkappa": [], "Total": 0, "SNallkappas": {}}
            d_arr = {}
            counts = {}
            for num, (key, zs) in enumerate(cone_zs.items()):
                bin_c = range(int(np.argmin(np.abs(limits - SN_zs[num]))))
                counts[key] = np.zeros(len(bin_c))
                for num2 in bin_c:
                    tmp = [np.logical_and(limits[num2] < zs, zs <= limits[num2 + 1])]
                    if weighted:
                        counts[key][num2] = np.count_nonzero(tmp) / SN_weights[num]
                    else:
                        counts[key][num2] = np.count_nonzero(tmp)

            chiSNs = []
            for z in SN_zs:
                chi = Convergence.comoving(np.linspace(0, z, 1001), OM=0.25, OL=0.75, h=0.7)
                chiSNs.append(chi[-1])
            # c_arr = []
            for num, (key, cs) in enumerate(counts.items()):
                d_arr[key] = (cs - expected_counts[:len(cs)]) / expected_counts[:(len(cs))]
                SNkappa, allkappas = Convergence.general_convergence(chi_widths[:len(cs)], chi_bis[:len(cs)],
                                                             z_bins[:len(cs)], d_arr[key], chiSNs[num], OM=0.25, h=0.7)
                kappa[f"Radius{str(cone_radius)}"]["SNkappa"].append(SNkappa)
                kappa[f"Radius{str(cone_radius)}"]["SNallkappas"][key] = allkappas
                # c_arr.append(cs)
            # s = plt.scatter([sum(c_arr[i]) for i in range(1500)], kappa[f"Radius{str(cone_radius)}"]["SNkappa"],
            #                c=SN_zs, cmap='coolwarm')
            # cbar = plt.colorbar(s)
            # cbar.set_label('$z$')
            # plt.xlabel('Total Count')
            # plt.ylabel('$\kappa$')
            # plt.show()

            kappa[f"Radius{str(cone_radius)}"]["Total"] = np.sum(kappa[f"Radius{str(cone_radius)}"]["SNkappa"])
            print(f"Finished radius {str(cone_radius)}'")
        if not fis:
            if weighted:
                pickle_out = open("MICEkappa_weighted.pickle", "wb")
            else:
                pickle_out = open("MICEkappa.pickle", "wb")
        else:
            pickle_out = open("MICEkappa_fis.pickle", "wb")
        pickle.dump(kappa, pickle_out)
        pickle_out.close()
    else:
        if not fis:
            pickle_in = open("MICE_SN_data.pickle", "rb")
            SN_data = pickle.load(pickle_in)
            if weighted:
                pickle_in = open("MICEkappa_weighted.pickle", "rb")
            else:
                pickle_in = open("MICEkappa.pickle", "rb")
        else:
            pickle_in = open("MICE_SN_data_fis.pickle", "rb")
            SN_data = pickle.load(pickle_in)
            pickle_in = open("MICEkappa_fis.pickle", "rb")
        kappa = pickle.load(pickle_in)

    for cone_radius in [12.0]:
        if fis:
            SN_zs = SN_data[f"Radius{cone_radius}"]["SNZ"]
            SN_kappas = SN_data[f"Radius{cone_radius}"]["SNkappa"]
        else:
            SN_zs = SN_data["SNZ"]
            SN_kappas = SN_data["SNkappa"]
        bins = np.linspace(0.05, 1.4 - 0.05, 14)
        # print(bins)
        mean_kappa = []
        standard_error = []
        mean_MICEkappa = []
        standard_MICEerror = []
        conv = kappa[f"Radius{str(cone_radius)}"]["SNkappa"]

        for b in bins:
            ks = []
            MICEks = []
            for z, k, Mk in zip(SN_zs, conv, SN_kappas):
                if b - 0.05 < z <= b + 0.05:
                    ks.append(k)
                    MICEks.append(Mk)

            mean_kappa.append(np.mean(ks))
            mean_MICEkappa.append(np.mean(MICEks))
            standard_error.append(np.std(ks) / np.sqrt(len(ks)))
            standard_MICEerror.append(np.std(MICEks) / np.sqrt(len(MICEks)))

        if plot_scatter:
            conv = kappa[f"Radius{str(cone_radius)}"]["SNkappa"]
            ax = plt.subplot2grid((1, 4), (0, 0), colspan=3)
            # ax = plt.subplot2grid((1, 1), (0, 0))
            ax2 = plt.subplot2grid((1, 4), (0, 3))
            ax.set_ylabel("$\kappa$")
            ax.set_xlabel("$z$")
            ax2.set_xlabel("Count")
            ax.tick_params(labelsize=12)
            ax2.tick_params(labelsize=12)
            ax2.set_yticklabels([])
            plt.subplots_adjust(wspace=0, hspace=0)
            ax.plot([0, 1.42], [0, 0], color=[0.25, 0.25, 0.25], linestyle='--', zorder=10)
            ax.axis([0, 1.42, -0.05, 0.06])
            ax2.axis([0, 500, -0.05, 0.06])
            ax.set_xticklabels([0, 0.25, 0.50, 0.75, 1.00, 1.25])
            # ax.set_xticklabels([0, 0.2, 0.4, 0])
            ax.plot(SN_zs, conv, linestyle='', marker='o', markersize=2, color=colours[0], label='Cone Method')
            ax.plot(SN_zs, SN_kappas, linestyle='', marker='o', markersize=2, color=colours[1], label='MICE value')
            ax2.hist(conv, bins=np.arange(-0.05, 0.08 + 0.005, 0.005), orientation='horizontal',
                     fc=green, edgecolor=colours[0])
            ax2.hist(SN_kappas, bins=np.arange(-0.05, 0.08 + 0.005, 0.005), orientation='horizontal',
                     fc=yellow, edgecolor=colours[1])
            ax.errorbar(bins, mean_MICEkappa, standard_MICEerror, marker='d', color='b', markersize=3, capsize=3,
                        zorder=20)
            ax.errorbar(bins, mean_kappa, standard_error, marker='s', color='r', markersize=3, capsize=3, zorder=20)
            plt.show()

    if plot_total:
        conv_total = []
        for cone_radius in RADII:
            conv_total.append(kappa[f"Radius{str(cone_radius)}"]["Total"])
        plt.ylabel("$\kappa$")
        plt.xlabel("Cone Radius (arcmin)")
        plt.tick_params(labelsize=12)
        plt.plot([0, 30], [0, 0], color=grey, linestyle='--')
        plt.xlim([0, 30])
        plt.plot(RADII, conv_total, marker='o', markersize=2, color=colours[0])
        plt.show()

    return kappa


def find_correlation(convergence_data, radii, plot_correlation=False, plot_radii=False, fis=False, mu_diff=None):
    """Finds the value of the slope for plotting residuals against convergence. Magnitude of slope and error
    quantify correlation between the two.

    Inputs:
     conv -- convergence.
     mu_diff -- residuals.
    """
    correlations = []
    correlation_errs = []
    for cone_radius in radii:
        if fis:
            pickle_in = open("MICE_SN_data_fis.pickle", "rb")
            SN_data = pickle.load(pickle_in)
            mu_diff = SN_data[f"Radius{str(cone_radius)}"]["mu_diff"]
            conv = np.array(convergence_data[f"Radius{str(cone_radius)}"]["SNkappa"])
        else:
            pickle_in = open("MICE_SN_data.pickle", "rb")
            SN_data = pickle.load(pickle_in)
            # redshift_cut = [SN_data['SNZ'] > 0.2]
            if mu_diff is None:
                mu_diff = SN_data["mu_diff"]
            conv = np.array(convergence_data[f"Radius{str(cone_radius)}"]["SNkappa"])

        conv_rank = rankdata(conv)
        mu_rank = rankdata(mu_diff)
        diff = np.abs(conv_rank - mu_rank)
        rho = 1 - 6 / (len(conv) * (len(conv) ** 2 - 1)) * np.sum(diff ** 2)
        rho_err = np.sqrt((1 - rho ** 2) / (len(conv) - 1))
        correlations.append(rho)
        correlation_errs.append(rho_err)

        if plot_correlation:
            edges = np.linspace(-0.0065, 0.011, 6)
            bins = (edges[1:] + edges[:-1]) / 2
            mean_dmu = []
            standard_error = []
            for bin in bins:
                dmus = []
                for kappa, dmu in zip(conv, mu_diff):
                    if bin - 0.007 / 4 < kappa <= bin + 0.0007 / 4:
                        dmus.append(dmu)
                mean_dmu.append(np.mean(dmus))
                standard_error.append(np.std(dmus) / np.sqrt(len(dmus)))

            plt.plot([min(conv), max(conv)], [0, 0], color=grey, linestyle='--')
            plt.plot(conv, mu_diff, linestyle='', marker='o', markersize=2, color=colours[0])
            # plt.plot(conv, fit, color=colours[1], label=f'$\Delta\mu = {round(float(grad),3)}\kappa$')
            plt.errorbar(bins, mean_dmu, standard_error, marker='s', color='r', markersize=3, capsize=3, linestyle='')
            plt.xlabel('$\kappa$')
            plt.ylabel('$\Delta\mu$')
            plt.xlim([-0.008, 0.011])
            plt.legend(frameon=0, loc='lower right')
            plt.ylim([-0.3, 0.3])
            plt.text(0.0038, -0.19, f'$\\rho$ = {round(rho, 3)} $\pm$ {round(rho_err, 3)}', fontsize=16)
            plt.show()

    if plot_radii:
        u_err = [correlations[i] + correlation_errs[i] for i in range(len(correlations))]
        d_err = [correlations[i] - correlation_errs[i] for i in range(len(correlations))]
        smooth_corr = savgol_filter([correlations[i] for i in range(len(correlations))], 11, 4)
        smooth_u_err = savgol_filter(u_err, 11, 4)
        smooth_d_err = savgol_filter(d_err, 11, 4)
        plt.plot([0, 30], [0, 0], color=grey, linestyle='--')
        plt.plot(RADII, smooth_corr, color=colours[0])
        plt.plot(RADII, [correlations[i] for i in range(len(correlations))], marker='x', color=colours[1],
                 linestyle='')
        plt.fill_between(RADII, smooth_u_err, smooth_d_err, color=colours[0], alpha=0.4)

        plt.xlabel('Cone Radius (arcmin)')
        plt.ylabel("Spearman's Rank Coefficient")
        plt.gca().invert_yaxis()
        plt.show()
        return [correlations, smooth_corr, smooth_u_err, smooth_d_err, np.array(u_err) - np.array(correlations)]

    return correlations, correlation_errs


def plot_Hubble():
    """Plots the Hubble diagram (distance modulus against redshift), including the best fitting cosmology, and
    residuals from best cosmology.

    Inputs:
     z -- redshift of SNe.
     mu -- distance modulus of SNe.
     mu_err -- error in distance modulus of SNe.
     mu_cosm -- distance modulus of best fitting cosmology.
     mu_diff -- residuals from best fitting cosmology.
     z_arr -- array of redshifts used for best fitting cosmology.
    """
    pickle_in = open("MICE_SN_data.pickle", "rb")
    SN_data = pickle.load(pickle_in)
    z = SN_data["SNZ"]
    mu = SN_data['SNMU']
    mu_err = SN_data['SNMU_ERR']
    z_array = np.linspace(0.0, 1.5 + 0.01, 1001)
    mu_cosm = 5 * np.log10((1 + z_array) * Convergence.comoving(z_array, OM=0.25, OL=0.75, h=0.7) * 1000) + 25
    mu_diff = SN_data['mu_diff']
    ax = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    ax.set_ylabel("$\mu$")
    ax2.set_xlabel("$z$")
    ax2.set_ylabel("$\Delta\mu$")
    plt.subplots_adjust(wspace=0, hspace=0)
    ax.set_xticklabels([])
    ax.tick_params(labelsize=12)
    ax.errorbar(z[::2], mu[::2], mu_err[::2], linestyle='', linewidth=0.8, marker='o',
                markersize=2, capsize=2, color='C3', zorder=0, alpha=0.4)
    ax.plot(z[::2], mu[::2], linestyle='', marker='o', markersize=2, color='C3', alpha=0.25, markerfacecolor='C3')

    ax.set_ylim([38.5, 46])
    ax.set_xlim([0, 1.5])
    ax.plot(z_array, mu_cosm, linestyle='--', linewidth=0.8, color='C0', zorder=10)
    ax2.errorbar(z[::2], mu_diff[::2], mu_err[::2], linestyle='', linewidth=1, marker='o',
                 markersize=2, capsize=2, color='C3', zorder=0, alpha=0.4)
    ax2.plot(z[::2], mu_diff[::2], linestyle='', marker='o', markersize=2, color='C3', alpha=0.25, markerfacecolor='C3')
    ax2.plot(z_array, np.zeros(len(z_array)), zorder=10, color='C0', linewidth=0.8, linestyle='--')
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlim([0, 1.5])
    ax2.tick_params(labelsize=12)

    plt.show()


def degradation(radii):
    pickle_in = open("MICE_SN_data.pickle", "rb")
    SN_data = pickle.load(pickle_in)
    pickle_in = open("MICEkappa_fis.pickle", "rb")
    kappa = pickle.load(pickle_in)
    corrs = []
    errs = []
    degreds = np.arange(0.05, 0.35, 0.01)
    for gauss_width in degreds:
        mu_diff = - (5.0 / np.log(10) * SN_data["SNkappa"])
        corr = []
        err = []
        for j in range(1000):
            for i in range(len(mu_diff)):
                mu_diff[i] += random.gauss(0.0, gauss_width)
            c, e = find_correlation(kappa, radii, mu_diff=mu_diff)
            corr.append(c)
            err.append(e)
        print(gauss_width)
        corrs.append(np.mean(corr, 0))
        errs.append(np.mean(err, 0))
    for num, rad in enumerate(radii):
        plt.plot(degreds, [corrs[i][num] for i in range(len(degreds))], label=f"{rad}'")
    plt.legend(frameon=0)
    plt.xlabel('$\sigma_{\kappa}$')
    plt.ylabel('$r$')
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    use_weighted = False
    alldata = get_data()
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    c = ax.plot(np.array(alldata['RA'])[np.logical_and(np.array(alldata['z']) < 0.15, np.array(alldata['DEC']) > 0)],
                np.array(alldata['z'])[np.logical_and(np.array(alldata['z']) < 0.15, np.array(alldata['DEC']) > 0)],
                color=colours[0], marker='.', markersize=2, linestyle='', alpha=0.5)
    ax.set_thetamin(240)
    ax.set_thetamax(300)
    ax.set_theta_zero_location("W")
    plt.show()

    big_cone_centre = [(min(alldata['RA']) + max(alldata['RA'])) / 2, (min(alldata['DEC']) + max(alldata['DEC'])) / 2]
    big_cone_radius = round(min(max(alldata['RA']) - big_cone_centre[0], big_cone_centre[0] - min(alldata['RA']),
                                max(alldata['DEC']) - big_cone_centre[1], big_cone_centre[1] - min(alldata['DEC'])), 2)
    big_cone = make_big_cone(alldata, redo=False)
    exp_data = find_expected(big_cone, big_cone_radius, 111, redo=False, plot=False)
    get_random(alldata, redo=False)
    # plot_cones(alldata, plot_hist=True, cone_radius=6.0)
    # plot_Hubble()

    kappa = find_convergence(alldata, exp_data, redo=False, plot_total=False, plot_scatter=False, weighted=use_weighted)
    use_weighted = not use_weighted
    kappa_weighted = find_convergence(alldata, exp_data, redo=False, plot_total=False, plot_scatter=True,
                                      weighted=use_weighted)
    # degradation([5.0, 6.0, 7.0, 10.0, 12.0])
    pickle_in = open("MICE_SN_data.pickle", "rb")
    SN_data = pickle.load(pickle_in)
    # pickle_in = open("random_cones_new.pickle", "rb")
    # lens_data = pickle.load(pickle_in)
    # SN_z = SN_data["SNZ"]
    SN_kappa = SN_data["SNkappa"]
    # SN_mu = SN_data['SNMU']
    # SN_mu_err = SN_data['SNMU_ERR']
    # SN_chi = []
    # gal_zs = {}
    # for z in SN_z:
    #     chi = Convergence.comoving(np.linspace(0, z, 1001), OM=0.25, OL=0.75, h=0.7)
    #     SN_chi.append(chi[-1])
    # data = {}
    # for rad in RADII:
    #     data[f'Radius{rad}'] = {}
    #     for j, (key, SN) in enumerate(lens_data[f"Radius{rad}"].items()):
    #         if key != "WEIGHT":
    #             cone_IDs = np.array([], dtype=np.int16)
    #             for r in RADII[0:np.argmin(np.abs(RADII - np.array(rad)))]:
    #                 cone_IDs = np.append(cone_IDs, lens_data[f"Radius{r}"][f"Shell{j+1}"])
    #             gal_zs[key] = alldata['z'][cone_IDs]
    #             data[f'Radius{rad}'][key] = {"Zs": gal_zs[key], "SNZ": SN_z[j], "SNMU": SN_mu[j],
    #                                          "SNMU_ERR": SN_mu_err[j], "WEIGHT": lens_data[f"Radius{rad}"]['WEIGHT'][j]}
    # print(data["Radius30.0"].keys())
    # cones_MICE_conv = cones.find_convergence(data, exp_data, redo=False, plot_scatter=False, plot_total=True, MICE=True,
    #                                          weighted=True, max_z=1.5)
    pickle_in = open("MICEkappa.pickle", "rb")
    cones_MICE_conv = pickle.load(pickle_in)
    pickle_in = open("MICEkappa_weighted.pickle", "rb")
    cones_MICE_conv_weighted = pickle.load(pickle_in)
    unweighted = find_correlation(cones_MICE_conv, RADII, plot_radii=True)
    weighted = find_correlation(cones_MICE_conv_weighted, RADII, plot_radii=True)

    # use_weighted = True
    # lensing_gals_fully_in_sample = {}
    # number_fis = np.zeros(len(RADII))
    # for n, rad in enumerate(RADII):
    #     lensing_gals_fully_in_sample[f"Radius{rad}"] = {}
    #     for key2, SN in data[f"Radius{rad}"].items():
    #         if SN["WEIGHT"] == 1:
    #             lensing_gals_fully_in_sample[f"Radius{rad}"][key2] = SN
    #             number_fis[n] += 1
    # plt.plot(RADII, number_fis, '+')
    # plt.show()
    kappa_fis = find_convergence(alldata, exp_data, redo=False, plot_total=False, plot_scatter=False, weighted=False,
                                 fis=True)
    fully_in_sample = find_correlation(kappa_fis, RADII, plot_correlation=False, plot_radii=True, fis=True)
    # print(fully_in_sample[0][12], fully_in_sample[4][12])
    fig, ax = plt.subplots()
    # ax2 = fig.add_axes([0.55, 0.5, 0.35, 0.35])
    ax.plot([0, 30], [0, 0], color=grey, linestyle='--')
    ax.plot(RADII, unweighted[1], color=colours[0])
    ax.plot(RADII, unweighted[0], marker='x', linestyle='', color=[0, 0.5, 0.9])
    ax.fill_between(RADII, unweighted[2], unweighted[3], color=colours[0], alpha=0.3)
    ax.plot(RADII, weighted[1], color=colours[1])
    ax.plot(RADII, weighted[0], marker='x', linestyle='', color=[0.7, 0.3, 0])
    ax.fill_between(RADII, weighted[2], weighted[3], color=colours[1], alpha=0.3)
    ax.plot(RADII, fully_in_sample[1], color=colours[2])
    ax.plot(RADII, fully_in_sample[0], marker='x', linestyle='', color=[0.7, 0.1, 0.6])
    ax.fill_between(RADII, fully_in_sample[2], fully_in_sample[3], color=colours[2], alpha=0.3)
    kwargs1 = {'marker': 'x', 'markeredgecolor': [0, 0.5, 0.9], 'color': colours[0]}
    kwargs2 = {'marker': 'x', 'markeredgecolor': [0.7, 0.3, 0], 'color': colours[1]}
    kwargs3 = {'marker': 'x', 'markeredgecolor': [0.7, 0.1, 0.6], 'color': colours[2]}
    ax.plot([], [], label='Unweighted', **kwargs1)
    ax.plot([], [], label='Weighted', **kwargs2)
    ax.plot([], [], label='Fully In Sample', **kwargs3)
    ax.invert_yaxis()
    ax.set_xlim([0, 30])
    # ax.set_ylim([0.2, 0.65])
    ax.legend(frameon=0, loc='lower left')
    ax.set_xlabel('Cone Radius (arcmin)')
    ax.set_ylabel("Spearman's Rank Coefficient")
    # ax2.plot([0, 30], [0, 0], color=grey, linestyle='--')
    # ax2.plot(RADII, unweighted[1], color=colours[0])
    # ax2.plot(RADII, unweighted[0], marker='x', linestyle='', color=[0, 0.5, 0.9])
    # ax2.fill_between(RADII, unweighted[2], unweighted[3], color=colours[0], alpha=0.3)
    # ax2.plot(RADII, weighted[1], color=colours[1])
    # ax2.plot(RADII, weighted[0], marker='x', linestyle='', color=[0.7, 0.3, 0])
    # ax2.fill_between(RADII, weighted[2], weighted[3], color=colours[1], alpha=0.3)
    # ax2.plot(RADII, fully_in_sample[1], color=colours[2])
    # ax2.plot(RADII, fully_in_sample[0], marker='x', linestyle='', color=[0.7, 0.1, 0.6])
    # ax2.fill_between(RADII, fully_in_sample[2], fully_in_sample[3], color=colours[2], alpha=0.3)
    # ax2.set_xlim([0.6, 6.6])
    # ax2.set_ylim([0.535, 0.62])
    plt.show()

    conv_total = []
    conv_total_weighted = []
    conv_total_fis = []
    conv_total_MICE = sum(SN_kappa)
    for cone_radius in RADII:
        conv_total.append(kappa[f"Radius{str(cone_radius)}"]["Total"])
        conv_total_weighted.append(kappa_weighted[f"Radius{str(cone_radius)}"]["Total"])
        conv_total_fis.append(kappa_fis[f"Radius{str(cone_radius)}"]["Total"])
    plt.ylabel("Total Convergence")
    plt.xlabel("Cone Radius (arcmin)")
    plt.tick_params(labelsize=12)
    plt.plot([0, 30], [0, 0], color=grey, linestyle='--')
    plt.xlim([0, 30])
    plt.plot(RADII, conv_total, marker='o', markersize=2, color=colours[0], label='Unweighted')
    plt.plot(RADII, conv_total_weighted, marker='o', markersize=2, color=colours[1], label='Weighted')
    plt.plot(RADII, conv_total_fis, marker='o', markersize=2, color=colours[2], label='Fully in sample')
    plt.plot(RADII, [conv_total_MICE for i in range(84)], marker='o', markersize=2, color=colours[3], label='MICECAT')
    plt.legend(frameon=0)
    plt.show()

