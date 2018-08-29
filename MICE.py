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

colours = [[0, 150/255, 100/255], [225/255, 149/255, 0], [207/255, 0, 48/255], 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5']
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


def find_convergence_MICE(gal_chi, gal_z, limits, exp, chi_widths, chi_bins, z_bins, gal_kappa, data, redo=False):
    if redo:
        counts = {}
        num = 0
        for z in gal_z:
            bin_c = range(int(np.argmin(np.abs(limits - z))))
            counts[f"g{num}"] = np.zeros(len(bin_c))
            for num2 in bin_c:
                counts[f"g{num}"][num2] = sum([limits[num2] < data[f'SN{num+1}']['zs'][i] <= limits[num2 + 1]
                                               for i in range(len(data[f'SN{num+1}']['zs']))])
            if sum(counts[f"g{num}"]) == 0:
                print(f'Total void for {data[f"SN{num+1}"]}')
            num += 1
            if num % 5 == 0:
                print(f"Finished {num}/{len(gal_z)}")
        convergence = np.zeros(len(counts))
        conv_err = np.zeros(len(counts))
        num = 0
        d_arr = {}
        for key, SN in counts.items():
            d_arr[key] = (SN - exp[:len(SN)]) / exp[:(len(SN))]
            convergence[num] = Convergence.general_convergence(chi_widths[:len(SN)], chi_bins[:len(SN)], z_bins[:len(SN)],
                                                   d_arr[f"{key}"], gal_chi[num])
            conv_err[num] = Convergence.convergence_error(chi_widths[:len(SN)], chi_bins[:len(SN)], z_bins[:len(SN)],
                                              exp[:len(SN)], gal_chi[num])
            num += 1

        conv_err += 0
        pickle_out = open("conv_sim.pickle", "wb")
        pickle.dump([convergence, conv_err], pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("conv_sim.pickle", "rb")
        conv_sim = pickle.load(pickle_in)
        convergence = conv_sim[0]
        conv_err = conv_sim[1]

    ax = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    ax.set_ylabel("$\kappa$")
    ax.set_xlabel("$z$")
    ax2.set_xlabel("Count")
    ax.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    ax2.set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    ax.plot([0, 1.5], [0, 0], color=grey, linestyle='--')
    ax.axis([0, 1.5, -0.06, 0.12])
    ax2.axis([0, 330, -0.06, 0.12])
    # ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0])
    ax.set_xticklabels([0, 0.5, 1.0, 0])
    ax.plot(gal_z, convergence, linestyle='', marker='o', markersize=2, color=colours[0], label="Cone Method")
    ax.plot(gal_z, gal_kappa, linestyle='', marker='o', markersize=2, color=colours[1], label="Actual Value")
    ax2.hist(convergence, bins=np.arange(-0.12, 0.13 + 0.025/4, 0.025/4), orientation='horizontal',
             fc=green, edgecolor=colours[0])
    ax2.hist(gal_kappa, bins=np.arange(-0.12, 0.13 + 0.025/4, 0.025/4), orientation='horizontal',
             fc=yellow, edgecolor=colours[1])
    ax.legend(frameon=0)
    plt.show()

    plt.plot(gal_kappa, convergence, linestyle='', marker='o', markersize=2, color=colours[0])
    plt.plot(gal_kappa, gal_kappa, linestyle='--', color=colours[1])
    plt.ylim([-0.05, 0.04])
    plt.show()

    return convergence


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
        z = hdul1[1].data['z']
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
            big_cone['Zs'] = np.append(big_cone['Zs'], zs[(RAs - centre[0] + 2 * i * radius) ** 2 + ( DECs - centre[1]) ** 2 <= radius ** 2])
            big_cone['kappa'] = np.append(big_cone['kappa'], kappas[(RAs - centre[0] + 2 * i * radius) ** 2 + (DECs - centre[1]) ** 2 <= radius ** 2])
            big_cone['Zs'] = np.append(big_cone['Zs'], zs[(RAs - centre[0] - 2 * i * radius) ** 2 + (DECs - centre[1]) ** 2 <= radius ** 2])
            big_cone['kappa'] = np.append(big_cone['kappa'], kappas[(RAs - centre[0] - 2 * i * radius) ** 2 + (DECs - centre[1]) ** 2 <= radius ** 2])

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
    chi_bin_widths, chi_bins, z_bins, z_bin_widths = Convergence.create_z_bins(0.01, max_z, bins)
    limits = np.cumsum(z_bin_widths) + z_bins[0]
    expected = {}
    if redo:
        cumul_counts = []
        for num1, lim in enumerate(limits):
            cumul_counts.append(sum(big_cone['Zs'][big_cone['Zs'] < lim]) / 5.0)
            print(f"Sorted {num1+1}/{len(limits)}")

        pickle_out = open("MICEexpected.pickle", "wb")
        pickle.dump(cumul_counts, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("MICEexpected.pickle", "rb")
        cumul_counts = pickle.load(pickle_in)

    expected_big = np.diff([cumul_counts[i] for i in range(len(limits))])
    for cone_radius in RADII:
        expected[f"Radius{str(cone_radius)}"] = [expected_big[i] * (cone_radius / r_big / 60.0) ** 2
                                                 for i in range(len(expected_big))]

    if plot:
        for cone_radius in RADII:
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
    IDs = data['id']
    flagged_data = {}
    # Don't want to deal with up to 30' (0.5 degrees) cones that have any portion outside left and right bounds.
    SN_DECs = DECs[RAs < max(RAs) - 0.5]
    SN_zs = zs[RAs < max(RAs) - 0.5]
    SN_kappas = kappas[RAs < max(RAs) - 0.5]
    SN_RAs = RAs[RAs < max(RAs) - 0.5]
    SN_DECs = SN_DECs[SN_RAs > min(RAs) + 0.5]
    SN_zs = SN_zs[SN_RAs > min(RAs) + 0.5]
    SN_kappas = SN_kappas[SN_RAs > min(RAs) + 0.5]
    SN_RAs = SN_RAs[SN_RAs > min(RAs) + 0.5]

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
            chi_to_z = Convergence.comoving(np.linspace(0, z, 1001), OM=0.25, OL=0.75)
            dists.append(chi_to_z[-1] * (1 + z))
            rand_chis.append(chi_to_z[-1])
        mus = 5 * np.log10(np.array(dists) / 10 * 1E9)
        rand_mus = mus * (1 - 5 / np.log(10) * rand_kappas)
        rand_errs = np.array([abs(random.uniform(0.1+0.3*rand_zs[i], 0.14+0.75*rand_zs[i]))
                              for i in range(rand_samp_size)])

        SN_data = find_mu_diff(rand_zs, rand_mus)
        SN_data['SNZ'] = rand_zs
        SN_data['SNkappa'] = rand_kappas
        SN_data['SNRA'] = rand_RAs
        SN_data['SNDEC'] = rand_DECs
        SN_data['SNMU'] = rand_mus
        SN_data['SNMU_ERR'] = rand_errs
        pickle_out = open("MICE_SN_data.pickle", "wb")
        pickle.dump(SN_data, pickle_out)
        pickle_out.close()

        # Radii to split files into:
        lenses = {}
        for cone_radius in RADII:
            flagged_data[f"Radius{str(cone_radius)}"] = np.zeros(len(RAs), dtype=np.int8)
            for num, (RA, DEC) in enumerate(zip(rand_RAs, rand_DECs)):
                cone_indices = [(RAs - RA) ** 2 + (DECs - DEC) ** 2 <= (cone_radius / 60.0) ** 2]
                flagged_data[f"Radius{str(cone_radius)}"][cone_indices] = [num+1]
                print(flagged_data[f"Radius{str(cone_radius)}"][cone_indices])
            heights = np.zeros(rand_samp_size)
            outsides_u = [rand_DECs > 10.1 - cone_radius / 60.0]
            heights[outsides_u] = rand_DECs[outsides_u] - (10.1 - cone_radius / 60.0)
            outsides_d = [rand_DECs < cone_radius / 60.0]
            heights[outsides_d] = cone_radius / 60.0 - rand_DECs[outsides_d]
            thetas = 2 * np.arccos(1 - heights / (cone_radius / 60.0))
            fraction_outside = 1 / (2 * np.pi) * (thetas - np.sin(thetas))
            weights = 1.0 - fraction_outside
            lenses[f"Radius{str(cone_radius)}"] = {'WEIGHT': weights}
            print(f"Sorted radius {cone_radius}'")
        pickle_out = open(f"random_cones_new.pickle", "wb")
        pickle.dump(lenses, pickle_out)
        pickle_out.close()


def plot_cones(data, sorted_data, plot_hist=False, cone_radius=12.0):
    """Plots all galaxies and SNe along with visualisation of cones and galaxies contributing to lensing.

    Input:
     cut_data -- dictionary that contains all data (RA, DEC, z, etc.) of galaxies.
     sorted_data -- dictionary that contains all information for every SN sorted into cones.
     plot_hist -- boolean that determines if a histogram of the galaxy and SNe distribution is plotted. Defaults to
                  False.
     cone_radius -- the radius of the cones. Defaults to 12'.
    """
    lenses = sorted_data[f"Radius{str(cone_radius)}"]
    patches = []
    SNRA = []
    SNDEC = []
    SNz = []
    for SN, dict1, in lenses.items():
        SNRA.append(dict1['SNRA'])
        SNDEC.append(dict1['SNDEC'])
        SNz.append(dict1['SNZ'])
    for x, y in zip(SNRA, SNDEC):
        circle = Circle((x, y), cone_radius/60.0)
        patches.append(circle)

    RA_gal = data['RA']
    DEC_gal = data['DEC']
    z_gal = data['z']
    fig, ax = plt.subplots()
    ax.plot(RA_gal, DEC_gal, marker='o', linestyle='', markersize=1, color=[0.5, 0.5, 0.5])
    contRAs = []
    contDECs = []
    for SN, dict1, in lenses.items():
        RAs = np.array(dict1['RAs'])
        DECs = np.array(dict1['DECs'])
        indices1 = dict1['Zs'] <= dict1['SNZ']
        contRAs = np.append(contRAs, RAs[indices1])
        contDECs = np.append(contDECs, DECs[indices1])
    ax.plot(contRAs, contDECs, marker='o', linestyle='', markersize=3, color=colours[1])
    p = PatchCollection(patches, alpha=0.4, color=colours[1])
    ax.add_collection(p)

    ax.plot(SNRA, SNDEC, marker='o', linestyle='', markersize=3, label='Supernova', color=colours[3])
    plt.xlabel('$\\alpha$')
    plt.ylabel('$\delta$')
    plt.text(27, -0.8, f"{cone_radius}' radius")
    # plt.legend(loc='lower right')
    plt.axis('equal')
    plt.xlim([24.5, 27.5])
    plt.ylim([-1, 1])
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


def find_convergence(exp_data, redo=False, plot_scatter=False, plot_total=False, weighted=False):
    """Finds the convergence along each line of sight to a SN for a variety of cone_widths.

    Inputs:
     exp_data -- dictionary containing all expected counts per bin per cone width.
     SNz -- redshifts of each SN.
     redo -- boolean that determines whether convergence is calculated or loaded. Dafault false.
     plot_scatter -- boolean that determined whether scatter plot of convergence per SN redshift is plotted.
                     Default false.
     plot_total -- boolean that determines whether total convergence per cone radius is plotted. Default false.
    """
    limits = exp_data[0]
    chi_widths = exp_data[2]
    chi_bis = exp_data[3]
    z_bins = exp_data[4]
    if redo:                                                                                              #if redo:
        pickle_in = open("MICE_SN_data.pickle", "rb")                                                     #    counts = {}
        SN_data = pickle.load(pickle_in)                                                                  #    num = 0
        SN_zs = SN_data["SNZ"]                                                                            #    for z in gal_z:
        pickle_in = open("MICEkappa.pickle", "rb")                                                        #        bin_c = range(int(np.argmin(np.abs(limits - z))))
        kappa = pickle.load(pickle_in)                                                                    #        counts[f"g{num}"] = np.zeros(len(bin_c))
        r_i = 0                                                                                           #        for num2 in bin_c:
        r_i_old = 0                                                                                       #            counts[f"g{num}"][num2] = sum([limits[num2] < data[f'SN{num+1}']['zs'][i] <= limits[num2 + 1]
        for i in [1, 2, 3]:                                                                               #                                           for i in range(len(data[f'SN{num+1}']['zs']))])
            with open(f"random_cones{i}.pickle", "rb") as pickle_in:                                      #        if sum(counts[f"g{num}"]) == 0:
                lens_data = pickle.load(pickle_in)                                                        #            print(f'Total void for {data[f"SN{num+1}"]}')
            r_i += len(lens_data.keys())                                                                  #        num += 1
            for cone_radius in RADII[r_i_old:r_i]:                                                        #        if num % 5 == 0:
                Zs = []                                                                                   #            print(f"Finished {num}/{len(gal_z)}")
                SN_weights = []                                                                           #    convergence = np.zeros(len(counts))
                for SN_key in lens_data[f"Radius{cone_radius}"].keys():                                   #    conv_err = np.zeros(len(counts))
                    Zs.append(lens_data[f"Radius{cone_radius}"][SN_key]["Zs"])                            #    num = 0
                    SN_weights.append(lens_data[f"Radius{cone_radius}"][SN_key]["WEIGHT"])                #    d_arr = {}
                                                                                                          #    for key, SN in counts.items():
                expected_counts = exp_data[1][f"Radius{str(cone_radius)}"]                                #        d_arr[key] = (SN - exp[:len(SN)]) / exp[:(len(SN))]
                kappa[f"Radius{str(cone_radius)}"] = {"SNkappa": [], "Total": 0}                          #        convergence[num] = general_convergence(chi_widths[:len(SN)], chi_bins[:len(SN)], z_bins[:len(SN)],
                d_arr = []                                                                                #                                               d_arr[f"{key}"], gal_chi[num])
                counts = []                                                                               #         conv_err[num] = convergence_error(chi_widths[:len(SN)], chi_bins[:len(SN)], z_bins[:len(SN)],
                for num, z in enumerate(SN_zs):
                    bin_c = range(int(np.argmin(np.abs(limits - z))))
                    counts.append(np.zeros(len(bin_c)))
                    for num2 in bin_c:
                        tmp = [np.logical_and(limits[num2] < z, z <= limits[num2 + 1])]
                        if weighted:
                            counts[num][num2] = np.count_nonzero(tmp) / SN_weights[num]
                        else:
                            counts[num][num2] = np.count_nonzero(tmp)

                    # print(f"Counted SN {num+1}/1500")

                chiSNs = []
                for z in SN_zs:
                    chi = Convergence.comoving(np.linspace(0, z, 1001))
                    chiSNs.append(chi[-1])

                for num, SN in enumerate(counts):
                    d_arr.append((SN - expected_counts[:len(SN)]) / expected_counts[:(len(SN))])
                    SNkappa, _ = Convergence.general_convergence(chi_widths[:len(SN)], chi_bis[:len(SN)],
                                                                 z_bins[:len(SN)], d_arr[num], chiSNs[num])
                    kappa[f"Radius{str(cone_radius)}"]["SNkappa"].append(SNkappa)

                kappa[f"Radius{str(cone_radius)}"]["Total"] = np.sum(kappa[f"Radius{str(cone_radius)}"]["SNkappa"])
                print(f"Finished radius {str(cone_radius)}'")
            r_i_old = r_i
        pickle_out = open("MICEkappa.pickle", "wb")
        pickle.dump(kappa, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("MICEkappa.pickle", "rb")
        kappa = pickle.load(pickle_in)
        pickle_in = open("MICE_SN_data.pickle", "rb")
        SN_data = pickle.load(pickle_in)
        SN_zs = SN_data["SNZ"]

    for cone_radius in RADII:
        bins = np.linspace(0.025, 1.5 - 0.025, 12)
        mean_kappa = []
        standard_error = []
        conv = kappa[f"Radius{str(cone_radius)}"]["SNkappa"]

        for b in bins:
            ks = []
            for z, k in zip(SN_zs, conv):
                if b - 0.025 < z <= b + 0.025:
                    ks.append(k)
            mean_kappa.append(np.mean(ks))
            standard_error.append(np.std(ks) / np.sqrt(len(ks)))

        if plot_scatter:
            conv = kappa[f"Radius{str(cone_radius)}"]["SNkappa"]
            # ax = plt.subplot2grid((1, 4), (0, 0), colspan=3)
            ax = plt.subplot2grid((1, 1), (0, 0))
            # ax2 = plt.subplot2grid((1, 4), (0, 3))
            ax.set_ylabel("$\kappa$")
            ax.set_xlabel("$z$")
            # ax2.set_xlabel("Count")
            ax.tick_params(labelsize=12)
            # ax2.tick_params(labelsize=12)
            # ax2.set_yticklabels([])
            plt.subplots_adjust(wspace=0, hspace=0)
            ax.plot([0, 1.5], [0, 0], color=grey, linestyle='--')
            ax.axis([0, 1.5, -0.01, 0.01])
            # ax2.axis([0, 180, -0.01, 0.01])
            # ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0])
            # ax.set_xticklabels([0, 0.2, 0.4, 0])
            ax.plot(SN_zs, conv, linestyle='', marker='o', markersize=2, color=colours[0])
            # ax2.hist(conv_total, bins=np.arange(-0.015, 0.02 + 0.001, 0.001), orientation='horizontal',
            #          fc=green, edgecolor=colours[0])
            ax.errorbar(bins, mean_kappa, standard_error, marker='s', color='r', markersize=3, capsize=3)
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


def find_correlation(convergence_data, plot_correlation=False, plot_radii=False):
    """Finds the value of the slope for plotting residuals against convergence. Magnitude of slope and error
    quantify correlation between the two.

    Inputs:
     conv -- convergence.
     mu_diff -- residuals.
    """
    correlations = []
    correlation_errs = []
    for cone_radius in RADII:
        pickle_in = open("MICE_SN_data.pickle", "rb")
        SN_data = pickle.load(pickle_in)
        redshift_cut = [SN_data['SNZ'] > 0.2]
        mu_diff = SN_data["mu_diff"][redshift_cut]
        conv = np.array(convergence_data[f"Radius{str(cone_radius)}"]["SNkappa"])[redshift_cut]
        conv_mean = np.mean(conv)
        mu_mean = np.mean(mu_diff)
        conv_std = np.std(conv)
        mu_std = np.std(mu_diff)
        r = 1 / (len(conv) - 1) * np.sum(((mu_diff - mu_mean) / mu_std) * ((conv - conv_mean) / conv_std))
        r_err = np.sqrt((1 - r ** 2) / (len(conv) - 1))

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

    u_err = [correlations[i] + correlation_errs[i] for i in range(len(correlations))]
    d_err = [correlations[i] - correlation_errs[i] for i in range(len(correlations))]
    smooth_corr = savgol_filter([correlations[i] for i in range(len(correlations))], 11, 4)
    smooth_u_err = savgol_filter(u_err, 11, 4)
    smooth_d_err = savgol_filter(d_err, 11, 4)
    if plot_radii:
        plt.plot([0, 30], [0, 0], color=grey, linestyle='--')
        plt.plot(RADII, smooth_corr, color=colours[0])
        plt.plot(RADII, [correlations[i] for i in range(len(correlations))], marker='x', color=colours[1],
                 linestyle='')
        plt.fill_between(RADII, smooth_u_err, smooth_d_err, color=colours[0], alpha=0.4)

        plt.xlabel('Cone Radius (arcmin)')
        plt.ylabel("Spearman's Rank Coefficient")
        plt.show()

    return [correlations, smooth_corr, smooth_u_err, smooth_d_err]


def find_mu_diff(SN_zs, SN_mus, OM=0.25, OL=0.75):
    """Finds the distance modulus of best fitting cosmology and hence residuals.
    """
    z_array = np.linspace(0.0, 1.5 + 0.01, 1001)
    mu_cosm = 5 * np.log10((1 + z_array) * Convergence.comoving(z_array, OM=OM, OL=OL) * 1000) + 25
    mu_cosm_interp = np.interp(SN_zs, z_array, mu_cosm)
    mu_diff = SN_mus - mu_cosm_interp
    data = {"mu_diff": mu_diff, "mu_cosm": mu_cosm, "z_array": z_array}
    return data


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
    mu_cosm = SN_data['mu_cosm']
    mu_diff = SN_data['mu_diff']
    z_arr = SN_data['z_array']
    ax = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    ax.set_ylabel("$\mu$")
    ax2.set_xlabel("$z$")
    ax2.set_ylabel("$\Delta\mu$")
    plt.subplots_adjust(wspace=0, hspace=0)
    ax.set_xticklabels([])
    ax.tick_params(labelsize=12)
    ax.errorbar(z, mu, mu_err, linestyle='', linewidth=0.8, marker='o',
                markersize=2, capsize=2, color='C3', zorder=0)
    ax.set_ylim([35, max(mu)+1])
    ax.set_xlim([0, 1.5])
    ax.plot(z_arr, mu_cosm, linestyle='--', linewidth=0.8, color='C0', zorder=10)
    ax2.errorbar(z, mu_diff, mu_err, linestyle='', linewidth=1, marker='o',
                 markersize=2, capsize=2, color='C3', zorder=0)
    ax2.plot(z_arr, np.zeros(len(z_arr)), zorder=10, color='C0', linewidth=0.8, linestyle='--')
    ax2.set_ylim(-1.4, 1.4)
    ax2.set_xlim([0, 1.5])
    ax2.tick_params(labelsize=12)

    plt.show()


if __name__ == "__main__":
    use_weighted = True
    alldata = get_data()
    big_cone_centre = [(min(alldata['RA']) + max(alldata['RA'])) / 2, (min(alldata['DEC']) + max(alldata['DEC'])) / 2]
    big_cone_radius = round(min(max(alldata['RA']) - big_cone_centre[0], big_cone_centre[0] - min(alldata['RA']),
                                max(alldata['DEC']) - big_cone_centre[1], big_cone_centre[1] - min(alldata['DEC'])), 2)
    big_cone = make_big_cone(alldata, redo=False)
    exp_data = find_expected(big_cone, big_cone_radius, 111, redo=False, plot=False)
    get_random(alldata, redo=True)
    # plot_cones(alldata, sorted_data, plot_hist=True)
    # plot_Hubble()
    conv = find_convergence(exp_data, redo=False, plot_total=True, plot_scatter=False, weighted=use_weighted)
    # cones.find_correlation(MICEconv, sorted_data, plot_correlation=True)
    find_correlation(conv, plot_radii=True)
