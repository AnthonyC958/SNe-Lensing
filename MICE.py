import cones
import Convergence
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import random
from scipy.stats import rankdata
import csv
import pickle
from scipy.signal import savgol_filter
import collections

colours = [[0, 150/255, 100/255], [225/255, 149/255, 0], [207/255, 0, 48/255], 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5']
blue = [23/255, 114/255, 183/255, 0.75]
orange = [255/255, 119/255, 15/255, 0.75]
green = [0, 150/255, 100/255, 0.75]
yellow = [253/255, 170/255, 0, 0.75]
grey = [0.75, 0.75, 0.75]
RADII = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
         4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75,
         9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75, 13.0,
         13.25, 13.5, 13.75, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0,
         23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]

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
    with fits.open('MICEsim4.fits') as hdul1:
        RA = hdul1[1].data['ra_gal']
        DEC = hdul1[1].data['dec_gal']
        z = hdul1[1].data['z_cgal']
        kap = hdul1[1].data['kappa']

    RA = np.array(RA)[[z >= 0.01]]
    DEC = np.array(DEC)[[z >= 0.01]]
    z = np.array(z)[[z >= 0.01]]
    kap = np.array(kap)[[z >= 0.01]]
    cut_data = {'RA': RA, 'DEC': DEC, 'z': z, 'kappa': kap}

    return cut_data


def make_big_cone(data, redo=False):
    if redo:
        RAs = data['RA']
        DECs = data['DEC']
        zs = data['z']
        kappas = data['kappa']
        centre = [(min(RAs) + max(RAs)) / 2, (min(DECs) + max(DECs)) / 2]
        radius = round(min(max(RAs) - centre[0], centre[0] - min(RAs), max(DECs) - centre[1], centre[1] - min(DECs)), 2)
        big_cone = {'Zs': zs[(RAs - centre[0]) ** 2 + (DECs - centre[1]) ** 2 <= radius ** 2],
                    'kappa': kappas[(RAs - centre[0]) ** 2 + (DECs - centre[1]) ** 2 <= radius ** 2]}

        pickle_out = open(f"big_cone.pickle", "wb")
        pickle.dump(big_cone, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open(f"big_cone.pickle", "rb")
        big_cone = pickle.load(pickle_in)

    return big_cone


def get_random(data, redo=False):
    RAs = data['RA']
    DECs = data['DEC']
    zs = data['z']
    kappas = data['kappa']
    # Don't want to deal with up to 30' (0.5 degrees) cones that have any portion outside left and right bounds.
    SN_DECs = DECs[RAs < max(RAs) - 0.5]
    SN_zs = zs[RAs < max(RAs) - 0.5]
    SN_kappas = kappas[RAs < max(RAs) - 0.5]
    SN_RAs = RAs[RAs < max(RAs) - 0.5]
    SN_DECs = SN_DECs[SN_RAs > min(RAs) + 0.5]
    SN_zs = SN_zs[SN_RAs > min(RAs) + 0.5]
    SN_kappas = SN_kappas[SN_RAs > min(RAs) + 0.5]
    SN_RAs = SN_RAs[SN_RAs > min(RAs) + 0.5]
    cone_radius = 12.0

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
        rand_mus = mus * (1 - 2 * rand_kappas)
        rand_errs = np.array([abs(random.uniform(0.14+0.43*rand_zs[i], 0.14+0.76*rand_zs[i]))
                              for i in range(rand_samp_size)])

        # Weight cones based off fraction outside sample
        heights = np.zeros(rand_samp_size)
        outsides_u = [rand_DECs > 10.1 - cone_radius / 60.0]
        heights[outsides_u] = rand_DECs[outsides_u] - (10.1 - cone_radius / 60.0)
        outsides_d = [rand_DECs < cone_radius / 60.0]
        heights[outsides_d] = cone_radius / 60.0 - rand_DECs[outsides_d]
        thetas = 2 * np.arccos(1 - heights / (cone_radius / 60.0))
        fraction_outside = 1 / (2 * np.pi) * (thetas - np.sin(thetas))
        weights = 1.0 - fraction_outside

        r_cuts = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84]
        file = 1
        for i in range(11):
            lenses = {}
            for cone_radius in RADII[r_cuts[i]:r_cuts[i+1]]:
                lenses[f"Radius{str(cone_radius)}"] = {}
                for num, (RandRA, RandDEC, Randz, Randkappa) in enumerate(zip(rand_RAs, rand_DECs, rand_zs,
                                                                              rand_kappas)):

                    lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}'] = {'IPs': [], 'Zs': [],
                                                                              'SNZ': Randz, 'SNkappa': Randkappa,
                                                                              'SNRA': RandRA, 'SNDEC': RandDEC,
                                                                              'SNMU': rand_mus[num],
                                                                              'SNMU_ERR': rand_errs[num],
                                                                              'WEIGHT': weights[num]}
                    cone_indices = [(RAs - RandRA) ** 2 + (DECs - RandDEC) ** 2 <= (cone_radius/60.0) ** 2]
                    lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['Zs'] = zs[cone_indices]
                    lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['IPs'] = (RAs[cone_indices] - RandRA) ** 2 +\
                                                                                    (DECs[cone_indices] - RandDEC) ** 2

                    print(f"Sorted {num+1}/{rand_samp_size} for radius {cone_radius}'")
            pickle_out = open(f"random_cones{file}.pickle", "wb")
            pickle.dump(lenses, pickle_out)
            pickle_out.close()
            print(f"Finished file {file}")
            file += 1

        lenses = {}
        for x in [1, 2, 3, 4]:
            pickle_in = open(f"random_cones_q{x}.pickle", "rb")
            rand_cones_quarter = pickle.load(pickle_in)
            lenses = deep_update(lenses, rand_cones_quarter)
    else:
        lenses = {"Radius12.0": {}}
        for x in [1, 2, 3, 4]:
            pickle_in = open(f"random_cones_q{x}.pickle", "rb")
            rand_cones_quarter = pickle.load(pickle_in)
            lenses = deep_update(lenses, rand_cones_quarter)

    return lenses


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


def find_expected(big_cone, r_big, bins, redo=False, plot=False):
    max_z = 1.41
    cone_radius = 12.0
    chi_bin_widths, chi_bins, z_bins, z_bin_widths = Convergence.create_z_bins(0.065, max_z, bins)
    limits = np.cumsum(z_bin_widths) + z_bins[0]
    expected = {}
    if redo:
        cumul_counts = []
        for num1, lim in enumerate(limits):
            cumul_counts.append(sum(big_cone['Zs'][big_cone['Zs'] < lim]))
            print(f"Sorted {num1+1}/{len(limits)}")

        pickle_out = open("MICEexpected.pickle", "wb")
        pickle.dump(cumul_counts, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("MICEexpected.pickle", "rb")
        cumul_counts = pickle.load(pickle_in)
    expected_big = np.diff([cumul_counts[i] for i in range(len(limits))])
    expected[f"Radius{str(cone_radius)}"] = [expected_big[i] * (cone_radius / r_big / 60.0) ** 2
                                             for i in range(len(expected_big))]

    if plot:
        plt.plot([0, 5], [0, 0], color=grey, linestyle='--')
        plt.plot((limits[1:]+limits[:-1])/2.0, expected[f"Radius{str(cone_radius)}"], marker='o', markersize=2.5,
                 color=colours[0])
        plt.xlabel('$z$')
        plt.ylabel('Expected Count')
        plt.xlim([0, 1.5])
        plt.show()

    return [limits, expected, chi_bin_widths, chi_bins, z_bins]


if __name__ == "__main__":
    use_weighted = True
    alldata = get_data()
    big_cone_centre = [(min(alldata['RA']) + max(alldata['RA'])) / 2, (min(alldata['DEC']) + max(alldata['DEC'])) / 2]
    big_cone_radius = round(min(max(alldata['RA']) - big_cone_centre[0], big_cone_centre[0] - min(alldata['RA']),
                                max(alldata['DEC']) - big_cone_centre[1], big_cone_centre[1] - min(alldata['DEC'])), 2)
    big_cone = make_big_cone(alldata, redo=False)
    sorted_data = get_random(alldata, redo=True)
    # plot_cones(alldata, sorted_data, plot_hist=True)
    exp_data = find_expected(big_cone, big_cone_radius, 111, redo=False, plot=True)
    # cones.plot_Hubble(sorted_data, OM=0.25, OL=0.75, max_z=1.5)
    conv = cones.find_convergence(sorted_data, exp_data, redo=False, plot_scatter=True, weighted=use_weighted,
                                  max_z=1.5, MICE=True)
    MICEconv = {"Radius12.0": {"SNkappa": []}}
    for key, val in sorted_data["Radius12.0"].items():
        MICEconv["Radius12.0"]["SNkappa"].append(val["SNkappa"])

    # cones.find_correlation(MICEconv, sorted_data, plot_correlation=True)
    cones.find_correlation(conv, sorted_data, plot_correlation=True)
