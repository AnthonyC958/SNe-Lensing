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
            convergence[num] = Convergence.general_convergence(chi_widths[:len(SN)], chi_bins[:len(SN)],
                                                               z_bins[:len(SN)], d_arr[f"{key}"], gal_chi[num])
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


def get_data():
    with fits.open('MICEsim3.fits') as hdul1:
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
        pass

    RAs = data['RA']
    DECs = data['DEC']
    zs = data['z']
    kappas = data['kappa']
    centre = [(min(RAs) + max(RAs)) / 2, (min(DECs) + max(DECs)) / 2]
    radius = round(min(max(RAs) - centre[0], centre[0] - min(RAs), max(DECs) - centre[1], centre[1] - min(DECs)), 2)
    big_cone = {'Zs': zs[(RAs - centre[0]) ** 2 + (DECs - centre[1]) ** 2 <= radius ** 2],
                'kappa': kappas[(RAs - centre[0]) ** 2 + (DECs - centre[1]) ** 2 <= radius ** 2]}

    return big_cone, radius


def get_random(data, redo=False, weighted=True):
    RAs = data['RA']
    DECs = data['DEC']
    zs = data['z']
    kappas = data['kappa']
    # Don't want to deal with up to 30' (0.5 degrees) cones that have any portion outside left and right bounds.
    DECs = DECs[RAs < max(RAs) - 0.5]
    zs = zs[RAs < max(RAs) - 0.5]
    kappas = kappas[RAs < max(RAs) - 0.5]
    RAs = RAs[RAs < max(RAs) - 0.5]
    DECs = DECs[RAs > min(RAs) + 0.5]
    zs = zs[RAs > min(RAs) + 0.5]
    kappas = kappas[RAs > min(RAs) + 0.5]
    RAs = RAs[RAs > min(RAs) + 0.5]
    cone_radius = 12.0

    if redo:
        random.seed(1337)
        rand_samp_size = 1600
        q1 = rand_samp_size / 4
        q2 = rand_samp_size / 2
        q3 = 3 * rand_samp_size / 4
        indices = random.sample(range(len(zs)), rand_samp_size)
        rand_zs = zs[indices]
        rand_RAs = RAs[indices]
        rand_DECs = DECs[indices]
        rand_kappas = kappas[indices]
        dists = []
        rand_chis = []
        for z in rand_zs:
            chi_to_z = Convergence.comoving(np.linspace(0, z, 1001), OM=0.25, OL=0.75)
            dists.append(chi_to_z[-1] * (1 + z))
            rand_chis.append(chi_to_z[-1])
        mus = 5 * np.log10(np.array(dists) / 10 * 1E9)
        rand_mus = mus * (1 + 2 * rand_kappas)
        rand_errs = np.array([abs(random.gauss(0.0, 0.2)) for i in range(rand_samp_size)])
        weights = np.ones(rand_samp_size)
        if weighted:
            heights = np.zeros(rand_samp_size)
            outsides_u = [rand_DECs > 10.1 - cone_radius / 60.0]
            heights[outsides_u] = rand_DECs[outsides_u] - (10.1 - cone_radius / 60.0)
            outsides_d = [rand_DECs < cone_radius / 60.0]
            heights[outsides_d] = cone_radius / 60.0 - rand_DECs[outsides_d]
            thetas = 2 * np.arccos(1 - heights / (cone_radius / 60.0))
            fraction_outside = 1 / (2 * np.pi) * (thetas - np.sin(thetas))
            weights = 1.0 - fraction_outside

        for q_num, q in enumerate([[0, q1], [q1, q2], [q2, q3], [q3, rand_samp_size]]):
            lenses = {'Radius12.0': {}}
            s = int(q[0])
            e = int(q[1])
            for num, (RandRA, RandDEC, Randz, Randkappa) in enumerate(zip(rand_RAs[s:e], rand_DECs[s:e], rand_zs[s:e],
                                                                          rand_kappas[s:e])):
                if q_num == 0:
                    num += 0
                elif q_num == 1:
                    num += 400
                elif q_num == 2:
                    num += 800
                elif q_num == 3:
                    num += 1200

                lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}'] = {'RAs': [], 'DECs': [], 'Zs': [], 'SNZ': Randz,
                                                                          'SNkappa': Randkappa, 'SNRA': RandRA,
                                                                          'SNDEC': RandDEC, 'SNMU': rand_mus[num],
                                                                          'SNMU_ERR': rand_errs[num],
                                                                          'WEIGHT': weights[num]}
                indices = [(RAs - RandRA) ** 2 + (DECs - RandDEC) ** 2 <= (cone_radius/60.0) ** 2]
                lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['Zs'] = zs[indices]
                lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['RAs'] = RAs[indices]
                lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['DECs'] = DECs[indices]

                print(f"Sorted {num+1}/{rand_samp_size}")

            pickle_out = open(f"random_cones_q{q_num+1}.pickle", "wb")
            pickle.dump(lenses, pickle_out)
            pickle_out.close()
            print(f"Finished quarter {q_num+1}")

        lenses = {}
        for x in [1, 2, 3, 4]:
            pickle_in = open(f"random_cones_q{x}.pickle", "rb")
            rand_cones_quarter = pickle.load(pickle_in)
            lenses.update(rand_cones_quarter)
    else:
        lenses = {}
        lense12 = {}
        for x in [1, 2, 3, 4]:
            pickle_in = open(f"random_cones_q{x}.pickle", "rb")
            rand_cones_quarter = pickle.load(pickle_in)
            r12 = rand_cones_quarter["Radius12.0"]
            lense12.update(r12)
        lenses[f"Radius12.0"] = lense12

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
    for SN, dict1, in lenses.items():
        RAs = np.array(dict1['RAs'])
        DECs = np.array(dict1['DECs'])
        indices2 = dict1['Zs'] > dict1['SNZ']
    ax.plot(RAs[indices2], DECs[indices2], marker='o', linestyle='', markersize=1, color='k')
    for SN, dict1, in lenses.items():
        RAs = np.array(dict1['RAs'])
        DECs = np.array(dict1['DECs'])
        indices1 = dict1['Zs'] <= dict1['SNZ']
    ax.plot(RAs[indices1], DECs[indices1], marker='o', linestyle='', markersize=3, color=colours[3])
    p = PatchCollection(patches, alpha=0.4, color=colours[3])
    ax.add_collection(p)

    ax.plot(SNRA, SNDEC, marker='o', linestyle='', markersize=3, label='Supernova', color=colours[0])
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
    max_z = 1.5
    cone_radius = 12.0
    chi_bin_widths, chi_bins, z_bins, z_bin_widths = Convergence.create_z_bins(0.01, max_z, bins)
    limits = np.cumsum(z_bin_widths)
    if redo:
        cumul_counts = []
        num = 0
        for num1, lim in enumerate(limits):
            cumul_counts.append(sum([big_cone['Zs'][i] < lim for i in range(len(big_cone['Zs']))]))
            num += 1
            print(f"Sorted {num}/{len(limits)}")

        pickle_out = open("MICEexpected.pickle", "wb")
        pickle.dump(cumul_counts, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("MICEexpected.pickle", "rb")
        cumul_counts = pickle.load(pickle_in)
    expected_big = np.diff([cumul_counts[i] for i in range(len(limits))])
    expected = [expected_big[i] * (cone_radius / r_big / 60.0) ** 2 for i in range(len(expected_big))]
    plt.plot([0, 5], [0, 0], color=grey, linestyle='--')
    plt.plot(limits[1:], expected, marker='o', markersize=2.5, color=colours[0])
    plt.xlabel('$z$')
    plt.ylabel('Expected Count')
    plt.xlim([0, 1.5])
    plt.show()


if __name__ == "__main__":
    alldata = get_data()
    big_cone, big_cone_radius = make_big_cone(alldata, redo=True)
    sorted_data = get_random(alldata, redo=False, weighted=True)
    # plot_cones(alldata, sorted_data, plot_hist=True)

    # with open('random_sample.csv', 'w', newline='') as csvfile:
    #     random_sample = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     random_sample.writerow(['ra', 'dec', 'z'])
    #     for key in rand_cones:
    #         for num in range(len(rand_cones[key]['ras'])):
    #             if rand_cones[key]['zs'][num] > rand_cones[key]['zSN']:
    #                 random_sample.writerow([rand_cones[key]['ras'][num], rand_cones[key]['decs'][num],
    #                                        rand_cones[key]['zs'][num]])

    find_expected(big_cone, big_cone_radius, 111, redo=True, plot=True)

    rand_gal_z = np.zeros(len(rand_cones))
    rand_gal_kap = np.zeros(len(rand_cones))
    c = 0
    for _, supernova in rand_cones.items():
        rand_gal_z[c] = supernova['SNZ']
        rand_gal_kap[c] = supernova['SNkappa']
        c += 1

    ds = []
    chis = []
    for z in rand_gal_z:
        chi_to_z = Convergence.comoving(np.linspace(0, z, 1000), OM=0.25, OL=0.75)
        ds.append(chi_to_z[-1] * (1 + z))
        chis.append(chi_to_z[-1])
    mu = 5 * np.log10(np.array(ds) / 10 * 1E9)
    data = {"Radius12.0": rand_cones}
    exp_radii = {"Radius12.0": expected}
    exp_data = [lims, exp_radii, chi_bin_widths, chi_bins, z_bins]
    # plot_Hubble(data, OM=0.25, OL=0.75, max_z=1.5)
    # conv = find_convergence(data, exp_data, redo=False, plot_scatter=True, weighted=False, max_z=1.5)
    pickle_in = open("kappaMICE.pickle", "rb")
    conv = pickle.load(pickle_in)
    convergence_data = {"Radius12.0": conv}
    MICEconv = {"Radius12.0": {"SNkappa": rand_gal_kap}}
    cones.find_correlation(MICEconv, data, plot_correlation=True)
