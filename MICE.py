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
        cumul_counts = []
        for num1, lim in enumerate(limits):
            cumul_counts.append(sum([big_cone['Zs'] < lim][0] / 5.0))  # Made 5 cones, so take average
            print(f"Sorted {num1+1}/{len(limits)}")

        expected_big = np.diff([cumul_counts[i] for i in range(len(limits))])
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
            chi_to_z = Convergence.comoving(np.linspace(0, z, 1001), OM=0.25, OL=0.75, h=0.7)
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
            outsides_u = [rand_DECs > 10.1 - cone_radius / 60.0]
            heights[outsides_u] = rand_DECs[outsides_u] - (10.1 - cone_radius / 60.0)
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
        pickle_in = open("MICE_SN_data.pickle", "rb")
        SN_data = pickle.load(pickle_in)
        SN_zs = SN_data["SNZ"]
        kappa = {}
        if fis:
            pickle_in = open("random_cones_new_fis.pickle", "rb")
        else:
            pickle_in = open("random_cones_new.pickle", "rb")
        lens_data = pickle.load(pickle_in)

        cone_zs = {}
        for cone_radius in RADII:
            if weighted:
                SN_weights = lens_data[f"Radius{cone_radius}"]["WEIGHT"]
            # Go through all SNe
            for SN_num, key in enumerate(lens_data[f"Radius{cone_radius}"].keys()):
                if SN_num > 0:
                    cone_indices = np.array([], dtype=np.int16)
                    # Get shells from all previous RADII
                    for r in RADII[0:np.argmin(np.abs(RADII - np.array(cone_radius))) + 1]:
                        cone_indices = np.append(cone_indices, lens_data[f"Radius{r}"][key])
                    # Get redshifts of all galaxies in each SN cone
                    cone_zs[key] = all_zs[cone_indices]
            print(len(cone_zs))
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

            for num, (key, cs) in enumerate(counts.items()):
                d_arr[key] = (cs - expected_counts[:len(cs)]) / expected_counts[:(len(cs))]
                SNkappa, allkappas = Convergence.general_convergence(chi_widths[:len(cs)], chi_bis[:len(cs)],
                                                             z_bins[:len(cs)], d_arr[key], chiSNs[num], OM=0.25, h=0.7)
                kappa[f"Radius{str(cone_radius)}"]["SNkappa"].append(SNkappa)
                kappa[f"Radius{str(cone_radius)}"]["SNallkappas"][key] = allkappas

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
            if weighted:
                pickle_in = open("MICEkappa_weighted.pickle", "rb")
            else:
                pickle_in = open("MICEkappa.pickle", "rb")
        else:
            pickle_in = open("MICEkappa_fis.pickle", "rb")
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
            # ax.axis([0, 1.5, -0.01, 0.01])
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


def find_correlation(convergence_data, plot_correlation=False, plot_radii=False, fis=False):
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
        if fis:
            pickle_in = open("MICEkappa.pickle", "rb")
            kappa = pickle.load(pickle_in)
            fis_indices = []
            for key in convergence_data[f'Radius{cone_radius}']['SNallkappas'].keys():
                fis_indices.append(int(key[5:]) - 1)
            redshift_cut = [SN_data['SNZ'][fis_indices] > 0.2]
            mu_diff = SN_data["mu_diff"][fis_indices][redshift_cut]
            conv = np.array(kappa[f"Radius{str(cone_radius)}"]["SNkappa"])[fis_indices][redshift_cut]
        else:
            redshift_cut = [SN_data['SNZ'] > 0.2]
            mu_diff = SN_data["mu_diff"][redshift_cut]
            conv = np.array(convergence_data[f"Radius{str(cone_radius)}"]["SNkappa"])[redshift_cut]

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
    mu_cosm = 5 * np.log10((1 + z_array) * Convergence.comoving(z_array, OM=OM, OL=OL, h=0.7) * 1000) + 25
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
    get_random(alldata, redo=False)
    # plot_cones(alldata, sorted_data, plot_hist=True)
    # plot_Hubble()
    conv = find_convergence(alldata, exp_data, redo=False, plot_total=False, plot_scatter=False, weighted=use_weighted)
    use_weighted = not use_weighted
    conv = find_convergence(alldata, exp_data, redo=False, plot_total=False, plot_scatter=False, weighted=use_weighted)

    # pickle_in = open("MICE_SN_data.pickle", "rb")
    # SN_data = pickle.load(pickle_in)
    # pickle_in = open("random_cones_new.pickle", "rb")
    # lens_data = pickle.load(pickle_in)
    # SN_z = SN_data["SNZ"]
    # SN_kappa = SN_data["SNkappa"]
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
    #         if j < 1500:
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
    unweighted = find_correlation(cones_MICE_conv, plot_radii=False)
    weighted = find_correlation(cones_MICE_conv_weighted, plot_radii=False)

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
    kappa_fis = find_convergence(alldata, exp_data, redo=False, plot_total=True, plot_scatter=True, weighted=False, fis=True)
    pickle_out = open("MICEkappa_fis.pickle", "wb")
    pickle.dump(kappa_fis, pickle_out)
    pickle_out.close()
    pickle_in = open("MICEkappa_fis.pickle", "rb")
    kappa_fis = pickle.load(pickle_in)
    fully_in_sample = find_correlation(kappa_fis, plot_correlation=False, plot_radii=False, fis=True)

    plt.plot([0, 30], [0, 0], color=grey, linestyle='--')
    plt.plot(RADII, unweighted[1], color=colours[0])
    plt.plot(RADII, unweighted[0], marker='x', linestyle='', color=[0, 0.5, 0.9])
    plt.fill_between(RADII, unweighted[2], unweighted[3], color=colours[0], alpha=0.3)
    plt.plot(RADII, weighted[1], color=colours[1])
    plt.plot(RADII, weighted[0], marker='x', linestyle='', color=[0.7, 0.2, 0])
    plt.fill_between(RADII, weighted[2], weighted[3], color=colours[1], alpha=0.3)
    plt.plot(RADII, fully_in_sample[1], color=colours[2])
    plt.plot(RADII, fully_in_sample[0], marker='x', linestyle='', color=[0.7, 0.1, 0.6])
    plt.fill_between(RADII, fully_in_sample[2], fully_in_sample[3], color=colours[2], alpha=0.3)
    kwargs1 = {'marker': 'x', 'markeredgecolor': [0, 0.5, 0.9], 'color': colours[0]}
    kwargs2 = {'marker': 'x', 'markeredgecolor': [0.7, 0.2, 0], 'color': colours[1]}
    kwargs3 = {'marker': 'x', 'markeredgecolor': [0.7, 0.1, 0.6], 'color': colours[2]}
    plt.plot([], [], label='Unweighted', **kwargs1)
    plt.plot([], [], label='Weighted', **kwargs2)
    plt.plot([], [], label='Fully In Sample', **kwargs3)
    plt.gca().invert_yaxis()
    plt.xlim([0, 30.1])
    plt.legend(frameon=0)
    plt.xlabel('Cone Radius (arcmin)')
    plt.ylabel("Spearman's Rank Coefficient")
    plt.show()

    pickle_in = open("MICEkappa.pickle", "rb")
    kappa = pickle.load(pickle_in)
    pickle_in = open("MICEkappa_weighted.pickle", "rb")
    kappa_weighted = pickle.load(pickle_in)
    pickle_in = open("MICEkappa_fis.pickle", "rb")
    kappa_fis = pickle.load(pickle_in)
    conv_total = []
    conv_total_weighted = []
    conv_total_fis = []
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
    plt.legend(frameon=0)
    plt.show()
