from Convergence import *
from astropy.io import fits
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from scipy.optimize import curve_fit
from scipy.stats import rankdata
import csv
import pickle

colours = [[0, 150/255, 100/255], [253/255, 170/255, 0], 'C2', 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5']
blue = [23/255, 114/255, 183/255, 0.75]
orange = [255/255, 119/255, 15/255, 0.75]
green = [0, 150/255, 100/255, 0.75]
yellow = [253/255, 170/255, 0, 0.75]
grey = [0.75, 0.75, 0.75]
names = ['STRIPE82_SPECTROSCOPIC_CHAZ_NOTCLEANED_ms77.fit', 'boss_206+SDSS_213_all_cuts_new_mu_dmu1_new.fits',
         'Smithdata.csv']
radii = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
         7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
         24.0, 25.0]

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


def f(x, A):
    """Linear function for fitting correlation."""
    return A * x


def get_data(new_data=False):
    """Either gets data from FITS files and creates new output file, or just loads data from previous output.

    Inputs:
     new_data -- boolean that determines whether data is loaded or read from FITS files. Default false.
    """
    if new_data:
        S_CID = []
        S_z = []
        S_ngal = []
        S_delta = []
        S_kappa = []
        S_dkappa = []
        S_HR = []
        S_dHR = []

        with open(names[2], 'r') as f:
            CSV = csv.reader(f, delimiter=',')
            for line in CSV:
                S_CID.append(int(float(line[0].strip())))
                S_z.append(float(line[1].strip()))
                S_ngal.append(int(float(line[2].strip())))
                S_delta.append(float(line[3].strip()))
                S_kappa.append(float(line[4].strip()))
                S_dkappa.append(float(line[5].strip()))
                S_HR.append(float(line[6].strip()))
                S_dHR.append(float(line[7].strip()))

        Smith_data = {'CID': S_CID, 'z': S_z, 'ngal': S_ngal, 'delta': S_delta, 'kappa': S_kappa, 'dkappa': S_dkappa,
                      'HR': S_HR, 'dHR': S_dHR}
        with fits.open(names[0])as hdul1:
            with fits.open(names[1]) as hdul2:
                low_z = 0.04  # should be 0.05
                RA1 = [hdul1[1].data['RA'][i] for i in np.arange(len(hdul1[1].data['RA'])) if
                       hdul1[1].data['CLASS'][i] == 'GALAXY' and hdul1[1].data['Z'][i] >= 0.01]
                DEC1 = [hdul1[1].data['DEC'][i] for i in np.arange(len(hdul1[1].data['DEC'])) if
                        hdul1[1].data['CLASS'][i] == 'GALAXY' and hdul1[1].data['Z'][i] >= 0.01]
                for num, ra in enumerate(RA1):
                    if ra > 60:
                        RA1[num] -= 360
                RA2 = [hdul2[1].data['RA'][i] for i in np.arange(len(hdul2[1].data['RA'])) if
                       hdul2[1].data['CID'][i] in S_CID]
                DEC2 = [hdul2[1].data['DECL'][i] for i in np.arange(len(hdul2[1].data['DECL'])) if
                        hdul2[1].data['CID'][i] in S_CID]

                z1 = [hdul1[1].data['Z'][i] for i in np.arange(len(hdul1[1].data['RA'])) if
                      hdul1[1].data['CLASS'][i] == 'GALAXY' and hdul1[1].data['Z'][i] >= 0.01]
                z2 = [hdul2[1].data['Z_BOSS'][i] for i in np.arange(len(hdul2[1].data['RA'])) if
                      hdul2[1].data['CID'][i] in S_CID]
                mu = [hdul2[1].data['MU'][i] for i in np.arange(len(hdul2[1].data['RA'])) if
                      hdul2[1].data['CID'][i] in S_CID]
                mu_err = [hdul2[1].data['DMU1'][i] for i in np.arange(len(hdul2[1].data['RA'])) if
                          hdul2[1].data['CID'][i] in S_CID]
                CID = [hdul2[1].data['CID'][i] for i in np.arange(len(hdul2[1].data['RA'])) if
                       hdul2[1].data['CID'][i] in S_CID]

            cut_data = {'RA1': RA1, 'DEC1': DEC1, 'RA2': RA2, 'DEC2': DEC2, 'z1': z1, 'z2': z2, 'mu': mu,
                        'mu_err': mu_err, 'CID': CID}
            pickle_out = open("cut_data.pickle", "wb")
            pickle.dump(cut_data, pickle_out)
            pickle_out.close()
            pickle_out = open("Smith_data.pickle", "wb")
            pickle.dump(Smith_data, pickle_out)
            pickle_out.close()

    else:
        pickle_in = open("cut_data.pickle", "rb")
        cut_data = pickle.load(pickle_in)
        pickle_in = open("Smith_data.pickle", "rb")
        Smith_data = pickle.load(pickle_in)

    return cut_data, Smith_data


def sort_SN_gals(cut_data, redo=False):
    """Either sorts galaxies into SN cones or loads sorted data from file.

    Inputs:
     cut_data -- contains all RA, DEC and z for all SNe and galaxies as well as distance moduli and CIDs.
     redo -- boolean that determines whether data is loaded or sorted. Default false.
    """
    RA1 = cut_data['RA1']
    DEC1 = cut_data['DEC1']
    RA2 = cut_data['RA2']
    DEC2 = cut_data['DEC2']
    z1 = cut_data['z1']
    z2 = cut_data['z2']
    mu = cut_data['mu']
    mu_err = cut_data['mu_err']
    CID = cut_data['CID']
    if redo:
        lenses = {}
        for cone_radius in radii:
            lenses[f"Radius{str(cone_radius)}"] = {}
            for num, SRA, SDE, SZ, SM, SE, C in zip(np.linspace(0, len(RA2) - 1, len(RA2)), RA2, DEC2, z2, mu, mu_err, CID):
                lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}'] = {'RAs': [], 'DECs': [], 'Zs': [], 'SNZ': SZ,
                                                                          'SNMU': SM, 'SNMU_ERR': SE,  'SNRA': SRA,
                                                                          'SNDEC': SDE, 'WEIGHT': 1, 'CID': C}
                if SDE > 1.28 - cone_radius/60.0:
                    h = SDE - (1.28 - cone_radius/60.0)
                elif SDE < -(1.28 - cone_radius/60.0):
                    h = -(1.28 - cone_radius/60.0) - SDE
                else:
                    h = 0
                theta = 2 * np.arccos(1 - h / (cone_radius/60.0))
                fraction_outside = 1 / (2 * np.pi) * (theta - np.sin(theta))
                lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['WEIGHT'] = 1 - fraction_outside
                for GRA, GDE, GZ in zip(RA1, DEC1, z1):
                    if (GRA - SRA) ** 2 + (GDE - SDE) ** 2 <= (cone_radius/60.0) ** 2 and GZ <= SZ:
                        lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['RAs'].append(GRA)
                        lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['DECs'].append(GDE)
                        lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['Zs'].append(GZ)
                # print(f'Finished {int(num)+1}/{len(RA2)}')
            print(f"Finished radius {str(cone_radius)}'")
        pickle_out = open("lenses.pickle", "wb")
        pickle.dump(lenses, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("lenses.pickle", "rb")
        lenses = pickle.load(pickle_in)
        # total = 0
        # for key, val in lenses.items():
        #     print(f"CID {val['CID']}   {len(val['RAs'])}")
        #     total += len(val['RAs'])
        # print("Total:", total)

    return lenses


def plot_cones(cut_data, sorted_data, plot_hist=False, cone_radius=12.0):
    """Plots all galaxies and SNe along with visualisation of cones and galaxies contributing to lensing.

    Input:
     RA1 -- right ascensions of all galaxies.
     DEC1 -- declinations of all galaxies.
     z1 -- redshifts of all galaxies.
     lenses -- dictionary of galaxies into cones.
     patches -- circles that represent the cones in the plot.
    """
    patches = []
    for x, y in zip(cut_data['RA2'], cut_data['DEC2']):
        circle = Circle((x, y), cone_radius/60.0)
        patches.append(circle)

    lenses = sorted_data[f"Radius{str(cone_radius)}"]
    RA1 = cut_data['RA1']
    DEC1 = cut_data['DEC1']
    fig, ax = plt.subplots()
    ax.plot(RA1, DEC1, marker='o', linestyle='', markersize=1, color=[0.5, 0.5, 0.5])
    for SN, dict1, in lenses.items():
        RAs = np.array(dict1['RAs'])
        DECs = np.array(dict1['DECs'])
        indices2 = dict1['Zs'] > dict1['SNZ']
        ax.plot(RAs[indices2], DECs[indices2], marker='o', linestyle='', markersize=1, color='k',
                label="Background" if SN == 'SN1' else "")
    for SN, dict1, in lenses.items():
        RAs = np.array(dict1['RAs'])
        DECs = np.array(dict1['DECs'])
        indices1 = dict1['Zs'] <= dict1['SNZ']
        ax.plot(RAs[indices1], DECs[indices1], marker='o', linestyle='', markersize=3, color=colours[0],
                label="Foreground" if SN == 'SN1' else "")
        p = PatchCollection(patches, alpha=0.4, color=colours[0])
    ax.add_collection(p)
    SNRA = []
    SNDEC = []
    for SN, dict1, in lenses.items():
        SNRA.append(dict1['SNRA'])
        SNDEC.append(dict1['SNDEC'])

    ax.plot(SNRA, SNDEC, marker='o', linestyle='', markersize=3, label='Supernova', color=colours[1])
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
        for num, z in enumerate([cut_data['z1'], cut_data['z2']]):
            plt.hist([i for i in z if i <= 0.6], bins=np.arange(0, 0.6 + 0.025, 0.025), normed='max', linewidth=1,
                     fc=cols[num], label=f'{labels[num]}', edgecolor=colours[num])
        plt.xlabel('$z$')
        plt.ylabel('Normalised Count')
        plt.legend(frameon=0)

        plt.show()


def make_test_cones(cut_data, redo=False):
    """Creates an array of 12 arcmin cones all across data or loads test cones from file.
    Also distribution of galaxy count per bin.

    Inputs:
     RA1 -- right ascensions of galaxies.
     DEC1 -- declinations of galaxies.
     z1 -- redshifts of galaxies.
     redo -- boolean that determines whether cones are created or loaded. Default false.
    """
    RA1 = cut_data['RA1']
    DEC1 = cut_data['DEC1']
    z1 = cut_data['z1']
    if redo:
        test_cones = {}
        for cone_radius in radii:
            tests = []
            if cone_radius > 12.0:
                for a in range(int((60.0 * (50.6 + 58.1)) / (2 * cone_radius))):  # Bounds [-50.6, -1.2, 58.1, 1.2]
                    for b in range(int((60.0 * 2.4) / (2 * cone_radius))):        # degrees (convert to arcmin)
                        test = [-50.6 * 60.0 + cone_radius, 1.2 * 60.0 - cone_radius]
                        test[0] += a * 2 * cone_radius
                        test[1] -= b * 2 * cone_radius
                        test[0] = round(test[0] / 60.0, 1)
                        test[1] = round(test[1] / 60.0, 1)
                        tests.append(test)
            # elif 6.0 < cone_radius <= 12.0:
            #     for a in range(int((60.0 * (50.6 + 58.1)) / (2 * 12.0))):
            #         for b in range(int((60.0 * 2.4) / (2 * 12.0))):
            #             test = [60.0 * (-50.6 + 0.1), 60.0 * (1.2 - 0.1)]
            #             test[0] += a * 2 * 0.1 * 60.0
            #             test[1] -= b * 2 * 0.1 * 60.0
            #             test[0] = round(test[0] / 60.0, 1)
            #             test[1] = round(test[1] / 60.0, 1)
            #             tests.append(test)
            # elif cone_radius <= 6.0:
            #     for a in range(int((60.0 * (50.6 + 58.1)) / (2 * 6.0))):
            #         for b in range(int((60.0 * 2.4) / (2 * 6.0))):
            #             test = [60.0 * (-50.6 + 0.05), 60.0 * (1.2 - 0.05)]
            #             test[0] += a * 2 * 0.05 * 60.0
            #             test[1] -= b * 2 * 0.05 * 60.0
            #             test[0] = round(test[0] / 60.0, 1)
            #             test[1] = round(test[1] / 60.0, 1)
            #             tests.append(test)
            test_cones[f"Radius{str(cone_radius)}"] = {}
            for num, loc, in enumerate(tests):
                test_cones[f"Radius{str(cone_radius)}"][f'c{int(num)+1}'] = {'Total': 0, 'Zs': []}
                for GRA, GDE, GZ in zip(RA1, DEC1, z1):
                    if (GRA - loc[0]) ** 2 + (GDE - loc[1]) ** 2 <= (cone_radius/60.0) ** 2:
                        test_cones[f"Radius{str(cone_radius)}"][f'c{int(num)+1}']['Zs'].append(GZ)
                    test_cones[f"Radius{str(cone_radius)}"][f'c{int(num)+1}']['Total'] = \
                        len(test_cones[f"Radius{str(cone_radius)}"][f'c{int(num)+1}']['Zs'])
                print(f'Finished {int(num)+1}/{len(tests)}')
            print(f"Finished radius {str(cone_radius)}'")

        pickle_out = open("test_cones.pickle", "wb")
        pickle.dump(test_cones, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("test_cones.pickle", "rb")
        test_cones = pickle.load(pickle_in)

    # plt.hist([test_cones[f'c{i+1}']['Total'] for i in range(len(test_cones))], density=1,
    #          bins=20, edgecolor=colours[0], fc=green, linewidth=1)
    # plt.xlabel('Number of Galaxies')
    # plt.ylabel('Count')
    # plt.show()

    return test_cones


def find_expected_counts(test_cones, bins, redo=False, plot=False):
    """Uses the test cones to find the expected number of galaxies per bin, for bins of even comoving distance.
    Inputs:
     test_cones -- data to obtain expected counts from.
     bins -- number of bins along the line of sight to maximum SN comoving distance.
     redo -- boolean that determines whether expected counts are calculated or loaded. Default false.
    """
    max_z = 0.6
    chi_bin_widths, chi_bins, z_bins, z_bin_widths = create_z_bins(0.01, max_z, bins)
    limits = np.cumsum(z_bin_widths)
    if redo:
        expected = {}
        for cone_radius in radii:
            test_cone = test_cones[f"Radius{str(cone_radius)}"]
            cumul_tot = np.zeros((len(limits), len(test_cone)))
            # num = 0
            for num1, lim in enumerate(limits):
                for num2, _ in enumerate(test_cone.items()):
                    cumul_tot[num1][num2] = sum([test_cone[f'c{num2+1}']['Zs'][i] < lim
                                                for i in range(len(test_cone[f'c{num2+1}']['Zs']))])
                    # num += 1
                    # if num % 1000 == 0:
                    #     print(f"Finished {num}/{len(limits)*len(test_cones)}")
            expected[f"Radius{str(cone_radius)}"] = np.diff([np.mean(cumul_tot[i][:]) for i in range(len(limits))])
            print(f"Finished radius {str(cone_radius)}'")

        pickle_out = open("expected.pickle", "wb")
        pickle.dump(expected, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("expected.pickle", "rb")
        expected = pickle.load(pickle_in)

    if plot:
        for cone_radius in radii:
            plt.plot([0, 0.6], [0, 0], color=grey, linestyle='--')
            plt.plot((limits[1:]+limits[:-1])/2.0, expected[f"Radius{str(cone_radius)}"], marker='o',
                     markersize=2.5, color=colours[0])
            plt.xlabel('$z$')
            plt.ylabel('Expected Count')
            plt.xlim([0, 0.6])
            plt.show()

    return [limits, expected, chi_bin_widths, chi_bins, z_bins]


def find_convergence(lenses, exp_data, SNz, cut2, plot=False):
    """Finds the convergence along each line of sight to a SN.
    Inputs:
     lenses -- dictionary containing all galaxies that contribute to lensing.
     SNz -- redshifts of each SN.
     cut -- logical array that select SNe that have z<0.65.
     cut2 -- logical array that select SNe that are <5sigma from mean.
     limits -- bin edges.
    """
    limits = exp_data[0]
    expected_counts = exp_data[1]
    chi_widths = exp_data[2]
    chis = exp_data[3]
    zs = exp_data[4]

    chiSNs = []
    for SN in SNz:
        chi = comoving(np.linspace(0, SN, 1001))
        chiSNs.append(chi[-1])

    counts = {}
    num = 0
    for num1 in range(len(lenses)):
        bin_c = range(int(np.argmin(np.abs(limits - lenses[f"SN{num1+1}"]['SNZ']))))
        counts[f"SN{num1+1}"] = np.zeros(len(bin_c))
        for num2 in bin_c:
            counts[f"SN{num1+1}"][num2] = sum([limits[num2] < lenses[f'SN{num1+1}']['Zs'][i] <= limits[num2 + 1]
                                               for i in range(len(lenses[f'SN{num1+1}']['Zs']))]) / lenses[
                f'SN{num1+1}']['WEIGHT']
        num += 1
        if num % 50 == 0:
            print(f"Finished {num}/{len(lenses)}")

    d_arr = {}
    conv_total = np.zeros(len(counts))
    conv = {}
    conv_err = np.zeros(len(counts))
    num = 0
    for key, SN in counts.items():
        d_arr[f"{key}"] = (SN - expected_counts[:len(SN)]) / expected_counts[:(len(SN))]
        conv_total[num], conv[f"{key}"] = general_convergence(chi_widths[:len(SN)], chis[:len(SN)], zs[:len(SN)],
                                               d_arr[f"{key}"], chiSNs[num])
        conv_err[num] = convergence_error(chi_widths[:len(SN)], chis[:len(SN)], zs[:len(SN)],
                                            expected_counts[:len(SN)], chiSNs[num])
        num += 1

    SN_num = 669
    SN_key = f"SN{str(SN_num)}"
    SN = counts[SN_key]
    plt.plot(zs[:len(SN)], SN, label='Counts')
    plt.plot(zs[:len(SN)], 10000*conv[SN_key], label='10000$\kappa$')
    plt.plot(zs[:len(SN)], d_arr[SN_key], label='Overdensity')
    plt.text(0.0, 6, f'$\kappa$ = {round(conv_total[SN_num-1], 4)}')
    plt.text(0, 4, f"($\\alpha$, $\delta$), ({round(lenses[SN_key]['SNRA'],2)}, {round(lenses[SN_key]['SNDEC'],2)})")
    plt.text(0, 2, f"CID {lenses[SN_key]['CID']}")
    plt.legend(frameon=0)
    # plt.ylim([-2, 8])

    plt.plot([0, 0.3], [0, 0], linestyle='--', color=[0.5, 0.5, 0.5])
    plt.show()

    bins = np.linspace(0.025, 0.575, 12)
    edges = np.linspace(0, 0.6, 13)
    mean_kappa = []
    standard_error = []
    for bin in bins:
        kappas = []
        for z, kappa in zip(SNzs[cut2], conv_total[cut2]):
            if bin - 0.025 < z <= bin + 0.025:
                kappas.append(kappa)
        mean_kappa.append(np.mean(kappas))
        standard_error.append(np.std(kappas) / np.sqrt(len(kappas)))
    if plot:
        ax = plt.subplot2grid((1, 4), (0, 0), colspan=3)
        ax2 = plt.subplot2grid((1, 4), (0, 3))
        ax.set_ylabel("$\kappa$")
        ax.set_xlabel("$z$")
        ax2.set_xlabel("Count")
        ax.tick_params(labelsize=12)
        ax2.tick_params(labelsize=12)
        ax2.set_yticklabels([])
        plt.subplots_adjust(wspace=0, hspace=0)
        ax.plot([0, 0.6], [0, 0], color=grey, linestyle='--')
        ax.axis([0, 0.6, -0.01, 0.01])
        ax2.axis([0, 180, -0.01, 0.01])
        ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0])
        # ax.set_xticklabels([0, 0.2, 0.4, 0])
        ax.plot(SNzs[cut2], conv_total[cut2], linestyle='', marker='o', markersize=2, color=colours[0])
        ax2.hist(conv_total[cut2], bins=np.arange(-0.015, 0.02 + 0.001, 0.001), orientation='horizontal',
                 fc=green, edgecolor=colours[0])
        ax.errorbar(bins, mean_kappa, standard_error, marker='s', color='r', markersize=3, capsize=3)
        plt.show()
    return conv_total


def plot_Hubble(z, mu, mu_err, mu_diff, z_arr):
    """Plots the Hubble diagram (distance modulus agaionst redshift), including the best fitting cosmology, and
    residuals from best cosmology.
    """
    ax = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    # ax.axvspan(0, 0.2, alpha=0.1, color=colours[1])
    # ax.axvspan(0.2, 0.6, alpha=0.1, color=colours[0])
    # ax2.axvspan(0, 0.2, alpha=0.1, color=colours[1])
    # ax2.axvspan(0.2, 0.6, alpha=0.1, color=colours[0])
    # ax.text(0.05, 41, 'Peculiar\nVelocities', color=colours[1], fontsize=16)
    # ax.text(0.35, 38, 'Lensing', color=colours[0], fontsize=16)
    ax.set_ylabel("$\mu$")
    ax2.set_xlabel("$z$")
    ax2.set_ylabel("$\Delta\mu$")
    plt.subplots_adjust(wspace=0, hspace=0)
    ax.set_xticklabels([])
    ax.tick_params(labelsize=12)
    ax.errorbar(z, mu, mu_err, linestyle='', linewidth=0.8, marker='o',
                markersize=2, capsize=2, color='C3', zorder=0)
    ax.set_ylim([35, 45])
    ax.set_xlim([0, 0.6])
    ax.plot(z_arr, mu_cosm, linestyle='--', linewidth=0.8, color='C0', zorder=10)
    ax2.errorbar(z, mu_diff, mu_err, linestyle='', linewidth=1, marker='o',
                 markersize=2, capsize=2, color='C3', zorder=0)
    ax2.plot(z_arr, np.zeros(len(z_arr)), zorder=10, color='C0', linewidth=0.8, linestyle='--')
    ax2.set_ylim(-1.4, 1.4)
    ax2.set_xlim([0, 0.6])
    ax2.tick_params(labelsize=12)

    plt.show()


def find_correlation(conv, mu_diff):
    """Finds the value of the slope for plotting residuals against convergence. Magnitude of slope and error
    quantify correlation between the two.

    Inputs:
     conv -- convergence.
     mu_diff -- residuals.
    """
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
    print(f"Pearson Correlation: {round(r, 3)} +/- {round(r_err, 3)}.")
    print(f"Spearman Rank: {round(rho, 3)} +/- {round(rho_err, 3)}.")
    grad = curve_fit(f, conv, mu_diff)[0]
    fit = conv * grad

    edges = np.linspace(-0.0065, 0.011, 6)
    bins = (edges[1:] + edges[:-1]) / 2
    mean_dmu = []
    standard_error = []
    for bin in bins:
        dmus = []
        for kappa, dmu in zip(conv, mu_diff):
            if bin - 0.007/4 < kappa <= bin + 0.0007/4:
                dmus.append(dmu)
        mean_dmu.append(np.mean(dmus))
        standard_error.append(np.std(dmus) / np.sqrt(len(dmus)))

    plt.plot([min(conv), max(conv)], [0, 0], color=grey, linestyle='--')
    plt.plot(conv, mu_diff, linestyle='', marker='o', markersize=2, color=colours[0])
    plt.plot(conv, fit, color=colours[1], label=f'$\Delta\mu = {round(float(grad),3)}\kappa$')
    plt.errorbar(bins, mean_dmu, standard_error, marker='s', color='r', markersize=3, capsize=3, linestyle='')
    plt.xlabel('$\kappa$')
    plt.ylabel('$\Delta\mu$')
    plt.xlim([-0.008, 0.011])
    plt.legend(frameon=0, loc='lower right')
    plt.ylim([-0.3, 0.3])
    plt.text(0.0038, -0.19, f'$\\rho$ = {round(rho, 3)} $\pm$ {round(rho_err, 3)}', fontsize=16)
    # print([convergence_cut[cuts2][i] for i in range(len(convergence_cut[cuts2]))])
    # print([mu_diff_cut[cuts2][i] for i in range(len(convergence_cut[cuts2]))])
    # print([SNmu_err_cut[cuts2][i] for i in range(len(convergence_cut[cuts2]))])
    print("Gradient:", grad)
    plt.show()


if __name__ == "__main__":
    radius = 6.0
    data, S_data = get_data(new_data=False)
    lensing_gals = sort_SN_gals(data, redo=False)
    # for rad in radii[5::5]:
    #     plot_cones(data, lensing_gals, plot_hist=False, cone_radius=rad)
    cone_array = make_test_cones(data, redo=True)
    exp_data = find_expected_counts(cone_array, 51, redo=True, plot=True)
    SNzs = np.zeros(len(lensing_gals))
    SNmus = np.zeros(len(lensing_gals))
    SNmu_err = np.zeros(len(lensing_gals))
    c = 0
    for _, supernova in lensing_gals.items():
        SNzs[c] = supernova['SNZ']
        SNmus[c] = supernova['SNMU']
        SNmu_err[c] = supernova['SNMU_ERR']
        c += 1

    z_array = np.linspace(0.0, 0.61, 1001)
    mu_cosm = 5 * np.log10((1 + z_array) * comoving(z_array) * 1000) + 25
    mu_cosm_interp = np.interp(SNzs, z_array, mu_cosm)
    mu_diff_cut = SNmus - mu_cosm_interp
    mu_diff_std = np.std(mu_diff_cut)
    mu_diff_mean = np.mean(mu_diff_cut)
    cuts2 = [-3.9 * mu_diff_std < mu_diff_cut[i] < 3.9 * mu_diff_std and SNzs[i] > 0.2
             for i in range(len(mu_diff_cut))]  # really broken

    convergence_cut = find_convergence(lensing_gals, exp_data, SNzs, cuts2, plot=True)
    plt.plot(S_data['kappa'], S_data['kappa'], color=colours[1])
    plt.plot(S_data['kappa'], convergence_cut, color=colours[0], marker='o', markersize=3, linestyle='')
    for i in range(len(convergence_cut)):
        if convergence_cut[i] < 0.0031 and S_data['kappa'][i] > 0.0057:
        # if convergence_cut[i] > 0.016:
            print(f"Outlier is SN {i+1}/{len(convergence_cut)}")
    plt.xlabel('$\kappa$ Smith')
    plt.ylabel('My $\kappa$')
    plt.show()

    # plot_Hubble(SNzs[cuts2], SNmus[cuts2], SNmu_err[cuts2], mu_diff_cut[cuts2], z_array)

    find_correlation(convergence_cut[cuts2], mu_diff_cut[cuts2])
