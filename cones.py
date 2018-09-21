from Convergence import *
from astropy.io import fits
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from scipy.optimize import curve_fit
from scipy.stats import rankdata
import csv
import pickle
from scipy.signal import savgol_filter

colours = [[0, 150/255, 100/255], [225/255, 149/255, 0], [207/255, 0, 48/255], [30/255, 10/255, 171/255],
           'C4', 'C9', 'C6', 'C7', 'C8', 'C5']
blue = [23/255, 114/255, 183/255, 0.75]
orange = [255/255, 119/255, 15/255, 0.75]
green = [0, 150/255, 100/255, 0.75]
yellow = [253/255, 170/255, 0, 0.75]
grey = [0.75, 0.75, 0.75]
NAMES = ['STRIPE82_SPECTROSCOPIC_CHAZ_NOTCLEANED_ms77.fit', 'boss_206+SDSS_213_all_cuts_new_mu_dmu1_new.fits',
         'Smithdata.csv']
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

        with open(NAMES[2], 'r') as f:
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
        with fits.open(NAMES[0])as hdul1:
            with fits.open(NAMES[1]) as hdul2:
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


def sort_SN_gals(cut_data, redo=False, weighted=False):
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
        if weighted:
            pickle_in = open("lenses_weighted.pickle", "rb")
        else:
            pickle_in = open("lenses.pickle", "rb")
        lenses = pickle.load(pickle_in)
        for cone_radius in [13.75]:
            lenses[f"Radius{str(cone_radius)}"] = {}
            for num, (SRA, SDE, SZ, SM, SE, C) in enumerate(zip(RA2, DEC2, z2, mu, mu_err, CID)):
                lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}'] = {'RAs': [], 'DECs': [], 'Zs': [], 'SNZ': SZ,
                                                                          'SNMU': SM, 'SNMU_ERR': SE,  'SNRA': SRA,
                                                                          'SNDEC': SDE, 'WEIGHT': 1.0, 'CID': C}
                if SDE > 1.28 - cone_radius/60.0:
                    h = SDE - (1.28 - cone_radius/60.0)
                elif SDE < -(1.28 - cone_radius/60.0):
                    h = -(1.28 - cone_radius/60.0) - SDE
                else:
                    h = 0
                theta = 2 * np.arccos(1 - h / (cone_radius/60.0))
                fraction_outside = 1 / (2 * np.pi) * (theta - np.sin(theta))
                if weighted:
                    lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['WEIGHT'] = 1.0 - fraction_outside
                else:
                    lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['WEIGHT'] = 1.0
                for GRA, GDE, GZ in zip(RA1, DEC1, z1):
                    if (GRA - SRA) ** 2 + (GDE - SDE) ** 2 <= (cone_radius/60.0) ** 2 and GZ <= SZ:
                        lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['RAs'].append(GRA)
                        lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['DECs'].append(GDE)
                        lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['Zs'].append(GZ)
                # print(f'Finished {int(num)+1}/{len(RA2)}')
            print(f"Finished radius {str(cone_radius)}'")
        if weighted:
            pickle_out = open("lenses_weighted.pickle", "wb")
        else:
            pickle_out = open("lenses.pickle", "wb")
        pickle.dump(lenses, pickle_out)
        pickle_out.close()
    else:
        if weighted:
            pickle_in = open("lenses_weighted.pickle", "rb")
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
     cut_data -- dictionary that contains all data (RA, DEC, z, etc.) of galaxies.
     sorted_data -- dictionary that contains all information for every SN sorted into cones.
     plot_hist -- boolean that determines if a histogram of the galaxy and SNe distribution is plotted. Defaults to
                  False.
     cone_radius -- the radius of the cones. Defaults to 12'.
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
    contRAs = []
    contDECs = []
    for SN, dict1, in lenses.items():
        RAs = np.array(dict1['RAs'])
        DECs = np.array(dict1['DECs'])
        indices1 = dict1['Zs'] <= dict1['SNZ']
        contRAs = np.append(contRAs, RAs[indices1])
        contDECs = np.append(contDECs, DECs[indices1])
    ax.plot(contRAs, contDECs, marker='o', linestyle='', markersize=3, color=colours[0])
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
    # plt.text(27, -0.8, f"{cone_radius}' radius")
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


def make_test_cones(cut_data, redo=False, plot=False):
    """Creates an array of cones all across data or loads test cones from file for a variety of cone widths.

    Inputs:
     cut_data -- dictionary that contains all data (RA, DEC, z, etc.) of galaxies.
     redo -- boolean that determines whether cones are created or loaded. Default false.
     plot -- boolean that determines whether a plot of the data field with test_cones overplotted. Default false.
    """
    RA1 = cut_data['RA1']
    DEC1 = cut_data['DEC1']
    z1 = cut_data['z1']
    if redo:
        pickle_in = open("test_cones.pickle", "rb")
        test_cones = pickle.load(pickle_in)
        x0 = -50.6 * 60.0  # Convert bounds in degrees to radians
        x1 = 58.1 * 60.0
        y0 = 1.25 * 60.0
        y1 = -1.25 * 60.0
        for cone_radius in [6.0]:
            tests = []
            if cone_radius > 12.0:
                for a in range(int((x1 - x0) / (2 * cone_radius))):
                    for b in range(int((y0 - y1) / (2 * cone_radius))):
                        test = [x0 + cone_radius, y0 - cone_radius]
                        test[0] += a * 2 * cone_radius
                        test[1] -= b * 2 * cone_radius
                        test[0] /= 60.0  # Back to degrees
                        test[1] /= 60.0
                        tests.append(test)
            if 6.0 < cone_radius <= 12.0:
                for a in range(int((x1 - x0) / 24.0)):
                    for b in range(int((y0 - y1) / 24.0)):
                        test = [x0 + 12.0, y0 - 12.0]
                        test[0] += a * 2 * 12.0
                        test[1] -= b * 2 * 12.0
                        test[0] /= 60.0
                        test[1] /= 60.0
                        tests.append(test)
            elif cone_radius <= 6.0:
                for a in range(int((x1 - x0) / 12.0)):
                    for b in range(int((y0 - y1) / 12.0)):
                        test = [x0 + 6.0, y0 - 6.0]
                        test[0] += a * 2 * 6.0
                        test[1] -= b * 2 * 6.0
                        test[0] /= 60.0
                        test[1] /= 60.0
                        tests.append(test)
            if plot:
                fig, ax = plt.subplots()
                patches = []
                for r in tests:
                    circle = Circle((r[0], r[1]), cone_radius / 60.0)
                    patches.append(circle)
                p = PatchCollection(patches, alpha=0.4, color=colours[0])
                ax.plot(RA1, DEC1, linestyle='', marker='o', markersize=1)
                ax.add_collection(p)
                plt.show()

            test_cones[f"Radius{str(cone_radius)}"] = {}
            for num, loc, in enumerate(tests):
                test_cones[f"Radius{str(cone_radius)}"][f'c{int(num)+1}'] = {'Total': 0, 'Zs': []}
                test_cones[f"Radius{str(cone_radius)}"][f'c{int(num)+1}']['Zs'].append(
                    np.array(z1)[(np.array(RA1) - np.array(loc[0])) ** 2 + (np.array(DEC1) - np.array(loc[1])) ** 2 <= (cone_radius/60.0) ** 2])
                test_cones[f"Radius{str(cone_radius)}"][f'c{int(num)+1}']['Total'] = len(
                    test_cones[f"Radius{str(cone_radius)}"][f'c{int(num)+1}']['Zs'])
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
    """Uses the test cones to find the expected number of galaxies per bin, for bins of even redshift.

    Inputs:
     test_cones -- dictionary of data to obtain expected counts from for a variety of cone widths.
     bins -- number of bins along the line of sight to maximum SN comoving distance.
     redo -- boolean that determines whether expected counts are calculated or loaded. Default false.
     plot -- boolean that determines whether expected counts are plotted. Default false.
    """
    max_z = 0.6
    chi_bin_widths, chi_bins, z_bins, z_bin_widths = create_z_bins(0.01, max_z, bins)
    limits = np.cumsum(z_bin_widths)
    if redo:
        pickle_in = open("expected.pickle", "rb")
        expected = pickle.load(pickle_in)
        for cone_radius in RADII:
            test_cone = test_cones[f"Radius{str(cone_radius)}"]
            cumul_tot = np.zeros((len(limits), len(test_cone)))
            for num1, lim in enumerate(limits):
                for num2, _ in enumerate(test_cone.items()):
                    cumul_tot[num1][num2] = np.count_nonzero([test_cone[f'c{num2+1}']['Zs'] < lim])
            expected[f"Radius{str(cone_radius)}"] = np.diff(np.mean(cumul_tot, 1))
            for index, count in enumerate(expected[f"Radius{str(cone_radius)}"]):
                if count == 0:
                    try:
                        expected[f"Radius{str(cone_radius)}"][index] = 0.5 * (expected[f"Radius{str(cone_radius)}"]
                                                                              [index+1] + expected[
                                                                               f"Radius{str(cone_radius)}"][index-1])
                    except IndexError:
                        expected[f"Radius{str(cone_radius)}"][index] = 0.5 * (expected[f"Radius{str(cone_radius)}"]
                                                                              [index] + expected[
                                                                               f"Radius{str(cone_radius)}"][index - 1])

            print(f"Finished radius {str(cone_radius)}'")

        pickle_out = open("expected.pickle", "wb")
        pickle.dump(expected, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("expected.pickle", "rb")
        expected = pickle.load(pickle_in)

    if plot:
        for cone_radius in RADII:
            plt.plot([0, 0.6], [0, 0], color=grey, linestyle='--')
            plt.plot((limits[1:]+limits[:-1])/2.0, expected[f"Radius{str(cone_radius)}"], marker='o',
                     markersize=2.5, color=colours[0])
            plt.xlabel('$z$')
            plt.ylabel('Expected Count')
            plt.xlim([0, 0.6])
            plt.show()

    return [limits, expected, chi_bin_widths, chi_bins, z_bins]


def find_convergence(lens_data, exp_data, redo=False, plot_scatter=False, plot_total=False, weighted=False, max_z=0.6,
                     fis=False, impact=False):
    """Finds the convergence along each line of sight to a SN for a variety of cone_widths.

    Inputs:
     lens_data -- dictionary containing all galaxies that contribute to lensing.
     exp_data -- dictionary containing all expected counts per bin per cone width.
     SNz -- redshifts of each SN.
     redo -- boolean that determines whether convergence is calculated or loaded. Dafault false.
     plot_scatter -- boolean that determined whether scatter plot of convergence per SN redshift is plotted.
                     Default false.
     plot_total -- boolean that determines whether total convergence per cone radius is plotted. Default false.
    """
    limits = exp_data[0]
    chi_widths = exp_data[2]
    chis = exp_data[3]
    zs = exp_data[4]
    if redo:
        if weighted:
            pickle_in = open("kappa_weighted.pickle", "rb")
        elif fis:
            pickle_in = open("kappa_fis.pickle", "rb")
        elif impact:
            pickle_in = open("kappa_impact.pickle", "rb")
        else:
            pickle_in = open("kappa.pickle", "rb")
        kappa = pickle.load(pickle_in)
        # kappa = {}
        for cone_radius in RADII:
            expected_counts = exp_data[1][f"Radius{str(cone_radius)}"]
            lenses = lens_data[f"Radius{str(cone_radius)}"]

            kappa[f"Radius{str(cone_radius)}"] = {"Counts": {}, "delta": {}, "SNkappa": [], "SNallkappas": {},
                                                  "SNerr": [], "Total": 0}
            d_arr = {}
            counts = {}
            for key in lenses.keys():
                bin_c = range(int(np.argmin(np.abs(limits - lenses[key]['SNZ']))))
                counts[key] = np.zeros(len(bin_c))
                for num2 in bin_c:
                    tmp = [np.logical_and(limits[num2] < lenses[key]['Zs'], (lenses[key]['Zs'] <= limits[num2 + 1]))]
                    IPs = np.array(lenses[key]["IPWEIGHT"])[tmp]

                    if weighted:
                        counts[key][num2] = np.count_nonzero(tmp) / lenses[key]['WEIGHT']
                    elif impact:
                        if len(IPs) == 0:
                            counts[key][num2] = 0.0
                        else:
                            counts[key][num2] = sum(IPs)
                            # print(cone_radius, key, num2+1, counts[key][num2], len(IPs))
                    else:
                        counts[key][num2] = np.count_nonzero(tmp)

            SNe_data_radius = find_mu_diff(lens_data, cone_radius=cone_radius)
            chiSNs = []
            for SN in SNe_data_radius['z']:
                chi = comoving(np.linspace(0, SN, 1001))
                chiSNs.append(chi[-1])
            # c_arr = []
            for num, (key, SN) in enumerate(counts.items()):
                d_arr[key] = (SN - expected_counts[:len(SN)]) / expected_counts[:(len(SN))]
                # kappa[f"Radius{str(cone_radius)}"]["SNkappa"][num], kappa[f"Radius{str(cone_radius)}"][key] =
                SNkappa, allkappas = general_convergence(chi_widths[:len(SN)], chis[:len(SN)], zs[:len(SN)], d_arr[key],
                                                 chiSNs[num])
                kappa[f"Radius{str(cone_radius)}"]["SNkappa"].append(SNkappa)
                kappa[f"Radius{str(cone_radius)}"]["SNallkappas"][key] = allkappas

                SNkappa_err = convergence_error(chi_widths[:len(SN)], chis[:len(SN)], zs[:len(SN)],
                                                expected_counts[:len(SN)], chiSNs[num])
                kappa[f"Radius{str(cone_radius)}"]["SNerr"].append(SNkappa_err)
                # c_arr.append(SN)
            kappa[f"Radius{str(cone_radius)}"]["Total"] = np.sum(kappa[f"Radius{str(cone_radius)}"]["SNkappa"])
            kappa[f"Radius{str(cone_radius)}"]["Counts"] = counts
            kappa[f"Radius{str(cone_radius)}"]["delta"] = d_arr

            # s = plt.scatter(SNe_data_radius['z'], kappa[f"Radius{str(cone_radius)}"]["SNkappa"],
            #                 c=[sum(c_arr[i]) for i in range(len(c_arr))], cmap='coolwarm')
            # cbar = plt.colorbar(s)
            # cbar.set_label('$z$')
            # plt.xlabel('Total Count')
            # plt.ylabel('$\kappa$')
            # plt.show()
            print(f"Finished radius {str(cone_radius)}'")
        if weighted:
            pickle_out = open("kappa_weighted.pickle", "wb")
        elif fis:
            pickle_out = open("kappa_fis.pickle", "wb")
        elif impact:
            pickle_out = open("kappa_impact.pickle", "wb")
        else:
            pickle_out = open("kappa.pickle", "wb")
        pickle.dump(kappa, pickle_out)
        pickle_out.close()

    else:
        if weighted:
            pickle_in = open("kappa_weighted.pickle", "rb")
        elif fis:
            pickle_in = open("kappa_fis.pickle", "rb")
        elif impact:
            pickle_in = open("kappa_impact.pickle", "rb")
        else:
            pickle_in = open("kappa.pickle", "rb")
        kappa = pickle.load(pickle_in)

    for cone_radius in [12.0]:
        SNe_data_radius = find_mu_diff(lens_data, cone_radius=cone_radius)
        lenses = lens_data[f"Radius{str(cone_radius)}"]
        bins = np.linspace(0.025, max_z - 0.025, 12)
        edges = np.linspace(0, 0.6, 13)
        mean_kappa = []
        standard_error = []
        conv = kappa[f"Radius{str(cone_radius)}"]["SNkappa"]
        # counts = kappa[f"Radius{str(cone_radius)}"]["Counts"]
        # d_arr = kappa[f"Radius{str(cone_radius)}"]["delta"]

        # SN_num = 669
        # SN_key = f"SN{str(SN_num)}"
        # SN = counts[SN_key]
        # allkappas = kappa[f"Radius{str(cone_radius)}"]["SNallkappas"][SN_key]
        # plt.plot(zs[:len(SN)], SN, label='Counts')
        # plt.plot(zs[:len(SN)], 10000 * allkappas, label='10000$\kappa$')
        # plt.plot(zs[:len(SN)], d_arr[SN_key], label='Overdensity')
        # plt.text(0.0, 6, f'$\kappa$ = {round(conv_total[SN_num-1], 4)}')
        # plt.text(0, 4, f"($\\alpha$, $\delta$), ({round(lenses[SN_key]['SNRA'], 2)}, "
        #                f"{round(lenses[SN_key]['SNDEC'], 2)})")
        # plt.text(0, 2, f"CID {lenses[SN_key]['CID']}")
        # plt.legend(frameon=0)
        # plt.plot([0, 0.3], [0, 0], linestyle='--', color=[0.5, 0.5, 0.5])
        # plt.show()

        for b in bins:
            ks = []
            for z, k in zip(SNe_data_radius['z'], conv):
                if b - 0.025 < z <= b + 0.025:
                    ks.append(k)
            mean_kappa.append(np.mean(ks))
            standard_error.append(np.std(ks) / np.sqrt(len(ks)))

        if plot_scatter:
            conv = kappa[f"Radius{str(cone_radius)}"]["SNkappa"]
            ax = plt.subplot2grid((1, 4), (0, 0), colspan=3)
            # ax = plt.subplot2grid((1, 1), (0, 0))
            ax2 = plt.subplot2grid((1, 4), (0, 3))
            ax.set_ylabel("$\kappa$")
            ax.set_xlabel("$z$")
            ax2.set_xlabel("Count")
            ax2.set_yticklabels([])
            plt.subplots_adjust(wspace=0, hspace=0)
            ax.plot([0, max_z], [0, 0], color=grey, linestyle='--')
            ax.axis([0, max_z, -0.01, 0.01])
            ax2.axis([0, 180, -0.01, 0.01])
            # ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0])
            ax.set_xticklabels([0, 0.2, 0.4, 0])
            ax.plot(SNe_data_radius['z'], conv, linestyle='', marker='o', markersize=2, color=colours[0])
            ax2.hist(conv, bins=np.arange(-0.015, 0.02 + 0.001, 0.001), orientation='horizontal',
                     fc=green, edgecolor=colours[0])
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
        plt.axis([0, 30, -0.5, 1.5])
        plt.plot(RADII, conv_total, marker='o', markersize=2, color=colours[0])
        plt.show()

    return kappa


def plot_Hubble(lenses, OM=0.27, OL=0.73, max_z=0.6):
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
    data = find_mu_diff(lenses, OM=OM, OL=OL, max_z=max_z)
    z = data['z']
    mu = data['mu']
    mu_err = data['mu_err']
    mu_cosm = data['mu_cosm']
    mu_diff = data['mu_diff']
    z_arr = data['z_arr']
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
    ax.set_xlim([0, max_z])
    ax.plot(z_arr, mu_cosm, linestyle='--', linewidth=0.8, color='C0', zorder=10)
    ax2.errorbar(z, mu_diff, mu_err, linestyle='', linewidth=1, marker='o',
                 markersize=2, capsize=2, color='C3', zorder=0)
    ax2.plot(z_arr, np.zeros(len(z_arr)), zorder=10, color='C0', linewidth=0.8, linestyle='--')
    ax2.set_ylim(-1.4, 1.4)
    ax2.set_xlim([0, max_z])
    ax2.tick_params(labelsize=12)

    plt.show()


def find_correlation(convergence_data, lens_data, plot_correlation=False, plot_radii=False):
    """Finds the value of the slope for plotting residuals against convergence. Magnitude of slope and error
    quantify correlation between the two.

    Inputs:
     conv -- convergence.
     mu_diff -- residuals.
    """
    correlations = []
    correlation_errs = []
    for cone_radius in RADII:
        SNe_data = find_mu_diff(lens_data, cone_radius=cone_radius)
        redshift_cut = np.logical_and(SNe_data['z'] > 0.3, SNe_data['z'] < 0.5)
        mu_diff = SNe_data["mu_diff"][redshift_cut]
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
            plt.errorbar(bins, mean_dmu, standard_error, marker='s', color='r', markersize=3, capsize=3, linestyle='')
            plt.xlabel('$\kappa$')
            plt.ylabel('$\Delta\mu$')
            # plt.xlim([-0.008, 0.011])
            # plt.legend(frameon=0, loc='lower right')
            # plt.ylim([-0.3, 0.3])
            plt.text(0.0038, -0.19, f'$\\rho$ = {round(rho, 3)} $\pm$ {round(rho_err, 3)}', fontsize=16)
            # print([convergence_cut[cuts2][i] for i in range(len(convergence_cut[cuts2]))])
            # print([mu_diff_cut[cuts2][i] for i in range(len(convergence_cut[cuts2]))])
            # print([SNmu_err_cut[cuts2][i] for i in range(len(convergence_cut[cuts2]))])
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
        plt.gca().invert_yaxis()
        plt.show()

    return [correlations, smooth_corr, smooth_u_err, smooth_d_err, np.array(u_err) - np.array(correlations)]


def find_mu_diff(lenses, OM=0.27, OL=0.73, h=0.738, max_z=0.6, cone_radius=12.0):
    """Finds the distance modulus of best fitting cosmology and hence residuals.

    Inputs:
     lenses -- data that contains distance modulus and redshift of each SN.
    """
    lens_gal_single_rad = lenses[f"Radius{cone_radius}"]
    SNzs = np.zeros(len(lens_gal_single_rad))
    SNmus = np.zeros(len(lens_gal_single_rad))
    SNmu_err = np.zeros(len(lens_gal_single_rad))
    c = 0
    for SN_key, SN in lens_gal_single_rad.items():
        SNzs[c] = SN['SNZ']
        SNmus[c] = SN['SNMU']
        SNmu_err[c] = SN['SNMU_ERR']
        c += 1
    z_array = np.linspace(0.0, max_z+0.01, 1001)
    mu_cosm = 5 * np.log10((1 + z_array) * comoving(z_array, OM=OM, OL=OL, h=h) * 1000) + 25
    mu_cosm_interp = np.interp(SNzs, z_array, mu_cosm)
    mu_diff = SNmus - mu_cosm_interp

    data = {"z": SNzs, "mu": SNmus, "mu_err": SNmu_err, "mu_diff": mu_diff, "mu_cosm": mu_cosm, "z_arr": z_array}
    return data


if __name__ == "__main__":
    use_weighted = False
    data, S_data = get_data(new_data=False)
    lensing_gals = sort_SN_gals(data, redo=False, weighted=use_weighted)
    # plot_cones(data, lensing_gals, plot_hist=False, cone_radius=12.0)
    cone_array = make_test_cones(data, redo=False, plot=False)
    exp_data = find_expected_counts(cone_array, 51, redo=False, plot=False)
    redo_conv = False
    kappa = find_convergence(lensing_gals, exp_data, redo=redo_conv, plot_scatter=False,
                             plot_total=False, weighted=use_weighted)

    # plt.plot(S_data['kappa'], S_data['kappa'], color=colours[1])
    # plt.plot(S_data['kappa'], convergence, color=colours[0], marker='o', markersize=3, linestyle='')
    # for SN in range(len(convergence)):
    #     if convergence[SN] < 0.0031 and S_data['kappa'][SN] > 0.0057:
    #     # if convergence_cut[SN] > 0.016:
    #         print(f"Outlier is SN {SN+1}/{len(convergence)}")
    # plt.xlabel('$\kappa$ Smith')
    # plt.ylabel('My $\kappa$')
    # plt.show()

    # plot_Hubble(lensing_gals)

    unweighted = find_correlation(kappa, lensing_gals, plot_correlation=False, plot_radii=True)
    use_weighted = True
    lensing_gals = sort_SN_gals(data, redo=False, weighted=use_weighted)
    kappa_weighted = find_convergence(lensing_gals, exp_data, redo=redo_conv, plot_scatter=False, plot_total=False,
                                      weighted=use_weighted)
    weighted = find_correlation(kappa_weighted, lensing_gals, plot_correlation=False, plot_radii=True)

    lensing_gals_fully_in_sample = {}
    number_fis = np.zeros(len(RADII))
    num = 0
    for rad in RADII:
        lensing_gals_fully_in_sample[f"Radius{rad}"] = {}
        for key2, SN in lensing_gals[f"Radius{rad}"].items():
            if SN["WEIGHT"] == 1:
                lensing_gals_fully_in_sample[f"Radius{rad}"][key2] = SN
                number_fis[num] += 1
        num += 1
    # plt.plot(RADII, number_fis, '+')
    # plt.show()
    lensing_gals = sort_SN_gals(data, redo=False, weighted=False)
    kappa_fis = find_convergence(lensing_gals_fully_in_sample, exp_data, redo=redo_conv, plot_total=False, fis=True)
    fully_in_sample = find_correlation(kappa_fis, lensing_gals_fully_in_sample, plot_correlation=False,
                                       plot_radii=True)

    kappa_impact = find_convergence(lensing_gals_fully_in_sample, exp_data, redo=False, plot_total=False, impact=True)
    impact = find_correlation(kappa_impact, lensing_gals_fully_in_sample, plot_radii=True)
    # print(unweighted[0][53], unweighted[4][53], weighted[0][53], weighted[4][53], fully_in_sample[0][53],
    #       fully_in_sample[4][53], fully_in_sample[0][58], fully_in_sample[4][58])
    # plt.plot([0, 30], [0, 0], color=grey, linestyle='--')
    # plt.plot(RADII, unweighted[1], color=colours[0])
    # plt.plot(RADII, unweighted[0], marker='x', linestyle='', color=[0, 0.5, 0.9])
    # plt.fill_between(RADII, unweighted[2], unweighted[3], color=colours[0], alpha=0.3)
    # plt.plot(RADII, weighted[1], color=colours[1])
    # plt.plot(RADII, weighted[0], marker='x', linestyle='', color=[0.7, 0.3, 0])
    # plt.fill_between(RADII, weighted[2], weighted[3], color=colours[1], alpha=0.3)
    # plt.plot(RADII, fully_in_sample[1], color=colours[2])
    # plt.plot(RADII, fully_in_sample[0], marker='x', linestyle='', color=[0.7, 0.1, 0.6])
    # plt.fill_between(RADII, fully_in_sample[2], fully_in_sample[3], color=colours[2], alpha=0.3)
    kwargs1 = {'marker': 'x', 'markeredgecolor': [0, 0.5, 0.9], 'color': colours[0]}
    kwargs2 = {'marker': 'x', 'markeredgecolor': [0.7, 0.3, 0], 'color': colours[1]}
    kwargs3 = {'marker': 'x', 'markeredgecolor': [0.7, 0.1, 0.6], 'color': colours[2]}
    # plt.plot([], [], label='Unweighted', **kwargs1)
    # plt.plot([], [], label='Weighted', **kwargs2)
    # plt.plot([], [], label='Fully In Sample', **kwargs3)
    # plt.gca().invert_yaxis()
    # plt.xlim([0, 30.0])
    # plt.legend(frameon=0)
    # plt.xlabel('Cone Radius (arcmin)')
    # plt.ylabel("Spearman's Rank Coefficient")
    # plt.show()

    plt.plot(RADII, fully_in_sample[1], color=colours[2])
    plt.plot(RADII, fully_in_sample[0], marker='x', linestyle='', color=[0.7, 0.1, 0.6])
    plt.fill_between(RADII, fully_in_sample[2], fully_in_sample[3], color=colours[2], alpha=0.3)
    plt.plot(RADII, impact[1], color=colours[3])
    plt.plot(RADII, impact[0], marker='x', linestyle='', color=[60/255, 90/255, 240/255])
    plt.fill_between(RADII, impact[2], impact[3], color=colours[3], alpha=0.3)
    kwargs4 = {'marker': 'x', 'markeredgecolor': [60/255, 90/255, 240/255], 'color': colours[3]}
    plt.plot([], [], label='Fully In Sample', **kwargs3)
    plt.plot([], [], label='Impact Parameter', **kwargs4)
    plt.gca().invert_yaxis()
    plt.xlim([0, 30.0])
    plt.legend(frameon=0)
    plt.xlabel('Cone Radius (arcmin)')
    plt.ylabel("Spearman's Rank Coefficient")
    plt.show()

    conv_total = []
    conv_total_weighted = []
    conv_total_fis = []
    conv_total_impact = []
    for cone_radius in RADII:
        conv_total.append(kappa[f"Radius{str(cone_radius)}"]["Total"])
        conv_total_weighted.append(kappa_weighted[f"Radius{str(cone_radius)}"]["Total"])
        conv_total_fis.append(kappa_fis[f"Radius{str(cone_radius)}"]["Total"])
        conv_total_impact.append(kappa_impact[f"Radius{str(cone_radius)}"]["Total"]/100)
    # plt.ylabel("Total Convergence")
    # plt.xlabel("Cone Radius (arcmin)")
    # plt.tick_params(labelsize=12)
    # plt.plot([0, 30], [0, 0], color=grey, linestyle='--')
    # plt.axis([0, 30, -1, 1.5])
    # plt.plot(RADII, conv_total, marker='o', markersize=2, color=colours[0], label='Unweighted')
    # plt.plot(RADII, conv_total_weighted, marker='o', markersize=2, color=colours[1], label='Weighted')
    # plt.plot(RADII, conv_total_fis, marker='o', markersize=2, color=colours[2], label='Fully in sample')
    # plt.legend(frameon=0)
    # plt.show()

    plt.ylabel("Total Convergence")
    plt.xlabel("Cone Radius (arcmin)")
    plt.tick_params(labelsize=12)
    plt.plot([0, 30], [0, 0], color=grey, linestyle='--')
    plt.axis([0, 30, -1, 1.5])
    plt.plot(RADII, conv_total_fis, marker='o', markersize=2, color=colours[2], label='Fully in sample')
    plt.plot(RADII, conv_total_impact, marker='o', markersize=2, color=colours[3], label='Impact')
    plt.legend(frameon=0)
    plt.show()
