from Convergence import *
from astropy.io import fits
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from scipy.optimize import curve_fit
from scipy.stats import rankdata
import csv
import pickle
from scipy.signal import savgol_filter

colours = [[0, 150/255, 100/255], [253/255, 170/255, 0], 'C2', 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5']
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
         13.25, 13.5, 12.75, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0,
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
        pickle_in = open("lenses.pickle", "rb")
        # pickle_in = open("lenses.pickle", "rb")
        lenses = pickle.load(pickle_in)
        for cone_radius in [1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25,
                            8.75, 9.25, 9.75, 10.25, 10.5, 10.75, 11.25, 11.5, 11.75, 12.25, 12.5, 12.75, 13.25, 13.5,
                            13.75, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 26.0, 27.0, 28.0, 29.0, 30.0]:
            lenses[f"Radius{str(cone_radius)}"] = {}
            for num, SRA, SDE, SZ, SM, SE, C in zip(np.linspace(0, len(RA2) - 1, len(RA2)), RA2, DEC2, z2, mu, mu_err,
                                                    CID):
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
                lenses[f"Radius{str(cone_radius)}"][f'SN{int(num)+1}']['WEIGHT'] = 1 #- fraction_outside
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
        pickle_in = open("lenses_weighted.pickle", "rb")
        # pickle_in = open("lenses.pickle", "rb")
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
        x0 = -50.6 * 60.0  # Convert ounds in degrees to radians
        x1 = 58.1 * 60.0
        y0 = 1.25 * 60.0
        y1 = -1.25 * 60.0
        for cone_radius in [1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25,
                            8.75, 9.25, 9.75, 10.25, 10.5, 10.75, 11.25, 11.5, 11.75, 12.25, 12.5, 12.75, 13.25, 13.5,
                            13.75, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 26.0, 27.0, 28.0, 29.0, 30.0]:
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
        for cone_radius in [1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25,
                            8.75, 9.25, 9.75, 10.25, 10.5, 10.75, 11.25, 11.5, 11.75, 12.25, 12.5, 12.75, 13.25, 13.5,
                            13.75, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 26.0, 27.0, 28.0, 29.0, 30.0]:
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


def find_convergence(lens_data, exp_data, SNz, redo=False, plot_scatter=True, plot_total=False):
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
        pickle_in = open("kappa_weighted.pickle", "rb")
        # pickle_in = open("kappa.pickle", "rb")
        kappa = pickle.load(pickle_in)

        chiSNs = []
        for SN in SNz:
            chi = comoving(np.linspace(0, SN, 1001))
            chiSNs.append(chi[-1])

        for cone_radius in [1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25,
                            8.75, 9.25, 9.75, 10.25, 10.5, 10.75, 11.25, 11.5, 11.75, 12.25, 12.5, 12.75, 13.25, 13.5,
                            13.75, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 26.0, 27.0, 28.0, 29.0, 30.0]:
            expected_counts = exp_data[1][f"Radius{str(cone_radius)}"]
            lenses = lens_data[f"Radius{str(cone_radius)}"]

            kappa[f"Radius{str(cone_radius)}"] = {"Counts": {}, "delta": {}, "SNkappa": [], "SNallkappas": {},
                                                  "SNerr": [], "Total": 0}
            d_arr = {}
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

            num = 0
            for key, SN in counts.items():
                d_arr[key] = (SN - expected_counts[:len(SN)]) / expected_counts[:(len(SN))]
                # kappa[f"Radius{str(cone_radius)}"]["SNkappa"][num], kappa[f"Radius{str(cone_radius)}"][key] =
                SNkappa, allkappas = general_convergence(chi_widths[:len(SN)], chis[:len(SN)], zs[:len(SN)], d_arr[key],
                                                 chiSNs[num])
                kappa[f"Radius{str(cone_radius)}"]["SNkappa"].append(SNkappa)
                kappa[f"Radius{str(cone_radius)}"]["SNallkappas"][key] = allkappas

                SNkappa_err = convergence_error(chi_widths[:len(SN)], chis[:len(SN)], zs[:len(SN)],
                                                expected_counts[:len(SN)], chiSNs[num])
                kappa[f"Radius{str(cone_radius)}"]["SNerr"].append(SNkappa_err)
                num += 1
            kappa[f"Radius{str(cone_radius)}"]["Total"] = np.sum(kappa[f"Radius{str(cone_radius)}"]["SNkappa"])
            kappa[f"Radius{str(cone_radius)}"]["Counts"] = counts
            kappa[f"Radius{str(cone_radius)}"]["delta"] = d_arr
            print(f"Finished radius {str(cone_radius)}'")

        pickle_out = open("kappa.pickle", "wb")
        pickle.dump(kappa, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("kappa_weighted.pickle", "rb")
        # pickle_in = open("kappa.pickle", "rb")
        kappa = pickle.load(pickle_in)

    for cone_radius in RADII:
        lenses = lens_data[f"Radius{str(cone_radius)}"]
        bins = np.linspace(0.025, 0.575, 12)
        edges = np.linspace(0, 0.6, 13)
        mean_kappa = []
        standard_error = []
        conv = kappa[f"Radius{str(cone_radius)}"]["SNkappa"]
        counts = kappa[f"Radius{str(cone_radius)}"]["Counts"]
        d_arr = kappa[f"Radius{str(cone_radius)}"]["delta"]

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
            for z, k in zip(SNz, conv):
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
            ax.plot([0, 0.6], [0, 0], color=grey, linestyle='--')
            ax.axis([0, 0.6, -0.01, 0.01])
            # ax2.axis([0, 180, -0.01, 0.01])
            # ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0])
            # ax.set_xticklabels([0, 0.2, 0.4, 0])
            ax.plot(SNz, conv, linestyle='', marker='o', markersize=2, color=colours[0])
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
        plt.plot([0, 26], [0, 0], color=grey, linestyle='--')
        # plt.axis([0, 0.6, -0.01, 0.01])
        plt.plot(RADII, conv_total, marker='o', markersize=2, color=colours[0])
        plt.show()

    return kappa


def plot_Hubble(z, mu, mu_err, mu_cosm, mu_diff, z_arr):
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


def find_correlation(convergence_data, mu_diff, plot_correlation=False, plot_radii=False):
    """Finds the value of the slope for plotting residuals against convergence. Magnitude of slope and error
    quantify correlation between the two.

    Inputs:
     conv -- convergence.
     mu_diff -- residuals.
    """
    correlations = []
    correlation_errs = []
    for cone_radius in RADII:
        conv = convergence_data[f"Radius{str(cone_radius)}"]["SNkappa"]
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
        # print(f"Pearson Correlation: {round(r, 3)} +/- {round(r_err, 3)}.")
        # print(f"Spearman Rank: {round(rho, 3)} +/- {round(rho_err, 3)}.")
        correlations.append(rho)
        correlation_errs.append(rho_err)
        grad = curve_fit(f, conv, mu_diff)[0]
        fit = conv * grad

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

    if plot_radii:
        # for r, c, e in zip(RADII, correlations, correlation_errs):
        #     print(r, c, e)
        plt.plot([0, 25], [0, 0], color=grey, linestyle='--')
        smooth_corr = savgol_filter([abs(correlations[i]) for i in range(len(correlations))], 11, 4)
        plt.plot(RADII, smooth_corr, marker='o', markersize=3,
                 color=colours[0])
        plt.fill_between(RADII, [abs(correlations[i]) - correlation_errs[i] for i in range(len(correlations))],
                         [abs(correlations[i]) + correlation_errs[i] for i in range(len(correlations))], color=green)
        plt.xlabel('Cone Radius (arcmin)')
        plt.ylabel('$\\rho$')
        plt.show()


def find_mu_diff(lenses):
    """Finds the distance modulus of best fitting cosmology and hence residuals.

    Inputs:
     lenses -- data that contains distance modulus and redshift of each SN.
    """
    # Arbitrarily pick 12'
    lens_gal_single_rad = lenses["Radius12.0"]
    SNzs = np.zeros(len(lens_gal_single_rad))
    SNmus = np.zeros(len(lens_gal_single_rad))
    SNmu_err = np.zeros(len(lens_gal_single_rad))
    c = 0
    for SN_key, SN in lens_gal_single_rad.items():
        SNzs[c] = SN['SNZ']
        SNmus[c] = SN['SNMU']
        SNmu_err[c] = SN['SNMU_ERR']
        c += 1
    z_array = np.linspace(0.0, 0.61, 1001)
    mu_cosm = 5 * np.log10((1 + z_array) * comoving(z_array) * 1000) + 25
    mu_cosm_interp = np.interp(SNzs, z_array, mu_cosm)
    mu_diff = SNmus - mu_cosm_interp
    mu_diff_std = np.std(mu_diff)

    data = {"z": SNzs, "mu": SNmus, "mu_err": SNmu_err, "mu_diff": mu_diff, "mu_cosm": mu_cosm}
    return data


if __name__ == "__main__":
    radius = 12.0
    data, S_data = get_data(new_data=False)
    lensing_gals = sort_SN_gals(data, redo=False)
    SNe_data = find_mu_diff(lensing_gals)
    # for rad in radii[5::5]:
    #     plot_cones(data, lensing_gals, plot_hist=False, cone_radius=rad)
    cone_array = make_test_cones(data, redo=False, plot=False)
    exp_data = find_expected_counts(cone_array, 51, redo=False, plot=False)

    convergence = find_convergence(lensing_gals, exp_data, SNe_data['z'], redo=False, plot_scatter=False,
                                   plot_total=False)
    # plt.plot(S_data['kappa'], S_data['kappa'], color=colours[1])
    # plt.plot(S_data['kappa'], convergence, color=colours[0], marker='o', markersize=3, linestyle='')
    # for SN in range(len(convergence)):
    #     if convergence[SN] < 0.0031 and S_data['kappa'][SN] > 0.0057:
    #     # if convergence_cut[SN] > 0.016:
    #         print(f"Outlier is SN {SN+1}/{len(convergence)}")
    # plt.xlabel('$\kappa$ Smith')
    # plt.ylabel('My $\kappa$')
    # plt.show()

    # plot_Hubble(SNe_data['z'], SNe_data['mu'], SNe_data['mu_err'], SNe_data['mu_diff'], SNe_data['mu_cosm'],
    #             np.linspace(0.0, 0.61, 1001))

    cut_for_corr = [SNe_data['z'][i] > 0.2 for i in range(len(SNe_data['z']))]

    find_correlation(convergence, SNe_data['mu_diff'], plot_correlation=False, plot_radii=True)
