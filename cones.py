from Convergence import *
from astropy.io import fits
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from scipy.optimize import curve_fit
from scipy.stats import rankdata
import pickle

colours = [[0, 150/255, 100/255], [253/255, 170/255, 0], 'C2', 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5']
blue = [23/255, 114/255, 183/255, 0.75]
orange = [255/255, 119/255, 15/255, 0.75]
green = [0, 150/255, 100/255, 0.75]
yellow = [253/255, 170/255, 0, 0.75]
grey = [0.75, 0.75, 0.75]
names = ['STRIPE82_SPECTROSCOPIC_CHAZ_NOTCLEANED_ms77.fit', 'boss_206+SDSS_213_all_cuts_new_mu_dmu1_new.fits']

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
        with fits.open(names[0])as hdul1:
            with fits.open(names[1]) as hdul2:
                RA1 = [hdul1[1].data['RA'][i] for i in np.arange(len(hdul1[1].data['RA'])) if
                       hdul1[1].data['CLASS'][i] == 'GALAXY' and hdul1[1].data['Z'][i] >= 0.01]
                DEC1 = [hdul1[1].data['DEC'][i] for i in np.arange(len(hdul1[1].data['DEC'])) if
                        hdul1[1].data['CLASS'][i] == 'GALAXY' and hdul1[1].data['Z'][i] >= 0.01]
                for num, ra in enumerate(RA1):
                    if ra > 60:
                        RA1[num] -= 360
                RA2 = [hdul2[1].data['RA'][i] for i in np.arange(len(hdul2[1].data['RA'])) if
                       hdul2[1].data['Z_BOSS'][i] >= 0.05]
                DEC2 = [hdul2[1].data['DECL'][i] for i in np.arange(len(hdul2[1].data['DECL'])) if
                        hdul2[1].data['Z_BOSS'][i] >= 0.05]

                z1 = hdul1[1].data['Z']
                z2 = hdul2[1].data['Z_BOSS']
                mu = hdul2[1].data['MU']
                mu_err = hdul2[1].data['DMU1']

                cut_data = np.array([RA1, DEC1, RA2, DEC2, z1, z2, mu, mu_err])
                pickle_out = open("cut_data.pickle", "wb")
                pickle.dump(cut_data, pickle_out)
                pickle_out.close()
    else:
        pickle_in = open("cut_data.pickle", "rb")
        cut_data = pickle.load(pickle_in)
        RA1 = cut_data[0]
        DEC1 = cut_data[1]
        RA2 = cut_data[2]
        DEC2 = cut_data[3]
        z1 = cut_data[4]
        z2 = cut_data[5]
        mu = cut_data[6]
        mu_err = cut_data[7]

    patches = []
    for x, y in zip(RA2, DEC2):
        circle = Circle((x, y), 0.2)
        patches.append(circle)

    return RA1, DEC1, RA2, DEC2, patches, z1, z2, mu, mu_err


def sort_SN_gals(RA1, DEC1, RA2, DEC2, z1, z2, mu, mu_err, redo=False):
    """Either sorts galaxies into SN cones or loads sorted data from file.

    Inputs:
     RA1 -- right ascensions of galaxies.
     RA2 -- right ascensions of SNe.
     DEC1 -- declinations of galaxies.
     DEC2 -- declinations of SNe.
     z1 -- redshifts of galaxies.
     z2 -- redshifts of SNe
     mu -- distance moduli of SNe
     mu_err -- error in distance moduli.
     redo -- boolean that determines whether data is loaded or sorted. Default false.
    """
    if redo:
        lenses = {}
        for num, SRA, SDE, SZ, SM, SE in zip(np.linspace(0, len(RA2) - 1, len(RA2)), RA2, DEC2, z2, mu, mu_err):
            lenses[f'SN{int(num)+1}'] = {'RAs': [], 'DECs': [], 'Zs': [], 'SNZ': SZ, 'SNMU': SM, 'SNMU_ERR': SE}
            for GRA, GDE, GZ in zip(RA1, DEC1, z1):
                if (GRA - SRA) ** 2 + (GDE - SDE) ** 2 <= 0.2 ** 2:
                    lenses[f'SN{int(num)+1}']['RAs'].append(GRA)
                    lenses[f'SN{int(num)+1}']['DECs'].append(GDE)
                    lenses[f'SN{int(num)+1}']['Zs'].append(GZ)
            print(f'Finished {int(num)+1}/{len(RA2)}')

        pickle_out = open("lenses.pickle", "wb")
        pickle.dump(lenses, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("lenses.pickle", "rb")
        lenses = pickle.load(pickle_in)

    return lenses


def plot_cones(RA1, RA2, DEC1, DEC2, z1, z2, lenses, patches):
    """Plots all galaxies and SNe along with visualisation of cones and galaxies contributing to lensing.

    Input:
     RA1 -- right ascensions of galaxies.
     RA2 -- right ascensions of SNe.
     DEC1 -- declinations of galaxies.
     DEC2 -- declinations of SNe.
     lenses -- sorted galaxy dictionary.
     patches -- circles that represent the cones in the plot.
    """
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
        indices1 = dict1['Zs'] < dict1['SNZ']
        ax.plot(RAs[indices1], DECs[indices1], marker='o', linestyle='', markersize=3, color=colours[0],
                label="Foreground" if SN == 'SN1' else "")
    p = PatchCollection(patches, alpha=0.4, color=colours[0])
    ax.add_collection(p)
    ax.plot(RA2, DEC2, marker='o', linestyle='', markersize=3, label='Supernova', color=colours[1])
    plt.xlabel('$\\alpha$')
    plt.ylabel('$\delta$')
    plt.legend(loc='lower right')
    plt.axis('equal')
    plt.xlim([24.5, 27.5])
    plt.ylim([-1, 1])
    plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False)
    plt.show()

    labels = ['Galaxies', 'Supernovae']
    cols = [green, yellow]
    for num, z in enumerate([z1, z2]):
        plt.hist([i for i in z if i <= 0.6], bins=np.arange(0, 0.6 + 0.025, 0.025), normed='max', linewidth=1,
                 fc=cols[num], label=f'{labels[num]}', edgecolor=colours[num])
    plt.xlabel('$z$')
    plt.ylabel('Normalised Count')
    plt.legend(frameon=0)
    
    plt.show()


def make_test_cones(RA1, DEC1, z1, redo=False):
    """Creates an array of 12 arcmin cones all across data or loads test cones from file.
    Also distribution of galaxy count per bin.

    Inputs:
     RA1 -- right ascensions of galaxies.
     DEC1 -- declinations of galaxies.
     z1 -- redshifts of galaxies.
     redo -- boolean that determines whether cones are created or loaded. Default false.
    """
    tests = []
    for a in range(272):
        for b in range(6):
            test = [-50.6, 1.0]  # Upper left corner of STRIPE82
            test[0] += a * 0.4
            test[1] -= b * 0.4
            test[0] = round(test[0], 1)
            test[1] = round(test[1], 1)
            tests.append(test)
    if redo:
        test_cones = {}
        for num, loc, in enumerate(tests):
            test_cones[f'c{int(num)+1}'] = {'Total': 0, 'Zs': []}
            for GRA, GDE, GZ in zip(RA1, DEC1, z1):
                if (GRA - loc[0]) ** 2 + (GDE - loc[1]) ** 2 <= 0.2 ** 2:
                    test_cones[f'c{int(num)+1}']['Zs'].append(GZ)
                test_cones[f'c{int(num)+1}']['Total'] = len(test_cones[f'c{int(num)+1}']['Zs'])
            print(f'Finished {int(num)+1}/{len(tests)}')

        pickle_out = open("test_cones.pickle", "wb")
        pickle.dump(test_cones, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("test_cones.pickle", "rb")
        test_cones = pickle.load(pickle_in)

    plt.hist([test_cones[f'c{i+1}']['Total'] for i in range(len(test_cones))], density=1,
             bins=20, edgecolor=colours[0], fc=green, linewidth=1)
    plt.xlabel('Number of Galaxies')
    plt.ylabel('Count')
    plt.show()

    return test_cones


def find_expected_counts(test_cones, bins, redo=False):
    """Uses the test cones to find the expected number of galaxies per bin, for bins of even comoving distance.

    Inputs:
     test_cones -- data to obtain expected counts from.
     bins -- number of bins along the line of sight to maximum SN comoving distance.
     redo -- boolean that determines whether expected counts are calculated or loaded. Default false.
    """
    max_z = max([max(test_cones[f'c{i+1}']['Zs']) for i in range(len(test_cones))])
    chi_bin_widths, chi_bins, z_bins, z_bin_widths = create_chi_bins(0, max_z, bins)
    limits = np.cumsum(z_bin_widths)
    if redo:
        expected = np.zeros((len(limits), len(test_cones)))
        num = 0
        for num1, lim in enumerate(limits):
            for num2, _ in enumerate(test_cones.items()):
                expected[num1][num2] = sum([test_cones[f'c{num2+1}']['Zs'][i] < lim
                                            for i in range(len(test_cones[f'c{num2+1}']['Zs']))])
                num += 1
                if num % 1000 == 0:
                    print(f"Finished {num}/{len(limits)*len(test_cones)}")
        print("Finished")

        pickle_out = open("expected.pickle", "wb")
        pickle.dump(expected, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("expected.pickle", "rb")
        expected = pickle.load(pickle_in)

    expected = np.diff([np.mean(expected[i][:]) for i in range(len(limits))])
    plt.plot([0, 5], [0, 0], color=grey, linestyle='--')
    plt.plot(limits[1:], expected, marker='o', markersize=2.5, color=colours[0])
    plt.xlabel('$z$')
    plt.ylabel('Expected Count')
    plt.xlim([0, 3])
    plt.show()

    return limits, expected, chi_bin_widths, chi_bins, z_bins


def find_convergence(lenses, SNz, cut, cut2, limits):
    # ################################# Still references globals; need to fix ##################################### #
    """Finds the convergence along each line of sight to a SN.

    Inputs:
     lenses -- dictionary containing all galaxies that contribute to lensing.
     SNz -- redshifts of each SN.
     cut -- logical array that select SNe that have z<0.65.
     cut2 -- logical array that select SNe that are <5sigma from mean.
     limits -- bin edges.
    """
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
                                               for i in range(len(lenses[f'SN{num1+1}']['Zs']))])
        num += 1
        if num % 50 == 0:
            print(f"Finished {num}/{len(lenses)}")

    SNzs_new = SNz[cut]
    d_arr = {}
    convergence = np.zeros(len(counts))
    conv_err = np.zeros(len(counts))
    num = 0
    for key, SN in counts.items():
        d_arr[f"{key}"] = (SN - exp[:len(SN)]) / exp[:(len(SN))]
        convergence[num] = general_convergence(chi_widths[:len(SN)], chis[:len(SN)], zs[:len(SN)],
                                               d_arr[f"{key}"], chiSNs[num])
        conv_err[num] = convergence_error(chi_widths[:len(SN)], chis[:len(SN)], zs[:len(SN)],
                                            exp[:len(SN)], chiSNs[num])
        num += 1

    convergence_new = convergence[cut]
    conv_err_new = conv_err[cut]
    conv_err_new += 0

    ax = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    ax.set_ylabel("$\kappa$")
    ax.set_xlabel("$z$")
    ax2.set_xlabel("Count")
    ax.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    ax2.set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    ax.plot([0, 0.6], [0, 0], color=grey, linestyle='--')
    ax.axis([0, 0.6, -0.015, 0.02])
    ax2.axis([0, 160, -0.015, 0.02])
    # ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0])
    ax.set_xticklabels([0, 0.2, 0.4, 0])
    ax.plot(SNzs_new[cut2], convergence_new[cut2], linestyle='', marker='o', markersize=2, color=colours[0])
    ax2.hist(convergence_new[cut2], bins=np.arange(-0.015, 0.02 + 0.001, 0.001), orientation='horizontal',
             fc=green, edgecolor=colours[0])
    plt.show()

    return convergence_new


# def plot_Hubble(z, mu, mu_err, OM=0.27, OL=0.73, max_x=0.6):
#     """Plots the Hubble diagram (distance modulus against redshift), including the best fitting cosmology, and
#     residuals from best cosmology.
#     """
#     z_arr = np.linspace(0.0, max(z) + 0.2, 1001)
#     cosm = 5 * np.log10((1 + z_arr) * b_comoving(0, max(z), OM, OL) * 1000) + 25
#     cosm_interp = np.interp(z, z_arr, cosm)
#     mu_diff = mu - cosm_interp
#     ax = plt.subplot2grid((2, 1), (0, 0))
#     ax2 = plt.subplot2grid((2, 1), (1, 0))
#     ax.set_ylabel("$\mu$")
#     ax2.set_xlabel("$z$")
#     ax2.set_ylabel("$\Delta\mu$")
#     plt.subplots_adjust(wspace=0, hspace=0)
#     ax.set_xticklabels([])
#     ax.tick_params(labelsize=12)
#     ax.errorbar(z, mu, mu_err, linestyle='', linewidth=0.8, marker='o',
#                 markersize=2, capsize=2, color='C3', zorder=0)
#     ax.set_ylim([35, 45])
#     ax.set_xlim([0, max_x])
#     ax.plot(z_arr, cosm, linestyle='--', linewidth=0.8, color='C0', zorder=10)
#     ax2.errorbar(z, mu_diff, mu_err, linestyle='', linewidth=1, marker='o',
#                  markersize=2, capsize=2, color='C3', zorder=0)
#     ax2.plot(z_arr, np.zeros(len(z_arr)), zorder=10, color='C0', linewidth=0.8, linestyle='--')
#     ax2.set_ylim(-1.4, 1.4)
#     ax2.set_xlim([0, max_x])
#     ax2.tick_params(labelsize=12)
#     ax.axvspan(0, 0.2, alpha=0.1, color=colours[1])
#     ax.axvspan(0.2, 0.6, alpha=0.1, color=colours[0])
#     ax2.axvspan(0, 0.2, alpha=0.1, color=colours[1])
#     ax2.axvspan(0.2, 0.6, alpha=0.1, color=colours[0])
#     ax.text(0.05, 41, 'Peculiar\nVelocities', color=colours[1], fontsize=16)
#     ax.text(0.35, 38, 'Lensing', color=colours[0], fontsize=16)
#
#     plt.show()


def plot_Hubble(z, mu, mu_err, mu_diff, z_arr):
    """Plots the Hubble diagram (distance modulus agaionst redshift), including the best fitting cosmology, and
    residuals from best cosmology.
    """
    ax = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    ax.axvspan(0, 0.2, alpha=0.1, color=colours[1])
    ax.axvspan(0.2, 0.6, alpha=0.1, color=colours[0])
    ax2.axvspan(0, 0.2, alpha=0.1, color=colours[1])
    ax2.axvspan(0.2, 0.6, alpha=0.1, color=colours[0])
    ax.text(0.05, 41, 'Peculiar\nVelocities', color=colours[1], fontsize=16)
    ax.text(0.35, 38, 'Lensing', color=colours[0], fontsize=16)
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
    plt.plot([min(conv), max(conv)], [0, 0], color=grey, linestyle='--')
    plt.plot(conv, mu_diff, linestyle='', marker='o', markersize=2, color=colours[0])
    plt.plot(conv, fit, color=colours[1], label=f'$\Delta\mu = {round(float(grad),3)}\kappa$')
    plt.xlabel('$\kappa$')
    plt.ylabel('$\Delta\mu$')
    plt.xlim([-0.008, 0.011])
    plt.legend(frameon=0, loc='lower right')
    plt.ylim([-0.3, 0.3])
    plt.text(0.0038, 0.09, f'$\\rho$ = {round(rho, 3)} $\pm$ {round(rho_err, 3)}', fontsize=16)
    # print([convergence_cut[cuts2][i] for i in range(len(convergence_cut[cuts2]))])
    # print([mu_diff_cut[cuts2][i] for i in range(len(convergence_cut[cuts2]))])
    # print([SNmu_err_cut[cuts2][i] for i in range(len(convergence_cut[cuts2]))])
    print("Gradient:", grad)
    plt.show()


if __name__ == "__main__":
    (RAgal, DECgal, RASN, DECSN, circles, zgal, zSN, muSN, mu_errSN) = get_data()
    lensing_gals = sort_SN_gals(RAgal, DECgal, RASN, DECSN, zgal, zSN, muSN, mu_errSN)
    # plot_cones(RAgal, RASN, DECgal, DECSN, zgal, zSN, lensing_gals, circles)
    cone_array = make_test_cones(RAgal, DECgal, zgal)
    bin_limits, exp, chi_widths, chis, zs = find_expected_counts(cone_array, 100)
    print(max([max(cone_array[f'c{i+1}']['Zs']) for i in range(len(cone_array))]))
    SNzs = np.zeros(len(lensing_gals))
    SNmus = np.zeros(len(lensing_gals))
    SNmu_err = np.zeros(len(lensing_gals))
    c = 0
    for _, supernova in lensing_gals.items():
        SNzs[c] = supernova['SNZ']
        SNmus[c] = supernova['SNMU']
        SNmu_err[c] = supernova['SNMU_ERR']
        c += 1

    cuts1 = [SNzs[i] < 0.65 for i in range(len(SNzs))]
    SNzs_cut = SNzs[cuts1]
    SNmus_cut = SNmus[cuts1]
    SNmu_err_cut = SNmu_err[cuts1]

    z_array = np.linspace(0.0, 0.61, 1001)
    mu_cosm = 5 * np.log10((1 + z_array) * comoving(z_array) * 1000) + 25
    mu_cosm_interp = np.interp(SNzs_cut, z_array, mu_cosm)
    mu_diff_cut = SNmus_cut - mu_cosm_interp
    mu_diff_std = np.std(mu_diff_cut)
    mu_diff_mean = np.mean(mu_diff_cut)
    cuts2 = [-3.9 * mu_diff_std < mu_diff_cut[i] < 3.9 * mu_diff_std and SNzs_cut[i] > 0.2
             for i in range(len(mu_diff_cut))]  # really broken

    convergence_cut = find_convergence(lensing_gals, SNzs, cuts1, cuts2, bin_limits)

    # plot_Hubble(SNzs_cut[cuts2], SNmus_cut[cuts2], SNmu_err_cut[cuts2], mu_diff_cut[cuts2])
    plot_Hubble(SNzs_cut[cuts2], SNmus_cut[cuts2], SNmu_err_cut[cuts2], mu_diff_cut[cuts2], z_array)

    find_correlation(convergence_cut[cuts2], mu_diff_cut[cuts2])
