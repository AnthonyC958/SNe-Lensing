import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages

colours = [[0, 165/255, 124/255], [253/255, 170/255, 0], 'C2', 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5']
h = 0.738
H0 = 1000 * 100 * h  # km/s/Gpc
c = 2.998E5  # km/s
G = 6.67E-11
MSOL = 1.989E30

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


def get_h_inv(z_val, OM=0.27, OL=0.73):
    """Integrand for calculating comoving distance.

    Inputs:
     z_val -- redshift value integrand is evaluated at.
    """
    OK = 1.0 - OM - OL
    H = np.sqrt(OK * (1.0 + z_val) ** 2 + OM * (1.0 + z_val) ** 3 + OL)
    return 1.0 / H


def get_adot_inv(a_val, OM=0.27, OL=0.73):
    """Integrand for calculating comoving distance.

    Inputs:
     a_val -- scalefactor value integrand is evaluated at.
    """
    OK = 1.0 - OM - OL
    adot = np.sqrt(OK + OM / a_val + OL * a_val ** 2)
    return 1.0 / adot


def comoving(zs_array, OM=0.27, OL=0.73):
    """Numerical integration of get_h_inv to create an array of comoving values.

    Inputs:
     zs_array -- array of redshifts to evaluate cumulative comoving distance to.
     """
    vecGet_h_inv = np.vectorize(get_h_inv)
    h_invs = vecGet_h_inv(zs_array, OM, OL)
    comoving_coord = sp.cumtrapz(h_invs, x=zs_array, initial=0)

    dist = comoving_coord * c / H0
    return dist


def create_chi_bins(z_lo, z_hi, num_bins, plot=False):
    """Takes a line sight from z_lo to z_hi and divides it into bins even in comoving distance.

    Inputs:
     z_lo -- beginning redshift.
     z_hi -- end redshift.
     num_bins -- number of bins to create.
     plot -- boolean to create plot of chi versus z with bins. Defaults to False.
     """
    z_to_end = np.linspace(z_lo, z_hi, 1001)
    chi_to_end = comoving(z_to_end)
    chi_start = chi_to_end[0]
    chi_end = chi_to_end[-1]

    chi_values = np.linspace(chi_start, chi_end, num_bins * 2 - 1)
    chi_bin_edges = chi_values[0::2]
    chi_widths = chi_bin_edges[1:] - chi_bin_edges[:-1]
    chis = chi_values[1::2]

    z_values = np.interp(chi_values, chi_to_end, z_to_end)
    z_bin_edges = z_values[0::2]
    z_widths = z_bin_edges[1:] - z_bin_edges[:-1]
    zs = z_values[1::2]

    if plot:
        plt.plot([z_bin_edges, z_bin_edges], [chi_bin_edges[0], chi_bin_edges[-1]], color=[0.75, 0.75, 0.73],
                 linestyle='-', linewidth=0.8)
        plt.plot([z_lo, z_hi], [chi_bin_edges, chi_bin_edges], color=[0.75, 0.75, 0.75], linestyle='-', linewidth=0.8)
        plt.plot(z_to_end, chi_to_end, color=colours[1])
        plt.plot(zs, chis, linestyle='', marker='o', markersize=3, color=colours[0])
        plt.xlabel(' $z$')
        
        plt.axis([0, z_hi, 0, chi_end])
        plt.ylabel('$R_0\chi$ (Gpc)')
        plt.savefig("figure_67.eps", format="pdf")
        plt.show()

    return chi_widths, chis, zs, z_widths


def create_z_bins(z_lo, z_hi, num_bins, plot=False):
    """Takes a line sight from z_lo to z_hi and divides it into bins even in redshift.

    Inputs:
     z_lo -- beginning redshift.
     z_hi -- end redshift.
     num_bins -- number of bins to create.
     plot -- boolean to create plot of chi versus z with bins. Defaults to False.
    """
    z_values = np.linspace(z_lo, z_hi, num_bins * 2 - 1)
    z_bin_edges = z_values[0::2]
    z_widths = z_bin_edges[1:] - z_bin_edges[:-1]
    zs = z_values[1::2]

    chi_values = np.linspace(0, 0, len(z_values))
    for k in range(len(z_values)):
        chi = comoving(np.linspace(z_lo, z_values[k], 1001))
        chi_values[k] = chi[-1]

    chi_bin_edges = chi_values[0::2]
    chis = chi_values[1::2]
    chi_widths = chi_bin_edges[1:] - chi_bin_edges[:-1]

    if plot:
        plt.plot([z_bin_edges, z_bin_edges], [chi_bin_edges[0], chi_bin_edges[-1]], color=[0.75, 0.75, 0.75],
                 linestyle='-', linewidth=0.8)
        plt.plot([z_lo, z_hi], [chi_bin_edges, chi_bin_edges], color=[0.75, 0.75, 0.75], linestyle='-', linewidth=0.8)
        plt.plot(np.linspace(z_lo, z_hi, 1001), comoving(np.linspace(z_lo, z_hi, 1001)), color=colours[1])
        plt.plot(zs, chis, linestyle='', marker='o', markersize=3, color=colours[0])
        plt.xlabel(' $z$')
        
        plt.ylabel('$R_0\chi$ (Gpc)')
        plt.axis([0, z_hi, 0, chis[-1] + chi_widths[-1]/2])
        plt.show()

    return chi_widths, chis, zs, z_widths


def single_d_convergence(chi_widths, chis, zs, index, density, SN_dist, OM=0.27):
    """Calculates convergence along the line of sight for a single overdensity in redshift bin i.

    Inputs:
     chi_widths -- the width of the comoving distance bins.
     chis -- the mean comoving distance of each bin.
     zs -- the mean redshift of each bin, for the scale factor.
     index -- which redshift bin will contain the over density.
     density -- the value of the overdensity. Corresponds to
                (observed-expected)/expected when galaxy counting (>= -1).
     SN_dist -- comoving distance to SN along line of sight.
    """
    coeff = 3.0 * H0 ** 2 * OM / (2.0 * c ** 2)
    d_arr = np.linspace(0, 0, len(zs))
    d_arr[index] = density
    sf_arr = 1.0 / (1.0 + zs)
    k_i = coeff * chis * chi_widths * (SN_dist - chis) / SN_dist * d_arr / sf_arr
    return np.sum(k_i)


def single_d_convergence_z(z_widths, chi_widths, chis, zs, index, density, SN_dist, OM=0.27):
    """Same as single_d_convergence but for making dealing with bins equal in z.
    """
    coeff = 3.0 * H0 ** 2 * OM / (2.0 * c ** 2)
    d_arr = np.linspace(0, 0, len(zs))
    d_arr[index] = density
    sf_arr = 1.0 / (1.0 + zs)
    k_i = coeff * chis * chi_widths * (SN_dist - chis) / SN_dist * d_arr / sf_arr * (c * get_h_inv(zs) / H0)
    return np.sum(k_i)


def convergence_error(chi_widths, chis, zs, expected_arr, SN_dist, OM=0.27):
    """Calculates the error in convergence due to Poisson noise in galaxy distribution.

    Inputs:
     chi_widths -- the width of the comoving distance bins.
     chis -- the mean comoving distance of each bin.
     zs -- the mean redshift of each bin, for the scale factor.
     expected_arr -- the array of expected galaxy counts per bin.
     SN_dist -- comoving distance to SN along line of sight.
    """
    coeff = 3.0 * H0 ** 2 * OM / (2.0 * c ** 2)
    sf_arr = 1.0 / (1.0 + zs)
    k_i = coeff * chis * chi_widths * (SN_dist - chis) / SN_dist / expected_arr / sf_arr
    return np.sum(k_i)


def general_convergence(chi_widths, chis, zs, d_arr, SN_dist, OM=0.27):
    """Calculates convergence from an array of overdesnities for all bins along line of sight.

    Inputs:
     chi_widths -- the width of the comoving distance bins.
     chis -- the mean comoving distances of each bin.
     zs -- the mean redshift of each bin, for the scale factor.
     d_arr -- overdensity array.
     SN_dist -- comoving distance to SN along line of sight.
    """
    coeff = 3.0 * H0 ** 2 * OM / (2.0 * c ** 2)
    sf_arr = 1.0 / (1.0 + zs)
    k_i = coeff * chis * chi_widths * (SN_dist - chis) / SN_dist * d_arr / sf_arr
    return np.sum(k_i)


def calc_single_d(chi_widths, chis, zs, z_widths, z_SN, use_chi=True):
    """Uses single_m_convergence with index starting at 0 and going along the entire line of sight.

    Inputs:
     chi_widths -- the width of the comoving distance bins.
     chis -- the mean comoving distances of each bin.
     zs -- the mean redshift of each bin, for the scale factor.
     z_SN -- the reshift of the SN.
     """
    comoving_to_SN = comoving(np.linspace(0, z_SN, 1001))
    chi_SN = comoving_to_SN[-1]

    convergence = np.linspace(0, 0, len(chis))
    delta = 1.0
    for i in range(0, len(chis)):
        if use_chi:
            convergence[i] = single_d_convergence(chi_widths, chis, zs, i, delta, chi_SN)
        else:
            convergence[i] = single_d_convergence_z(z_widths, chi_widths, chis, zs, i, delta, chi_SN)

    return convergence


def plot_smoothed_d(chi_widths, chis, zs, z_SN):
    """Plots general_convergence for overdensities that are increasingly smoothed over the line of sight.
    Also plots the case where the overdensity along the entire line of sight is 0.

    Inputs:
     chi_widths -- the width of the comoving distance bins.
     chis -- the mean comoving distances of each bin.
     zs -- the mean redshift of each bin, for the scale factor.
     z_SN -- the reshift of the SN.
     """
    comoving_to_SN = comoving(np.linspace(0, z_SN, 1001))
    chi_SN = comoving_to_SN[-1]

    size = 2 * len(zs)//2 + 1
    delta = np.zeros((size, len(zs)))

    delta1 = 1
    correction = delta1 / len(zs)
    delta[0][int(len(zs) // 2):int(len(zs) // 2) + 1] = delta1
    delta[-1][int(len(zs) // 2):int(len(zs) // 2) + 1] = -delta1

    for i, s in enumerate(np.arange(1, len(zs)//2 + 1, 1)):
        delta[s][int(len(zs) // 2) - s:int(len(zs) // 2) + s + 1] = delta1 / (2 * s + 1)
        delta[-s-1][int(len(zs) // 2) - s:int(len(zs) // 2) + s + 1] = -delta1 / (2 * s + 1)
    convergence = np.zeros(size)
    convergence_cor = np.zeros(size)

    delta_cor = np.zeros((size, len(zs)))
    delta_cor[0:size//2] = delta[0:size//2]-correction
    delta_cor[size//2:] = delta[size//2:]+correction

    for j in range(size):
        convergence[j] = (general_convergence(chi_widths, chis, zs, delta[j], chi_SN))
        convergence_cor[j] = (general_convergence(chi_widths, chis, zs, delta_cor[j], chi_SN))

    # convergence = np.delete(convergence, size // 2, 0)
    convergence_cor = np.delete(convergence_cor, size // 2, 0)

    plt.plot([size // 2 - 1, size // 2 - 1], [min(convergence)-0.0003, max(convergence)+0.0003],
             color=[0.75, 0.75, 0.75], linestyle='--')
    plt.plot([0, size - 1], [0, 0], color=[0.75, 0.75, 0.75], linestyle='--')
    plt.plot(range(size // 2), convergence[:size // 2], label=f'Total $|\delta|$ = 1', color=colours[0])
    plt.plot(range(size // 2 - 1, size - 1), convergence[size // 2:], color=colours[0])
    plt.plot(range(size - 1), convergence_cor, label=f'Total $|\delta|$ = 0', color=colours[1])
    plt.xlabel("Number of bins smoothed over")
    plt.ylabel(" $\kappa$")
    
    # plt.title(f"Convergence as a function of central overdensity smoothing (z$_S$$_N$ = {z_SN})")
    plt.legend(frameon=0)
    plt.xticks([0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100], [0, 25, 50, 75, 100, 75, 50, 25, 0])
    plt.axis([0, size - 1, min(convergence)-0.0003, max(convergence)+0.0003])

    plt.show()


def compare_z_chi(conv_c_arr, conv_z_arr, chi_bins_c, chi_bins_z, z_bins_z, SN_dist, z_SN):
    """Plots the convergence distribution for even chi and z, over chi or z.

    Inputs:
     conv_c_arr -- array of convergence for even comoving bins.
     conv_z_arr -- array of convergence for even redshift bins.
     chi_bins_c -- mean comoving distance values of the even comoving bins.
     chi_bins_z -- mean comoving distance values of the even redshift bins.
     z_bins_z --
     SN_dist -- comoving distance to SN along line of sight.
     z_SN -- the reshift of the SN.
    """
    plt.plot([SN_dist / 2, SN_dist / 2], [0, 1.1 * max(conv_c_arr)], linestyle='--', color=[0.75, 0.75, 0.75],
             linewidth=1)
    plt.plot(chi_bins_c, conv_c_arr, label='Even $\chi$', color=colours[0])
    plt.plot(chi_bins_z, conv_z_arr / (get_h_inv(z_bins_z)), label='Even $z$', color=colours[1])
    plt.xlabel("$\chi_{Overdensity}$ (Gpc)")
    plt.ylabel("$\kappa$")
    
    plt.legend(frameon=0)
    plt.axis([0, SN_dist, 0, 1.1 * max(conv_c_arr)])
    plt.show()

    plt.plot([z_SN / 2, z_SN / 2], [0, 1.1 * max(conv_c_arr)], linestyle='--',
             color=[0.75, 0.75, 0.75], linewidth=1)
    plt.plot(z_binsc, conv_c_arr, label='Even $\chi$', color=colours[0])
    print("Peak at z =", z_binsc[np.argmin(np.abs(conv_c_arr - max(conv_c_arr)))])
    plt.plot(z_bins_z, conv_z_arr / (get_h_inv(z_bins_z)), label='Even $z$', color=colours[1])
    plt.xlabel("$z_{Overdensity}$")
    plt.ylabel("$\kappa$")
    plt.legend(frameon=0)
    
    plt.axis([0, z_SN, 0, 1.1 * max(conv_c_arr)])
    plt.show()


def smoothed_m_convergence(tests, OM=0.27):
    test_range = np.arange(3, tests, 2)
    conv = np.zeros(len(test_range))
    mass_mag = 15
    mass = MSOL * 10 ** mass_mag
    bin_lengths = np.zeros(len(test_range))
    for num, y in enumerate(test_range):
        (comoving_binwidths, comoving_bins, z_bins, z_widths) = create_chi_bins(0, SN_redshift, y + 1)
        cone_rad = comoving_bins[len(z_bins) // 2] * (1 + z_bins[len(z_bins) // 2]) * 0.00349066
        # distance * 12 arcmin = 0.00349066 rad
        vol_bin = (comoving_binwidths[0] * (1 + z_bins[len(z_bins) // 2])) * np.pi * cone_rad ** 2
        Hz = get_h_inv(z_bins[len(z_bins) // 2]) ** (-1) * H0
        d_m = 8 * np.pi * G * mass / (3 * OM * vol_bin * Hz ** 2 * 3.086E31) - 1
        conv[num] = single_d_convergence(comoving_binwidths, comoving_bins, z_bins, len(z_bins) // 2, d_m, SN_chi)
        bin_lengths[num] = round(1000 * comoving_binwidths[0], 1)
    plt.plot(test_range[10::], conv[10::], label='$M_{{cluster}} = 10^{0} M_\odot$'.format({mass_mag}),
             color=colours[1])
    plt.plot(test_range[10::], np.zeros(len(test_range[10::])), color=[0.75, 0.75, 0.75], linestyle='--')
    plt.xticks(test_range[10::tests // 20], bin_lengths[10::tests // 20], rotation=45)
    plt.xlabel("Bin length (Mpc)")
    plt.ylabel("$\kappa$")
    plt.legend(frameon=0)
    plt.axis([15, 799, -0.002325, 0.0017])
    plt.show()


if __name__ == "__main__":
    SN_redshift = 7.0
    num_bin = 100

    chi_to_SN = comoving(np.linspace(0, SN_redshift, 501))
    SN_chi = chi_to_SN[-1]
    print("SN redshift", SN_redshift, "\nSN comoving distace", SN_chi)
    (comoving_binwidthsc, comoving_binsc, z_binsc, z_widthsc) = create_chi_bins(0, SN_redshift, num_bin)
    (comoving_binwidthsz, comoving_binsz, z_binsz, z_widthsz) = create_z_bins(0, SN_redshift, num_bin)

    single_conv_c = calc_single_d(comoving_binwidthsc, comoving_binsc, z_binsc, z_widthsc, SN_redshift)
    single_conv_z = calc_single_d(comoving_binwidthsz, comoving_binsz, z_binsz, z_widthsz, SN_redshift)
    plot_smoothed_d(comoving_binwidthsc, comoving_binsc, z_binsc, SN_redshift)

    compare_z_chi(single_conv_c, single_conv_z, comoving_binsc, comoving_binsz, z_binsz, SN_chi, SN_redshift)

    num_test = 800
    smoothed_m_convergence(num_test)
