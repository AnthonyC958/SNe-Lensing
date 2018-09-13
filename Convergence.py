import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp

colours = [[0, 150/255, 100/255], [225/255, 149/255, 0], [207/255, 0, 48/255], 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5']
blue = [23/255, 114/255, 183/255, 0.75]
orange = [255/255, 119/255, 15/255, 0.75]
green = [0, 150/255, 100/255, 0.25]
yellow = [253/255, 170/255, 0, 0.75]
grey = [0.75, 0.75, 0.75]
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
     OM -- matter density parameter. Defaults to 0.27.
     OL -- dark energy density parameter. Defaults to 0.73.
    """
    OK = 1.0 - OM - OL
    H = np.sqrt(OK * (1.0 + z_val) ** 2 + OM * (1.0 + z_val) ** 3 + OL)
    return 1.0 / H


def comoving(zs_array, OM=0.27, OL=0.73, h=0.738):
    """Numerical integration of get_h_inv to create an array of comoving values.

    Inputs:
     zs_array -- array of redshifts to evaluate cumulative comoving distance to.
     OM -- matter density parameter. Defaults to 0.27.
     OL -- dark energy density parameter. Defaults to 0.73.
    """
    vecGet_h_inv = np.vectorize(get_h_inv)
    h_invs = vecGet_h_inv(zs_array, OM, OL)
    comoving_coord = sp.cumtrapz(h_invs, x=zs_array, initial=0)
    H0 = 1000 * 100 * h  # km/s/Gpc
    dist = comoving_coord * c / H0
    return dist


def b_comoving_integrand(a_val, OM=0.27, OL=0.73):
    """Numerical integration of get_h_inv to create an array of comoving values.

    Inputs:
     a_val -- scalefactor value.
     OM -- matter density parameter. Defaults to 0.27.
     OL -- dark energy density parameter. Defaults to 0.73.
    """
    OK = 1 - OM - OL
    return 1 / np.sqrt(a_val * OM + a_val ** 2 * OK + a_val ** 4 * OL)


def b_comoving(z_lo, z_hi, OM=0.27, OL=0.73, n=1001, h=0.738):
    """Numerical integration of b_comoving_integrand to create an array of comoving values. Uses start and end redshift
    as opposed to an array of z values.

    Inputs:
     z_lo -- start redshift.
     z_hi -- end redshift.
     OM -- matter density parameter. Defaults to 0.27.
     OL -- dark energy density parameter. Defaults to 0.73.
     n -- number of integration steps. Defaults to 1001.
    """
    vecIntegrand = np.vectorize(b_comoving_integrand)
    a1 = 1 / (1 + z_hi)  # backwards in a
    a2 = 1 / (1 + z_lo)
    a_arr = np.linspace(a1, a2, n)
    integrands = vecIntegrand(a_arr, OM, OL)
    comoving_coord = sp.cumtrapz(integrands, x=a_arr, initial=0)
    H0 = 1000 * 100 * h  # km/s/Gpc
    return comoving_coord * c / H0


def create_chi_bins(z_lo, z_hi, num_bins, plot=False):
    """Takes a line sight from z_lo to z_hi and divides it into bins even in comoving distance.

    Inputs:
     z_lo -- beginning redshift.
     z_hi -- end redshift.
     num_bins -- number of bins to create.
     plot -- boolean to create plot of chi versus z with bins. Defaults to False.
    """
    z_to_end = np.linspace(z_lo, z_hi, 1001)
    chi_to_end = b_comoving(z_lo, z_hi)
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


def create_z_bins(z_lo, z_hi, num_bins, plot=False, OM=0.27, OL=0.73, h=0.738):
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
        chi = b_comoving(z_lo, z_values[k], OM=OM, OL=OL, h=h)
        chi_values[k] = chi[-1]

    chi_bin_edges = chi_values[0::2]
    chis = chi_values[1::2]
    chi_widths = chi_bin_edges[1:] - chi_bin_edges[:-1]

    if plot:
        plt.plot([z_bin_edges, z_bin_edges], [chi_bin_edges[0], chi_bin_edges[-1]], color=[0.75, 0.75, 0.75],
                 linestyle='-', linewidth=0.8)
        plt.plot([z_lo, z_hi], [chi_bin_edges, chi_bin_edges], color=[0.75, 0.75, 0.75], linestyle='-', linewidth=0.8)
        plt.plot(np.linspace(z_lo, z_hi, 1001), b_comoving(z_lo, z_hi, OM=OM, OL=OL, h=h), color=colours[1])
        plt.plot(zs, chis, linestyle='', marker='o', markersize=3, color=colours[0])
        plt.xlabel(' $z$')
        
        plt.ylabel('$R_0\chi$ (Gpc)')
        plt.axis([0, z_hi, 0, chis[-1] + chi_widths[-1]/2])
        plt.show()

    return chi_widths, chis, zs, z_widths


def single_d_convergence(chi_widths, chis, zs, index, mass, SN_dist, OM=0.27, h=0.738):
    """Calculates convergence along the line of sight for a single overdensity in redshift bin i.

    Inputs:
     chi_widths -- the width of the comoving distance bins.
     chis -- the mean comoving distance of each bin.
     zs -- the mean redshift of each bin, for the scale factor.
     index -- which redshift bin will contain the over density.
     density -- the value of the overdensity. Corresponds to (observed-expected)/expected when galaxy counting (>= -1).
     SN_dist -- comoving distance to SN along line of sight.
     OM -- matter density parameter. Defaults to 0.27.
    """
    H0 = 1000 * 100 * h  # km/s/Gpc
    coeff = 3.0 * H0 ** 2 * OM / (2.0 * c ** 2)
    # print(chi_widths)
    chi_widths[0] = chis[1] / 2
    chi_widths[-1] = (SN_dist - chis[-2]) / 2
    chi_widths[1:-1] = (chis[2:] - chis[:-2]) / 2
    # print(chi_widths)
    d_arr = np.linspace(0, 0, len(zs))
    # rho_0 = 3 * OM * H0 ** 2 / (8 * np.pi * G)
    # rho_bar = 1 / (1 + zs[index]) ** 3 * rho_0
    # rho = 10E17
    # d_m = rho / rho_bar - 1
    d_m = 1
    d_arr[index] = d_m
    # print(d_arr[index])
    sf_arr = 1.0 / (1.0 + zs)
    k_i = coeff * chis * chi_widths * (SN_dist - chis) / SN_dist * d_arr / sf_arr
    return np.sum(k_i)


def single_d_convergence_z(z_widths, chis, zs, index, mass, SN_dist, OM=0.27, h=0.738):
    """Same as single_d_convergence but for making dealing with bins equal in z.

    Inputs:
     z_widths -- the width of the redshift bins.
     chis -- the mean comoving distance of each bin.
     zs -- the mean redshift of each bin, for the scale factor.
     index -- which redshift bin will contain the over density.
     density -- the value of the overdensity. Corresponds to (observed-expected)/expected when galaxy counting (>= -1).
     SN_dist -- comoving distance to SN along line of sight.
     OM -- matter density parameter. Defaults to 0.27.
    """
    H0 = 1000 * 100 * h  # km/s/Gpc
    coeff = 3.0 * H0 ** 2 * OM / (2.0 * c ** 2)
    d_arr = np.linspace(0, 0, len(zs))
    # rho_0 = 3 * OM * H0 ** 2 / (8 * np.pi * G)
    # rho_bar = 1 / (1 + zs[index]) ** 3 * rho_0
    # rho = 10E17
    # d_m = rho / rho_bar - 1
    d_m = 1
    d_arr[index] = d_m
    # print(d_arr[index])
    sf_arr = 1.0 / (1.0 + zs)
    k_i = coeff * chis * z_widths * (SN_dist - chis) / SN_dist * d_arr / sf_arr * c / H0 * get_h_inv(zs)
    return np.sum(k_i)


def convergence_error(chi_widths, chis, zs, expected_arr, SN_dist, OM=0.27, h=0.738):
    """Calculates the error in convergence due to Poisson noise in galaxy distribution.

    Inputs:
     chi_widths -- the width of the comoving distance bins.
     chis -- the mean comoving distance of each bin.
     zs -- the mean redshift of each bin, for the scale factor.
     expected_arr -- the array of expected galaxy counts per bin.
     SN_dist -- comoving distance to SN along line of sight.
     OM -- matter density parameter. Defaults to 0.27.
    """
    H0 = 1000 * 100 * h  # km/s/Gpc
    coeff = 3.0 * H0 ** 2 * OM / (2.0 * c ** 2)
    sf_arr = 1.0 / (1.0 + zs)
    k_i = coeff * chis * chi_widths * (SN_dist - chis) / SN_dist / expected_arr / sf_arr
    return np.sum(k_i)


def general_convergence(chi_widths, chis, zs, d_arr, SN_dist, OM=0.27, h=0.738):
    """Calculates convergence from an array of overdesnities for all bins along line of sight.

    Inputs:
     chi_widths -- the width of the comoving distance bins.
     chis -- the mean comoving distances of each bin.
     zs -- the mean redshift of each bin, for the scale factor.
     d_arr -- overdensity array.
     SN_dist -- comoving distance to SN along line of sight.
     OM -- matter density parameter. Defaults to 0.27.
    """
    H0 = 1000 * 100 * h  # km/s/Gpc
    coeff = 3.0 * H0 ** 2 * OM / (2.0 * c ** 2)
    sf_arr = 1.0 / (1.0 + zs)
    k_i = coeff * chis * chi_widths * (SN_dist - chis) / SN_dist * d_arr / sf_arr
    return np.sum(k_i), k_i


def calc_single_d(chi_widths, chis, zs, z_widths, z_SN, use_chi=True):
    """Uses single_m_convergence with index starting at 0 and going along the entire line of sight.

    Inputs:
     chi_widths -- the width of the comoving distance bins.
     chis -- the mean comoving distances of each bin.
     zs -- the mean redshift of each bin, for the scale factor.
     z_SN -- the reshift of the SN.
     use_chi -- boolean that determined whether equal comoving distance or redshift bins are used.
    """
    comoving_to_SN = b_comoving(0, z_SN)
    chi_SN = comoving_to_SN[-1]

    convergence = np.linspace(0, 0, len(chis))
    mass = MSOL * 10 ** 15
    for i in range(0, len(chis)):
        if use_chi:
            convergence[i] = single_d_convergence(chi_widths, chis, zs, i, mass, chi_SN)
        else:
            convergence[i] = single_d_convergence_z(z_widths, chis, zs, i, mass, chi_SN)

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
    comoving_to_SN = b_comoving(0, z_SN)
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
        convergence[j], _ = general_convergence(chi_widths, chis, zs, delta[j], chi_SN)
        convergence_cor[j], _ = general_convergence(chi_widths, chis, zs, delta_cor[j], chi_SN)

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

    # for array in delta_cor[:len(delta_cor) // 2]:
    #     plt.bar(chis, array, width=chi_widths[0], edgecolor=[colours[0]]*len(chis), fc=green)
    #     plt.tick_params(labelsize=26)
    #     plt.xlabel("$\chi$ (Gpc)", fontsize=30)
    #     plt.ylabel("$\delta_i$", fontsize=30)
    #     plt.plot([-0.2, 8.68563], [0, 0], color='k', linestyle='-')
    #     plt.xlim([-0.2, 8.68563])
    #     plt.ylim([-0.08, 1])
    # fig = matplotlib.pyplot.gcf()
    # fig.tight_layout()
    # fig.set_size_inches(7, 5.5)
    # plt.show()


def compare_z_chi(conv_c_arr, conv_z_arr, chi_bins_c, chi_bins_z, z_bins_z, z_bins_c, SN_dist, z_SN):
    """Plots the convergence distribution for even chi and z, over chi or z.

    Inputs:
     conv_c_arr -- array of convergence for even comoving bins.
     conv_z_arr -- array of convergence for even redshift bins.
     chi_bins_c -- mean comoving distance values of the equal comoving bins.
     chi_bins_z -- mean comoving distance values of the equal redshift bins.
     z_bins_c -- mean redshift values of the equal comoving bins.
     z_bins_z -- mean redshift values of the equal redshift bins.
     SN_dist -- comoving distance to SN along line of sight.
     z_SN -- the reshift of the SN.
    """
    plt.plot([SN_dist / 2, SN_dist / 2], [0, 1.1 * max(conv_c_arr)], linestyle='--', color=[0.75, 0.75, 0.75],
             linewidth=1)
    chi_peak_c = np.array(chi_bins_c)[np.argmin(np.abs(conv_c_arr - max(conv_c_arr)))]
    chi_peak_z = np.array(chi_bins_z)[np.argmin(np.abs(conv_z_arr - max(conv_z_arr)))]

    plt.plot(chi_bins_c, 1000 * conv_c_arr, label='Even $\chi$', color=colours[0])
    plt.plot(chi_peak_c, 1000 * max(conv_c_arr), marker='x', color=colours[0])
    plt.text((chi_peak_z + chi_peak_c) / 2, 1000 * max(conv_c_arr)*3.5/5, f'$\chi$ = {round(chi_peak_c, 2)} Gpc',
             fontsize=16, ha='center', color=colours[0])

    plt.plot(chi_bins_z, 1000 * conv_z_arr, label='Even $z$', color=colours[1])
    plt.plot(chi_peak_z, 1000 * max(conv_z_arr), marker='x', color=colours[1])
    plt.text((chi_peak_z + chi_peak_c) / 2, 1000 * max(conv_c_arr)*3/5, f'$\chi$ = {round(chi_peak_z, 2)} Gpc',
             fontsize=16, ha='center', color=colours[1])
    plt.xlabel("$\chi_L$ (Gpc)")
    plt.ylabel("$\kappa$ ($\\times 10^{-3}$)")
    
    plt.legend(frameon=0, loc='upper left')
    # plt.axis([0, SN_dist, 0, 1.1 * max(conv_c_arr)])
    plt.show()

    plt.plot([z_SN / 2, z_SN / 2], [0, 1.1 * max(conv_c_arr)], linestyle='--',
             color=[0.75, 0.75, 0.75], linewidth=1)
    z_peak_c = np.array(z_bins_c)[np.argmin(np.abs(conv_c_arr - max(conv_c_arr)))]
    z_peak_z = np.array(z_bins_z)[np.argmin(np.abs(conv_z_arr - max(conv_z_arr)))]

    plt.plot(z_bins_c, 1000 * conv_c_arr, label='Even $\chi$', color=colours[0])
    plt.plot(z_peak_c, 1000 * max(conv_c_arr), marker='x', color=colours[0])
    plt.text((z_peak_z + z_peak_c) / 2, 1000 * max(conv_z_arr) * 3.5 / 5, f'$z$ = {round(z_peak_c, 2)}',
             fontsize=16, ha='center', color=colours[0])

    plt.plot(z_bins_z, 1000 * conv_z_arr, label='Even $z$', color=colours[1])
    plt.plot(z_peak_z, 1000 * max(conv_z_arr), marker='x', color=colours[1])
    plt.text((z_peak_z + z_peak_c) / 2, 1000 * max(conv_z_arr) * 3 / 5, f'$z$ = {round(z_peak_z, 2)}',
             fontsize=16, ha='center', color=colours[1])

    plt.xlabel("$z_L$")
    plt.ylabel("$\kappa$ ($\\times 10^{-3}$)")
    plt.legend(frameon=0, loc='upper right')

    # plt.axis([0, z_SN, 0, 1.1 * max(conv_c_arr)])
    plt.show()


def smoothed_m_convergence(tests, SN_dist, z_SN, OM=0.27, h=0.738):
    """Plots the convergence of a single mass confined to the centre of the LOS with decreasing bin width.

    Inputs:
     tests -- number of bin widths.
     SN_dist -- comoving distance to supernova.
     z_SN -- redshift of supernova.
     OM -- matter density parameter. Defaults to 0.27.
    """
    H0 = 1000 * 100 * h  # km/s/Gpc
    test_range = np.arange(3, tests, 2)
    conv = np.zeros(len(test_range))
    mass_mag = 15
    mass = MSOL * 10 ** mass_mag
    bin_lengths = np.zeros(len(test_range))
    for num, y in enumerate(test_range):
        (comoving_binwidths, comoving_bins, z_bins, z_widths) = create_chi_bins(0, z_SN, y + 1)
        cone_rad = comoving_bins[len(z_bins) // 2] * (1 + z_bins[len(z_bins) // 2]) * 0.00349066
        # distance * 12 arcmin = 0.00349066 rad
        vol_bin = (comoving_binwidths[0] * (1 + z_bins[len(z_bins) // 2])) * np.pi * cone_rad ** 2
        Hz = get_h_inv(z_bins[len(z_bins) // 2]) ** (-1) * H0
        d_m = 8 * np.pi * G * mass / (3 * OM * vol_bin * Hz ** 2 * 3.086E31) - 1
        conv[num] = single_d_convergence(comoving_binwidths, comoving_bins, z_bins, len(z_bins) // 2, mass, SN_dist)
        bin_lengths[num] = round(1000 * comoving_binwidths[0], 1)

    plt.plot(test_range[10::], conv[10::], label='$M_{{cluster}} = 10^{0} M_\odot$'.format({mass_mag}),
             color=colours[0])
    plt.plot(test_range[10::], np.zeros(len(test_range[10::])), color=[0.75, 0.75, 0.75], linestyle='--')
    plt.xticks(test_range[10::tests // 20], bin_lengths[10::tests // 20], rotation=45)
    plt.xlabel("Bin length (Mpc)")
    plt.ylabel("$\kappa$")
    plt.legend(frameon=0)
    plt.axis([15, 799, -0.002325, 0.0017])
    plt.show()


def distance_ratio(z_source):
    """Compares the `convergence' obtained from two equivalent forms of the equation.

    Inputs:
     z_source -- redshift of supernova.
    """
    _, chis, zs, _ = create_chi_bins(0, z_source, 1002)
    z_source = zs[-1]
    # z_arr = np.linspace(0, z_source, 1001)
    D_S = b_comoving(0, z_source, 1, 0, 1001)[-1] / (1+z_source)
    chi_S = b_comoving(0, z_source, 1, 0, 1001)[-1]
    D_L = np.array([(b_comoving(0, i, 1, 0, 1001)[-1] / (1+i)) for i in zs])
    chi_L = np.array([(b_comoving(0, i, 1, 0, 1001)[-1]) for i in zs])
    D_LS = np.array([((b_comoving(0, z_source, 1, 0, 1001)[-1] - b_comoving(0, i, 1, 0, 1001)[-1]) / (1+z_source)) for i in zs])
    D_ratio = D_L * D_LS / D_S
    chi_ratio = chi_L * (np.linspace(chi_S, chi_S, 1001) - chi_L) / np.linspace(chi_S, chi_S, 1001) * (1 + zs)
    D_A = comoving(zs) / (1 + zs)
    z_peak = np.array(zs)[np.argmin(np.abs(D_ratio - max(D_ratio)))]
    z_peak_chi = np.array(zs)[np.argmin(np.abs(chi_ratio - max(chi_ratio)))]
    plt.plot(zs, np.linspace(chi_S, chi_S, 1001), color=[0.75, 0.75, 0.75], linestyle='--', label='$\chi_S$')
    plt.plot(zs, chi_L, color=colours[0], label='$\chi_L$')
    plt.plot(zs, (np.linspace(chi_S, chi_S, 1001) - chi_L), color=colours[1], label='$\chi_{LS}$')
    # plt.plot(zs, D_ratio, color=colours[2], label='$D_LD_{LS}/D_S$')
    plt.plot(zs, chi_ratio, color=colours[2], label='$\chi_L\chi_{LS}/\chi_Sa_L$')
    plt.legend(frameon=0)
    # plt.plot(z_peak, max(D_ratio), marker='x', color=colours[2])
    # plt.text(z_peak, D_S / 4, f'$z$ = {round(z_peak, 2)}', fontsize=16, ha='center', color=colours[2])
    plt.plot(z_peak_chi, max(chi_ratio), marker='x', color=colours[2])
    plt.text(z_peak_chi, chi_S / 3.5, f'$z$ = {round(z_peak_chi, 4)}', fontsize=16, ha='center', color=colours[2])
    plt.xlabel('$z$')
    plt.ylabel('$D_A$ (Gpc)')
    plt.show()

    z_peak_D = np.array(zs)[np.argmin(np.abs(D_ratio - max(D_ratio)))]
    chi_peak2 = np.array(chis)[np.argmin(np.abs(chi_ratio - max(chi_ratio)))]
    plt.plot(zs, np.linspace(D_S, D_S, 1001), color=[0.75, 0.75, 0.75], linestyle='--', label='$D_S$')
    plt.plot(zs, D_L, color=colours[0], label='$D_L$')
    plt.plot(zs, D_LS, color=colours[1], label='$D_{LS}$')
    plt.plot(zs, D_ratio, color=colours[2], label='$D_LD_{LS}/D_S$')
    # plt.plot(chis, chi_ratio, color=colours[4], label='$\chi_L\chi_{LS}/\chi_Sa_L$')
    plt.legend(frameon=0)
    plt.plot(z_peak_D, max(D_ratio), marker='x', color=colours[2])
    plt.text(z_peak_D, D_S / 5, f'$\chi$ = {round(z_peak_D, 4)} Gpc', fontsize=16, ha='center', color=colours[2])
    # plt.plot(chi_peak2, max(chi_ratio), marker='x', color=colours[4])
    # plt.text(chi_peak2, chi_S / 2.5, f'$\chi$ = {round(chi_peak2, 2)} Gpc', fontsize=16, ha='center', color=colours[4])
    plt.xlabel('$z$')
    plt.ylabel('$D_A$ (Gpc)')
    plt.show()


if __name__ == "__main__":
    SN_redshift = 1.0
    num_bin = 50

    chi_to_SN = b_comoving(0, SN_redshift)
    # chi_to_SN = b_comoving(0, SN_redshift)
    SN_chi = chi_to_SN[-1]
    print("SN redshift", SN_redshift, "\nSN comoving distace", SN_chi)
    (comoving_binwidthsc, comoving_binsc, z_binsc, z_widthsc) = create_chi_bins(0, SN_redshift, num_bin)
    (comoving_binwidthsz, comoving_binsz, z_binsz, z_widthsz) = create_z_bins(0, SN_redshift, num_bin)

    single_conv_c = calc_single_d(comoving_binwidthsc, comoving_binsc, z_binsc, z_widthsc, SN_redshift)
    single_conv_z = calc_single_d(comoving_binwidthsz, comoving_binsz, z_binsz, z_widthsz, SN_redshift, use_chi=False)
    plot_smoothed_d(comoving_binwidthsz, comoving_binsz, z_binsz, SN_redshift)

    compare_z_chi(single_conv_c, single_conv_z, comoving_binsc, comoving_binsz, z_binsz, z_binsc, SN_chi, SN_redshift)

    num_test = 800
    # smoothed_m_convergence(num_test, SN_chi, SN_redshift)
    # distance_ratio(SN_redshift)

    scalefactor = np.linspace(1, 0.5, 101)
    rho_crit = scalefactor ** (-3)
    plt.plot(scalefactor, rho_crit, color=colours[0])
    plt.plot(scalefactor, (1 - rho_crit) / rho_crit, color=colours[1])
    plt.plot(scalefactor, 2 * rho_crit, color=colours[2])
    plt.ylim([-2, 20])
    plt.gca().invert_xaxis()
    plt.show()
