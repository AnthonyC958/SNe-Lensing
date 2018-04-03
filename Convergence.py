import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp

colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
h = 0.738
H0 = 1000 * 100 * h  # km/s/Gpc
# H0 = 100000  # in terms of h
c = 2.998E5  # km/s
OM = 0.27
OL = 0.73
G = 6.67E-11
MSOL = 1.989E30


def get_h_inv(z_val):
    """Integrand for calculating comoving distance.
    Assumes the lower bound on redshift is 0.

    Inputs:
     z_val -- upper redshift bound.
    """
    OK = 1.0 - OM - OL
    H = np.sqrt(OK * (1.0 + z_val) ** 2 + OM * (1.0 + z_val) ** 3 + OL)
    return 1.0 / H


def comoving(zs_array):
    h_invs = vecGet_h_inv(zs_array)
    comoving_coord = sp.cumtrapz(h_invs, x=zs_array, initial=0)

    dist = comoving_coord * c / H0

    return dist


def create_chi_bins(z_lo, z_hi, num_bins):
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

    # plt.plot(z_to_end, chi_to_end)
    # plt.plot(zs, chis, linestyle='', marker='o', markersize=3)
    # plt.plot([z_bin_edges, z_bin_edges], [chi_bin_edges[0], chi_bin_edges[-1]], color=[0.5, 0.5, 0.5],
    #          linestyle='--', linewidth=0.5)
    # plt.plot([z_lo, z_hi], [chi_bin_edges, chi_bin_edges], color=[0.5, 0.5, 0.5], linestyle='--', linewidth=0.5)
    # plt.xlabel('$z$')
    # plt.ylabel('$R_0\chi$')
    # plt.show()

    return chi_widths, chis, zs, z_widths


def create_z_bins(z_lo, z_hi, num_bins):
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

    # plt.plot(np.linspace(z_lo, z_hi, 1001), comoving(np.linspace(z_lo, z_hi, 1001)))
    # plt.plot(zs, chis, linestyle='', marker='o', markersize=3)
    # plt.plot([z_bin_edges, z_bin_edges], [chi_bin_edges[0], chi_bin_edges[-1]], color=[0.5, 0.5, 0.5],
    #          linestyle='--', linewidth=0.5)
    # plt.plot([z_lo, z_hi], [chi_bin_edges, chi_bin_edges], color=[0.5, 0.5, 0.5], linestyle='--', linewidth=0.5)
    # plt.xlabel('$z$')
    # plt.ylabel('$R_0\chi$')
    # plt.show()

    return chi_widths, chis, zs, z_widths


def single_m_convergence(chi_widths, chis, zs, index, density, SN_dist):
    """Calculates convergence from an overdensity in redshift bin i.

    Inputs:
     chi_widths -- the width of the comoving distance bins.
     chis -- the mean comoving distance of each bin.
     zs -- the mean redshift of each bin, for the scale factor.
     index -- which redshift bin will contain the over density.
     density -- the value of the overdensity. Corresponds to (observed-expected)/expected when galaxy counting (>= -1).
     SN_dist -- comoving distance to SN along line of sight.
    """
    coeff = 3.0 * H0 ** 2 * OM / (2.0 * c ** 2)
    d_arr = np.linspace(0, 0, len(zs))
    d_arr[index] = density
    sf_arr = 1.0 / (1.0 + zs)
    k_i = coeff * chis * chi_widths * (SN_dist - chis) / SN_dist * d_arr / sf_arr
    return np.sum(k_i)


def calc_single_m(chi_widths, chis, zs, z_SN):
    comoving_to_SN = comoving(np.linspace(0, z_SN, 1001))
    chi_SN = comoving_to_SN[-1]
    # print(chi_SN)

    convergence = np.linspace(0, 0, len(chis))
    delta = 1.0
    for i in range(0, len(chis)):
        convergence[i] = (single_m_convergence(chi_widths, chis, zs, i, delta, chi_SN))

    return convergence
    # # plt.plot(chis, convergence, label=f'$\delta$ = {delta}')
    # plt.plot(zs, convergence, label=f'$\delta$ = {delta}')
    # # plt.plot(chis[-1] - chis, convergence, label=f'$\delta$ = {delta}')
    # # plt.plot(zs[-1] - zs, convergence, label=f'$\delta$ = {delta}')
    # # plt.xlabel("Comoving Distance of Overdensity (Gpc)")
    # plt.xlabel("Redshift of Overdensity")
    # plt.ylabel("Convergence $\kappa$")
    # # plt.title(f"Convergence as a function of overdensity location for SN at $\chi$ = {np.round(chi_SN, 2)} Gpc")
    # plt.legend(frameon=0)
    # plt.show()


def plot_smoothed_m(chi_widths, chis, zs, z_SN, z_widths):
    """Creates an array of density arrays with progressively smoothed overdensity.
    Assumes the array of bins is odd."""
    comoving_to_SN = comoving(np.linspace(0, z_SN, 1001))
    chi_SN = comoving_to_SN[-1]

    size = 2 * len(zs)//2 + 1
    delta = np.zeros((size, len(zs)))

    delta1 = 10
    correction = delta1 / len(zs)
    delta[0][int(len(zs) // 2):int(len(zs) // 2) + 1] = delta1
    delta[-1][int(len(zs) // 2):int(len(zs) // 2) + 1] = -delta1

    for i, s in enumerate(np.arange(1, len(zs)//2 + 1, 1)):
        delta[s][int(len(zs) // 2) - s:int(len(zs) // 2) + s + 1] = delta1 / (2 * s + 1)
        delta[-s-1][int(len(zs) // 2) - s:int(len(zs) // 2) + s + 1] = -delta1 / (2 * s + 1)
    convergence = np.zeros(size)
    convergence_cor = np.zeros(size)

    for array in delta[0:len(delta)//2]:
        plt.bar(chis, array, width=chi_widths[0], alpha=0.5, edgecolor='k', color=[0.5, 0.5, 0.5])
    plt.xlabel("Comoving Distance (Gpc)")
    plt.ylabel("$\delta_i$")
    plt.show()

    delta_cor = np.zeros((size, len(zs)))
    delta_cor[0:size//2] = delta[0:size//2]-correction
    delta_cor[size//2:] = delta[size//2:]+correction

    for j in range(size):
        convergence[j] = (smoothed_m_convergence(chi_widths, chis, zs, delta[j], chi_SN))
        convergence_cor[j] = (smoothed_m_convergence(chi_widths, chis, zs, delta_cor[j], chi_SN))

    # convergence = np.delete(convergence, size // 2, 0)
    convergence_cor = np.delete(convergence_cor, size // 2, 0)

    plt.plot(range(size // 2), convergence[:size // 2], label=f'Total $|\delta|$ = 10', color=colours[0])
    plt.plot(range(size // 2 - 1, size - 1), convergence[size // 2:], color=colours[0])
    plt.plot([size // 2 - 1, size // 2 - 1], [convergence[0], convergence[-1]], color=[0.5, 0.5, 0.5],
             linestyle='--')
    plt.plot([0, size - 1], [0, 0], color=[0.5, 0.5, 0.5], linestyle='--')
    plt.plot(range(size - 1), convergence_cor, label=f'Total $|\delta|$ = 0', color=colours[1])
    plt.xlabel("Number of bins smoothed over")
    plt.ylabel("$\kappa$")
    # plt.title(f"Convergence as a function of central overdensity smoothing (z$_S$$_N$ = {z_SN})")
    plt.legend(frameon=0)
    plt.show()


def smoothed_m_convergence(chi_widths, chis, zs, d_arr, SN_dist):
    """Calculates convergence from an overdensity in redshift bin i.

    Inputs:
     matter_dp -- matter density parameter.
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


if __name__ == "__main__":
    vecGet_h_inv = np.vectorize(get_h_inv)

    SN_redshift = 0.5
    chi_to_SN = comoving(np.linspace(0, SN_redshift, 1001))
    SN_chi = chi_to_SN[-1]
    print(SN_redshift, SN_chi)
    (comoving_binwidthsc, comoving_binsc, z_binsc, z_widthsc) = create_chi_bins(0, SN_redshift, 36)
    (comoving_binwidthsz, comoving_binsz, z_binsz, z_widthsz) = create_z_bins(0, SN_redshift, 36)

    # single_conv_c = calc_single_m(comoving_binwidthsc, comoving_binsc, z_binsc, SN_redshift)
    # single_conv_z = calc_single_m(comoving_binwidthsz, comoving_binsz, z_binsz, SN_redshift)
    plot_smoothed_m(comoving_binwidthsc, comoving_binsc, z_binsc, SN_redshift, z_widthsc)

    # plt.plot(comoving_binsc, single_conv_c, label=f'Even $\chi$')
    # plt.plot(comoving_binsz, single_conv_z, label=f'Even z')
    # # plt.plot(comoving_binsc[-1] - comoving_binsc, single_conv_c, color=colours[0], alpha=0.5, linestyle='--')
    # # plt.plot(comoving_binsz[-1] - comoving_binsz, single_conv_z, color=colours[1], alpha=0.5, linestyle='--')
    # plt.xlabel("Comoving Distance of Overdensity (Gpc)")
    # plt.ylabel("Convergence $\kappa$")
    # plt.legend(frameon=0)
    # plt.show()
    #
    # plt.plot(z_binsc, single_conv_c, label=f'Even $\chi$')
    # plt.plot(z_binsz, single_conv_z, label=f'Even z')
    # # plt.plot(z_binsc[-1] - z_binsc, single_conv_c, color=colours[0], alpha=0.5, linestyle='--')
    # # plt.plot(z_binsz[-1] - z_binsz, single_conv_z, color=colours[1], alpha=0.5, linestyle='--')
    # plt.xlabel("Redshift of Overdensity")
    # plt.ylabel("Convergence $\kappa$")
    # plt.legend(frameon=0)
    # plt.show()

    num_test = 80
    test_range = np.arange(3, num_test, 2)
    conv = np.zeros(len(test_range))
    mass_mag = 15
    mass = MSOL * 10 ** mass_mag

    for num, y in enumerate(test_range):
        (comoving_binwidths, comoving_bins, z_bins, z_widthsz) = create_chi_bins(0, SN_redshift, y+1)
        vol_bin = (comoving_binwidths[0] * (1 + z_bins[len(z_bins) // 2]) * 3.086E22) ** 3  # Gpc^3 -> m^3
        # d_m = 2.0 * G * get_h_inv(z_bins[len(z_bins) // 2]) ** 2 / H0 ** 2 / \
        #      OM * mass / ((comoving_binwidths[0] / (2.0 * (1 + z_bins[len(z_bins) // 2]))) ** 3) - 1
        # d_m = mass / vol_bin / h ** 2 / OM / 1.88E-26 - 1
        d_m = mass / vol_bin * (8 * np.pi * G / 3 * (get_h_inv(z_bins[len(z_bins) // 2]) / H0 / 1000 * 3.068E22) ** 2)\
            - 1
        print(1/(8 * np.pi * G / 3 * (get_h_inv(z_bins[len(z_bins) // 2]) / H0 / 1000 * 3.068E22) ** 2))
        conv[num] = single_m_convergence(comoving_binwidths, comoving_bins, z_bins, len(z_bins) // 2, d_m, SN_chi)

    plt.plot(test_range, conv, label='$M_{{gal}} = 10^{0} M_\odot$'.format({mass_mag}))
    plt.plot(test_range, np.zeros(len(test_range)), color=[0.5, 0.5, 0.5], linestyle='--')
    plt.xlabel("Number of bins")
    plt.ylabel("Convergence $\kappa$")
    plt.legend(frameon=0)
    plt.show()
