import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp

colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
H0 = 73800  # km/s/Gpc assuming h = 0.738
# H0 = 100000  # in terms of h
c = 2.998e5  # km/s
OM = 0.27
OL = 0.73


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
    zs = z_values[1::2]

    # plt.plot(z_to_end, chi_to_end)
    # plt.plot(zs, chis, linestyle='', marker='o', markersize=3)
    # plt.plot([z_bin_edges, z_bin_edges], [chi_bin_edges[0], chi_bin_edges[-1]], color=[0.5, 0.5, 0.5],
    #          linestyle='--', linewidth=0.5)
    # plt.plot([z_lo, z_hi], [chi_bin_edges, chi_bin_edges], color=[0.5, 0.5, 0.5], linestyle='--', linewidth=0.5)
    # plt.xlabel('$z$')
    # plt.ylabel('$R_0\chi$')
    # plt.show()

    return chi_widths, chis, zs


def create_z_bins(z_lo, z_hi, num_bins):
    z_values = np.linspace(z_lo, z_hi, num_bins * 2 - 1)
    zs = z_values[1::2]

    chi_values = np.linspace(0, 0, len(z_values))
    for k in range(len(z_values)):
        chi = comoving(np.linspace(z_lo, z_values[k], 1001))
        chi_values[k] = chi[-1]

    chi_bin_edges = chi_values[0::2]
    chis = chi_values[1::2]
    chi_widths = chi_bin_edges[1:] - chi_bin_edges[:-1]

    # plt.plot(z_to_end, chi_to_end)
    # plt.plot(zs, chis, linestyle='', marker='o', markersize=3)
    # plt.plot([z_bin_edges, z_bin_edges], [chi_bin_edges[0], chi_bin_edges[-1]], color=[0.5, 0.5, 0.5],
    #          linestyle='--', linewidth=0.5)
    # plt.plot([z_lo, z_hi], [chi_bin_edges, chi_bin_edges], color=[0.5, 0.5, 0.5], linestyle='--', linewidth=0.5)
    # plt.xlabel('$z$')
    # plt.ylabel('$R_0\chi$')
    # plt.show()

    return chi_widths, chis, zs


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


def plot_single_m(chi_widths, chis, zs, z_SN):
    comoving_to_SN = comoving(np.linspace(0, z_SN, 1001))
    chi_SN = comoving_to_SN[-1]
    # print(chi_SN)

    convergence = np.linspace(0, 0, len(chis))
    delta = 1.0
    for i in range(0, len(chis)):
        convergence[i] = (single_m_convergence(chi_widths, chis, zs, i, delta, chi_SN))

    # plt.plot(chis, convergence, label=f'$\delta$ = {delta}')
    plt.plot(zs, convergence, label=f'$\delta$ = {delta}')
    # plt.plot(chis[-1] - chis, convergence, label=f'$\delta$ = {delta}')
    # plt.plot(zs[-1] - zs, convergence, label=f'$\delta$ = {delta}')
    # plt.xlabel("Comoving Distance of Overdensity (Gpc)")
    plt.xlabel("Redshift of Overdensity")
    plt.ylabel("Convergence $\kappa$")
    # plt.title(f"Convergence as a function of overdensity location for SN at $\chi$ = {np.round(chi_SN, 2)} Gpc")
    plt.legend(frameon=0)
    plt.show()


def plot_smoothed_m(chi_widths, chis, zs, z_SN):
    """Creates an array of density arrays with progressively smoothed overdensity.
    Assumes the array of bins is odd."""
    comoving_to_SN = comoving(np.linspace(0, z_SN, 1001))
    chi_SN = comoving_to_SN[-1]

    delta = np.zeros((len(zs)//2 + 1, len(zs)))

    delta1 = 10
    correction = delta1 / len(zs)

    for s in np.arange(0, len(zs)//2 + 1, 1):
        delta[s][int(len(zs)//2)-s:int(len(zs)//2)+s+1] = delta1/(2*s+1)

    convergence = np.zeros(len(zs)//2 + 1)
    convergence_cor = np.zeros(len(zs) // 2 + 1)

    for array in delta:
        plt.bar(chis, array, width=chi_widths[0], alpha=0.5, edgecolor='k', color=[0.5, 0.5, 0.5])
    plt.xlabel("Comoving Distance (Gpc)")
    plt.ylabel("$\delta_i$")
    plt.show()

    for j in range(len(zs)//2 + 1):
        convergence[j] = (smoothed_m_convergence(chi_widths, chis, zs, delta[j], chi_SN))
        convergence_cor[j] = (smoothed_m_convergence(chi_widths, chis, zs, delta[j]-correction, chi_SN))

    plt.plot(range(len(zs) // 2 + 1), convergence, label=f'Total $\delta$ = 10')
    plt.plot(range(len(zs) // 2 + 1), convergence_cor, label=f'Total $\delta$ = 0')
    plt.xlabel("Number of bins smoothed over")
    plt.ylabel("$\kappa$")
    plt.title(f"Convergence as a function of central overdensity smoothing (z$_S$$_N$ = {z_SN})")
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

    SN_redshift = 2.0
    chi_to_SN = comoving(np.linspace(0, SN_redshift, 1001))
    SN_chi = chi_to_SN[-1]
    (comoving_binwidthsc, comoving_binsc, z_binsc) = create_chi_bins(0, SN_redshift, 36)
    # (comoving_binwidthsz, comoving_binsz, z_binsz) = create_z_bins(0, SN_redshift, 36)
    # plot_single_m(comoving_binwidthsz, comoving_binsz, z_binsz, SN_redshift)
    plot_smoothed_m(comoving_binwidthsc, comoving_binsc, z_binsc, SN_redshift)

    num_test = 80
    test_range = np.arange(3, num_test, 2)
    conv = np.zeros(len(test_range))

    for num, y in enumerate(test_range):
        (comoving_binwidths, comoving_bins, z_bins) = create_chi_bins(0, SN_redshift, y+1)
        conv[num] = single_m_convergence(comoving_binwidths, comoving_bins, z_bins, len(z_bins)//2, 1, SN_chi)

    plt.plot(test_range, conv, label=f'$\delta$ = 1.0')
    plt.xlabel("Number of bins")
    plt.ylabel("Convergence $\kappa$")
    plt.legend(frameon=0)
    plt.show()
