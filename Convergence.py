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
    curvature_dp = 1.0 - OM - OL
    H = np.sqrt(curvature_dp * (1 + z_val) ** 2 + OM * (1 + z_val) ** 3 + OL)
    return 1. / H


def comoving(zs_array):
    h_invs = vecGet_h_inv(zs_array)
    comoving_coord = sp.cumtrapz(h_invs, x=zs_array, initial=0)

    dist = comoving_coord * c / H0

    return dist


def single_m_convergence(chi_widths, chis, zs, index, density, SN_dist):
    """Calculates convergence from an overdensity in redshfit bin i.

    Inputs:
     chi_widths -- the width of the comoving distance bins.
     chis -- the mean comoving distances of each bin.
     zs -- the mean redshift of each bin, for the scale factor.
     index -- which redshift bin will contain the over density.
     density -- the value of the overdensity. Corresponds to (observed-expected)/expected when galaxy counting (>= -1).
     SN_dist -- comoving distance to SN along line of sight.
    """
    coeff = 3 * H0 ** 2 * OM / (2 * c ** 2)
    d_arr = np.linspace(0, 0, len(zs))
    d_arr[index] = density
    sf_arr = 1 / (1 + zs)
    k_i = coeff * chis * chi_widths * (SN_dist - chis) / SN_dist * d_arr / sf_arr
    return np.sum(k_i)


def plot_single_m(chi_widths, chis, zs):
    SN_redshift = 1.6
    comoving_to_SN = comoving(np.linspace(0, SN_redshift, 1000))
    chi_SN = comoving_to_SN[-1]

    convergence = np.linspace(0, 0, len(z_bins))
    delta = 1
    for i in range(0, len(z_bins)):
        convergence[i] = (single_m_convergence(chi_widths, chis, zs, i, delta, chi_SN))

    plt.plot(comoving_bins, convergence, label=f'$\delta$ = {delta}')
    # plt.plot([1, 3, 5, 7, 9, 11, 13, 15], convergence, label=f'$\delta$ = 10')
    plt.xlabel("Comoving Distance of Overdensity (Gpc)")
    # plt.xlabel("Number of bins smoothed over")
    plt.ylabel("$\kappa$")
    plt.title("Convergence as a function of overdensity location for SN at $\chi$ = 4.42 Gpc")
    # plt.title("Convergence as a function of central overdensity smoothing (z$_{SN}$ = 1.5)")
    plt.legend()
    plt.show()


def plot_smoothed_m(chi_widths, chis, zs):
    SN_redshift = 1.5
    comoving_to_SN = comoving(np.linspace(0, SN_redshift, 1000))
    chi_SN = comoving_to_SN[-1]

    delta = [[0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 10 / 3, 10 / 3, 10 / 3, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 10 / 7, 10 / 7, 10 / 7, 10 / 7, 10 / 7, 10 / 7, 10 / 7, 0, 0, 0, 0],
             [0, 0, 0, 10 / 9, 10 / 9, 10 / 9, 10 / 9, 10 / 9, 10 / 9, 10 / 9, 10 / 9, 10 / 9, 0, 0, 0],
             [0, 0, 10 / 11, 10 / 11, 10 / 11, 10 / 11, 10 / 11, 10 / 11, 10 / 11, 10 / 11, 10 / 11, 10 / 11, 10 / 11,
              0, 0],
             [0, 10 / 13, 10 / 13, 10 / 13, 10 / 13, 10 / 13, 10 / 13, 10 / 13, 10 / 13, 10 / 13, 10 / 13, 10 / 13,
              10 / 13, 10 / 13, 0],
             [10 / 15, 10 / 15, 10 / 15, 10 / 15, 10 / 15, 10 / 15, 10 / 15, 10 / 15, 10 / 15, 10 / 15, 10 / 15,
              10 / 15, 10 / 15, 10 / 15, 10 / 15]]
    print(len(delta))
    convergence = np.linspace(0, 0, len(delta))

    for j in range(8):
        convergence[j] = (smoothed_m_convergence(chi_widths, chis, zs, delta[j], chi_SN))

    plt.plot([1, 3, 5, 7, 9, 11, 13, 15], convergence, label=f'$\delta$ = 10')
    plt.xlabel("Number of bins smoothed over")
    plt.ylabel("$\kappa$")
    plt.title("Convergence as a function of central overdensity smoothing (z$_{SN}$ = 1.5)")
    plt.legend()
    plt.show()


def smoothed_m_convergence(chi_widths, chis, zs, d_arr, SN_dist):
    """Calculates convergence from an overdensity in redshfit bin i.

    Inputs:
     matter_dp -- matter density parameter.
     chi_widths -- the width of the comoving distance bins.
     chis -- the mean comoving distances of each bin.
     zs -- the mean redshift of each bin, for the scale factor.
     d_arr -- the vector of overdensities.
     SN_dist -- comoving distance to SN along line of sight.
    """
    coeff = 3 * H0 ** 2 * OM / (2 * c ** 2)
    # d_arr = np.linspace(0, 0, len(zs))
    # d_arr[index] = density
    sf_arr = 1 / (1 + zs)
    k_i = coeff * chis * chi_widths * (SN_dist - chis) / SN_dist * d_arr / sf_arr
    return np.sum(k_i)


if __name__ == "__main__":
    vecGet_h_inv = np.vectorize(get_h_inv)

    z_lo = 0.1
    z_hi = 1.5
    z_nbin = 15
    z_values = np.linspace(0, z_hi, z_nbin * 2 + 1)
    z_bin_edges = z_values[0::2]
    z_bins = z_values[1::2]
    z_binwidth = z_bin_edges[1] - z_bin_edges[0]

    comoving_values = np.linspace(0, 0, z_nbin*2+1)
    for k in range(1, z_nbin*2+1):
        comoving_distances = comoving(np.linspace(0, z_values[k], 1000))
        comoving_values[k] = comoving_distances[-1]

    comoving_bin_edges = comoving_values[0::2]
    comoving_bins = comoving_values[1::2]
    comoving_binwidths = comoving_bin_edges[1:] - comoving_bin_edges[:-1]

    plot_single_m(comoving_binwidths, comoving_bins, z_bins)
    plot_smoothed_m(comoving_binwidths, comoving_bins, z_bins)
