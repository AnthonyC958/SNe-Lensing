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


def create_bins(z_lo, z_hi, num_bins):
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
    zs = z_values[1::2]

    plt.plot(z_to_end, chi_to_end)
    plt.plot(zs, chis, linestyle='', marker='o', markersize=4)
    plt.plot([z_bin_edges, z_bin_edges], [chi_bin_edges[0], chi_bin_edges[-1]], color=[0.5, 0.5, 0.5],
             linestyle='--')
    plt.plot([z_lo, z_hi], [chi_bin_edges, chi_bin_edges], color=[0.5, 0.5, 0.5], linestyle='--')
    plt.show()

    return chi_widths, chis, zs


def single_m_convergence(chi_widths, chis, zs, index, density, SN_dist):
    """Calculates convergence from an overdensity in redshfit bin i.

    Inputs:
     chi_widths -- the width of the comoving distance bins.
     chis -- the mean comoving distance of each bin.
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


def plot_single_m(chi_widths, chis, zs, SN_redshift):
    comoving_to_SN = comoving(np.linspace(0, SN_redshift, 1001))
    chi_SN = comoving_to_SN[-1]

    convergence = np.linspace(0, 0, len(chis))
    delta = 1
    for i in range(0, len(chis)):
        convergence[i] = (single_m_convergence(chi_widths, chis, zs, i, delta, chi_SN))

    plt.plot(chis, convergence, label=f'$\delta$ = {delta}')
    plt.xlabel("Comoving Distance of Overdensity (Gpc)")
    plt.ylabel("$\kappa$")
    plt.title(f"Convergence as a function of overdensity location for SN at $\chi$ = {np.round(chi_SN, 2)} Gpc")
    plt.legend(frameon=0)
    plt.show()


def plot_smoothed_m(chi_widths, chis, zs, SN_redshift):
    comoving_to_SN = comoving(np.linspace(0, SN_redshift, 1000))
    chi_SN = comoving_to_SN[-1]

    delta = np.arange((2 * len(chis) + 1) * len(chis), dtype=np.float64).reshape(2 * len(chis) + 1, len(chis))
    delta_range = np.linspace(-3, 3, len(chis))

    for m, i in enumerate(np.linspace(0.1, 1, 2 * len(chis))):
        delta[m][:] = 1 / (i * np.sqrt(2 * np.pi)) * np.exp(-delta_range ** 2 / (2 * i ** 2))
        plt.plot(delta_range, delta[m][:])
    delta[-1][:] = 1/9
    plt.plot(delta_range, delta[-1][:])
    plt.show()

    convergence = np.zeros(((2 * len(chis) + 1), 1), dtype=np.float64)

    for j in range(2 * len(chis) + 1):
        convergence[j] = (smoothed_m_convergence(chi_widths, chis, zs, delta[j], chi_SN))

    plt.plot(range(2 * len(chis)), convergence[0:-1], label=f'$\delta$ = 10')
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

    z_SN = 10
    (comoving_binwidths, comoving_bins, z_bins) = create_bins(0, z_SN, 50)
    plot_single_m(comoving_binwidths, comoving_bins, z_bins, z_SN)
    plot_smoothed_m(comoving_binwidths, comoving_bins, z_bins, z_SN)
