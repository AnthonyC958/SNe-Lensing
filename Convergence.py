import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sp

colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
H0 = 73800  # km/s/Gpc assuming h = 0.738
# H0 = 100000  # in terms of h
c = 2.998e5  # km/s


def get_h_inv(z_val, matter_dp, lambda_dp):
    """Integrand for calculating comoving distance.
    Assumes the lower bound on redshift is 0.

    Inputs:
     z_val -- upper redshift bound.
     matter_dp -- matter density parameter.
     lambda_dp -- dark energy density parameter.
    """
    curvature_dp = 1.0 - matter_dp - lambda_dp
    H = np.sqrt(curvature_dp * (1 + z_val) ** 2 + matter_dp * (1 + z_val) ** 3 + lambda_dp)
    return 1. / H


def comoving(matter_dp, lambda_dp, zs_array):
    h_invs = vecGet_h_inv(zs_array, matter_dp, lambda_dp)
    comoving_coord = sp.cumtrapz(h_invs, x=zs_array, initial=0)

    dist = comoving_coord * c / H0

    return dist


def single_m_convergence(matter_dp, chi_widths, chis, zs, index, density, SN_dist):
    """Calculates convergence from an overdensity in redshfit bin i.

    Inputs:
     matter_dp -- matter density parameter.
     chi_widths -- the width of the comoving distance bins.
     chis -- the mean comoving distances of each bin.
     zs -- the mean redshift of each bin, for the scale factor.
     index -- which redshift bin will contain the over density.
     density -- the value of the overdensity. Corresponds to (observed-expected)/expected when galaxy counting (>= -1).
     SN_dist -- comoving distance to SN along line of sight.
    """
    coeff = 3 * H0 ** 2 * matter_dp / (2 * c ** 2)
    d_arr = np.linspace(0, 0, len(zs))
    d_arr[index] = density
    sf_arr = 1 / (1 + zs)
    k_i = coeff * chis * chi_widths * (SN_dist - chis) / SN_dist * d_arr / sf_arr
    return np.sum(k_i)


if __name__ == "__main__":
    vecGet_h_inv = np.vectorize(get_h_inv)

    om = 0.27
    ol = 0.73

    z_lo = 0.1
    z_hi = 1.5
    z_nstep = 1001
    z_arr = np.linspace(0, z_hi, z_nstep)

    z_nbin = 15
    z_values = np.linspace(0, z_hi, z_nbin * 2 + 1)
    z_bin_edges = z_values[0::2]
    z_bins = z_values[1::2]
    z_binwidth = z_bin_edges[1] - z_bin_edges[0]

    comoving_values = np.linspace(0, 0, z_nbin*2+1)
    for k in range(1, z_nbin*2+1):
        comoving_distances = comoving(om, ol, np.linspace(0, z_values[k], 1000))
        comoving_values[k] = comoving_distances[-1]

    comoving_bin_edges = comoving_values[0::2]
    comoving_bins = comoving_values[1::2]
    comoving_binwidths = comoving_bin_edges[1:] - comoving_bin_edges[:-1]

    # plt.plot(z_arr, comoving(om, ol, z_arr))
    # plt.plot(z_bins, comoving_bins, linestyle='', marker='o', markersize=4)
    # plt.plot([z_bin_edges, z_bin_edges], [0, 4.24], color=[0.5, 0.5, 0.5], linestyle='--')
    # plt.plot([0, 1.5], [comoving_bin_edges, comoving_bin_edges], color=[0.5, 0.5, 0.5], linestyle='--')
    # plt.show()

    SN_redshift = 1.6
    comoving_to_SN = comoving(om, ol, np.linspace(0, SN_redshift, 1000))
    chi_SN = comoving_to_SN[-1]
    print(chi_SN)
    convergence = np.linspace(0, 0, len(z_bins))
    delta = 1
    for i in range(0, len(z_bins)):
        convergence[i] = (single_m_convergence(om, comoving_binwidths, comoving_bins, z_bins, i, delta, chi_SN))

    plt.plot(comoving_bins, convergence, label=f'$\delta$ = {delta}')
    plt.xlabel("Comoving Distance of Overdensity (Gpc)")
    plt.ylabel("$\kappa$")
    plt.title("Convergence as a function of overdensity location for SN at $\chi$ = 4.42 Gpc")
    plt.legend()
    plt.show()
