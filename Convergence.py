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


def single_m_convergence(matter_dp, chi_widths, chis, zs, index, density):
    """Calculates convergence from an overdensity in redshfit bin i.

    Inputs:
     matter_dp -- matter density parameter.
     index -- which redshift bin will contain the over density.
     density -- the value of the overdensity. Any rational number. Corresponds to (observed-expected)/expected when
                galaxy counting.
    """
    coeff = 3 * H0 ** 2 * matter_dp / (2 * c ** 2)
    chi_SN = 1.6


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
    print(comoving_values, comoving_bin_edges, comoving_binwidths, comoving_bins)

    plt.plot(z_arr, comoving(om, ol, z_arr))
    plt.plot(z_values, comoving_values, linestyle='', marker='o')
    plt.show()
