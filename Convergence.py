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

    Inputs:
     z_val -- redshift value integrand is evaluated at.
    """
    OK = 1.0 - OM - OL
    H = np.sqrt(OK * (1.0 + z_val) ** 2 + OM * (1.0 + z_val) ** 3 + OL)
    return 1.0 / H


def get_adot_inv(a_val):
    """Integrand for calculating comoving distance.

    Inputs:
     a_val -- scalefactor value integrand is evaluated at.
    """
    OK = 1.0 - OM - OL
    adot = np.sqrt(OK + OM / a_val + OL * a_val ** 2)
    return 1.0 / adot


def comoving(zs_array):
    vecGet_h_inv = np.vectorize(get_h_inv)
    h_invs = vecGet_h_inv(zs_array)
    comoving_coord = sp.cumtrapz(h_invs, x=zs_array, initial=0)

    dist = comoving_coord * c / H0

    return dist


# def scalefactor(as_array):
#     adot_invs = vecGet_adot_inv(as_array)
#     time = sp.cumtrapz(adot_invs, x=as_array, initial=0)
#
#     return time
#
#
# def plot_scalefactor(z):
#     aarr = np.linspace(1/(z+1), 1, 1001)
#     zarr = 1 / aarr - 1
#     tarr = scalefactor(aarr)
#     mid_t = max(tarr) / 2
#     mid_a = aarr[np.argmin(np.abs(tarr - mid_t))]
#     mid_z = zarr[np.argmin(np.abs(tarr - mid_t))]
#
#     plt.plot(tarr, aarr)
#     plt.xlabel('t')
#     plt.ylabel('a')
#     plt.show()
#
#     plt.plot(tarr, zarr)
#     plt.xlabel('t')
#     plt.ylabel('z')
#     plt.show()
#
#     print("t/2 =", mid_t)
#     print("a(t/2) =", mid_a)
#     print("z(t/2) =", mid_z)
#
#
# def plot_comoving(z):
#     aarr = np.linspace(1 / (z + 1), 1, 1001)
#     zarr = 1 / aarr - 1
#     tarr = scalefactor(aarr)
#     carr = comoving(zarr[::-1])
#     mid_t = max(tarr) / 2
#     mid_a = aarr[np.argmin(np.abs(tarr - mid_t))]
#     mid_z = zarr[np.argmin(np.abs(tarr - mid_t))]
#     mid_c = carr[np.argmin(np.abs(tarr - mid_t))]
#
#     plt.plot(zarr, carr)
#     plt.xlabel('z')
#     plt.ylabel('$R_0\chi$')
#     plt.show()
#
#     plt.plot(aarr, carr)
#     plt.xlabel('a')
#     plt.ylabel('$R_0\chi$')
#     plt.show()
#
#     plt.plot(tarr, carr)
#     plt.xlabel('t')
#     plt.ylabel('$R_0\chi$')
#     plt.show()
#
#     print("t/2 =", mid_t)
#     print("a(t/2) =", mid_a)
#     print("z(t/2) =", mid_z)
#     print("$R_0\chi$(t/2) =", mid_c)


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
    # plt.plot([z_bin_edges, z_bin_edges], [chi_bin_edges[0], chi_bin_edges[-1]], color=[0.75, 0.75, 0.73],
    #          linestyle='-', linewidth=0.8)
    # plt.plot([z_lo, z_hi], [chi_bin_edges, chi_bin_edges], color=[0.75, 0.75, 0.75], linestyle='-', linewidth=0.8)
    # plt.xlabel('$z$', fontsize=16)
    # plt.tick_params(labelsize=12)
    # plt.axis([0, z_hi, 0, chi_end])
    # plt.ylabel('$R_0\chi$ (Gpc)', fontsize=16)
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
    # plt.plot([z_bin_edges, z_bin_edges], [chi_bin_edges[0], chi_bin_edges[-1]], color=[0.75, 0.75, 0.75],
    #          linestyle='-', linewidth=0.8)
    # plt.plot([z_lo, z_hi], [chi_bin_edges, chi_bin_edges], color=[0.75, 0.75, 0.75], linestyle='-', linewidth=0.8)
    # plt.xlabel('$z$', fontsize=16)
    # plt.tick_params(labelsize=12)
    # plt.ylabel('$R_0\chi$ (Gpc)', fontsize=16)
    # plt.axis([0, z_hi, 0, chis[-1]+chi_widths[-1]/2])
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


def single_m_conv_z(chi_widths, chis, zs, index, density, SN_dist):
    coeff = 3.0 * H0 ** 2 * OM / (2.0 * c ** 2)
    d_arr = np.linspace(0, 0, len(zs))
    d_arr[index] = density
    sf_arr = 1.0 / (1.0 + zs)
    k_i = coeff * chis * chi_widths * (c / H0 * get_h_inv(zs)) * (SN_dist - chis) / SN_dist * d_arr / sf_arr
    return np.sum(k_i)


def calc_single_m(chi_widths, chis, zs, z_SN):
    comoving_to_SN = comoving(np.linspace(0, z_SN, 1001))
    chi_SN = comoving_to_SN[-1]

    convergence = np.linspace(0, 0, len(chis))
    delta = 1.0
    for i in range(0, len(chis)):
        convergence[i] = (single_m_convergence(chi_widths, chis, zs, i, delta, chi_SN))

    return convergence


def calc_single_m_z(chi_widths, chis, zs, z_SN):
    comoving_to_SN = comoving(np.linspace(0, z_SN, 1001))
    chi_SN = comoving_to_SN[-1]

    convergence = np.linspace(0, 0, len(chis))
    delta = 1.0
    for i in range(0, len(chis)):
        convergence[i] = (single_m_conv_z(chi_widths, chis, zs, i, delta, chi_SN))

    return convergence


def plot_smoothed_m(chi_widths, chis, zs, z_SN, z_widths):
    """Creates an array of density arrays with progressively smoothed overdensity.
    Assumes the array of bins is odd."""
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

    # for array in delta[0:len(delta)//2]:
    #     plt.bar(chis, array, width=chi_widths[0], alpha=0.5, edgecolor='k', color=[0.5, 0.5, 0.5])
    # plt.xlabel("Comoving Distance (Gpc)")
    # plt.ylabel("$\delta_i$")
    # plt.show()

    delta_cor = np.zeros((size, len(zs)))
    delta_cor[0:size//2] = delta[0:size//2]-correction
    delta_cor[size//2:] = delta[size//2:]+correction

    for j in range(size):
        convergence[j] = (general_convergence(chi_widths, chis, zs, delta[j], chi_SN))
        convergence_cor[j] = (general_convergence(chi_widths, chis, zs, delta_cor[j], chi_SN))

    # convergence = np.delete(convergence, size // 2, 0)
    convergence_cor = np.delete(convergence_cor, size // 2, 0)

    plt.plot([size // 2 - 1, size // 2 - 1], [convergence[0], convergence[-1]], color=[0.75, 0.75, 0.75],
             linestyle='--')
    plt.plot([0, size - 1], [0, 0], color=[0.75, 0.75, 0.75], linestyle='--')
    plt.plot(range(size // 2), convergence[:size // 2], label=f'Total $|\delta|$ = 1', color=colours[0])
    plt.plot(range(size // 2 - 1, size - 1), convergence[size // 2:], color=colours[0])
    plt.plot(range(size - 1), convergence_cor, label=f'Total $|\delta|$ = 0', color=colours[1])
    plt.xlabel("Number of bins smoothed over", fontsize=16)
    plt.ylabel("$\kappa$", fontsize=16)
    plt.tick_params(labelsize=12)
    # plt.title(f"Convergence as a function of central overdensity smoothing (z$_S$$_N$ = {z_SN})")
    plt.legend(frameon=0, fontsize=12)
    plt.xticks([0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100], [0, 25, 50, 75, 100, 75, 50, 25, 0])
    plt.axis([0, size, min(convergence)-0.0003, max(convergence)+0.0003])

    plt.show()


def general_convergence(chi_widths, chis, zs, d_arr, SN_dist):
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


def convergence_error(chi_widths, chis, zs, expected_arr, SN_dist):
    coeff = 3.0 * H0 ** 2 * OM / (2.0 * c ** 2)
    sf_arr = 1.0 / (1.0 + zs)
    k_i = coeff * chis * chi_widths * (SN_dist - chis) / SN_dist / sf_arr / expected_arr
    return np.sum(k_i)


if __name__ == "__main__":
    vecGet_adot_inv = np.vectorize(get_adot_inv)

    SN_redshift = 1.0

    # plot_scalefactor(SN_redshift)
    # plot_comoving(SN_redshift)

    chi_to_SN = comoving(np.linspace(0, SN_redshift, 501))
    SN_chi = chi_to_SN[-1]
    print("SN redshift", SN_redshift, "\nSN comoving distace", SN_chi)
    (comoving_binwidthsc, comoving_binsc, z_binsc, z_widthsc) = create_chi_bins(0, SN_redshift, 100)
    (comoving_binwidthsz, comoving_binsz, z_binsz, z_widthsz) = create_z_bins(0, SN_redshift, 100)

    single_conv_c = calc_single_m(comoving_binwidthsc, comoving_binsc, z_binsc, SN_redshift)
    single_conv_z = calc_single_m(comoving_binwidthsz, comoving_binsz, z_binsz, SN_redshift)
    plot_smoothed_m(comoving_binwidthsc, comoving_binsc, z_binsc, SN_redshift, z_widthsc)

    plt.plot([SN_chi / 2, SN_chi / 2], [0, 1.1 * max(single_conv_c)], linestyle='--', color=[0.75, 0.75, 0.75],
             linewidth=1)
    plt.plot(comoving_binsc, single_conv_c, label='Even $\chi$')
    plt.plot(comoving_binsz, single_conv_z / (c / H0 * get_h_inv(z_binsz)), label='Even z')
    plt.xlabel("$\chi_{Overdensity}$ (Gpc)", fontsize=16)
    plt.ylabel("$\kappa$", fontsize=16)
    plt.tick_params(labelsize=12)
    plt.legend(frameon=0, fontsize=12)
    plt.axis([0, SN_chi, 0, 1.1 * max(single_conv_c)])
    plt.show()

    plt.plot([SN_redshift / 2, SN_redshift / 2], [0, 1.1 * max(single_conv_c)], linestyle='--',
             color=[0.75, 0.75, 0.75], linewidth=1)
    plt.plot(z_binsc, single_conv_c, label='Even $\chi$')
    print("Peak at z =", z_binsc[np.argmin(np.abs(single_conv_c - max(single_conv_c)))])
    plt.plot(z_binsz, single_conv_z / (c / H0 * get_h_inv(z_binsz)), label='Even z')
    plt.xlabel("$z_{Overdensity}$", fontsize=16)
    plt.ylabel("$\kappa$", fontsize=16)
    plt.legend(frameon=0, fontsize=12)
    plt.tick_params(labelsize=12)
    plt.axis([0, SN_redshift, 0, 1.1 * max(single_conv_c)])
    plt.show()

    num_test = 800
    test_range = np.arange(3, num_test, 2)
    # test_range = 3*(np.arange(1, num_test))
    conv = np.zeros(len(test_range))
    mass_mag = 15
    mass = MSOL * 10 ** mass_mag
    bin_lengths = np.zeros(len(test_range))
    stop_num_bins = 0
    stop_index = 0
    # cluster_size = 0.0  # Mpc
    d_final = 0

    for num, y in enumerate(test_range):
        (comoving_binwidths, comoving_bins, z_bins, z_widths) = create_chi_bins(0, SN_redshift, y+1)
        cone_rad = comoving_bins[len(z_bins) // 2] * (1 + z_bins[len(z_bins) // 2]) * 0.00349066
        # distance * 12 arcmin = 0.00349066 rad
        vol_bin = (comoving_binwidths[0] * (1 + z_bins[len(z_bins) // 2])) * np.pi * cone_rad ** 2
        Hz = get_h_inv(z_bins[len(z_bins) // 2]) ** (-1) * H0
        d_m = 8 * np.pi * G * mass / (3 * OM * vol_bin * Hz ** 2 * 3.086E31) - 1
        conv[num] = single_m_convergence(comoving_binwidths, comoving_bins, z_bins, len(z_bins) // 2, d_m, SN_chi)
        bin_lengths[num] = round(1000*comoving_binwidths[0], 1)
        # print(d_m)
    #     if bin_lengths[num] <= cluster_size:
    #         stop_index += num
    #         stop_num_bins += y
    #         break
    #
    # if stop_index > 0:
    #     for new_num, y in enumerate(test_range[stop_index+1::]):
    #         num = new_num + stop_index + 1
    #         (comoving_binwidths, comoving_bins, z_bins, z_widths) = create_chi_bins(0, SN_redshift, y+1)
    #         cone_rad = comoving_bins[len(z_bins) // 2] * (1 + z_bins[len(z_bins) // 2]) * 0.00349066 / 2
    #         # distance * 12 arcmin / 2
    #         vol_bin = (comoving_binwidths[0] * (1 + z_bins[len(z_bins) // 2])) * np.pi * cone_rad ** 2
    #         Hz = get_h_inv(z_bins[len(z_bins) // 2]) ** (-1) * H0
    #         d_m = 8 * np.pi * G * mass / (new_num + 2) / (3 * OM * vol_bin * Hz ** 2 * 3.086E31) - 1
    #         d_arr = np.zeros(y)
    #         if new_num % 2 == 0:
    #             pos = (new_num + 2) // 2
    #             d_arr[len(z_bins) // 2 - pos:len(z_bins) // 2 + pos] = d_m
    #         else:
    #             pos = (new_num + 1) // 2
    #             d_arr[len(z_bins) // 2 - pos:len(z_bins) // 2 + pos + 1] = d_m
    #         # print(sum(d_arr), d_m)
    #         conv[num] = smoothed_m_convergence(comoving_binwidths, comoving_bins, z_bins, d_arr, SN_chi)
    #         bin_lengths[num] = round(1000*comoving_binwidths[0], 1)

    # size_num = np.argmin(np.abs(bin_lengths - cluster_size))
    plt.plot(test_range[10::], conv[10::], label='$M_{{cluster}} = 10^{0} M_\odot$'.format({mass_mag}))
    plt.plot(test_range[10::], np.zeros(len(test_range[10::])), color=[0.75, 0.75, 0.75], linestyle='--')
    # plt.plot([test_range[size_num], test_range[size_num]], [conv[10], -conv[12]], color=[0.5, 0.5, 0.5], linestyle='--')
    plt.xticks(test_range[10::num_test//20], bin_lengths[10::num_test//20], rotation=45)
    plt.xlabel("Bin length (Mpc)", fontsize=16)
    plt.ylabel("$\kappa$", fontsize=16)
    plt.tick_params(labelsize=12)
    plt.legend(frameon=0, fontsize=12)
    plt.axis([18, 799, -0.002325, 0.0017])
    plt.show()
