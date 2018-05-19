from cones import *
import random

colours = [[0, 165/255, 124/255], [253/255, 170/255, 0], 'C2', 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5']
blue = [23/255, 114/255, 183/255, 0.75]
orange = [255/255, 119/255, 15/255, 0.75]
green = [0, 165/255, 124/255, 0.75]
yellow = [253/255, 170/255, 0, 0.75]
grey = [0.75, 0.75, 0.75]

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


def find_convergence(gal_chi, gal_z, limits, exp, chi_widths, chi_bins, z_bins, gal_kappa, z_all, redo=False):
    if redo:
        counts = {}
        num = 0
        for z in gal_z:
            bin_c = range(int(np.argmin(np.abs(limits - z))))
            counts[f"g{num}"] = np.zeros(len(bin_c))
            for num2 in bin_c:
                counts[f"g{num}"][num2] = sum([limits[num2] < z_all[i] <= limits[num2 + 1] for i in range(len(z_all))])
            num += 1
            if num % 5 == 0:
                print(f"Finished {num}/{len(gal_z)}")
        convergence = np.zeros(len(counts))
        conv_err = np.zeros(len(counts))
        num = 0
        d_arr = {}
        for key, SN in counts.items():
            d_arr[f"{key}"] = (SN - exp[:len(SN)]) / exp[:(len(SN))]
            convergence[num] = general_convergence(chi_widths[:len(SN)], chi_bins[:len(SN)], z_bins[:len(SN)],
                                                   d_arr[f"{key}"], gal_chi[num])
            conv_err[num] = convergence_error(chi_widths[:len(SN)], chi_bins[:len(SN)], z_bins[:len(SN)],
                                              exp[:len(SN)], gal_chi[num])
            num += 1

        conv_err += 0
        pickle_out = open("conv_sim.pickle", "wb")
        pickle.dump([convergence, conv_err], pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("conv_sim.pickle", "rb")
        conv_sim = pickle.load(pickle_in)
        convergence = conv_sim[0]
        conv_err = conv_sim[1]

    ax = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    ax.set_ylabel("$\kappa$")
    ax.set_xlabel("$z$")
    ax2.set_xlabel("Count")
    ax.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    ax2.set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.plot([0, 1.5], [0, 0], color=grey, linestyle='--')
    ax.axis([0, 1.5, -0.04, 0.14])
    ax2.axis([0, 270, -0.04, 0.14])
    ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0])
    ax.set_xticklabels([0, 0.2, 0.4, 0])
    ax.plot(gal_z, convergence, linestyle='', marker='o', markersize=2, color=colours[0], label="Cone Method")
    ax.plot(gal_z, gal_kappa, linestyle='', marker='o', markersize=2, color=colours[1], label="Actual Value")
    ax2.hist(convergence, bins=np.arange(-0.04, 0.14 + 0.025/4, 0.025/4), orientation='horizontal',
             fc=green, edgecolor=colours[0])
    ax2.hist(gal_kappa, bins=np.arange(-0.04, 0.14 + 0.025/4, 0.025/4), orientation='horizontal',
             fc=yellow, edgecolor=colours[1])
    ax.legend(frameon=0)
    plt.show()

    return convergence


if __name__ == "__main__":
    with fits.open('MICEsim.fits')as hdul1:
        # print(repr(hdul1[1].header))
        # print(repr(hdul1[1].data['dec']))

        max_z = max(hdul1[1].data['z'])
        redo = False
        bins = 100
        chi_bin_widths, chi_bins, z_bins, z_bin_widths = create_chi_bins(0, max_z, bins)
        lims = np.cumsum(z_bin_widths)
        RAs = hdul1[1].data['ra']
        DECs = hdul1[1].data['dec']
        zs = hdul1[1].data['z']
        kappas = hdul1[1].data['kappa']
        MICEdata = [RAs, DECs, zs, kappas]

        loc = [(min(RAs) + max(RAs)) / 2, (min(DECs) + max(DECs)) / 2]
        counter = 0
        if redo:
            big_cone = {'zs': [], 'kaps': []}
            for RA, DEC, z, kap in zip(RAs, DECs, zs, kappas):
                if (RA - loc[0]) ** 2 + (DEC - loc[1]) ** 2 <= 3.0 ** 2 and z >= 0.01:
                    big_cone['zs'].append(z)
                    big_cone['kaps'].append(kap)
                    counter += 1
                if counter % 10000 == 0:
                    print(f"Found {counter}/{len(RAs)}")

            little_cone = {'zs': [], 'kaps': []}
            counter = 0
            for RA, DEC, z, kap in zip(RAs, DECs, zs, kappas):
                if (RA - loc[0]) ** 2 + (DEC - loc[1]) ** 2 <= 0.2 ** 2 and z >= 0.01:
                    little_cone['zs'].append(z)
                    little_cone['kaps'].append(kap)
                    counter += 1
                if counter % 10000 == 0:
                    print(f"Found {counter}/{len(RAs)}")

            pickle_out = open("big_cone.pickle", "wb")
            pickle.dump([big_cone, little_cone], pickle_out)
            pickle_out.close()
        else:
            pickle_in = open("big_cone.pickle", "rb")
            cones = pickle.load(pickle_in)
            big_cone = cones[0]
            little_cone = cones[1]

        if redo:
            MICEexpected = []
            num = 0
            for num1, lim in enumerate(lims):
                MICEexpected.append(sum([big_cone['zs'][i] < lim for i in range(len(big_cone['zs']))]))
                num += 1
                print(f"Sorted {num}/{len(lims)}")

            pickle_out = open("MICEexpected.pickle", "wb")
            pickle.dump(MICEexpected, pickle_out)
            pickle_out.close()
        else:
            pickle_in = open("MICEexpected.pickle", "rb")
            MICEexpected = pickle.load(pickle_in)

        expected_big = np.diff([MICEexpected[i] for i in range(len(lims))])
        expected = [expected_big[i] * (0.2 / 3.0) ** 2 for i in range(len(expected_big))]
        plt.plot([0, 5], [0, 0], color=grey, linestyle='--')
        plt.plot(lims[1:], expected, marker='o', markersize=2.5, color=colours[0])
        plt.xlabel('$z$')
        plt.ylabel('Expected Count')
        plt.xlim([0, 1.5])
        plt.show()

        if redo:
            random.seed(1337)
            rand_samp_size = 1000
            sample = random.sample(little_cone['zs'], rand_samp_size)
            indices = [np.argmin(np.abs(little_cone['zs'] - sample[i])) for i in range(rand_samp_size)]
            kaps = [little_cone['kaps'][indices[i]] for i in range(rand_samp_size)]
            random_gals = [sample, kaps]
            pickle_out = open("random_gals.pickle", "wb")
            pickle.dump(random_gals, pickle_out)
            pickle_out.close()
        else:
            pickle_in = open("random_gals.pickle", "rb")
            random_gals = pickle.load(pickle_in)
            sample = random_gals[0]
            kaps = random_gals[1]

        ds = []
        chis = []
        for z in sample:
            chi_to_z = comoving(np.linspace(0, z, 1000), 0.25, 0.75)
            ds.append(chi_to_z[-1] * (1 + z))
            chis.append(chi_to_z[-1])

        mu = 5 * np.log10(np.array(ds) / 10 * 1E9)

        # plot_Hubble(sample, mu, np.zeros(len(sample)), OM=0.25, OL=0.75, max_x=1.5)
        conv = find_convergence(chis, sample, lims, expected, chi_bin_widths, chi_bins, z_bins, kaps, little_cone['zs'],
                                redo=False)
