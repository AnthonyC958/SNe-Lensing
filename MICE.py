from cones import *
import random
import csv

colours = [[0, 150/255, 100/255], [253/255, 170/255, 0], 'C2', 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5']
blue = [23/255, 114/255, 183/255, 0.75]
orange = [255/255, 119/255, 15/255, 0.75]
green = [0, 150/255, 100/255, 0.75]
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


def find_convergence(gal_chi, gal_z, limits, exp, chi_widths, chi_bins, z_bins, gal_kappa, data, redo=False):
    if redo:
        counts = {}
        num = 0
        for z in gal_z:
            bin_c = range(int(np.argmin(np.abs(limits - z))))
            counts[f"g{num}"] = np.zeros(len(bin_c))
            for num2 in bin_c:
                counts[f"g{num}"][num2] = sum([limits[num2] < data[f'SN{num+1}']['zs'][i] <= limits[num2 + 1]
                                               for i in range(len(data[f'SN{num+1}']['zs']))])
            if sum(counts[f"g{num}"]) == 0:
                print(f'Total void for {data[f"SN{num+1}"]}')
            num += 1
            if num % 5 == 0:
                print(f"Finished {num}/{len(gal_z)}")
        convergence = np.zeros(len(counts))
        conv_err = np.zeros(len(counts))
        num = 0
        d_arr = {}
        for key, SN in counts.items():
            d_arr[key] = (SN - exp[:len(SN)]) / exp[:(len(SN))]
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
    ax.plot([0, 1.5], [0, 0], color=grey, linestyle='--')
    ax.axis([0, 1.5, -0.06, 0.12])
    ax2.axis([0, 330, -0.06, 0.12])
    # ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0])
    ax.set_xticklabels([0, 0.5, 1.0, 0])
    ax.plot(gal_z, convergence, linestyle='', marker='o', markersize=2, color=colours[0], label="Cone Method")
    ax.plot(gal_z, gal_kappa, linestyle='', marker='o', markersize=2, color=colours[1], label="Actual Value")
    ax2.hist(convergence, bins=np.arange(-0.12, 0.13 + 0.025/4, 0.025/4), orientation='horizontal',
             fc=green, edgecolor=colours[0])
    ax2.hist(gal_kappa, bins=np.arange(-0.12, 0.13 + 0.025/4, 0.025/4), orientation='horizontal',
             fc=yellow, edgecolor=colours[1])
    ax.legend(frameon=0)
    plt.show()

    return convergence


if __name__ == "__main__":
    with fits.open('MICEsim2.fits')as hdul1:
        # print(repr(hdul1[1].header))
        # print(repr(hdul1[1].data['dec']))

        max_z = max(hdul1[1].data['z'])
        print(len(hdul1[1].data['z']))
        redo_random = False
        redo_expected = False
        bins = 100
        chi_bin_widths, chi_bins, z_bins, z_bin_widths = create_chi_bins(0, max_z, bins)
        lims = np.cumsum(z_bin_widths)
        RAs = hdul1[1].data['ra']
        DECs = hdul1[1].data['dec']
        zs = hdul1[1].data['z']
        kappas = hdul1[1].data['kappa']
        MICEdata = [RAs, DECs, zs, kappas]

    r_small = 0.2  # degrees
    r_big = 3.2  # degrees
    print((min(RAs) + max(RAs)) / 2, (min(DECs) + max(DECs)) / 2, min(RAs), max(RAs), min(DECs), max(DECs))
    loc = [(min(RAs) + max(RAs)) / 2, (min(DECs) + max(DECs)) / 2]
    counter = 0
    if redo_random:
        big_cone = {'zs': [], 'kaps': []}
        for RA, DEC, z, kap in zip(RAs, DECs, zs, kappas):
            if (RA - loc[0]) ** 2 + (DEC - loc[1]) ** 2 <= r_big ** 2 and z >= 0.01:
                big_cone['zs'].append(z)
                big_cone['kaps'].append(kap)
            counter += 1
            if counter % 10000 == 0:
                print(f"Found {counter}/{len(RAs)}")

        pickle_out = open("big_cone.pickle", "wb")
        pickle.dump(big_cone, pickle_out)
        pickle_out.close()

        random.seed(1337)
        rand_samp_size = 1000
        q1 = rand_samp_size/4
        q2 = rand_samp_size/2
        q3 = 3*rand_samp_size/4
        rand_gal_z = random.sample(list(zs), rand_samp_size)
        indices = [np.argmin(np.abs(zs - rand_gal_z[i])) for i in range(rand_samp_size)]
        rand_gal_RA = RAs[indices]
        rand_gal_DEC = DECs[indices]
        rand_gal_kap = kappas[indices]
        counter = 0
        for q_num, q in enumerate([[0, q1], [q1, q2], [q2, q3], [q3, rand_samp_size]]):
            s = int(q[0])
            e = int(q[1])
            rand_cones = {}
            for randRA, randDEC, randZ, randKap in zip(rand_gal_RA[s:e], rand_gal_DEC[s:e],
                                                       rand_gal_z[s:e], rand_gal_kap[s:e]):
                rand_cones[f'SN{int(counter)+1}'] = {'ras': [], 'decs': [], 'zs': [], 'zSN': randZ, 'kapSN': randKap,
                                                     'raSN': randRA, 'decSN': randDEC}
                for RA, DEC, z in zip(RAs, DECs, zs):
                    if (randRA - RA) ** 2 + (randDEC - DEC) ** 2 <= r_small ** 2 and z >= 0.01:
                        rand_cones[f'SN{int(counter)+1}']['zs'].append(z)
                        rand_cones[f'SN{int(counter)+1}']['ras'].append(RA)
                        rand_cones[f'SN{int(counter)+1}']['decs'].append(DEC)
                counter += 1
                print(f"Sorted {counter}/{rand_samp_size}")

            pickle_out = open(f"random_cones_q{q_num+1}.pickle", "wb")
            pickle.dump(rand_cones, pickle_out)
            pickle_out.close()
            print(f"Finished quarter {q_num+1}")
    else:
        pickle_in = open("big_cone.pickle", "rb")
        big_cone = pickle.load(pickle_in)
        rand_cones = {}
        for x in [1, 2, 3, 4]:
            pickle_in = open(f"random_cones_q{x}.pickle", "rb")
            rand_cones_quarter = pickle.load(pickle_in)
            rand_cones.update(rand_cones_quarter)

        # with open('random_sample.csv', 'w', newline='') as csvfile:
        #     random_sample = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     random_sample.writerow(['ra', 'dec', 'z'])
        #     for key in rand_cones:
        #         for num in range(len(rand_cones[key]['ras'])):
        #             if rand_cones[key]['zs'][num] > rand_cones[key]['zSN']:
        #                 random_sample.writerow([rand_cones[key]['ras'][num], rand_cones[key]['decs'][num],
        #                                        rand_cones[key]['zs'][num]])

    if redo_expected:
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
    expected = [expected_big[i] * (r_small/r_big) ** 2 for i in range(len(expected_big))]
    plt.plot([0, 5], [0, 0], color=grey, linestyle='--')
    plt.plot(lims[1:], expected, marker='o', markersize=2.5, color=colours[0])
    plt.xlabel('$z$')
    plt.ylabel('Expected Count')
    plt.xlim([0, 1.5])
    plt.show()

    rand_gal_z = np.zeros(len(rand_cones))
    rand_gal_kap = np.zeros(len(rand_cones))
    c = 0
    for _, supernova in rand_cones.items():
        rand_gal_z[c] = supernova['zSN']
        rand_gal_kap[c] = supernova['kapSN']
        c += 1

    ds = []
    chis = []
    for z in rand_gal_z:
        chi_to_z = comoving(np.linspace(0, z, 1000), 0.25, 0.75)
        ds.append(chi_to_z[-1] * (1 + z))
        chis.append(chi_to_z[-1])
    mu = 5 * np.log10(np.array(ds) / 10 * 1E9)
    # plot_Hubble(sample, mu, np.zeros(len(sample)), OM=0.25, OL=0.75, max_x=1.5)
    conv = find_convergence(chis, rand_gal_z, lims, expected, chi_bin_widths, chi_bins, z_bins, rand_gal_kap, rand_cones,
                                redo=False)
