from Convergence import *
from mpl_toolkits.mplot3d import Axes3D
# from astropy.visualization import astropy_mpl_style
# from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import pickle

colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

if __name__ == "__main__":
    names = ['STRIPE82_SPECTROSCOPIC_CHAZ_NOTCLEANED_ms77.fit', 'boss_206+SDSS_213_all_cuts_new_mu_dmu1_new.fits']
    with fits.open(names[0])as hdul1:
        with fits.open(names[1]) as hdul2:
            if False:
                RA1 = [hdul1[1].data['RA'][i] for i in np.arange(len(hdul1[1].data['RA'])) if
                       hdul1[1].data['CLASS'][i] == 'GALAXY' and hdul1[1].data['Z'][i] >= 0.01]
                DEC1 = [hdul1[1].data['DEC'][i] for i in np.arange(len(hdul1[1].data['DEC'])) if
                        hdul1[1].data['CLASS'][i] == 'GALAXY' and hdul1[1].data['Z'][i] >= 0.01]
                for num, ra in enumerate(RA1):
                    if ra > 60:
                        RA1[num] -= 360
                RA2 = [hdul2[1].data['RA'][i] for i in np.arange(len(hdul2[1].data['RA'])) if
                       hdul2[1].data['Z_BOSS'][i] >= 0.05]
                DEC2 = [hdul2[1].data['DECL'][i] for i in np.arange(len(hdul2[1].data['DECL'])) if
                        hdul2[1].data['Z_BOSS'][i] >= 0.05]

                cut_data = np.array([RA1, DEC1, RA2, DEC2])
                pickle_out = open("cut_data.pickle", "wb")
                pickle.dump(cut_data, pickle_out)
                pickle_out.close()
            else:
                pickle_in = open("cut_data.pickle", "rb")
                cut_data = pickle.load(pickle_in)
                RA1 = cut_data[0]
                DEC1 = cut_data[1]
                RA2 = cut_data[2]
                DEC2 = cut_data[3]

            patches = []
            for x, y in zip(RA2, DEC2):
                circle = Circle((x, y), 0.2)
                patches.append(circle)

            z1 = hdul1[1].data['Z']
            z2 = hdul2[1].data['Z_BOSS']

            if False:  # Change if dictionary needs to be made again
                lenses = {}
                for num, SRA, SDE, SZ in zip(np.linspace(0, len(RA2)-1, len(RA2)), RA2, DEC2, z2):
                    lenses[f'SN{int(num)+1}'] = {}
                    lenses[f'SN{int(num)+1}']['RAs'] = []
                    lenses[f'SN{int(num)+1}']['DECs'] = []
                    lenses[f'SN{int(num)+1}']['Zs'] = []
                    lenses[f'SN{int(num)+1}']['ZSN'] = SZ
                    for GRA, GDE, GZ in zip(RA1, DEC1, z1):
                        if (GRA - SRA) ** 2 + (GDE - SDE) ** 2 <= 0.2 ** 2:
                            lenses[f'SN{int(num)+1}']['RAs'].append(GRA)
                            lenses[f'SN{int(num)+1}']['DECs'].append(GDE)
                            lenses[f'SN{int(num)+1}']['Zs'].append(GZ)
                    print(f'Finished {int(num)+1}/{len(RA2)}')
                    # print(contribs[f'SN{int(num)+1}'])

                pickle_out = open("conts.pickle", "wb")
                pickle.dump(lenses, pickle_out)
                pickle_out.close()
            else:
                pickle_in = open("conts.pickle", "rb")
                lenses = pickle.load(pickle_in)

    fig, ax = plt.subplots()
    ax.plot(RA1, DEC1, marker='o', linestyle='', markersize=1, color=[0.5, 0.5, 0.5])
    for SN, dict1, in lenses.items():
        RAs = np.array(dict1['RAs'])
        DECs = np.array(dict1['DECs'])
        indices1 = dict1['Zs'] < dict1['ZSN']
        indices2 = dict1['Zs'] > dict1['ZSN']
        # print("Galaxies:", len(dict1['RAs']), "with z < z_SN:", len(RAs[indices]),
        #       f"at around: ({RAs[0]}, {DECs[0]})")
        ax.plot(RAs[indices1], DECs[indices1], marker='o', linestyle='', markersize=3, color=colours[0],
                label="Foreground" if SN == 'SN1' else "")
        ax.plot(RAs[indices2], DECs[indices2], marker='o', linestyle='', markersize=1, color='k',
                label="Background" if SN == 'SN1' else "")
    p = PatchCollection(patches, alpha=0.4)
    ax.add_collection(p)
    ax.plot(RA2, DEC2, marker='o', linestyle='', markersize=3, label='Supernova', color=colours[1])
    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')
    plt.legend(loc='lower right')
    plt.axis('equal')
    plt.xlim([24.5, 27.5])
    plt.ylim([-1, 1])
    # plt.show()

    # print(repr(hdul1[1].header))
    labels = ['Galaxies', 'Supernovae']
    for num, z in enumerate([z1, z2]):
        plt.hist([i for i in z if i <= 0.6], bins=np.arange(0, 0.6+0.025, 0.025), normed='max', alpha=0.5,
                 edgecolor=colours[num], linewidth=2, label=f'{labels[num]}')
    plt.xlabel('z')
    plt.ylabel('Normalised Count')
    plt.legend(frameon=0)
    # plt.show()

    tests = []
    for a in range(272):
        for b in range(6):
            test = [-50.6, 1.0]
            test[0] += a * 0.4
            test[1] -= b * 0.4
            test[0] = round(test[0], 1)
            test[1] = round(test[1], 1)
            tests.append(test)

    if False:
        cones = {}
        for num, loc, in enumerate(tests):
            cones[f'c{int(num)+1}'] = {}
            cones[f'c{int(num)+1}']['Total'] = 0
            cones[f'c{int(num)+1}']['Zs'] = []
            for GRA, GDE, GZ in zip(RA1, DEC1, z1):
                if (GRA - loc[0]) ** 2 + (GDE - loc[1]) ** 2 <= 0.2 ** 2:
                    cones[f'c{int(num)+1}']['Zs'].append(GZ)
                cones[f'c{int(num)+1}']['Total'] = len(cones[f'c{int(num)+1}']['Zs'])
            print(f'Finished {int(num)+1}/{len(tests)}')

        pickle_out = open("cones.pickle", "wb")
        pickle.dump(cones, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("cones.pickle", "rb")
        cones = pickle.load(pickle_in)

    plt.hist([cones[f'c{i+1}']['Total'] for i in range(len(cones))], density=1,
             bins=20, edgecolor=colours[0], alpha=.5, linewidth=2)
    # plt.show()
    # print("Max:", max([max(cones[f'c{i+1}']['Zs']) for i in range(len(cones))]))

    chi_widths, chis, zs, widths = create_chi_bins(0, max([max(cones[f'c{i+1}']['Zs']) for i in range(len(cones))]), 100)
    limits = np.cumsum(widths)
    print(limits)
    if False:
        expected = np.zeros((len(limits), len(cones)))
        c = 0
        for num1, lim in enumerate(limits):
            for num2, _ in enumerate(cones.items()):
                expected[num1][num2] = sum([cones[f'c{num2+1}']['Zs'][i] < lim
                                            for i in range(len(cones[f'c{num2+1}']['Zs']))])
                c += 1
                if c % 1000 == 0:
                    print(f"Finished {c}/{len(limits)*len(cones)}")
        print("Finished")

        pickle_out = open("expected.pickle", "wb")
        pickle.dump(expected, pickle_out)
        pickle_out.close()
    else:
        pickle_in = open("expected.pickle", "rb")
        expected = pickle.load(pickle_in)

    expected_counts = np.diff([np.mean(expected[i][:]) for i in range(len(limits))])
    plt.plot(limits[1:], expected_counts)
    plt.show()

    ZSNs = []
    for _, SN in lenses.items():
        ZSNs.append(SN['ZSN'])

    chiSNs = []
    for SN in ZSNs:
        chi = comoving(np.linspace(0, SN, 1001))
        chiSNs.append(chi[-1])

    counts = {}
    c = 0
    for num1 in range(len(lenses)):
        bin_c = range(np.argmin(np.abs(limits - lenses[f"SN{num1+1}"]["ZSN"])))
        counts[f"SN{num1+1}"] = np.zeros(len(bin_c))
        for num2 in bin_c:
            counts[f"SN{num1+1}"][num2] = sum([limits[num2] < lenses[f'SN{num1+1}']['Zs'][i]
                                               <= limits[num2 + 1]
                                               for i in range(len(lenses[f'SN{num1+1}']['Zs']))])
        c += 1
        print(f"Finished {c}/{len(lenses)}")

    density = {}
    convergence = []
    c = 0
    for key, SN in counts.items():
        density[f"{key}"] = (SN - expected_counts[:len(SN)])/expected_counts[:(len(SN))]
        convergence.append(smoothed_m_convergence(chi_widths[:len(SN)], chis[:len(SN)], zs[:len(SN)],
                                                  density[f"{key}"], chiSNs[c]))
        c += 1

    plt.plot(ZSNs, [abs(convergence[i]) for i in range(len(convergence))], linestyle='', marker='o')
    plt.show()

    #     pickle_out = open("counts.pickle", "wb")
    #     pickle.dump(counts, pickle_out)
    #     pickle_out.close()
    # else:
    #     pickle_in = open("counts.pickle", "rb")
    #     counts = pickle.load(pickle_in)

    # plt.style.use(astropy_mpl_style)
    # img_file = get_pkg_data_filename('tutorials/FITS-images/HorseHead.fits')
    # img_data = fits.getdata(img_file, ext=0)
    # plt.figure()
    # plt.imshow(img_data)
    # plt.colorbar()
    # plt.show()
