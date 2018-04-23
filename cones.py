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
            patches = []
            RA1 = hdul1[1].data['RA']
            DEC1 = hdul1[1].data['DEC']
            for num, ra in enumerate(RA1):
                if ra > 60:
                    RA1[num] -= 360
            RA2 = hdul2[1].data['RA']
            DEC2 = hdul2[1].data['DECL']
            for x, y in zip(RA2, DEC2):
                circle = Circle((x, y), 0.2)
                patches.append(circle)

            z1 = hdul1[1].data['Z']
            z2 = hdul2[1].data['Z_BOSS']

            if False:  # Change if dictionary needs to be made again
                contribs = {}
                for num, SRA, SDE, SZ in zip(np.linspace(0, len(RA2)-1, len(RA2)), RA2, DEC2, z2):
                    contribs[f'SN{int(num)+1}'] = {}
                    contribs[f'SN{int(num)+1}']['RAs'] = []
                    contribs[f'SN{int(num)+1}']['DECs'] = []
                    contribs[f'SN{int(num)+1}']['Zs'] = []
                    contribs[f'SN{int(num)+1}']['ZSN'] = SZ
                    for GRA, GDE, GZ in zip(RA1, DEC1, z1):
                        if (GRA - SRA) ** 2 + (GDE - SDE) ** 2 <= 0.2 ** 2:
                            contribs[f'SN{int(num)+1}']['RAs'].append(GRA)
                            contribs[f'SN{int(num)+1}']['DECs'].append(GDE)
                            contribs[f'SN{int(num)+1}']['Zs'].append(GZ)
                    print(f'Finished {int(num)+1}/{len(RA2)}')
                    # print(contribs[f'SN{int(num)+1}'])

                pickle_out = open("conts.pickle", "wb")
                pickle.dump(contribs, pickle_out)
                pickle_out.close()
            else:
                pickle_in = open("conts.pickle", "rb")
                contribs = pickle.load(pickle_in)

            fig, ax = plt.subplots()
            ax.plot(RA1, DEC1, marker='s', linestyle='', markersize=1, label='Background', color=[0.5, 0.5, 0.5])
            for SN, dict1, in contribs.items():
                RAs = np.array(dict1['RAs'])
                DECs = np.array(dict1['DECs'])
                indices = dict1['Zs'] < dict1['ZSN']
                ax.plot(RAs[indices], DECs[indices], marker='o', linestyle='', markersize=2, color=colours[0],
                        label="Foreground" if SN == 'SN1' else "")
            p = PatchCollection(patches, alpha=0.4)
            ax.add_collection(p)
            ax.plot(RA2, DEC2, marker='o', linestyle='', markersize=2, label='Supernova', color=colours[1])
            plt.xlabel('Right Ascension')
            plt.ylabel('Declination')
            plt.legend(loc='lower right')
            plt.axis('equal')
            plt.xlim([24.5, 27.5])
            plt.ylim([-1, 1])
            plt.show()

            # print(repr(hdul2[1].header))
            labels = ['Galaxies', 'Supernovae']
            for num, z in enumerate([z1, z2]):
                plt.hist([i for i in z if i <= 0.6], bins=np.arange(0, 0.6+0.025, 0.025), normed='max', alpha=0.5,
                         edgecolor=colours[num], linewidth=2, label=f'{labels[num]}')
            plt.xlabel('z')
            plt.ylabel('Normalised Count')
            plt.legend()
            plt.show()

    # plt.style.use(astropy_mpl_style)
    # img_file = get_pkg_data_filename('tutorials/FITS-images/HorseHead.fits')
    # img_data = fits.getdata(img_file, ext=0)
    # plt.figure()
    # plt.imshow(img_data)
    # plt.colorbar()
    # plt.show()
