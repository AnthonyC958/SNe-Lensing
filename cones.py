import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

names = ['boss_206+SDSS_213_all_cuts_new_mu_dmu1_new.fits', 'STRIPE82_SPECTROSCOPIC_CHAZ_NOTCLEANED_ms77.fit']
for file in names:
    with fits.open(file) as hdul:
        hdul.info()

plt.style.use(astropy_mpl_style)
img_file = get_pkg_data_filename('tutorials/FITS-images/HorseHead.fits')
img_data = fits.getdata(img_file, ext=0)
plt.figure()
plt.imshow(img_data)
plt.colorbar()
plt.show()
