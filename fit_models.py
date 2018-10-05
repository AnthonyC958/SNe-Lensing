import os
import numpy as np
from scipy.integrate import quad
import matplotlib
import pickle
import csv
from astropy.io import fits

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def Hz_inverse(z, om, ol):
    """ Calculate 1/H(z). Will integrate this function. """
    Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
    return 1.0 / Hz


def Hz_inversew(z, om, ox, w):
    """ Calculate 1/H(z). Will integrate this function. """
    ok = 1.0 - om - ox
    Hz = np.sqrt(om * (1 + z) ** 3 + ox * (1 + z) ** (3 * (1 + w)) + ok * (1 + z) ** 2)
    return 1.0 / Hz


def dist_mod(zs, om, ol, w):
    """ Calculate the distance modulus, correcting for curvature"""
    ok = 1.0 - om - ol
    x = np.array([quad(Hz_inversew, 0, z, args=(om, ol, w))[0] for z in zs])
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
    else:
        D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    return dist_mod


# ---------- Uncomment to load SDSS data  -------------------------------------
# S_CID = []
# with open('Smithdata.csv', 'r') as f:
#     CSV = csv.reader(f, delimiter=',')
#     for line in CSV:
#         S_CID.append(int(float(line[0].strip())))
#
# with fits.open('boss_206+SDSS_213_all_cuts_new_mu_dmu1_new.fits')as hdul1:
#     zz = np.array([hdul1[1].data['Z_BOSS'][i] for i in np.arange(len(hdul1[1].data['RA'])) if
#                    hdul1[1].data['CID'][i] in S_CID])
#     mu = np.array([hdul1[1].data['MU'][i] for i in np.arange(len(hdul1[1].data['RA'])) if
#                    hdul1[1].data['CID'][i] in S_CID])
#     mu_error = np.array([hdul1[1].data['DMU1'][i] for i in np.arange(len(hdul1[1].data['RA'])) if
#                          hdul1[1].data['CID'][i] in S_CID])
# test     = g10f['M0DIF']

# ---------- Uncomment to load MICECAT data  -------------------------------------
with open("MICE_SN_data.pickle", "rb") as pickle_in:
    SN_data = pickle.load(pickle_in)
zz = SN_data['SNZ']
mu = SN_data['SNMU']
mu_error = SN_data['SNMU_ERR']
mu_error2 = mu_error ** 2  # squared for ease of use later
# data = np.genfromtxt("jla_lcparams_simple_new.txt",names=True,comments='#',dtype=None, skip_header=11)
# zz = data['zcmb']
# mu = data['mb']
# mu_error = data['dmb']
# mu_error2 = mu_error**2 # squared for ease of use later

# Define cosntants
H0 = 70.0
c_H0 = 2.998E5 / H0

# ---------- Set up fitting ranges ---------------------------
n = 501  # Increase this for a finer grid
oms = np.linspace(0, 0.5, n)  # Array of matter densities
ws = np.linspace(-0.5, -1.5, n)  # Array of cosmological constant values
chi2 = np.ones((n, n)) * np.inf  # Array to hold our chi2 values, set initially to super large values

n_marg = 500  # Number of steps in marginalisation
mscr_guess = 5.0 * np.log10(c_H0) + 25  # Initial guess for best mscr
mscr = np.linspace(mscr_guess - 1.0, mscr_guess + 1.0, n_marg)  # Array of mscr values to marginalise over
mscr_used = np.zeros((n, n)) # Array to hold the best fit mscr value for each om, ol combination
z_small = np.linspace(0, max(zz), 100)
# ---------- Do the fit ---------------------------
saved_output_filename = "saved_grid_%d.txt" % n

if os.path.exists(saved_output_filename):  # Load the last run with n grid if we can find it
    print("Loading saved data. Delete %s if you want to regenerate the points\n" % saved_output_filename)
    chi2 = np.loadtxt(saved_output_filename)
else:
    for i, om in enumerate(oms):
        for j, w in enumerate(ws):
            mu_model_small = dist_mod(z_small, om, 1 - om, w)
            mu_model = np.interp(zz, z_small, mu_model_small)
            # mu_model_norm = np.array(np.repeat(mu_model, len(mscr)), dtype=object)
            for k, m in enumerate(mscr):
                mu_model_norm = mu_model + m
                chi2_test = np.sum((mu_model_norm - mu) ** 2 / mu_error2) + ((om - 0.25) / 0.08) ** 2 + (
                            (w + 1.0) / 0.1) ** 2  # Priors might be added in the wrong scope
                if chi2_test < chi2[i, j]:
                    chi2[i, j] = chi2_test
                    mscr_used[i, j] = k
        print("Done %d out of %d" % (i + 1, oms.size))
    np.savetxt(saved_output_filename, chi2, fmt="%10.4f")

likelihood = np.exp(-0.5 * (chi2 - np.amin(chi2)))
chi2_reduced = chi2 / (len(zz) - 1 - 2)

indbest = np.argmin(chi2)  # Gives index of best fit but where the indices are just a single number
ibest = np.unravel_index(indbest, [n, n])  # Converts the best fit index to the 2d version (i,j)
print('Best fit values are (om,w)=(%s,%s)' % (oms[ibest[0]], ws[ibest[1]]))
print('Reduced chi^2 for the best fit is %s' % chi2_reduced[ibest[0], ibest[1]])

# Plot contours of 1, 2, and 3 sigma
omlikelihood = np.sum(likelihood, 1)
wlikelihood = np.sum(likelihood, 0)
ax1 = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
ax1.contour(oms, ws, np.transpose(chi2 - np.amin(chi2)), cmap="winter", **{'levels': [2.30, 6.18, 11.83]})
ax1.set_xlabel("$\Omega_m$", fontsize=12)
ax1.set_ylabel("$w$", fontsize=12)
ax2 = plt.subplot2grid((4, 4), (0, 0), colspan=3)
ax2.plot(oms, omlikelihood)
ax2.set_xticklabels([])
ax3 = plt.subplot2grid((4, 4), (1, 3), rowspan=3)
ax3.plot(wlikelihood, ws)
# ax3.set_yticklabels([])
plt.subplots_adjust(wspace=0, hspace=0)
ax1.set_xlim([0.1, 0.35])
ax2.set_xlim([0.1, 0.35])
ax1.set_ylim([-1.4, -0.7])
ax3.set_ylim([-1.4, -0.7])
plt.show()
print(oms[np.where(abs(-2*np.log(omlikelihood) - 1.0) < 0.5)])
# plt.savefig("contours.pdf", bbox_inches="tight", transparent=True)
# plt.close()
