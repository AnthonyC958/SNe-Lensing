import numpy as np
from scipy.integrate import quad
import matplotlib
import csv
from astropy.io import fits
import pickle

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def Hz_inverse(z, om, ox, w):
    """ Calculate 1/H(z). Will integrate this function. """
    ok = 1.0 - om - ox
    Hz = np.sqrt(om * (1 + z) ** 3 + ox * (1 + z) ** (3 * (1 + w)) + ok * (1 + z) ** 2)
    return 1.0 / Hz


def dist_mod(zs, om, ox, w):
    """ Calculate the distance modulus, correcting for curvature"""
    ok = 1.0 - om - ox
    x = np.array([quad(Hz_inverse, 0, z, args=(om, ox, w))[0] for z in zs])
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


def load_fitres(fitres_file):
    # scan header for variable names and number of lines to skip
    header = open(fitres_file, 'r').readlines()[
             :60]  # todo Increased to 60 from 30 to read in Dillons files... check this still works for others.
    for i, line in enumerate(header):
        if line.startswith('NVAR'):
            nvar = int(line.split()[1])
            varnames = header[i + 1].split()
        elif line.startswith('SN'):
            nskip = i
            break
    print('Reading in %s' % fitres_file)
    fitres = np.genfromtxt(fitres_file, skip_header=nskip, usecols=tuple(range(1, nvar + 1)), names=varnames[1:])
    return fitres


def load_m0dif(m0dif_file):
    # scan header for variable names and number of lines to skip
    header = open(m0dif_file, 'r').readlines()[
             :30]  # todo Increased to 60 from 30 to read in Dillons files... check this still works for others.
    for i, line in enumerate(header):
        if line.startswith('NVAR'):
            nvar = int(line.split()[1])
            varnames = header[i + 1].split()
        elif line.startswith('ROW'):
            nskip = i
            break
    print('Reading in %s' % m0dif_file)
    m0dif = np.genfromtxt(m0dif_file, skip_header=nskip, usecols=tuple(range(1, nvar + 1)), names=varnames[1:])
    return m0dif


# ---------- Uncomment to load SDSS data  -------------------------------------
S_CID = []
with open('Smithdata.csv', 'r') as f:
    CSV = csv.reader(f, delimiter=',')
    for line in CSV:
        S_CID.append(int(float(line[0].strip())))

with fits.open('boss_206+SDSS_213_all_cuts_new_mu_dmu1_new.fits')as hdul1:
    zz = np.array([hdul1[1].data['Z_BOSS'][i] for i in np.arange(len(hdul1[1].data['RA'])) if
                   hdul1[1].data['CID'][i] in S_CID])
    mu = np.array([hdul1[1].data['MU'][i] for i in np.arange(len(hdul1[1].data['RA'])) if
                   hdul1[1].data['CID'][i] in S_CID])
    mu_error = np.array([hdul1[1].data['DMU1'][i] for i in np.arange(len(hdul1[1].data['RA'])) if
                         hdul1[1].data['CID'][i] in S_CID])

# ---------- Uncomment to load MICECAT data  -------------------------------------
# with open("MICE_SN_data.pickle", "rb") as pickle_in:
#     SN_data = pickle.load(pickle_in)
# zz = SN_data['SNZ']
# mu = SN_data['SNMU']
# mu_error = SN_data['SNMU_ERR']
print(len(zz))
# pickle_in = open("MICEkappa_weighted.pickle", "rb")
# kappa_weighted = pickle.load(pickle_in)
# kappa_est = SN_data["SNkappa"]
# kappa_est = kappa_weighted["Radius6.25"]["SNkappa"]
# mu = mu + (5.0 / np.log(10) * np.array(kappa_est))
ombest = 0.271
# olbest = 0.760
wbest  = -0.913
best = (ombest, 1-ombest, wbest)
test1 = (0.29, 0.71, -0.96)
# test2 = (0.25, 0.75, -0.85)
test3 = (0.0, 1.0, -0.6)

# Uncomment for MICECAT fits
# ombest = 0.20
# wbest = -1.0
# best = (ombest, 0.75, wbest)
# test1 = (0.225, 1-0.225, -1.0)
# test2 = (0.25, 0.75, -1)
zz_model = np.linspace(0, 1.45, 500)  # Make logarithmic redshift array to better sample low-z
mu_model = dist_mod(zz_model, *best)  # Calculate the distance modulus corresponding to the model redshifts
mu_model_dataz = dist_mod(zz, *best)

# Calculate arbitrary offset (a weighted mean):
mscr = np.sqrt(np.sum(((mu - mu_model_dataz) / mu_error) ** 2) / np.sum(1. / mu_error ** 2))
# mscr = np.mean(mu-mu_model_dataz) #43.18
mu_model = mu_model + mscr
mu_model_dataz = mu_model_dataz + mscr

# Calculate some other models
mu_model_Cbell = dist_mod(zz_model, *test1) + mscr
# testmodel = dist_mod(zz_model, *test2) + mscr
testmode2   = dist_mod(zz_model, *test3) + mscr


# ---------- Plot a combined Hubble diagram ----------------
font = {'family': ' serif',
        # 'weight' : 'normal',
        'size': 18}
matplotlib.rc('font', **font)
from matplotlib import rcParams, rc

rcParams['mathtext.fontset'] = 'stix'

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 1]})
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0)
# f, (a0, a1) = plt.subplots(1,2, gridspec_kw = {'width_ratios':[3, 1]})

colours = 'r'
# colors = ['#E7717D','#AFD275','black']
# colors = ['#F13C20','#D79922','black']
# colors = ['#FF652F','#14A76C','#272727','grey','#FFE400','#AC3B61']
# colors = ['red','#86C232','black']
linecolours = ['grey', [0, 150 / 255, 100 / 255], [145 / 255, 4 / 255, 180 / 255], [225 / 255, 149 / 255, 0]]
# linecolors = colors[3:]
markersizes = 8
alphas = 0.65
capsize = 2
elinewidths = 2.0
# Plot each graph, and manually set the y tick values
axs[0].plot(zz_model, mu_model, color='white', zorder=1)  # ,label='($\Omega_M$, $\Omega_\Lambda$, $w$)'
axs[0].plot(zz_model, mu_model, '-', color=linecolours[0], linewidth=1, zorder=2)  # , label='(0.319, 0.681, -0.967)'
# axs[0].plot(zz_model, testmodel, ':', color=linecolours[1], linewidth=1, zorder=3)  # , label='(0.3, 0, 0)'
axs[0].plot(zz_model, mu_model_Cbell, '-.', color=linecolours[2], linewidth=1, zorder=4)  # , label='(1.0, 0, 0)'
axs[0].plot(zz_model, testmode2, '--', color=linecolours[3], linewidth=1, zorder=4) #, label='(1.0, 0, 0)'
subset = np.arange(len(zz))
axs[0].errorbar(zz[subset], mu[subset], yerr=mu_error[subset], fmt='.',
                elinewidth=0.7, markersize=markersizes, alpha=alphas, color=colours)
axs[0].legend(loc='upper left', frameon=False, fontsize=18)
axs[0].set_ylabel('$\mu$', fontsize=24)
# axs[0].set_ylim(33,45) #SDSS only
axs[0].set_ylim(35, 44)

fonts = 18
space = 0.15
start = 1.5
plt.text(0.55, start-space, '($\Omega_M$, $\Omega_\Lambda$, $w$)', family='serif', color='black', rotation=0, fontsize=fonts,
         ha='right')
# plt.text(0.45, start - space, f'$\Omega_M$ = {best[0]}''$^{+0.020}_{-0.019}$', color='k', rotation=0, fontsize=fonts, ha='left')
# plt.text(0.45, start - space * 2.3, '$w$ = 'f'{best[1]}''$^{+0.065}_{-0.067}$', color='k', rotation=0, fontsize=fonts, ha='left')
plt.text(0.55, start - space * 2, f'${best}$', color=linecolours[0], rotation=0, fontsize=fonts, ha='right')
plt.text(0.55, start - space * 3, f'{test1}', color=linecolours[2], rotation=0, fontsize=fonts, ha='right')
plt.text(0.55,start-space*4,f'{test3}', color=linecolours[3],rotation=0,fontsize=fonts,ha='right') #SDSS only

axs[1].plot(zz_model, mu_model - mu_model, '-', color=linecolours[0], linewidth=1, zorder=1)
# axs[1].plot(zz_model, testmodel - mu_model, ':', color=linecolours[1], linewidth=1, zorder=2)
axs[1].plot(zz_model, mu_model_Cbell - mu_model, '-.', color=linecolours[2], linewidth=1, zorder=3)
axs[1].plot(zz_model, testmode2 - mu_model, '--', color=linecolours[3], linewidth=1, zorder=3) # SDSS only
subset = np.arange(len(zz))
axs[1].errorbar(zz[subset], mu[subset] - mu_model_dataz[subset], yerr=mu_error[subset], fmt='.',
                elinewidth=0.7, markersize=markersizes, alpha=alphas, color=colours)
axs[1].xaxis.set_major_formatter(ScalarFormatter())
axs[1].ticklabel_format(style='plain')
axs[1].set_xlim(0.009,0.6) #SDSS only
# axs[1].set_xlim(0.009, 1.45)
axs[1].set_ylim(-0.8, 0.8)
axs[1].set_xlabel('$z$', fontsize=24)
axs[1].set_ylabel('$\Delta\mu$', fontsize=24)

# plt.savefig("hubble_diagram_combined.pdf", bbox_inches='tight')
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()
plt.close()
