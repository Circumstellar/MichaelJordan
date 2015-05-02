# Using the samples for teff, logg, logL, Av, and dpc, convert into estimates
# of the stellar radii based upon the assumption that they have equal mass ratio
# and the same Teff, same radius

import numpy as np
import triangle

# Constants
R_sun = 6.96e10 # [cm]
pc = 3.086e18 # [cm]
L_sun = 3.9e33 # [erg/s]
sigma_k = 5.67051e-5 # [erg cm-2 K-4 s-1] Stefan-Boltzman constant


def npread(fname, burn=0, thin=1):
    '''
    Read the flatchain
    '''
    return np.load(fname)[burn::thin]


def plot(flatchain, format=".png"):
    '''
    Make a triangle plot
    '''

    labels = [r"$T_\textrm{eff}$ [K]", r"$\log g$ [dex]", r"$\log_{10} L$ [$L_\odot$]", r"$A_V$ [mag]", r"$d$ [pc]", r"$R_1$ [$R_\odot$]"]
    figure = triangle.corner(flatchain, quantiles=[0.16, 0.5, 0.84],
        plot_contours=True, plot_datapoints=False, labels=labels, show_titles=True)
    figure.savefig("Rtriangle" + format)


flatchain = npread("eparams_emcee.npy")[:, 0:5]

temp = flatchain[:, 0]
L = 10**flatchain[:,2] * L_sun

R_1 = np.sqrt(L / (8 * np.pi * sigma_k * temp**4)) / R_sun # [R_sun]

new_flat = np.hstack((flatchain, R_1[:, np.newaxis]))
np.save("eparams_R.npy", new_flat)

# Evaluate covariance of new_flat
print(np.mean(temp))
print(np.mean(R_1))
cov = np.cov(np.array([temp, R_1]).T, rowvar=0)
print(cov)

plot(new_flat)
