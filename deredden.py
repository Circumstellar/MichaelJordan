#==============================================================================
# DEREDDEN.py Sean Andrews's deredden.pro ported to python3
#
# A simple function to provide the de-reddening factor in either magnitudes
# (with keyword /mags set) or flux density at a range of input wavelengths,
# given a visual extinction (Av).
#
# made composite extinction curves for different Av
#			regimes: at higher Av, use McClure 2009 model, but at
#			lower Av can use the Rv = 3.1 (DISM) Mathis 1990 model.
#           the McClure 2009 model switches at Ak = 1
#==============================================================================

from astropy.io import ascii
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

data = ascii.read("ext_curves.dat")
awl = data['wl'] #wavelength grid for extinction [microns]
A1 = data['A1'] #Mathis Law
A2 = data['A2'] # Valid 0.3 < Ak < 1
A3 = data['A3'] # Valid 1 < Ak < 7

#what is Av_me? An arbitrary cutoff assumed by Sean?
# Alambda = Av * (1/7.75  *  interpol(A2,awl,wl))

def deredden(wl, Av, thres=None, mags=True):
    '''Takes in wavelength array in microns. Valid between .1200 um and 1e4 microns.'''
    #- thresholds for different extinction curve regimes
    if thres is not None:
        Av_lo = thresh
    else:
        Av_lo = 0.0

    Av_me = 2.325 #McClure 2009 threshold: AK = 0.3

    if (Av_lo >= Av_me):
        Av_lo = 0.0

    Av_hi = 7.75 #McClure 2009 threshold: AK = 1.0

    if (Av >= Av_hi):
        AA = A3
        AvAk = 7.75

    if (Av >= Av_me) and (Av < Av_hi):
        AA = A2
        AvAk = 7.75

    if (Av >= Av_lo) and (Av < Av_me):
        AA = A2
        AvAk = 7.75

    if (Av < Av_lo):
        AA = A1
        AvAk = 9.03

    AK_AV = 1. / AvAk

    #interpolate extinction curve onto input wavelength grid
    Alambda_func = interp1d(awl, Av * AK_AV * AA)
    Alambda = Alambda_func(wl)

    # - return the extinction at input wavelengths
    #at this point, Alambda is in magnitudes

    if mags:
        return Alambda
    else:
        # to convert to flux, raise 10^(0.4 * Alambda)
        return 10. ** (0.4 * Alambda)

def av_point(wl):
    '''call this, get grid. multiply grid by Av to get redenning at that wavelength.'''
    # Using A2
    AK_AV = 1 / 7.75
    Alambda_func = interp1d(awl, AK_AV * A2, kind='linear')
    return Alambda_func(wl)

def create_red_grid(wl):
    avs = av_points(wl)
    np.save('red_grid.npy',avs)


def plot_curve():
    '''To test implementation'''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    wl = np.linspace(.13, 10, num=300)
    ax.plot(wl, deredden(wl, .2, mags=False), label="0.2 mags")
    ax.plot(wl, deredden(wl, 1.0, mags=False), label="1.0 mags")
    ax.plot(wl, deredden(wl, 2.0, mags=False), label="2.0 mags")
    avs = av_points(wl)
    ax.plot(wl, 10**(0.4 * avs), "k:", label="fiducial")
    ax.legend(loc="upper right")
    ax.set_xlabel(r"$\lambda\quad[\AA]$")
    ax.set_ylabel(r"$A_\lambda$")
    plt.savefig("redenning_curves.png")


def main():
    plot_curve()

if __name__ == "__main__":
    main()
