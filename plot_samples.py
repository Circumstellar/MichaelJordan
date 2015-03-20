#!/usr/bin/env python

import numpy as np
import triangle

def npread(fname, burn=0, thin=1):
    '''
    Read the flatchain
    '''
    return np.load(fname)[burn::thin]


def plot(flatchain, format=".png"):
    '''
    Make a triangle plot
    '''

    labels = [r"$T_\textrm{eff}$ [K]", r"$\log g$ [dex]", r"$R$ [$R_\odot$]", r"$A_V$ [mag]", r"$d$ [pc]"]
    figure = triangle.corner(flatchain, quantiles=[0.16, 0.5, 0.84],
        plot_contours=True, plot_datapoints=False, labels=labels, show_titles=True)
    figure.savefig("triangle" + format)

def main():
    plot(npread("eparams_emcee.npy"))

if __name__=="__main__":
    main()
