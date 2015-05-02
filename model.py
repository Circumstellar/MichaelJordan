import numpy as np
from astropy.io import ascii
from emcee import EnsembleSampler

# Interpolators for temp and logg
import model_mag
import data_mag
# from deredden import av_point
from unred import A_lams

# Constants
pc = 3.086e18 # [cm]
R_sun = 6.96e10 # [cm]
L_sun = 3.9e33 # [erg/s]
M_sun = 1.99e33 # [g]
sigma_k = 5.67051e-5 # [erg cm-2 K-4 s-1] Stefan-Boltzman constant
G = 6.67259e-8 # [cm^3 /g /s^2] Gravitational constant

class ModelError(Exception):
    '''
    Raised when Model parameters are outside the grid.
    '''
    def __init__(self, msg):
        self.msg = msg

filters = ascii.read("FILTERS.dat")

# Define the filter indices you want to sample
my_filters = ["U", "B", "V", "R", "I"]#, "H_2M"] #, "J_2M"]

# Get the iso-photal wavelength for each filter
def get_filt(filter_name):
    return lambda row: row["name"] == filter_name

def get_row(filter_name):
    "Assumes rows are unique so we just return the first"
    rows = list(filter(get_filt(filter_name), filters))
    assert len(rows) == 1, "Filter names in FILTERS.dat are not unique!"
    return rows[0]

# Get the iso_photal wavelength for each filter
wl_isos = {name: get_row(name)["wl_iso"] for name in my_filters}

wl_iso_arr = 1e4 * np.array([get_row(name)["wl_iso"] for name in my_filters]) # [AA]

# Get the redenning coefficients at these wavelengths
# avs = {name: av_point(wl_isos[name]) for name in my_filters}

# Get the zero point for each filter
zps = {name: get_row(name)["Fnu"] for name in my_filters}

# Get the temp, logg interpolators for each filter
interps = {name: model_mag.interp[name] for name in my_filters}

# # Get the arrays of data points measured in each filter
fluxes = {name: data_mag.fluxes[name] for name in my_filters}
errs = {name: data_mag.errs[name] for name in my_filters}

# This is how to convert to redenning (in magnitudes), using the Av
# alambda = Av * av

def mod_fluxes(p):
    '''
    Calculate the model magnitudes for these parameters.
    '''
    temp, logg, logL, Av, dpc = p

    if Av < 0.0 or Av > 7.75: # or Rv < 3.0 or Rv > 5.0:
        raise ModelError("Av outside of allowable range.")

    L = 10**logL * L_sun # [erg/s]

    Omega = L / ((dpc * pc)**2 * 4 * np.pi * sigma_k * temp**4)

    fluxes = {}

    # Get the extinction in magnitudes
    avs = dict(zip(my_filters, A_lams(wl_iso_arr, Av, 4.3)))

    # Iterate through the filters and sum the lnprob
    for name in my_filters:
        # get the raw model magnitudes predicted at this temp, logg

        try:
            raw_mag = interps[name]((temp, logg))
        except ValueError as e:
            # Tried to interpolate outside the grid
            raise ModelError("Interpolating outside grid edges. {}".format(e))

        if raw_mag >= model_mag.max_val:
            # Interpolating outside jagged edges of grid.
            raise ModelError("Interpolating outside jagged edges of grid.")

        # Use the radius and distance to convert to physical model magnitudes
        mod_mag = raw_mag - 2.5 * np.log10(Omega)

        # Get the redenning at this wavelength in magnitudes
        Afilt = avs[name] * Av

        # Add/Subtract from the model_mag
        red_mag = mod_mag + Afilt

        fluxes[name] = zps[name] * 10**(-0.4 * red_mag)


    return fluxes

def lnprior(p):
    temp, logg, logL, Av, dpc = p
    mu_d = 142. # [pc]
    sig_d = 6. # [pc]


    # Prior on temperature
    mu_T = 6450 # K
    sig_T = 150 # K


    # For using the Mass-distance joint prior from disk-fitting
    # L = 10**logL * L_sun # [erg/s] total luminosity for both stars
    # R_1 = np.sqrt(L / (8 * np.pi * sigma_k * temp**4))# [cm]  # The radius obtained for a single star
    # M = 10**logg * R_1**2/ G  / M_sun # [M_sun] Mass obtained for single star
    # M2 = 2 * M # Mass obtained for the pair of stars
    #
    # # Evaluate the joint prior for Mass and Distance obtained from the sub-mm modelling
    # # Covariance matrix determined from disk modeling estimate
    # Sigma = np.array([[1.0e-2, 0.5624], [0.5624, 34.]])
    # x = np.array([M2 - 2.485, dpc - mu_d]) # Residual vector
    # invSigma = np.linalg.inv(Sigma)
    # s, logdet = np.linalg.slogdet(Sigma)
    # lnp = -0.5 * (x.dot(invSigma).dot(x) + logdet + 2 * np.log(2 * np.pi))

    lnp =  (-0.5 * (temp - mu_T)**2 / sig_T**2) + (-0.5 * (dpc - mu_d)**2 / sig_d**2)

    return lnp

def lnprob(p):
    '''
    Calculate the ln of the probability distribution.

    :param p: model parameters
    :type p: 1D numpy array

    '''

    lnp = 0

    try:
        mfluxes = mod_fluxes(p[:5])
    except ModelError as e:
        # We tried interpolating outside the grid or Av is non-physical
        # print("ModelError", e)
        return -np.inf

    if np.any(p[5:] < 0.0):
        return -np.inf

    jitter = dict(zip(my_filters, p[5:]))
    # Iterate through the filters and sum the lnprob
    for name in my_filters:
        # Add in the jitter term to the noise, plus a penalty

        # Evaluate chi2 for all values in this range
        var = errs[name]**2 + jitter[name]**2
        lnp += -0.5 * np.sum((fluxes[name] - mfluxes[name])**2/var + np.log(var))

    return lnp + lnprior(p[:5])


def main():
    # print(lnprob(np.array([6550., 4.1, 1.0, 0.1, 3.1, 142., .1, .1, .1, .1, .1])))
    # import sys
    # sys.exit()

    ndim = 10
    nwalkers = 6 * ndim

    p0 = np.array([ np.random.uniform(5800, 6800, nwalkers),
                    np.random.uniform(3.2, 4.99, nwalkers),
                    np.random.uniform(0.6, 1.0, nwalkers),
                    np.random.uniform(0.0, 0.8, nwalkers),
                    # np.random.uniform(3.1, 5.0, nwalkers),
                    np.random.uniform(135, 150, nwalkers),
                    np.random.uniform(0.01, 0.1, nwalkers),
                    np.random.uniform(0.01, 0.1, nwalkers),
                    np.random.uniform(0.01, 0.1, nwalkers),
                    np.random.uniform(0.01, 0.1, nwalkers),
                    np.random.uniform(0.01, 0.1, nwalkers)]).T
    sampler = EnsembleSampler(nwalkers, ndim, lnprob, threads=3)

    # burn in
    pos, prob, state = sampler.run_mcmc(p0, 5000)
    sampler.reset()
    print("Burned in")

    # actual run
    pos, prob, state = sampler.run_mcmc(pos, 5000)

    # Save the last position of the walkers
    np.save("walkers_emcee.npy", pos)
    np.save("eparams_emcee.npy", sampler.flatchain)

if __name__=="__main__":
    main()
