import numpy as np
from astropy.io import ascii
from emcee import EnsembleSampler

# Interpolators for temp and logg
import model_mag
import data_mag
from deredden import av_point

# Constants
R_sun = 6.96e10 # [cm]
pc = 3.086e18 # [cm]

class ModelError(Exception):
    '''
    Raised when Model parameters are outside the grid.
    '''
    def __init__(self, msg):
        self.msg = msg

filters = ascii.read("FILTERS.dat")

# Define the filter indices you want to sample
my_filters = ["U", "B", "V", "R", "I"] #, "J_2M"]

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

# Get the redenning coefficients at these wavelengths
avs = {name: av_point(wl_isos[name]) for name in my_filters}

# Get the zero point for each filter
zps = {name: get_row(name)["Fnu"] for name in my_filters}

# Get the temp, logg interpolators for each filter
interps = {name: model_mag.interp[name] for name in my_filters}

# Get the arrays of data points measured in each filter
fluxes = {name: data_mag.fluxes[name] for name in my_filters}
errs = {name: data_mag.errs[name] for name in my_filters}

# This is how to convert to redenning (in magnitudes), using the Av
# alambda = Av * av

def mod_fluxes(p):
    '''
    Calculate the model magnitudes for these parameters.
    '''
    temp, logg, R, Av, dpc = p

    if Av < 0.0 or Av > 7.75:
        raise ModelError("Av outside of allowable range.")

    R = R * R_sun # [cm]
    d = dpc * pc # [cm]

    fluxes = {}
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
        mod_mag = raw_mag - 5 * np.log10(R / d)

        # Get the redenning at this wavelength in magnitudes
        Afilt = avs[name] * Av

        # Add/Subtract from the model_mag
        red_mag = mod_mag + Afilt

        fluxes[name] = zps[name] * 10**(-0.4 * red_mag)


    return fluxes

def lnprior(p):
    temp, logg, R, Av, dpc = p
    mu_d = 142. # [pc]
    sig_d = 6. # [pc]
    return -0.5 * (dpc - mu_d)**2 / sig_d**2


def lnprob(p):
    '''
    Calculate the ln of the probability distribution.

    :param p: model parameters
    :type p: 1D numpy array

    '''

    lnp = 0

    try:
        mfluxes = mod_fluxes(p)
    except ModelError:
        # We tried interpolating outside the grid or Av is non-physical
        return -np.inf

    # Iterate through the filters and sum the lnprob
    for name in my_filters:
        # Add in the jitter term to the noise (Not yet implemented)

        # Evaluate chi2 for all values in this range
        lnp += -0.5 * np.sum((fluxes[name] - mfluxes[name])**2/errs[name])

    return lnp + lnprior(p)


def main():
    # print(lnprob(np.array([6550., 5.1, 1.0, 0.1, 142.])))

    ndim = 5
    nwalkers = 4 * ndim # about the minimum per dimension we can get by with

    p0 = np.array([ np.random.uniform(5000, 7000, nwalkers),
                    np.random.uniform(3.0, 5.0, nwalkers),
                    np.random.uniform(0.8, 2.0, nwalkers),
                    np.random.uniform(0.0, 3.5, nwalkers),
                    np.random.uniform(135, 150, nwalkers)]).T
    sampler = EnsembleSampler(nwalkers, ndim, lnprob) #, threads=mp.cpu_count())

    # burn in
    pos, prob, state = sampler.run_mcmc(p0, 4000)
    sampler.reset()
    print("Burned in")

    # actual run
    pos, prob, state = sampler.run_mcmc(pos, 4000)

    # Save the last position of the walkers
    np.save("walkers_emcee.npy", pos)
    np.save("eparams_emcee.npy", sampler.flatchain)

if __name__=="__main__":
    main()
