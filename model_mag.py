import numpy as np
from astropy.io import fits, ascii
from scipy.interpolate import RegularGridInterpolator

# Grid point locations
temps = np.array([2000., 2100., 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7200, 7400, 7600, 7800, 8000, 8200, 8400, 8600, 8800, 9000, 9200, 9400, 9600, 9800, 10000, 10200, 10400, 10600, 10800, 11000, 11200, 11400, 11600, 11800, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000])

loggs = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])

# index 	name	wl_iso (um)	Fnu (zp)	source

filters = ascii.read("FILTERS.dat")
filt_indexes = filters["filt_index"]
filt_names = filters["name"]

hdulist = fits.open("raw_mags_BTSettl.fits")
data = hdulist[0].data.T

# Flag anything larger than 8e29 as being a bad value.
inds = data > 8e29
# This is the largest possible physical value from the grid.
max_val = np.max(data[~inds])

# Dictionary to hold interpolators
interp = {}

for i, filt_name in enumerate(filt_names):
    # Bi-Linear interpolators
    interp[filt_name] = RegularGridInterpolator((temps, loggs), data[:, :, i])

hdulist.close()
