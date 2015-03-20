import numpy as np
from astropy.io import ascii

data = ascii.read("aksco_phot.dat")
filters = ascii.read("FILTERS.dat")

# Find all unique filters in the data set
all_filters = np.unique(data["filt_index"])

# Generate a function that returs True when we've reached a row that
# has the filter_index we want
def get_filt(filter_index):
    return lambda row: row["filt_index"] == filter_index

# Get the filter name corresponding to a filter id
def get_name(filter_index):
    "Assumes rows are unique so we just return the first"
    rows = list(filter(get_filt(filter_index), filters))
    assert len(rows) == 1, "Filter names in FILTERS.dat are not unique!"
    return rows[0]["name"]

all_names = [get_name(filter_index) for filter_index in all_filters]

fluxes = {}
errs = {}
for filt, name in zip(all_filters, all_names):
    flux = []
    err = []
    for row in filter(get_filt(filt), data):
        flux.append(row["Jy"])
        err.append(row["err"])

    fluxes[name] = np.array(flux)
    errs[name] = np.array(err)
