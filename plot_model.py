# Plot the data and the model, and see what the discrepancy is.

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter as FSF
from matplotlib.ticker import MultipleLocator
import numpy as np

import model
import data_mag

p = np.array([6363., 3.4, 0.78, 0.70, 142., 0.05, 0.1, 0.13, 0.14, 0.14])
# p = np.array([7680., 4.23, 2.2, 1.52, 142.])
mod_fluxes = model.mod_fluxes(p[:5])

print(mod_fluxes)

jitter = dict(zip(model.my_filters, p[5:]))

fig, ax = plt.subplots(nrows=2, figsize=(3.5,2.5), sharex=True)

# Go through each filter and plot the data, model, and residuals
for name in model.my_filters:
    data = data_mag.fluxes[name]
    ndata = len(data)
    err = data_mag.errs[name]

    avg_err = np.average(err)
    jit = jitter[name]

    tot_err = np.sqrt(avg_err**2 + jit**2)

    mod = mod_fluxes[name]
    wl = model.wl_isos[name]
    wl_scat = wl * (1 + np.random.uniform(low=-0.02, high=0.02, size=ndata))

    ax[0].plot(wl_scat, data, "o", color="0.5", ms=3.5, alpha=0.3)
    ax[0].plot(wl, mod, "bo", ms=7, alpha=0.8)

    ax[1].plot(wl_scat, data - mod, "o", color="0.5", ms=3.5, alpha=0.3)
    ax[1].errorbar(wl, 0, yerr=tot_err, ecolor="k", zorder=100)

# ax[0].yaxis.set_major_formatter(FSF("%.0f"))
ax[0].yaxis.set_major_locator(MultipleLocator(0.3))

labels = ["data", "residuals"]
for a,label in zip(ax, labels):
    a.annotate(label, (0.05, 0.8), xycoords="axes fraction")

vvmax = np.max(np.abs(data - mod))
ax[1].yaxis.set_major_locator(MultipleLocator(0.2))
ax[1].set_ylim(-vvmax, vvmax)

ax[0].set_ylabel(r"$F_\nu$ [Jy]")
ax[1].set_xlabel(r"$\lambda$ [$\mu$m]")
# ax[1].set_ylabel(r"$F_\nu$ [Jy]")
fig.subplots_adjust(left=0.19, right=0.81, top=0.96, bottom=0.2)


fig.savefig("residuals.pdf")
