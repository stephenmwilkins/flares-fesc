import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt

import flares
import flare.plt as fplt


fl = flares.flares('/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5', sim_type='FLARES')
halo = fl.halos
tag = fl.tags[-3]  #This would be z=7





spec_type = 'DustModelI'


quantities = []
# quantities.append(['Galaxy', 'SFR_inst_30']) # needed for sSFR
quantities.append(['Galaxy', 'Mstar_30']) # needed for selection
quantities.append([f'Galaxy/BPASS_2.2.1/Chabrier300/Lines/{spec_type}/HI4861', 'EW', 'HbetaEW'])
# quantities.append([f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic', 'FUV', 'IntrinsicFUV']) # needed for AFUV
quantities.append([f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/{spec_type}', 'FUV']) # needed for beta
quantities.append([f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/{spec_type}', 'NUV']) # needed for beta
# quantities.append()
# quantities.append()



qs = []
d = {}
D = {}
for Q in quantities:
    if len(Q)>2:
        qpath, q, qname = Q
    else:
        qpath, q = Q
        qname = q

    d[qname] = fl.load_dataset(q, arr_type=qpath)
    D[qname] = np.array([])
    qs.append(qname)

# --- read in weights
df = pd.read_csv('/cosma/home/dp004/dc-wilk2/data/flare/modules/flares/weight_files/weights_grid.txt')
weights = np.array(df['weights'])
ws = np.array([])


for ii in range(len(halo)):
    ws = np.append(ws, np.ones(np.shape(d[q][halo[ii]][tag]))*weights[ii])
    for q in qs:
        D[q] = np.append(D[q], d[q][halo[ii]][tag])



# ----------------------------------------------
# adjust units

D['Mstar_30'] *= 1E10

# ----------------------------------------------
# define selection

s = np.log10(D['Mstar_30'])>8.5

# ----------------------------------------------
# derive new quantities

beta = np.log10(D['FUV']/D['NUV'])/np.log10(1500/2500)-2.0

# ----------------------------------------------
# Print range of third quantity

print(f'Total number of galaxies: {len(ws[s])}')
print('weight:', np.min(ws[s]), np.max(ws[s]))




# ----------------------------------------------
# ----------------------------------------------
# Plot

cmap = cm.copper

fig, ax, cax = fplt.single_wcbar(base_size=3.5)

norm = mpl.colors.Normalize(vmin=-5, vmax=-1)
ax.scatter(np.log10(D['HbetaEW'][s]), beta[s], s=1, alpha=0.5, c = cmap(norm(np.log10(ws[s]))))

ax.set_ylim([-3, -1.0])
ax.set_xlim([2.5, 0.0])

ax.set_xlabel(r'$\rm log_{10}(H\beta\ EW/\AA)$')
ax.set_ylabel(r'$\rm \beta$')

cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
cmap.set_array([])
cbar = fig.colorbar(cmap, cax=cax)
cbar.ax.tick_params(labelsize=6)
cbar.set_label(r'$\rm weight$', fontsize=8)


fig.savefig(f'figs/beta_HbEW_weights.pdf')
fig.clf()
