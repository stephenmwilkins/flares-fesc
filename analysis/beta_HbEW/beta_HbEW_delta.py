import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt

import flares
import flare.plt as fplt


deltas = np.array([0.969639,0.918132,0.851838,0.849271,0.845644,0.842128,0.841291,0.83945,0.838891,0.832753,0.830465,0.829349,0.827842,0.824159,0.821425,0.820476,0.616236,0.616012,0.430745,0.430689,0.266515,0.266571,0.121315,0.121147,-0.007368,-0.007424,-0.121207,-0.121319,-0.222044,-0.222156,-0.311441,-0.311329,-0.066017,-0.066185,-0.00748,-0.007424,0.055076,0.054909,-0.47874,-0.433818])


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
delta = np.array([])

for ii in range(len(halo)):
    ws = np.append(ws, np.ones(np.shape(d[q][halo[ii]][tag]))*weights[ii])
    delta = np.append(delta, np.ones(np.shape(d[q][halo[ii]][tag]))*deltas[ii])
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

cmap = cm.cividis

fig, ax, cax = fplt.single_wcbar(base_size=3.5)

norm = mpl.colors.Normalize(vmin=-1, vmax=1)
ax.scatter(np.log10(D['HbetaEW'][s]), beta[s], s=1, alpha=0.5, c = cmap(norm(delta[s])))

ax.set_ylim([-3, -1.0])
ax.set_xlim([2.5, 0.0])

ax.set_xlabel(r'$\rm log_{10}(H\beta\ EW/\AA)$')
ax.set_ylabel(r'$\rm \beta$')

cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
cmap.set_array([])
cbar = fig.colorbar(cmap, cax=cax)
cbar.ax.tick_params(labelsize=6)
cbar.set_label(r'$\rm delta$', fontsize=8)


fig.savefig(f'figs/beta_HbEW_delta.pdf')
fig.clf()
