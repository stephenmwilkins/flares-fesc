import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt

import flares
import flare.plt as fplt

# ----------------------------------------------------------------------
# --- open data

deltas = np.array([0.969639,0.918132,0.851838,0.849271,0.845644,0.842128,0.841291,0.83945,0.838891,0.832753,0.830465,0.829349,0.827842,0.824159,0.821425,0.820476,0.616236,0.616012,0.430745,0.430689,0.266515,0.266571,0.121315,0.121147,-0.007368,-0.007424,-0.121207,-0.121319,-0.222044,-0.222156,-0.311441,-0.311329,-0.066017,-0.066185,-0.00748,-0.007424,0.055076,0.054909,-0.47874,-0.433818])

fl = flares.flares('/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5', sim_type='FLARES')
halo = fl.halos



# ----------------------------------------------------------------------
# --- define parameters

tag = fl.tags[-3]  # --- select tag -3 = z=7
spec_type = 'DustModelI' # --- select photometry type


# ----------------------------------------------------------------------
# --- define quantities to read in [not those for the corner plot, that's done later]

quantities = []
quantities.append(['Galaxy', 'SFR_inst_30']) # needed for sSFR
quantities.append(['Galaxy', 'Mstar_30']) # needed for selection
quantities.append([f'Galaxy/BPASS_2.2.1/Chabrier300/Lines/{spec_type}/HI4861', 'EW', 'HbetaEW'])
quantities.append([f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic', 'FUV', 'IntrinsicFUV']) # needed for AFUV
quantities.append([f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/{spec_type}', 'FUV']) # needed for beta
quantities.append([f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/{spec_type}', 'NUV']) # needed for beta
# quantities.append()
# quantities.append()

# ----------------------------------------------------------------------
# --- get all the required quantities, including the weights and delta

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
D['log10Mstar_30'] = np.log10(D['Mstar_30'])

# ----------------------------------------------
# define selection
s = D['log10Mstar_30']>8.5

# ----------------------------------------------
# derive new quantities
D['log10HbetaEW'] = np.log10(D['HbetaEW'])
D['log10FUV'] = np.log10(D['FUV'])
D['beta'] = np.log10(D['FUV']/D['NUV'])/np.log10(1500/2500)-2.0
D['log10sSFR'] = np.log10(D['SFR_inst_30'])-np.log10(D['Mstar_30'])+9
D['AFUV'] = -2.5*np.log10(D['FUV']/D['IntrinsicFUV'])

# ----------------------------------------------
# Print info
print(f'Total number of galaxies: {len(ws[s])}')







# ----------------------------------------------
# define quantities for including in the corner plot

properties = ['log10Mstar_30', 'log10sSFR', 'log10FUV',  'AFUV', 'beta', 'log10HbetaEW']

labels = {}
labels['log10Mstar_30'] = r'\log_{10}({\rm M_{\star}}/{\rm M_{\odot})}'
labels['beta'] = r'\beta'
labels['log10sSFR'] = r'\log_{10}({\rm sSFR}/{\rm Gyr^{-1})}'
labels['log10HbetaEW'] = r'\log_{10}(H\beta\ EW/\AA)'
labels['log10FUV'] = r'\log_{10}(L_{FUV}/erg\ s^{-1}\ Hz^{-1})'
labels['AFUV'] = r'A_{FUV}'

limits = {}
limits['log10Mstar_30'] = [8.5,11]
limits['beta'] = [-2.9,-1.1]
limits['log10sSFR'] = [-0.9,1.9]
limits['log10HbetaEW'] = [0.01,2.49]
limits['AFUV'] = [0,3.9]
limits['log10FUV'] = [28.1,29.9]

# ----------------------------------------------
# ----------------------------------------------
# Corner Plot


norm = mpl.colors.Normalize(vmin=8.5, vmax=10.5)
cmap = cm.viridis

N = len(properties)

fig, axes = plt.subplots(N, N, figsize = (7,7))
plt.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.02, hspace=0.02)

for i in np.arange(N):
    for j in np.arange(N):
        axes[i, j].set_axis_off()

for i,x in enumerate(properties):
    for j,y in enumerate(properties[1:][::-1]):

        jj = N-1-j
        ii = i

        ax = axes[jj, ii]

        if j+i<(N-1):
            ax.set_axis_on()

            # plot here
            ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = cmap(norm(D['log10Mstar_30'][s])))
            ax.set_xlim(limits[x])
            ax.set_ylim(limits[y])

        if i == 0: # first column
            ax.set_ylabel(rf'$\rm {labels[y]}$', fontsize = 7)
        else:
            ax.yaxis.set_ticklabels([])

        if j == 0: # first row
            ax.set_xlabel(rf'$\rm {labels[x]}$', fontsize = 7)
        else:
            ax.xaxis.set_ticklabels([])

        # ax.text(0.5, 0.5, f'x{i}-y{j}', transform = ax.transAxes)


    # --- histograms

    bins = 50

    ax = axes[ii, ii]
    ax.set_axis_on()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    X = D[x][s]

    H, bin_edges = np.histogram(X, bins = bins, range = limits[x])
    Hw, bin_edges = np.histogram(X, bins = bins, range = limits[x], weights = ws[s])

    Hw *= np.max(H)/np.max(Hw)

    bin_centres = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])*0.5


    ax.fill_between(bin_centres, H*0.0, H, color='0.9')
    ax.plot(bin_centres, Hw, c='0.7', lw=1)

    ax.set_ylim([0.0,np.max(H)*1.2])





# --- add colourbar

cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
cmapper.set_array([])

cax = fig.add_axes([0.25, 0.87, 0.5, 0.015])
fig.colorbar(cmapper, cax=cax, orientation='horizontal')
cax.set_xlabel(r'$\rm\log_{10}({\rm M_{\star}}/{\rm M_{\odot})}$')



fig.savefig(f'figs/corner.pdf')
