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





for spec_type in ['DustModelI',  'Intrinsic']:


    quantities = []
    quantities.append(['Galaxy', 'SFR_inst_30'])
    quantities.append(['Galaxy', 'Mstar_30'])
    quantities.append([f'Galaxy/BPASS_2.2.1/Chabrier300/Lines/{spec_type}/HI4861', 'EW', 'HbetaEW'])
    quantities.append([f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/{spec_type}', 'FUV'])
    quantities.append([f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/{spec_type}', 'NUV'])
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

    D['Mstar_30'] *= 1E10
    s = np.log10(D['Mstar_30'])>8.5

    sSFR = np.log10(D['SFR_inst_30'])-np.log10(D['Mstar_30'])+9

    print(np.min(sSFR),np.max(sSFR))

    beta = np.log10(D['FUV']/D['NUV'])/np.log10(1500/2500)-2.0

    # --- print quantity medians

    # for q in qs:
    #     print(q, D[q].shape, np.median(D[q]))

    # --- print quantity medians for M*>1E8
    for q in qs:
        print(q, D[q][s].shape, np.median(D[q][s]), np.std(D[q][s]))

    print(np.median(beta[s]), np.std(beta[s]))


    cmap = cm.Spectral

    fig, ax, cax = fplt.single_wcbar(base_size=3.5)

    norm = mpl.colors.Normalize(vmin=-0.5, vmax=1.5)
    ax.scatter(np.log10(D['HbetaEW'][s]), beta[s], s=1, alpha=0.5, c = cmap(norm(sSFR[s])))

    ax.set_ylim([-3, -1.0])
    ax.set_xlim([2.5, 0.0])

    ax.set_xlabel(r'$\rm log_{10}(H\beta\ EW/\AA)$')
    ax.set_ylabel(r'$\rm \beta$')

    cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
    cmap.set_array([])
    cbar = fig.colorbar(cmap, cax=cax)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(r'$\log_{10}({\rm sSFR}/{\rm Gyr^{-1})}$', fontsize=8)


    fig.savefig(f'figs/beta_HbEW_sSFR_{spec_type}.pdf')
    fig.clf()


# bins = np.arange(0, 4, 0.5)
# bincen = (bins[:-1]+bins[1:])/2.
# out = flares.binned_weighted_quantile(x,y,ws,bins,quantiles)
#
#
# fig = plt.figure(figsize=(3,3))
#
# left  = 0.2
# bottom = 0.2
# width = 0.75
# height = 0.75
#
# ax = fig.add_axes((left, bottom, width, height))
#
#
# ax.legend()
#
# ax.set_xlabel(r'$\rm log_{10}(f_{H}/nJy)$')
# ax.set_ylabel(r'$\rm log_{10}(N(>z))$')
#
# fig.savefig(f'figs/beta_HbEW.pdf')
# fig.clf()
