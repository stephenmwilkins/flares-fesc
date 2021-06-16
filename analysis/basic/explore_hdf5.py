

import h5py

printname = lambda name: print(name)


filedir = '/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/'

f = h5py.File(filedir + 'flares.hdf5', 'r')

f.visit(printname)
