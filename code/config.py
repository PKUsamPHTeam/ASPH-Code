import numpy as np

small = 0.0001
cut = 8.0
rs = 0.25
# change it to your data directory
data_dir = "../data"
dt = np.dtype([('typ', 'S2'), ('pos', float, (3, ))])
lth = int(np.rint(cut/rs))
# feature name
fname = "feature_topo_compo"

USE_MULTIPROCESS = True
