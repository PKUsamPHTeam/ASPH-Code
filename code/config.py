import numpy as np

small = 0.0001
cut = 12.0
rs = 0.25
icsd_dir = "/udata/yjiang/Topology_ML/data/cutoff_" + str(int(cut))
dt = np.dtype([('typ', 'S2'), ('pos', float, (3, ))])
lth = int(np.rint(cut/rs))
# feature name
fname = "feature_topo_compo"

USE_MULTIPROCESS = True
