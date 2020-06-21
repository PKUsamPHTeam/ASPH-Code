import numpy as np

small = 0.0001
cut = 12.0
rs = 0.25
icsd_dir = "/udata/yjiang/Topology_ML/data/cutoff_" + str(int(cut))
dt = np.dtype([('typ', 'S2'), ('pos', float, (3, ))])
lth = int(np.rint(cut/rs))

ele_list = []
# feature_len = pair_len*lth*9
with open(icsd_dir + '/atom_single_sorted', 'r') as sf:
    slines = sf.read().splitlines()
for line in slines:
    s = line.split()[0]
    if s not in ['Xe', 'Kr', 'He', 'Ne', 'Ar', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Tc']:
        ele_list.append(s)
with open(icsd_dir + '/atom_pair_sorted', 'r') as pf:
    plines = pf.read().splitlines()
common_pair = {}
i = 0
for line in slines:
    s, n = line.split()
    if int(n) < 100:
        break
    common_pair[s] = i
    i += 1
for line in plines:
    p, n = line.split()
    if int(n) < 200:
        break
    # pre_atom = p[0:2] if p[1].islower() else p[0]
    # suf_atom = p[2:] if p[1].islower() else p[1:]
    # if ('O' == pre_atom or 'O' == suf_atom) or ('N' == pre_atom or 'N' == suf_atom):
    #     continue
    common_pair[p] = i
    i += 1
# for e in ele_list:
#     for p in [e+'N', 'N'+e, e+'O', 'O'+e]:
#         if p not in common_pair:
#             common_pair[p] = i
#             i += 1
com_len = len(common_pair)
# print(common_pair)
# print(com_len)

unuse_ele = ['Xe', 'Kr', 'He', 'Ne', 'Ar', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Tc']