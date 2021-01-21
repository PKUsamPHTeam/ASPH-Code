import numpy as np
import pandas as pd
import os
import math
import re
from structure import get_betti_num
from config import *

# composition feature
def get_feature_composition(data_dir, id):
    with open(data_dir + '/atoms/' + id + '_enlarge.npz', 'rb') as structfile:
        data = np.load(structfile)
        center_atom_vec=data['CAV']
    typ_dict = {}
    for vec in center_atom_vec:
        typ = vec['typ'].decode()
        if typ not in typ_dict:
            typ_dict[typ] = 1
        else:
            typ_dict[typ] += 1

    # get element properties
    element_properties = pd.read_csv(data_dir + '/element_properties.csv')
    element_properties.set_index('Abbr', inplace=True)
    # element_properties.drop(columns=['NfUnfilled', 'NfValence'])

    Feature = []; tmp_array = []
    for ele, n in typ_dict.items():
        ele_list = list(element_properties.loc[ele])
        for i in range(n):
            tmp_array.append(ele_list)

    Feature.append(np.mean(tmp_array, axis=0))
    Feature.append(np.std(tmp_array, axis=0))
    Feature.append(np.sum(tmp_array, axis=0))
    Feature.append(np.max(tmp_array, axis=0))
    Feature.append(np.min(tmp_array, axis=0))

    Feature_1 = np.asarray(Feature, float)
    Feature_1 = np.concatenate(Feature_1, axis=0)

    if not os.path.exists(data_dir + "/feature_composition"):
        os.makedirs(data_dir + "/feature_composition")

    with open(data_dir + '/feature_composition/' + id + '_feature.npy', 'wb') as outfile:
        np.save(outfile, Feature_1)


def get_feature_with_s_nobin(data_dir, id):
    # get atoms frequencey
    atom_single = atoms_frequency()
    common_pair = {}
    i = 0
    for key in sorted(atom_single, key=atom_single.__getitem__,reverse=True):
        common_pair[key] = i
        i += 1
    com_len = len(common_pair)
    with open(data_dir + '/atoms/' + id + '_enlarge.npz', 'rb') as structfile:
        data = np.load(structfile)
        center_atom_vec=data['CAV']
    typ_dict = {}
    ele_Barcode = {}
    for vec in center_atom_vec:
        typ = vec['typ'].decode()
        if typ not in typ_dict:
            typ_dict[typ] = 1
        else:
            typ_dict[typ] += 1
        if typ in common_pair and typ not in ele_Barcode:
            ele_Barcode[typ] = [[], [], [], [], []] # Bar0Death Bar1Birth Bar1Death Bar2Birth Bar2Death
    
    # get structure feature
    with open(data_dir + '/betti_num/' + id, 'r') as phfile:
        lines = phfile.read().splitlines()

    if lines == []:
        get_betti_num(data_dir, id)
    pair_dict = {}
    pair_i = 0
    for line in lines:
        typ = line.split()[0]
        if typ not in pair_dict:
            pair_dict[typ] = pair_i
            pair_i += 1

    # Barcode without bin
    Bar0Death = []; Bar1Birth = []; Bar1Death = []; Bar2Birth = []; Bar2Death = []
    WBar0Death = []; WBar1Birth = []; WBar1Death = []; WBar2Birth = []; WBar2Death = []
    for line in lines:
        typ, dim, birth, death = line.split()
        center_atom = typ[0:2] if typ[1].islower() else typ[0]
        ca_num = typ_dict[center_atom]

        dim = int(dim); birth = float(birth); death = float(death)
        # Birth
        if dim == 1:
            Bar1Birth.append(birth)
            WBar1Birth.append(birth/ca_num)
        elif dim == 2:
            Bar2Birth.append(birth)
            WBar2Birth.append(birth/ca_num)
        # Death
        if death == float('inf'): continue
        if dim == 0:
            Bar0Death.append(death)
            WBar0Death.append(death/ca_num)
        elif dim == 1:
            Bar1Death.append(death)
            WBar1Death.append(death/ca_num)
        elif dim == 2:
            Bar2Death.append(death)
            WBar2Death.append(death/ca_num)

    Bar0Death = np.asarray(Bar0Death); Bar1Birth = np.asarray(Bar1Birth); Bar1Death = np.asarray(Bar1Death); Bar2Birth = np.asarray(Bar2Birth); Bar2Death = np.asarray(Bar2Death); 
    WBar0Death = np.asarray(WBar0Death); WBar1Birth = np.asarray(WBar1Birth); WBar1Death = np.asarray(WBar1Death); WBar2Birth = np.asarray(WBar2Birth); WBar2Death = np.asarray(WBar2Death); 
    Feature_2 = []
    # Betti0
    if len(Bar0Death) > 0:
        Feature_2.append(np.mean(Bar0Death, axis=0))
        Feature_2.append(np.std(Bar0Death, axis=0))
        Feature_2.append(np.max(Bar0Death, axis=0))
        Feature_2.append(np.min(Bar0Death, axis=0))
        Feature_2.append(np.sum(WBar0Death, axis=0))
        # Feature_2.append(len(Bar0Death))
    else:
        Feature_2.extend([0.]*5)
    # Betti1
    if len(Bar1Death) > 0:
        Feature_2.append(np.mean(Bar1Death - Bar1Birth, axis=0))
        Feature_2.append(np.std(Bar1Death - Bar1Birth, axis=0))
        Feature_2.append(np.max(Bar1Death - Bar1Birth, axis=0))
        Feature_2.append(np.min(Bar1Death - Bar1Birth, axis=0))
        Feature_2.append(np.sum(WBar1Death - WBar1Birth, axis=0))
        Feature_2.append(np.mean(Bar1Birth, axis=0))
        Feature_2.append(np.std(Bar1Birth, axis=0))
        Feature_2.append(np.max(Bar1Birth, axis=0))
        Feature_2.append(np.min(Bar1Birth, axis=0))
        Feature_2.append(np.sum(WBar1Birth, axis=0))
        Feature_2.append(np.mean(Bar1Death, axis=0))
        Feature_2.append(np.std(Bar1Death, axis=0))
        Feature_2.append(np.max(Bar1Death, axis=0))
        Feature_2.append(np.min(Bar1Death, axis=0))
        Feature_2.append(np.sum(WBar1Death, axis=0))
        # Feature_2.append(len(Bar1Death))
    else:
        Feature_2.extend([0.]*15)
    # Betti2
    if len(Bar2Death) > 0:
        Feature_2.append(np.mean(Bar2Death - Bar2Birth, axis=0))
        Feature_2.append(np.std(Bar2Death - Bar2Birth, axis=0))
        Feature_2.append(np.max(Bar2Death - Bar2Birth, axis=0))
        Feature_2.append(np.min(Bar2Death - Bar2Birth, axis=0))
        Feature_2.append(np.sum(WBar2Death - WBar2Birth, axis=0))
        Feature_2.append(np.mean(Bar2Birth, axis=0))
        Feature_2.append(np.std(Bar2Birth, axis=0))
        Feature_2.append(np.max(Bar2Birth, axis=0))
        Feature_2.append(np.min(Bar2Birth, axis=0))
        Feature_2.append(np.sum(WBar2Birth, axis=0))
        Feature_2.append(np.mean(Bar2Death, axis=0))
        Feature_2.append(np.std(Bar2Death, axis=0))
        Feature_2.append(np.max(Bar2Death, axis=0))
        Feature_2.append(np.min(Bar2Death, axis=0))
        Feature_2.append(np.sum(WBar2Death, axis=0))
        # Feature_2.append(len(Bar2Death))
    else:
        Feature_2.extend([0.]*15)
    Feature_2 = np.asarray(Feature_2, float)

    # Barcode for each single element and element pair
    Feature_3 = np.zeros([com_len, 7*5], float)
    with open(data_dir + '/betti_num/' + id, 'r') as phfile:
        lines = phfile.read().splitlines()
    for line in lines:
        typ, dim, birth, death = line.split()
        center_atom = typ[0:2] if typ[1].islower() else typ[0]
        ca_num = typ_dict[center_atom]

        dim = int(dim); birth = float(birth); death = float(death)
        if center_atom in common_pair:
            # Birth
            if dim == 1:
                ele_Barcode[center_atom][1].append(birth)
            elif dim == 2:
                ele_Barcode[center_atom][3].append(birth)
            # Death
            if death == float('inf'): continue
            if dim == 0:
                ele_Barcode[center_atom][0].append(death)
            elif dim == 1:
                ele_Barcode[center_atom][2].append(death)
            elif dim == 2:
                ele_Barcode[center_atom][4].append(death)

    for key in ele_Barcode.keys():
        ele_Barcode[key] = [np.asarray(bar) for bar in ele_Barcode[key]]
    for ele, bar in ele_Barcode.items():
        ca_num = typ_dict[ele]
        if len(bar[0]) > 0:
            Feature_3[common_pair[ele],0] = np.mean(bar[0], axis=0)
            Feature_3[common_pair[ele],1] = np.std(bar[0], axis=0)
            Feature_3[common_pair[ele],2] = np.max(bar[0], axis=0)
            Feature_3[common_pair[ele],3] = np.min(bar[0], axis=0)
            Feature_3[common_pair[ele],4] = np.sum(bar[0], axis=0)/ca_num
        if len(bar[2]) > 0:
            Feature_3[common_pair[ele],5] = np.mean(bar[2] - bar[1], axis=0)
            Feature_3[common_pair[ele],6] = np.std(bar[2] - bar[1], axis=0)
            Feature_3[common_pair[ele],7] = np.max(bar[2] - bar[1], axis=0)
            Feature_3[common_pair[ele],8] = np.min(bar[2] - bar[1], axis=0)
            Feature_3[common_pair[ele],9] = np.sum(bar[2] - bar[1], axis=0)/ca_num
            Feature_3[common_pair[ele],10] = np.mean(bar[1], axis=0)
            Feature_3[common_pair[ele],11] = np.std(bar[1], axis=0)
            Feature_3[common_pair[ele],12] = np.max(bar[1], axis=0)
            Feature_3[common_pair[ele],13] = np.min(bar[1], axis=0)
            Feature_3[common_pair[ele],14] = np.sum(bar[1], axis=0)/ca_num
            Feature_3[common_pair[ele],15] = np.mean(bar[2], axis=0)
            Feature_3[common_pair[ele],16] = np.std(bar[2], axis=0)
            Feature_3[common_pair[ele],17] = np.max(bar[2], axis=0)
            Feature_3[common_pair[ele],18] = np.min(bar[2], axis=0)
            Feature_3[common_pair[ele],19] = np.sum(bar[2], axis=0)/ca_num
        if len(bar[4]) > 0:
            Feature_3[common_pair[ele],20] = np.mean(bar[4] - bar[3], axis=0)
            Feature_3[common_pair[ele],21] = np.std(bar[4] - bar[3], axis=0)
            Feature_3[common_pair[ele],22] = np.max(bar[4] - bar[3], axis=0)
            Feature_3[common_pair[ele],23] = np.min(bar[4] - bar[3], axis=0)
            Feature_3[common_pair[ele],24] = np.sum(bar[4] - bar[3], axis=0)/ca_num
            Feature_3[common_pair[ele],25] = np.mean(bar[3], axis=0)
            Feature_3[common_pair[ele],26] = np.std(bar[3], axis=0)
            Feature_3[common_pair[ele],27] = np.max(bar[3], axis=0)
            Feature_3[common_pair[ele],28] = np.min(bar[3], axis=0)
            Feature_3[common_pair[ele],29] = np.sum(bar[3], axis=0)/ca_num
            Feature_3[common_pair[ele],30] = np.mean(bar[4], axis=0)
            Feature_3[common_pair[ele],31] = np.std(bar[4], axis=0)
            Feature_3[common_pair[ele],32] = np.max(bar[4], axis=0)
            Feature_3[common_pair[ele],33] = np.min(bar[4], axis=0)
            Feature_3[common_pair[ele],34] = np.sum(bar[4], axis=0)/ca_num

    Feature_3 = np.concatenate(Feature_3, axis=0)
    Feature = np.concatenate((Feature_2, Feature_3), axis=0)
    # print(Feature.shape)
    if not os.path.exists(data_dir + "/feature_add_s_nobin"):
        os.makedirs(data_dir + "/feature_add_s_nobin")

    with open(data_dir + '/feature_add_s_nobin/' + id + '_feature.npy', 'wb') as outfile:
        np.save(outfile, Feature)

def get_feature_with_s_nobin_Bar0(data_dir, id):
    atom_single = atoms_frequency()
    common_pair = {}
    i = 0
    for key in sorted(atom_single, key=atom_single.__getitem__,reverse=True):
        common_pair[key] = i
        i += 1
    com_len = len(common_pair)
    with open(data_dir + '/atoms/' + id + '_enlarge.npz', 'rb') as structfile:
        data = np.load(structfile)
        center_atom_vec=data['CAV']
    typ_dict = {}
    ele_Barcode = {}
    for vec in center_atom_vec:
        typ = vec['typ'].decode()
        if typ not in typ_dict:
            typ_dict[typ] = 1
        else:
            typ_dict[typ] += 1
        if typ in common_pair and typ not in ele_Barcode:
            ele_Barcode[typ] = [] # Bar0Death
    
    # get structure feature
    with open(data_dir + '/betti_num/' + id, 'r') as phfile:
        lines = phfile.read().splitlines()

    if lines == []:
        get_betti_num(data_dir, id)
    pair_dict = {}
    pair_Barcode = {}
    pair_i = 0
    for line in lines:
        typ = line.split()[0]
        if typ not in pair_dict:
            pair_dict[typ] = pair_i
            pair_i += 1
    
    # Barcode without bin
    Bar0Death = []; WBar0Death = []
    for line in lines:
        typ, dim, birth, death = line.split()
        center_atom = typ[0:2] if typ[1].islower() else typ[0]
        ca_num = typ_dict[center_atom]

        dim = int(dim); birth = float(birth); death = float(death)
        # Death
        if death == float('inf'): continue
        if dim == 1:
            Bar0Death.append(birth)
            WBar0Death.append(birth/ca_num)

    Bar0Death = np.asarray(Bar0Death); WBar0Death = np.asarray(WBar0Death)
    Feature_2 = []
    # Betti0
    if len(Bar0Death) > 0:
        Feature_2.append(np.mean(Bar0Death, axis=0))
        Feature_2.append(np.std(Bar0Death, axis=0))
        Feature_2.append(np.max(Bar0Death, axis=0))
        Feature_2.append(np.min(Bar0Death, axis=0))
        Feature_2.append(np.sum(WBar0Death, axis=0))
    else:
        Feature_2.extend([0.]*5)
    Feature_2 = np.asarray(Feature_2, float)

    # Barcode for each single element and element pair
    Feature_3 = np.zeros([com_len, 5], float)
    with open(data_dir + '/betti_num/' + id, 'r') as phfile:
        lines = phfile.read().splitlines()
    for line in lines:
        typ, dim, birth, death = line.split()
        center_atom = typ[0:2] if typ[1].islower() else typ[0]
        ca_num = typ_dict[center_atom]

        dim = int(dim); birth = float(birth); death = float(death)
        if center_atom in common_pair:
            # Death
            if death == float('inf'): continue
            if dim == 0:
                ele_Barcode[center_atom].append(death)

    for key in ele_Barcode.keys():
        ele_Barcode[key] = [np.asarray(bar) for bar in ele_Barcode[key]]
    for ele, bar in ele_Barcode.items():
        ca_num = typ_dict[ele]
        if len(bar) > 0:
            Feature_3[common_pair[ele],0] = np.mean(bar, axis=0)
            Feature_3[common_pair[ele],1] = np.std(bar, axis=0)
            Feature_3[common_pair[ele],2] = np.max(bar, axis=0)
            Feature_3[common_pair[ele],3] = np.min(bar, axis=0)
            Feature_3[common_pair[ele],4] = np.sum(bar, axis=0)/ca_num

    Feature_3 = np.concatenate(Feature_3, axis=0)
    # print(Feature_3.shape)
    Feature = np.concatenate((Feature_2, Feature_3), axis=0)
    if not os.path.exists(data_dir + "/feature_add_s_nobin_Bar0"):
        os.makedirs(data_dir + "/feature_add_s_nobin_Bar0")

    with open(data_dir + '/feature_add_s_nobin_Bar0/' + id + '_feature.npy', 'wb') as outfile:
        np.save(outfile, Feature)

# topological feature plus composition feature
def get_feature_topo_compo(data_dir, id):
    name = id + "_feature.npy"

    if not os.path.exists(data_dir + '/feature_add_s_nobin/' + name):
        get_feature_with_s_nobin(data_dir, id)

    with open(data_dir + '/feature_add_s_nobin/' + name, "rb") as f:
        feature_topo = np.load(f)

    if not os.path.exists(data_dir + '/feature_composition/' + name):
        get_feature_composition(data_dir, id)

    with open(data_dir + '/feature_composition/' + name, "rb") as f:
        feature_compo = np.load(f)

    Feature = np.concatenate((feature_compo, feature_topo), axis=0)

    if not os.path.exists(data_dir + "/feature_topo_compo"):
        os.makedirs(data_dir + "/feature_topo_compo")

    with open(data_dir + "/feature_topo_compo/" + name, "wb") as outfile:
        np.save(outfile, Feature)

def atoms_frequency():
    sub_dict = split_element(data_dir)
    # print(sub_dict)
    data_len = len(sub_dict)
    atom_single = {}
    for sub in sub_dict.keys():
        for i in sub_dict[sub]:
            if i not in atom_single:
                atom_single[i] = 0.0
            atom_single[i] += 1.0
    return atom_single
    # f = open(data_dir + '/atoms_frequecy.txt', 'w')
    # for key in sorted(atom_single, key=atom_single.__getitem__,reverse=True):
    #     f.write(key + " " + str(int(atom_single[key])) + "\n")

    # f.close()

def split_element(data_dir):
    pattern = re.compile("[A-Z]{1}[a-z]{0,1}")
    # substance to all pair
    sub_dict = {}
    ele_set = set()
    with open(data_dir + '/properties.txt', 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        sub = line.split()[0]
        with open(data_dir + "/structure/" + sub, 'r') as tmp:
            ls = tmp.read().splitlines()[0]
        eles = set(pattern.findall(ls))
        sub_dict[sub] = eles
    return sub_dict