import numpy as np
import pandas as pd
import os
import math
from betti_num import get_betti_num
from resource import *

def get_feature_withoutbin(data_dir, id):
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
    Feature_1 = []; tmp_array = []
    for ele, n in typ_dict.items():
        ele_list = list(element_properties.loc[ele])
        for i in range(n):
            tmp_array.append(ele_list)
    Feature_1.append(np.mean(tmp_array, axis=0))
    Feature_1.append(np.std(tmp_array, axis=0))
    # Feature_1.append(np.median(tmp_array, axis=0))
    Feature_1.append(np.sum(tmp_array, axis=0))
    Feature_1 = np.asarray(Feature_1, float)
    Feature_1 = np.concatenate(Feature_1, axis=0)
    with open(data_dir + '/feature_element/' + id + '_feature.npy', 'wb') as outfile:
        np.save(outfile, Feature_1)

    # get structure feature
    with open(data_dir + '/betti_num/' + id, 'r') as phfile:
        lines = phfile.read().splitlines()
    if lines == []:
        get_betti_num(data_dir, id)
    pair_dict = {}
    pair_i = 0
    for line in lines:
        typ, dim, birth, death = line.split()
        dim = int(dim); death = float(death)
        if typ not in pair_dict:
            pair_dict[typ] = pair_i
            pair_i += 1


def get_feature_with_s_nobin(data_dir, id):
    with open(data_dir + '/atom_single_sorted', 'r') as sf:
        slines = sf.read().splitlines()
    common_pair = {}
    i = 0
    for line in slines:
        s, n = line.split()
        if int(n) < 100:
            break
        common_pair[s] = i
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
    try:
        with open(data_dir + '/betti_num/' + id, 'r') as phfile:
            lines = phfile.read().splitlines()
    except:
        get_betti_num(data_dir, id)
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
    with open(data_dir + '/feature_add_s_nobin/' + id + '_feature.npy', 'wb') as outfile:
        np.save(outfile, Feature)

def get_feature_with_s_nobin_Bar0(data_dir, id):
    with open(data_dir + '/atom_single_sorted', 'r') as sf:
        slines = sf.read().splitlines()
    common_pair = {}
    i = 0
    for line in slines:
        s, n = line.split()
        # if int(n) < 100:
        #     break
        common_pair[s] = i
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
    with open(data_dir + '/feature_add_s_nobin_Bar0/' + id + '_feature.npy', 'wb') as outfile:
        np.save(outfile, Feature)

# 拓扑+晶体组成的性质
def get_feature_topo_compo(icsd_dir):
    with open("11.txt", 'r') as f:
        lines = f.read().splitlines()
    id_list = []
    for l in lines:
        id_list.append(l.split()[0])
    
    csv_data = pd.read_csv(icsd_dir + '/icsd-all.csv')
    csv_data.fillna(0, inplace=True)
    
    # 组成性质从第127个开始
    columns_list = csv_data.columns.values.tolist()[126:-1]
    compo_prop = np.asarray(csv_data.loc[:, columns_list])

    for i in range(len(id_list)):
        id = id_list[i]
        with open(icsd_dir + '/feature_add_s_nobin/' + id + "_feature.npy", "rb") as f:
            feature = np.load(f)
        Feature = np.concatenate((compo_prop[i], feature), axis=0)

        with open(icsd_dir + "/feature_topo_compo/" + id + "_feature.npy", "wb") as outfile:
            np.save(outfile, Feature)
