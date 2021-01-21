import os
import numpy as np
from config import *
from ripser import Rips, ripser

# poscar to numpy array
def get_prim_structure_info(data_dir, id):
    # need to change if predict icsd data
    with open(data_dir + '/structure/' + id, 'r') as f:
        lines = f.read().splitlines()
    atom_map = []
    index_up = 0
    for atom, num in zip(lines[5].split(), lines[6].split()):
        index_up += int(num)
        atom_map.append((atom, index_up))
    lattice_vec = []
    for line in lines[2:5]:
        x, y, z = line.split()
        lattice_vec.append([float(x), float(y), float(z)])
    lattice_vec = np.array(lattice_vec)
    #get atom position
    index_atom = 0
    atom_nums = len(lines[8:])
    atom_vec = np.zeros([atom_nums], dtype=dt)
    for i in range(atom_nums):
        line = lines[8+i]
        x, y, z = line.split()
        if i < atom_map[index_atom][1]:
            atom_vec[i]['typ'] = atom_map[index_atom][0]
        else:
            index_atom += 1
            atom_vec[i]['typ'] = atom_map[index_atom][0]
        atom_vec[i]['pos'][:] = np.array([float(x), float(y), float(z)])
    
    if not os.path.exists(data_dir + "/atoms"):
        os.makedirs(data_dir + "/atoms")

    with open(data_dir + '/atoms/' + id + '_original.npz', 'wb') as out_file:
        np.savez(out_file, lattice_vec=lattice_vec, atom_vec=atom_vec)


# enlarge the unit cell to each atom in unit cell can form a ball with radius value cut
def enlarge_cell(data_dir, id):
    with open(data_dir + '/atoms/' + id + '_original.npz', 'rb') as structfile:
        data = np.load(structfile)
        lattice_vec = data['lattice_vec']; atom_vec = data['atom_vec']
    min_lattice = min([np.linalg.norm(i) for i in lattice_vec])
    mul_time = int(np.ceil(cut/min_lattice))
    center_atom_vec = atom_vec.copy()
    center_atom_vec['pos'][:] += mul_time
    center_atom_vec['pos'][:] = np.matmul(center_atom_vec['pos'][:], lattice_vec)

    enlarge_dict = {}
    atom_nums = 0
    for atom in atom_vec:
        typ = atom['typ']
        tmp = []
        if typ not in enlarge_dict:
            enlarge_dict[typ] = []
        for i in range(mul_time*2 + 2):
            tmp.append(atom['pos'][0]+i)
            for j in range(mul_time*2 + 2):
                tmp.append(atom['pos'][1]+j)
                for k in range(mul_time*2 + 2):
                    tmp.append(atom['pos'][2]+k)
                    point = tmp.copy()
                    if point not in enlarge_dict[typ]:
                        enlarge_dict[typ].append(point)
                        atom_nums += 1
                    tmp.pop()
                tmp.pop()
            tmp.pop()
    # print(enlarge_dict, atom_nums)
    enlarge_vec = np.zeros([atom_nums], dtype=dt)
    cart_enlarge_vec = np.zeros([atom_nums], dtype=dt)
    atom_index = 0
    for typ, vec in enlarge_dict.items():
        for v in vec:
            enlarge_vec[atom_index]['typ'] = typ
            cart_enlarge_vec[atom_index]['typ'] = typ
            enlarge_vec[atom_index]['pos'][:] = np.array(v)
            cart_enlarge_vec[atom_index]['pos'][:] = np.matmul(np.array(v), lattice_vec)
            atom_index += 1
    with open(data_dir + '/atoms/' + id + '_enlarge.npz', 'wb') as out_file:
        np.savez(out_file, CAV=center_atom_vec, CEV=cart_enlarge_vec)


# betti number for one structure
def get_betti_num(data_dir, id):
    if os.path.exists(data_dir + '/betti_num/' + id):
        return "exists"
    with open(data_dir + '/atoms/' + id + '_enlarge.npz', 'rb') as structfile:
        data = np.load(structfile)
        center_atom_vec=data['CAV']; cart_enlarge_vec=data['CEV']
    typ_dict = {}
    for vec in center_atom_vec:
        typ = vec['typ'].decode()
        if typ not in typ_dict:
            typ_dict[typ] = 1
        else:
            typ_dict[typ] += 1
    
    # for every atom in center_atom
    # first calculate it neighbor atom with same element
    # then with other element within distance cut
    if not os.path.exists(data_dir + "/betti_num"):
        os.makedirs(data_dir + "/betti_num")

    out_File = open(data_dir + '/betti_num/' + id, 'w')
    for cav in center_atom_vec:
        center_atom_type = cav['typ'].decode()
        for ele in typ_dict.keys():
            # get atom postion in each pair
            pair_index = []
            for i in range(len(cart_enlarge_vec)):
                vec = cart_enlarge_vec[i]
                # make change
                if (vec['typ'].decode() == ele or vec['typ'].decode() == center_atom_type) and np.linalg.norm(vec['pos'][:] - cav['pos'][:]) <= cut:
                    pair_index.append(i)
            if len(pair_index) == 0:
                continue
            points_num = len(pair_index)
            pair_pos = np.zeros((points_num+1, 3))
            pair_pos[0][:] = cav['pos'][:]
            index = 1
            for i in pair_index:
                atom = cart_enlarge_vec[i]['pos']
                pair_pos[index][:] = np.array([atom[0], atom[1], atom[2]])
                index += 1
            
            # calculate barcode
            dgms = ripser(pair_pos, maxdim=2, thresh=cut)['dgms']
            for i, dgm in enumerate(dgms):
                for p in dgm:
                    out_File.write(center_atom_type+ele+' '+str(i)+' '+str(p[0])+' '+str(p[1])+'\n')
    out_File.close()