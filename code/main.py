from feature import *
from learning import *
from config import *
from structure import *
from multiprocessing import Pool

import numpy as np
import os

def main():
    id_list = get_id_list(data_dir)

    # get feature
    if USE_MULTIPROCESS:
        pool = Pool(10)
        pool.map(batch_handle, split_list(id_list))
    else:
        batch_handle(id_list)
    
    learning_cv(data_dir)

if __name__ == '__main__':
    main()