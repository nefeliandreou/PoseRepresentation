""" Run generate_paired_data_nearest_representatives """
from generate_motion_in_dualquaternions import allbvh2other
from rhythmdata.foot_contact import create_fc_of_dir
import os
import numpy as np
from loadbvh import loadbvh
from dualquats import currentdq2localquats, translation_np
    


def step1(path_to_read, path_to_write, order, qfix=False):
    """
    Step1: Export the quaternions from bvh files!
    :return:
    """
    z = allbvh2other(path_to_read, path_to_write, order)
    z.writeallq(True)


def step2(path_to_read, path_to_write, order):
    """
    Step3 : export ortho6D data
    """
    z = allbvh2other(path_to_read, path_to_write, order)
    z.writeallortho6D()


def step3(path_to_read, path_to_write, order, qfix=False):
    """
    Step3 : export dualquaternion data
    """
    z = allbvh2other(path_to_read, path_to_write, order)
    z.writealldq(qfix)


def step4(path_to_read, path_to_write, order):
    """
    Step3 : export positional data
    """
    z = allbvh2other(path_to_read, path_to_write, order)
    z.writeallpos()

def step6(path_to_read, path_to_write, order):
    """
    Step3 : export positional data
    """
    z = allbvh2other(path_to_read, path_to_write, order)
    z.writeallqpos()
def step7(path_to_read, path_to_write, order):
    """
    Step7 : export axis angle positional data
    """
    z = allbvh2other(path_to_read, path_to_write, order)
    z.writeallaxanglepos()

# def step5(dirname, order,mixamo):
#     create_fc_of_dir(dirname,
#                      dirname + "/FC", order,mixamo=mixamo)

def step8(path_to_read,path_to_write):
    """
        Step7 : export quatpositional31 reduced from dquats
        """
    parents = [None, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 13, 17, 18, 19, 20, 21, 20, 13, 24, 25, 26,
               27, 28, 27]
    for i in [i for i in os.listdir(path_to_read) if i.endswith(".npy")]:
        root, dq = np.load(path_to_read + "/" + i)[:, :3], np.load(path_to_read + "/" + i)[:, 3:]
        localquats = qfix(currentdq2localquats(dq, parents, 31).reshape(-1,31,4)).reshape(-1,31*4)
        positions = translation_np(dq.reshape(-1, 8)).reshape(root.shape[0], 3 * 31)
        np.save(path_to_write + "/" + i, np.concatenate((root,localquats, positions), axis=1))

def step9(path_to_readortho6D,path_to_readpos,path_to_write):
    path_to_readortho6D = "/DATA/Projects/2020/lstm_pro/src/databases/61/ortho6D"
    path_to_readpos = "/DATA/Projects/2020/lstm_pro/src/databases/61/quatspos31"
    path_to_write = "/DATA/Projects/2020/lstm_pro/src/databases/61/ortho6Dpos"
   
    for i in [i for i in os.listdir(path_to_readortho6D) if i.endswith(".npy")]:
        o = np.load(path_to_readortho6D + "/" + i)
        p = np.load(path_to_readpos + "/" + i)[:,127:]
        np.save(path_to_write + "/" + i, np.concatenate((o,p),axis=1))


path_to_read = "/DATA/Projects/2020/lstm_pro/src/databases/86__/"
path_to_write = "/DATA/Projects/2020/lstm_pro/src/databases/61/quatspos31"
qfix = True
order = "zyx"

from common.quaternion import qfix
mixamo=False
# path_to_read ="../../../../../home/nefeli-x/Desktop/testbvh/"

if __name__ == '__main__':
    # step1(path_to_read, path_to_write, order, qfix)
    # step2(path_to_read,path_to_write,order)
    # step3(path_to_read,path_to_write,order,qfix)
    step4(path_to_read,path_to_write,order)
    # step5(path_to_read,order,mixamo=mixamo)
    # step6(path_to_read,path_to_write,order)
    # step7(path_to_read, path_to_write, order)
    # step8(path_to_read, path_to_write)
    path_to_readortho6D = "/DATA/Projects/2020/lstm_pro/src/databases/61/ortho6D"
    path_to_readpos = "/DATA/Projects/2020/lstm_pro/src/databases/61/quatspos31"
    path_to_write = "/DATA/Projects/2020/lstm_pro/src/databases/61/ortho6Dpos"
    step9(path_to_readortho6D,path_to_readpos, path_to_write)

