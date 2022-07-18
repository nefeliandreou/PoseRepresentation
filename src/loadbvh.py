from sklearn import preprocessing

import numpy as np
import torch
from utils.bvh import read_bvh_hierarchy
from utils.bvh import read_bvh

import transforms3d.euler as euler
from transforms3d.euler import euler2quat

# ##pyquaternioin
# from DualQuaternion import DualQuaternion
# from pyquaternion import Quaternion

###uncomment for numpy quaternion
# numpy_quat = False


import quaternion
Quaternion = quaternion.quaternion
from rhythmdata.DualQuaternion2 import DualQuaternion
numpy_quat = True

class loadbvh:
    def __init__(self, bvh_file, order):
        self.bvh_file = bvh_file
        weight_translation = 0.01
        self.order = 'r' + order
        if self.order == 'rzyx':
            self.o = [2, 1, 0]
        else:
            self.o = [2, 0, 1]
        oldskeleton, non_end_bones = read_bvh_hierarchy.read_bvh_hierarchy(bvh_file)
        self.data = read_bvh.parse_frames(bvh_file)

        self.sample_data = loadbvh.permuteRawData(self.data, self.o)
        self.root_joint = [joint for joint in oldskeleton.keys() if oldskeleton[joint]['parent'] == None][0]
        self.Nframes = self.sample_data.shape[0]
        self.end_bones = [i for i in oldskeleton.keys() if oldskeleton[i]['channels'] == []]  # 7
        self.non_end_bones = [i for i in oldskeleton.keys() if oldskeleton[i]['channels'] != []]  # 31 root included
        self.skeleton = self.augmentSkeleton(oldskeleton)  # add Dxyz, rxyz, trans

        # joint_index= read_bvh.get_pos_joints_index(self.sample_data[0],self.non_end_bones, self.skeleton)

    @staticmethod
    def Mat2DQuat(M):
        """matrix to dual quaternion"""
        dq = DualQuaternion.from_homogeneous_matrix(M).normalized()
        if numpy_quat:
            dq = dq.dq_array()
        else:
            dq = list(dq.q_r) + list(dq.q_d)

        return dq

    @staticmethod
    def DQ2Mat(dq):
        """dual quaternion to matrix"""
        if numpy_quat == False:
            M = DualQuaternion(Quaternion(dq[:4]), Quaternion(dq[4:])).normalized().homogeneous_matrix()
        else:
            M = DualQuaternion(Quaternion(*dq[:4]), Quaternion(*dq[4:]),
                               normalize=True).normalized().homogeneous_matrix()

        return M

    @staticmethod
    def transformation_matrix(displ, rxyz, order):
        """give displacement and rotation calculate transformation matrix"""
        c = [np.cos(i * np.pi / 180) for i in rxyz]
        s = [np.sin(i * np.pi / 180) for i in rxyz]

        RxRyRz = np.zeros((3, 3, 3))
        RxRyRz[:, :, 0] = [[1, 0, 0], [0, c[0], -s[0]], [0, s[0], c[0]]]
        RxRyRz[:, :, 1] = [[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]]
        RxRyRz[:, :, 2] = [[c[2], -s[2], 0], [s[2], c[2], 0], [0, 0, 1]]
        rotM = RxRyRz[:, :, order[0]] @ RxRyRz[:, :, order[1]] @ RxRyRz[:, :, order[2]]

        transM = np.zeros((4, 4))
        transM[:3, :3] = rotM
        transM[3, :] = [0, 0, 0, 1]
        transM[:3, 3] = displ
        return transM

    @staticmethod
    def rotation_matrix(rxyz, order):
        """give displacement and rotation calculate transformation matrix"""
        c = [np.cos(i * np.pi / 180) for i in rxyz]
        s = [np.sin(i * np.pi / 180) for i in rxyz]
        RxRyRz = np.zeros((3, 3, 3))
        RxRyRz[:, :, 0] = [[1, 0, 0], [0, c[0], -s[0]], [0, s[0], c[0]]]
        RxRyRz[:, :, 1] = [[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]]
        RxRyRz[:, :, 2] = [[c[2], -s[2], 0], [s[2], c[2], 0], [0, 0, 1]]
        rotM = RxRyRz[:, :, order[0]] @ RxRyRz[:, :, order[1]] @ RxRyRz[:, :, order[2]]

        return rotM

    @staticmethod
    def euler2mat(rxyz, order):
        c = [np.cos(i * np.pi / 180) for i in rxyz]
        s = [np.sin(i * np.pi / 180) for i in rxyz]

        RxRyRz = np.zeros((3, 3, 3))
        RxRyRz[:, :, 0] = [[1, 0, 0], [0, c[0], -s[0]], [0, s[0], c[0]]]
        RxRyRz[:, :, 1] = [[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]]
        RxRyRz[:, :, 2] = [[c[2], -s[2], 0], [s[2], c[2], 0], [0, 0, 1]]
        return RxRyRz[:, :, order[0]] @ RxRyRz[:, :, order[1]] @ RxRyRz[:, :, order[2]]

    @staticmethod
    def permuteRawData(rawBVH, order):
        """change order from zxy to xyz"""
        new_data = rawBVH.copy()
        if order == [2, 0, 1]:
            new_data[:, [[3 + 3 * i, 4 + 3 * i, 5 + 3 * i] for i in range(0, 31)]] = new_data[:,
                                                                                     [[4 + i * 3, 5 + i * 3, 3 + 3 * i]
                                                                                      for i in range(0, 31)]]
        elif order == [2, 1, 0]:
            new_data[:, [[3 + 3 * i, 4 + 3 * i, 5 + 3 * i] for i in range(0, 31)]] = new_data[:,
                                                                                     [[5 + i * 3, 4 + i * 3, 3 + 3 * i]
                                                                                      for i in range(0, 31)]]
        return new_data

    @staticmethod
    def permute4bvh(frames, order):
        """change order from xyz to zxy(or zyx)
        inverse of permuteRawData"""

        new_data = frames.copy()
        if order == [2, 0, 1]:
            # new_data[:,[4, 5, 3]] = new_data[:,[3, 4,5]]
            new_data[:, [[4 + 3 * i, 5 + 3 * i, 3 + 3 * i] for i in range(0, 31)]] = new_data[:,
                                                                                     [[3 + i * 3, 4 + i * 3, 5 + 3 * i]
                                                                                      for i in range(0, 31)]]
        elif order == [2, 1, 0]:
            # new_data[:, [5, 4, 3]] = new_data[:, [3, 4, 5]]
            new_data[:, [[5 + 3 * i, 4 + 3 * i, 3 + 3 * i] for i in range(0, 31)]] = new_data[:,
                                                                                     [[3 + i * 3, 4 + i * 3, 5 + 3 * i]
                                                                                      for
                                                                                      i in range(0, 31)]]
        return new_data

    @staticmethod
    # from rotation continuity
    def normalize_vector(v, return_mag=False):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])).cuda())
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        if (return_mag == True):
            return v, v_mag[:, 0]
        else:
            return v

    # u, v batch*n
    @staticmethod
    # from rotation continuity
    def cross_product(u, v):
        batch = u.shape[0]
        # print (u.shape)
        # print (v.shape)

        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

        return out

    @staticmethod
    # from rotation continuity
    def compute_rotation_matrix_from_ortho6d(ortho6d):

        x_raw = ortho6d[:, 0:3]  # batch*3
        y_raw = ortho6d[:, 3:6]  # batch*3

        x = loadbvh.normalize_vector(x_raw)  # batch*3
        z = loadbvh.cross_product(x, y_raw)  # batch*3
        z = loadbvh.normalize_vector(z)  # batch*3
        y = loadbvh.cross_product(z, x)  # batch*3

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)
        matrix = torch.cat((x, y, z), 2)  # batch*3*3
        return matrix

    @staticmethod
    def rotfromortho6d(ortho6D):
        x_raw = ortho6D[0:3]
        y_raw = ortho6D[3:]

        x = preprocessing.normalize(x_raw.reshape(1, -1))
        z = np.cross(x, y_raw)
        z = preprocessing.normalize(z.reshape(1, -1))
        y = np.cross(z, x)
        matrix = np.zeros((3, 3))
        matrix[:, 0] = x
        matrix[:, 1] = y
        matrix[:, 2] = z
        return matrix

    def seq6D2euler(seq6D, order, non_end_bones=31):  # 1,189
        o = 'r' + order
        temp = list(seq6D[:3])
        for i in range(3, non_end_bones * 6, 6):
            ang = [j * 180 / np.pi for j in euler.mat2euler(loadbvh.rotfromortho6d(seq6D[i:i + 6]), axes=o)]
            temp += ang
        return temp

    def seq6Dframes2euler(seq6Dframes, order='zxy'):  # 1,189
    """convert 6D frames to Euler"""
        temp = []
        for seq in seq6Dframes:
            temp.append(loadbvh.seq6D2euler(seq, order))
        return np.array(temp)


    def skeleton2dq(self):
        """given a skeleton which includes Dxyz, rxyz, transM per frame, converts all bvh data to dq"""
        framesDQC = []
        for ff in range(self.sample_data.shape[0]):
            GlobalTransM = {}
            CurrentTransM = {}
            DQC = {}
            DQG = {}
            for nn in self.skeleton.keys():
                if self.skeleton[nn]['parent'] == None:
                    # root
                    CurrentTransM[nn] = loadbvh.transformation_matrix([0, 0, 0], self.skeleton[nn]['rxyz'][:, ff],
                                                                      self.o)
                    DQC[nn] = loadbvh.Mat2DQuat(CurrentTransM[nn])

                elif self.skeleton[nn]['channels'] != []:
                    parent = self.skeleton[nn]['parent']
                    LocalHomoTransM = loadbvh.transformation_matrix(self.skeleton[nn]['offsets'],
                                                                    self.skeleton[nn]['rxyz'][:, ff], self.o)
                    CurrentTransM[nn] = self.skeleton[parent]['transC'][:, :, ff] @ LocalHomoTransM
                    DQC[nn] = loadbvh.Mat2DQuat(CurrentTransM[nn])
            framesDQC.append(DQC)
     
        p = []
        for ff in range(self.Nframes):
            temp = list(self.skeleton[self.root_joint]['Dxyz'][:, ff])
            for i in framesDQC[ff].items():
                for j in i[1]:
                    temp.append(j)
            p.append(temp)
        return np.array(p)

    def skeleton2ortho6D(self, order='zxy'):
        """given a skeleton which includes Dxyz, rxyz, transM per frame, converts all bvh data to dq"""

        framesDQG = []
        o = 'r' + order

        for ff in range(self.sample_data.shape[0]):
            DQG = {}
            for nn in self.skeleton.keys():
                if self.skeleton[nn]['channels'] == []:
                    pass
                else:
                    DQG[nn] = list(loadbvh.rotation_matrix(
                        self.skeleton[nn]['rxyz'][:, ff], self.o)[:, 0]) + list(loadbvh.rotation_matrix(
                        self.skeleton[nn]['rxyz'][:, ff], self.o)[:, 1])
                    assert np.isclose(loadbvh.rotfromortho6d(np.array(DQG[nn]))
                                      - loadbvh.rotation_matrix(self.skeleton[nn]['rxyz'][:, ff], self.o), 0).all()

            framesDQG.append(DQG)

        p = []
        for ff in range(self.Nframes):
            temp = list(self.skeleton[self.root_joint]['Dxyz'][:, ff])
            for i in framesDQG[ff].items():
                for j in i[1]:
                    temp.append(j)
            p.append(temp)
        return np.array(p)

    def skeleton2dq2(self, data, skeleton):
        """given a skeleton which includes Dxyz, rxyz, transM per frame, converts all bvh data to dq"""
        framesDQC = []
        framesDQG = []
        framesGTransM = []
        framesCTransM = []

        for ff in range(data.shape[0]):
            GlobalTransM = {}
            CurrentTransM = {}
            DQC = {}
            DQG = {}
            for nn in self.skeleton.keys():
                if self.skeleton[nn]['parent'] == None:
                    # root

                    GlobalTransM[nn] = loadbvh.transformation_matrix(skeleton[nn]['Dxyz'][:, ff],
                                                                     skeleton[nn]['rxyz'][:, ff], self.o)
                    CurrentTransM[nn] = loadbvh.transformation_matrix([0, 0, 0], skeleton[nn]['rxyz'][:, ff], self.o)
                    DQG[nn] = loadbvh.Mat2DQuat(GlobalTransM[nn])
                    DQC[nn] = loadbvh.Mat2DQuat(CurrentTransM[nn])

                elif self.skeleton[nn]['channels'] != []:
                    parent = skeleton[nn]['parent']
                    LocalHomoTransM = loadbvh.transformation_matrix(skeleton[nn]['offsets'],
                                                                    skeleton[nn]['rxyz'][:, ff], self.o)
                    CurrentTransM[nn] = skeleton[parent]['transC'][:, :, ff] @ LocalHomoTransM
                    GlobalTransM[nn] = skeleton[parent]['transG'][:, :, ff] @ LocalHomoTransM
                    DQC[nn] = loadbvh.Mat2DQuat(CurrentTransM[nn])
                    DQG[nn] = loadbvh.Mat2DQuat(GlobalTransM[nn])
            framesDQC.append(DQC)
            framesDQG.append(DQG)
            framesGTransM.append(GlobalTransM)
            framesCTransM.append(CurrentTransM)

        k = []
        for ff in range(self.Nframes):
            temp = list(skeleton[self.root_joint]['Dxyz'][:, ff])
            for i in framesDQG[ff].items():
                for j in i[1]:
                    temp.append(j)
            k.append(temp)
        p = []
        for ff in range(self.Nframes):
            temp = list(skeleton[self.root_joint]['Dxyz'][:, ff])
            for i in framesDQC[ff].items():
                for j in i[1]:
                    temp.append(j)
            p.append(temp)
        return np.array(k), np.array(p), framesGTransM, framesCTransM

    def skeleton2bvh(self, framesskeleton):
        """convert Euler skeleton to bvh format"""
        bvh = []
        for ff in framesskeleton:
            temp = []
            for joint in self.non_end_bones:
                if joint == self.root_joint:
                    temp.append(ff[joint]['Dxyz'][0] - self.skeleton[joint]['offsets'][0])
                    temp.append(ff[joint]['Dxyz'][1] - self.skeleton[joint]['offsets'][1])
                    temp.append(ff[joint]['Dxyz'][2] - self.skeleton[joint]['offsets'][2])

                temp.append(ff[joint]['rxyz'][0])
                temp.append(ff[joint]['rxyz'][1])
                temp.append(ff[joint]['rxyz'][2])

            bvh.append(temp)
        return loadbvh.permute4bvh(np.array(bvh), self.o)

    def skeleton2q(self):
        """given a skeleton which includes Dxyz, rxyz, transM per frame, converts all bvh data to dq"""
        frames = []

        for ff in range(self.Nframes):
            Q = {}

            for nn in self.skeleton.keys():
                if nn == 'Hips' or nn in self.non_end_bones:
                    x, y, z = self.skeleton[nn]['rxyz'][:, ff]
                    if self.order == 'rzxy':
                        ang = [z * np.pi / 180, x * np.pi / 180, y * np.pi / 180]
                    else:
                        ang = [z * np.pi / 180, y * np.pi / 180, x * np.pi / 180]
                    quat = euler2quat(*ang, axes=self.order)
                    Q[nn] = quat
            frames.append(Q)

        k = []
        for ff in range(self.Nframes):
            temp = list(self.skeleton[self.root_joint]['Dxyz'][:, ff])
            for i in frames[ff].items():
                for j in i[1]:
                    temp.append(j)
            k.append(temp)

        return np.array(k)

    def dqframes2skeletonthrq(self, frames):
    """convert dq frames to skeleton via quaternions"""
        framesSkeleton = []
        root_pos = ff[:3]
        for idx, ff in enumerate(frames):
            skeleton_ = {}
            for i in self.non_end_bones:
                skeleton_[i] = {}

            for joint_idx, joint in enumerate(self.non_end_bones):
                skeleton_[joint]['CurrentTransM'] = loadbvh.DQ2Mat(ff[3 + joint_idx * 8:3 + 8 + joint_idx * 8])[:3, :3]

            for joint_idx, joint in enumerate(self.non_end_bones):
                if joint == self.root_joint:
                    skeleton_[joint]['LocalTransMC'] = skeleton_[joint]['CurrentTransM'].copy()
                elif joint in self.end_bones:
                    continue
                else:
                    skeleton_[joint]['LocalTransMC'] = np.linalg.inv(
                        skeleton_[self.skeleton[joint]['parent']]['CurrentTransM']) @ skeleton_[joint]['CurrentTransM']

                if self.order == 'rzxy':

                    angles = [i * 180 / np.pi for i in
                              euler.mat2euler(skeleton_[joint]['LocalTransMC'][:3, :3], axes=self.order)]
                    skeleton_[joint]['rxyz'] = [angles[1], angles[2], angles[0]]
                elif self.order == 'rzyx':

                    angles = euler.mat2euler(skeleton_[joint]['LocalTransMC'][:3, :3])
                    skeleton_[joint]['rxyz'] = np.array(angles) * 180 / np.pi
                skeleton_[joint]['Dxyz'] = root_pos
            framesSkeleton.append(skeleton_)
        return framesSkeleton

    def dqframes2skeleton(self, frames):
        """convert dq frames to skeleton via rotation matrix"""
        framesSkeleton = []
        for idx, ff in enumerate(frames):
            skeleton_ = {}
            for i in self.non_end_bones:
                skeleton_[i] = {}
            root_pos = ff[:3]

            for joint_idx, joint in enumerate(self.non_end_bones):
                # here globaltransM is taken directly from BVH
                skeleton_[joint]['CurrentTransM'] = loadbvh.DQ2Mat(ff[3 + joint_idx * 8:3 + 8 + joint_idx * 8])

                skeleton_[joint]['quat'] = ff[3 + joint_idx * 8:3 + 8 + joint_idx * 8][:4]
                skeleton_[joint]['dualquat'] = ff[3 + joint_idx * 8:3 + 8 + joint_idx * 8][4:]

            for joint_idx, joint in enumerate(self.non_end_bones):
                if joint == self.root_joint:
                    skeleton_[joint]['LocalTransMC'] = skeleton_[joint]['CurrentTransM'].copy()
                    skeleton_[joint]['LocalTransMC'][:3, 3] = [i for i in self.skeleton[joint]['offsets']]

                elif joint in self.end_bones:
                    continue

                else:
                    skeleton_[joint]['LocalTransMC'] = np.linalg.inv(
                        skeleton_[self.skeleton[joint]['parent']]['CurrentTransM']) @ skeleton_[joint]['CurrentTransM']
                if self.order == 'rzxy':

                    angles = [i * 180 / np.pi for i in
                              euler.mat2euler(skeleton_[joint]['LocalTransMC'][:3, :3], axes=self.order)]
                    skeleton_[joint]['rxyz'] = [angles[1], angles[2], angles[0]]

                elif self.order == 'rzyx':

                    angles = euler.mat2euler(skeleton_[joint]['LocalTransMC'][:3, :3])
                    skeleton_[joint]['rxyz'] = np.array(angles) * 180 / np.pi
                    skeleton_[joint]['Dxyz'] = root_pos
            framesSkeleton.append(skeleton_)
        return framesSkeleton

    def qframes2skeleton(self, frames):
        """convert q frames to skeleton"""

        framesSkeleton = []
        print(self.order)
        for idx, ff in enumerate(frames):
            skeleton_ = {}
            for i in self.non_end_bones:
                skeleton_[i] = {}
            root_pos = ff[:3]

            for joint_idx, joint in enumerate(self.non_end_bones):
                # here globaltransM is taken directly from BVH
                quat = ff[3 + joint_idx * 4:3 + 4 + joint_idx * 4]
                if joint == 'Hips':
                    skeleton_[joint]['Dxyz'] = root_pos

                if self.order == 'rzxy':

                    angles = [i * 180 / np.pi for i in
                              euler.quat2euler(quat, axes=self.order)]
                    skeleton_[joint]['rxyz'] = [angles[1], angles[2], angles[0]]

                elif self.order == 'rzyx':

                    angles = euler.quat2euler(quat)
                    skeleton_[joint]['rxyz'] = np.array(angles) * 180 / np.pi

            framesSkeleton.append(skeleton_)

        return framesSkeleton

    def augmentSkeleton(self, skel):
        """calculate positions"""

        skeleton = skel.copy()
        channel_count = 0;
        for nn in skel.keys():
            if len(skeleton[nn]['channels']) == 6:
                # assume translational data always XYZ
                Dxyz = np.zeros((self.Nframes, 3))

                Dxyz = np.array(self.sample_data)[:, :3] + np.array(skeleton[nn]['offsets'])[None, :]
                skeleton[nn]['Dxyz'] = Dxyz.T

                # rotational data in raw_data in xyz format
                rxyz = np.zeros((self.Nframes, 3))
                rxyz = self.sample_data[:, [3, 4, 5]]
                skeleton[nn]['rxyz'] = rxyz.T
                skeleton[nn]['transC'] = np.zeros((4, 4, self.Nframes))
                skeleton[nn]['transG'] = np.zeros((4, 4, self.Nframes))
                for ff in range(self.Nframes):
                    skeleton[nn]['transC'][:, :, ff] = loadbvh.transformation_matrix([0, 0, 0],
                                                                                     skeleton[nn]['rxyz'][:, ff],
                                                                                     self.o)
                    skeleton[nn]['transG'][:, :, ff] = loadbvh.transformation_matrix(skeleton[nn]['Dxyz'][:, ff],
                                                                                     skeleton[nn]['rxyz'][:, ff],
                                                                                     self.o)

            elif len(skeleton[nn]['channels']) == 3:
                rxyz = np.zeros((self.Nframes, 3))
                rxyz = self.sample_data[:, [channel_count, channel_count + 1, channel_count + 2]]
                skeleton[nn]['rxyz'] = rxyz.T
                Dxyz = np.zeros((3, self.Nframes))
                skeleton[nn]['Dxyz'] = Dxyz
                skeleton[nn]['transC'] = np.zeros((4, 4, self.Nframes))
                skeleton[nn]['transG'] = np.zeros((4, 4, self.Nframes))

            elif len(skeleton[nn]['channels']) == 0:
                Dxyz = np.zeros((3, self.Nframes))
                skeleton[nn]['Dxyz'] = Dxyz
            channel_count += len(skeleton[nn]['channels'])

        for nn in [i for i in skeleton.keys() if (len(skeleton[i]['channels']) != 0) & (skeleton[i]['parent'] != None)]:
            parent = skeleton[nn]['parent']
            for ff in range(self.Nframes):
                transM = loadbvh.transformation_matrix(skeleton[nn]['offsets'], skeleton[nn]['rxyz'][:, ff], self.o)
                skeleton[nn]['transC'][:, :, ff] = skeleton[parent]['transC'][:, :, ff] @ transM

                skeleton[nn]['transG'][:, :, ff] = skeleton[parent]['transG'][:, :, ff] @ transM
                skeleton[nn]['Dxyz'][:, ff] = skeleton[nn]['transG'][:3, 3, ff]

        for nn in [i for i in skeleton.keys() if (len(skeleton[i]['channels']) == 0)]:
            parent = skeleton[nn]['parent']
            for ff in range(self.Nframes):
                tempTrans = np.eye(4)
                tempTrans[:3, 3] = skeleton[nn]['offsets']
                transM = skeleton[parent]['transG'][:, :, ff] @ tempTrans
                skeleton[nn]['Dxyz'][:, ff] = transM[:3, 3]
           
        return skeleton

    def exportpositions(self, outfolder='',save=True):
        positions = np.zeros((self.data.shape[0] , len(self.skeleton.keys()) * 3))
        for ff in range(self.data.shape[0]):
            for i, nn in enumerate(self.skeleton.keys()):
                if nn == 'Hips':
                    positions[ff, i * 3:i * 3 + 3] = self.skeleton[nn]['Dxyz'][:, ff]
                else:
                    positions[ff, i * 3:i * 3 + 3] = self.skeleton[nn]['Dxyz'][:, ff] - self.skeleton['Hips']['Dxyz'][:,
                                                                                        ff]
        fname = str(self.bvh_file.split('.bvh')[0] + '.npy')
        if save:
            np.save(file=fname, arr=positions)
        return positions

    def check(self):
        if np.isclose(self.skeleton2bvh(self.sk) - self.sample_data, 0).all() == False:
            idx = np.where(np.isclose(self.skeleton2bvh(self.sk) - self.data, 0) == False)
            assert np.isclose(self.skeleton2bvh(self.sk)[idx] - self.data[idx], 360).any() or np.isclose(
                self.skeleton2bvh(self.sk)[idx] - self.data[idx], -360).any()

    def __call__(self):
        # self.framesDQG, self.framesDQC, self.framesGTransM, self.framesCTransM = self.skeleton2dq()
        self.framesDQC = self.skeleton2dq()
        self.sk = self.dqframes2skeleton(self.framesDQC)
    #         self.check()

# k = loadbvh("/DATA/Projects/2020/lstm_pro/src/databases/contemporary/bvh/MaritsaElia_Excited.bvh")

# count=0
# for ff in range(k.Nframes):
#     if np.isclose(k.skeleton2bvh(k.sk)[ff]-k.data[ff],0).all()==False:
#         idx = np.where(np.isclose(k.skeleton2bvh(k.sk)[ff]-k.data[ff],0)==False)
#         if np.isclose((k.skeleton2bvh(k.sk)[ff][idx[0][0]]-k.data[ff][idx[0][0]]),360) or np.isclose((k.skeleton2bvh(k.sk)[ff][idx[0][0]]-k.data[ff][idx[0][0]]),-360):
#             count+=1
