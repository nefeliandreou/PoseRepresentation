import numpy as np
from utils.bvh import read_bvh_hierarchy
from utils.bvh import read_bvh
import transforms3d.euler as euler
from transforms3d.euler import euler2quat
import quaternion

Quaternion = quaternion.quaternion
from DualQuaternion2 import DualQuaternion


class loadbvh2:
    def __init__(self, bvh_file, order):

        self.bvh_file = bvh_file
        weight_translation = 0.01
        oldskeleton, non_end_bones = read_bvh_hierarchy.read_bvh_hierarchy(bvh_file)
        self.order = order
        if self.order == 'zxy':
            self.o = [2, 0, 1]
        elif order =="zyx":
            self.o = [2, 1, 0]
        elif order=="xyz":
            self.o =[0,1,2]
        self.data = read_bvh.parse_frames(bvh_file)
        self.sample_data = loadbvh2.permuteRawData(self.data, self.o)
        self.root_joint = [joint for joint in oldskeleton.keys() if oldskeleton[joint]['parent'] == None][0]
        self.Nframes = self.sample_data.shape[0]
        self.end_bones = [i for i in oldskeleton.keys() if oldskeleton[i]['channels'] == []]  # 7
        self.non_end_bones = [i for i in oldskeleton.keys() if oldskeleton[i]['channels'] != []]  # 31 root included
        self.skeleton = self.augmentSkeleton(oldskeleton)  # add Dxyz, rxyz, trans

    @staticmethod
    def permuteRawData(rawBVH, order):
        """change order from zxy to xyz"""
        new_data = rawBVH.copy()
        nj = int((new_data.shape[1] - 3) / 3)
        if order == [2, 0, 1]:
            new_data[:, [[3 + 3 * i, 4 + 3 * i, 5 + 3 * i] for i in range(0, nj)]] = new_data[:,
                                                                                     [[4 + i * 3, 5 + i * 3, 3 + 3 * i]
                                                                                      for i in range(0, nj)]]
        elif order == [2, 1, 0]:
            new_data[:, [[3 + 3 * i, 4 + 3 * i, 5 + 3 * i] for i in range(0, nj)]] = new_data[:,
                                                                                     [[5 + i * 3, 4 + i * 3, 3 + 3 * i]
                                                                                      for i in range(0, nj)]]
        return new_data

    @staticmethod
    def permute4bvh(frames, order):
        """change order from xyz to zxy(or zyx)
        inverse of permuteRawData"""

        new_data = frames.copy()
        nj = int((new_data.shape[1] - 3) / 3)
        if order == [2, 0, 1]:
            new_data[:, [[4 + 3 * i, 5 + 3 * i, 3 + 3 * i] for i in range(0, nj)]] = new_data[:,
                                                                                     [[3 + i * 3, 4 + i * 3, 5 + 3 * i]
                                                                                      for i in range(0, nj)]]
        elif order == [2, 1, 0]:
            new_data[:, [[5 + 3 * i, 4 + 3 * i, 3 + 3 * i] for i in range(0, nj)]] = new_data[:,
                                                                                     [[3 + i * 3, 4 + i * 3, 5 + 3 * i]
                                                                                      for
                                                                                      i in range(0, nj)]]
        return new_data

    def skeleton2dq(self):
        """given a skeleton which includes Dxyz, rxyz, transM per frame, converts all bvh data to dq"""
        framesDQC = []
        framesDQG = []

        for ff in range(self.sample_data.shape[0]):
            GlobalTransM = {}
            CurrentTransM = {}
            DQC = {}
            DQG = {}
            for nn in self.skeleton.keys():
                if self.skeleton[nn]['parent'] == None:
                    # root

                    x, y, z = self.skeleton[nn]['rxyz'][:, ff]
                    if self.order == 'zxy':
                        ang = [z * np.pi / 180, x * np.pi / 180, y * np.pi / 180]
                    elif self.order=="zyx":
                        ang = [z * np.pi / 180, y * np.pi / 180, x * np.pi / 180]
                    elif self.order=="xyz":
                        ang = [x* np.pi / 180, y * np.pi / 180, z * np.pi / 180]
                    quat = euler2quat(*ang, axes='r' + self.order)
                    DQC[nn] = DualQuaternion.from_quat_pose_array(list(quat) +
                                                                  list([0, 0, 0])).normalized().dq_array()

                elif self.skeleton[nn]['channels'] != []:
                    parent = self.skeleton[nn]['parent']
                    x, y, z = self.skeleton[nn]['rxyz'][:, ff]
                    if self.order == 'zxy':
                        ang = [z * np.pi / 180, x * np.pi / 180, y * np.pi / 180]
                    elif self.order=="zyx":
                        ang = [z * np.pi / 180, y * np.pi / 180, x * np.pi / 180]
                    elif self.order == "xyz":
                        ang = [x * np.pi / 180, y * np.pi / 180, z * np.pi / 180]
                    quat = euler2quat(*ang, axes='r' + self.order)
                    LocalHomoTransM = DualQuaternion.from_quat_pose_array(
                        list(quat) + list(self.skeleton[nn]['offsets'])).normalized()
                    DQC[nn] = (self.skeleton[parent]['transC'][ff] * LocalHomoTransM).normalized().dq_array()

            framesDQC.append(DQC)

        p = []
        for ff in range(self.Nframes):
            temp = list(self.skeleton[self.root_joint]['Dxyz'][:, ff])
            for i in framesDQC[ff].items():
                for j in i[1]:
                    temp.append(j)
            p.append(temp)
        return np.array(p)

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
                    quat = euler2quat(*ang, axes="r" + self.order)
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

    def skeleton2bvh(self, framesskeleton):
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
        return loadbvh2.permute4bvh(np.array(bvh), self.o)

    def dqframes2skeleton(self, frames):
        framesSkeleton = []
        for ff in frames:
            skeleton_ = {}
            for i in self.non_end_bones:
                skeleton_[i] = {}
            root_pos = ff[:3]

            def make(dq):
                dq = np.array(dq.dq_array())
                norm = np.linalg.norm(dq[:4])
                q = dq[:4] / norm
                d = dq[4:] / norm - np.dot(dq[:4], dq[4:]) / norm ** 2 * dq[:4] / norm
                dq = np.concatenate((q,d),axis=0)
                return DualQuaternion.from_dq_array(dq)
            for joint_idx, joint in enumerate(self.non_end_bones):
                # here globaltransM is taken directly from BVH
                skeleton_[joint]['CurrentTransM'] = DualQuaternion.from_dq_array(
                    ff[3 + joint_idx * 8:3 + 8 + joint_idx * 8]).normalized()         
                skeleton_[joint]['CurrentTransM']  = make(skeleton_[joint]['CurrentTransM'] )
                assert skeleton_[joint]['CurrentTransM'].q_r.norm()-1<1e-5
                assert  np.dot(quaternion.as_float_array( skeleton_[joint]['CurrentTransM'].q_r),quaternion.as_float_array( skeleton_[joint]['CurrentTransM'].q_d))<1e-5
            for joint_idx, joint in enumerate(self.non_end_bones):
                if joint == self.root_joint:
                    skeleton_[joint]['LocalTransMC'] = DualQuaternion.from_quat_pose_array(
                        list(skeleton_[joint]['CurrentTransM'].dq_array()[:4]) + [i for i in self.skeleton[joint][
                            'offsets']]).normalized()
                    skeleton_[joint]['LocalTransMC'] = make( skeleton_[joint]['LocalTransMC'])

                elif joint in self.end_bones:
                    continue

                else:
                    skeleton_[joint]['LocalTransMC'] = (
                            skeleton_[self.skeleton[joint]['parent']]['CurrentTransM'].inverse() * skeleton_[joint][
                        'CurrentTransM'])
                    assert skeleton_[joint]['LocalTransMC'].q_r.norm() - 1 < 1e-5
                    assert np.dot(quaternion.as_float_array( skeleton_[joint]['LocalTransMC'].q_r),quaternion.as_float_array( skeleton_[joint]['LocalTransMC'].q_d))<1e-5

            
                qt = skeleton_[joint]['LocalTransMC'].q_r.normalized()
                angles = [i * 180 / np.pi for i in
                              euler.quat2euler([qt.w, qt.x, qt.y, qt.z], axes='r' + self.order)]
           
                if self.order == "zxy":
                    skeleton_[joint]['rxyz'] = [angles[1], angles[2], angles[0]]
                elif self.order == "zyx":
                    skeleton_[joint]['rxyz'] = [angles[2], angles[1], angles[0]]
                skeleton_[joint]['Dxyz'] = root_pos
            framesSkeleton.append(skeleton_)
        return framesSkeleton

    def dqframes2skeletonthrq(self, frames):
        framesSkeleton = []
        for ff in frames:
            skeleton_ = {}
            for i in self.non_end_bones:
                skeleton_[i] = {}
            root_pos = ff[:3]

            for joint_idx, joint in enumerate(self.non_end_bones):
                # here globaltransM is taken directly from BVH
                skeleton_[joint]['CurrentTransM'] = DualQuaternion.from_dq_array(
                    ff[3 + joint_idx * 8:3 + 8 + joint_idx * 8]).normalized().q_r
                if joint == self.root_joint:
                    skeleton_[joint]['GlobalTransM'] = DualQuaternion.from_quat_pose_array(
                        list(ff[3 + joint_idx * 8:3 + 4 + joint_idx * 8]) + list(root_pos)).normalized().q_r

            for joint_idx, joint in enumerate(self.non_end_bones):
                if joint == self.root_joint:
                    skeleton_[joint]['LocalTransMG'] = skeleton_[joint]['GlobalTransM'].dq_array()[:4]

                    skeleton_[joint]['LocalTransMC'] = skeleton_[joint]['CurrentTransM'].dq_array()[:4]

                elif joint in self.end_bones:
                    continue

                else:
                    skeleton_[joint]['LocalTransMC'] = (
                                skeleton_[self.skeleton[joint]['parent']]['CurrentTransM'].inverse() * skeleton_[joint][
                            'CurrentTransM'])

                    skeleton_[joint]['GlobalTransM'] = (skeleton_[self.skeleton[joint]['parent']]['GlobalTransM'] * \
                                                        skeleton_[joint]['LocalTransMC']).normalized()
                    skeleton_[joint]['LocalTransMG'] = (
                                skeleton_[self.skeleton[joint]['parent']]['GlobalTransM'].inverse() * skeleton_[joint][
                            'GlobalTransM'])
                angles = [i * 180 / np.pi for i in
                          euler.quat2euler(skeleton_[joint]['LocalTransMC'], axes='r' + self.order)]
                skeleton_[joint]['rxyz'] = [angles[1], angles[2], angles[0]]
                t1 = quaternion.as_euler_angles(skeleton_[joint]['LocalTransMC'])
                assert t1[0] * 180 / np.pi == skeleton_[joint]['rxyz'][0]
                assert t1[1] * 180 / np.pi == skeleton_[joint]['rxyz'][1]
                assert t1[2] * 180 / np.pi == skeleton_[joint]['rxyz'][2]
            framesSkeleton.append(skeleton_)
        return framesSkeleton

    def qframes2skeleton(self, frames):
        framesSkeleton = []
        for idx, ff in enumerate(frames):
            skeleton_ = {}
            for i in self.non_end_bones:
                skeleton_[i] = {}
            root_pos = ff[:3]

            for joint_idx, joint in enumerate(self.non_end_bones):

                quat = ff[3 + joint_idx * 4:3 + 4 + joint_idx * 4]
                if joint == 'Hips':
                    skeleton_[joint]['Dxyz'] = root_pos

                if self.order == 'zxy':

                    angles = [i * 180 / np.pi for i in
                              euler.quat2euler(quat, axes="r" + self.order)]

                    angles2 = [i * 180 / np.pi for i in
                              euler.quat2euler(quat2, axes="r" + self.order)]
                    skeleton_[joint]['rxyz'] = [angles[1], angles[2], angles[0]]

                elif self.order == 'zyx':

                    angles = euler.quat2euler(quat)
                    skeleton_[joint]['rxyz'] = np.array(angles) * 180 / np.pi

            framesSkeleton.append(skeleton_)

        return framesSkeleton

    def augmentSkeleton(self, skel):
        skeleton = skel.copy()
        channel_count = 0;
        for nn in skel.keys():
            if len(skeleton[nn]['channels']) == 6:

                Dxyz = np.zeros((self.Nframes, 3))

                Dxyz = np.array(self.sample_data)[:, :3] + np.array(skeleton[nn]['offsets'])[None, :]
                skeleton[nn]['Dxyz'] = Dxyz.T

                # rotational data in raw_data in xyz format
                rxyz = np.zeros((self.Nframes, 3))
                rxyz = self.sample_data[:, [3, 4, 5]]
                skeleton[nn]['rxyz'] = rxyz.T
                skeleton[nn]['transC'] = np.empty(self.Nframes, dtype=DualQuaternion)
                skeleton[nn]['transG'] = np.empty(self.Nframes, dtype=DualQuaternion)
                for ff in range(self.Nframes):
                    x, y, z = skeleton[nn]['rxyz'][:, ff]
                    if self.order == 'zxy':
                        ang = [z * np.pi / 180, x * np.pi / 180, y * np.pi / 180]
                    else:
                        ang = [z * np.pi / 180, y * np.pi / 180, x * np.pi / 180]

                    quat = euler2quat(*ang, axes='r' + self.order)
                    skeleton[nn]['transC'][ff] = DualQuaternion.from_quat_pose_array(list(quat) + [0, 0, 0])

                    skeleton[nn]['transG'][ff] = DualQuaternion.from_quat_pose_array(list(quat) + \
                                                                                     list(skeleton[nn]['Dxyz'][:, ff]))


            elif len(skeleton[nn]['channels']) == 3:
                rxyz = np.zeros((self.Nframes, 3))
                rxyz = self.sample_data[:, [channel_count, channel_count + 1, channel_count + 2]]
                skeleton[nn]['rxyz'] = rxyz.T
                Dxyz = np.zeros((3, self.Nframes))
                skeleton[nn]['Dxyz'] = Dxyz
                skeleton[nn]['transC'] = np.empty(self.Nframes, dtype=DualQuaternion)
                skeleton[nn]['transG'] = np.empty(self.Nframes, dtype=DualQuaternion)

            elif len(skeleton[nn]['channels']) == 0:
                Dxyz = np.zeros((3, self.Nframes))
                skeleton[nn]['Dxyz'] = Dxyz
            channel_count += len(skeleton[nn]['channels'])

        for nn in [i for i in skeleton.keys() if (len(skeleton[i]['channels']) != 0) & \
                                                 (skeleton[i]['parent'] != None)]:
            parent = skeleton[nn]['parent']
            for ff in range(self.Nframes):
                x, y, z = skeleton[nn]['rxyz'][:, ff]
                if self.order == 'zxy':
                    ang = [z * np.pi / 180, x * np.pi / 180, y * np.pi / 180]
                else:
                    ang = [z * np.pi / 180, y * np.pi / 180, x * np.pi / 180]
                quat = euler2quat(*ang, axes='r' + self.order)
                transM = DualQuaternion.from_quat_pose_array(list(quat) + skeleton[nn]['offsets'])
                skeleton[nn]['transC'][ff] = skeleton[parent]['transC'][ff] * transM

                skeleton[nn]['transG'][ff] = skeleton[parent]['transG'][ff] * transM
                skeleton[nn]['Dxyz'][:, ff] = skeleton[nn]['transG'][ff].translation()

        for nn in [i for i in skeleton.keys() if (len(skeleton[i]['channels']) == 0)]:
            parent = skeleton[nn]['parent']
            for ff in range(self.Nframes):
                quat = euler2quat(0, 0, 0, axes='r' + self.order)
                tempTrans = DualQuaternion.from_quat_pose_array(list(quat) + skeleton[nn]['offsets'])
                transM = skeleton[parent]['transG'][ff] * tempTrans
                skeleton[nn]['Dxyz'][:, ff] = transM.translation()
        return skeleton

    def check(self):
        assert np.isclose(self.skeleton2bvh(self.sk) - self.data, 0).all()

    def __call__(self):
        newframes = []
        self.framesDQG, self.framesDQC = self.skeleton2dq()
        self.sk = self.dqframes2skeleton(self.framesDQC)
        self.check()
