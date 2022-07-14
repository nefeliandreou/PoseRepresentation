import numpy as np, math, re
from collections import OrderedDict
import transforms3d.euler as euler, cv2 as cv

"""
Code from: https://github.com/papagina/Auto_Conditioned_RNN_motion
"""


def get_child_dict(skel):
    child_dict = {}
    for t in skel.keys():
        parent = skel[t]['parent']
        if parent in child_dict.keys():
            child_dict[parent].append(t)
        else:
            child_dict[parent] = [
                t]

    return child_dict


def get_hip_transform(motion, skel):
    offsets_t = motion[0:3]
    Zrotation = motion[3]
    Yrotation = motion[4]
    Xrotation = motion[5]
    theta = [
        Xrotation, Yrotation, Zrotation]
    Rotation = eulerAnglesToRotationMatrix_hip(theta)
    Transformation = np.zeros((4, 4))
    Transformation[0:3, 0:3] = Rotation
    Transformation[3][3] = 1
    Transformation[0][3] = offsets_t[0]
    Transformation[1][3] = offsets_t[1]
    Transformation[2][3] = offsets_t[2]
    return Transformation


def get_skeleton_position(motion, non_end_bones, skel):
    pos_dict = OrderedDict()
    for bone in skel.keys():
        pos = get_pos(bone, motion, non_end_bones, skel)
        pos_dict[bone] = pos[0:3]

    return pos_dict


def get_bone_start_end(positions, skeleton, hips_name):
    bone_list = []
    for bone in positions.keys():
        if bone != hips_name:
            bone_end = positions[bone]
            bone_start = positions[skeleton[bone]['parent']]
            bone_tuple = (bone_start, bone_end)
            bone_list.append(bone_tuple)

    return bone_list


def rotation_dic_to_vec(rotation_dictionary, non_end_bones, position, hips_name, o=[2, 0, 1]):
    motion_vec = np.zeros(6 + len(non_end_bones) * 3)
    motion_vec[0:3] = position[hips_name]

    motion_vec[3] = rotation_dictionary[hips_name][o[0]]
    motion_vec[4] = rotation_dictionary[hips_name][o[1]]
    motion_vec[5] = rotation_dictionary[hips_name][o[2]]
    for i in range(0, len(non_end_bones)):
        motion_vec[3 * (i + 2)] = rotation_dictionary[non_end_bones[i]][o[0]]
        motion_vec[3 * (i + 2) + 1] = rotation_dictionary[non_end_bones[i]][o[1]]
        motion_vec[3 * (i + 2) + 2] = rotation_dictionary[non_end_bones[i]][o[2]]

    return motion_vec


def get_pos(bone, motion, non_end_bones, skel):
    global_transform = np.dot(get_hip_transform(motion, skel), get_global_transform(bone, skel, motion, non_end_bones))
    position = np.dot(global_transform, np.array([0, 0, 0, 1])[:, np.newaxis])
    return position


def get_global_transform(bone, skel, motion, non_end_bones):
    parent = skel[bone]['parent']
    Transformation = get_relative_transformation(bone, non_end_bones, motion, skel)
    while parent != None:
        parent_transformation = get_relative_transformation(parent, non_end_bones, motion, skel)
        Transformation = np.dot(parent_transformation, Transformation)
        parent = skel[parent]['parent']

    return Transformation


def get_relative_transformation(bone, non_end_bones, motion, skel):
    end_bone = 0
    try:
        bone_index = non_end_bones.index(bone)
    except:
        end_bone = 1

    if end_bone == 0:
        Zrotation = motion[6 + 3 * bone_index]
        Xrotation = motion[6 + 3 * bone_index + 2]
        Yrotation = motion[6 + 3 * bone_index + 1]
        theta = [
            Xrotation, Yrotation, Zrotation]
        Rotation = eulerAnglesToRotationMatrix_hip(theta)
    else:
        Rotation = np.identity(3)
    Transformation = np.zeros((4, 4))
    Transformation[0:3, 0:3] = Rotation
    Transformation[3][3] = 1
    offsets_t = np.array(skel[bone]['offsets'])
    Transformation[0][3] = offsets_t[0]
    Transformation[1][3] = offsets_t[1]
    Transformation[2][3] = offsets_t[2]
    return Transformation


def eulerAnglesToRotationMatrix(theta1):
    theta = np.array(theta1) * (math.pi / 180)
    R_x = np.array([[1, 0, 0],
                    [
                        0, math.cos(theta[0]), -math.sin(theta[0])],
                    [
                        0, math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [
                        0, 1, 0],
                    [
                        -math.sin(theta[1]), 0, math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [
                        math.sin(theta[2]), math.cos(theta[2]), 0],
                    [
                        0, 0, 1]])
    R = np.dot(R_z, np.dot(R_x, R_y))
    return R


def eulerAnglesToRotationMatrix_hip(theta1):
    theta = np.array(theta1) * (math.pi / 180)
    R_x = np.array([[1, 0, 0],
                    [
                        0, math.cos(theta[0]), -math.sin(theta[0])],
                    [
                        0, math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [
                        0, 1, 0],
                    [
                        -math.sin(theta[1]), 0, math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [
                        math.sin(theta[2]), math.cos(theta[2]), 0],
                    [
                        0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def inside_image(x, y):
    return x >= 0 and x < 512 and y >= 0 and y < 424


def visualize_joints(bone_list, focus):
    m = np.zeros((424, 600, 3))
    m.astype(np.uint8)
    for bone in bone_list:
        p1x = bone[0][0]
        p1y = bone[0][1]
        p1z = bone[0][2] + 400
        p2x = bone[1][0]
        p2y = bone[1][1]
        p2z = bone[1][2] + 400
        p1 = (
            int(p1x * focus / p1z + 300.0), int(-p1y * focus / p1z + 204.0))
        p2 = (int(p2x * focus / p2z + 300.0), int(-p2y * focus / p2z + 204.0))
        if inside_image(p1[0], p1[1]) and inside_image(p2[0], p2[1]):
            cv.line(m, p1, p2, (255, 0, 0), 2)
            cv.circle(m, p1, 2, (0, 255, 255), -1)
            cv.circle(m, p2, 2, (0, 255, 255), -1)

    return m


def visualize_joints2(bone_list, focus):
    m = np.zeros((424, 600, 3))
    m.astype(np.uint8)
    for bone in bone_list:
        p1x = bone[0][0]
        p1y = bone[0][1]
        p1z = bone[0][2] + 400
        p2x = bone[1][0]
        p2y = bone[1][1]
        p2z = bone[1][2] + 400
        p1 = (
            int(p1x * focus / p1z + 300.0), int(-p1y * focus / p1z + 204.0))
        p2 = (int(p2x * focus / p2z + 300.0), int(-p2y * focus / p2z + 204.0))
        if inside_image(p1[0], p1[1]) and inside_image(p2[0], p2[1]):
            cv.line(m, p1, p2, (255, 0, 0), 2)
            cv.circle(m, p1, 2, (0, 255, 255), -1)
            cv.circle(m, p2, 2, (0, 255, 255), -1)

    return m


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-06


def rotationMatrixToEulerAngles(R):
    assert isRotationMatrix(R)
    sy = math.sqrt(R[(0, 0)] * R[(0, 0)] + R[(1, 0)] * R[(1, 0)])
    singular = sy < 1e-06
    if not singular:
        x = math.atan2(R[(2, 1)], R[(2, 2)])
        y = math.atan2(-R[(2, 0)], sy)
        z = math.atan2(R[(1, 0)], R[(0, 0)])
    else:
        x = math.atan2(-R[(1, 2)], R[(1, 1)])
        y = math.atan2(-R[(2, 0)], sy)
        z = 0
    return np.array([x, y, z])


def xyz_to_rotations(skel, position, hips_name):
    all_rotations = {}
    for bone in skel.keys():
        if bone != hips_name:
            parent = skel[bone]['parent']
            parent_xyz = position[parent]
            bone_xyz = position[bone]
            displacement = bone_xyz - parent_xyz
            displacement_normalized = displacement / np.linalg.norm(displacement)
            orig_offset = np.array(skel[bone]['offsets'])
            orig_offset_normalized = orig_offset / np.linalg.norm(orig_offset)
            rotation = rel_rotation(orig_offset_normalized, np.transpose(displacement_normalized))
            all_rotations[parent] = rotationMatrixToEulerAngles(rotation) * (180 / math.pi)

    return all_rotations


def rel_rotation(a, b):
    v = np.cross(a, b)
    c = np.dot(a, b)
    ssc = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Rotation = np.identity(3) + ssc + np.dot(ssc, ssc) * (1 / (1 + c))
    return Rotation


def xyz_to_rotations_debug(skel, position, hips_name, rotations_order):
    all_rotations = {}
    all_rotation_matrices = {}
    children_dict = get_child_dict(skel)
    while len(children_dict.keys()) - 1 > len(all_rotation_matrices.keys()):
        for bone in children_dict.keys():
            if bone == None:
                continue
            parent = skel[bone]['parent']
            if bone in all_rotation_matrices.keys():
                continue
            if parent not in all_rotation_matrices.keys() and parent != None:
                continue
            upper = parent
            parent_rot = np.identity(3)
            while upper != None:
                upper_rot = all_rotation_matrices[upper]
                parent_rot = np.dot(upper_rot, parent_rot)
                upper = skel[upper]['parent']

            children = children_dict[bone]
            children_xyz = np.zeros([len(children), 3])
            children_orig = np.zeros([len(children), 3])
            for i in range(len(children)):
                children_xyz[i, :] = np.array(position[children[i]]) - np.array(position[bone])
                children_orig[i, :] = np.array(skel[children[i]]['offsets'])
                divider = np.linalg.norm(children_xyz[i, :].astype(float))
                if divider != 0:
                    children_xyz[i, :] = children_xyz[i, :] * np.linalg.norm(children_orig[i, :]) / divider
                # allclose returns true if the arrays are equal within a tolerance.
                a = np.linalg.norm(children_xyz[i, :].astype(float))
                b = np.linalg.norm(children_orig[i, :])
                # print(bone)
                # print(a)
                # print(b)
                # assert np.allclose(a, b)

            parent_space_children_xyz = np.dot(children_xyz, parent_rot)
            rotation = kabsch(parent_space_children_xyz, children_orig)
            if bone == hips_name:
                # Order of translation
                multiplication_order = "r" + rotations_order
                angles = np.array(euler.mat2euler(rotation, "r" + rotations_order)) * (180.0 / math.pi)
                if rotations_order == 'zxy':
                    all_rotations[bone] = [
                        angles[1], angles[2], angles[0]]
                elif rotations_order == 'zyx':
                    all_rotations[bone] = [
                        angles[2], angles[1], angles[0]]
            else:
                # Different bvh can have different order of rotations.
                # We take the order of rotations, and we reverse it, to get the multiplication order,
                # As mat2euler returns Euler angles from rotation matrix for specified axis sequence.
                multiplication_order = "r" + rotations_order
                angles = np.array(euler.mat2euler(rotation, multiplication_order)) * (180.0 / math.pi)
                if rotations_order == 'zxy':
                    all_rotations[bone] = [
                        angles[1], angles[2], angles[0]]
                elif rotations_order == 'zyx':
                    all_rotations[bone] = [
                        angles[2], angles[1], angles[0]]

            all_rotation_matrices[bone] = rotation

    return all_rotation_matrices, all_rotations


def kabsch(p, q):
    A = np.dot(np.transpose(p), q)
    V, s, W = np.linalg.svd(A)
    A_2 = np.dot(np.dot(V, np.diag(s)), W)
    assert np.allclose(A, A_2)
    d = np.sign(np.linalg.det(np.dot(np.transpose(W), np.transpose(V))))
    s_2 = np.ones(len(s))
    s_2[len(s) - 1] = d
    rotation = np.dot(np.transpose(W), np.dot(np.diag(s_2), np.transpose(V)))
    assert isRotationMatrix(rotation)
    return np.transpose(rotation)
