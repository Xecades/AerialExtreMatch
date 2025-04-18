import numpy as np
from scipy.spatial.transform import Rotation as R
import pyproj
import math
import cv2

def get_CRS():
    wgs84 = pyproj.CRS('EPSG:4326')
    cgcs2000 = pyproj.CRS('EPSG:4547')  # 长沙
    return wgs84, cgcs2000

def cgcs2000towgs84(c2w_t):
    """Convert coordinates from CGCS2000 to WGS84.
    
    Args:
        c2w_t (list): [x, y, z] in CGCS2000 format
    """
    x, y = c2w_t[0], c2w_t[1]
    
    wgs84, cgcs2000 = get_CRS()
    
    transformer = pyproj.Transformer.from_crs(cgcs2000, wgs84, always_xy=True)
    lon, lat = transformer.transform(x, y)
    height = c2w_t[2]
    return [lon, lat, height]

def convert_quaternion_to_euler(q_c2w):
    """
    Convert a quaternion to Euler angles in 'xyz' order, with angles in degrees.
    
    Parameters:
    qvec (numpy.ndarray): The quaternion vector as [x, y, z, w].
    R (numpy.matrix, optional): If provided, this rotation matrix will be used
                                 instead of calculating it from the quaternion.
    
    Returns:
    numpy.ndarray: The Euler angles in 'xyz' order in degrees.
    """
    # Convert the quaternion in COLMAP format [QW, QX, QY, QZ] 
    # to the correct order for scipy's Rotation object: [x, y, z, w]
    q_xyzw = [np.float64(q_c2w[1]), np.float64(q_c2w[2]), np.float64(q_c2w[3]), np.float64(q_c2w[0])] 
    
    # Create a Rotation object from the quaternion
    ret = R.from_quat(q_xyzw)
    
    # Convert the quaternion to Euler angles in 'xyz' order, with angles in degrees
    euler_xyz = ret.as_euler('xyz', degrees=True)    
    euler_xyz[0] = euler_xyz[0] + 180 
   
    return list(euler_xyz)

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotation_matrix_x(angle_deg):
    angle_rad = np.radians(angle_deg)
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(angle_rad), -math.sin(angle_rad)],
        [0, math.sin(angle_rad), math.cos(angle_rad)]
    ])
    return R_x

def parse_pose_list(path):
    """
    poses:w2c
    """
    poses = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')

            info = line.split(' ')
            name = info[0].split('.')[0]
            q, t = np.split(np.array(info[1:], float), [4])
            R = np.asmatrix(qvec2rotmat(q))
            # R = np.asmatrix(qvec2rotmat(q)).transpose()

            T = np.identity(4)
            T[0:3,0:3] = R
            T[0:3,3] = t
            # T[0:3,3] = -R.dot(t)
            poses[name] = T
    
    assert len(poses) > 0
    return poses


def parse_intrinsic_list(path):
    """
    intrinsic:w2c
    """

    images = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0:
                continue

            info = line.split(' ')
            name = info[0].split('.')[0]
            _,_,fx,fy,cx,cy = list(map(float, info[2:8]))[:]

            K = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ])
            images[name] = K
    
    return images

def parse_pose_osg(path):

    poses = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            info = line.split(' ')
            name = info[0].split('.')[0]
            q_w2c, t_w2c = np.split(np.array(info[1:], float), [4])
            qmat = qvec2rotmat(q_w2c)
            qmat = qmat.T
            q_c2w = rotmat2qvec(qmat)
            euler_angles = convert_quaternion_to_euler(q_c2w)
            R_c2w = np.asmatrix(qvec2rotmat(q_w2c)).transpose()  
            t_c2w = np.array(-R_c2w.dot(t_w2c))  
            translation = cgcs2000towgs84(t_c2w[0])
            poses[name] = [euler_angles, translation]
            # print([euler_angles, translation])
            # import pdb;pdb.set_trace()
    
    return poses


def rotate_image(image, angle):
    """    
    参数:
    - image: 输入图像 (numpy array)
    - angle: 旋转角度 (度)
    
    返回:
    - 旋转后的图像 (numpy array)
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    new_w = int(h * abs(np.sin(np.radians(angle))) + w * abs(np.cos(np.radians(angle))))
    new_h = int(w * abs(np.sin(np.radians(angle))) + h * abs(np.cos(np.radians(angle))))

    # 构造旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    M[0, 2] += (new_w - w) // 2
    M[1, 2] += (new_h - h) // 2

    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

    additional_row = np.array([[0, 0, 1]])
    M = np.vstack((M, additional_row))

    return rotated, M