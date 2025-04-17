
import torch
import numpy as np
import pycolmap



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

class QueryLocalizer:
    def __init__(self, K1, K2, pose2, depth2, matches):

        self.K1 = K1
        self.K2 = K2
        self.pose2 = pose2
        self.depth2 = depth2
        self.matches = matches
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def interpolate_depth(self, pos, depth):
        ids = torch.arange(0, pos.shape[0])
        if depth.ndim != 2:
            if depth.ndim == 3:
                depth = depth[:,:,0]
            else:
                raise Exception("Invalid depth image!")
        h, w = depth.size()
        
        i = pos[:, 0]
        j = pos[:, 1]

        i_top_left = torch.clamp(torch.floor(i).long(), 0, h - 1)
        j_top_left = torch.clamp(torch.floor(j).long(), 0, w - 1)

        i_top_right = torch.clamp(torch.floor(i).long(), 0, h - 1)
        j_top_right = torch.clamp(torch.ceil(j).long(), 0, w - 1)

        i_bottom_left = torch.clamp(torch.ceil(i).long(), 0, h - 1)
        j_bottom_left = torch.clamp(torch.floor(j).long(), 0, w - 1)

        i_bottom_right = torch.clamp(torch.ceil(i).long(), 0, h - 1)
        j_bottom_right = torch.clamp(torch.ceil(j).long(), 0, w - 1)
        
        valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)
        valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)
        valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)
        valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

        valid_corners = torch.min(
            torch.min(valid_top_left, valid_top_right),
            torch.min(valid_bottom_left, valid_bottom_right)
        )

        i_top_left = i_top_left[valid_corners]
        j_top_left = j_top_left[valid_corners]

        i_top_right = i_top_right[valid_corners]
        j_top_right = j_top_right[valid_corners]

        i_bottom_left = i_bottom_left[valid_corners]
        j_bottom_left = j_bottom_left[valid_corners]

        i_bottom_right = i_bottom_right[valid_corners]
        j_bottom_right = j_bottom_right[valid_corners]

        # Valid depth
        valid_depth = torch.min(
            torch.min(
                depth[i_top_left, j_top_left] > 0,
                depth[i_top_right, j_top_right] > 0
            ),
            torch.min(
                depth[i_bottom_left, j_bottom_left] > 0,
                depth[i_bottom_right, j_bottom_right] > 0
            )
        )

        i_top_left = i_top_left[valid_depth]
        j_top_left = j_top_left[valid_depth]

        i_top_right = i_top_right[valid_depth]
        j_top_right = j_top_right[valid_depth]

        i_bottom_left = i_bottom_left[valid_depth]
        j_bottom_left = j_bottom_left[valid_depth]

        i_bottom_right = i_bottom_right[valid_depth]
        j_bottom_right = j_bottom_right[valid_depth]
        # vaild index
        ids = ids.to(valid_depth.device)
        ids = ids[valid_depth]
        
        i = i[ids]
        j = j[ids]
        dist_i_top_left = i - i_top_left.double()
        dist_j_top_left = j - j_top_left.double()
        w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
        w_top_right = (1 - dist_i_top_left) * dist_j_top_left
        w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
        w_bottom_right = dist_i_top_left * dist_j_top_left

        #depth is got from interpolation
        interpolated_depth = (
            w_top_left * depth[i_top_left, j_top_left] +
            w_top_right * depth[i_top_right, j_top_right] +
            w_bottom_left * depth[i_bottom_left, j_bottom_left] +
            w_bottom_right * depth[i_bottom_right, j_bottom_right]
        )

        pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

        return [interpolated_depth, pos, ids]
    
    def read_valid_depth(self, mkpts1r, depth=None):
        
        mkpts1r_a = torch.unsqueeze(mkpts1r[:,0],0)
        mkpts1r_b =  torch.unsqueeze(mkpts1r[:,1],0)
        mkpts1r_inter = torch.cat((mkpts1r_b ,mkpts1r_a),0).transpose(1,0)

        depth, _, valid = self.interpolate_depth(mkpts1r_inter, depth)

        return depth, valid
    
    def get_Points3D(self, depth, R, t, K, points):   # points[n,2]
    
    
        if points.shape[-1] != 3:
            points_2D = torch.cat([points, torch.ones_like(points[ :, [0]])], dim=-1)
            points_2D = points_2D.T  
        t = torch.unsqueeze(t,-1).repeat(1, points_2D.shape[-1])
        Points_3D = R @ K @ (depth * points_2D) + t   
        return (Points_3D.T).cpu().numpy().astype(np.float64)    #[3,n]
    
    def localize(self, points3D, points2D):
        
        self.K1 = self.K1.cpu().numpy()
        fx, fy, cx, cy = self.K1[0,0], self.K1[1,1], self.K1[0,2], self.K1[1,2]
        width, height = self.K1[0,2] * 2, self.K1[1,2] * 2
        
        camera = pycolmap.Camera(
            model="PINHOLE",
            width=int(width),
            height=int(height),
            params=[fx, fy, cx, cy]
        )

        estimation_options = pycolmap.AbsolutePoseEstimationOptions()
        estimation_options.ransac.max_error = 5
        estimation_options.ransac.min_inlier_ratio = 0.01
        estimation_options.ransac.min_num_trials = 1000
        estimation_options.ransac.max_num_trials = 100000
        estimation_options.ransac.confidence = 0.9999

        ret = pycolmap.estimate_absolute_pose(
            points2D=points2D.astype(np.float64),
            points3D=points3D.astype(np.float64),
            camera=camera,
            estimation_options=estimation_options,
        )
        
        return ret  
    def project_3d_to_2d(self, points_3D, K, w2c):
 
        # 确保 3D 点是齐次坐标
        num_points = points_3D.shape[0]
        points_3D_hom = np.hstack((points_3D, np.ones((num_points, 1))))  # 转为齐次坐标 (n, 4)

        # 使用外参将 3D 点变换到相机坐标系
        points_camera = (w2c @ points_3D_hom.T).T  # (n, 4) -> (n, 4)

        # 丢弃齐次坐标的最后一列，得到相机坐标系中的 3D 点 (n, 3)
        points_camera = points_camera[:, :3]

        # 使用内参将相机坐标系中的点投影到图像平面
        points_2D_hom = (K @ points_camera.T).T  # (3x3) x (n, 3).T -> (n, 3)

        # 归一化齐次坐标，转换为非齐次的 2D 点坐标
        points_2D = points_2D_hom[:, :2] / points_2D_hom[:, 2:]

        return points_2D
        
    def main_localize(self, matchesq, qpose):
        
        # read valid depth
        depth, valid = self.read_valid_depth(self.matches, self.depth2)
        # reference pose
        self.poses2 = torch.tensor(self.pose2)
        poses2_c2w = self.poses2.inverse()
        # compute 3D points
        K2_c2w = self.K2.inverse()
        points_3D = self.get_Points3D(
                    depth,
                    poses2_c2w[:3, :3],
                    poses2_c2w[:3, 3],
                    K2_c2w,
                    self.matches,
                )
        # 怀疑point3D点计算有误，3D点投影过去
        q_matches = self.project_3d_to_2d(points_3D, self.K1.squeeze(0).cpu().numpy(), qpose)

        points2D = matchesq[valid].cpu().numpy()
        ret = self.localize(points_3D, points2D)
        if ret is not None:
            # w2c results
            pred_matrix = ret['cam_from_world'].matrix()
        
        else:
            return None
        
        return pred_matrix
    
    def eval(self, pred_matrix, gt_pose):

        if pred_matrix is None:
            return np.nan, 180.0
        
        gt_pose = gt_pose.inverse().cpu().numpy()
        gt_t = gt_pose[:3, 3]
        gt_R = gt_pose[:3, :3]
        pred_t = -pred_matrix[:3, :3].T @ pred_matrix[:3, 3]
        pred_R = pred_matrix[:3, :3].T
        e_t = np.linalg.norm(gt_t - pred_t, axis=0)
        cos = np.clip((np.trace(np.dot(gt_R.T, pred_R)) - 1) / 2, -1., 1.)
        e_R = np.rad2deg(np.abs(np.arccos(cos)))
        return e_t, e_R


        