import torch
import numpy as np
import pycolmap

from immatch.utils.visualize import plot_match_2view

device = "cuda" if torch.cuda.is_available() else "cpu"

class DOMLocalizer:
    def __init__(self, K1, points_path, matches):

        self.K1 = K1
        self.world_points = torch.tensor(np.load(points_path)).to(device)
        self.matches = matches
    
    def interpolate_depth(self, pos):
        ids = torch.arange(0, pos.shape[0])
        
        h, w, _ = self.world_points.shape
        # !!! DSM 数据的格式是x和y，所以需要对应
        # depth是 H，W
        j = pos[:, 0] # W
        i = pos[:, 1] # H

        # 
        i_top_left = torch.clamp(torch.floor(i).long(), 0, h - 1)
        j_top_left = torch.clamp(torch.floor(j).long(), 0, w - 1)
        
        i_top_right = torch.clamp(torch.floor(i).long(), 0, h - 1)
        j_top_right = torch.clamp(torch.ceil(j).long(), 0, w - 1)
        
        i_bottom_left = torch.clamp(torch.ceil(i).long(), 0, h - 1)
        j_bottom_left = torch.clamp(torch.floor(j).long(), 0, w - 1)
        
        i_bottom_right = torch.clamp(torch.ceil(i).long(), 0, h - 1)
        j_bottom_right = torch.clamp(torch.ceil(j).long(), 0, w - 1)
        
        # 
        valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)
        valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)
        valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)
        valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)
        
        valid_corners = torch.min(
            torch.min(valid_top_left, valid_top_right),
            torch.min(valid_bottom_left, valid_bottom_right)
        )
        
        #
        i_top_left = i_top_left[valid_corners]
        j_top_left = j_top_left[valid_corners]
        i_top_right = i_top_right[valid_corners]
        j_top_right = j_top_right[valid_corners]
        i_bottom_left = i_bottom_left[valid_corners]
        j_bottom_left = j_bottom_left[valid_corners]
        i_bottom_right = i_bottom_right[valid_corners]
        j_bottom_right = j_bottom_right[valid_corners]
        
        # 
        valid_depth = torch.ones_like(valid_corners, dtype=torch.bool)
        for c in range(3):
            channel_valid = torch.min(
                torch.min(
                    self.world_points[i_top_left, j_top_left, c] > 0,
                    self.world_points[i_top_right, j_top_right, c] > 0
                ),
                torch.min(
                    self.world_points[i_bottom_left, j_bottom_left, c] > 0,
                    self.world_points[i_bottom_right, j_bottom_right, c] > 0
                )
            )
            valid_depth = torch.min(valid_depth, channel_valid)
        
        # 
        i_top_left = i_top_left[valid_depth]
        j_top_left = j_top_left[valid_depth]
        i_top_right = i_top_right[valid_depth]
        j_top_right = j_top_right[valid_depth]
        i_bottom_left = i_bottom_left[valid_depth]
        j_bottom_left = j_bottom_left[valid_depth]
        i_bottom_right = i_bottom_right[valid_depth]
        j_bottom_right = j_bottom_right[valid_depth]
        
        ids = ids.to(valid_depth.device)
        ids = ids[valid_depth]
        
        i = i[ids]
        j = j[ids]
        
        # 
        dist_i_top_left = i - i_top_left.double()
        dist_j_top_left = j - j_top_left.double()
        w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
        w_top_right = (1 - dist_i_top_left) * dist_j_top_left
        w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
        w_bottom_right = dist_i_top_left * dist_j_top_left
       
        interpolated_depths = []
        for c in range(3):
            interpolated_depth = (
                w_top_left * self.world_points[i_top_left, j_top_left, c] +
                w_top_right * self.world_points[i_top_right, j_top_right, c] +
                w_bottom_left * self.world_points[i_bottom_left, j_bottom_left, c] +
                w_bottom_right * self.world_points[i_bottom_right, j_bottom_right, c]
            )
            interpolated_depths.append(interpolated_depth)
        
        # 
        interpolated_depth = torch.stack(interpolated_depths, dim=-1)
        
        pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)
        
        return interpolated_depth.cpu().numpy(), pos, ids.cpu().numpy()


    def localize(self, points3D, points2D):

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
    
    def main_localize(self, qpath, qpose, rot_R=None):

        matches_ref = np.column_stack((self.matches[:, 2], self.matches[:, 3]))
        world_points, _, valid = self.interpolate_depth(torch.tensor(matches_ref).to('cuda'))
        matches_query = np.column_stack((self.matches[:, 0], self.matches[:, 1]))
        if rot_R is not None:
            ones_column = np.ones((matches_query.shape[0], 1))
            query_pts2d_homo = np.hstack((matches_query, ones_column))
            query_pts2d_homo = np.dot(np.linalg.inv(rot_R), query_pts2d_homo.T)
            matches_query = (query_pts2d_homo[:2, ] / query_pts2d_homo[2,:]).T
        matches_query = matches_query[valid]


        # TODO 3D->2D vis, rot matches
        matches_compute = self.project_3d_to_2d(world_points, self.K1, qpose)
        plot_match_2view(qpath, matches_query, matches_compute, '1.jpg')


        ret = self.localize(world_points, matches_query)
        if ret is not None:
            # w2c results
            pred_matrix = ret['cam_from_world'].matrix()
        
        else:
            return None
        
        return pred_matrix
    
    def eval(self, pred_matrix, gt_pose):

        if pred_matrix is None:
            return np.nan, 180.0
        
        gt_pose = np.linalg.inv(gt_pose)
        gt_t = gt_pose[:3, 3]
        gt_R = gt_pose[:3, :3]
        pred_t = -pred_matrix[:3, :3].T @ pred_matrix[:3, 3]
        pred_R = pred_matrix[:3, :3].T
        e_t = np.linalg.norm(gt_t - pred_t, axis=0)
        cos = np.clip((np.trace(np.dot(gt_R.T, pred_R)) - 1) / 2, -1., 1.)
        e_R = np.rad2deg(np.abs(np.arccos(cos)))

        return e_t, e_R