import os
import numpy as np
import rasterio
from PIL import Image
import gc
from tqdm import tqdm
# from transform import parse_pose_list, parse_intrinsic_list

def is_valid_projection_old(world_points, K_w2c, pose_w2c, image_width, image_height):

    H, W, _ = world_points.shape
    world_points_reshaped = world_points.reshape(-1, 3)  
    valid_depth = world_points_reshaped[:, 2] > 0
    # pose_w2c:3*4
    R = pose_w2c[:3, :3]  
    T = pose_w2c[:3, 3]  
    camera_points = (R @ world_points_reshaped.T).T + T  # (N, 3)
    projected_points = (K_w2c @ camera_points.T).T  # (N, 3)
    projected_points /= projected_points[:, 2:3]  # (N, 2)
    u = projected_points[:, 0]  
    v = projected_points[:, 1] 
    
    # 判断投影是否在有效范围内
    valid_points = (u >= 0.2*image_width) & (u <= 0.8*image_width) & (v >= 0.2*image_height) & (v <= 0.8*image_height)
    # valid_points =   (u > 0) &(u < image_width)& (v >0) & (v < image_height)
    valid_points = valid_points & valid_depth
    
    return valid_points.reshape(H, W)

def is_valid_projection(world_points, K_w2c, pose_w2c, image_width, image_height, chunk_size=100):
    """
    验证投影是否有效，支持逐块处理以减少内存占用
    """
    H, W, _ = world_points.shape
    valid_points = np.zeros((H, W), dtype=bool)  # 结果数组

    # 分块处理
    for start_row in tqdm(range(0, H, chunk_size)):
        end_row = min(start_row + chunk_size, H)

        # 提取当前块
        world_points_chunk = world_points[start_row:end_row, :, :]
        world_points_reshaped = world_points_chunk.reshape(-1, 3)  # 当前块的形状 (chunk_size * W, 3)
        
        # 判断深度是否有效
        valid_depth = world_points_reshaped[:, 2] > 0
        
        # 计算投影
        R = pose_w2c[:3, :3]
        T = pose_w2c[:3, 3]
        camera_points = (R @ world_points_reshaped.T).T + T  # (N, 3)
        projected_points = (K_w2c @ camera_points.T).T  # (N, 3)
        projected_points /= projected_points[:, 2:3]  # (N, 2)
        u = projected_points[:, 0]
        v = projected_points[:, 1]
        
        # 判断投影是否在有效范围内
        valid_chunk = (u >= 0.2 * image_width) & (u <= 0.8 * image_width) & \
                      (v >= 0.2 * image_height) & (v <= 0.8 * image_height)
        valid_chunk = valid_chunk & valid_depth
        
        # 将结果存入总数组
        valid_points[start_row:end_row, :] = valid_chunk.reshape(end_row - start_row, W)

        # 显式释放内存
        del world_points_chunk, world_points_reshaped, valid_depth, camera_points, projected_points, valid_chunk
        gc.collect()

    return valid_points


def read_dsm(ref_dsm, npy_save_path):

    with rasterio.open(ref_dsm) as src:
        shape = src.shape
        world_points = np.empty((shape[0], shape[1], 3), dtype=np.float32)
        # 逐行处理
        for i in tqdm(range(shape[0])):
            dsm_row = src.read(1, window=rasterio.windows.Window(0, i, shape[1], 1))
            xs, ys = rasterio.transform.xy(src.transform, [i] * shape[1], range(shape[1]))
            world_points[i, :, 0] = xs
            world_points[i, :, 1] = ys
            world_points[i, :, 2] = dsm_row

            del dsm_row, xs, ys
            gc.collect()  # 调用垃圾回收
        gc.collect()
        # save
        np.save(npy_save_path, world_points)  
    return world_points


def generate_ref_map(query_intrinsics, query_poses, ref_dsm, ref_dom, ref_npy, crop_output):

    crop_output_rgb = os.path.join(crop_output, 'rgb')
    crop_output_npy = os.path.join(crop_output, 'npy')
    if not os.path.exists(crop_output_rgb):
        os.makedirs(crop_output_rgb)
    if not os.path.exists(crop_output_npy):
        os.makedirs(crop_output_npy)
 
    if not os.path.exists(ref_npy):
        world_points = read_dsm(ref_dsm, ref_npy)
    else:
        world_points = np.load(ref_npy)

    data = {}
    
   
    for name in tqdm(query_intrinsics.keys()):
        K_w2c = query_intrinsics[name]
        pose_w2c = query_poses[name]
        output_img_path = os.path.join(crop_output_rgb, f'{name}_dom.jpg')
        output_npy_path = os.path.join(crop_output_npy, f'{name}_dom.npy')
        # reference exist
        if os.path.exists(output_img_path) and os.path.exists(output_npy_path):
            data[name] = {
                'imgr_name':name+'_dom',
                'exr_name':name+'_dom'
            }
            continue
        width, height = K_w2c[0,2]*2, K_w2c[1,2]*2
        valid_points = is_valid_projection(world_points, K_w2c, pose_w2c, width, height)
        with rasterio.open(ref_dom) as data_dom:
            dom_data = data_dom.read([1, 2, 3])
            dom_data = np.moveaxis(dom_data, 0, -1) 
            # 获取有效点的边界框
            valid_coords = np.column_stack(np.where(valid_points))
            min_row, min_col = valid_coords.min(axis=0)
            max_row, max_col = valid_coords.max(axis=0)
            # valid area
            mask = np.zeros_like(valid_points, dtype=bool)
            mask[min_row:max_row+1, min_col:max_col+1] = True  # 设置有效区域为True
            # 
            valid_dom_img = np.zeros_like(dom_data)  
            valid_dom_img[mask] = dom_data[mask]  
            # 
            valid_dom_rgb_img = valid_dom_img  
            # 
            cropped_dom_img = valid_dom_rgb_img[min_row:max_row+1, min_col:max_col+1, :]
            # 
            Image.fromarray(cropped_dom_img).save(output_img_path) 
            valid_dsm = world_points[min_row:max_row+1, min_col:max_col+1, :]
            np.save(output_npy_path, valid_dsm)
            data[name] = {
                'imgr_name':name+'_dom',
                'exr_name':name+'_dom'
            }
    
    return data


# query_intrinsic = "/media/guan/data/sparse/ExtreBenchmark/data/query/syn/q_intrinsic.txt"
# query_poses = "/media/guan/data/sparse/ExtreBenchmark/data/query/syn/q_pose.txt" 
# ref_dsm = "/media/guan/data/sparse/ExtreBenchmark/data/ref/Production_4_DSM_merge.tif"
# ref_dom = "/media/guan/data/sparse/ExtreBenchmark/data/ref/Production_4_ortho_merge.tif"
# ref_npy = "/media/guan/data/sparse/ExtreBenchmark/data/ref/world_point.npy"
# crop_output = "/media/guan/data/sparse/ExtreBenchmark/data/query/syn"

# q_intrinsic = parse_intrinsic_list(query_intrinsic)
# q_pose = parse_pose_list(query_poses)

# generate_ref_map(q_intrinsic, q_pose, ref_dsm, ref_dom, ref_npy, crop_output)