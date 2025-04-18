from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm
from PIL import Image

def plot_match(q_path, rot_path, ref_path, matches, save_path, name, rot_R = None, percent=0.2):

    query_img = Image.open(rot_path)
    map_img = Image.open(ref_path)
    ori_img = Image.open(q_path)

    num2plot = int(percent * len(matches))
    selected_indices = np.random.choice(len(matches), num2plot, replace=False)

    qmatches = np.column_stack((matches[:, 0], matches[:, 1]))
    rmatches = np.column_stack((matches[:, 2], matches[:, 3]))
    if rot_R is not None:
        ones_column = np.ones((qmatches.shape[0], 1))
        query_pts2d_homo = np.hstack((qmatches, ones_column))
        query_pts2d_homo = np.dot(np.linalg.inv(rot_R), query_pts2d_homo.T)
        matches_query_rot = (query_pts2d_homo[:2, ] / query_pts2d_homo[2,:]).T
    
    query_selected = qmatches[selected_indices]
    new_selected = matches_query_rot[selected_indices]
    map_selected = rmatches[selected_indices]
    
    Color = np.zeros(num2plot)
    col_max = np.max(query_selected, axis=0)
    x = query_selected[:,0]/col_max[0]
    y = query_selected[:,1]/col_max[1]
    Color = np.sqrt(x**2+y**2)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 8), dpi=150)
    axes[0].imshow(query_img)
    axes[1].imshow(ori_img)
    axes[2].imshow(map_img)
     # 使用一次性调用scatter绘制所有点
    axes[0].scatter(query_selected[:, 0], query_selected[:, 1], c=Color, cmap=cm.hsv,s=4)
    axes[1].scatter(new_selected[:, 0], new_selected[:, 1], c=Color, cmap=cm.hsv,s=4)
    axes[2].scatter(map_selected[:, 0], map_selected[:, 1], c=Color, cmap=cm.hsv, s=4)

    # 隐藏坐标轴
    for ax in axes:
        ax.axis('off')
    
    # 调整布局
    plt.tight_layout()

    write_path = save_path + '/' + name + '.jpg'
    plt.savefig(str(write_path), bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_match_2view(q_path, matches1, matches2, save_path,percent=0.2):

    ori_img = Image.open(q_path)

    num2plot = int(percent * len(matches1))
    selected_indices = np.random.choice(len(matches1), num2plot, replace=False)
    
    query_selected = matches1[selected_indices]
    new_selected = matches2[selected_indices]
    
    Color = np.zeros(num2plot)
    col_max = np.max(query_selected, axis=0)
    x = query_selected[:,0]/col_max[0]
    y = query_selected[:,1]/col_max[1]
    Color = np.sqrt(x**2+y**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), dpi=150)
    axes[0].imshow(ori_img)
    axes[1].imshow(ori_img)
     # 使用一次性调用scatter绘制所有点
    axes[0].scatter(query_selected[:, 0], query_selected[:, 1], c=Color, cmap=cm.hsv,s=4)
    axes[1].scatter(new_selected[:, 0], new_selected[:, 1], c=Color, cmap=cm.hsv,s=4)

    # 隐藏坐标轴
    for ax in axes:
        ax.axis('off')
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(str(save_path), bbox_inches='tight', pad_inches=0)
    plt.close()