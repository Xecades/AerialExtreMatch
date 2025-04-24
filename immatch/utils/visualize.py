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


def plot_match_2view(q_path, r_path, matches, save_path,percent=0.2):

    ori_img = Image.open(q_path)
    ref_img = Image.open(r_path)
    
    # ori_img = q_path.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # ref_img = r_path.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # ori_img =q_path
    # ref_img = r_path

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
    axes[1].imshow(ref_img)
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




def plot_match_2viewtest(q_path, r_path, matches, save_path, percent=0.5):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from matplotlib import cm
    from skimage.measure import ransac
    from skimage.transform import FundamentalMatrixTransform
    from matplotlib.collections import LineCollection
        # 加载图像并获取尺寸
    ori_img = Image.open(q_path)
    ref_img = Image.open(r_path)
    # 获取图像尺寸
    w1, h1 = ori_img.size
    w2, h2 = ref_img.size

    # 拼接两张图像
    combined_img = Image.new('RGB', (w1 + w2, max(h1, h2)))  # 创建拼接后的图像
    combined_img.paste(ori_img, (0, 0))  # 左侧放置查询图像
    combined_img.paste(ref_img, (w1, 0))  # 右侧放置参考图像

    # 准备匹配点数据
    matches1 = np.column_stack((matches[:, 0], matches[:, 1]))
    matches2 = np.column_stack((matches[:, 2], matches[:, 3]))

    # 随机选择指定比例的匹配点
    num2plot = int(percent * len(matches1))
    selected_indices = np.random.choice(len(matches1), num2plot, replace=False)
    query_selected = matches1[selected_indices]
    ref_selected = matches2[selected_indices]

    # RANSAC处理
    inliers = None
    if len(query_selected) > 8:
        try:
            model, inliers = ransac(
                data=(query_selected, ref_selected),
                model_class=FundamentalMatrixTransform,
                min_samples=8,
                residual_threshold=1,
                max_trials=2000
            )
            inlier_count = np.sum(inliers)
        except Exception as e:
            print(f"RANSAC error: {e}")
            inliers = np.ones(len(query_selected), dtype=bool)

    # 可视化拼接图像和连线
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(combined_img)
    ax.axis("off")  # 关闭坐标轴

    # 绘制内点连线
    for i, (q_pt, r_pt) in enumerate(zip(query_selected, ref_selected)):
        if inliers[i]:  # 只绘制内点
            color = 'g'  # 内点连线为绿色
            ax.plot(
                [q_pt[0], r_pt[0] + w1],  # x 坐标：左图点到右图点
                [q_pt[1], r_pt[1]],       # y 坐标：左图点到右图点
                color=color, alpha=0.6, linewidth=1
            )
    text = f"Inliers: {inlier_count}"
    ax.text(10, 10, text, color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
    # 保存可视化后的图像
    plt.savefig(str(save_path), bbox_inches='tight', pad_inches=0, dpi=150)

