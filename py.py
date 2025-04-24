import os
from PIL import Image

def create_gif_from_jpgs(folder_path, output_gif_path, duration=500):
    """
    将文件夹中的所有 .jpg 文件拼接成一个 GIF 动画。
    
    参数:
        folder_path (str): 包含 .jpg 文件的文件夹路径。
        output_gif_path (str): 输出 GIF 文件的路径。
        duration (int): 每帧的持续时间（毫秒）。
    """
    # 获取文件夹中的所有 .jpg 文件
    jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    jpg_files.sort()  # 按文件名排序（可根据需要调整排序方式）

    # 检查是否有 .jpg 文件
    if not jpg_files:
        print("文件夹中没有找到 .jpg 文件！")
        return

    # 加载所有 .jpg 文件为 PIL 图像对象
    images = [Image.open(os.path.join(folder_path, f)) for f in jpg_files]

    # 创建 GIF 动画
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],  # 添加后续帧
        duration=duration,         # 每帧持续时间（毫秒）
        loop=0                     # 循环次数，0 表示无限循环
    )
    print(f"GIF 动画已保存到: {output_gif_path}")

# 使用示例
folder_path = "/media/guan/ZX1/ExeBenchmark/results/roma_ort/test/"  # 替换为你的文件夹路径
output_gif_path = "/media/guan/ZX1/ExeBenchmark/results/roma_ort/test/output.gif"       # 替换为输出 GIF 的路径
create_gif_from_jpgs(folder_path, output_gif_path, duration=500)
