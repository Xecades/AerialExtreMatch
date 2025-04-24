import os
import glob

# 设定图像文件夹路径
image_folder_path = '/media/guan/data/sparse/ExtreBenchmark/data/query/seq3/rgb/'  # 替换为实际路径
output_txt_path = '/media/guan/data/sparse/ExtreBenchmark/data/query/seq3/intrinsic.txt'  # 输出的txt文件名称

# 获取所有图像文件（假设是常见的图像格式）
image_files = glob.glob(image_folder_path + '*.JPG')

# 创建并写入txt文件
with open(output_txt_path, 'w') as f:
    for image_file in image_files:
        # 构造每一行的内容
        line = f"{image_file} PINHOLE 4056 3040 2893.37890625 2892.4970703125 2028.730906251585 1520.6780418291455\n"
        f.write(line)

print(f"已创建文件 {output_txt_path}，包含 {len(image_files)} 行。")
