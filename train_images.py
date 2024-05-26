import os
import subprocess

# 图片集所在的绝对路径
print("注意，你的照片要放到dataname/input文件夹下，不要放到其他文件夹下")
# images_path = r"D:\ZhangZhilong\Download\3dgslaptting\data\test\input"
images_path = r"D:\ZhangZhilong\Download\3dgslaptting\data\实验保存\nerfdata_vasedeck\input"


# 上一级文件夹所在路径
folder_path = os.path.dirname(images_path)

# 脚本运行
# COLMAP估算相机位姿
# command = f'python convert.py -s {folder_path}'
# subprocess.run(command, shell=True)
# 模型训练脚本，模型会保存在output路径下
command = f'python train.py -s {folder_path} --iterations 30000 '
subprocess.run(command, shell=True)
#注意，--start_checkpoint D:\...