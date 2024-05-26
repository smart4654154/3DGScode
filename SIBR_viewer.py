import subprocess

# output保存路径
# model_path = r'D:\ZhangZhilong\Download\3dgslaptting\gaussian-splatting\output\pig'

# model_path = r'D:\ZhangZhilong\Download\3dgslaptting\gaussian-splatting\output\nerfdata_room'
# model_path = r'D:\ZhangZhilong\Download\3dgslaptting\gaussian-splatting\output\nerfdata_ship'
model_path = r'E:\WSL\zzl\data\mega_nerfdata\3dgs_origin\magenerf_300_3dgs\3wresult'
# model_path = r'E:\WSL\zzl\data\mega_nerfdata\3dgs_origin\magenerf_102_3dgs\magenerf_102_3dgs'

# model_path = r'D:\ZhangZhilong\Download\3dgslaptting\gaussian-splatting\output\makesi\makesi_20w_540p'
# model_path = r'D:\ZhangZhilong\Download\3dgslaptting\gaussian-splatting\output\makesixueyuan_5w_1080p'
# model_path = r'D:\ZhangZhilong\Download\3dgslaptting\gaussian-splatting\output\kongzi_30000_540p'

# 脚本执行
command = f'SIBR_gaussianViewer_app.exe -m {model_path}'#
# -m是指定模型存放的文件夹
run_path = 'external/viewers/bin'
subprocess.run(command, shell=True, cwd=run_path)
