import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from natsort import natsorted
import torch
import os
import imageio

with open("/home/a/acw799/data/put-block-in-bowl-seen-colors-test/color/000000-10001.pkl", "rb") as f:
    data = pkl.load(f)

print(type(data))
print(data.shape)

color_dir = "/home/a/acw799/concept-fusion/data/bowl/view/color"

for sample_idx, sample in enumerate(data):
    print(f"Sample {sample_idx}:")
    sample_dir = f"{sample_idx:05}"
    os.makedirs(color_dir, exist_ok=True)
    
    # 遍历每个时间帧
    for frame_idx, frame in enumerate(sample):
        print(f" Saving Frame {frame_idx}:")
        # save .png files
        color_filename = os.path.join(color_dir, f"{frame_idx:05}.png")
        # 保存彩色图像
        imageio.imwrite(color_filename, frame)  # color(H, W, 3)

        # # Matplotlib 显示图像
        # plt.imshow(frame.astype('uint8'))  # 转为 uint8 类型以正确显示
        # plt.title(f"Sample {sample_idx}, Frame {frame_idx}")
        # plt.axis('off')  # 去掉坐标轴
        # plt.show()
    break