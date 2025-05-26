import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2 (prerequisite!)
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor
import imageio
from pathlib import Path
import shutil
import cv2
import imageio


# ----------------------
# 參數設定
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='D:/Olia/結構化機器學習/SINet-master/SINet_Final.pth')
parser.add_argument('--test_save', type=str,
                    default='D:/Olia/結構化機器學習/SINet-master/Result/')  # 注意：不要結尾就寫死 CAMO
opt = parser.parse_args()

# ----------------------
# 模型載入
# ----------------------
model = SINet_ResNet50().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()
print("Model loaded:", any(p.requires_grad for p in model.parameters()))

# ----------------------
# 測試資料集設定
# ----------------------
datasets = ['CAMO', 'CHAMELEON', 'COD10K']
dataset_mae_dict = {}
all_mae_list = []

for dataset in datasets:
    print(f"\n>>> Testing on dataset: {dataset}")

    save_path = os.path.join(opt.test_save, dataset)
    if Path(save_path).exists():
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    image_root = f'D:/Olia/DATASET/COD10K/TestDataset/TestDataset/{dataset}/Imgs/'
    gt_root = f'D:/Olia/DATASET/COD10K/TestDataset/TestDataset/{dataset}/GT/'
    test_loader = test_dataset(image_root=image_root, gt_root=gt_root, testsize=opt.testsize)

    mae_list = []
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        with torch.no_grad():
            _, cam = model(image)
            cam = F.interpolate(cam, size=gt.shape, mode='bilinear', align_corners=True)
            cam = cam.sigmoid().data.cpu().numpy().squeeze()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam_uint8 = (cam * 255).astype(np.uint8)

        imageio.imsave(os.path.join(save_path, name), cam_uint8)

        # 評估 MAE
        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        mae_list.append(mae.item())

        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {:.4f}'.format(
            dataset, name, i + 1, test_loader.size, mae))

    # 每個資料集平均 MAE
    avg_mae = np.mean(mae_list)
    dataset_mae_dict[dataset] = avg_mae
    all_mae_list.extend(mae_list)

# ----------------------
# 結果總結輸出
# ----------------------
print("\n[Congratulations! Testing Done]\n")
print("===== [MAE Summary for SINet] =====")
for dataset, mae_val in dataset_mae_dict.items():
    print("Dataset: {:10s} | Average MAE: {:.4f}".format(dataset, mae_val))

if all_mae_list:
    print("-------------------------------")
    print("All Dataset Average MAE: {:.4f}".format(np.mean(all_mae_list)))
    print("==================================\n")