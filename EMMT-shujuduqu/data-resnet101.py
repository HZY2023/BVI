# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
# import os
# import numpy as np
# import pandas as pd
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
#
# # 自定义数据集
# class ImageDataset(Dataset):
#     def __init__(self, dataframe, image_folder, transform=None):
#         self.dataframe = dataframe
#         self.image_folder = image_folder
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.dataframe)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.image_folder, self.dataframe.iloc[idx]['image']['path'])
#         image = Image.open(img_name).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image
#
# # 文件路径
# parquet_file = '/gb/HZY/EMMT/EMMT-test/0000.parquet'
# image_folder = '/gb/HZY/EMMT/EMMT-test/images'
#
# # 使用pandas读取parquet文件
# df = pd.read_parquet(parquet_file, engine='pyarrow')
#
# # 定义预处理操作
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# # 创建数据集
# dataset = ImageDataset(dataframe=df, image_folder=image_folder, transform=preprocess)
#
# # 创建DataLoader
# data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
#
# # 确保CUDA可用于加速
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 加载预训练的ResNet-101模型
# model = models.resnet101(pretrained=True)
# model = torch.nn.Sequential(*(list(model.children())[:-2])).to(device)
# model.eval()
#
# features_list = []
#
# # 批量处理图像
# for inputs in data_loader:
#     inputs = inputs.to(device)
#     with torch.no_grad():
#         features = model(inputs)
#         features = torch.nn.functional.adaptive_avg_pool2d(features, (7, 7))
#         features_list.append(features.view(features.size(0), 49, 2048).cpu().numpy())
#
# # 将特征列表转换为numpy数组
# features_array = np.vstack(features_list)
# np.save(r'/gb/HZY/EMMT/EMMT-train/test.npy', features_array)
# # print(features_array.shape)


import torch
import numpy as np
features = []
numpy_feature = ['/gb/HZY/EMMT/EMMT-train/train_1.npy', '/gb/HZY/EMMT/EMMT-train/train_2.npy', '/gb/HZY/EMMT/EMMT-train/train_3.npy',  '/gb/HZY/EMMT/EMMT-train/train_4.npy',]
for i in numpy_feature:
    feature = np.load(i)
    features.append(feature)
final_feature = np.concatenate(features, axis=0)
np.save('/gb/HZY/EMMT/EMMT-train/train.npy',final_feature)