import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import tqdm
from concurrent.futures import ThreadPoolExecutor

# 记录已处理的数据的文件路径
resume_file = 'resume_checkpoint.txt'

# 定义一个全局变量，用于记录当前已处理的文件夹
current_folder = None

# 初始化或加载已处理的文件夹信息
if os.path.exists(resume_file):
    with open(resume_file, 'r') as f:
        current_folder = f.read().strip()

# Gaussian Kernel
class GaussianKernel(nn.Module):
    def __init__(self, sigma):
        super(GaussianKernel, self).__init__()
        self.sigma = sigma
        self.weight = nn.Parameter(torch.randn(1, 49, 2048))  # 添加权重参数

    def forward(self, x1, x2):
        # 计算两个张量之间的欧几里得距离的平方
        distance_squared = torch.sum((x1 - x2) ** 2, dim=1)
        # 计算高斯核函数的值
        kernel_value = torch.exp(-distance_squared / (2 * self.sigma**2))
        return kernel_value

# Custom Model for Multi-Kernel Learning
class CustomModel(nn.Module):
    def __init__(self, num_features, num_gaussian_kernels, final_dim, input_dim):
        super(CustomModel, self).__init__()
        self.num_gaussian_kernels = num_gaussian_kernels
        self.final_dim = final_dim
        self.input_dim = input_dim
        self.gaussian_kernels = nn.ModuleList(
            [GaussianKernel(sigma) for sigma in range(1, num_gaussian_kernels + 1)])

        self.weights = nn.Parameter(torch.ones(num_gaussian_kernels))
        self.bn = nn.BatchNorm1d(num_gaussian_kernels)
        self.relu = nn.ReLU()

    def forward(self, x):
        representations = []
        sigmas = [1.0, 2.0, 3.0, 4.0, 5.0]

        # 对每个σ值应用高斯核函数并获得五组特征表示
        for sigma in sigmas:
            kernel = GaussianKernel(sigma)
            features_per_sigma = []

            # 对每个图片应用高斯核
            for i in range(x.size(0)):  # 遍历图片数量
                feature = kernel(x[i], self.gaussian_kernels[0].weight)  # 使用第一个高斯核的权重作为参考
                features_per_sigma.append(feature.unsqueeze(0).repeat(1,49,1))

            representations.append(torch.cat(features_per_sigma, dim=0))

        final_representation = torch.cat(representations, dim=0)  # 沿着特征维度连接

        return final_representation


# Bilinear Interaction Module
class BilinearInteraction(nn.Module):
    def __init__(self, input_dim):
        super(BilinearInteraction, self).__init__()
        self.weight_matrix = nn.Parameter(torch.randn(49, 49))  # 权重矩阵

    def forward(self, x1, x2):
        x1_transposed = x1.transpose(0, 1)  # 转置第一个输入
        interaction = torch.mm(x1_transposed, self.weight_matrix) *  x2.transpose(0,1)
        return interaction

# MyModel for Bilinear Interaction
class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.interaction1 = BilinearInteraction(input_dim)
        self.interaction2 = BilinearInteraction(input_dim)
        self.interaction3 = BilinearInteraction(input_dim)
        self.interaction4 = BilinearInteraction(input_dim)
        self.interaction5 = BilinearInteraction(input_dim)
        self.interaction6 = BilinearInteraction(input_dim)
        self.interaction7 = BilinearInteraction(input_dim)
        self.interaction8 = BilinearInteraction(input_dim)
        self.interaction9 = BilinearInteraction(input_dim)
        self.interaction10 = BilinearInteraction(input_dim)
        self.Linear = nn.Linear(input_dim, output_dim)

    def forward(self, *features):
        interaction1 = self.interaction1(features[0], features[1])
        interaction2 = self.interaction2(features[0], features[2])
        interaction3 = self.interaction3(features[0], features[3])
        interaction4 = self.interaction4(features[0], features[4])
        interaction5 = self.interaction5(features[1], features[2])
        interaction6 = self.interaction6(features[1], features[3])
        interaction7 = self.interaction7(features[1], features[4])
        interaction8 = self.interaction8(features[2], features[3])
        interaction9 = self.interaction9(features[2], features[4])
        interaction10 = self.interaction10(features[3], features[4])

        interaction = []

        interaction1 = F.relu(interaction1)
        interaction.append(interaction1.unsqueeze(0))
        interaction2 = F.relu(interaction2)
        interaction.append(interaction2.unsqueeze(0))
        interaction3 = F.relu(interaction3)
        interaction.append(interaction3.unsqueeze(0))
        interaction4 = F.relu(interaction4)
        interaction.append(interaction4.unsqueeze(0))
        interaction5 = F.relu(interaction5)
        interaction.append(interaction5.unsqueeze(0))
        interaction6 = F.relu(interaction6)
        interaction.append(interaction6.unsqueeze(0))
        interaction7 = F.relu(interaction7)
        interaction.append(interaction7.unsqueeze(0))
        interaction8 = F.relu(interaction8)
        interaction.append(interaction8.unsqueeze(0))
        interaction9 = F.relu(interaction9)
        interaction.append(interaction9.unsqueeze(0))
        interaction10 = F.relu(interaction10)
        interaction.append(interaction10.unsqueeze(0))

        interaction_sum = torch.cat(interaction,dim=0).mean(dim=0,keepdim=True)
        interaction_sum = 1/2 *interaction_sum + 1/2 * (interaction1+interaction2+interaction3+interaction4+interaction5+interaction6+interaction7+interaction8+interaction9+interaction10)

        return  interaction_sum.transpose(1,2)

# 示例用法
if __name__ == "__main__":
    num_features = 2048
    num_gaussian_kernels = 5
    final_dim = 2048
    input_dim = 10
    output_dim = 1
    model = CustomModel(num_features, num_gaussian_kernels, final_dim, input_dim)

    path = 'E:/all-images-listdir'
    save_path = 'E:/kernel-biliner-final'
    a = os.listdir(path)
    concatenated_features = []
    interaction_num = []

    # 找到上次中断的位置
    if current_folder is not None:
        try:
            start_index = a.index(current_folder)
            a = a[start_index + 1:]  # 从上次中断的文件夹之后开始处理
        except ValueError:
            pass

    # 创建进度条
    with tqdm.tqdm(total=len(a)) as pbar:
        for each_folder in a:
            # 更新当前文件夹信息
            current_folder = each_folder
            # 保存当前已处理的文件夹信息
            with open(resume_file, 'w') as f:
                f.write(current_folder)

            # 检查文件是否已存在
            output_file = os.path.join(save_path, each_folder+'.npy')
            if os.path.exists(output_file):
                pbar.update(1)
                continue

            new_each_feature = os.listdir(os.path.join(path, each_folder))
            features_per_folder = []

            def process_image(each_feature):
                each_feature_tensor = torch.tensor(np.load(os.path.join(path, each_folder, each_feature)))
                features_per_folder.append(each_feature_tensor)

            # 使用多线程处理每张图像
            with ThreadPoolExecutor() as executor:
                executor.map(process_image, new_each_feature)

            result = torch.cat(features_per_folder, dim=0)

            output = model(result)

            interaction_model = MyModel(input_dim, output_dim)
            interaction_result = interaction_model(*output)

            interaction_numpy = interaction_result.cpu().detach().numpy()
            np.save(output_file, interaction_numpy)

            pbar.update(1)  # 更新进度条
