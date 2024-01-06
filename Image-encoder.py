import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import tqdm
from itertools import combinations




def generate_image_feature_pairs(root_folder):
    # 获取所有文件夹的列表
    folders = [folder for folder in os.listdir(root_folder)]

    # 存储组合后的图像特征对
    image_feature_pairs = []

    if len(folders) == 1:
        single_feature = np.load(folders[0])
        noise = np.random.normal(0, 0.1, single_feature.shape)
        noisy_data = single_feature+noise
        image_feature_pairs.append(noisy_data)

        # 将图像特征两两组合成一对
    pairs = list(combinations(folders, 2))


    for pair in pairs:
        image_feature_pairs.append((os.path.join(root_folder, pair[0]), os.path.join(root_folder, pair[1])))

    return image_feature_pairs


# Gaussian Kernel
class GaussianKernel(nn.Module):
    def __init__(self, num_regions, num_features):
        super(GaussianKernel, self).__init__()
        self.sigma = nn.Parameter(torch.randn(1)* 100)  # 方差作为可训练参数
        self.mean = nn.Parameter(torch.randn(1))  # 均值作为可训练参数
        self.num_regions = num_regions
        self.num_features = num_features

    def forward(self, x1, x2):
        # 将输入张量重塑为 (num_regions, num_features)
        x1_reshaped = x1.view(self.num_regions, self.num_features)
        x2_reshaped = x2.view(self.num_regions, self.num_features)

        # 计算两个张量之间的欧几里得距离的平方
        distance_squared = torch.sum((x1_reshaped.unsqueeze(1) - x2_reshaped.unsqueeze(0) - self.mean) ** 2, dim=2)

        distance_squared = torch.clamp(distance_squared, min=1e-6, max=1e6)  # 限制距离的平方在一个合理的范围内

        # 计算高斯核函数的值
        kernel_value = torch.exp(-distance_squared / (2 * self.sigma ** 2))

        # 在第一个维度上执行 softmax 操作
        softmaxed_similarity_matrix = torch.softmax(kernel_value, dim=1)

        return softmaxed_similarity_matrix


# Custom Model for Multi-Kernel Learning
class CustomModel(nn.Module):
    def __init__(self, num_regions, num_features, num_gaussian_kernels, final_dim):
        super(CustomModel, self).__init__()
        self.num_gaussian_kernels = num_gaussian_kernels
        self.final_dim = final_dim
        self.gaussian_kernels = nn.ModuleList(
            [GaussianKernel(num_regions, num_features) for _ in range(num_gaussian_kernels)])

        self.sigma_parameters = nn.Parameter(torch.randn(num_gaussian_kernels))
        # self.weights = nn.Parameter(torch.randn(num_gaussian_kernels))  # 使用随机初始化的权重参数
        self.softmax = nn.Softmax(dim=0)
        self.bn = nn.BatchNorm1d(num_gaussian_kernels)
        self.relu = nn.ReLU()
        # 使用映射函数计算权重
        self.weights = 1 / (self.sigma_parameters**2)

    def forward(self, x1, x2):
        representations = []

        # 对每个高斯核计算相似度矩阵
        for kernel, weight in zip(self.gaussian_kernels, self.weights):
            similarity_matrix = kernel(x1, x2)
            representations.append(similarity_matrix.unsqueeze(0))  # 使用权重参数加权

        final_representation = torch.cat(representations, dim=0)  # 沿着特征维度连接

        # 使用 softmax 归一化权重
        normalized_weights = self.softmax(self.weights)

        # 使用归一化后的权重对相似度矩阵进行加权
        final_representation_weighted = final_representation * normalized_weights.view(-1, 1, 1)

        # 沿着特征维度求和以获得最终输出
        final_output = torch.sum(final_representation_weighted, dim=0)

        return final_output

# 定义双线性交互模块
class BilinearInteraction(nn.Module):
    def __init__(self, input_dim, weight_dim):
        super(BilinearInteraction, self).__init__()
        self.weight = nn.Parameter(torch.randn(weight_dim, input_dim, input_dim))

    def forward(self, x1, x2, similarity_matrix=None):
        batch_size = x1.size(0)
        # 计算 x1 的转置与权重矩阵的内积
        interaction = torch.matmul(x1.transpose(1, 2).squeeze(0), similarity_matrix) *  x2.transpose(1,2).squeeze(0)
        return interaction.transpose(0,1).unsqueeze(0)


# 定义带权重的双线性模型
class WeightedBilinearModel(nn.Module):
    def __init__(self, input_dim, weight_dim):
        super(WeightedBilinearModel, self).__init__()
        self.interaction = BilinearInteraction(input_dim, weight_dim)

    def forward(self, tuples, similarity_matrix=None):
        # 对每个 tuple 进行双线性交互
        interaction = self.interaction(tuples[0], tuples[1], similarity_matrix)
        return interaction


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob=0.3):
        super(MLP, self).__init__()

        # 第一个全连接层（1x49x2048 -> 1x49x1024）
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # 第二个全连接层（1x49x1024 -> 1x49x512）
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # 输出层（1x49x512 -> 1x49xoutput_dim）
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 主模型
class MainModel(nn.Module):
    def __init__(self, num_regions, num_features, num_gaussian_kernels, input_dim, weight_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MainModel, self).__init__()
        self.gaussian_kernel = GaussianKernel(num_regions, num_features)
        self.bilinear_interaction = BilinearInteraction(input_dim, weight_dim)
        self.mlp = MLP(input_dim, hidden_dim1, hidden_dim2, output_dim)

    def forward(self, x1, x2):
        similarity_matrix = self.gaussian_kernel(x1, x2)
        interaction = self.bilinear_interaction(x1, x2, similarity_matrix)
        output = self.mlp(interaction)
        return output






# 示例用法
if __name__ == "__main__":
    num_features = 2048
    num_gaussian_kernels = 5
    final_dim = 2048
    num_regions = 49
    # 创建MLP模型
    input_dim = 2048
    hidden_dim1 = 1024
    hidden_dim2 = 512
    output_dim = 512
    dropout_prob = 0.3
    mlp_model = MLP(input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob)
    model = CustomModel(num_regions,num_features, num_gaussian_kernels,final_dim)
    path = 'E:/all-images-listdir-shiyan'
    save_path = 'E:/kernel-biliner-final'
    all_final_representation = []
    # 创建双线性模型
    model_bilinear = WeightedBilinearModel(final_dim, num_regions)
    for each_folder in os.listdir(path):
        all_feature_pair = generate_image_feature_pairs(os.path.join(path, each_folder))
        representation_bilinear_each_folder = []

        for each_feature_pair in all_feature_pair:
            each_feature_tensor_1 = torch.tensor(np.load(each_feature_pair[0])['arr_0'])
            each_feature_tensor_2 = torch.tensor(np.load(each_feature_pair[1])['arr_0'])
            feature_pair_tuple_tensor = (each_feature_tensor_1, each_feature_tensor_2)
            output = model(feature_pair_tuple_tensor[0], feature_pair_tuple_tensor[1])
            #########################双线性交互########################
            # 计算双线性交互后的结果
            representation_bilinear = model_bilinear(feature_pair_tuple_tensor, output)
            each_bilinear_after_mlp = mlp_model(representation_bilinear)
            representation_bilinear_each_folder.append(each_bilinear_after_mlp)
        each_folder_final_output = torch.cat(representation_bilinear_each_folder, dim=0)
        each_folder_final_output =  torch.sum((torch.stack(each_folder_final_output) / len(all_feature_pair)),dim=0).cpu().detach().numpy()
        all_feature_pair.append(each_folder_final_output)
    final_output = torch.cat(all_final_representation, dim=0)