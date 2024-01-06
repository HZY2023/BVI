import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from itertools import combinations
from torch.utils.data import DataLoader, Dataset
import gc



def generate_image_feature_pairs(root_folder):
    # 获取所有文件夹的列表
    if len(os.listdir(os.path.join(path, root_folder))) >= 5:
        folders = [folder for folder in os.listdir(root_folder)][0:5]
    else:
        folders = [folder for folder in os.listdir(root_folder)]

    # 存储组合后的图像特征对
    image_feature_pairs = []

    if len(folders) == 1:
        single_feature = np.load(os.path.join(root_folder,folders[0]))
        noise = np.random.normal(0, 0.1, single_feature['arr_0'].shape)
        noisy_data = single_feature['arr_0']+noise
        image_feature_pairs.append(noisy_data)
        image_feature_pairs.append(folders[0])
        pairs = list(combinations(image_feature_pairs, 2))

    else:
        # 将图像特征两两组合成一对
        pairs = list(combinations(folders, 2))
        for pair in pairs:
            image_feature_pairs.append((os.path.join(root_folder, pair[0]), os.path.join(root_folder, pair[1])))


    if len(folders) == 1:
        image_feature_pairs[1]= os.path.join(root_folder,image_feature_pairs[1])
        return image_feature_pairs
    else:
        return image_feature_pairs

class GaussianKernel(nn.Module):
    def __init__(self, num_regions, num_features):
        super(GaussianKernel, self).__init__()
        self.sigma = nn.Parameter(torch.randn(1) * 100).cuda()  # 方差作为可训练参数
        self.mean = nn.Parameter(torch.randn(1)).cuda()  # 均值作为可训练参数
        self.num_regions = num_regions
        self.num_features = num_features

    def forward(self, x1, x2):
        # 张量形状保持为 (N, num_regions, num_features)
        x1, x2 = x1.cuda(), x2.cuda()
        N = x1.size(0)

        # 扩展维度以便可以进行张量广播
        x1_reshaped = x1.view(N, self.num_regions, 1, self.num_features)
        x2_reshaped = x2.view(N, 1, self.num_regions, self.num_features)

        # 计算欧几里得距离的平方
        distance_squared = torch.sum((x1_reshaped - x2_reshaped - self.mean) ** 2, dim=-1)

        # 计算高斯核函数值
        kernel_value = torch.exp(-distance_squared / (2 * self.sigma ** 2))

        # 在第一个维度上执行 softmax 操作
        softmaxed_similarity_matrix = torch.softmax(kernel_value, dim=-1)

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


        # 确保 normalized_weights 的维度是 [num_gaussian_kernels]
        assert normalized_weights.shape[0] == self.num_gaussian_kernels

        # 扩展 normalized_weights 的维度以匹配 final_representation
        # 从 [num_gaussian_kernels] 扩展为 [1, num_gaussian_kernels, 1, 1]
        normalized_weights_expanded = normalized_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # 使用归一化后的权重对相似度矩阵进行加权
        final_representation_weighted = (final_representation.transpose(0,1).to(device) * normalized_weights_expanded.to(device))

        # 沿着特征维度求和以获得最终输出
        final_output = torch.sum(final_representation_weighted, dim=1)

        return final_output

# 定义双线性交互模块
class BilinearInteraction(nn.Module):
    def __init__(self, input_dim, weight_dim):
        super(BilinearInteraction, self).__init__()
        self.weight = nn.Parameter(torch.randn(weight_dim, input_dim, input_dim))

    def forward(self, x1, x2, similarity_matrix=None):
        # 计算 x1 的转置与权重矩阵的内积
        interaction = torch.matmul(x1.float().transpose(1, 2).to(torch.float16).cuda(), similarity_matrix.to(torch.float16).cuda()) *  x2.transpose(1,2).cuda()
        return interaction.transpose(0,1)


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
        x = x.cuda().to(torch.float32)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x.half()

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, folder_pairs):
        self.folder_pairs = folder_pairs

    def __len__(self):
        return len(self.folder_pairs)

    def __getitem__(self, index):
        if len(self.folder_pairs) != 2:
            feature_1 = torch.tensor(np.load(self.folder_pairs[index][0], allow_pickle=True)['arr_0'])
            feature_2 = torch.tensor(np.load(self.folder_pairs[index][1], allow_pickle=True)['arr_0'])
        else:
            feature_1 = torch.tensor(self.folder_pairs[0])
            feature_2 = torch.tensor(np.load(self.folder_pairs[1], allow_pickle=True)['arr_0'])
        return feature_1, feature_2



# 示例用法
if __name__ == "__main__":
    num_features = 512
    num_gaussian_kernels = 5
    final_dim = 512
    num_regions = 49
    # 创建MLP模型
    input_dim = 2048
    hidden_dim1 = 1024
    hidden_dim2 = 512
    output_dim = 512
    dropout_prob = 0.3

    # 初始化AMP的scaler
    scaler = torch.cuda.amp.GradScaler()


    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    mlp_model = MLP(input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob)
    model = CustomModel(num_regions,num_features, num_gaussian_kernels,final_dim)
    path = '/home/gb/hzy/second-article/all-images-listdir'
    # batch_size = 64  # 选择适当的批处理大小
    all_final_representation = []
    # 创建双线性模型
    model_bilinear = WeightedBilinearModel(final_dim, num_regions)

    model.to(device)
    model_bilinear.to(device)
    mlp_model.to(device)


    for each_folder in tqdm(os.listdir(path)[107140:], desc="Processing Folders"):
        all_feature_pair = generate_image_feature_pairs(os.path.join(path, each_folder))
        dataset = MyDataset(all_feature_pair)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True)
        representation_bilinear_each_folder = []

        for batch in dataloader:
            feature_pair_tuple_tensor_1, feature_pair_tuple_tensor_2 = batch
            feature_pair_tuple_tensor_1 = mlp_model(feature_pair_tuple_tensor_1)
            feature_pair_tuple_tensor_2 = mlp_model(feature_pair_tuple_tensor_2)
            with torch.cuda.amp.autocast():
                output = model(feature_pair_tuple_tensor_1.to(device), feature_pair_tuple_tensor_2.to(device))

                # 计算双线性交互后的结果
                representation_bilinear = model_bilinear((feature_pair_tuple_tensor_1.squeeze(1), feature_pair_tuple_tensor_2.squeeze(1)), output)
                # each_bilinear_after_mlp = mlp_model(representation_bilinear)
                # representation_bilinear_each_folder.append(each_bilinear_after_mlp)
                # del each_bilinear_after_mlp,representation_bilinear,output
                representation_bilinear_each_folder.append(representation_bilinear)
                del output
                torch.cuda.empty_cache()
            # each_folder_final_output = torch.cat(representation_bilinear_each_folder, dim=0)
            each_folder_final_output = torch.sum(torch.stack(representation_bilinear_each_folder).squeeze(0) / len(all_feature_pair)*2, dim=0,keepdim=True).cpu().detach().numpy()
            all_final_representation.append(each_folder_final_output)
            del representation_bilinear_each_folder
            gc.collect()
            torch.cuda.empty_cache()
    final_output = np.concatenate(all_final_representation, axis=0)
    123












        # for each_feature_pair in all_feature_pair:
        #     each_feature_tensor_1 = torch.tensor(np.load(each_feature_pair[0])['arr_0'])
        #     each_feature_tensor_2 = torch.tensor(np.load(each_feature_pair[1])['arr_0'])
        #     feature_pair_tuple_tensor = (each_feature_tensor_1, each_feature_tensor_2)
        #     output = model(feature_pair_tuple_tensor[0], feature_pair_tuple_tensor[1])
            ########################双线性交互########################
            # 计算双线性交互后的结果
            # representation_bilinear = model_bilinear(feature_pair_tuple_tensor, output)
            # each_bilinear_after_mlp = mlp_model(representation_bilinear)
            # representation_bilinear_each_folder.append(each_bilinear_after_mlp)
        # each_folder_final_output = torch.cat(representation_bilinear_each_folder, dim=0)
        # each_folder_final_output =  torch.sum((torch.stack(representation_bilinear_each_folder) / len(all_feature_pair)),dim=0).cpu().detach().numpy()
        # all_feature_pair.append(each_folder_final_output)
    # final_output = torch.cat(all_final_representation, dim=0)