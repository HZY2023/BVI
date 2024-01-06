# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task, visual_halllucination =None, src_img_features= None):
        super().__init__(args, task)
        self.eps = args.label_smoothing
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model=None, sample=None, visual_hallucination=None, src_img_features=None,reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if sample['net_input'].get('src_img_features') is not None:
            # 收集所有的pair[0]和pair[1]
            all_pair_0_tensor = [torch.tensor(p[0].reshape(1, 49, 2048)).half().cuda() for p in sample['net_input']['src_img_features']]
            all_pair_1_tensor = [torch.tensor(p[1].reshape(1, 49, 2048)).half().cuda() for p in sample['net_input']['src_img_features']]

            all_pair_0_tensor = torch.cat(all_pair_0_tensor, dim=0)
            all_pair_1_tensor = torch.cat(all_pair_1_tensor, dim=0)


            # 一次性将它们转换为张量并移到GPU
            # all_pair_0_tensor = torch.tensor(np.array(all_pair_0)).half().to('cuda:1')
            # all_pair_1_tensor = torch.tensor(np.array(all_pair_1)).half().to('cuda:1')
            # all_pair_0_tensor = torch.tensor(all_pair_0).half().cuda()
            # all_pair_1_tensor = torch.tensor(all_pair_1).half().cuda()

            # all_pair_0_tensor = torch.tensor(all_pair_0)
            # all_pair_1_tensor = torch.tensor(all_pair_1)

            # 如果你需要更新原始sample字典，可以这样做：
            # 这里假设src_img_features的每个元素都是一个pair，现在我们用新的值更新它
            # sample['net_input']['src_img_features'] = list(zip(all_pair_0_tensor.unbind(), all_pair_1_tensor.unbind()))
            all_pair_0_list = list(all_pair_0_tensor.unbind())
            all_pair_1_list = list(all_pair_1_tensor.unbind())

            # 在每个张量中添加一个额外的维度
            all_pair_0_expanded = [x.unsqueeze(0) for x in all_pair_0_list]
            all_pair_1_expanded = [x.unsqueeze(0) for x in all_pair_1_list]

            # 使用 zip 将它们重新组合成列表的列表
            src_img_features = list(zip(all_pair_0_expanded, all_pair_1_expanded))

            # 将列表的列表转换为张量
            sample['net_input']['src_img_features'] = src_img_features




            net_output = model(**sample['net_input'])
        else:
            net_output = model(**sample['net_input'])


        # net_output = model(**sample['net_input'])

        loss, nll_loss = self.compute_loss(model, net_output[0], sample, reduce=reduce)

        #######将KL散度的值添加到损失中############
        # with torch.no_grad():
        if net_output[1].encoder_out is not None:
            if net_output[1].src_img_features is not None:
                kl_divergence = self.kl_divergence_loss(net_output[1].hallucination_train.float().cpu(), net_output[1].src_img_features.float().cpu(), net_output[1].src_img_features.size(-1), net_output[1].hallucination_train.size(-1))
                # kl_divergence = self.kl_divergence_loss(net_output[1].hallucination_train.float(), net_output[1].src_img_features.float())
                loss = loss.to(device) + 15*kl_divergence

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    # def compute_kl_divergence(self, text_features, img_features):
        # Step 1: 计算文本特征和图像特征的均值和方差
        # mean_text = torch.mean(text_features, dim=(0, 1))
        # variance_text = torch.var(text_features, dim=(0, 1))
        # mean_img = torch.mean(img_features, dim=(0, 1))
        # variance_img = torch.var(img_features, dim=(0, 1))
        #
        # # Step 2: 构建正态分布
        # std_text = torch.sqrt(variance_text)
        # std_img = torch.sqrt(variance_img)
        #
        # text_distribution = dist.Normal(mean_text, std_text)
        # img_distribution = dist.Normal(mean_img, std_img)

        # Step 3: 计算KL散度
        # kl_divergence = torch.distributions.kl.kl_divergence(text_features, img_features)
        # total_kl_divergence = torch.sum(kl_divergence)
        #
        # return total_kl_divergence

    # def kl_divergence_loss(self, visual_hallucination, src_img_features):
    #     """
    #     计算KL散度损失，使 visual_hallucination 逼近 src_img_features。
    #
    #     Args:
    #     visual_hallucination (torch.Tensor): 待训练的 visual_hallucination 张量，形状为 (batch_size, 49, 512)
    #     src_img_features (torch.Tensor): 目标特征张量，形状同样为 (batch_size, 49, 512)
    #
    #     Returns:
    #     torch.Tensor: KL散度损失值
    #     """
    #
    #     # 使用softmax将输入张量转换为概率分布
    #     visual_hallucination_prob = torch.nn.functional.softmax(visual_hallucination, dim=-1)
    #     src_img_features_prob = torch.nn.functional.softmax(src_img_features, dim=-1)
    #
    #     # 直接对softmax后的结果取对数
    #     visual_hallucination_log_prob = torch.log(visual_hallucination_prob + 1e-10)
    #     src_img_features_log_prob = torch.log(src_img_features_prob + 1e-10)
    #
    #     # 计算KL散度损失
    #     kl_loss = torch.sum(visual_hallucination_prob.to('cuda:1') * (visual_hallucination_log_prob.to('cuda:1') - src_img_features_log_prob),
    #                         dim=(1, 2))
    #     average_kl_loss = kl_loss.sum()
    #
    #     return average_kl_loss

    # def compute_loss(self, model, net_output, sample, reduce=True):
    #     net_output_x = net_output[0:2]
    #     net_output_x_txt = net_output[2:4]
    #     net_output_x_img = net_output[4:6]
    #     net_output_x_img_no = net_output[6:8]
    #
    #     lprobs_txt = model.get_normalized_probs(net_output_x, log_probs=True)
    #     lprobs_x_txt = model.get_normalized_probs(net_output_x_txt, log_probs=True)
    #     lprobs_x_img = model.get_normalized_probs(net_output_x_img, log_probs=False)
    #     lprobs_x_img_no = model.get_normalized_probs(net_output_x_img_no, log_probs=False)
    #
    #     lprobs = lprobs_txt.view(-1, lprobs_txt.size(-1))
    #     # lprobs_img = lprobs_img.view(-1, lprobs.size(-1))
    #
    #     target = model.get_targets(sample, net_output)
    #     pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
    #
    #     target = target.view(-1, 1)
    #
    #     loss, nll_loss = label_smoothed_nll_loss(
    #         lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
    #     )
    #
    #     # txt_out = net_output[0]
    #     # img_out = net_output[2]
    #
    #     lprobs_x_txt = torch.exp(lprobs_x_txt)
    #     lprobs_x_img = torch.exp(lprobs_x_img)
    #     lprobs_x_img_no = torch.exp(lprobs_x_img_no)
    #     consis_loss_help = utils.multimodel_consis_loss(lprobs_x_txt, lprobs_x_img)
    #     consis_loss_help_no = utils.multimodel_consis_loss(lprobs_x_txt, lprobs_x_img_no)
    #     # if consis_loss_help > 0:
    #     #     if consis_loss_help_no > 0:
    #     #         consis_loss_all = consis_loss_help + consis_loss_help_no
    #     #     else:
    #     #         consis_loss_all = consis_loss_help - consis_loss_help_no
    #     # if consis_loss_help <= 0:
    #     #     if consis_loss_help_no <= 0:
    #     #         consis_loss_all = -consis_loss_help - consis_loss_help_no
    #     #     else:
    #     #         consis_loss_all = -consis_loss_help + consis_loss_help_no
    #     # consis_loss_all = -math.fabs(consis_loss_help) + math.fabs(consis_loss_help_no)
    #     # consis_loss_all = torch.einsum('nc,nc->n',[lprobs_x_txt,lprobs_x_img])
    #     # consis_loss_all = torch.log(-consis_loss_help_no)
    #     consis_loss_all = -torch.log(consis_loss_help / consis_loss_help_no)

    # def compute_kl_divergence(self, p, q):
    #     """
    #     计算两个二维张量之间的KL散度。
    #
    #     Args:
    #     p (torch.Tensor): 输入张量 p，形状为 (batch_size, features)
    #     q (torch.Tensor): 输入张量 q，形状为 (batch_size, features)
    #
    #     Returns:
    #     torch.Tensor: KL散度的计算结果
    #     """
    #     # 确保 p 和 q 都是正数
    #     p = torch.clamp(p, min=1e-10)
    #     q = torch.clamp(q, min=1e-10)
    #
    #     # 规范化为概率分布
    #     p = p / p.sum(dim=1, keepdim=True)
    #     q = q / q.sum(dim=1, keepdim=True)
    #
    #     # 使用 torch.log 函数计算 p 的对数
    #     p_log = torch.log(p)
    #
    #     # 使用 torch.nn.functional.kl_div 计算 KL 散度
    #     kl_divergence = torch.nn.functional.kl_div(p_log, q, reduction='batchmean')
    #
    #     return kl_divergence
    # def kl_divergence_loss(self, visual_hallucination, src_img_features, epsilon=1e-10):
    #     """
    #     计算KL散度损失，使 visual_hallucination 逼近 src_img_features。
    #     Args:
    #     visual_hallucination (torch.Tensor): 待训练的 visual_hallucination 张量，形状为 (batch_size, 49, dim)
    #     src_img_features (torch.Tensor): 目标特征张量，形状同样为 (batch_size, 49, dim)
    #     epsilon (float): 为避免对数为无穷大的正值
    #     Returns:
    #     torch.Tensor: KL散度损失值
    #     """
    #
    #     # 使用softmax将输入张量转换为概率分布
    #     visual_hallucination_prob = F.softmax(visual_hallucination, dim=-1)
    #     src_img_features_prob = F.softmax(src_img_features, dim=-1)
    #
    #     # 为避免对数为负无穷，给概率分布中的每个值添加一个小的epsilon
    #     visual_hallucination_prob = visual_hallucination_prob + epsilon
    #     src_img_features_prob = src_img_features_prob + epsilon
    #
    #     # 使用对数softmax输出来计算log-probabilities，以便直接与概率分布相乘
    #     log_visual_hallucination_prob = torch.log(visual_hallucination_prob)
    #
    #     # 计算KL散度
    #     kl_div = F.kl_div(log_visual_hallucination_prob.to('cuda:0'), src_img_features_prob, reduction='batchmean')
    #
    #     return kl_div

#    def kl_divergence_loss(self, visual_hallucination, src_img_features, dim, dim_2, epsilon=1e-10):
#        """
#        计算KL散度损失，使 visual_hallucination 逼近 src_img_features。
#
#        Args:
#        visual_hallucination (torch.Tensor): 待训练的 visual_hallucination 张量，形状为 (batch_size, 49, dim)
#        src_img_features (torch.Tensor): 目标特征张量，形状同样为 (batch_size, 49, dim_2)
#        dim (int): visual_hallucination的最后一个维度大小
#        dim_2 (int): src_img_features的最后一个维度大小
#        epsilon (float): 为避免对数为无穷大的正值
#
#        Returns:
#        torch.Tensor: KL散度损失值
#        """
#
#        # 全连接层，将dim维度转换为dim_2维度
#        fc = nn.Linear(dim, dim_2)
#
#        # 使用全连接层改变visual_hallucination的最后一维
#        src_img_features = fc(src_img_features)
#
#        # 使用softmax将输入张量转换为概率分布
#        visual_hallucination_prob = F.softmax(visual_hallucination, dim=-1)
#        src_img_features_prob = F.softmax(src_img_features, dim=-1)
#
#        # 直接对softmax后的结果取对数
#        visual_hallucination_log_prob = torch.log(visual_hallucination_prob + epsilon)
#        src_img_features_log_prob = torch.log(src_img_features_prob + epsilon)
#
#        # 计算KL散度损失
#        kl_div = F.kl_div(visual_hallucination_log_prob, src_img_features_log_prob.cuda(), reduction='batchmean')
#
#        return kl_div


    def kl_divergence_loss(self, visual_hallucination, src_img_features, dim, dim_2, epsilon=1e-10):
        """
        计算KL散度损失，使 visual_hallucination 逼近 src_img_features。

        Args:
        visual_hallucination (torch.Tensor): 待训练的 visual_hallucination 张量，形状为 (batch_size, 49, dim)
        src_img_features (torch.Tensor): 目标特征张量，形状同样为 (batch_size, 49, dim_2)
        dim (int): visual_hallucination的最后一个维度大小
        dim_2 (int): src_img_features的最后一个维度大小
        epsilon (float): 为避免对数为无穷大的正值

        Returns:
        torch.Tensor: KL散度损失值
        """

        # 全连接层，将dim维度转换为dim_2维度
        fc = nn.Linear(dim, dim_2)
        # 使用全连接层改变visual_hallucination的最后一维
        src_img_features = fc(src_img_features)

        # 使用softmax将输入张量转换为概率分布，并加上epsilon避免log(0)
        visual_hallucination_prob = F.softmax(visual_hallucination, dim=-1) + epsilon
        src_img_features_prob = F.softmax(src_img_features, dim=-1) + epsilon

        # 计算对数概率
        log_visual_hallucination_prob = torch.log(visual_hallucination_prob)

        # 计算KL散度损失
        kl_div = F.kl_div(log_visual_hallucination_prob.to(device), src_img_features_prob.to(device), reduction='batchmean')

        return kl_div




    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: round(2**meters['nll_loss'].avg, 3))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


def tensor_statistics(tensor, name="Tensor"):
    print(f"--- {name} Statistics ---")
    print("Mean:", tensor.mean().item())
    print("Std Dev:", tensor.std().item())
    print("Max Value:", tensor.max().item())
    print("Min Value:", tensor.min().item())
    print("-------------------------\n")