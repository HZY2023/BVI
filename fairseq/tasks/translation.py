# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import json
import itertools
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from itertools import combinations
from torch.utils.data import DataLoader, Dataset
import numpy as np
import gc
from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)

from fairseq.tasks import FairseqTask, register_task

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False,
    src_img_features = None
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                'src_dict.eos()',
            )
        src_datasets.append(src_dataset)
        tgt_datasets.append(
            data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        )

        if split == 'train':
            ####################################second-article##############################################
            # src_img_features = sorted(os.listdir('/home/gb/hzy/second-article/all-images-listdir'),key=lambda x: int(os.path.splitext(x)[0]))

            #################EMMT######################
            src_img_features = np.load('EMMT/train.npy')
            ###########################################
        elif split == 'valid':
            ##################################second-article###############################
            # src_img_features = sorted(os.listdir('/home/gb/hzy/second-article/all-images-listdir'),key=lambda x: int(os.path.splitext(x)[0]))
            ###############################################################

            ###################EMMT########################################
            src_img_features = np.load('EMMT/valid.npy')


        else:
            ##################Fashion-MMT####################
            # src_img_features = None
            #################################################

            ############EMMT#######################
            src_img_features = None


        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)
    if src_img_features is not None:
        return LanguagePairDataset(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset.sizes, tgt_dict,
            src_img_features=src_img_features,
            # bpe_txt_relations=bpe_txt_relations_list,
            # img_txt_relations=img_txt_relations_list,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            max_source_positions=max_source_positions,
            max_target_positions=max_target_positions,
            align_dataset=align_dataset,
        )
    else:
        return LanguagePairDataset(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset.sizes, tgt_dict,
            # bpe_txt_relations=bpe_txt_relations_list,
            # img_txt_relations=img_txt_relations_list,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            max_source_positions=max_source_positions,
            max_target_positions=max_target_positions,
            align_dataset=align_dataset,
        )


@register_task('translation')
class TranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenizer before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='if setting, we compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = args.data.split(os.pathsep)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(os.pathsep)
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_model(self, args):
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator(Namespace(**gen_args))
        return super().build_model(args)

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                escape_unk=escape_unk,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            try:
                hyps.append(decode(gen_out[i][0]['tokens']))
                refs.append(decode(
                    utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                ))
            except :
                print(gen_out[i])
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        tokenize = sacrebleu.DEFAULT_TOKENIZER if not self.args.eval_tokenized_bleu else 'none'
        return sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize)



###################################################kernel-bilinear-module#####################################################

def generate_pairs(root_folder):
        path = '/home/gb/hzy/second-article/all-images-listdir'
        if len(os.listdir(os.path.join(path, root_folder))) >= 5:
            folders = [folder for folder in os.listdir(root_folder)][0:5]
        else:
            folders = [folder for folder in os.listdir(root_folder)]

        image_feature_pairs = []

        if len(folders) == 1:
            single_feature = np.load(os.path.join(root_folder, folders[0]))
            noise = np.random.normal(0, 0.1, single_feature['arr_0'].shape)
            noisy_data = single_feature['arr_0'] + noise
            image_feature_pairs.append(noisy_data)
            image_feature_pairs.append(folders[0])
            pairs = list(combinations(image_feature_pairs, 2))
        else:
            pairs = list(combinations(folders, 2))
            for pair in pairs:
                image_feature_pairs.append((os.path.join(root_folder, pair[0]), os.path.join(root_folder, pair[1])))

        if len(folders) == 1:
            image_feature_pairs[1] = os.path.join(root_folder, image_feature_pairs[1])
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
        final_representation_weighted = (final_representation.transpose(0,1).cuda() * normalized_weights_expanded.cuda())

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
        interaction = torch.matmul(x1.float().transpose(1, 2).to(torch.float16).cuda(), similarity_matrix.to(torch.float16).cuda()) *  x2.float().transpose(1,2).cuda()
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
        x = x.view(x.size(0), 49, 1024)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, folder_pairs):
        self.folder_pairs = folder_pairs

    def __len__(self):
        return len(self.folder_pairs)

    def __getitem__(self, index):
        if len(self.folder_pairs) != 2:
            feature_1 = torch.tensor(np.load(self.folder_pairs[index][0])['arr_0'])
            feature_2 = torch.tensor(np.load(self.folder_pairs[index][1])['arr_0'])
        else:
            feature_1 = torch.tensor(self.folder_pairs[0])
            feature_2 = torch.tensor(np.load(self.folder_pairs[1])['arr_0'])
        return feature_1, feature_2