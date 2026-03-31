# -*- coding: utf-8 -*-
"""
小样本学习（Few-Shot Learning）训练框架
======================================
基于 DINO 预训练视觉模型 + 原型网络（Prototypical Network）的小样本分类系统。

核心思路：
    1. 用 DINO 提取图像特征（骨干网络冻结，不参与梯度更新）
    2. 通过投影层将特征映射到统一的嵌入空间
    3. 在嵌入空间中为每个类别计算原型（类中心向量）
    4. 用查询样本与各类原型的相似度完成分类

支持三个可独立开关的消融模块：
    - USE_MULTI_LAYER:    多层特征融合（融合 DINO 最后 N 层的 [CLS] 特征）
    - USE_ATTENTION_PROTO: 注意力加权原型（样本级自注意力替代简单均值）
    - USE_MULTI_METRIC:   多度量学习（多个马氏距离度量的加权融合替代余弦相似度）
"""

import os
import random
import logging
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoImageProcessor, AutoConfig
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# ==================== 全局设备设置 ====================
# 自动检测 GPU，有则用 CUDA，否则回退到 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# 开启 cuDNN 自动调优：对固定输入尺寸的模型可加速卷积运算
torch.backends.cudnn.benchmark = True

# ==================== 加载 DINO 图片预处理器 ====================
# AutoImageProcessor 会读取 DINO 目录下的 preprocessor_config.json，
# 自动获得与预训练模型匹配的 resize、normalize 等预处理流程。
# 后续 Dataset 中会用它把 PIL Image 转为模型所需的 pixel_values 张量。
processor = AutoImageProcessor.from_pretrained("DINO")
print("加载器加载成功")


class Config:
    """
    全局超参数配置类
    ================
    集中管理所有实验超参数，便于消融实验中快速切换配置。
    按功能分区组织，路径类字段由 load_config() 根据实验名自动填充。
    """

    # =========================
    # 数据参数
    # =========================
    SPLIT_DIR = './split_dir'   # 数据集根目录，下设 train/ 和 val/ 子目录，每个子目录按类别再分子文件夹
    K_SHOT = 5                  # 支持集中每个类别的样本数（即 "5-shot"）
    Q_QUERY = 2                 # 查询集中每个类别的样本数

    # =========================
    # 消融实验开关（训练期）
    # =========================
    USE_MULTI_LAYER = False      # 是否启用多层特征融合（融合 DINO 最后 FUSION_LAYERS 层）
    USE_ATTENTION_PROTO = False  # 是否启用注意力加权原型（用样本间相似度加权代替简单均值）
    USE_MULTI_METRIC = False     # 是否启用多度量学习（多个可学习马氏距离度量的加权融合）

    # 实验名称标识符，用于区分不同消融实验的输出文件（建议与 A1/A2/A3/A4/A5 对齐）
    EXPERIMENT_NAME = "A1"

    # =========================
    # 模型参数
    # =========================
    PRETRAINED_MODEL_PATH = 'DINO'  # DINO 预训练权重的本地目录路径
    N_WAY = 10                      # N-way 分类：每个 episode 中随机抽取的类别数
    PROJECTION_DIM = 1280           # 投影层输出维度，即嵌入空间的维度
    DROPOUT_RATE = 0.3              # 投影层中 Dropout 的丢弃概率，防止过拟合

    # =========================
    # 训练参数
    # =========================
    NUM_EPOCHS = 100     # 最大训练轮数
    MIN_EPOCHS = 40      # 最少训练轮数，早停机制在此之前不会触发
    BATCH_SIZE = 8       # 每个 batch 包含的 episode 数量
    LEARNING_RATE = 1e-5 # Adam 优化器初始学习率
    WEIGHT_DECAY = 1e-4  # Adam 优化器的 L2 权重衰减系数
    PATIENCE = 10        # 早停耐心值：验证准确率连续 PATIENCE 轮未提升 DELTA 则停止
    DELTA = 0.3          # 早停灵敏度：验证准确率需超过历史最佳至少 DELTA 个百分点才算提升
    REG_WEIGHT = 0.1     # 多度量正则损失的权重系数（仅 USE_MULTI_METRIC=True 时生效）

    # =========================
    # 多度量学习参数
    # =========================
    NUM_METRICS = 3            # 可学习马氏距离度量矩阵的数量
    METRIC_TEMP = 0.07         # 温度系数，用于缩放相似度分数（越小分布越尖锐）
    METRIC_FUSION_HIDDEN = 256 # 度量融合网络的隐藏层维度
    METRIC_DIFF_WEIGHT = 0.1   # 正交正则项的权重，鼓励不同度量矩阵学到不同的距离空间
    FUSION_DROPOUT = 0.2       # 度量融合网络中 Dropout 的丢弃概率

    # =========================
    # 多层融合参数
    # =========================
    FUSION_LAYERS = 9  # 融合 DINO 最后多少层的 [CLS] 特征（仅 USE_MULTI_LAYER=True 时有意义）

    # =========================
    # 优化参数
    # =========================
    SCHEDULER_T_MAX = 20  # CosineAnnealingLR 调度器的半周期长度（学习率在 T_MAX 轮后降至最低）

    # 以下路径在 load_config() 中根据 EXPERIMENT_NAME 自动生成
    SAVE_PATH = None  # 最佳模型权重的保存路径
    LOG_PATH = None   # 训练日志的保存路径
    PLOT_PATH = None  # 训练/验证曲线图的保存路径


def set_seed(seed=42):
    """
    设置全局随机种子，确保实验可复现。

    参数:
        seed (int): 随机种子值，默认 42

    说明:
        同时固定 Python 内置 random、PyTorch CPU 和所有 CUDA 设备的随机数生成器。
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info(f"随机种子已设置为 {seed}")


def get_device():
    """
    检测并返回当前可用的计算设备（GPU 或 CPU）。

    返回:
        torch.device: cuda 或 cpu
    """
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备：{_device}")
    return _device


def get_train_transforms():
    """
    构建训练集的数据增强管线。

    增强策略（按顺序执行）：
        1. RandomResizedCrop(224):  随机裁剪并缩放到 224×224，裁剪面积比例 [0.6, 1.0]
        2. RandomHorizontalFlip:    以 50% 概率水平翻转
        3. RandomRotation(15):      随机旋转 ±15°
        4. ColorJitter:             随机调整亮度/对比度/饱和度（±0.5）和色相（±0.2）
        5. RandomGrayscale(p=0.2):  以 20% 概率转为灰度图（仍保持 3 通道）

    注意:
        这里不包含 ToTensor 和 Normalize，因为后续由 DINO 的 processor 完成。

    返回:
        transforms.Compose: 组合后的变换管线
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
        transforms.RandomGrayscale(p=0.2)
    ])


def get_val_transforms():
    """
    构建验证集的预处理管线（无随机增强，保证评估一致性）。

    预处理步骤：
        1. Resize(256):      短边缩放到 256
        2. CenterCrop(224):  中心裁剪 224×224

    返回:
        transforms.Compose: 组合后的变换管线
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])


class FewShotDataset(Dataset):
    """
    小样本学习 Episode 采样数据集
    ==============================
    每次 __getitem__ 调用构造一个完整的 episode：
        - 从所有类别中随机抽取 n_way 个类
        - 每个类随机抽取 k_shot 张作为支持集，q_query 张作为查询集
        - 对图像施加数据增强（如有），再通过 DINO processor 转为模型输入张量

    参数:
        root_dir  (str):            数据根目录，下设各类别子文件夹
        transform (callable|None):  PIL Image 上的数据增强变换
        n_way     (int):            每个 episode 的类别数
        k_shot    (int):            每个类在支持集中的样本数
        q_query   (int):            每个类在查询集中的样本数
    """

    def __init__(self, root_dir, transform=None, n_way=10, k_shot=5, q_query=2):
        self.root_dir = root_dir    # 数据集根目录路径
        self.transform = transform  # 数据增强变换（训练集有，验证集无随机增强）
        self.n_way = n_way          # 每 episode 采样的类别数
        self.k_shot = k_shot        # 支持集每类样本数
        self.q_query = q_query      # 查询集每类样本数

        # 扫描根目录，获取所有类别名称（即子文件夹名），并排序保证顺序一致
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort()
        self.num_classes = len(self.classes)  # 总类别数

        # 建立 {类别名: [图片路径列表]} 的映射字典，只保留常见图片格式
        self.class_to_images = {
            cls: [
                os.path.join(root_dir, cls, img)
                for img in os.listdir(os.path.join(root_dir, cls))
                if img.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            for cls in self.classes
        }

        # 对图片数不足 (k_shot + q_query) 的类别，通过有放回重采样补齐
        for cls, imgs in self.class_to_images.items():
            required = self.k_shot + self.q_query  # 每 episode 单类最少需要的图片数
            if len(imgs) < required:
                shortfall = required - len(imgs)
                self.class_to_images[cls].extend(random.choices(imgs, k=shortfall))

        # 过滤掉仍然不满足最低样本要求的类别（理论上经过补齐后不会出现）
        self.classes = [
            cls for cls, imgs in self.class_to_images.items()
            if len(imgs) >= (self.k_shot + self.q_query)
        ]
        self.num_classes = len(self.classes)  # 更新有效类别数
        logging.info(f"有效类别数量：{self.num_classes}")

    def __len__(self):
        """
        返回每个 epoch 的 episode 总数。
        这是一个固定值 1000，因为 episode 是随机采样的，与实际图片总数无关。
        """
        return 1000

    def __getitem__(self, idx):
        """
        构造一个 episode 并返回支持集/查询集的图像张量和标签。

        参数:
            idx (int): 索引（实际未使用，因为每次都是随机采样）

        返回:
            support_images (Tensor): 支持集图像，形状 [n_way * k_shot, 3, 224, 224]
            support_labels (Tensor): 支持集标签，形状 [n_way * k_shot]，值域 [0, n_way-1]
            query_images   (Tensor): 查询集图像，形状 [n_way * q_query, 3, 224, 224]
            query_labels   (Tensor): 查询集标签，形状 [n_way * q_query]，值域 [0, n_way-1]
        """
        # 从所有有效类别中随机抽取 n_way 个类
        selected_classes = random.sample(self.classes, self.n_way)

        support_images = []   # 存放支持集图像张量
        support_labels = []   # 存放支持集标签（episode 内局部编号 0 ~ n_way-1）
        query_images = []     # 存放查询集图像张量
        query_labels = []     # 存放查询集标签

        for i, cls in enumerate(selected_classes):
            # 从当前类中随机抽取 k_shot + q_query 张图片
            selected_images = random.sample(self.class_to_images[cls], self.k_shot + self.q_query)
            support_imgs = selected_images[:self.k_shot]   # 前 k_shot 张作为支持集
            query_imgs = selected_images[self.k_shot:]     # 剩余作为查询集

            # 处理支持集图片
            for img_path in support_imgs:
                image = Image.open(img_path).convert('RGB')  # 统一转 RGB
                if self.transform:
                    image = self.transform(image)  # 施加数据增强
                # 用 DINO processor 做 resize + normalize，得到 pixel_values 张量
                inputs = processor(images=image, return_tensors='pt')
                support_images.append(inputs['pixel_values'].squeeze(0))  # 去掉 batch 维
                support_labels.append(i)  # episode 内局部类别编号

            # 处理查询集图片（流程同上）
            for img_path in query_imgs:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                inputs = processor(images=image, return_tensors="pt")
                query_images.append(inputs['pixel_values'].squeeze(0))
                query_labels.append(i)

        # 将列表堆叠为张量
        support_images = torch.stack(support_images)  # [S, 3, 224, 224]，S = n_way * k_shot
        support_labels = torch.tensor(support_labels)  # [S]
        query_images = torch.stack(query_images)      # [Q, 3, 224, 224]，Q = n_way * q_query
        query_labels = torch.tensor(query_labels)      # [Q]

        return support_images, support_labels, query_images, query_labels


def custom_collate_fn(batch):
    """
    自定义 DataLoader 的 collate 函数，将多个 episode 堆叠为一个 batch。

    默认 collate 无法正确处理形状不规则的 episode 数据，
    因此需要手动在第 0 维（batch 维）进行堆叠。

    参数:
        batch (list[tuple]): DataLoader 收集的一组 episode，
            每个元素为 (support_images, support_labels, query_images, query_labels)

    返回:
        support_images (Tensor): [B, S, 3, 224, 224]  B=batch_size, S=n_way*k_shot
        support_labels (Tensor): [B, S]
        query_images   (Tensor): [B, Q, 3, 224, 224]  Q=n_way*q_query
        query_labels   (Tensor): [B, Q]
    """
    support_images = torch.stack([item[0] for item in batch], dim=0)  # [B, S, 3, 224, 224]
    support_labels = torch.stack([item[1] for item in batch], dim=0)  # [B, S]
    query_images = torch.stack([item[2] for item in batch], dim=0)    # [B, Q, 3, 224, 224]
    query_labels = torch.stack([item[3] for item in batch], dim=0)    # [B, Q]
    return support_images, support_labels, query_images, query_labels


class FewShotModel(nn.Module):
    """
    小样本学习模型（基于原型网络 + DINO 骨干）
    ============================================
    整体流程：
        图像 → DINO 特征提取（冻结） → 多层融合（可选） → 投影层 → L2 归一化
        → 原型计算（均值 / 注意力加权） → 相似度打分（余弦 / 多度量马氏距离） → 分类

    参数:
        config          (Config):     超参数配置对象
        initialize_clip (bool):       是否加载预训练权重。True=加载并冻结，False=仅加载结构
        model_config    (PretrainedConfig|None): 若提供，则用此配置直接构建空模型（用于特殊加载场景）
    """

    def __init__(self, config, initialize_clip=True, model_config=None):
        super(FewShotModel, self).__init__()
        self.config = config  # 保存配置引用，供各方法读取开关和超参

        # ---- DINO 骨干网络初始化（三种模式） ----
        if model_config:
            # 模式1：用外部提供的 config 构建空模型（不加载权重）
            self.clip = AutoModel.from_config(model_config).to(device)
        elif initialize_clip:
            # 模式2（默认）：加载预训练权重并冻结所有参数（不参与梯度更新）
            self.clip = AutoModel.from_pretrained(config.PRETRAINED_MODEL_PATH).to(device)
            for param in self.clip.parameters():
                param.requires_grad = False
        else:
            # 模式3：仅加载模型结构，不加载权重（用于后续手动 load_state_dict）
            self.clip = AutoModel.from_config(
                AutoConfig.from_pretrained(config.PRETRAINED_MODEL_PATH)
            ).to(device)

        # DINO 单层隐藏维度（如 ViT-B/16 为 768，ViT-L/14 为 1024）
        self.vision_hidden_size = self.clip.config.hidden_size
        # 多层融合后的总维度 = 单层维度 × 融合层数
        total_dim = self.vision_hidden_size * config.FUSION_LAYERS

        # ---- 多层融合的可学习层权重 ----
        # 初始化为均匀分布（每层权重相等），训练时通过 softmax 归一化后加权求和
        # 即使 USE_MULTI_LAYER=False，也需要此参数以保持 state_dict 键一致
        self.layer_weights = nn.Parameter(
            torch.ones(config.FUSION_LAYERS) / config.FUSION_LAYERS
        )

        # ---- 投影层 ----
        # 将拼接后的多层特征（total_dim 维）映射到统一的嵌入空间（PROJECTION_DIM 维）
        # Linear → LayerNorm → ReLU → Dropout
        self.projection = nn.Sequential(
            nn.Linear(total_dim, config.PROJECTION_DIM),
            nn.LayerNorm(config.PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
        )

        # ---- 多度量学习：可学习的马氏距离矩阵 ----
        # 每个 metric 是一个 [D, D] 的方阵 M，实际距离计算为 d = diff^T (M^T M) diff
        # 初始化为 单位矩阵 + 小噪声，确保训练初期接近欧氏距离
        self.metrics = nn.ParameterList([
            nn.Parameter(
                torch.eye(config.PROJECTION_DIM) +
                torch.randn(config.PROJECTION_DIM, config.PROJECTION_DIM) * 0.01
            )
            for _ in range(config.NUM_METRICS)
        ])

        # ---- 度量融合网络 ----
        # 根据查询样本的特征动态生成各度量的权重（自适应融合）
        # 输入: 查询特征 [*, D] → 输出: 各度量权重 [*, NUM_METRICS]，经 Softmax 归一化
        self.metric_fusion = nn.Sequential(
            nn.Linear(config.PROJECTION_DIM, config.METRIC_FUSION_HIDDEN),
            nn.LayerNorm(config.METRIC_FUSION_HIDDEN),
            nn.ReLU(),
            nn.Dropout(config.FUSION_DROPOUT),
            nn.Linear(config.METRIC_FUSION_HIDDEN, config.METRIC_FUSION_HIDDEN // 2),
            nn.LayerNorm(config.METRIC_FUSION_HIDDEN // 2),
            nn.ReLU(),
            nn.Dropout(config.FUSION_DROPOUT),
            nn.Linear(config.METRIC_FUSION_HIDDEN // 2, config.NUM_METRICS),
            nn.Softmax(dim=1)
        )

        # ---- 兼容性 buffer ----
        # 此 buffer 不参与当前版本的前向计算，仅为兼容旧版本保存的 state_dict 键而保留，
        # 防止 load_state_dict 时因键缺失而报错
        self.register_buffer(
            'episode_prototypes',
            torch.zeros(config.N_WAY, config.PROJECTION_DIM)
        )

        # 温度系数：用于缩放相似度分数，使 softmax 分布更尖锐或更平滑
        self.temperature = config.METRIC_TEMP

    # =========================================================
    # 特征提取：支持开关 USE_MULTI_LAYER
    # =========================================================
    def get_features(self, images):
        """
        从 DINO 骨干中提取图像特征。

        当 USE_MULTI_LAYER=True 时：
            取最后 FUSION_LAYERS 层的 [CLS] token，用可学习权重加权后拼接，
            得到 (vision_hidden_size × FUSION_LAYERS) 维的融合特征。

        当 USE_MULTI_LAYER=False 时：
            仅取最后一层的 [CLS] token，右侧补零至与多层融合相同的总维度，
            保证投影层输入维度一致，从而共享同一套投影层权重。

        参数:
            images (Tensor): 输入图像，形状 [N, 3, 224, 224]

        返回:
            fused_features (Tensor): 融合后的特征，形状 [N, vision_hidden_size × FUSION_LAYERS]
        """
        # 前向传播 DINO，开启 output_hidden_states 以获取所有中间层输出
        vision_outputs = self.clip(
            pixel_values=images,
            output_hidden_states=True,
            return_dict=True
        )

        if self.config.USE_MULTI_LAYER:
            # 取最后 FUSION_LAYERS 层的隐藏状态
            last_n_layers = vision_outputs.hidden_states[-self.config.FUSION_LAYERS:]
            # 对层权重做 softmax 归一化，确保权重和为 1
            layer_weights = F.softmax(self.layer_weights, dim=0)

            features = []
            for i, layer_features in enumerate(last_n_layers):
                cls_features = layer_features[:, 0]              # 取 [CLS] token: [N, D]
                weighted_features = cls_features * layer_weights[i]  # 乘以该层的权重标量
                features.append(weighted_features)

            # 将各层加权特征在特征维度上拼接: [N, D * FUSION_LAYERS]
            fused_features = torch.cat(features, dim=-1)
        else:
            # 仅使用最后一层的 [CLS] token
            last_feat = vision_outputs.hidden_states[-1][:, 0]  # [N, D]
            # 右侧补零，使总维度与多层融合模式一致: [N, D * (FUSION_LAYERS - 1)]
            pad_size = self.config.FUSION_LAYERS - 1
            zeros = torch.zeros(
                last_feat.shape[0],
                last_feat.shape[1] * pad_size,
                device=last_feat.device,
                dtype=last_feat.dtype
            )
            fused_features = torch.cat([last_feat, zeros], dim=-1)  # [N, D * FUSION_LAYERS]

        return fused_features

    # =========================================================
    # 多度量相似度
    # =========================================================
    def compute_single_metric_similarity(self, query_features, prototypes, metric_idx):
        """
        使用第 metric_idx 个马氏距离度量矩阵计算查询与原型之间的负距离（相似度）。

        马氏距离公式: d(q, p) = (q - p)^T L (q - p)，其中 L = M^T M（半正定矩阵）
        返回 -d 作为相似度（距离越小相似度越高）。

        支持两种输入形状：
            - 2D: query_features [Q, D], prototypes [N, D] → 返回 [Q, N]
            - 3D: query_features [B, Q, D], prototypes [B, N, D] → 返回 [B, Q, N]

        参数:
            query_features (Tensor): 查询样本特征
            prototypes     (Tensor): 各类原型特征
            metric_idx     (int):    度量矩阵索引

        返回:
            Tensor: 负马氏距离（即相似度分数）
        """
        metric = self.metrics[metric_idx]          # [D, D] 可学习矩阵 M
        L = torch.matmul(metric, metric.T)         # L = M^T M，保证半正定

        if query_features.dim() == 2 and prototypes.dim() == 2:
            diff = query_features.unsqueeze(1) - prototypes.unsqueeze(0)   # [Q, N, D]
            dist = torch.sum(torch.matmul(diff, L) * diff, dim=-1)         # [Q, N]
            return -dist

        if query_features.dim() == 3 and prototypes.dim() == 3:
            diff = query_features.unsqueeze(2) - prototypes.unsqueeze(1)   # [B, Q, N, D]
            dist = torch.sum(torch.matmul(diff, L) * diff, dim=-1)         # [B, Q, N]
            return -dist

        raise ValueError(
            f"Unsupported shapes: query={query_features.shape}, prototypes={prototypes.shape}"
        )

    def compute_metric_weights(self, query_features):
        """
        通过度量融合网络，根据查询样本特征动态生成各度量矩阵的融合权重。

        参数:
            query_features (Tensor): [Q, D] 或 [B, Q, D]

        返回:
            Tensor: 各度量的权重，[Q, M] 或 [B, Q, M]，M = NUM_METRICS，已经过 Softmax
        """
        if query_features.dim() == 2:
            return self.metric_fusion(query_features)  # [Q, M]

        if query_features.dim() == 3:
            bsz, nq, dim = query_features.shape
            # 展平为 2D 送入融合网络，再恢复形状
            weights = self.metric_fusion(query_features.reshape(-1, dim))
            return weights.reshape(bsz, nq, -1)  # [B, Q, M]

        raise ValueError(f"Unsupported query_features shape: {query_features.shape}")

    def compute_multi_metric_similarity(self, query_features, prototypes):
        """
        多度量融合相似度计算：
            1. 分别用每个度量矩阵计算负马氏距离
            2. 用融合网络动态生成各度量的权重
            3. 加权求和得到最终相似度
            4. 除以温度系数缩放

        参数:
            query_features (Tensor): [Q, D] 或 [B, Q, D]
            prototypes     (Tensor): [N, D] 或 [B, N, D]

        返回:
            Tensor: 缩放后的融合相似度分数，[Q, N] 或 [B, Q, N]
        """
        # 收集每个度量矩阵的相似度
        all_similarities = []
        for i in range(len(self.metrics)):
            sim = self.compute_single_metric_similarity(query_features, prototypes, i)
            all_similarities.append(sim)

        # 在最后一维堆叠: [..., M]
        similarities = torch.stack(all_similarities, dim=-1)
        # 获取自适应融合权重
        metric_weights = self.compute_metric_weights(query_features)

        if query_features.dim() == 2:
            # similarities: [Q, N, M], metric_weights: [Q, M] → unsqueeze(1) → [Q, 1, M]
            weighted_similarities = torch.sum(
                similarities * metric_weights.unsqueeze(1),
                dim=-1
            )  # [Q, N]
        elif query_features.dim() == 3:
            # similarities: [B, Q, N, M], metric_weights: [B, Q, M] → unsqueeze(2) → [B, Q, 1, M]
            weighted_similarities = torch.sum(
                similarities * metric_weights.unsqueeze(2),
                dim=-1
            )  # [B, Q, N]
        else:
            raise ValueError(f"Unsupported query_features shape: {query_features.shape}")

        return weighted_similarities / self.temperature

    # =========================================================
    # 注意力原型：样本级自注意力加权
    # =========================================================
    def compute_attention_weights(self, features):
        """
        计算同一类别内各支持样本的注意力权重。

        核心思想：与类内其它样本越相似的样本（"越中心"的样本）应获得更高权重，
        从而使原型更稳健、不易受离群点干扰。

        消融隔离设计：
            - USE_MULTI_METRIC=True 时：利用马氏距离空间计算样本间相似度
            - USE_MULTI_METRIC=False 时：退化为纯点积（余弦相似度），彻底切断
              对 self.metrics 的梯度流，实现完美的消融隔离

        参数:
            features (Tensor): 同一类别的支持集特征，形状 [K, D]（K = k_shot）

        返回:
            attention_weights (Tensor): 归一化后的样本权重，形状 [K, 1]
        """
        if self.config.USE_MULTI_METRIC:
            # 用所有度量矩阵分别计算自相似度，取平均
            all_similarities = []
            for i in range(len(self.metrics)):
                sim = self.compute_single_metric_similarity(features, features, i)  # [K, K]
                all_similarities.append(sim)
            similarities = torch.stack(all_similarities).mean(0)  # [K, K]
        else:
            # 纯点积相似度，不依赖任何可学习度量矩阵
            similarities = torch.matmul(features, features.T)  # [K, K]

        # 每个样本与同类所有样本的平均相似度，作为"中心度"得分
        sample_scores = similarities.mean(dim=1)  # [K]

        # 用温度系数缩放后做 softmax，得到归一化的注意力权重
        attention_weights = F.softmax(sample_scores / self.temperature, dim=0).unsqueeze(1)  # [K, 1]

        return attention_weights

    # =========================================================
    # 原型计算：支持单 episode 和 batch episode
    # =========================================================
    def compute_prototypes(self, support_features, support_labels):
        """
        根据支持集特征和标签，计算每个类别的原型向量。

        当 USE_ATTENTION_PROTO=True 时：
            用 compute_attention_weights 为同类样本分配不同权重，加权求和得到原型。
        当 USE_ATTENTION_PROTO=False 时：
            简单取同类样本特征的均值作为原型。

        最终原型会经过 L2 归一化。

        支持两种输入：
            - 单 episode: support_features [S, D], support_labels [S] → 返回 [N, D]
            - batch episode: support_features [B, S, D], support_labels [B, S] → 返回 [B, N, D]

        参数:
            support_features (Tensor): 支持集经过投影+归一化后的特征
            support_labels   (Tensor): 支持集的 episode 内局部标签

        返回:
            prototypes (Tensor): 各类原型向量
        """
        # ---------- 单 episode（2D 输入）----------
        if support_features.dim() == 2:
            prototypes = []
            unique_labels = torch.unique(support_labels)  # 获取所有不重复的类别标签

            for label in unique_labels:
                mask = (support_labels == label)           # 布尔掩码：选出当前类别的样本
                class_features = support_features[mask]    # 当前类别的所有支持特征 [K, D]

                if self.config.USE_ATTENTION_PROTO:
                    # 注意力加权原型：离群样本权重低，中心样本权重高
                    attention_weights = self.compute_attention_weights(class_features)  # [K, 1]
                    prototype = (class_features * attention_weights).sum(dim=0)         # [D]
                else:
                    # 简单均值原型
                    prototype = class_features.mean(dim=0)  # [D]

                prototype = F.normalize(prototype, dim=0)  # L2 归一化到单位球面
                prototypes.append(prototype)

            return torch.stack(prototypes, dim=0)  # [N, D]

        # ---------- batch episode（3D 输入）----------
        if support_features.dim() == 3:
            batch_prototypes = []
            for b in range(support_features.size(0)):
                # 递归调用单 episode 版本
                proto_b = self.compute_prototypes(support_features[b], support_labels[b])  # [N, D]
                batch_prototypes.append(proto_b)

            return torch.stack(batch_prototypes, dim=0)  # [B, N, D]

        raise ValueError(f"Unsupported support_features shape: {support_features.shape}")

    # =========================================================
    # 相似度计算：支持开关 USE_MULTI_METRIC
    # =========================================================
    def compute_similarity(self, query_features, prototypes):
        """
        计算查询样本与各类原型之间的相似度分数。

        当 USE_MULTI_METRIC=True 时：
            调用 compute_multi_metric_similarity，使用多个马氏距离度量的自适应融合。
        当 USE_MULTI_METRIC=False 时：
            使用简单的点积（等价于余弦相似度，因为特征已 L2 归一化），
            再除以温度系数缩放。

        参数:
            query_features (Tensor): [Q, D] 或 [B, Q, D]
            prototypes     (Tensor): [N, D] 或 [B, N, D]

        返回:
            scores (Tensor): 缩放后的相似度分数，[Q, N] 或 [B, Q, N]
        """
        if self.config.USE_MULTI_METRIC:
            scores = self.compute_multi_metric_similarity(query_features, prototypes)
        else:
            if query_features.dim() == 2 and prototypes.dim() == 2:
                scores = torch.mm(query_features, prototypes.T)  # [Q, N]
            elif query_features.dim() == 3 and prototypes.dim() == 3:
                scores = torch.matmul(query_features, prototypes.transpose(1, 2))  # [B, Q, N]
            else:
                raise ValueError(
                    f"Unsupported shapes: query={query_features.shape}, prototypes={prototypes.shape}"
                )
            scores = scores / self.temperature  # 温度缩放
        return scores

    # =========================================================
    # 正则项：仅在 USE_MULTI_METRIC=True 时启用
    # =========================================================
    def compute_orthogonality_loss(self):
        """
        计算多个度量矩阵之间的正交正则损失。

        目的：鼓励不同度量矩阵学到不同的距离空间（多样性），
        避免所有度量退化为同一个距离函数。

        方法：对每对度量的 Gram 矩阵 (L_i = M_i^T M_i) 计算归一化内积，
        越小说明两个度量越正交（不相关）。

        返回:
            orth_loss (float): 所有度量对的正交损失之和
        """
        orth_loss = 0.0
        for i in range(len(self.metrics)):
            for j in range(i + 1, len(self.metrics)):
                Li = torch.matmul(self.metrics[i], self.metrics[i].T)  # Gram 矩阵 i
                Lj = torch.matmul(self.metrics[j], self.metrics[j].T)  # Gram 矩阵 j
                # 归一化 Frobenius 内积：衡量两个矩阵的相似程度
                orth_loss += torch.abs(torch.sum(Li * Lj)) / (torch.norm(Li) * torch.norm(Lj) + 1e-12)
        return orth_loss

    def metric_regularization(self):
        """
        计算多度量学习的综合正则化损失。

        当 USE_MULTI_METRIC=False 时直接返回 0（不添加任何正则项）。

        三个正则化组件：
            1. identity_reg:       各度量 Gram 矩阵与单位矩阵的距离，防止度量偏离太远
            2. orthogonality_loss: 度量间的正交损失（由 compute_orthogonality_loss 计算）
            3. layer_reg:          层权重的熵正则（仅 USE_MULTI_LAYER=True 时），
                                   鼓励层权重分布更均匀，避免退化为只用单层

        返回:
            Tensor: 综合正则化损失标量
        """
        if not self.config.USE_MULTI_METRIC:
            return torch.tensor(0.0, device=device)

        # 组件1：度量矩阵接近单位矩阵的正则
        identity_reg = 0.0
        for metric in self.metrics:
            L = torch.matmul(metric, metric.T)            # Gram 矩阵
            eye = torch.eye(L.size(0), device=L.device)   # 同维单位矩阵
            identity_reg += torch.norm(L - eye)            # Frobenius 范数

        # 组件2：度量矩阵间的正交正则
        orthogonality_loss = self.compute_orthogonality_loss()

        # 组件3：层权重熵正则（可选）
        if self.config.USE_MULTI_LAYER:
            layer_weights = F.softmax(self.layer_weights, dim=0)
            # 负熵：越大表示分布越均匀（鼓励均匀利用各层）
            layer_reg = -torch.sum(layer_weights * torch.log(layer_weights + 1e-10))
        else:
            layer_reg = torch.tensor(0.0, device=device)

        return identity_reg + self.config.METRIC_DIFF_WEIGHT * orthogonality_loss + 0.1 * layer_reg

    # =========================================================
    # 前向传播：支持 batch episode
    # =========================================================
    def forward(self, support_images, support_labels, query_images, is_training=True):
        """
        模型前向传播，完成从原始图像到分类分数的完整流程。

        兼容两种输入格式：
            - 单 episode:  support_images [S, 3, 224, 224]
            - batch episode: support_images [B, S, 3, 224, 224]
          内部统一升维为 batch 格式处理，单 episode 输入会在输出时自动降维。

        参数:
            support_images (Tensor): 支持集图像
            support_labels (Tensor): 支持集标签
            query_images   (Tensor): 查询集图像
            is_training    (bool):   是否处于训练模式（当前版本未对此做差异化处理，预留接口）

        返回:
            relation_scores (Tensor): 查询样本对各类的相似度分数，[B, Q, N] 或 [Q, N]
            prototypes      (Tensor): 各类原型向量，[B, N, D] 或 [N, D]
            query_features  (Tensor): 查询样本的嵌入特征，[B, Q, D] 或 [Q, D]
        """
        # ---- 兼容单 episode 输入：升维为 batch 格式 ----
        if support_images.dim() == 4:
            support_images = support_images.unsqueeze(0)  # [1, S, 3, 224, 224]
            support_labels = support_labels.unsqueeze(0)  # [1, S]
            query_images = query_images.unsqueeze(0)      # [1, Q, 3, 224, 224]
            squeeze_back = True  # 标记：输出时需要去掉 batch 维
        else:
            squeeze_back = False

        bsz, s_num = support_images.shape[:2]  # batch_size, 支持集样本数
        q_num = query_images.shape[1]           # 查询集样本数

        # ---- 支持集特征提取 ----
        # 将 [B, S, 3, 224, 224] 展平为 [B*S, 3, 224, 224] 以便批量送入 DINO
        support_images_flat = support_images.reshape(-1, *support_images.shape[2:])
        with torch.no_grad():  # DINO 骨干冻结，不需要梯度
            support_features = self.get_features(support_images_flat)  # [B*S, total_dim]
        support_features = self.projection(support_features)           # [B*S, PROJECTION_DIM]
        support_features = F.normalize(support_features, dim=1)        # L2 归一化
        support_features = support_features.reshape(bsz, s_num, -1)   # [B, S, D]

        # ---- 查询集特征提取（流程同上）----
        query_images_flat = query_images.reshape(-1, *query_images.shape[2:])
        with torch.no_grad():
            query_features = self.get_features(query_images_flat)      # [B*Q, total_dim]
        query_features = self.projection(query_features)               # [B*Q, PROJECTION_DIM]
        query_features = F.normalize(query_features, dim=1)            # L2 归一化
        query_features = query_features.reshape(bsz, q_num, -1)       # [B, Q, D]

        # ---- 计算原型并打分 ----
        prototypes = self.compute_prototypes(support_features, support_labels)  # [B, N, D]
        relation_scores = self.compute_similarity(query_features, prototypes)    # [B, Q, N]

        # ---- 单 episode 输入时去掉 batch 维 ----
        if squeeze_back:
            relation_scores = relation_scores.squeeze(0)  # [Q, N]
            prototypes = prototypes.squeeze(0)            # [N, D]
            query_features = query_features.squeeze(0)    # [Q, D]

        return relation_scores, prototypes, query_features


def train_epoch(model, dataloader, optimizer, criterion, device, config, log_interval=100):
    """
    执行一个完整的训练 epoch。

    参数:
        model        (FewShotModel):  模型实例
        dataloader   (DataLoader):    训练数据加载器
        optimizer    (Optimizer):     优化器（Adam）
        criterion    (Loss):          分类损失函数（CrossEntropyLoss）
        device       (torch.device):  计算设备
        config       (Config):        超参数配置
        log_interval (int):           每隔多少个 batch 打印一次日志

    返回:
        avg_loss (float): 本 epoch 的平均损失
        accuracy (float): 本 epoch 的训练准确率（百分比）
    """
    model.train()       # 切换到训练模式（启用 Dropout 等）
    running_loss = 0.0  # 累计损失
    correct = 0         # 累计正确预测数
    total = 0           # 累计总预测数

    for batch_idx, (support_images, support_labels, query_images, query_labels) in enumerate(
        tqdm(dataloader, desc='Training')
    ):
        # 将数据移到计算设备
        support_images = support_images.to(device)   # [B, S, 3, 224, 224]
        support_labels = support_labels.to(device)   # [B, S]
        query_images = query_images.to(device)       # [B, Q, 3, 224, 224]
        query_labels = query_labels.to(device)       # [B, Q]

        optimizer.zero_grad()  # 清零梯度

        # 前向传播
        relation_scores, prototypes, query_features = model(
            support_images, support_labels, query_images, is_training=True
        )  # relation_scores: [B, Q, N]

        # 交叉熵损失：将 [B, Q, N] 展平为 [B*Q, N]，标签展平为 [B*Q]
        ce_loss = criterion(
            relation_scores.reshape(-1, relation_scores.size(-1)),
            query_labels.reshape(-1)
        )

        # 如果启用多度量学习，附加正则损失
        if config.USE_MULTI_METRIC:
            reg_loss = model.metric_regularization()
            loss = ce_loss + config.REG_WEIGHT * reg_loss
        else:
            reg_loss = torch.tensor(0.0, device=device)
            loss = ce_loss

        loss.backward()    # 反向传播计算梯度
        optimizer.step()   # 更新参数

        # 统计准确率
        preds = torch.argmax(relation_scores, dim=-1)   # 取分数最高的类别: [B, Q]
        correct += (preds == query_labels).sum().item()
        total += query_labels.numel()
        running_loss += loss.item()

        # 定期打印训练日志
        if batch_idx % log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100 * correct / total

            # 如果启用多层融合，打印各层权重以便观察融合分布变化
            if config.USE_MULTI_LAYER:
                layer_weights = F.softmax(model.layer_weights, dim=0)
                layer_info = [f'{w:.3f}' for w in layer_weights.tolist()]
            else:
                layer_info = ["single-layer"]

            logging.info(
                f"Batch {batch_idx}, Loss: {avg_loss:.4f}, "
                f"CE Loss: {ce_loss.item():.4f}, "
                f"Reg Loss: {reg_loss.item():.4f}, "
                f"Accuracy: {accuracy:.2f}%, "
                f"Layer Info: {layer_info}"
            )

    return running_loss / len(dataloader), 100 * correct / total


def validate_epoch(model, dataloader, criterion, device, config):
    """
    执行一个完整的验证 epoch（不计算梯度）。

    参数:
        model     (FewShotModel):  模型实例
        dataloader(DataLoader):    验证数据加载器
        criterion (Loss):          分类损失函数
        device    (torch.device):  计算设备
        config    (Config):        超参数配置（当前验证阶段未直接使用，预留接口）

    返回:
        avg_loss (float): 本 epoch 的平均验证损失
        accuracy (float): 本 epoch 的验证准确率（百分比）
    """
    model.eval()        # 切换到评估模式（关闭 Dropout 等）
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算，节省显存和算力
        for batch_idx, (support_images, support_labels, query_images, query_labels) in enumerate(
            tqdm(dataloader, desc='Validation')
        ):
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)

            relation_scores, prototypes, query_features = model(
                support_images, support_labels, query_images, is_training=False
            )  # [B, Q, N]

            loss = criterion(
                relation_scores.reshape(-1, relation_scores.size(-1)),
                query_labels.reshape(-1)
            )

            preds = torch.argmax(relation_scores, dim=-1)
            correct += (preds == query_labels).sum().item()
            total += query_labels.numel()
            running_loss += loss.item()

    return running_loss / len(dataloader), 100 * correct / total


def save_model(model, save_path):
    """
    将模型权重（state_dict）保存到指定路径。

    参数:
        model     (nn.Module): 要保存的模型
        save_path (str):       保存路径（含文件名），如 './checkpoints/best_model_A1.pth'

    说明:
        自动创建目录（如不存在），保存失败时记录错误日志而非抛出异常。
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logging.info(f"模型已成功保存至 {save_path}")
    except Exception as e:
        logging.error(f"保存模型失败: {e}")


def load_model(model_path, config, device):
    """
    从磁盘加载模型权重并返回完整模型实例。

    参数:
        model_path (str):          模型权重文件路径
        config     (Config):       超参数配置（用于重建模型结构）
        device     (torch.device): 目标设备

    返回:
        FewShotModel | None: 加载成功返回模型实例，失败返回 None

    说明:
        使用 initialize_clip=False 构建空模型结构，再加载权重，
        避免重复下载/加载预训练权重。
    """
    try:
        model = FewShotModel(config, initialize_clip=False).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"完整模型已成功从 {model_path} 加载")
        return model
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        return None


def setup_logging(log_path):
    """
    配置日志系统：同时输出到文件和控制台。

    参数:
        log_path (str): 日志文件保存路径

    说明:
        日志格式为 "时间 - 级别 - 消息"，级别设为 INFO。
        自动创建日志目录（如不存在）。
    """
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_path),   # 写入日志文件
            logging.StreamHandler()           # 同步输出到控制台
        ]
    )

    logging.info("日志记录已成功配置")


def load_config():
    """
    创建 Config 实例并根据实验名自动填充输出路径。

    返回:
        Config: 填充完毕的配置对象

    自动生成的路径：
        - SAVE_PATH: './checkpoints/best_model_{实验名}.pth'
        - LOG_PATH:  './logs/training_{实验名}.log'
        - PLOT_PATH: './plots/training_validation_curves_{实验名}.png'
    """
    config = Config()
    config.SAVE_PATH = f'./checkpoints/best_model_{config.EXPERIMENT_NAME}.pth'
    config.LOG_PATH = f'./logs/training_{config.EXPERIMENT_NAME}.log'
    config.PLOT_PATH = f'./plots/training_validation_curves_{config.EXPERIMENT_NAME}.png'
    return config


def main():
    """
    主训练流程入口函数。

    执行流程：
        1. 加载配置 & 初始化日志
        2. 设置随机种子 & 检测设备
        3. 构建训练/验证数据集和数据加载器
        4. 初始化模型、损失函数、优化器、学习率调度器
        5. 训练循环（含早停机制）
        6. 绘制并保存训练损失和验证准确率曲线
    """
    # ---- 初始化 ----
    config = load_config()
    setup_logging(config.LOG_PATH)
    logging.info("开始主执行流程")

    # 打印当前消融实验配置，方便回溯
    logging.info(
        f"实验配置: {config.EXPERIMENT_NAME} | "
        f"USE_MULTI_LAYER={config.USE_MULTI_LAYER}, "
        f"USE_ATTENTION_PROTO={config.USE_ATTENTION_PROTO}, "
        f"USE_MULTI_METRIC={config.USE_MULTI_METRIC}"
    )

    set_seed(42)
    _device = get_device()

    # ---- 数据准备 ----
    train_transform = get_train_transforms()  # 训练集增强
    val_transform = get_val_transforms()      # 验证集预处理

    train_dataset = FewShotDataset(
        root_dir=os.path.join(config.SPLIT_DIR, 'train'),
        transform=train_transform,
        n_way=config.N_WAY,
        k_shot=config.K_SHOT,
        q_query=config.Q_QUERY
    )

    val_dataset = FewShotDataset(
        root_dir=os.path.join(config.SPLIT_DIR, 'val'),
        transform=val_transform,
        n_way=config.N_WAY,
        k_shot=config.K_SHOT,
        q_query=config.Q_QUERY
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,               # 训练集打乱顺序
        num_workers=10,              # 10 个子进程并行加载数据
        pin_memory=True,             # 将数据锁页到内存，加速 CPU→GPU 传输
        prefetch_factor=2,           # 每个 worker 预取 2 个 batch
        persistent_workers=True,     # 保持 worker 进程存活，避免每 epoch 重建开销
        collate_fn=custom_collate_fn # 自定义 batch 堆叠逻辑
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,               # 验证集不打乱
        num_workers=10,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )

    logging.info(f"训练集大小：{len(train_loader)} batches")
    logging.info(f"验证集大小：{len(val_loader)} batches")

    # ---- 模型 & 优化器 ----
    model = FewShotModel(config).to(_device)

    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # 只优化可训练参数（排除冻结的 DINO）
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # 余弦退火学习率调度器：学习率从初始值平滑衰减
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.SCHEDULER_T_MAX
    )

    # ---- 训练循环 ----
    best_val_accuracy = 0.0  # 历史最佳验证准确率
    trigger_times = 0        # 早停计数器：验证准确率未提升的连续 epoch 数

    train_losses = []    # 记录每 epoch 训练损失（用于绘图）
    val_accuracies = []  # 记录每 epoch 验证准确率（用于绘图）

    for epoch in range(1, config.NUM_EPOCHS + 1):
        logging.info(f"\n开始 Epoch {epoch}/{config.NUM_EPOCHS}")

        # 训练一个 epoch
        avg_train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=_device,
            config=config,
            log_interval=100
        )
        train_losses.append(avg_train_loss)
        logging.info(f"训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.2f}%")

        # 验证一个 epoch
        avg_val_loss, val_acc = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=_device,
            config=config
        )
        val_accuracies.append(val_acc)
        logging.info(f"验证损失: {avg_val_loss:.4f}, 验证准确率: {val_acc:.2f}%")

        # ---- 早停机制 ----
        # 验证准确率需超过历史最佳至少 DELTA 个百分点才算"有效提升"
        if val_acc > best_val_accuracy + config.DELTA:
            best_val_accuracy = val_acc
            trigger_times = 0  # 重置计数器
            save_model(model, config.SAVE_PATH)
            logging.info("保存了新的最佳模型。")
        else:
            trigger_times += 1
            logging.info(f"早停计数：{trigger_times}/{config.PATIENCE}")

        # 超过最少训练轮数且耐心耗尽 → 触发早停
        if epoch >= config.MIN_EPOCHS and trigger_times >= config.PATIENCE:
            logging.info("早停触发，停止训练。")
            break

        # 更新学习率
        scheduler.step()

    logging.info(f"\n训练完成！最佳验证准确率：{best_val_accuracy:.2f}%")

    # ---- 绘制训练曲线 ----
    plt.figure(figsize=(12, 5))

    # 子图1：训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    # 子图2：验证准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    os.makedirs('./plots', exist_ok=True)
    plt.savefig(config.PLOT_PATH)
    plt.show()

    logging.info(f"最佳验证准确率：{best_val_accuracy:.2f}%")
    logging.info("主执行流程已完成。")


if __name__ == "__main__":
    main()

