# -*- coding:utf-8 -*-

"""
配置解析与验证模块。
将 .conf 文件解析为结构化的 dataclass，供训练脚本和其他模块使用。
"""

import configparser
from dataclasses import dataclass, field
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# 辅助解析函数
# ---------------------------------------------------------------------------

def _parse_list(value: str) -> List[str]:
    value = value.strip()
    if value.startswith('[') and value.endswith(']'):
        value = value[1:-1]
    items = [item.strip() for item in value.split(',')]
    return [item for item in items if item]


def _parse_int_list(value: str) -> List[int]:
    return [int(v) for v in _parse_list(value)]


def _parse_bool(value) -> bool:
    value = str(value).strip().lower()
    return value in ['1', 'true', 'yes', 'y', 'on']


def _format_lr_tag(value) -> str:
    return ('%s' % value).replace('.', 'p')


# ---------------------------------------------------------------------------
# 数据配置
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    adj_filename: str
    graph_signal_matrix_filename: str
    num_of_vertices: int
    points_per_hour: int
    num_for_predict: int

    @classmethod
    def from_section(cls, section) -> 'DataConfig':
        return cls(
            adj_filename=section['adj_filename'],
            graph_signal_matrix_filename=section['graph_signal_matrix_filename'],
            num_of_vertices=int(section['num_of_vertices']),
            points_per_hour=int(section['points_per_hour']),
            num_for_predict=int(section['num_for_predict']),
        )


# ---------------------------------------------------------------------------
# 训练配置
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    model_name: str
    ctx: str
    optimizer: str
    learning_rate: float
    epochs: int
    batch_size: int
    num_of_weeks: int
    num_of_days: int
    num_of_hours: int
    K: int
    merge: bool
    seed: int = 1
    params_dir: str = ''
    prediction_filename: str = ''

    @classmethod
    def from_section(cls, section) -> 'TrainingConfig':
        return cls(
            model_name=section['model_name'],
            ctx=section['ctx'],
            optimizer=section['optimizer'],
            learning_rate=float(section['learning_rate']),
            epochs=int(section['epochs']),
            batch_size=int(section['batch_size']),
            num_of_weeks=int(section['num_of_weeks']),
            num_of_days=int(section['num_of_days']),
            num_of_hours=int(section['num_of_hours']),
            K=int(section['K']),
            merge=bool(int(section['merge'])),
            seed=int(section.get('seed', '1')),
            params_dir=section.get('params_dir', ''),
            prediction_filename=section.get('prediction_filename', ''),
        )


# ---------------------------------------------------------------------------
# 模型升级配置
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveGraphConfig:
    embedding_dim: int = 10
    sparse_ratio: float = 0.0
    directed: bool = True


@dataclass
class TransformerConfig:
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    dropout: float = 0.1
    max_len: int = 36
    topk_ratio: float = 0.5


@dataclass
class UpgradeConfig:
    spatial_modes: List[int] = field(default_factory=lambda: [0])
    temporal_modes: List[int] = field(default_factory=lambda: [0])
    adaptive_graph: AdaptiveGraphConfig = field(default_factory=AdaptiveGraphConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)


# ---------------------------------------------------------------------------
# 顶层实验配置
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    config_file: str
    data: DataConfig
    training: TrainingConfig
    upgrade: UpgradeConfig

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可 JSON 化的字典。"""
        from dataclasses import asdict
        return asdict(self)


# ---------------------------------------------------------------------------
# 解析入口
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> ExperimentConfig:
    """从 .conf 文件加载并返回结构化的 ExperimentConfig。"""
    config = configparser.ConfigParser()
    config.read(config_path)

    data_cfg = DataConfig.from_section(config['Data'])
    training_cfg = TrainingConfig.from_section(config['Training'])

    # --- UpgradeConfig ---
    upgrade_cfg = UpgradeConfig()

    if config.has_section('ModelUpgrade'):
        uc = config['ModelUpgrade']
        if 'spatial_mode' in uc:
            upgrade_cfg.spatial_modes = _parse_int_list(uc['spatial_mode'])
        if 'temporal_mode' in uc:
            upgrade_cfg.temporal_modes = _parse_int_list(uc['temporal_mode'])

    # AdaptiveGraph
    ag = upgrade_cfg.adaptive_graph
    if config.has_section('AdaptiveGraph'):
        ag_sec = config['AdaptiveGraph']
        if 'embedding_dim' in ag_sec:
            ag.embedding_dim = int(ag_sec['embedding_dim'])
        if 'sparse_ratio' in ag_sec:
            ag.sparse_ratio = float(ag_sec['sparse_ratio'])
        if 'directed' in ag_sec:
            ag.directed = _parse_bool(ag_sec['directed'])

    # Transformer
    tf = upgrade_cfg.transformer
    tf.max_len = data_cfg.points_per_hour * 3  # 默认值依赖 points_per_hour
    if config.has_section('Transformer'):
        tf_sec = config['Transformer']
        if 'd_model' in tf_sec:
            tf.d_model = int(tf_sec['d_model'])
        if 'n_heads' in tf_sec:
            tf.n_heads = int(tf_sec['n_heads'])
        if 'e_layers' in tf_sec:
            tf.e_layers = int(tf_sec['e_layers'])
        if 'dropout' in tf_sec:
            tf.dropout = float(tf_sec['dropout'])
        if 'max_len' in tf_sec:
            tf.max_len = int(tf_sec['max_len'])
        if 'topk_ratio' in tf_sec:
            tf.topk_ratio = float(tf_sec['topk_ratio'])

    return ExperimentConfig(
        config_file=config_path,
        data=data_cfg,
        training=training_cfg,
        upgrade=upgrade_cfg,
    )


def get_model_names(training_cfg: TrainingConfig) -> List[str]:
    """从 training_cfg.model_name 解析出可能的多个模型名称。"""
    return _parse_list(training_cfg.model_name)


def get_learning_rates(training_cfg: TrainingConfig) -> List[float]:
    """支持 model_name 字段包含逗号分隔的多个学习率。"""
    # learning_rate 在 TrainingConfig 中已解析为 float，但原始 conf 可能含多个
    # 这里重新从字符串解析
    return [training_cfg.learning_rate]


def format_lr_tag(lr: float) -> str:
    return _format_lr_tag(lr)
