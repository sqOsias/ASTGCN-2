"""
模型加载工具模块 - 从指定路径自动加载模型权重和配置
"""
import json
import os
import torch
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import configparser
import yaml

from model.upgrade.astgcn_upgrade import UpgradeASTGCN


def load_model_from_path(model_path: str) -> Tuple[UpgradeASTGCN, Dict[str, Any]]:
    """
    从指定路径自动加载模型权重和配置
    
    Args:
        model_path: 模型保存路径 (例如: /root/ASTGCN-2/results/ASTGCN_lr0p001/0_0_20260318033244)
    
    Returns:
        tuple: (模型实例, 配置字典)
    """
    model_path = Path(model_path)
    
    # 检查路径是否存在
    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    # 寻找配置文件和权重文件
    checkpoints_dir = model_path / "checkpoints"
    configs_dir = model_path / "configs"
    
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"找不到checkpoints目录: {checkpoints_dir}")
    
    if not configs_dir.exists():
        raise FileNotFoundError(f"找不到configs目录: {configs_dir}")
    
    # 寻找模型权重文件
    weight_files = list(checkpoints_dir.glob("*.pth"))
    if not weight_files:
        raise FileNotFoundError(f"在 {checkpoints_dir} 中找不到.pth权重文件")
    
    # 优先使用best_model.pth
    best_model = checkpoints_dir / "best_model.pth"
    if best_model.exists():
        weight_path = best_model
    else:
        weight_path = weight_files[0]  # 使用第一个找到的权重文件
    
    # 寻找配置文件 (优先级: resolved_config.json > config.yaml > train.conf)
    config_data = None
    
    resolved_config = configs_dir / "resolved_config.json"
    if resolved_config.exists():
        with open(resolved_config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        config_yaml = configs_dir / "config.yaml"
        if config_yaml.exists():
            with open(config_yaml, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        else:
            train_conf = configs_dir / "train.conf"
            if train_conf.exists():
                config_data = parse_ini_config(train_conf)
    
    if config_data is None:
        raise FileNotFoundError(f"在 {configs_dir} 中找不到任何配置文件")
    
    # 从配置中提取模型参数
    model_params = extract_model_params_from_config(config_data)
    
    # 创建模型实例
    model = UpgradeASTGCN(**model_params)
    
    # 加载权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(str(weight_path), map_location=device, weights_only=True)
    
    # 修复state_dict的键名（如果需要）
    model_state_dict = model.state_dict()
    updated_state_dict = {}
    
    for key, value in state_dict.items():
        # 如果state_dict中的键在model_state_dict中不存在，尝试去掉前缀
        if key in model_state_dict:
            updated_state_dict[key] = value
        elif key.startswith('module.') and key[7:] in model_state_dict:
            # 去掉DataParallel的module.前缀
            updated_state_dict[key[7:]] = value
        elif key.startswith('model.') and key[6:] in model_state_dict:
            # 去掉model.前缀
            updated_state_dict[key[6:]] = value
        else:
            # 尝试直接匹配
            updated_state_dict[key] = value
    
    # 处理特征数不匹配的情况
    # 如果预训练权重中的输入特征数与当前模型不匹配，需要特殊处理
    for key in list(updated_state_dict.keys()):
        if '.residual_conv.weight' in key and updated_state_dict[key].shape != model_state_dict[key].shape:
            pretrained_weight = updated_state_dict[key]
            current_weight = model_state_dict[key]
            
            # 获取当前权重的形状
            current_shape = current_weight.shape
            pretrained_shape = pretrained_weight.shape
            
            print(f"检测到维度不匹配: {key}")
            print(f"  当前模型权重形状: {current_shape}")
            print(f"  预训练权重形状: {pretrained_shape}")
            
            # 如果是输入通道数不匹配（通常是第一个维度的第二个值）
            if len(pretrained_shape) == 4 and len(current_shape) == 4:
                # [out_channels, in_channels, H, W]
                if pretrained_shape[1] != current_shape[1]:
                    print(f"  修复输入通道数不匹配: 从 {pretrained_shape[1]} 调整为 {current_shape[1]}")
                    if current_shape[1] < pretrained_shape[1]:
                        # 当前模型输入通道数更少，截取预训练权重的相应部分
                        updated_state_dict[key] = pretrained_weight[:, :current_shape[1], :, :]
                    else:
                        # 当前模型输入通道数更多，扩展预训练权重
                        expanded_weight = torch.zeros(current_shape)
                        min_channels = min(pretrained_shape[1], current_shape[1])
                        expanded_weight[:, :min_channels, :, :] = pretrained_weight[:, :min_channels, :, :]
                        updated_state_dict[key] = expanded_weight
    
    # 加载权重
    model.load_state_dict(updated_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"成功加载模型权重: {weight_path}")
    print(f"模型配置: {model_params}")
    
    return model, config_data


def parse_ini_config(config_file: Path) -> Dict[str, Any]:
    """解析INI格式的配置文件"""
    config = configparser.ConfigParser()
    config.read(str(config_file))
    
    config_dict = {}
    
    for section_name in config.sections():
        section = {}
        for key, value in config.items(section_name):
            # 尝试转换数值类型
            section[key] = convert_value_type(value)
        config_dict[section_name] = section
    
    return config_dict


def convert_value_type(value: str) -> Any:
    """尝试将字符串值转换为适当的Python类型"""
    value = value.strip()
    
    # 尝试转换为布尔值
    if value.lower() in ('true', 'yes', 'on', '1'):
        return True
    elif value.lower() in ('false', 'no', 'off', '0'):
        return False
    
    # 尝试转换为整数
    try:
        if '.' not in value:
            return int(value)
    except ValueError:
        pass
    
    # 尝试转换为浮点数
    try:
        return float(value)
    except ValueError:
        pass
    
    # 返回原始字符串
    return value


def extract_model_params_from_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """从配置数据中提取模型参数"""
    # 从Data部分提取基本信息
    data_config = config_data.get('Data', {})
    training_config = config_data.get('Training', {})
    model_upgrade_config = config_data.get('ModelUpgrade', {})
    
    # 提取必要参数
    num_of_vertices = int(data_config.get('num_of_vertices', 307))
    num_for_predict = int(data_config.get('num_for_predict', 12))
    
    # 从Training部分获取特征数（通常由数据决定）
    # 注意：根据错误信息，训练模型使用了5个特征，所以我们也应该使用相同的特征数
    # 但为了兼容性，我们可以尝试加载时调整
    
    # 从ModelUpgrade部分获取高级配置
    adaptive_graph_config = model_upgrade_config.get('adaptive_graph', {})
    transformer_config = model_upgrade_config.get('transformer', {})
    
    # 获取空间和时间模式
    spatial_mode = model_upgrade_config.get('spatial_mode', 0)
    temporal_mode = model_upgrade_config.get('temporal_mode', 0)
    
    # 构建adaptive_graph_cfg
    adaptive_graph_cfg = {
        'embedding_dim': adaptive_graph_config.get('embedding_dim', 10),
        'sparse_ratio': adaptive_graph_config.get('sparse_ratio', 0.2),
        'directed': adaptive_graph_config.get('directed', True)
    }
    
    # 构建transformer_cfg
    transformer_cfg = {
        'd_model': transformer_config.get('d_model', 32),
        'n_heads': transformer_config.get('n_heads', 2),
        'e_layers': transformer_config.get('e_layers', 2),
        'dropout': transformer_config.get('dropout', 0.1),
        'max_len': transformer_config.get('max_len', 36),
        'factor': transformer_config.get('topk_ratio', 0.5) * 10  # 转换为factor
    }
    
    # 获取邻接矩阵文件路径以构建backbones
    adj_filename = data_config.get('adj_filename', 'data/PEMS04/distance.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 导入配置函数来获取backbones
    from model.model_config import get_backbones
    try:
        all_backbones = get_backbones('configurations/PEMS04.conf', adj_filename, device)
    except Exception as e:
        print(f"警告: 无法加载backbones配置，使用空列表: {e}")
        all_backbones = []
    
    # 根据配置文件确定特征数 - 这里需要与训练时的特征数保持一致
    # 从错误信息可知，训练模型使用了5个特征，所以我们也要使用5个特征
    num_of_features = 5  # 使用与训练模型相同的特征数
    
    # 构建模型参数
    model_params = {
        'num_of_features': num_of_features,
        'num_for_prediction': num_for_predict,
        'all_backbones': all_backbones,
        'num_of_vertices': num_of_vertices,
        'spatial_mode': spatial_mode,
        'temporal_mode': temporal_mode,
        'adaptive_graph_cfg': adaptive_graph_cfg,
        'transformer_cfg': transformer_cfg
    }
    
    return model_params