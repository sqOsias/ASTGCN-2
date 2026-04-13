# -*- coding:utf-8 -*-

import numpy as np

from .utils import get_sample_indices


def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray

    Returns
    ----------
    stats: dict, two keys: mean and std

    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original

    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'mean': mean, 'std': std}, train_norm, val_norm, test_norm


def _clean_data_sequence(data_seq, clip_percentiles=None):
    """
    默认只修复 NaN/Inf，避免影响“速度为 0”这种可能真实的拥堵场景
    可选“分位数截断”抑制极端噪声（如 clip_percentiles=(0.1, 99.9) ），但不建议默认开启。
    """
    data_seq = np.asarray(data_seq)
    if not np.isfinite(data_seq).all():
        data_seq = data_seq.copy()
        data_seq[~np.isfinite(data_seq)] = 0.0
    if clip_percentiles:
        lower, upper = clip_percentiles
        clipped = data_seq.copy()
        for feature_idx in range(clipped.shape[2]):
            feature_values = clipped[:, :, feature_idx]
            low_value = np.percentile(feature_values, lower)
            high_value = np.percentile(feature_values, upper)
            clipped[:, :, feature_idx] = np.clip(feature_values, low_value, high_value)
        data_seq = clipped
    return data_seq


def _build_time_features(sequence_length, num_of_vertices, points_per_hour, dtype):
    points_per_day = 24 * points_per_hour
    time_index = np.arange(sequence_length)
    time_of_day = (time_index % points_per_day) / float(points_per_day)
    day_of_week = ((time_index // points_per_day) % 7) / 7.0
    time_of_day = np.repeat(time_of_day[:, np.newaxis], num_of_vertices, axis=1)
    day_of_week = np.repeat(day_of_week[:, np.newaxis], num_of_vertices, axis=1)
    time_of_day = time_of_day[:, :, np.newaxis]
    day_of_week = day_of_week[:, :, np.newaxis]
    features = np.concatenate([time_of_day, day_of_week], axis=2).astype(dtype)
    return features


def read_and_generate_dataset(graph_signal_matrix_filename,
                              num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=12, merge=False,
                              add_time_features=True,
                              clip_percentiles=None):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file

    num_of_weeks, num_of_days, num_of_hours: int

    num_for_predict: int

    points_per_hour: int, default 12, depends on data

    merge: boolean, default False,
           whether to merge training set and validation set to train model

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_batches * points_per_hour,
                       num_of_vertices, num_of_features)

    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)

    '''
    data_seq = np.load(graph_signal_matrix_filename)['data']
    data_seq = _clean_data_sequence(data_seq, clip_percentiles=clip_percentiles)
    if add_time_features:
        time_features = _build_time_features(
            data_seq.shape[0], data_seq.shape[1], points_per_hour, data_seq.dtype)
        data_seq = np.concatenate([data_seq, time_features], axis=2)

    # 首先找出所有有效的样本索引
    valid_indices = []
    for idx in range(data_seq.shape[0]):
        # 先检查是否能生成有效样本，而不实际生成数据
        sample_check = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                          num_of_hours, idx, num_for_predict,
                                          points_per_hour, only_check=True)
        if sample_check:
            valid_indices.append(idx)
    
    print(f'Total valid samples: {len(valid_indices)}')
    # todo 打印前10个valid_indices
    print(f'[INFO]First 10 valid indices: {valid_indices[:10]}')  
    
    # 计算分割点
    split_line1 = int(len(valid_indices) * 0.6)
    split_line2 = int(len(valid_indices) * 0.8)
    
    # 分批处理数据以节省内存
    def process_batch(indices):
        batch_samples = []
        for idx in indices:
            sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if sample:
                week_sample, day_sample, hour_sample, target = sample
                batch_samples.append((
                    np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
                    np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
                    np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
                    np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 2, :]
                ))
        if batch_samples:
            """
            在每个元组内沿 batch 维度（axis=0）拼接，把多个 (1, N, F, H) 拼成 (B, N, F, H)。
            最终返回 4 个数组，形状分别为：
            week: (B, N, F, num_of_weeks*points_per_hour)
            day: (B, N, F, num_of_days*points_per_hour)
            hour: (B, N, F, num_of_hours*points_per_hour)
            target: (B, N, num_for_predict)
            """
            return [np.concatenate(i, axis=0) for i in zip(*batch_samples)]
        else:
            return [np.array([]) for _ in range(4)]  # 返回空数组列表
    
    # 处理训练集
    if not merge:
        train_indices = valid_indices[:split_line1]
    else:
        print('Merge training set and validation set!')
        train_indices = valid_indices[:split_line2]
    
    training_set = process_batch(train_indices)
    
    # 处理验证集
    val_indices = valid_indices[split_line1:split_line2]
    validation_set = process_batch(val_indices)
    
    # 处理测试集
    test_indices = valid_indices[split_line2:]
    testing_set = process_batch(test_indices)
    
    train_week, train_day, train_hour, train_target = training_set
    val_week, val_day, val_hour, val_target = validation_set
    test_week, test_day, test_hour, test_target = testing_set

    print('training data: week: {}, day: {}, recent: {}, target: {}'.format(
        train_week.shape, train_day.shape,
        train_hour.shape, train_target.shape))
    print('validation data: week: {}, day: {}, recent: {}, target: {}'.format(
        val_week.shape, val_day.shape, val_hour.shape, val_target.shape))
    print('testing data: week: {}, day: {}, recent: {}, target: {}'.format(
        test_week.shape, test_day.shape, test_hour.shape, test_target.shape))

    # 清理临时变量以释放内存
    del training_set, validation_set, testing_set

    (week_stats, train_week_norm,
     val_week_norm, test_week_norm) = normalization(train_week,
                                                    val_week,
                                                    test_week)

    (day_stats, train_day_norm,
     val_day_norm, test_day_norm) = normalization(train_day,
                                                  val_day,
                                                  test_day)

    (recent_stats, train_recent_norm,
     val_recent_norm, test_recent_norm) = normalization(train_hour,
                                                        val_hour,
                                                        test_hour)

    all_data = {
        'train': {
            'week': train_week_norm,
            'day': train_day_norm,
            'recent': train_recent_norm,
            'target': train_target,
        },
        'val': {
            'week': val_week_norm,
            'day': val_day_norm,
            'recent': val_recent_norm,
            'target': val_target
        },
        'test': {
            'week': test_week_norm,
            'day': test_day_norm,
            'recent': test_recent_norm,
            'target': test_target
        },
        'stats': {
            'week': week_stats,
            'day': day_stats,
            'recent': recent_stats
        }
    }

    return all_data