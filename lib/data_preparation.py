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

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if not sample:
            continue

        week_sample, day_sample, hour_sample, target = sample
        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 2, :]
        ))

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    if not merge:
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]
    else:
        print('Merge training set and validation set!')
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line2])]

    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

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
