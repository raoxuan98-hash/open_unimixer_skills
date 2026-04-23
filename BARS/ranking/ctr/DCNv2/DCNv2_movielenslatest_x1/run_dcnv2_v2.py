#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run DCNv2 on MovielensLatest_x1 using FuxiCTR v2 API.
Compatible with BARS benchmark configs.
"""

import os
import sys

# Add FuxiCTR to path
sys.path.insert(0, '/home/raoxuan/projects/open_unimixer_skills/FuxiCTR')

import logging
import gc
import argparse
from datetime import datetime
from pathlib import Path

from fuxictr import datasets
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from model_zoo.DCNv2.src.DCNv2 import DCNv2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./DCNv2_movielenslatest_x1_tuner_config_01/',
                        help='The config directory.')
    parser.add_argument('--expid', type=str, default='DCNv2_movielenslatest_x1_016_98ea1c72',
                        help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())

    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']

    # Override data paths for FuxiCTR v2 with preprocessed parquet data
    params['data_root'] = '/home/raoxuan/projects/open_unimixer_skills/data'
    params['dataset_id'] = 'movielenslatest_x1'
    params['data_format'] = 'parquet'
    params['train_data'] = os.path.join(params['data_root'], params['dataset_id'], 'train')
    params['valid_data'] = os.path.join(params['data_root'], params['dataset_id'], 'valid')
    params['test_data'] = os.path.join(params['data_root'], params['dataset_id'], 'test')

    # FuxiCTR v2 uses early_stop_patience instead of patience
    if 'patience' in params:
        params['early_stop_patience'] = params['patience']

    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    # Load feature map
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    # Initialize model
    model = DCNv2(feature_map, **params)
    model.count_parameters()

    # Data loaders
    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()

    # Train
    model.fit(train_gen, validation_data=valid_gen, **params)

    # Validation evaluation
    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()

    # Test evaluation
    test_result = {}
    if params["test_data"]:
        logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result = model.evaluate(test_gen)

    # Save results
    result_filename = 'DCNv2_movielenslatest_x1_v2_results.csv'
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n'
                 .format(datetime.now().strftime('%Y%m%d-%H%M%S'),
                         ' '.join(sys.argv), experiment_id, params['dataset_id'],
                         "N.A.", print_to_list(valid_result), print_to_list(test_result)))

    print("\n===== Results =====")
    print(f"Validation: {valid_result}")
    print(f"Test:       {test_result}")
