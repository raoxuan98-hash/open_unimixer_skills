#!/bin/bash

# Run UniMixer_lite_sinkhorn across all datasets.
# Small datasets (frappe_x1, movielenslatest_x1) share GPU 0.
# Other datasets each occupy a dedicated GPU.

python main/run_expid_sinkhorn.py --gpu 0 --dataset frappe_x1 --optimizer adamw --early_stop_patience 2 &
python main/run_expid_sinkhorn.py --gpu 0 --dataset movielenslatest_x1 --optimizer adamw --early_stop_patience 2 &
python main/run_expid_sinkhorn.py --gpu 2 --dataset taobaoad_x1 --optimizer adamw --early_stop_patience 2 &
python main/run_expid_sinkhorn.py --gpu 1 --dataset microvideo1.7m_x1 --optimizer adamw --early_stop_patience 2 &
python main/run_expid_sinkhorn.py --gpu 3 --dataset kuaivideo_x1 --optimizer adamw --early_stop_patience 2 &

wait