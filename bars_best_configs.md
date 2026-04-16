# BARS Best Configs Collection

> **Source**: [BARS Benchmark](https://github.com/reczoo/BARS) 

> **Methodology**: Following the Wukong paper, we directly use the best searched model configs on BARS whenever possible, and use the provided model default hyperparameters for the rest.

This document collects the best searched configurations for various CTR models evaluated on public datasets via the BARS framework.

## Supported Datasets

- **Frappe**
- **MicroVideo1.7M**
- **MovielensLatest**
- **KuaiVideo**
- **TaobaoAd**

## Supported Models

- **AFN**
- **AutoInt**
- **DCNv2**
- **DLRM**
- **DeepFM**
- **DIN**
- **FinalMLP**
- **MaskNet**
- **xDeepFM**

---

## AFN

### Frappe

- **ExpID**: `AFN_frappe_x1_008_f15b0bf0`
- **BARS Directory**: `BARS/ranking/ctr/AFN/AFN_frappe_x1`

#### Model Config

```yaml
AFN_frappe_x1_008_f15b0bf0:
  afn_activations: relu
  afn_dropout: '0.4'
  afn_hidden_units: '[400]'
  batch_norm: 'False'
  batch_size: '4096'
  debug: 'False'
  dnn_activations: relu
  dnn_dropout: '0'
  dnn_hidden_units: '[]'
  embedding_dim: '10'
  embedding_regularizer: '0.001'
  ensemble_dnn: 'False'
  epochs: '100'
  every_x_epochs: '1'
  gpu: '1'
  learning_rate: '0.001'
  logarithmic_neurons: '1000'
  loss: binary_crossentropy
  metrics: '[''AUC'', ''logloss'']'
  model: AFN
  model_id: AFN_frappe_x1_008_f15b0bf0
  model_root: ./Frappe/AFN_frappe_x1/
  monitor: AUC
  monitor_mode: max
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  partition_block_size: '-1'
  patience: '2'
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
  version: pytorch
```

#### Dataset Config

```yaml
frappe_x1_04e961e9:
  data_format: csv
  data_root: ../data/Frappe/
  dataset_id: frappe_x1_04e961e9
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user'', ''item'',
    ''daytime'', ''weekday'', ''isweekend'', ''homework'', ''cost'', ''weather'',
    ''country'', ''city''], ''type'': ''categorical''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Frappe/Frappe_x1/test.csv
  train_data: ../data/Frappe/Frappe_x1/train.csv
  use_hdf5: 'True'
  valid_data: ../data/Frappe/Frappe_x1/valid.csv
```

### MovielensLatest

- **ExpID**: `AFN_movielenslatest_x1_015_9d6fa874`
- **BARS Directory**: `BARS/ranking/ctr/AFN/AFN_movielenslatest_x1`
- **BARS AUC**: `0.9566881481481482`
- **BARS LogLoss**: `0.31259344444444437`

#### Model Config

```yaml
AFN_movielenslatest_x1_015_9d6fa874:
  afn_activations: relu
  afn_dropout: '0.4'
  afn_hidden_units: '[200]'
  batch_norm: 'False'
  batch_size: '4096'
  debug: 'False'
  dnn_activations: relu
  dnn_dropout: '0'
  dnn_hidden_units: '[]'
  embedding_dim: '10'
  embedding_regularizer: '0.001'
  ensemble_dnn: 'False'
  epochs: '100'
  every_x_epochs: '1'
  gpu: '1'
  learning_rate: '0.001'
  logarithmic_neurons: '800'
  loss: binary_crossentropy
  metrics: '[''AUC'', ''logloss'']'
  model: AFN
  model_id: AFN_movielenslatest_x1_015_9d6fa874
  model_root: ./Movielens/AFN_movielenslatest_x1/
  monitor: AUC
  monitor_mode: max
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  partition_block_size: '-1'
  patience: '2'
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
  version: pytorch
```

#### Dataset Config

```yaml
movielenslatest_x1_cd32d937:
  data_format: csv
  data_root: ../data/Movielens/
  dataset_id: movielenslatest_x1_cd32d937
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user_id'',
    ''item_id'', ''tag_id''], ''type'': ''categorical''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Movielens/MovielensLatest_x1/test.csv
  train_data: ../data/Movielens/MovielensLatest_x1/train.csv
  use_hdf5: 'True'
  valid_data: ../data/Movielens/MovielensLatest_x1/valid.csv
```

---

## AutoInt

### Frappe

- **ExpID**: `AutoInt_frappe_x1_005_2b296630`
- **BARS Directory**: `BARS/ranking/ctr/AutoInt/AutoInt_frappe_x1`

#### Model Config

```yaml
AutoInt_frappe_x1_005_2b296630:
  attention_dim: '128'
  attention_layers: '6'
  batch_norm: 'False'
  batch_size: '4096'
  debug: 'False'
  dnn_activations: relu
  dnn_hidden_units: '[]'
  embedding_dim: '10'
  embedding_regularizer: '0.05'
  epochs: '100'
  every_x_epochs: '1'
  gpu: '0'
  layer_norm: 'False'
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''AUC'', ''logloss'']'
  model: AutoInt
  model_id: AutoInt_frappe_x1_005_2b296630
  model_root: ./Frappe/AutoInt_frappe_x1/
  monitor: AUC
  monitor_mode: max
  net_dropout: '0'
  net_regularizer: '0'
  num_heads: '4'
  num_workers: '3'
  optimizer: adam
  patience: '2'
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  task: binary_classification
  use_residual: 'True'
  use_scale: 'False'
  use_wide: 'False'
  verbose: '1'
  version: pytorch
```

#### Dataset Config

```yaml
frappe_x1_04e961e9:
  data_format: csv
  data_root: ../data/Frappe/
  dataset_id: frappe_x1_04e961e9
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user'', ''item'',
    ''daytime'', ''weekday'', ''isweekend'', ''homework'', ''cost'', ''weather'',
    ''country'', ''city''], ''type'': ''categorical''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Frappe/Frappe_x1/test.csv
  train_data: ../data/Frappe/Frappe_x1/train.csv
  use_hdf5: 'True'
  valid_data: ../data/Frappe/Frappe_x1/valid.csv
```

### KuaiVideo

- **ExpID**: `AutoInt_kuaivideo_x1_011_9ed2831b`
- **BARS Directory**: `BARS/ranking/ctr/AutoInt/AutoInt+_kuaivideo_x1`

#### Model Config

```yaml
AutoInt_kuaivideo_x1_011_9ed2831b:
  attention_dim: '512'
  attention_layers: '3'
  batch_norm: 'False'
  batch_size: '8192'
  debug_mode: 'False'
  dnn_activations: relu
  dnn_hidden_units: '[1024, 512, 256]'
  early_stop_patience: '2'
  embedding_dim: '64'
  embedding_regularizer: '0.0001'
  epochs: '100'
  eval_interval: '1'
  gpu: '0'
  group_id: group_id
  layer_norm: 'False'
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''gAUC'', ''AUC'', ''logloss'']'
  model: AutoInt
  model_id: AutoInt_kuaivideo_x1_011_9ed2831b
  model_root: ./checkpoints/AutoInt_kuaivideo_x1/
  monitor: '{''AUC'': 1, ''gAUC'': 1}'
  monitor_mode: max
  net_dropout: '0.3'
  net_regularizer: '0'
  num_heads: '4'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  save_best_only: 'True'
  seed: '20222023'
  shuffle: 'True'
  task: binary_classification
  use_residual: 'True'
  use_scale: 'False'
  use_wide: 'False'
  verbose: '1'
```

#### Dataset Config

```yaml
kuaivideo_x1_dc7a3035:
  data_format: csv
  data_root: ../data/KuaiShou/
  dataset_id: kuaivideo_x1_dc7a3035
  feature_cols: '[{''active'': True, ''dtype'': ''int'', ''name'': ''group_id'', ''preprocess'':
    ''copy_from(user_id)'', ''remap'': False, ''type'': ''meta''}, {''active'': True,
    ''dtype'': ''str'', ''name'': ''user_id'', ''type'': ''categorical''}, {''active'':
    True, ''dtype'': ''str'', ''name'': ''item_id'', ''type'': ''categorical''}, {''active'':
    True, ''dtype'': ''str'', ''embedding_dim'': 64, ''min_categr_count'': 1, ''name'':
    ''item_emb'', ''preprocess'': ''copy_from(item_id)'', ''pretrained_emb'': ''../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5'',
    ''type'': ''categorical''}, {''active'': True, ''dtype'': ''str'', ''max_len'':
    100, ''name'': ''pos_items'', ''padding'': ''pre'', ''share_embedding'': ''item_id'',
    ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': True, ''dtype'': ''str'',
    ''max_len'': 100, ''name'': ''neg_items'', ''padding'': ''pre'', ''share_embedding'':
    ''item_id'', ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': True,
    ''dtype'': ''str'', ''embedding_dim'': 64, ''max_len'': 100, ''min_categr_count'':
    1, ''name'': ''pos_items_emb'', ''padding'': ''pre'', ''preprocess'': ''copy_from(pos_items)'',
    ''pretrained_emb'': ''../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5'',
    ''share_embedding'': ''item_emb'', ''splitter'': ''^'', ''type'': ''sequence''},
    {''active'': True, ''dtype'': ''str'', ''embedding_dim'': 64, ''max_len'': 100,
    ''min_categr_count'': 1, ''name'': ''neg_items_emb'', ''padding'': ''pre'', ''preprocess'':
    ''copy_from(neg_items)'', ''pretrained_emb'': ''../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5'',
    ''share_embedding'': ''item_emb'', ''splitter'': ''^'', ''type'': ''sequence''}]'
  feature_specs: '[{''feature_encoder'': ''nn.Linear(64, 64, bias=False)'', ''name'':
    ''item_emb''}, {''feature_encoder'': ''layers.MaskedAveragePooling()'', ''name'':
    ''pos_items''}, {''feature_encoder'': ''layers.MaskedAveragePooling()'', ''name'':
    ''neg_items''}, {''feature_encoder'': [''layers.MaskedAveragePooling()'', ''nn.Linear(64,
    64, bias=False)''], ''name'': ''pos_items_emb''}, {''feature_encoder'': [''layers.MaskedAveragePooling()'',
    ''nn.Linear(64, 64, bias=False)''], ''name'': ''neg_items_emb''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''is_click''}'
  min_categr_count: '10'
  pickle_feature_encoder: 'True'
  test_data: ../data/KuaiShou/KuaiVideo_x1/test.csv
  train_data: ../data/KuaiShou/KuaiVideo_x1/train.csv
  valid_data: ../data/KuaiShou/KuaiVideo_x1/test.csv
```

### MicroVideo1.7M

- **ExpID**: `AutoInt_microvideo1.7m_x1_029_f813da5f`
- **BARS Directory**: `BARS/ranking/ctr/AutoInt/AutoInt+_microvideo1.7m_x1`

#### Model Config

```yaml
AutoInt_microvideo1.7m_x1_029_f813da5f:
  attention_dim: '128'
  attention_layers: '3'
  batch_norm: 'True'
  batch_size: '2048'
  debug_mode: 'False'
  dnn_activations: relu
  dnn_hidden_units: '[1024, 512, 256]'
  early_stop_patience: '2'
  embedding_dim: '64'
  embedding_regularizer: '0.001'
  epochs: '100'
  eval_interval: '1'
  gpu: '4'
  group_id: group_id
  layer_norm: 'True'
  learning_rate: '0.0005'
  loss: binary_crossentropy
  metrics: '[''gAUC'', ''AUC'', ''logloss'']'
  model: AutoInt
  model_id: AutoInt_microvideo1.7m_x1_029_f813da5f
  model_root: ./checkpoints/AutoInt_microvideo1.7m_x1/
  monitor: '{''AUC'': 1, ''gAUC'': 1}'
  monitor_mode: max
  net_dropout: '0.2'
  net_regularizer: '0'
  num_heads: '2'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  save_best_only: 'True'
  seed: '2022'
  shuffle: 'True'
  task: binary_classification
  use_residual: 'True'
  use_scale: 'False'
  use_wide: 'False'
  verbose: '1'
```

#### Dataset Config

```yaml
microvideo1.7m_x1_0d855fe6:
  data_format: csv
  data_root: ../data/MicroVideo1.7M/
  dataset_id: microvideo1.7m_x1_0d855fe6
  feature_cols: '[{''active'': True, ''dtype'': ''int'', ''name'': ''group_id'', ''preprocess'':
    ''copy_from(user_id)'', ''remap'': False, ''type'': ''meta''}, {''active'': True,
    ''dtype'': ''str'', ''name'': ''user_id'', ''type'': ''categorical''}, {''active'':
    True, ''dtype'': ''str'', ''embedding_dim'': 64, ''name'': ''item_id'', ''pretrained_emb'':
    ''../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5'', ''type'':
    ''categorical''}, {''active'': True, ''dtype'': ''str'', ''name'': ''cate_id'',
    ''type'': ''categorical''}, {''active'': True, ''dtype'': ''str'', ''embedding_dim'':
    64, ''max_len'': 100, ''name'': ''clicked_items'', ''padding'': ''pre'', ''pretrained_emb'':
    ''../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5'', ''splitter'':
    ''^'', ''type'': ''sequence''}, {''active'': True, ''dtype'': ''str'', ''max_len'':
    100, ''name'': ''clicked_categories'', ''padding'': ''pre'', ''share_embedding'':
    ''cate_id'', ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': False,
    ''dtype'': ''str'', ''name'': ''timestamp'', ''type'': ''categorical''}]'
  feature_specs: '[{''feature_encoder'': ''nn.Linear(64, 64, bias=False)'', ''name'':
    ''item_id''}, {''feature_encoder'': [''layers.MaskedAveragePooling()'', ''nn.Linear(64,
    64, bias=False)''], ''name'': ''clicked_items''}, {''feature_encoder'': ''layers.MaskedAveragePooling()'',
    ''name'': ''clicked_categories''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''is_click''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv
  train_data: ../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv
  valid_data: ../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv
```

### MovielensLatest

- **ExpID**: `AutoInt_movielenslatest_x1_004_4795ccb3`
- **BARS Directory**: `BARS/ranking/ctr/AutoInt/AutoInt_movielenslatest_x1`

#### Model Config

```yaml
AutoInt_movielenslatest_x1_004_4795ccb3:
  attention_dim: '128'
  attention_layers: '2'
  batch_norm: 'False'
  batch_size: '4096'
  debug: 'False'
  dnn_activations: relu
  dnn_hidden_units: '[]'
  embedding_dim: '10'
  embedding_regularizer: '0.01'
  epochs: '100'
  every_x_epochs: '1'
  gpu: '1'
  layer_norm: 'True'
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''AUC'', ''logloss'']'
  model: AutoInt
  model_id: AutoInt_movielenslatest_x1_004_4795ccb3
  model_root: ./Movielens/AutoInt_movielenslatest_x1/
  monitor: AUC
  monitor_mode: max
  net_dropout: '0'
  net_regularizer: '0'
  num_heads: '4'
  num_workers: '3'
  optimizer: adam
  patience: '2'
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  task: binary_classification
  use_residual: 'True'
  use_scale: 'True'
  use_wide: 'True'
  verbose: '1'
  version: pytorch
```

#### Dataset Config

```yaml
movielenslatest_x1_cd32d937:
  data_format: csv
  data_root: ../data/Movielens/
  dataset_id: movielenslatest_x1_cd32d937
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user_id'',
    ''item_id'', ''tag_id''], ''type'': ''categorical''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Movielens/MovielensLatest_x1/test.csv
  train_data: ../data/Movielens/MovielensLatest_x1/train.csv
  use_hdf5: 'True'
  valid_data: ../data/Movielens/MovielensLatest_x1/valid.csv
```

### TaobaoAd

- **ExpID**: `AutoInt_taobaoad_x1_012_778cf7e5`
- **BARS Directory**: `BARS/ranking/ctr/AutoInt/AutoInt+_taobaoad_x1`

#### Model Config

```yaml
AutoInt_taobaoad_x1_012_778cf7e5:
  attention_dim: '128'
  attention_layers: '3'
  batch_norm: 'False'
  batch_size: '8192'
  debug_mode: 'False'
  dnn_activations: relu
  dnn_hidden_units: '[512, 256, 128]'
  early_stop_patience: '1'
  embedding_dim: '32'
  embedding_regularizer: 5e-06
  epochs: '100'
  eval_interval: '1'
  gpu: '3'
  group_id: group_id
  layer_norm: 'False'
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''gAUC'', ''AUC'', ''logloss'']'
  model: AutoInt
  model_id: AutoInt_taobaoad_x1_012_778cf7e5
  model_root: ./checkpoints/AutoInt_taobaoad_x1/
  monitor: '{''AUC'': 1, ''gAUC'': 1}'
  monitor_mode: max
  net_dropout: '0.1'
  net_regularizer: '0'
  num_heads: '1'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  save_best_only: 'True'
  seed: '20222023'
  shuffle: 'True'
  task: binary_classification
  use_residual: 'True'
  use_scale: 'False'
  use_wide: 'False'
  verbose: '1'
```

#### Dataset Config

```yaml
taobaoad_x1_2753db8a:
  data_format: csv
  data_root: ../data/Taobao/
  dataset_id: taobaoad_x1_2753db8a
  feature_cols: '[{''active'': True, ''dtype'': ''int'', ''name'': ''group_id'', ''preprocess'':
    ''copy_from(userid)'', ''remap'': False, ''type'': ''meta''}, {''active'': True,
    ''dtype'': ''str'', ''name'': [''userid'', ''cms_segid'', ''cms_group_id'', ''final_gender_code'',
    ''age_level'', ''pvalue_level'', ''shopping_level'', ''occupation'', ''new_user_class_level'',
    ''adgroup_id'', ''cate_id'', ''campaign_id'', ''customer'', ''brand'', ''pid'',
    ''btag''], ''type'': ''categorical''}, {''active'': True, ''dtype'': ''float'',
    ''name'': ''price'', ''type'': ''numeric''}, {''active'': True, ''dtype'': ''str'',
    ''max_len'': 50, ''name'': ''cate_his'', ''padding'': ''pre'', ''share_embedding'':
    ''cate_id'', ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': True,
    ''dtype'': ''str'', ''max_len'': 50, ''name'': ''brand_his'', ''padding'': ''pre'',
    ''share_embedding'': ''brand'', ''splitter'': ''^'', ''type'': ''sequence''},
    {''active'': True, ''dtype'': ''str'', ''max_len'': 50, ''name'': ''btag_his'',
    ''padding'': ''pre'', ''share_embedding'': ''btag'', ''splitter'': ''^'', ''type'':
    ''sequence''}]'
  feature_specs: None
  label_col: '{''dtype'': ''float'', ''name'': ''clk''}'
  min_categr_count: '10'
  pickle_feature_encoder: 'True'
  test_data: ../data/Taobao/TaobaoAd_x1/test.csv
  train_data: ../data/Taobao/TaobaoAd_x1/train.csv
  valid_data: ../data/Taobao/TaobaoAd_x1/test.csv
```

---

## DCNv2

### Frappe

- **ExpID**: `DCNv2_frappe_x1_007_c207b717`
- **BARS Directory**: `BARS/ranking/ctr/DCNv2/DCNv2_frappe_x1`
- **BARS AUC**: `0.983824`
- **BARS LogLoss**: `0.152036`

#### Model Config

```yaml
DCNv2_frappe_x1_007_c207b717:
  batch_norm: 'True'
  batch_size: '4096'
  debug: 'False'
  dnn_activations: relu
  embedding_dim: '10'
  embedding_regularizer: '0.05'
  epochs: '100'
  every_x_epochs: '1'
  gpu: '1'
  learning_rate: '0.001'
  loss: binary_crossentropy
  low_rank: '32'
  metrics: '[''AUC'', ''logloss'']'
  model: DCNv2
  model_id: DCNv2_frappe_x1_007_c207b717
  model_root: ./Frappe/DCN_frappe_x1/
  model_structure: parallel
  monitor: AUC
  monitor_mode: max
  net_dropout: '0.4'
  net_regularizer: '0'
  num_cross_layers: '4'
  num_experts: '4'
  num_workers: '3'
  optimizer: adam
  parallel_dnn_hidden_units: '[400, 400, 400]'
  patience: '2'
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  stacked_dnn_hidden_units: '[500, 500, 500]'
  task: binary_classification
  use_low_rank_mixture: 'False'
  verbose: '1'
  version: pytorch
```

#### Dataset Config

```yaml
frappe_x1_04e961e9:
  data_format: csv
  data_root: ../data/Frappe/
  dataset_id: frappe_x1_04e961e9
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user'', ''item'',
    ''daytime'', ''weekday'', ''isweekend'', ''homework'', ''cost'', ''weather'',
    ''country'', ''city''], ''type'': ''categorical''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Frappe/Frappe_x1/test.csv
  train_data: ../data/Frappe/Frappe_x1/train.csv
  use_hdf5: 'True'
  valid_data: ../data/Frappe/Frappe_x1/valid.csv
```

### KuaiVideo

- **ExpID**: `DCNv2_kuaivideo_x1_008_7047bff3`
- **BARS Directory**: `BARS/ranking/ctr/DCNv2/DCNv2_kuaivideo_x1`

#### Model Config

```yaml
DCNv2_kuaivideo_x1_008_7047bff3:
  batch_norm: 'True'
  batch_size: '8192'
  debug_mode: 'False'
  dnn_activations: relu
  early_stop_patience: '2'
  embedding_dim: '64'
  embedding_regularizer: '0.0001'
  epochs: '100'
  eval_interval: '1'
  gpu: '7'
  group_id: group_id
  learning_rate: '0.001'
  loss: binary_crossentropy
  low_rank: '32'
  metrics: '[''gAUC'', ''AUC'', ''logloss'']'
  model: DCNv2
  model_id: DCNv2_kuaivideo_x1_008_7047bff3
  model_root: ./checkpoints/DCNv2_kuaivideo_x1/
  model_structure: parallel
  monitor: '{''AUC'': 1, ''gAUC'': 1}'
  monitor_mode: max
  net_dropout: '0.1'
  net_regularizer: '0'
  num_cross_layers: '2'
  num_experts: '4'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  parallel_dnn_hidden_units: '[1024, 512, 256]'
  save_best_only: 'True'
  seed: '20222023'
  shuffle: 'True'
  stacked_dnn_hidden_units: '[500, 500, 500]'
  task: binary_classification
  use_low_rank_mixture: 'False'
  verbose: '1'
```

#### Dataset Config

```yaml
kuaivideo_x1_dc7a3035:
  data_format: csv
  data_root: ../data/KuaiShou/
  dataset_id: kuaivideo_x1_dc7a3035
  feature_cols: '[{''active'': True, ''dtype'': ''int'', ''name'': ''group_id'', ''preprocess'':
    ''copy_from(user_id)'', ''remap'': False, ''type'': ''meta''}, {''active'': True,
    ''dtype'': ''str'', ''name'': ''user_id'', ''type'': ''categorical''}, {''active'':
    True, ''dtype'': ''str'', ''name'': ''item_id'', ''type'': ''categorical''}, {''active'':
    True, ''dtype'': ''str'', ''embedding_dim'': 64, ''min_categr_count'': 1, ''name'':
    ''item_emb'', ''preprocess'': ''copy_from(item_id)'', ''pretrained_emb'': ''../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5'',
    ''type'': ''categorical''}, {''active'': True, ''dtype'': ''str'', ''max_len'':
    100, ''name'': ''pos_items'', ''padding'': ''pre'', ''share_embedding'': ''item_id'',
    ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': True, ''dtype'': ''str'',
    ''max_len'': 100, ''name'': ''neg_items'', ''padding'': ''pre'', ''share_embedding'':
    ''item_id'', ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': True,
    ''dtype'': ''str'', ''embedding_dim'': 64, ''max_len'': 100, ''min_categr_count'':
    1, ''name'': ''pos_items_emb'', ''padding'': ''pre'', ''preprocess'': ''copy_from(pos_items)'',
    ''pretrained_emb'': ''../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5'',
    ''share_embedding'': ''item_emb'', ''splitter'': ''^'', ''type'': ''sequence''},
    {''active'': True, ''dtype'': ''str'', ''embedding_dim'': 64, ''max_len'': 100,
    ''min_categr_count'': 1, ''name'': ''neg_items_emb'', ''padding'': ''pre'', ''preprocess'':
    ''copy_from(neg_items)'', ''pretrained_emb'': ''../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5'',
    ''share_embedding'': ''item_emb'', ''splitter'': ''^'', ''type'': ''sequence''}]'
  feature_specs: '[{''feature_encoder'': ''nn.Linear(64, 64, bias=False)'', ''name'':
    ''item_emb''}, {''feature_encoder'': ''layers.MaskedAveragePooling()'', ''name'':
    ''pos_items''}, {''feature_encoder'': ''layers.MaskedAveragePooling()'', ''name'':
    ''neg_items''}, {''feature_encoder'': [''layers.MaskedAveragePooling()'', ''nn.Linear(64,
    64, bias=False)''], ''name'': ''pos_items_emb''}, {''feature_encoder'': [''layers.MaskedAveragePooling()'',
    ''nn.Linear(64, 64, bias=False)''], ''name'': ''neg_items_emb''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''is_click''}'
  min_categr_count: '10'
  pickle_feature_encoder: 'True'
  test_data: ../data/KuaiShou/KuaiVideo_x1/test.csv
  train_data: ../data/KuaiShou/KuaiVideo_x1/train.csv
  valid_data: ../data/KuaiShou/KuaiVideo_x1/test.csv
```

### MicroVideo1.7M

- **ExpID**: `DCNv2_microvideo1.7m_x1_017_9199218b`
- **BARS Directory**: `BARS/ranking/ctr/DCNv2/DCNv2_microvideo1.7m_x1`

#### Model Config

```yaml
DCNv2_microvideo1.7m_x1_017_9199218b:
  batch_norm: 'True'
  batch_size: '2048'
  debug_mode: 'False'
  dnn_activations: relu
  early_stop_patience: '2'
  embedding_dim: '64'
  embedding_regularizer: '0.001'
  epochs: '100'
  eval_interval: '1'
  gpu: '0'
  group_id: group_id
  learning_rate: '0.0005'
  loss: binary_crossentropy
  low_rank: '32'
  metrics: '[''gAUC'', ''AUC'', ''logloss'']'
  model: DCNv2
  model_id: DCNv2_microvideo1.7m_x1_017_9199218b
  model_root: ./checkpoints/DCNv2_microvideo1.7m_x1/
  model_structure: parallel
  monitor: '{''AUC'': 1, ''gAUC'': 1}'
  monitor_mode: max
  net_dropout: '0.1'
  net_regularizer: '0'
  num_cross_layers: '3'
  num_experts: '4'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  parallel_dnn_hidden_units: '[1024, 512, 256]'
  save_best_only: 'True'
  seed: '2022'
  shuffle: 'True'
  stacked_dnn_hidden_units: '[500, 500, 500]'
  task: binary_classification
  use_low_rank_mixture: 'False'
  verbose: '1'
```

#### Dataset Config

```yaml
microvideo1.7m_x1_0d855fe6:
  data_format: csv
  data_root: ../data/MicroVideo1.7M/
  dataset_id: microvideo1.7m_x1_0d855fe6
  feature_cols: '[{''active'': True, ''dtype'': ''int'', ''name'': ''group_id'', ''preprocess'':
    ''copy_from(user_id)'', ''remap'': False, ''type'': ''meta''}, {''active'': True,
    ''dtype'': ''str'', ''name'': ''user_id'', ''type'': ''categorical''}, {''active'':
    True, ''dtype'': ''str'', ''embedding_dim'': 64, ''name'': ''item_id'', ''pretrained_emb'':
    ''../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5'', ''type'':
    ''categorical''}, {''active'': True, ''dtype'': ''str'', ''name'': ''cate_id'',
    ''type'': ''categorical''}, {''active'': True, ''dtype'': ''str'', ''embedding_dim'':
    64, ''max_len'': 100, ''name'': ''clicked_items'', ''padding'': ''pre'', ''pretrained_emb'':
    ''../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5'', ''splitter'':
    ''^'', ''type'': ''sequence''}, {''active'': True, ''dtype'': ''str'', ''max_len'':
    100, ''name'': ''clicked_categories'', ''padding'': ''pre'', ''share_embedding'':
    ''cate_id'', ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': False,
    ''dtype'': ''str'', ''name'': ''timestamp'', ''type'': ''categorical''}]'
  feature_specs: '[{''feature_encoder'': ''nn.Linear(64, 64, bias=False)'', ''name'':
    ''item_id''}, {''feature_encoder'': [''layers.MaskedAveragePooling()'', ''nn.Linear(64,
    64, bias=False)''], ''name'': ''clicked_items''}, {''feature_encoder'': ''layers.MaskedAveragePooling()'',
    ''name'': ''clicked_categories''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''is_click''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv
  train_data: ../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv
  valid_data: ../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv
```

### MovielensLatest

- **ExpID**: `DCNv2_movielenslatest_x1_016_98ea1c72`
- **BARS Directory**: `BARS/ranking/ctr/DCNv2/DCNv2_movielenslatest_x1`
- **BARS AUC**: `0.968657`
- **BARS LogLoss**: `0.216136`

#### Model Config

```yaml
DCNv2_movielenslatest_x1_016_98ea1c72:
  batch_norm: 'True'
  batch_size: '4096'
  debug: 'False'
  dnn_activations: relu
  embedding_dim: '10'
  embedding_regularizer: '0.01'
  epochs: '100'
  every_x_epochs: '1'
  gpu: '0'
  learning_rate: '0.001'
  loss: binary_crossentropy
  low_rank: '32'
  metrics: '[''AUC'', ''logloss'']'
  model: DCNv2
  model_id: DCNv2_movielenslatest_x1_016_98ea1c72
  model_root: ./Movielens/DCN_movielenslatest_x1/
  model_structure: parallel
  monitor: AUC
  monitor_mode: max
  net_dropout: '0.2'
  net_regularizer: '0'
  num_cross_layers: '5'
  num_experts: '4'
  num_workers: '3'
  optimizer: adam
  parallel_dnn_hidden_units: '[400, 400, 400]'
  patience: '2'
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  stacked_dnn_hidden_units: '[500, 500, 500]'
  task: binary_classification
  use_low_rank_mixture: 'False'
  verbose: '1'
  version: pytorch
```

#### Dataset Config

```yaml
movielenslatest_x1_cd32d937:
  data_format: csv
  data_root: ../data/Movielens/
  dataset_id: movielenslatest_x1_cd32d937
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user_id'',
    ''item_id'', ''tag_id''], ''type'': ''categorical''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Movielens/MovielensLatest_x1/test.csv
  train_data: ../data/Movielens/MovielensLatest_x1/train.csv
  use_hdf5: 'True'
  valid_data: ../data/Movielens/MovielensLatest_x1/valid.csv
```

### TaobaoAd

- **ExpID**: `DCNv2_taobaoad_x1_026_55d3948a`
- **BARS Directory**: `BARS/ranking/ctr/DCNv2/DCNv2_taobaoad_x1`

#### Model Config

```yaml
DCNv2_taobaoad_x1_026_55d3948a:
  batch_norm: 'False'
  batch_size: '8192'
  debug_mode: 'False'
  dnn_activations: relu
  early_stop_patience: '1'
  embedding_dim: '32'
  embedding_regularizer: 5e-06
  epochs: '100'
  eval_interval: '1'
  gpu: '6'
  group_id: group_id
  learning_rate: '0.001'
  loss: binary_crossentropy
  low_rank: '32'
  metrics: '[''gAUC'', ''AUC'', ''logloss'']'
  model: DCNv2
  model_id: DCNv2_taobaoad_x1_026_55d3948a
  model_root: ./checkpoints/DCNv2_taobaoad_x1/
  model_structure: parallel
  monitor: '{''AUC'': 1, ''gAUC'': 1}'
  monitor_mode: max
  net_dropout: '0.1'
  net_regularizer: '0'
  num_cross_layers: '4'
  num_experts: '4'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  parallel_dnn_hidden_units: '[512, 256, 128]'
  save_best_only: 'True'
  seed: '20222023'
  shuffle: 'True'
  stacked_dnn_hidden_units: '[500, 500, 500]'
  task: binary_classification
  use_low_rank_mixture: 'False'
  verbose: '1'
```

#### Dataset Config

```yaml
taobaoad_x1_2753db8a:
  data_format: csv
  data_root: ../data/Taobao/
  dataset_id: taobaoad_x1_2753db8a
  feature_cols: '[{''active'': True, ''dtype'': ''int'', ''name'': ''group_id'', ''preprocess'':
    ''copy_from(userid)'', ''remap'': False, ''type'': ''meta''}, {''active'': True,
    ''dtype'': ''str'', ''name'': [''userid'', ''cms_segid'', ''cms_group_id'', ''final_gender_code'',
    ''age_level'', ''pvalue_level'', ''shopping_level'', ''occupation'', ''new_user_class_level'',
    ''adgroup_id'', ''cate_id'', ''campaign_id'', ''customer'', ''brand'', ''pid'',
    ''btag''], ''type'': ''categorical''}, {''active'': True, ''dtype'': ''float'',
    ''name'': ''price'', ''type'': ''numeric''}, {''active'': True, ''dtype'': ''str'',
    ''max_len'': 50, ''name'': ''cate_his'', ''padding'': ''pre'', ''share_embedding'':
    ''cate_id'', ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': True,
    ''dtype'': ''str'', ''max_len'': 50, ''name'': ''brand_his'', ''padding'': ''pre'',
    ''share_embedding'': ''brand'', ''splitter'': ''^'', ''type'': ''sequence''},
    {''active'': True, ''dtype'': ''str'', ''max_len'': 50, ''name'': ''btag_his'',
    ''padding'': ''pre'', ''share_embedding'': ''btag'', ''splitter'': ''^'', ''type'':
    ''sequence''}]'
  feature_specs: None
  label_col: '{''dtype'': ''float'', ''name'': ''clk''}'
  min_categr_count: '10'
  pickle_feature_encoder: 'True'
  test_data: ../data/Taobao/TaobaoAd_x1/test.csv
  train_data: ../data/Taobao/TaobaoAd_x1/train.csv
  valid_data: ../data/Taobao/TaobaoAd_x1/test.csv
```

---

## DIN

### KuaiVideo

- **ExpID**: `DIN_kuaivideo_x1_013_fc4bf206`
- **BARS Directory**: `BARS/ranking/ctr/DIN/DIN_kuaivideo_x1`

#### Model Config

```yaml
DIN_kuaivideo_x1_013_fc4bf206:
  attention_dropout: '0.2'
  attention_hidden_activations: ReLU
  attention_hidden_units: '[512, 256]'
  attention_output_activation: None
  batch_norm: 'False'
  batch_size: '8192'
  debug_mode: 'False'
  din_sequence_field: '[(''pos_items'', ''pos_items_emb''), (''neg_items'', ''neg_items_emb'')]'
  din_target_field: '[(''item_id'', ''item_emb''), (''item_id'', ''item_emb'')]'
  din_use_softmax: 'False'
  dnn_activations: Dice
  dnn_hidden_units: '[1024, 512, 256]'
  early_stop_patience: '2'
  embedding_dim: '64'
  embedding_regularizer: '0.0001'
  epochs: '100'
  eval_interval: '1'
  gpu: '6'
  group_id: group_id
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''gAUC'', ''AUC'', ''logloss'']'
  model: DIN
  model_id: DIN_kuaivideo_x1_013_fc4bf206
  model_root: ./checkpoints/DIN_kuaivideo_x1/
  monitor: '{''AUC'': 1, ''gAUC'': 1}'
  monitor_mode: max
  net_dropout: '0.1'
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  save_best_only: 'True'
  seed: '20222023'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
```

#### Dataset Config

```yaml
kuaivideo_x1_dc7a3035:
  data_format: csv
  data_root: ../data/KuaiShou/
  dataset_id: kuaivideo_x1_dc7a3035
  feature_cols: '[{''active'': True, ''dtype'': ''int'', ''name'': ''group_id'', ''preprocess'':
    ''copy_from(user_id)'', ''remap'': False, ''type'': ''meta''}, {''active'': True,
    ''dtype'': ''str'', ''name'': ''user_id'', ''type'': ''categorical''}, {''active'':
    True, ''dtype'': ''str'', ''name'': ''item_id'', ''type'': ''categorical''}, {''active'':
    True, ''dtype'': ''str'', ''embedding_dim'': 64, ''min_categr_count'': 1, ''name'':
    ''item_emb'', ''preprocess'': ''copy_from(item_id)'', ''pretrained_emb'': ''../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5'',
    ''type'': ''categorical''}, {''active'': True, ''dtype'': ''str'', ''max_len'':
    100, ''name'': ''pos_items'', ''padding'': ''pre'', ''share_embedding'': ''item_id'',
    ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': True, ''dtype'': ''str'',
    ''max_len'': 100, ''name'': ''neg_items'', ''padding'': ''pre'', ''share_embedding'':
    ''item_id'', ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': True,
    ''dtype'': ''str'', ''embedding_dim'': 64, ''max_len'': 100, ''min_categr_count'':
    1, ''name'': ''pos_items_emb'', ''padding'': ''pre'', ''preprocess'': ''copy_from(pos_items)'',
    ''pretrained_emb'': ''../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5'',
    ''share_embedding'': ''item_emb'', ''splitter'': ''^'', ''type'': ''sequence''},
    {''active'': True, ''dtype'': ''str'', ''embedding_dim'': 64, ''max_len'': 100,
    ''min_categr_count'': 1, ''name'': ''neg_items_emb'', ''padding'': ''pre'', ''preprocess'':
    ''copy_from(neg_items)'', ''pretrained_emb'': ''../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5'',
    ''share_embedding'': ''item_emb'', ''splitter'': ''^'', ''type'': ''sequence''}]'
  feature_specs: '[{''feature_encoder'': ''nn.Linear(64, 64, bias=False)'', ''name'':
    ''item_emb''}, {''feature_encoder'': None, ''name'': ''pos_items''}, {''feature_encoder'':
    None, ''name'': ''neg_items''}, {''feature_encoder'': [''nn.Linear(64, 64, bias=False)''],
    ''name'': ''pos_items_emb''}, {''feature_encoder'': [''nn.Linear(64, 64, bias=False)''],
    ''name'': ''neg_items_emb''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''is_click''}'
  min_categr_count: '10'
  pickle_feature_encoder: 'True'
  test_data: ../data/KuaiShou/KuaiVideo_x1/test.csv
  train_data: ../data/KuaiShou/KuaiVideo_x1/train.csv
  valid_data: ../data/KuaiShou/KuaiVideo_x1/test.csv
```

### MicroVideo1.7M

- **ExpID**: `DIN_microvideo1.7m_x1_006_ab4e3b7f`
- **BARS Directory**: `BARS/ranking/ctr/DIN/DIN_microvideo1.7m_x1`

#### Model Config

```yaml
DIN_microvideo1.7m_x1_006_ab4e3b7f:
  attention_dropout: '0.2'
  attention_hidden_activations: ReLU
  attention_hidden_units: '[512, 256]'
  attention_output_activation: None
  batch_norm: 'True'
  batch_size: '2048'
  debug_mode: 'False'
  din_sequence_field: ('clicked_items', 'clicked_categories')
  din_target_field: ('item_id', 'cate_id')
  din_use_softmax: 'True'
  dnn_activations: relu
  dnn_hidden_units: '[1024, 512, 256]'
  early_stop_patience: '2'
  embedding_dim: '64'
  embedding_regularizer: '0.001'
  epochs: '100'
  eval_interval: '1'
  gpu: '5'
  group_id: group_id
  learning_rate: '0.0005'
  loss: binary_crossentropy
  metrics: '[''gAUC'', ''AUC'', ''logloss'']'
  model: DIN
  model_id: DIN_microvideo1.7m_x1_006_ab4e3b7f
  model_root: ./checkpoints/DIN_microvideo1.7m_x1/
  monitor: '{''AUC'': 1, ''gAUC'': 1}'
  monitor_mode: max
  net_dropout: '0.2'
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  save_best_only: 'True'
  seed: '2022'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
```

#### Dataset Config

```yaml
microvideo1.7m_x1_0d855fe6:
  data_format: csv
  data_root: ../data/MicroVideo1.7M/
  dataset_id: microvideo1.7m_x1_0d855fe6
  feature_cols: '[{''active'': True, ''dtype'': ''int'', ''name'': ''group_id'', ''preprocess'':
    ''copy_from(user_id)'', ''remap'': False, ''type'': ''meta''}, {''active'': True,
    ''dtype'': ''str'', ''name'': ''user_id'', ''type'': ''categorical''}, {''active'':
    True, ''dtype'': ''str'', ''embedding_dim'': 64, ''name'': ''item_id'', ''pretrained_emb'':
    ''../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5'', ''type'':
    ''categorical''}, {''active'': True, ''dtype'': ''str'', ''name'': ''cate_id'',
    ''type'': ''categorical''}, {''active'': True, ''dtype'': ''str'', ''embedding_dim'':
    64, ''max_len'': 100, ''name'': ''clicked_items'', ''padding'': ''pre'', ''pretrained_emb'':
    ''../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5'', ''splitter'':
    ''^'', ''type'': ''sequence''}, {''active'': True, ''dtype'': ''str'', ''max_len'':
    100, ''name'': ''clicked_categories'', ''padding'': ''pre'', ''share_embedding'':
    ''cate_id'', ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': False,
    ''dtype'': ''str'', ''name'': ''timestamp'', ''type'': ''categorical''}]'
  feature_specs: '[{''feature_encoder'': ''nn.Linear(64, 64, bias=False)'', ''name'':
    ''item_id''}, {''feature_encoder'': ''nn.Linear(64, 64, bias=False)'', ''name'':
    ''clicked_items''}, {''feature_encoder'': None, ''name'': ''clicked_categories''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''is_click''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv
  train_data: ../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv
  valid_data: ../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv
```

### TaobaoAd

- **ExpID**: `DIN_taobaoad_x1_012_13c23a36`
- **BARS Directory**: `BARS/ranking/ctr/DIN/DIN_taobaoad_x1`

#### Model Config

```yaml
DIN_taobaoad_x1_012_13c23a36:
  attention_dropout: '0.4'
  attention_hidden_activations: ReLU
  attention_hidden_units: '[512, 256]'
  attention_output_activation: None
  batch_norm: 'False'
  batch_size: '8192'
  debug_mode: 'False'
  din_sequence_field: ('cate_his', 'brand_his', 'btag_his')
  din_target_field: ('cate_id', 'brand', 'btag')
  din_use_softmax: 'False'
  dnn_activations: relu
  dnn_hidden_units: '[512, 256, 128]'
  early_stop_patience: '1'
  embedding_dim: '32'
  embedding_regularizer: 5e-06
  epochs: '100'
  eval_interval: '1'
  gpu: '3'
  group_id: group_id
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''gAUC'', ''AUC'', ''logloss'']'
  model: DIN
  model_id: DIN_taobaoad_x1_012_13c23a36
  model_root: ./checkpoints/DIN_taobaoad_x1/
  monitor: '{''AUC'': 1, ''gAUC'': 1}'
  monitor_mode: max
  net_dropout: '0.1'
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  save_best_only: 'True'
  seed: '20222023'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
```

#### Dataset Config

```yaml
taobaoad_x1_2753db8a:
  data_format: csv
  data_root: ../data/Taobao/
  dataset_id: taobaoad_x1_2753db8a
  feature_cols: '[{''active'': True, ''dtype'': ''int'', ''name'': ''group_id'', ''preprocess'':
    ''copy_from(userid)'', ''remap'': False, ''type'': ''meta''}, {''active'': True,
    ''dtype'': ''str'', ''name'': [''userid'', ''cms_segid'', ''cms_group_id'', ''final_gender_code'',
    ''age_level'', ''pvalue_level'', ''shopping_level'', ''occupation'', ''new_user_class_level'',
    ''adgroup_id'', ''cate_id'', ''campaign_id'', ''customer'', ''brand'', ''pid'',
    ''btag''], ''type'': ''categorical''}, {''active'': True, ''dtype'': ''float'',
    ''name'': ''price'', ''type'': ''numeric''}, {''active'': True, ''dtype'': ''str'',
    ''max_len'': 50, ''name'': ''cate_his'', ''padding'': ''pre'', ''share_embedding'':
    ''cate_id'', ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': True,
    ''dtype'': ''str'', ''max_len'': 50, ''name'': ''brand_his'', ''padding'': ''pre'',
    ''share_embedding'': ''brand'', ''splitter'': ''^'', ''type'': ''sequence''},
    {''active'': True, ''dtype'': ''str'', ''max_len'': 50, ''name'': ''btag_his'',
    ''padding'': ''pre'', ''share_embedding'': ''btag'', ''splitter'': ''^'', ''type'':
    ''sequence''}]'
  feature_specs: '[{''feature_encoder'': None, ''name'': [''cate_his'', ''brand_his'',
    ''btag_his'']}]'
  label_col: '{''dtype'': ''float'', ''name'': ''clk''}'
  min_categr_count: '10'
  pickle_feature_encoder: 'True'
  test_data: ../data/Taobao/TaobaoAd_x1/test.csv
  train_data: ../data/Taobao/TaobaoAd_x1/train.csv
  valid_data: ../data/Taobao/TaobaoAd_x1/test.csv
```

---

## DLRM

### Frappe

- **ExpID**: `DLRM_frappe_x1_006_216831a3`
- **BARS Directory**: `BARS/ranking/ctr/DLRM/DLRM_frappe_x1`

#### Model Config

```yaml
DLRM_frappe_x1_006_216831a3:
  batch_norm: 'True'
  batch_size: '4096'
  bottom_mlp_activations: ReLU
  bottom_mlp_dropout: '0'
  bottom_mlp_units: None
  debug: 'False'
  embedding_dim: '10'
  embedding_regularizer: '0.1'
  epochs: '100'
  every_x_epochs: '1'
  gpu: '0'
  interaction_op: cat
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''AUC'', ''logloss'']'
  model: DLRM
  model_id: DLRM_frappe_x1_006_216831a3
  model_root: ./Frappe/DLRM_frappe_x1/
  monitor: AUC
  monitor_mode: max
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  patience: '2'
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  task: binary_classification
  top_mlp_activations: ReLU
  top_mlp_dropout: '0.4'
  top_mlp_units: '[400, 400, 400]'
  verbose: '1'
  version: pytorch
```

#### Dataset Config

```yaml
frappe_x1_04e961e9:
  data_format: csv
  data_root: ../data/Frappe/
  dataset_id: frappe_x1_04e961e9
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user'', ''item'',
    ''daytime'', ''weekday'', ''isweekend'', ''homework'', ''cost'', ''weather'',
    ''country'', ''city''], ''type'': ''categorical''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Frappe/Frappe_x1/test.csv
  train_data: ../data/Frappe/Frappe_x1/train.csv
  use_hdf5: 'True'
  valid_data: ../data/Frappe/Frappe_x1/valid.csv
```

### MovielensLatest

- **ExpID**: `DLRM_movielenslatest_x1_002_333e0a39`
- **BARS Directory**: `BARS/ranking/ctr/DLRM/DLRM_movielenslatest_x1`

#### Model Config

```yaml
DLRM_movielenslatest_x1_002_333e0a39:
  batch_norm: 'True'
  batch_size: '4096'
  bottom_mlp_activations: ReLU
  bottom_mlp_dropout: '0'
  bottom_mlp_units: None
  debug: 'False'
  embedding_dim: '10'
  embedding_regularizer: '0.01'
  epochs: '100'
  every_x_epochs: '1'
  gpu: '1'
  interaction_op: cat
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''AUC'', ''logloss'']'
  model: DLRM
  model_id: DLRM_movielenslatest_x1_002_333e0a39
  model_root: ./Movielens/DLRM_movielenslatest_x1/
  monitor: AUC
  monitor_mode: max
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  patience: '2'
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  task: binary_classification
  top_mlp_activations: ReLU
  top_mlp_dropout: '0.2'
  top_mlp_units: '[400, 400, 400]'
  verbose: '1'
  version: pytorch
```

#### Dataset Config

```yaml
movielenslatest_x1_cd32d937:
  data_format: csv
  data_root: ../data/Movielens/
  dataset_id: movielenslatest_x1_cd32d937
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user_id'',
    ''item_id'', ''tag_id''], ''type'': ''categorical''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Movielens/MovielensLatest_x1/test.csv
  train_data: ../data/Movielens/MovielensLatest_x1/train.csv
  use_hdf5: 'True'
  valid_data: ../data/Movielens/MovielensLatest_x1/valid.csv
```

---

## DeepFM

### Frappe

- **ExpID**: `DeepFM_frappe_x1_001_4ae3a56e`
- **BARS Directory**: `BARS/ranking/ctr/DeepFM/DeepFM_frappe_x1`
- **BARS AUC**: `0.983947`
- **BARS LogLoss**: `0.149462`

#### Model Config

```yaml
DeepFM_frappe_x1_001_4ae3a56e:
  batch_norm: 'True'
  batch_size: '4096'
  debug: 'False'
  embedding_dim: '10'
  embedding_regularizer: '0.05'
  epochs: '100'
  every_x_epochs: '1'
  gpu: '0'
  hidden_activations: relu
  hidden_units: '[400, 400, 400]'
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''AUC'', ''logloss'']'
  model: DeepFM
  model_id: DeepFM_frappe_x1_001_4ae3a56e
  model_root: ./Frappe/DeepFM_frappe_x1/
  monitor: AUC
  monitor_mode: max
  net_dropout: '0.2'
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  patience: '2'
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
  version: pytorch
```

#### Dataset Config

```yaml
frappe_x1_04e961e9:
  data_format: csv
  data_root: ../data/Frappe/
  dataset_id: frappe_x1_04e961e9
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user'', ''item'',
    ''daytime'', ''weekday'', ''isweekend'', ''homework'', ''cost'', ''weather'',
    ''country'', ''city''], ''type'': ''categorical''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Frappe/Frappe_x1/test.csv
  train_data: ../data/Frappe/Frappe_x1/train.csv
  use_hdf5: 'True'
  valid_data: ../data/Frappe/Frappe_x1/valid.csv
```

### KuaiVideo

- **ExpID**: `DeepFM_kuaivideo_x1_003_a7784cdb`
- **BARS Directory**: `BARS/ranking/ctr/DeepFM/DeepFM_kuaivideo_x1`

#### Model Config

```yaml
DeepFM_kuaivideo_x1_003_a7784cdb:
  batch_norm: 'False'
  batch_size: '8192'
  debug_mode: 'False'
  early_stop_patience: '2'
  embedding_dim: '64'
  embedding_regularizer: '0.0001'
  epochs: '100'
  eval_interval: '1'
  gpu: '2'
  group_id: group_id
  hidden_activations: relu
  hidden_units: '[1024, 512, 256]'
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''gAUC'', ''AUC'', ''logloss'']'
  model: DeepFM
  model_id: DeepFM_kuaivideo_x1_003_a7784cdb
  model_root: ./checkpoints/DeepFM_kuaivideo_x1/
  monitor: '{''AUC'': 1, ''gAUC'': 1}'
  monitor_mode: max
  net_dropout: '0.2'
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  save_best_only: 'True'
  seed: '20222023'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
```

#### Dataset Config

```yaml
kuaivideo_x1_dc7a3035:
  data_format: csv
  data_root: ../data/KuaiShou/
  dataset_id: kuaivideo_x1_dc7a3035
  feature_cols: '[{''active'': True, ''dtype'': ''int'', ''name'': ''group_id'', ''preprocess'':
    ''copy_from(user_id)'', ''remap'': False, ''type'': ''meta''}, {''active'': True,
    ''dtype'': ''str'', ''name'': ''user_id'', ''type'': ''categorical''}, {''active'':
    True, ''dtype'': ''str'', ''name'': ''item_id'', ''type'': ''categorical''}, {''active'':
    True, ''dtype'': ''str'', ''embedding_dim'': 64, ''min_categr_count'': 1, ''name'':
    ''item_emb'', ''preprocess'': ''copy_from(item_id)'', ''pretrained_emb'': ''../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5'',
    ''type'': ''categorical''}, {''active'': True, ''dtype'': ''str'', ''max_len'':
    100, ''name'': ''pos_items'', ''padding'': ''pre'', ''share_embedding'': ''item_id'',
    ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': True, ''dtype'': ''str'',
    ''max_len'': 100, ''name'': ''neg_items'', ''padding'': ''pre'', ''share_embedding'':
    ''item_id'', ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': True,
    ''dtype'': ''str'', ''embedding_dim'': 64, ''max_len'': 100, ''min_categr_count'':
    1, ''name'': ''pos_items_emb'', ''padding'': ''pre'', ''preprocess'': ''copy_from(pos_items)'',
    ''pretrained_emb'': ''../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5'',
    ''share_embedding'': ''item_emb'', ''splitter'': ''^'', ''type'': ''sequence''},
    {''active'': True, ''dtype'': ''str'', ''embedding_dim'': 64, ''max_len'': 100,
    ''min_categr_count'': 1, ''name'': ''neg_items_emb'', ''padding'': ''pre'', ''preprocess'':
    ''copy_from(neg_items)'', ''pretrained_emb'': ''../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5'',
    ''share_embedding'': ''item_emb'', ''splitter'': ''^'', ''type'': ''sequence''}]'
  feature_specs: '[{''feature_encoder'': ''nn.Linear(64, 64, bias=False)'', ''name'':
    ''item_emb''}, {''feature_encoder'': ''layers.MaskedAveragePooling()'', ''name'':
    ''pos_items''}, {''feature_encoder'': ''layers.MaskedAveragePooling()'', ''name'':
    ''neg_items''}, {''feature_encoder'': [''layers.MaskedAveragePooling()'', ''nn.Linear(64,
    64, bias=False)''], ''name'': ''pos_items_emb''}, {''feature_encoder'': [''layers.MaskedAveragePooling()'',
    ''nn.Linear(64, 64, bias=False)''], ''name'': ''neg_items_emb''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''is_click''}'
  min_categr_count: '10'
  pickle_feature_encoder: 'True'
  test_data: ../data/KuaiShou/KuaiVideo_x1/test.csv
  train_data: ../data/KuaiShou/KuaiVideo_x1/train.csv
  valid_data: ../data/KuaiShou/KuaiVideo_x1/test.csv
```

### MicroVideo1.7M

- **ExpID**: `DeepFM_microvideo1.7m_x1_023_bda67d29`
- **BARS Directory**: `BARS/ranking/ctr/DeepFM/DeepFM_microvideo1.7m_x1`

#### Model Config

```yaml
DeepFM_microvideo1.7m_x1_023_bda67d29:
  batch_norm: 'True'
  batch_size: '2048'
  debug_mode: 'False'
  early_stop_patience: '2'
  embedding_dim: '64'
  embedding_regularizer: '0.0005'
  epochs: '100'
  eval_interval: '1'
  gpu: '6'
  group_id: group_id
  hidden_activations: relu
  hidden_units: '[1024, 512, 256]'
  learning_rate: '0.0005'
  loss: binary_crossentropy
  metrics: '[''gAUC'', ''AUC'', ''logloss'']'
  model: DeepFM
  model_id: DeepFM_microvideo1.7m_x1_023_bda67d29
  model_root: ./checkpoints/DeepFM_microvideo1.7m_x1/
  monitor: '{''AUC'': 1, ''gAUC'': 1}'
  monitor_mode: max
  net_dropout: '0.3'
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  save_best_only: 'True'
  seed: '2022'
  shuffle: 'True'
  task: binary_classification
  verbose: '0'
```

#### Dataset Config

```yaml
microvideo1.7m_x1_0d855fe6:
  data_format: csv
  data_root: ../data/MicroVideo1.7M/
  dataset_id: microvideo1.7m_x1_0d855fe6
  feature_cols: '[{''active'': True, ''dtype'': ''int'', ''name'': ''group_id'', ''preprocess'':
    ''copy_from(user_id)'', ''remap'': False, ''type'': ''meta''}, {''active'': True,
    ''dtype'': ''str'', ''name'': ''user_id'', ''type'': ''categorical''}, {''active'':
    True, ''dtype'': ''str'', ''embedding_dim'': 64, ''name'': ''item_id'', ''pretrained_emb'':
    ''../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5'', ''type'':
    ''categorical''}, {''active'': True, ''dtype'': ''str'', ''name'': ''cate_id'',
    ''type'': ''categorical''}, {''active'': True, ''dtype'': ''str'', ''embedding_dim'':
    64, ''max_len'': 100, ''name'': ''clicked_items'', ''padding'': ''pre'', ''pretrained_emb'':
    ''../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5'', ''splitter'':
    ''^'', ''type'': ''sequence''}, {''active'': True, ''dtype'': ''str'', ''max_len'':
    100, ''name'': ''clicked_categories'', ''padding'': ''pre'', ''share_embedding'':
    ''cate_id'', ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': False,
    ''dtype'': ''str'', ''name'': ''timestamp'', ''type'': ''categorical''}]'
  feature_specs: '[{''feature_encoder'': ''nn.Linear(64, 64, bias=False)'', ''name'':
    ''item_id''}, {''feature_encoder'': [''layers.MaskedAveragePooling()'', ''nn.Linear(64,
    64, bias=False)''], ''name'': ''clicked_items''}, {''feature_encoder'': ''layers.MaskedAveragePooling()'',
    ''name'': ''clicked_categories''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''is_click''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv
  train_data: ../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv
  valid_data: ../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv
```

### MovielensLatest

- **ExpID**: `DeepFM_movielenslatest_x1_005_0f6d2e8e`
- **BARS Directory**: `BARS/ranking/ctr/DeepFM/DeepFM_movielenslatest_x1`
- **BARS AUC**: `0.968223`
- **BARS LogLoss**: `0.213790`

#### Model Config

```yaml
DeepFM_movielenslatest_x1_005_0f6d2e8e:
  batch_norm: 'True'
  batch_size: '4096'
  debug: 'False'
  embedding_dim: '10'
  embedding_regularizer: '0.01'
  epochs: '100'
  every_x_epochs: '1'
  gpu: '0'
  hidden_activations: relu
  hidden_units: '[400, 400, 400]'
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''AUC'', ''logloss'']'
  model: DeepFM
  model_id: DeepFM_movielenslatest_x1_005_0f6d2e8e
  model_root: ./Movielens/DeepFM_movielenslatest_x1/
  monitor: AUC
  monitor_mode: max
  net_dropout: '0.3'
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  patience: '2'
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
  version: pytorch
```

#### Dataset Config

```yaml
movielenslatest_x1_cd32d937:
  data_format: csv
  data_root: ../data/Movielens/
  dataset_id: movielenslatest_x1_cd32d937
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user_id'',
    ''item_id'', ''tag_id''], ''type'': ''categorical''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Movielens/MovielensLatest_x1/test.csv
  train_data: ../data/Movielens/MovielensLatest_x1/train.csv
  use_hdf5: 'True'
  valid_data: ../data/Movielens/MovielensLatest_x1/valid.csv
```

### TaobaoAd

- **ExpID**: `DeepFM_taobaoad_x1_009_afab5940`
- **BARS Directory**: `BARS/ranking/ctr/DeepFM/DeepFM_taobaoad_x1`

#### Model Config

```yaml
DeepFM_taobaoad_x1_009_afab5940:
  batch_norm: 'False'
  batch_size: '8192'
  debug_mode: 'False'
  early_stop_patience: '1'
  embedding_dim: '32'
  embedding_regularizer: 5e-06
  epochs: '100'
  eval_interval: '1'
  gpu: '0'
  group_id: group_id
  hidden_activations: relu
  hidden_units: '[512, 256, 128]'
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''gAUC'', ''AUC'', ''logloss'']'
  model: DeepFM
  model_id: DeepFM_taobaoad_x1_009_afab5940
  model_root: ./checkpoints/DeepFM_taobaoad_x1/
  monitor: '{''AUC'': 1, ''gAUC'': 1}'
  monitor_mode: max
  net_dropout: '0'
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  save_best_only: 'True'
  seed: '20222023'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
```

#### Dataset Config

```yaml
taobaoad_x1_2753db8a:
  data_format: csv
  data_root: ../data/Taobao/
  dataset_id: taobaoad_x1_2753db8a
  feature_cols: '[{''active'': True, ''dtype'': ''int'', ''name'': ''group_id'', ''preprocess'':
    ''copy_from(userid)'', ''remap'': False, ''type'': ''meta''}, {''active'': True,
    ''dtype'': ''str'', ''name'': [''userid'', ''cms_segid'', ''cms_group_id'', ''final_gender_code'',
    ''age_level'', ''pvalue_level'', ''shopping_level'', ''occupation'', ''new_user_class_level'',
    ''adgroup_id'', ''cate_id'', ''campaign_id'', ''customer'', ''brand'', ''pid'',
    ''btag''], ''type'': ''categorical''}, {''active'': True, ''dtype'': ''float'',
    ''name'': ''price'', ''type'': ''numeric''}, {''active'': True, ''dtype'': ''str'',
    ''max_len'': 50, ''name'': ''cate_his'', ''padding'': ''pre'', ''share_embedding'':
    ''cate_id'', ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': True,
    ''dtype'': ''str'', ''max_len'': 50, ''name'': ''brand_his'', ''padding'': ''pre'',
    ''share_embedding'': ''brand'', ''splitter'': ''^'', ''type'': ''sequence''},
    {''active'': True, ''dtype'': ''str'', ''max_len'': 50, ''name'': ''btag_his'',
    ''padding'': ''pre'', ''share_embedding'': ''btag'', ''splitter'': ''^'', ''type'':
    ''sequence''}]'
  feature_specs: None
  label_col: '{''dtype'': ''float'', ''name'': ''clk''}'
  min_categr_count: '10'
  pickle_feature_encoder: 'True'
  test_data: ../data/Taobao/TaobaoAd_x1/test.csv
  train_data: ../data/Taobao/TaobaoAd_x1/train.csv
  valid_data: ../data/Taobao/TaobaoAd_x1/test.csv
```

---

## FinalMLP

### Frappe

- **ExpID**: `FinalMLP_frappe_x1_004_e1ab402f`
- **BARS Directory**: `BARS/ranking/ctr/FinalMLP/FinalMLP_frappe_x1`

#### Model Config

```yaml
FinalMLP_frappe_x1_004_e1ab402f:
  batch_size: '4096'
  debug_mode: 'False'
  early_stop_patience: '2'
  embedding_dim: '10'
  embedding_regularizer: '0.05'
  epochs: '100'
  eval_interval: '1'
  fs1_context: '[''user'']'
  fs2_context: '[''item'']'
  fs_hidden_units: '[400]'
  gpu: '1'
  group_id: None
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''AUC'', ''logloss'']'
  mlp1_batch_norm: 'True'
  mlp1_dropout: '0.4'
  mlp1_hidden_activations: relu
  mlp1_hidden_units: '[400]'
  mlp2_batch_norm: 'True'
  mlp2_dropout: '0.4'
  mlp2_hidden_activations: relu
  mlp2_hidden_units: '[100]'
  model: FinalMLP
  model_id: FinalMLP_frappe_x1_004_e1ab402f
  model_root: ./checkpoints/FinalMLP_frappe_x1/
  monitor: AUC
  monitor_mode: max
  net_regularizer: '0'
  num_heads: '5'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  task: binary_classification
  use_fs: 'True'
  verbose: '1'
```

#### Dataset Config

```yaml
frappe_x1_47e6e0df:
  data_format: csv
  data_root: ../data/Frappe/
  dataset_id: frappe_x1_47e6e0df
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user'', ''item'',
    ''daytime'', ''weekday'', ''isweekend'', ''homework'', ''cost'', ''weather'',
    ''country'', ''city''], ''type'': ''categorical''}]'
  feature_specs: None
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Frappe/Frappe_x1/test.csv
  train_data: ../data/Frappe/Frappe_x1/train.csv
  valid_data: ../data/Frappe/Frappe_x1/valid.csv
```

### MovielensLatest

- **ExpID**: `FinalMLP_movielenslatest_x1_004_498f3e4f`
- **BARS Directory**: `BARS/ranking/ctr/FinalMLP/FinalMLP_movielenslatest_x1`

#### Model Config

```yaml
FinalMLP_movielenslatest_x1_004_498f3e4f:
  batch_size: '4096'
  debug_mode: 'False'
  early_stop_patience: '2'
  embedding_dim: '10'
  embedding_regularizer: '0.01'
  epochs: '100'
  eval_interval: '1'
  fs1_context: '[]'
  fs2_context: '[]'
  fs_hidden_units: '[800]'
  gpu: '1'
  group_id: None
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''AUC'', ''logloss'']'
  mlp1_batch_norm: 'True'
  mlp1_dropout: '0.4'
  mlp1_hidden_activations: relu
  mlp1_hidden_units: '[400]'
  mlp2_batch_norm: 'True'
  mlp2_dropout: '0.2'
  mlp2_hidden_activations: relu
  mlp2_hidden_units: '[800]'
  model: FinalMLP
  model_id: FinalMLP_movielenslatest_x1_004_498f3e4f
  model_root: ./checkpoints/FinalMLP_movielenslatest_x1/
  monitor: AUC
  monitor_mode: max
  net_regularizer: '0'
  num_heads: '10'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  task: binary_classification
  use_fs: 'True'
  verbose: '1'
```

#### Dataset Config

```yaml
movielenslatest_x1_233328b6:
  data_format: csv
  data_root: ../data/Movielens/
  dataset_id: movielenslatest_x1_233328b6
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user_id'',
    ''item_id'', ''tag_id''], ''type'': ''categorical''}]'
  feature_specs: None
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Movielens/MovielensLatest_x1/test.csv
  train_data: ../data/Movielens/MovielensLatest_x1/train.csv
  valid_data: ../data/Movielens/MovielensLatest_x1/valid.csv
```

---

## MaskNet

### Frappe

- **ExpID**: `MaskNet_frappe_x1_028_015da53e`
- **BARS Directory**: `BARS/ranking/ctr/MaskNet/MaskNet_frappe_x1`

#### Model Config

```yaml
MaskNet_frappe_x1_028_015da53e:
  batch_size: '4096'
  debug: 'False'
  dnn_hidden_activations: relu
  dnn_hidden_units: '[400, 400, 400]'
  emb_layernorm: 'False'
  embedding_dim: '10'
  embedding_regularizer: '0.1'
  epochs: '100'
  every_x_epochs: '1'
  gpu: '1'
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''AUC'', ''logloss'']'
  model: MaskNet
  model_id: MaskNet_frappe_x1_028_015da53e
  model_root: ./Frappe/MaskNet_frappe_x1/
  model_type: ParallelMaskNet
  monitor: AUC
  monitor_mode: max
  net_dropout: '0.1'
  net_layernorm: 'True'
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  parallel_block_dim: '50'
  parallel_num_blocks: '3'
  patience: '2'
  reduction_ratio: '2'
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
  version: pytorch
```

#### Dataset Config

```yaml
frappe_x1_04e961e9:
  data_format: csv
  data_root: ../data/Frappe/
  dataset_id: frappe_x1_04e961e9
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user'', ''item'',
    ''daytime'', ''weekday'', ''isweekend'', ''homework'', ''cost'', ''weather'',
    ''country'', ''city''], ''type'': ''categorical''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Frappe/Frappe_x1/test.csv
  train_data: ../data/Frappe/Frappe_x1/train.csv
  use_hdf5: 'True'
  valid_data: ../data/Frappe/Frappe_x1/valid.csv
```

### MovielensLatest

- **ExpID**: `MaskNet_movielenslatest_x1_010_13a8d29c`
- **BARS Directory**: `BARS/ranking/ctr/MaskNet/MaskNet_movielenslatest_x1`

#### Model Config

```yaml
MaskNet_movielenslatest_x1_010_13a8d29c:
  batch_size: '4096'
  debug: 'False'
  dnn_hidden_activations: relu
  dnn_hidden_units: '[400, 400, 400]'
  emb_layernorm: 'False'
  embedding_dim: '10'
  embedding_regularizer: '0.005'
  epochs: '100'
  every_x_epochs: '1'
  gpu: '1'
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''AUC'', ''logloss'']'
  model: MaskNet
  model_id: MaskNet_movielenslatest_x1_010_13a8d29c
  model_root: ./Movielens/MaskNet_movielenslatest_x1/
  model_type: SerialMaskNet
  monitor: AUC
  monitor_mode: max
  net_dropout: '0.5'
  net_layernorm: 'True'
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  parallel_block_dim: '64'
  parallel_num_blocks: '1'
  patience: '2'
  reduction_ratio: '4'
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
  version: pytorch
```

#### Dataset Config

```yaml
movielenslatest_x1_cd32d937:
  data_format: csv
  data_root: ../data/Movielens/
  dataset_id: movielenslatest_x1_cd32d937
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user_id'',
    ''item_id'', ''tag_id''], ''type'': ''categorical''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Movielens/MovielensLatest_x1/test.csv
  train_data: ../data/Movielens/MovielensLatest_x1/train.csv
  use_hdf5: 'True'
  valid_data: ../data/Movielens/MovielensLatest_x1/valid.csv
```

---

## xDeepFM

### Frappe

- **ExpID**: `xDeepFM_frappe_x1_005_447fa536`
- **BARS Directory**: `BARS/ranking/ctr/xDeepFM/xDeepFM_frappe_x1`
- **BARS AUC**: `0.984138`
- **BARS LogLoss**: `0.146125`

#### Model Config

```yaml
xDeepFM_frappe_x1_005_447fa536:
  batch_norm: 'True'
  batch_size: '4096'
  cin_layer_units: '[64]'
  debug: 'False'
  dnn_hidden_units: '[400, 400, 400]'
  embedding_dim: '10'
  embedding_regularizer: '0.1'
  epochs: '100'
  every_x_epochs: '1'
  gpu: '0'
  hidden_activations: relu
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''AUC'', ''logloss'']'
  model: xDeepFM
  model_id: xDeepFM_frappe_x1_005_447fa536
  model_root: ./Frappe/xDeepFM_frappe_x1/
  monitor: AUC
  monitor_mode: max
  net_dropout: '0.2'
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  partition_block_size: '-1'
  patience: '2'
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
  version: pytorch
```

#### Dataset Config

```yaml
frappe_x1_04e961e9:
  data_format: csv
  data_root: ../data/Frappe/
  dataset_id: frappe_x1_04e961e9
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user'', ''item'',
    ''daytime'', ''weekday'', ''isweekend'', ''homework'', ''cost'', ''weather'',
    ''country'', ''city''], ''type'': ''categorical''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Frappe/Frappe_x1/test.csv
  train_data: ../data/Frappe/Frappe_x1/train.csv
  use_hdf5: 'True'
  valid_data: ../data/Frappe/Frappe_x1/valid.csv
```

### KuaiVideo

- **ExpID**: `xDeepFM_kuaivideo_x1_016_0372eae9`
- **BARS Directory**: `BARS/ranking/ctr/xDeepFM/xDeepFM_kuaivideo_x1`

#### Model Config

```yaml
xDeepFM_kuaivideo_x1_016_0372eae9:
  batch_norm: 'True'
  batch_size: '8192'
  cin_hidden_units: '[64, 64]'
  debug_mode: 'False'
  dnn_hidden_units: '[1024, 512, 256]'
  early_stop_patience: '2'
  embedding_dim: '64'
  embedding_regularizer: '0.0005'
  epochs: '100'
  eval_interval: '1'
  gpu: '5'
  group_id: group_id
  hidden_activations: relu
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''gAUC'', ''AUC'', ''logloss'']'
  model: xDeepFM
  model_id: xDeepFM_kuaivideo_x1_016_0372eae9
  model_root: ./checkpoints/xDeepFM_kuaivideo_x1/
  monitor: '{''AUC'': 1, ''gAUC'': 1}'
  monitor_mode: max
  net_dropout: '0.1'
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  save_best_only: 'True'
  seed: '20222023'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
```

#### Dataset Config

```yaml
kuaivideo_x1_dc7a3035:
  data_format: csv
  data_root: ../data/KuaiShou/
  dataset_id: kuaivideo_x1_dc7a3035
  feature_cols: '[{''active'': True, ''dtype'': ''int'', ''name'': ''group_id'', ''preprocess'':
    ''copy_from(user_id)'', ''remap'': False, ''type'': ''meta''}, {''active'': True,
    ''dtype'': ''str'', ''name'': ''user_id'', ''type'': ''categorical''}, {''active'':
    True, ''dtype'': ''str'', ''name'': ''item_id'', ''type'': ''categorical''}, {''active'':
    True, ''dtype'': ''str'', ''embedding_dim'': 64, ''min_categr_count'': 1, ''name'':
    ''item_emb'', ''preprocess'': ''copy_from(item_id)'', ''pretrained_emb'': ''../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5'',
    ''type'': ''categorical''}, {''active'': True, ''dtype'': ''str'', ''max_len'':
    100, ''name'': ''pos_items'', ''padding'': ''pre'', ''share_embedding'': ''item_id'',
    ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': True, ''dtype'': ''str'',
    ''max_len'': 100, ''name'': ''neg_items'', ''padding'': ''pre'', ''share_embedding'':
    ''item_id'', ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': True,
    ''dtype'': ''str'', ''embedding_dim'': 64, ''max_len'': 100, ''min_categr_count'':
    1, ''name'': ''pos_items_emb'', ''padding'': ''pre'', ''preprocess'': ''copy_from(pos_items)'',
    ''pretrained_emb'': ''../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5'',
    ''share_embedding'': ''item_emb'', ''splitter'': ''^'', ''type'': ''sequence''},
    {''active'': True, ''dtype'': ''str'', ''embedding_dim'': 64, ''max_len'': 100,
    ''min_categr_count'': 1, ''name'': ''neg_items_emb'', ''padding'': ''pre'', ''preprocess'':
    ''copy_from(neg_items)'', ''pretrained_emb'': ''../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5'',
    ''share_embedding'': ''item_emb'', ''splitter'': ''^'', ''type'': ''sequence''}]'
  feature_specs: '[{''feature_encoder'': ''nn.Linear(64, 64, bias=False)'', ''name'':
    ''item_emb''}, {''feature_encoder'': ''layers.MaskedAveragePooling()'', ''name'':
    ''pos_items''}, {''feature_encoder'': ''layers.MaskedAveragePooling()'', ''name'':
    ''neg_items''}, {''feature_encoder'': [''layers.MaskedAveragePooling()'', ''nn.Linear(64,
    64, bias=False)''], ''name'': ''pos_items_emb''}, {''feature_encoder'': [''layers.MaskedAveragePooling()'',
    ''nn.Linear(64, 64, bias=False)''], ''name'': ''neg_items_emb''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''is_click''}'
  min_categr_count: '10'
  pickle_feature_encoder: 'True'
  test_data: ../data/KuaiShou/KuaiVideo_x1/test.csv
  train_data: ../data/KuaiShou/KuaiVideo_x1/train.csv
  valid_data: ../data/KuaiShou/KuaiVideo_x1/test.csv
```

### MicroVideo1.7M

- **ExpID**: `xDeepFM_microvideo1.7m_x1_011_7df31553`
- **BARS Directory**: `BARS/ranking/ctr/xDeepFM/xDeepFM_microvideo1.7m_x1`

#### Model Config

```yaml
xDeepFM_microvideo1.7m_x1_011_7df31553:
  batch_norm: 'True'
  batch_size: '2048'
  cin_hidden_units: '[32]'
  debug_mode: 'False'
  dnn_hidden_units: '[1024, 512, 256]'
  early_stop_patience: '2'
  embedding_dim: '64'
  embedding_regularizer: '0.001'
  epochs: '100'
  eval_interval: '1'
  gpu: '2'
  group_id: group_id
  hidden_activations: relu
  learning_rate: '0.0005'
  loss: binary_crossentropy
  metrics: '[''gAUC'', ''AUC'', ''logloss'']'
  model: xDeepFM
  model_id: xDeepFM_microvideo1.7m_x1_011_7df31553
  model_root: ./checkpoints/xDeepFM_microvideo1.7m_x1/
  monitor: '{''AUC'': 1, ''gAUC'': 1}'
  monitor_mode: max
  net_dropout: '0.2'
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  save_best_only: 'True'
  seed: '2022'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
```

#### Dataset Config

```yaml
microvideo1.7m_x1_0d855fe6:
  data_format: csv
  data_root: ../data/MicroVideo1.7M/
  dataset_id: microvideo1.7m_x1_0d855fe6
  feature_cols: '[{''active'': True, ''dtype'': ''int'', ''name'': ''group_id'', ''preprocess'':
    ''copy_from(user_id)'', ''remap'': False, ''type'': ''meta''}, {''active'': True,
    ''dtype'': ''str'', ''name'': ''user_id'', ''type'': ''categorical''}, {''active'':
    True, ''dtype'': ''str'', ''embedding_dim'': 64, ''name'': ''item_id'', ''pretrained_emb'':
    ''../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5'', ''type'':
    ''categorical''}, {''active'': True, ''dtype'': ''str'', ''name'': ''cate_id'',
    ''type'': ''categorical''}, {''active'': True, ''dtype'': ''str'', ''embedding_dim'':
    64, ''max_len'': 100, ''name'': ''clicked_items'', ''padding'': ''pre'', ''pretrained_emb'':
    ''../data/MicroVideo1.7M/MicroVideo1.7M_x1/item_image_emb_dim64.h5'', ''splitter'':
    ''^'', ''type'': ''sequence''}, {''active'': True, ''dtype'': ''str'', ''max_len'':
    100, ''name'': ''clicked_categories'', ''padding'': ''pre'', ''share_embedding'':
    ''cate_id'', ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': False,
    ''dtype'': ''str'', ''name'': ''timestamp'', ''type'': ''categorical''}]'
  feature_specs: '[{''feature_encoder'': ''nn.Linear(64, 64, bias=False)'', ''name'':
    ''item_id''}, {''feature_encoder'': [''layers.MaskedAveragePooling()'', ''nn.Linear(64,
    64, bias=False)''], ''name'': ''clicked_items''}, {''feature_encoder'': ''layers.MaskedAveragePooling()'',
    ''name'': ''clicked_categories''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''is_click''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv
  train_data: ../data/MicroVideo1.7M/MicroVideo1.7M_x1/train.csv
  valid_data: ../data/MicroVideo1.7M/MicroVideo1.7M_x1/test.csv
```

### MovielensLatest

- **ExpID**: `xDeepFM_movielenslatest_x1_018_57ed221b`
- **BARS Directory**: `BARS/ranking/ctr/xDeepFM/xDeepFM_movielenslatest_x1`
- **BARS AUC**: `0.968098`
- **BARS LogLoss**: `0.230108`

#### Model Config

```yaml
xDeepFM_movielenslatest_x1_018_57ed221b:
  batch_norm: 'False'
  batch_size: '4096'
  cin_layer_units: '[64, 64, 64]'
  debug: 'False'
  dnn_hidden_units: '[400, 400, 400]'
  embedding_dim: '10'
  embedding_regularizer: '0.005'
  epochs: '100'
  every_x_epochs: '1'
  gpu: '0'
  hidden_activations: relu
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''AUC'', ''logloss'']'
  model: xDeepFM
  model_id: xDeepFM_movielenslatest_x1_018_57ed221b
  model_root: ./Movielens/xDeepFM_movielenslatest_x1/
  monitor: AUC
  monitor_mode: max
  net_dropout: '0.3'
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  partition_block_size: '-1'
  patience: '2'
  save_best_only: 'True'
  seed: '2021'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
  version: pytorch
```

#### Dataset Config

```yaml
movielenslatest_x1_cd32d937:
  data_format: csv
  data_root: ../data/Movielens/
  dataset_id: movielenslatest_x1_cd32d937
  feature_cols: '[{''active'': True, ''dtype'': ''float'', ''name'': [''user_id'',
    ''item_id'', ''tag_id''], ''type'': ''categorical''}]'
  label_col: '{''dtype'': ''float'', ''name'': ''label''}'
  min_categr_count: '1'
  pickle_feature_encoder: 'True'
  test_data: ../data/Movielens/MovielensLatest_x1/test.csv
  train_data: ../data/Movielens/MovielensLatest_x1/train.csv
  use_hdf5: 'True'
  valid_data: ../data/Movielens/MovielensLatest_x1/valid.csv
```

### TaobaoAd

- **ExpID**: `xDeepFM_taobaoad_x1_002_3f7806c5`
- **BARS Directory**: `BARS/ranking/ctr/xDeepFM/xDeepFM_taobaoad_x1`

#### Model Config

```yaml
xDeepFM_taobaoad_x1_002_3f7806c5:
  batch_norm: 'False'
  batch_size: '8192'
  cin_hidden_units: '[16]'
  debug_mode: 'False'
  dnn_hidden_units: '[512, 256, 128]'
  early_stop_patience: '1'
  embedding_dim: '32'
  embedding_regularizer: 5e-06
  epochs: '100'
  eval_interval: '1'
  gpu: '2'
  group_id: group_id
  hidden_activations: relu
  learning_rate: '0.001'
  loss: binary_crossentropy
  metrics: '[''gAUC'', ''AUC'', ''logloss'']'
  model: xDeepFM
  model_id: xDeepFM_taobaoad_x1_002_3f7806c5
  model_root: ./checkpoints/xDeepFM_taobaoad_x1/
  monitor: '{''AUC'': 1, ''gAUC'': 1}'
  monitor_mode: max
  net_dropout: '0.1'
  net_regularizer: '0'
  num_workers: '3'
  optimizer: adam
  ordered_features: None
  save_best_only: 'True'
  seed: '20222023'
  shuffle: 'True'
  task: binary_classification
  verbose: '1'
```

#### Dataset Config

```yaml
taobaoad_x1_2753db8a:
  data_format: csv
  data_root: ../data/Taobao/
  dataset_id: taobaoad_x1_2753db8a
  feature_cols: '[{''active'': True, ''dtype'': ''int'', ''name'': ''group_id'', ''preprocess'':
    ''copy_from(userid)'', ''remap'': False, ''type'': ''meta''}, {''active'': True,
    ''dtype'': ''str'', ''name'': [''userid'', ''cms_segid'', ''cms_group_id'', ''final_gender_code'',
    ''age_level'', ''pvalue_level'', ''shopping_level'', ''occupation'', ''new_user_class_level'',
    ''adgroup_id'', ''cate_id'', ''campaign_id'', ''customer'', ''brand'', ''pid'',
    ''btag''], ''type'': ''categorical''}, {''active'': True, ''dtype'': ''float'',
    ''name'': ''price'', ''type'': ''numeric''}, {''active'': True, ''dtype'': ''str'',
    ''max_len'': 50, ''name'': ''cate_his'', ''padding'': ''pre'', ''share_embedding'':
    ''cate_id'', ''splitter'': ''^'', ''type'': ''sequence''}, {''active'': True,
    ''dtype'': ''str'', ''max_len'': 50, ''name'': ''brand_his'', ''padding'': ''pre'',
    ''share_embedding'': ''brand'', ''splitter'': ''^'', ''type'': ''sequence''},
    {''active'': True, ''dtype'': ''str'', ''max_len'': 50, ''name'': ''btag_his'',
    ''padding'': ''pre'', ''share_embedding'': ''btag'', ''splitter'': ''^'', ''type'':
    ''sequence''}]'
  feature_specs: None
  label_col: '{''dtype'': ''float'', ''name'': ''clk''}'
  min_categr_count: '10'
  pickle_feature_encoder: 'True'
  test_data: ../data/Taobao/TaobaoAd_x1/test.csv
  train_data: ../data/Taobao/TaobaoAd_x1/train.csv
  valid_data: ../data/Taobao/TaobaoAd_x1/test.csv
```

---

## Missing Configs

The following model-dataset combinations are not available in the current BARS repository:

- AFN/MicroVideo1.7M
- AFN/KuaiVideo
- AFN/TaobaoAd
- DLRM/MicroVideo1.7M
- DLRM/KuaiVideo
- DLRM/TaobaoAd
- DIN/Frappe
- DIN/MovielensLatest
- FinalMLP/MicroVideo1.7M
- FinalMLP/KuaiVideo
- FinalMLP/TaobaoAd
- MaskNet/MicroVideo1.7M
- MaskNet/KuaiVideo
- MaskNet/TaobaoAd

