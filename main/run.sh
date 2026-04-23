dataset="frappe_x1"

python main/run_expid.py --model RankMixer --gpu 0 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model FAT --gpu 1 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model HiFormer --gpu 2 --dataset $dataset --optimizer adamw --early_stop_patience 2&
python main/run_expid.py --model TransformerCTR --gpu 3 --dataset $dataset --optimizer adamw --early_stop_patience 2&

wait

python main/run_expid.py --model Wukong --gpu 0 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model HeteroAttention --gpu 1 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model TokenMixer_Large --gpu 2 --dataset $dataset --optimizer adamw --early_stop_patience 2&
python main/run_expid.py --model UniMixer_lite --gpu 3 --dataset $dataset --optimizer adamw --early_stop_patience 2&

wait

dataset="movielenslatest_x1"

python main/run_expid.py --model RankMixer --gpu 0 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model FAT --gpu 1 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model HiFormer --gpu 2 --dataset $dataset --optimizer adamw --early_stop_patience 2&
python main/run_expid.py --model TransformerCTR --gpu 3 --dataset $dataset --optimizer adamw --early_stop_patience 2&

wait

python main/run_expid.py --model Wukong --gpu 0 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model HeteroAttention --gpu 1 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model TokenMixer_Large --gpu 2 --dataset $dataset --optimizer adamw --early_stop_patience 2&
python main/run_expid.py --model UniMixer_lite --gpu 3 --dataset $dataset --optimizer adamw --early_stop_patience 2&

wait

dataset="taobaoad_x1"

python main/run_expid.py --model RankMixer --gpu 0 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model FAT --gpu 1 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model HiFormer --gpu 2 --dataset $dataset --optimizer adamw --early_stop_patience 2&
python main/run_expid.py --model TransformerCTR --gpu 3 --dataset $dataset --optimizer adamw --early_stop_patience 2&

wait

python main/run_expid.py --model Wukong --gpu 0 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model HeteroAttention --gpu 1 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model TokenMixer_Large --gpu 2 --dataset $dataset --optimizer adamw --early_stop_patience 2&
python main/run_expid.py --model UniMixer_lite --gpu 3 --dataset $dataset --optimizer adamw --early_stop_patience 2&

wait

dataset="microvideo1.7m_x1"

python main/run_expid.py --model RankMixer --gpu 0 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model FAT --gpu 1 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model HiFormer --gpu 2 --dataset $dataset --optimizer adamw --early_stop_patience 2&
python main/run_expid.py --model TransformerCTR --gpu 3 --dataset $dataset --optimizer adamw --early_stop_patience 2&

wait

python main/run_expid.py --model Wukong --gpu 0 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model HeteroAttention --gpu 1 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model TokenMixer_Large --gpu 2 --dataset $dataset --optimizer adamw --early_stop_patience 2&
python main/run_expid.py --model UniMixer_lite --gpu 3 --dataset $dataset --optimizer adamw --early_stop_patience 2&

wait

dataset="kuaivideo_x1"

python main/run_expid.py --model RankMixer --gpu 0 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model FAT --gpu 1 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model HiFormer --gpu 2 --dataset $dataset --optimizer adamw --early_stop_patience 2&
python main/run_expid.py --model TransformerCTR --gpu 3 --dataset $dataset --optimizer adamw --early_stop_patience 2&

wait

python main/run_expid.py --model Wukong --gpu 0 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model HeteroAttention --gpu 1 --dataset $dataset --optimizer adamw --early_stop_patience 2& 
python main/run_expid.py --model TokenMixer_Large --gpu 2 --dataset $dataset --optimizer adamw --early_stop_patience 2&
python main/run_expid.py --model UniMixer_lite --gpu 3 --dataset $dataset --optimizer adamw --early_stop_patience 2&

wait
