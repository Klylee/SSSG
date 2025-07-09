python train.py -s ~/expdata/neuralto/yuanbao -m output_neuralto/yuanbao/test --quiet -r2 --ncc_scale 0.5

python train.py -s ~/expdata/neuralto/chinesedragon -m output_neuralto/chinesedragon/test --quiet -r2 --ncc_scale 0.5

python render.py -m output_neuralto/chinesedragon/test --max_depth 10.0 --voxel_size 0.01

python scripts/dtu_eval.py

CUDA_VISIBLE_DEVICES=1