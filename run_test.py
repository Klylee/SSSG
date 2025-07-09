import os

gpu = 1
scene = "nail"
# model_dir = f"output_neuralto/{scene}/test"
# model_dir = f"results/wo_fresnel/{scene}/test"
# model_dir = f"results/wo_pbr/{scene}/test"
# model_dir = "./results/pgsr"
# model_dir = f"debug/best/{scene}/test"
# model_dir = "./results/neuralto"
model_dir = f'results/2dgs'
# recon_f = model_dir + f'/mesh/tsdf_fusion_post-{scene}.obj'
recon_f = model_dir + f'/mesh/fuse_post-{scene}.ply'
# recon_f = model_dir + f'/mesh/00100000-{scene}.ply'

# cmd = f"python render.py -m {model_dir} --max_depth 10.0 --voxel_size 0.001"
# os.system(cmd)
cmd = f"python scripts/dtu_eval.py --s {scene} --recon_f {recon_f} --mc"
os.system(cmd)