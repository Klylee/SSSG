import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-s", type=str, default="chinesedragon")
args = parser.parse_args()

print(f"\nExtracting Mesh\n")
os.system(f"python render.py -m output_neuralto/{args.s}/test --max_depth 10.0 --voxel_size 0.003 --it 100000")

print(f"\nEvaluating\n")
os.system(f"python scripts/dtu_eval.py --s {args.s}")