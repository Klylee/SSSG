import os
import pynvml
import time

def is_gpu_idle(gpu_index=0, threshold=10):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    pynvml.nvmlShutdown()
    return util.gpu < threshold

gpu = 1
scene = "acientdragon"
output_dir = f"results/wo_fresnel/{scene}"
# output_dir = f"output_neuralto/{scene}"

check_interval = 60  # seconds
while True:
    if is_gpu_idle(gpu_index=gpu):
        print(f"GPU{gpu} idle，run task...")
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "nohup.out")
        if os.path.exists(log_file):
            os.remove(log_file)
        
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} nohup python train.py -s ~/expdata/neuralto/{scene} -m {output_dir}/test > {log_file} 2>&1 &"
        os.system(cmd)

        break
    else:
        print(f"GPU{gpu} busy, retry after {check_interval}s...")
        time.sleep(check_interval)

