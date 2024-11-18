# %%
from tqdm.auto import tqdm
import torch
from PIL import Image
from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import load_image, export_to_video
import time
import os
import json
import random
import numpy as np
import subprocess
import re
import logging

# %%

def setSeeds(seed):
    global logger
    # predefining random initial seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_torch_visible_gpu_info():
    # Check if CUDA is available and GPUs are visible to PyTorch
    if not torch.cuda.is_available():
        print("No GPU found by PyTorch.")
        return

    # Get the remapped GPU indices from the CUDA_VISIBLE_DEVICES environment variable
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        # CUDA_VISIBLE_DEVICES is a comma-separated list of device indices, e.g., "0,2,3"
        visible_gpu_indices = [int(idx) for idx in cuda_visible_devices.split(",")]
    else:
        # If CUDA_VISIBLE_DEVICES is not set, use the default order
        visible_gpu_indices = list(range(torch.cuda.device_count()))

    print(f"Found {len(visible_gpu_indices)} GPU(s) visible to PyTorch:")

    # Run nvidia-smi and parse the output
    try:
        nvidia_smi_output = subprocess.check_output(
            "nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits",
            shell=True
        )
        gpu_info = nvidia_smi_output.decode('utf-8').strip().split('\n')

        # Display only the GPUs that are visible to PyTorch, using CUDA_VISIBLE_DEVICES mapping
        gpu_free_mem = []
        for line in gpu_info:
            gpu_index, gpu_name, total_memory, used_memory = re.match(r"(\d+),\s*(.*?),\s*(\d+),\s*(\d+)", line).groups()
            gpu_index = int(gpu_index)
            total_memory = int(total_memory)
            used_memory = int(used_memory)

            # Check if the actual GPU index is in the remapped list
            if gpu_index in visible_gpu_indices:
                # Map the original index to the PyTorch-visible index
                pytorch_index = visible_gpu_indices.index(gpu_index)
                free_memory = total_memory - used_memory
                print(f"GPU {pytorch_index}: {gpu_name} - {used_memory}/{total_memory} MB used, {free_memory} MB free")
                gpu_free_mem.append(free_memory)
        return gpu_free_mem
    except subprocess.CalledProcessError as e:
        print("Error running nvidia-smi:", e)
    except Exception as e:
        print("An error occurred:", e)


def get_chunk_number(base_path):
    print(os.listdir(os.path.join(base_path, 'chunked')))
    return len(os.listdir(os.path.join(base_path, 'chunked')))

def get_next_chunk_fn_idx_and_update(base_path):
    with open(os.path.join(base_path,'next_chunk.json')) as f:
        next_chunk = json.load(f)
    idx = next_chunk["next_chunk_index"]
    next_chunck_fn = os.path.join(base_path, 'chunked', f'video_desc_train_{idx}.json')
    next_chunk["next_chunk_index"] += 1
    with open(os.path.join(base_path,'next_chunk.json'),'w') as f:
        json.dump(next_chunk, f, indent=4)
    return next_chunck_fn, idx

# %%
def main():
    PATH = './local_model'

    SEED = 424242

    OUTPUT_BASE_PATH = os.path.join('..','..','datasets', 'SemArtVideoArtGen')

    INPUT_PATH = os.path.join('..', '..', 'final_train')

    # Call the function to display info for GPUs visible to PyTorch
    gpu_free_mem = get_torch_visible_gpu_info()

    CUDA_DEVICE_ID = int(np.argmax(gpu_free_mem))
    print(f"device with most available memory is {CUDA_DEVICE_ID}")
    device = f"cuda:{str(CUDA_DEVICE_ID)}"
    torch.cuda.set_device(int(device[-1]))

    # %%
    logging.basicConfig()
    logging.getLogger().setLevel(logging.WARNING)
    #logging.getLogger().setLevel(logging.DEBUG)

    # %%
    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

    fps = 24

    model_dtype, torch_dtype = 'bf16', torch.bfloat16   # Use bf16 (not support fp16 yet)

    model = PyramidDiTForVideoGeneration(
        PATH,                                         # The downloaded checkpoint dir
        model_dtype,
        model_name="pyramid_flux",
        model_variant='diffusion_transformer_384p',     # SD3 supports 'diffusion_transformer_768p'
    )

    model.vae.enable_tiling()
    model.vae.to(device)
    model.dit.to(device)
    model.text_encoder.to(device)

    # if you're not using sequential offloading bellow uncomment the lines above ^
    #model.enable_sequential_cpu_offload()


    chunk_number = get_chunk_number(INPUT_PATH)
    chunck_fn, idx = get_next_chunk_fn_idx_and_update(INPUT_PATH)
    
    while(idx < chunk_number):
    # %%
        setSeeds(SEED)

        video_out_path = os.path.join(OUTPUT_BASE_PATH, 'videos')
        os.makedirs(video_out_path, exist_ok=True)

        with open(chunck_fn) as f:
            data = json.load(f)

        for el in tqdm(data['data'], desc=f'chunck {idx}'):
            prompt = el['generated_video_desc']
            video_output_fn = os.path.join(video_out_path, f"{'.'.join(el['painting_file'].split('.')[:-1])}.mp4")
            if not os.path.exists(video_output_fn):
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
                    frames = model.generate(
                        prompt=prompt,
                        num_inference_steps=[20, 20, 20],
                        video_num_inference_steps=[10, 10, 10],
                        height=384,     
                        width=640,
                        temp=16,                    # temp=16: 5s, temp=31: 10s
                        guidance_scale=7.0,         # The guidance for the first frame, set it to 7 for 384p variant
                        video_guidance_scale=5.0,   # The guidance for the other video latent
                        output_type="pil",
                        save_memory=False, #True,           # If you have enough GPU memory, set it to `False` to improve vae decoding speed
                        #save_memory doesn't seem to do much difference
                    )

                export_to_video(frames, os.path.join(video_out_path, f"{'.'.join(el['painting_file'].split('.')[:-1])}.mp4"), fps=fps)

            else:
                print(f'video already generated for {video_output_fn}')
        
        chunck_fn, idx = get_next_chunk_fn_idx_and_update(INPUT_PATH)

if __name__=='__main__':
    main()
