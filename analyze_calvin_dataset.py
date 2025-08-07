import glob
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import wandb
import clip
from tqdm import tqdm
import time
from itertools import chain
from PIL import Image, ImageDraw, ImageFont
import imageio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from models.seer_model import SeerAgent
from utils.train_utils import get_checkpoint, train_one_epoch_calvin, get_ckpt_name, AverageMeter, get_cast_dtype
from utils.arguments_utils import get_parser
from utils.data_utils import get_calvin_dataset, get_calvin_val_dataset, get_droid_dataset, get_libero_pretrain_dataset, get_libero_finetune_dataset, get_real_finetune_dataset, get_oxe_dataset
from utils.distributed_utils import init_distributed_device, world_info_from_env  


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def count_parameters(model):
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return total_params, trainable_params

def make_gif(rgb_statics, rgb_grippers, instruction, out_path="output.gif"):
        frames = []
        # font = ImageFont.load_default()  # Use default font
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=12)

        for i in range(len(rgb_statics)):
            rgb_static = rgb_statics[i]    # shape: (200, 200, 3)
            rgb_gripper = rgb_grippers[i]  # shape: (84, 84, 3)

            # Convert to PIL Images
            img_static = Image.fromarray(rgb_static.astype(np.uint8))
            img_gripper = Image.fromarray(rgb_gripper.astype(np.uint8)).resize((200, 200))

            # Combine side by side
            combined = Image.new("RGB", (400, 200))
            combined.paste(img_static, (0, 0))
            combined.paste(img_gripper, (200, 0))

            # Draw text (inst) on top
            draw = ImageDraw.Draw(combined)
            draw.text((10, 180), instruction, font=font, fill=(0, 0, 0))  # 빨간색 텍스트

            frames.append(np.array(combined))

        # Save as GIF
        imageio.mimsave(out_path, frames, duration=0.2, loop=0)  # duration per frame in seconds
        print(f"GIF saved to {out_path}")


def make_gif_eval(all_vis_logs, out_path):

    frames = []
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=12)
    
    for j in range(len(all_vis_logs)):
        rgb_statics = all_vis_logs[j][0]
        rgb_grippers = all_vis_logs[j][1]
        instruction = all_vis_logs[j][2]

        print(f'{j}th episode with {len(rgb_statics)} steps')
        for i in range(len(rgb_statics)):
            
            rgb_static = rgb_statics[i]    # shape: (200, 200, 3)
            rgb_gripper = rgb_grippers[i]  # shape: (84, 84, 3)

            # Convert to PIL Images
            img_static = Image.fromarray(rgb_static.astype(np.uint8))
            img_gripper = Image.fromarray(rgb_gripper.astype(np.uint8)).resize((200, 200))

            # Combine side by side
            combined = Image.new("RGB", (400, 200))
            combined.paste(img_static, (0, 0))
            combined.paste(img_gripper, (200, 0))

            # Draw text (inst) on top
            draw = ImageDraw.Draw(combined)
            draw.text((10, 180), instruction, font=font, fill=(0, 0, 0))  # 빨간색 텍스트

            frames.append(np.array(combined))

    # Save as GIF
    imageio.mimsave(out_path, frames, duration=0.2, loop=0)  # duration per frame in seconds
    print(f"GIF saved to {out_path}")


@record
def main(args):

    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    clip_device = device_id
    import clip
    clip_model, image_processor = clip.load("checkpoints/clip/ViT-B-32.pt", device=clip_device)

    calvin_dataset = get_calvin_dataset(args, image_processor, clip, epoch=0, except_lang=args.except_lang)
    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")
    device_id = args.rank % torch.cuda.device_count()


    lang_data = np.load("calvin/dataset/task_ABC_D/training/lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
    # print(lang_data)

    lang_lookup = lang_data["info"]["indx"]
    lang_ann = lang_data["language"]["ann"]
    print(len(lang_ann))
    
    rand_idxs = np.random.randint(0, len(lang_ann), 10)

    episodes = []
    for i in rand_idxs:
        start_idx, end_idx = lang_lookup[i]
        print(f"length of episode {i}: {end_idx - start_idx}")
        keys = list(chain(*calvin_dataset.dataset.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        samples = [
            calvin_dataset.dataset.load_file(calvin_dataset.dataset._get_episode_name(file_idx))
            for file_idx in range(start_idx, end_idx)
        ]
        episodes.append({key: np.stack([ep[key] for ep in samples]) for key in keys})

    for i, episode in enumerate(episodes):
        rgb_statics = episode["rgb_static"]  # L,200,200,3
        rgb_grippers = episode["rgb_gripper"]  # L,84,84,3
        robot_obss = episode["robot_obs"]  # L,15
        rel_actionss = episode["rel_actions"]  # L,7
        scene_obss = episode["scene_obs"]  # L,24
        make_gif(rgb_statics, rgb_grippers, lang_ann[i], out_path=f"calvin_analyze/output_{i}.gif")
    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    