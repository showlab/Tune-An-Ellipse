import os
from typing import List, Union, Tuple
import argparse
import torch
import cv2
from tqdm import tqdm
import numpy as np
from torch import multiprocessing

from tae.ellipse import MultiRotElpse
from tae.loss import SimMaxLoss
from tae.utils import (draw_ellipse_on_hd_image,
                       MultiCLIPGradCAM, MultiCLIPModel,
                       fix_seed, prepare_workspace, MultiImage)
from tae.initialize import EllipseInitializer

from dataset.refer.refer import REFER
from run import (ellipse_parser, tuning_an_ellipse
                 )


def process_train(p_id, args, ref_ids_list, r, gpu_ids):
    refer = REFER(args.root, args.reftype, splitBy=args.splitby)
    ref_ids = ref_ids_list[p_id]

    with torch.cuda.device(gpu_ids[p_id]):
        device = f'cuda'
        # ===================== Prepare =====================
        clipModel = MultiCLIPModel(args.sim_clip, device=device)

        GradCAM = MultiCLIPGradCAM(args.cam_clip, device=device)


        if p_id == 0:
            pbar = tqdm(total=len(ref_ids), desc=f"Process", dynamic_ncols=False, ascii=True)
        print(f"Process {p_id} is using GPU {gpu_ids[p_id]} and processing {len(ref_ids)} images.")
        for ref_id in ref_ids:
            ref = refer.Refs[ref_id]
            img_id = ref['image_id']
            file_name = refer.loadImgs(img_id)[0]['file_name']
            img_path = os.path.join(refer.IMAGE_DIR, file_name)
            Img = MultiImage(img_path, [224, 336, 384], device)
            # logger.info(f"ref_id: {ref_id}, img_path: {img_path}, caption: {caption}")

            for sentence in ref['sentences']:
                sent_id = sentence['sent_id']
                caption = sentence['raw']
                img_text_for_similarity = args.sim_text_prompt.format(caption)
                img_text_for_cam = args.cam_text_prompt.format(caption)
                result_path = os.path.join(args.workspace, 'npy', f'{ref_id}_{sent_id}.npy')

                if args.skip_exists and os.path.exists(result_path):
                    continue
                
                # ===================== Tuning =====================
                (cx, cy, a, b, t, similarity, model,
                ellipse_img, ellipse_mask, grad_cam_highres,
                init_grid_info, pred_ellipse_list) = (
                    tuning_an_ellipse(
                        args,
                        clipModel,
                        GradCAM, Img,
                        img_text_for_similarity, img_text_for_cam,
                        device, False)
                    )

                cv2.imwrite(os.path.join(args.workspace, 'img', f'{ref_id}_{sent_id}_{caption}.jpg'),
                            cv2.cvtColor(ellipse_img[0].detach().permute(1, 2, 0).cpu().numpy() * 255, cv2.COLOR_RGB2BGR))

                result = {
                    'ref_id': ref_id,
                    'sent_id': sent_id,
                    'file_name': file_name,
                    'caption': caption,
                    'ellipse_params': (cx, cy, a, b, t),
                    'ellipse_mask': ellipse_mask.detach().cpu().numpy(),
                    'cam': grad_cam_highres[0].detach().cpu().numpy(),
                    'sim': similarity.detach().cpu().numpy() if type(similarity) == torch.Tensor else similarity,
                }
                np.save(result_path, result)
                # torch.save(model.state_dict(), os.path.join(args.workspace, "model", f"{ref_id}_{sent_id}.pth"))

            if p_id == 0:
                pbar.update(1)


def arg_parser():
    parser, postprocess = ellipse_parser()

    parser.add_argument('--process_num', type=int, default=1)

    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--reftype', type=str, default='refcoco')
    parser.add_argument('--split', type=str, default='testA')
    parser.add_argument('--splitby', type=str, default='unc')

    parser.add_argument('--skip_exists', action='store_true')

    args = parser.parse_args()
    args = postprocess(args)

    return args


if __name__ == '__main__':
    # 固定随机种子
    seed = 0
    fix_seed(seed)
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"

    args = arg_parser()
    prepare_workspace(args.workspace)

    # logger = CircleLogger(None, os.path.join(args.workspace, 'log.txt'))
    print(f'\n=== Args {args} \n')

    refer = REFER(args.root, args.reftype, splitBy=args.splitby)
    ref_ids = refer.getRefIds(split=args.split)
    print(f"Total {len(ref_ids)} ref_ids in {args.split}")

    n_gpus = torch.cuda.device_count()
    n_process_per_gpu = args.process_num // n_gpus
    gpu_id_per_process = torch.arange(n_gpus).repeat(n_process_per_gpu).tolist()
    print(f'Process num: {args.process_num} on GPU {gpu_id_per_process}')

    # 将 list ref_ids 分成  args.process_num 份
    ref_ids_list = np.array_split(ref_ids, args.process_num)

    multiprocessing.spawn(process_train, nprocs=args.process_num,
                          args=(args, ref_ids_list, None, gpu_id_per_process),
                          join=True)
    torch.cuda.empty_cache()

