import os
from typing import List, Union, Tuple
import argparse
import torch
import cv2
from tqdm import tqdm

from tae.ellipse import MultiRotElpse
from tae.loss import SimMaxLoss
from tae.utils import (draw_ellipse_on_hd_image,
                       MultiCLIPGradCAM, MultiCLIPModel,
                       fix_seed, prepare_workspace, MultiImage)
from tae.initialize import EllipseInitializer
from tae.trainer import initialize_model, tune_one_step, initialize_ellipse_binary_cam


def save_hd_images(save_dir, img_path,
                   init_elp_params, pred_ellipse_list,
                   gt_box=None, steps=None):
    """
    保存tunning初始化和过程中椭圆图像
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    hd_elp_img = draw_ellipse_on_hd_image(img_path, init_elp_params, gt_box)
    target_path = os.path.join(save_dir,
                               f'init_'
                               f'({init_elp_params[0]:.3f}_{init_elp_params[1]:.3f}_{init_elp_params[2]:.3f}_{init_elp_params[3]:.3f}_{init_elp_params[4]:.3f}).png')
    cv2.imwrite(target_path, hd_elp_img)

    if steps is None:
        steps = [1, 5, 10, 20, 40, 80, 120, 160, 200, len(pred_ellipse_list)]
        steps = list(set(steps))
        steps = [step for step in steps if step <= len(pred_ellipse_list)]
        steps.sort()
    for step_i in steps:
        step_i = step_i - 1
        elp = pred_ellipse_list[step_i]
        hd_elp_img = draw_ellipse_on_hd_image(img_path, elp, gt_box)
        target_path = os.path.join(save_dir,
                                   f'step{step_i + 1}_({elp[0]:.3f}_{elp[1]:.3f}_{elp[2]:.3f}_{elp[3]:.3f}_{elp[4]:.3f}).png')
        cv2.imwrite(target_path, hd_elp_img)


def initialize_ellipse(args, Imgs: MultiImage, clipModel: MultiCLIPModel,
                       SimRotElpses: MultiRotElpse, CamRotElpses: MultiRotElpse,
                       GradCAMs: MultiCLIPGradCAM,
                       device) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    get the initial ellipse
    """
    ellipse_initializer = EllipseInitializer(Imgs, clipModel,
                                             SimRotElpses, CamRotElpses, GradCAMs)
    init_score, init_ellipse, chose_grid_size, grid_idx, init_similarity = (
        ellipse_initializer.anchor_topk_similarity_and_highest_mean_activation(
            args.topk,
            args.anchor_ratios,
            args.anchor_scales,
            base_circle_r=args.anchor_size,
            grid_size=args.anchor_grid_size,
        )
    )
    center_x, center_y, major_axis, minor_axis, angle = torch.tensor(init_ellipse, device=device)
    return (center_x, center_y, major_axis, minor_axis, angle,
            {'score': init_score, 'similarity': init_similarity, 'grid_size': chose_grid_size, 'grid_idx': grid_idx})


def tuning_an_ellipse(
    args,
    clipModel: MultiCLIPModel,
    GradCAM: MultiCLIPGradCAM,
    Img: MultiImage,
    text_for_sim: str,
    text_for_cam: str,
    device: str,
    show_pbar: bool = True,
):
    # ===================== Update Text =====================
    clipModel.update_text(text_for_sim)
    GradCAM.update_text_features(text_for_cam)

    # ===================== Prepare RotElpse =====================
    SimRotElpses = MultiRotElpse([224, 336, 384], sigma=args.sim_sigma, eta=args.eta)
    CamRotElpses = MultiRotElpse([224, 336, 384], sigma=args.cam_sigma, eta=args.eta)

    # ===================== Initialize Ellipse =====================
    center_x, center_y, major_axis, minor_axis, angle, init_grid_info = (
        initialize_ellipse(args, Img, clipModel,
                           SimRotElpses, CamRotElpses, GradCAM, device))
    predict_ellipse = (center_x, center_y, major_axis, minor_axis, angle)
    similarity = init_grid_info['similarity']
    init_grid_info['init_elpse'] = predict_ellipse

    # ===================== Prepare Init Grad-CAM =====================
    ellipse_mask, ellipse_img, grad_cam_highres = (
        initialize_ellipse_binary_cam(center_x, center_y, major_axis, minor_axis, angle,
                                      Img, GradCAM, CamRotElpses))

    # ===================== Model & Optimizer =====================
    model, optimizer, scheduler = initialize_model(args.num_step, args.lr,
                                                   (center_x, center_y, major_axis, minor_axis, angle),
                                                   224, 224,
                                                   args.sim_sigma, args.eta, device)

    # ===================== Loss =====================
    L_INF = SimMaxLoss(margin=0)
    L_SQZ = SimMaxLoss(margin=0)

    input_params = torch.zeros(1, 64).cuda()
    pred_ellipse_list = []

    # ===================== Differentiable Visual Prompting =====================
    import copy
    clipModel = copy.deepcopy(clipModel)
    del clipModel.models['ViT-L/14']
    if show_pbar:
        pbar = tqdm(total=args.num_step, desc=f"[LR=0.0000][Loss=0.000] Process", dynamic_ncols=False, ascii=True)
    for i in range(args.num_step):
        textFeats = {}
        for clip_name in args.sim_clip:
            textFeats[clip_name] = torch.cat([clipModel.text_feat(clip_name),
                                              *clipModel.background_feat(clip_name)], dim=0)
        (ellipse_img, ellipse_mask, predict_ellipse, similarity,
         m_act, m_act2, loss, l_sim, l_inf, l_sqz) = (
            tune_one_step(model, input_params, optimizer,
                          scheduler, clipModel, Img, textFeats, grad_cam_highres,
                          L_INF, L_SQZ))

        (cx, cy, a, b, t) = predict_ellipse
        pred_ellipse_list.append([cx.item(), cy.item(), a.item(), b.item(), t.item()])

        if show_pbar:
            pbar.update(1)
            pbar.set_description(f"[LR={optimizer.param_groups[0]['lr']:.4f}][Loss={loss.item():.4f}] Process")
    if show_pbar:
        pbar.close()

    (cx, cy, a, b, t) = (_p.detach().cpu().numpy() for _p in predict_ellipse)

    return (cx, cy, a, b, t, similarity, model,
            ellipse_img, ellipse_mask, grad_cam_highres,
            init_grid_info, pred_ellipse_list)


def train_ellipse(
        args,
        img_path: str,
        img_text: str,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    img_name = img_path.split('/')[-1].split('.')[0]
    process_name = f"{img_name}_({img_text.replace('/', '-')})"

    # ===================== CLIP Model =====================
    clipModel = MultiCLIPModel(args.sim_clip, device=device)

    # ===================== Prepare Text features =====================
    img_text_for_similarity = args.sim_text_prompt.format(img_text)
    img_text_for_cam = args.cam_text_prompt.format(img_text)

    # ===================== Prepare Img & RotElpse =====================
    Img = MultiImage(img_path, [224, 336, 384], device)
    GradCAM = MultiCLIPGradCAM(args.cam_clip, img_text=img_text_for_cam, device=device)

    # ===================== Tuning =====================
    (center_x, center_y, major_axis, minor_axis, angle, similarity, model,
     ellipse_img, ellipse_mask, grad_cam_highres,
     init_grid_info, pred_ellipse_list) = (
        tuning_an_ellipse(args,
                          clipModel,
                          GradCAM, Img,
                          img_text_for_similarity, img_text_for_cam,
                          device))

    # ===================== Save Tuning images =====================
    init_elp_params = init_grid_info['init_elpse']
    hd_tunning_dir = os.path.join(args.workspace, 'hd_tune', process_name)
    save_hd_images(hd_tunning_dir, img_path,
                   init_elp_params,
                   pred_ellipse_list, init_grid_info['similarity'])

    # ===================== Save Tuning Elps =====================
    result_file_path = os.path.join(args.workspace, 'result.txt')
    with open(result_file_path, 'a') as f:
        f.write(f"{process_name}_Init {init_elp_params}\n")
        for i, ellipse in enumerate(pred_ellipse_list):
            f.write(f"{process_name}_{i} {ellipse}\n")

    # ===================== Save Model =====================
    torch.save(model.state_dict(), os.path.join(args.workspace, "model", f"{process_name}.pth"))


def ellipse_parser():
    parser = argparse.ArgumentParser(description='Tuning-An-Ellipse')
    parser.add_argument('--workspace', type=str, default='workspace/test')
    parser.add_argument('--img_path', type=str, default='source/frog.png')
    parser.add_argument('--caption', type=str, default='the frog in the middle')

    # Rotated Ellipse
    parser.add_argument('--sim_sigma', type=float, default=0.05)
    parser.add_argument('--cam_sigma', type=float, default=0.05)
    parser.add_argument('--eta', type=float, default=50)

    # CLIP model
    parser.add_argument('--sim_clip', type=str, default="ViT-B/16,ViT-L/14",
                        help="CLIP model for calculating similarity")
    parser.add_argument('--cam_clip', type=str, default="ViT-B/16,ViT-L/14@336px",
                        help="CLIP model for calculating CAM")

    parser.add_argument('--sim_text_prompt', type=str, default="{}")
    parser.add_argument('--cam_text_prompt', type=str, default="a clean origami {}")

    # Anchor initialization
    parser.add_argument('--anchor_ratios', type=str, default='0.5,1,2')
    parser.add_argument('--anchor_scales', type=str, default='1,2')
    parser.add_argument('--anchor_size', type=float, default=0.1)
    parser.add_argument('--anchor_grid_size', type=int, default=9)
    parser.add_argument('--topk', type=float, default=10,
                        help="Anchor-Ellipse in top-k similarity and highest mean activation value will be chosen")

    # Tuning
    parser.add_argument('--num_step', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)

    def _args_postprocess(_args):
        _args.anchor_ratios = [float(x) for x in _args.anchor_ratios.split(',')]
        _args.anchor_scales = [float(x) for x in _args.anchor_scales.split(',')]
        _args.cam_clip = _args.cam_clip.split(',')
        _args.sim_clip = _args.sim_clip.split(',')
        return _args

    return parser, _args_postprocess


def arg_parser():
    parser, args_postprocess = ellipse_parser()
    args = parser.parse_args()
    args = args_postprocess(args)
    return args


if __name__ == '__main__':
    fix_seed(0)

    args = arg_parser()
    prepare_workspace(args.workspace)

    train_ellipse(
        args,
        img_path=args.img_path,
        img_text=args.caption,
    )
