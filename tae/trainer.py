import torch
import torch.nn.functional as F

from tae.net import EllipseGenerator
from tae.ellipse import MultiRotElpse
from tae.utils import (MultiImage, MultiCLIPModel, MultiCLIPGradCAM,
                       draw_ellipse_on_image_torch,
                       clip_norm)


def initialize_model(num_step, lr, init_ellipse_params, h, w, sigma, eta, device):
    model = EllipseGenerator(init_ellipse_params, h, w, sigma=sigma, eta=eta)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_step * 2, eta_min=0)
    return model, optimizer, scheduler


def _training_calc_similarity(clip_model, Img, r_ellipse, text_feat):
    # ===================== Ellipse on Image =====================
    img = Img[clip_model.target_size]
    ellipse_img = draw_ellipse_on_image_torch(img, r_ellipse)

    # ===================== Origin Image Similarity =====================
    norm_ellipse_img = clip_norm(ellipse_img)
    img_feat = clip_model.encode_image(norm_ellipse_img)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    similarity = img_feat @ text_feat.T

    return similarity, ellipse_img


def tune_one_step(model: EllipseGenerator, model_input,
                  optimizer, scheduler, clipModel: MultiCLIPModel,
                  Img: MultiImage, textFeats, cam,
                  L_INF, L_SQZ):
    """
    optimize the ellipse parameters (ensemble multiple CLIP models)
    """
    # ===================== forward =====================
    origin_r_ellipse, origin_ellipse_mask, predict_ellipse = model(model_input)

    # ===================== Similarity =====================
    all_similarity = []
    for clip_name, clip_model in clipModel.items():
        r_ellipse = F.interpolate(origin_r_ellipse[None, None, :, :],
                                  (clip_model.target_size, clip_model.target_size),
                                  mode='bilinear', align_corners=False)[0, 0]

        similarity, e_img = _training_calc_similarity(clip_model, Img, r_ellipse, textFeats[clip_name])

        all_similarity.append(similarity)
        if clip_model.target_size == 224:
            ellipse_img = e_img
    all_similarity = torch.stack(all_similarity).mean(dim=0)

    # ===================== Mean Activation Value =====================
    masked_grad_cam_highres = cam * origin_ellipse_mask
    mean_activation_value = masked_grad_cam_highres.sum() / (224 * 224)
    mean_activation_value2 = masked_grad_cam_highres.sum() / origin_ellipse_mask.sum()

    # ===================== Loss =====================
    l_sim = F.cross_entropy(all_similarity, torch.tensor([0], device=all_similarity.device))
    l_inf = L_INF(mean_activation_value, 1)
    l_sqz = L_SQZ(mean_activation_value2, 1)

    loss = l_sim + l_inf + l_sqz

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    all_similarity = all_similarity[0, 0]

    return (ellipse_img, origin_ellipse_mask, predict_ellipse,
            all_similarity, mean_activation_value, mean_activation_value2, loss, l_sim, l_inf, l_sqz)


def generate_binary_cam(ellipse_img, target_size, gradcam_object):
    norm_ellipse_img = clip_norm(ellipse_img)
    grad_cam_highres = gradcam_object(norm_ellipse_img, resize=True, resize_size=target_size)
    grad_cam_highres = grad_cam_highres.detach()
    return grad_cam_highres, None


def initialize_ellipse_binary_cam(center_x, center_y, major_axis, minor_axis, angle,
                                  Imgs: MultiImage, GradCAMs: MultiCLIPGradCAM, RotElpses: MultiRotElpse):
    cams = []
    ellipse_mask, ellipse_img = None, None
    for _, gradcam_object in GradCAMs.items():
        target_size = gradcam_object.target_size
        image = Imgs[target_size]
        r_e, e_mask = RotElpses[target_size](center_x, center_y, major_axis, minor_axis, angle)
        e_img = draw_ellipse_on_image_torch(image, r_e)

        if target_size == 224:
            ellipse_mask, ellipse_img = e_mask, e_img

        cam, _ = generate_binary_cam(e_img, 224, gradcam_object)
        cams.append(cam)

        cam, _ = generate_binary_cam(image, 224, gradcam_object)
        cams.append(cam)
    grad_cam_highres = torch.stack(cams).mean(dim=0)  # ensemble

    return ellipse_mask, ellipse_img, grad_cam_highres
