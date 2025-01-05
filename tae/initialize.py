import os
from typing import List, Union, Tuple
import numpy as np
import torch

from tae.utils import clip_norm, MultiCLIPModel, MultiCLIPGradCAM, MultiImage, MultiTargetImage, \
    draw_ellipse_on_image_torch
from tae.ellipse import MultiRotElpse


def get_grid_circles(img_w, img_h, grid_size) -> List[List[float]]:
    grid_w = img_w // grid_size
    grid_h = img_h // grid_size

    ellipse_list = []
    for i in range(1, grid_size + 1):
        for j in range(1, grid_size + 1):
            cx = (i - 1) * grid_w + grid_w // 2
            cy = (j - 1) * grid_h + grid_h // 2
            a = grid_w // 2
            b = grid_h // 2
            cx = cx / img_w
            cy = cy / img_h
            a = a / img_w
            b = b / img_h
            ellipse_list.append([cx, cy, a, b, 0])
    return ellipse_list


def get_anchor_ellipses(cx: float, cy: float, a: float, b: float, t: float,
                        ratios: List[float], scales: List[float]):
    ellipses = []
    for ratio in ratios:
        for scale in scales:
            a_ = a * scale / ratio
            b_ = b * scale * ratio
            ellipses.append([cx, cy, a_, b_, t])
    return ellipses


class EllipseInitializer:
    def __init__(self,
                 Imgs: MultiImage,
                 clipModels: MultiCLIPModel,
                 sim_RotElps: MultiRotElpse,
                 cam_RotElps: MultiRotElpse,
                 gradCAMs: MultiCLIPGradCAM,
                 ):
        self.Imgs = Imgs
        self.clipModels = clipModels
        self.target_size = 224
        self.sim_RotElps = sim_RotElps
        self.cam_RotElps = cam_RotElps
        self.gradCAMs = gradCAMs
        self.batch_size = 128

    def _normed_image_batch(self, image: torch.Tensor, ellipse_batch_list: List[List[float]]):
        """
        draw the ellipse on image, and normalize
        """
        # get ellipse region curve and ellipse mask
        img_size = image.shape[-1]
        ellipse_batch = torch.tensor(ellipse_batch_list, device=image.device)  # (batch_size, 5)
        cx, cy, a, b, t = (ellipse_batch[:, 0], ellipse_batch[:, 1], ellipse_batch[:, 2],
                           ellipse_batch[:, 3], ellipse_batch[:, 4])
        r_ellipse, r_mask = self.sim_RotElps[img_size](cx, cy, a, b, t)

        # draw ellipse on image
        ellipsed_img = [draw_ellipse_on_image_torch(image, r_ellipse_i) for r_ellipse_i in r_ellipse]
        # normalize
        norm_ellipsed_img = [clip_norm(ellipsed_img_i) for ellipsed_img_i in ellipsed_img]

        return norm_ellipsed_img, r_mask

    def _get_ellipse_cam_activation_values(self, ellipse_list: List[List[float]], batch_size: int = 1):
        """
        calculate the mean activation value of the ellipse region, increase batch_size to speed up
        """
        sizes = self.gradCAMs.target_sizes()
        activation_values = []
        for i in range(0, len(ellipse_list), batch_size):
            ellipse_batch = ellipse_list[i:i + batch_size]

            multi_size_imgs = {s: [] for s in sizes}
            r_mask = None
            for size in sizes:
                _s_img = self.Imgs[size]
                norm_ellipsed_img, _r_m = self._normed_image_batch(_s_img, ellipse_batch)
                multi_size_imgs[size] = norm_ellipsed_img
                if size == self.target_size:
                    r_mask = _r_m

            cams = []
            gradcam_origin_cams = []
            for _, gradcam_object in self.gradCAMs.items():
                _s = gradcam_object.target_size
                _cam = gradcam_object(torch.cat(multi_size_imgs[_s]), resize=True, resize_size=self.target_size).detach()
                _origin_cam = gradcam_object(self.Imgs[_s], resize=True, resize_size=self.target_size).detach()
                cams.append(_cam)
                gradcam_origin_cams.append(_origin_cam)
            cams = torch.stack(cams)    # (num_models, batch_size, 224, 224)
            gradcam_origin_cams = torch.stack(gradcam_origin_cams)    # (num_models, 1, 224, 224)
            gradcam_origin_cams = torch.repeat_interleave(gradcam_origin_cams, len(ellipse_batch), dim=1)
            cams = torch.cat([cams, gradcam_origin_cams], dim=0)    # (num_models*2, batch_size, 224, 224)
            cams = cams.mean(dim=0)  # (batch_size, 224, 224)
            masked_cams = cams * r_mask

            acts = masked_cams.sum(dim=[1, 2]) / r_mask.sum(dim=[1, 2])
            activation_values.extend(acts.tolist())

        return activation_values

    def _get_ellipse_similarities(self, ellipse_list: List[List[float]], batch_size: int = 1) \
            -> List[np.array]:
        """
        calculate the similarity between ellipsed-image and caption, increase batch_size to speed up
        """
        sizes = self.clipModels.target_sizes()

        similarity_list = []
        for i in range(0, len(ellipse_list), batch_size):
            ellipse_batch = ellipse_list[i:i + batch_size]

            multi_size_imgs = {s: [] for s in sizes}
            for size in sizes:
                _s_img = self.Imgs[size]
                norm_ellipsed_img, _ = self._normed_image_batch(_s_img, ellipse_batch)
                multi_size_imgs[size] = norm_ellipsed_img

            norm_ellipsed_imgs = MultiTargetImage([torch.cat(multi_size_imgs[size]) for size in sizes], sizes)
            similarities = self.clipModels(norm_ellipsed_imgs)
            similarity_list.extend(similarities.tolist())

        return similarity_list

    def anchor_topk_similarity_and_highest_mean_activation(
            self, topk: Union[float, int],
            ratios: List[float], scales: List[float], base_circle_r: float = 0.1, grid_size: int = 9):
        """
        Anchor-Ellipse, 1.Top-k Similarity; 2.Highest Mean Activation Value
        """
        # generate anchor ellipse
        center_ellipses = get_grid_circles(self.target_size, self.target_size, grid_size)
        ellipses = []
        for center_elp in center_ellipses:
            ellipses += get_anchor_ellipses(cx=center_elp[0], cy=center_elp[1],
                                            a=base_circle_r, b=base_circle_r, t=0.,
                                            ratios=ratios, scales=scales)

        # calculate similarity
        similarities = self._get_ellipse_similarities(ellipses, batch_size=self.batch_size)

        # get topk similarity
        if topk < 1:
            topk_num = int(topk * len(similarities))
        else:
            topk_num = int(topk)
        topk_idx = np.argsort(similarities, )[-topk_num:][::-1]
        ellipses = [ellipses[idx] for idx in topk_idx]
        similarities = [similarities[idx] for idx in topk_idx]

        # calculate mean activation value
        mean_values = self._get_ellipse_cam_activation_values(ellipses, batch_size=self.batch_size)

        # get the highest mean activation value
        chosed_idx = np.argmax(mean_values)

        score = similarities[chosed_idx]
        ellipse = ellipses[chosed_idx]
        return score, ellipse, grid_size, chosed_idx, score
