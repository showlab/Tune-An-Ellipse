import os
import random
from typing import Union, List
import clip
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch.nn.functional as F
from torchvision.transforms import Normalize, Compose, ToTensor, Resize

from tae.ellipse import RotatatedEllipse

from third_party.clipes.pytorch_grad_cam import GradCAM
import third_party.clipes.clip as clipes
from third_party.clipes.utils import BACKGROUND_CATEGORY, BACKGROUND_CATEGORY_COCO, CLIPES_PROMPT_TEMPLATE, \
    zeroshot_classifier, ClipOutputTarget, reshape_transform

imagenet_norm = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
clip_norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


class MultiImage:
    """
    MultiImage is a class that contains the image of different sizes,
    also contains the gray image and the gaussian image.
    All images are not normalized.
    Initialize the MultiImage with the image path and the target size list.
    """

    def __init__(self, image_path: str = None, target_size_list: List[int] = None, device=None, gaussian_std=60):
        self.image_path = image_path
        self.images = {}
        self.gaussian_images = {}
        self.gray_images = {}
        if image_path is None:
            return
        for target_size in target_size_list:
            if self.images.__contains__(target_size):
                continue
            transform = initialize_transform(target_size)
            img = Image.open(image_path)
            blurred_img = img.filter(ImageFilter.GaussianBlur(gaussian_std))
            blurred_img = transform(blurred_img).unsqueeze(0).to(device)
            self.gaussian_images[target_size] = blurred_img
            gray_img = img.convert('L').convert('RGB')
            gray_img = transform(gray_img).unsqueeze(0).to(device)
            self.gray_images[target_size] = gray_img
            image = preprocess_image(image_path, transform, device)
            self.images[target_size] = image

    def __getitem__(self, item):
        return self.images[item].clone()

    def gaussian(self, target_size):
        return self.gaussian_images[target_size].clone()

    def gray(self, target_size):
        return self.gray_images[target_size].clone()

    def set(self, target_size, image):
        self.images[target_size] = image

    @property
    def device(self):
        return list(self.images.values())[0].device

    def items(self):
        return self.images.items()

    def clone(self):
        return MultiImage(self.image_path, list(self.images.keys()), self.device)


class MultiTargetImage(MultiImage):
    """
    MultiTargetImage is a class that contains the image of different sizes.
    Initialize the MultiTargetImage with the image list and the target size list.
    """

    def __init__(self, images, target_sizes):
        super(MultiTargetImage, self).__init__()
        for size, img in zip(target_sizes, images):
            self.images[size] = img


class CLIPGradCAM:
    """
    CLIPGradCAM is a class that contains the CLIP model to compute the GradCAM.
    """
    def __init__(self, clip_type, bg_type='voc', img_text=None, device='cuda'):
        model, _ = clipes.load(clip_type, device=device)
        model = model.to(device)
        model.eval()

        target_layers = [model.visual.transformer.resblocks[-1].ln_1]
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform, use_cuda=True)

        self.model = model
        self.target_size = model.visual.input_resolution
        self.cam = cam
        self.img_text = img_text
        self.device = device
        self.bg_type = bg_type
        if img_text is not None:
            self.update_text_features(img_text)

    def update_text_features(self, img_text: str):
        with torch.no_grad():
            fg_text_features = self.model.encode_text(clipes.tokenize(img_text).to(self.device))
            fg_text_features /= fg_text_features.norm(dim=-1, keepdim=True)
            if self.bg_type == 'voc':
                bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, [CLIPES_PROMPT_TEMPLATE], self.model,
                                                       self.device)
            elif self.bg_type == 'coco':
                bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY_COCO, [CLIPES_PROMPT_TEMPLATE], self.model,
                                                       self.device)
            else:
                raise NotImplementedError
            fg_features_temp = fg_text_features.to(self.device)
            bg_features_temp = bg_text_features.to(self.device)
            text_features = torch.cat([fg_features_temp, bg_features_temp], dim=0)
        self.text_features = text_features

    def __call__(self, image, resize=True, resize_size=None):
        h, w = image.shape[-2], image.shape[-1]

        image_features, _ = self.model.encode_image(image, h, w)
        input_tensor = [image_features, self.text_features.to(image.device), h, w]
        targets = [ClipOutputTarget(0)] * image.shape[0]

        grayscale_cam, logits_per_image, _ = self.cam(
            input_tensor=input_tensor,
            targets=targets,
            target_size=None,
            clip_patch_size=self.model.visual.patch_size,
        )

        if resize:
            target_size = self.target_size if resize_size is None else resize_size
            grayscale_cam = F.interpolate(grayscale_cam.unsqueeze(0),
                                          size=(target_size, target_size), mode='bilinear',
                                          align_corners=False)[0]
        return grayscale_cam


class MultiCLIPGradCAM:
    """
    MultiCLIPGradCAM is a class that contains multiple CLIPGradCAM with different clip types.
    Use this class to compute the GradCAM of the image.
    """
    def __init__(self, clip_types: Union[str, List[str]], bg_type='voc', img_text=None, device='cuda'):
        if isinstance(clip_types, str):
            clip_types = [clip_types]
        self.clip_grad_cams = {
            clip_type: CLIPGradCAM(clip_type, bg_type, img_text, device)
            for clip_type in clip_types
        }

    def __getitem__(self, item):
        return self.clip_grad_cams[item]

    def items(self):
        return self.clip_grad_cams.items()

    def update_text_features(self, img_text: str):
        for clip_type, clip_grad_cam in self.clip_grad_cams.items():
            clip_grad_cam.update_text_features(img_text)

    def target_sizes(self):
        return [cam.target_size for cam in self.clip_grad_cams.values()]


class MultiCLIPModel:
    """
    MultiCLIPModel is a class that contains multiple CLIP models with different clip types.
    Use this class to compute the similarity between the image and the text.
    """
    def __init__(self, clip_types: Union[str, List[str]], bg_type="voc", text=None, device='cuda',
                 all_captions=None):
        if isinstance(clip_types, str):
            clip_types = [clip_types]
        self.models = {}
        self._sizes = {}
        self.background_text_features = {}
        for clip_type in clip_types:
            model, _ = load_clip_model(clip_type, device)
            self.models[clip_type] = model
            self._sizes[clip_type] = model.target_size
            if bg_type == 'voc':
                self.background_text_features[clip_type] = \
                    [gen_text_feat(model, "a photo of {}".format(bg)) for bg in BACKGROUND_CATEGORY]
            elif bg_type == 'coco':
                self.background_text_features[clip_type] = \
                    [gen_text_feat(model, "a photo of {}".format(bg)) for bg in BACKGROUND_CATEGORY_COCO]
            else:
                raise NotImplementedError
        self.text = text
        self.text_features = {}
        self.captions = all_captions

    def update_text(self, text: str):
        for clip_type, clip_model in self.models.items():
            self.text_features[clip_type] = gen_text_feat(clip_model, text)

    def __getitem__(self, item):
        return self.models[item]

    def items(self):
        return self.models.items()

    def text_feat(self, clip_type: str):
        return self.text_features[clip_type]

    def background_feat(self, clip_type: str) -> List[torch.Tensor]:
        return self.background_text_features[clip_type]

    def target_sizes(self):
        return list(set(self._sizes.values()))

    def __call__(self, image: Union[torch.Tensor, MultiImage]) -> torch.Tensor:
        similarities = []
        for mdoel_name, model in self.models.items():
            if isinstance(image, MultiImage) or isinstance(image, MultiTargetImage):
                img = image[model.target_size]
            else:  # torch.Tensor
                img = image
            img_feat = model.encode_image(img)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            text_feat = self.text_features[mdoel_name]
            similarity = img_feat @ text_feat.T
            similarities.append(similarity)
        similarities = torch.cat(similarities, dim=1)
        similarity = similarities.sum(dim=1)
        return similarity[0] if similarity.shape[0] == 1 else similarity


def draw_ellipse_on_image_torch(img: torch.Tensor, r_ellipse: torch.Tensor):
    # r_ellipse can be generated by RotatatedEllipse.forward()
    r_ellipse = torch.stack([r_ellipse, r_ellipse, r_ellipse], dim=0)
    r_ellipse[0, :, :] = r_ellipse[0, :, :] * 255
    r_ellipse[1, :, :] = r_ellipse[1, :, :] * -255
    r_ellipse[2, :, :] = r_ellipse[2, :, :] * -255
    ellipsed_img = img + r_ellipse
    ellipsed_img = ellipsed_img.clamp(0, 1)
    return ellipsed_img


def preprocess_image(img_path, transform, device, origin_size=False):
    image = Image.open(img_path)
    if origin_size:
        w, h = image.size
    image = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    image = image.to(device)
    if origin_size:
        return image, w, h
    return image


def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_clip_model(clip_type: str, device: Union[str, int] = 'cpu'):
    clip_model, preprocess = clip.load(clip_type, device=device)
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model.eval()
    clip_model.target_size = clip_model.visual.input_resolution
    return clip_model, preprocess


def gen_text_feat(clip_model, text: str):
    text_feat = clip_model.encode_text(clip.tokenize(text).cuda())
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat


def prepare_workspace(workspace: str, sub_dirs: list = None):
    if not os.path.exists(workspace):
        os.makedirs(workspace)
    if sub_dirs is None:
        sub_dirs = []
    sub_dirs += ['img', 'model', 'vis', 'pdf', 'init', 'npy']
    for sub_dir in sub_dirs:
        if not os.path.exists(os.path.join(workspace, sub_dir)):
            os.makedirs(os.path.join(workspace, sub_dir))


def initialize_transform(target_size):
    # We didn't normalize the image here.
    # The image will be normalized after drawn the ellipse.
    train_transform = Compose([
        lambda x: x.convert('RGB') if x.mode != 'RGB' else x,
        Resize((target_size, target_size), interpolation=Image.BICUBIC),
        ToTensor(),
    ])
    return train_transform


_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list, multi_contour_eval=False, scale=4):
    """
    borrowed from https://github.com/Sierkinhane/ORNet/blob/59e8ef5e461a5b00ca8c94d9fa0f5e58df193774/utils/func_utils.py#L55
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    # check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0 * scale, y0 * scale, x1 * scale, y1 * scale])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list


def draw_box_on_hd_image(image, box, color=(0, 255, 0), width=2):
    try:
        x, y, w, h = [int(_) for _ in box]
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, width)
    except Exception as e:
        # print(e, box)
        pass
    return image


def draw_ellipse_on_hd_image(_img_path, ellipse, _bbox=None, eta=50):
    (_cx, _cy, _a, _b, _t) = [torch.tensor(_, dtype=torch.float32, device='cpu') for _ in ellipse]

    _img = cv2.imread(_img_path)
    _h, _w, _ = _img.shape

    if _bbox is not None:
        _img = draw_box_on_hd_image(_img, _bbox, color=(0, 255, 0), width=3)

    rotElp_img = RotatatedEllipse(_w, _h, sigma=0.04, eta=eta)
    r_c, r_m = rotElp_img(_cx, _cy, _a, _b, _t)
    r_c = r_c.cpu().numpy()
    _img[r_c > 0.0001] = [0, 0, 255]

    return _img


