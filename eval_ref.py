import os

import PIL
import math
import numpy
import torch

from dataset.refer.refer import REFER
import argparse
import numpy as np
import cv2
from tqdm import tqdm

# from log_utils import CircleLogger
from tae.utils import prepare_workspace
from tae.ellipse import RotatatedEllipse


def get_ellipse_bb(x, y, major, minor, angle_deg):
    """
    Compute tight ellipse bounding box.

    see https://stackoverflow.com/questions/87734/how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse#88020
    """
    t = np.arctan(-minor / 2 * np.tan(np.radians(angle_deg)) / (major / 2))
    [min_x, max_x] = sorted([x + major / 2 * np.cos(t) * np.cos(np.radians(angle_deg)) -
                             minor / 2 * np.sin(t) * np.sin(np.radians(angle_deg)) for t in (t + np.pi, t)])
    t = np.arctan(minor / 2 * 1. / np.tan(np.radians(angle_deg)) / (major / 2))
    [min_y, max_y] = sorted([y + minor / 2 * np.sin(t) * np.cos(np.radians(angle_deg)) +
                             major / 2 * np.cos(t) * np.sin(np.radians(angle_deg)) for t in (t + np.pi, t)])
    return min_x, min_y, max_x, max_y


def ellipse_to_bbox(cx, cy, a, b, t):
    # _w = np.sin(t / 180 * np.pi) * b + np.cos(t / 180 * np.pi) * a
    # _h = np.cos(t / 180 * np.pi) * b + np.sin(t / 180 * np.pi) * a
    # _x = cx - _w / 2
    # _y = cy - _h / 2

    x1, y1, x2, y2 = get_ellipse_bb(cy, cx, b*2, a*2, -t)
    return x1, y1, x2 - x1, y2 - y1


def ellipse_mask_to_bbox(mask, th=0.9):
    # cv2.imwrite('mask.jpg', mask * 255)
    mask = np.where(mask > th, 1, 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    # exit(-1)
    if len(contours) == 0:
        return 0, 0, 0, 0
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    x, y, w, h = cv2.boundingRect(box)
    return x, y, w, h


def calculate_box_iou(box1, box2):
    # box: (x, y, w, h)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    left_x = max(x1, x2)
    left_y = max(y1, y2)
    right_x = min(x1 + w1, x2 + w2)
    right_y = min(y1 + h1, y2 + h2)
    inter_area = max(0, right_x - left_x) * max(0, right_y - left_y)
    union_area = w1 * h1 + w2 * h2
    iou_value = inter_area / (union_area - inter_area)
    return iou_value


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray):
    assert mask1.shape == mask2.shape
    inter = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(inter) / np.sum(union)
    return iou


def draw_box_on_image(image, box, color=(0, 255, 0), width=2):
    try:
        x, y, w, h = [int(_) for _ in box]
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, width)
    except Exception as e:
        # print(e, box)
        pass
    return image


def draw_hd_ellipse(_img_path, ellipse, _bbox, eta):
    (_cx, _cy, _a, _b, _t) = [torch.tensor(_, dtype=torch.float32, device='cpu') for _ in ellipse]

    _img = cv2.imread(_img_path)
    _h, _w, _ = _img.shape

    _img = draw_box_on_image(_img, _bbox, color=(0, 255, 0), width=3)

    rotElp_img = RotatatedEllipse(_w, _h, sigma=0.04, eta=eta)
    r_c, r_m = rotElp_img(_cx, _cy, _a, _b, _t)
    r_c = r_c.cpu().numpy()
    _img[r_c > 0.0001] = [0, 0, 255]

    return _img



def coco_object_area_type(area: int):
    # if area < 32 * 32:
    if area < 42160:
        return 'small'
    # elif area < 96 * 96:
    elif area < 168640:
        return 'medium'
    else:
        return 'large'


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default='workspace/test')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--reftype', type=str, default='refcoco')
    parser.add_argument('--split', type=str, default='testA')
    parser.add_argument('--splitby', type=str, default='unc')
    parser.add_argument('--eval_type', type=str, default='box2')
    parser.add_argument('--mask_cam_threshold', type=float, default=0.5)

    parser.add_argument('--log_vis', action='store_true', default=False)
    parser.add_argument('--log_hd', action='store_true', default=False)

    parser.add_argument('--iou_threshold', type=float, default=0.5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()
    prepare_workspace(args.workspace, sub_dirs=['bbox', 'low_iou', 'hd_elp', 'wrong_size'])

    # logger = CircleLogger(None, os.path.join(args.workspace, 'log.txt'))
    print(f'\n=== Eval args {args}')
    print(f'Eval threshold: {args.iou_threshold}, [{args.eval_type}]')

    refer = REFER(args.root, args.reftype, splitBy=args.splitby)
    ref_ids = refer.getRefIds(split=args.split)

    correct_count = 0
    total_count = 0
    ious = []
    size_correct = {'small': 0, 'medium': 0, 'large': 0}
    size_total = {'small': 0, 'medium': 0, 'large': 0}
    size_iou = {'small': [], 'medium': [], 'large': []}
    pbar = tqdm(total=len(ref_ids))
    for ref_id in ref_ids:
        ref = refer.Refs[ref_id]
        gt_box = refer.getRefBox(ref_id)  # (x, y, w, h)
        gt_mask = refer.getMask(ref)    # (h, w)
        gt_mask, gt_area = gt_mask['mask'], gt_mask['area']
        gt_area = gt_box[2] * gt_box[3]
        img_id = ref['image_id']
        file_name = refer.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(refer.IMAGE_DIR, file_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        gt_boxed_img = draw_box_on_image(img, gt_box, (0, 255, 0))
        for sentence in ref['sentences']:
            sent_id = sentence['sent_id']
            caption = sentence['raw']
            result_path = os.path.join(args.workspace, 'npy', f'{ref_id}_{sent_id}.npy')
            if args.eval_type != 'mask' and not os.path.exists(result_path):
                continue
            # import datetime
            # file_date = datetime.datetime.fromtimestamp(os.path.getmtime(result_path))
            # if file_date < datetime.datetime(2023, 11, 8, 16, 00):
            #     continue
            # assert os.path.exists(result_path), f'{result_path} does not exist'

            result = None
            if os.path.exists(result_path):
                try:
                    result = np.load(result_path, allow_pickle=True).item()
                except Exception as e:
                    print(f'Error: {e}, {result_path}')
                    continue

            if args.eval_type == 'box2':
                cx, cy, a, b, t = result['ellipse_params']
                # cx = cx * h
                # cy = cy * w
                # a = a * h
                # b = b * w
                # t = t * 180
                bbox = ellipse_to_bbox(cx, cy, a, b, t)
                bbox = [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                if bbox[0] + bbox[2] > w:
                    bbox[2] = w - bbox[0]
                if bbox[1] + bbox[3] > h:
                    bbox[3] = h - bbox[1]
                iou = calculate_box_iou(gt_box, bbox)
                if args.log_vis:
                    box_img = draw_box_on_image(gt_boxed_img.copy(), bbox, (0, 0, 255))
                    cv2.imwrite(os.path.join(args.workspace, 'bbox', f'{ref_id}_{sent_id}_{caption}_iou{iou}.jpg'), box_img)
                if args.log_hd and iou >= args.iou_threshold:
                    hd_img = draw_hd_ellipse(img_path, result['ellipse_params'], gt_box, 50)
                    cv2.imwrite(os.path.join(args.workspace, 'hd_elp',
                                             f"{ref_id}_{sent_id}_{caption}_iou{iou:.4f}_sim{result['sim']:.4f}.png"),
                                hd_img)

            elif args.eval_type == 'box':
                if result.__contains__('box') and result['box'] is not None:
                    bbox = result['box']
                    # bbox = [bbox[0] * (w / 224), bbox[1] * (h / 224), bbox[2] * (w / 224), bbox[3] * (h / 224)]
                else:
                    cx, cy, a, b, t = result['ellipse_params']
                    e_mask = result['ellipse_mask']
                    e_mask = cv2.resize(e_mask, (w, h))
                    bbox = ellipse_mask_to_bbox(e_mask)
                iou = calculate_box_iou(gt_box, bbox)
                if args.log_vis:
                    box_img = draw_box_on_image(gt_boxed_img.copy(), bbox, (0, 0, 255))
                    cv2.imwrite(os.path.join(args.workspace, 'bbox', f'{ref_id}_{sent_id}_{caption}_iou{iou}.jpg'), box_img)
            elif args.eval_type == 'mask':
                mask_path = os.path.join(args.workspace, 'mask', f'{ref_id}_{sent_id}.jpg')
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (w, h))
                    iou = calculate_mask_iou(gt_mask, mask)
                elif result is not None and result.__contains__('ellipse_mask') and result['ellipse_mask'] is not None:
                    mask = result['ellipse_mask']
                    mask = cv2.resize(mask, (w, h))
                    mask = np.where(mask >= 0.9, 1, 0)
                    iou = calculate_mask_iou(gt_mask, mask)
                else:
                    raise ValueError(f'No mask found for {ref_id}_{sent_id}_{caption}')

            else:
                raise ValueError(f'Unknown eval type: {args.eval_type}')

            if iou >= args.iou_threshold:
                correct_count += 1
                # if args.eval_type == 'mask':
                size_correct[coco_object_area_type(gt_area)] += 1
            else:
                if args.eval_type == 'box':
                    # cv2.imwrite(os.path.join(args.workspace, 'low_iou',
                    #                          f'{ref_id}_{sent_id}_{caption}_iou{iou}_{file_name.split(".")[0]}.jpg'),
                    #             box_img)
                    pass
                box_img = draw_box_on_image(gt_boxed_img.copy(), bbox, (0, 0, 255))
                cv2.imwrite(os.path.join(args.workspace, 'wrong_size',
                                         f'{ref_id}_{sent_id}_[{coco_object_area_type(gt_area)}]_{caption}_iou{iou}.jpg'),
                            box_img)

            total_count += 1
            ious.append(iou)
            size_iou[coco_object_area_type(gt_area)].append(iou)
            size_total[coco_object_area_type(gt_area)] += 1
        pbar.update(1)
        pbar.set_description(f'ACC: {(correct_count / total_count):.4f}')
    acc = correct_count / total_count
    small_acc = (size_correct['small'] / size_total['small']) if size_total['small'] != 0 else 0
    medium_acc = (size_correct['medium'] / size_total['medium']) if size_total['medium'] != 0 else 0
    large_acc = (size_correct['large'] / size_total['large']) if size_total['large'] != 0 else 0
    small_mean = np.mean(size_iou['small']) if len(size_iou['small']) != 0 else 0
    medium_mean = np.mean(size_iou['medium']) if len(size_iou['medium']) != 0 else 0
    large_mean = np.mean(size_iou['large']) if len(size_iou['large']) != 0 else 0
    print(f'Correct count: {correct_count}, total count: {total_count}, '
                f'ACC: {acc}, mIoU: {np.mean(ious)}; '
                f'ACC: [small]: {small_acc:.2f}({size_correct["small"]}/{size_total["small"]}), '
                f'[medium]: {medium_acc:.2f}({size_correct["medium"]}/{size_total["medium"]}), '
                f'[large]: {large_acc:.2f}({size_correct["large"]}/{size_total["large"]}),'
                f'mIoU: [small]: {small_mean}, [medium]: {medium_mean}, [large]: {large_mean}')

#  TODO:
# [] mask iou
# []
