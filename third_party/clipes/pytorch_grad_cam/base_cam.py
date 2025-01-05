import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple
from .activations_and_gradients import ActivationsAndGradients
from .utils.svd_on_activations import get_2d_projection
from .utils.image import scale_cam_image
from .utils.model_targets import ClassifierOutputTarget


class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)  # (1, 768)
        weighted_activations = weights[:, :, None, None] * activations  # (1, 768, 18, 32)
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)  # (1, 18, 32)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                target_size,
                eigen_smooth: bool = False,
                clip_patch_size: int = 16) -> np.ndarray:
        if self.cuda:
            # input_tensor = input_tensor.cuda()
            input_tensor = [input_tensor[0].cuda(), input_tensor[1].cuda(), input_tensor[2], input_tensor[3]]

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        W, H = self.get_target_width_height(input_tensor)
        outputs = self.activations_and_grads(input_tensor, H, W,
                                             clip_patch_size)  # (1, num(TEXT)), (1, 577, 577) attn_weight of last layer
        if targets is None:
            if isinstance(input_tensor, list):
                target_categories = np.argmax(outputs[0].cpu().data.numpy(), axis=-1)
            else:
                target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            # self.model.zero_grad()
            # if isinstance(input_tensor, list):
            #     loss = sum([target(output[0]) for target, output in zip(targets, outputs)])
            # else:
            #     loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss = sum(outputs[0][:, 0])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        # cam_per_layer = self.compute_cam_per_layer(input_tensor,
        #                                            targets,
        #                                            target_size,
        #                                            eigen_smooth)    # [(1, 1, 18, 32)]
        cam = self.compute_cam_per_layer(input_tensor,
                                         targets,
                                         target_size,
                                         eigen_smooth)  # [(1, 1, 18, 32)]
        if isinstance(input_tensor, list):
            # return self.aggregate_multi_layers(cam_per_layer), outputs[0], outputs[1]
            return cam, outputs[0], outputs[1]
        else:
            return self.aggregate_multi_layers(cam_per_layer), outputs

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        if isinstance(input_tensor, list):
            width, height = input_tensor[-1], input_tensor[-2]
        # width, height = input_tensor.size()[-1], input_tensor.size()[-2]
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            target_size,
            eigen_smooth: bool) -> np.ndarray:
        # activations_list = [a.cpu().data.numpy()
        #                     for a in self.activations_and_grads.activations]    # (1, 768, 18, 32)
        # grads_list = [g.cpu().data.numpy()
        #               for g in self.activations_and_grads.gradients]
        activations_list = self.activations_and_grads.activations  # (1, 768, 18, 32)
        grads_list = self.activations_and_grads.gradients

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            # cam = np.maximum(cam, 0).astype(np.float32)#float16->32
            # scaled = scale_cam_image(cam, target_size)
            # cam_per_target_layer.append(scaled[:, None, :]) # (1, 1, 18, 32)
            cam = torch.maximum(cam, torch.tensor(0., device='cuda')).type(torch.float32)
            # scaled = scale_cam_image(cam, target_size)
            # cam_per_target_layer.append(scaled[:, None, :]) # (1, 1, 18, 32)
            # cam = torch.where(cam, cam, torch.tensor(0., device='cuda')).type(torch.float32)
            if cam.shape[0] == 1:
                cam = cam - cam.min()
                cam_per_target_layer = cam / (1e-7 + cam.max())
            else:
                # cam: (batch_size, 14, 14)
                # 每一维度减去其最小值
                cam = cam - cam.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
                # 每一维度除以其最大值
                cam_per_target_layer = cam / (1e-7 + cam.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 target_size=None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False,
                 **kwargs
                 ) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, target_size, eigen_smooth, **kwargs)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
