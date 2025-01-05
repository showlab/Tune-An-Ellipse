class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation, self.height, self.width)
        # self.activations.append(activation.cpu().detach())
        self.activations.append(activation)

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad, self.height, self.width)
            # self.gradients = [grad.cpu().detach()] + self.gradients
            self.gradients = [grad] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x, H, W, clip_patch_size=16):
        # self.height = H // 16
        # self.width = W // 16
        self.height = H // clip_patch_size
        self.width = W // clip_patch_size
        self.gradients = []
        self.activations = []
        if isinstance(x, list):
            return self.model.forward_last_layer(x[0], x[1])
        else:
            return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()
