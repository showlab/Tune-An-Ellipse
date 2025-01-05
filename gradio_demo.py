import os
import numpy as np
import torch
import gradio as gr
import cv2

from tae.utils import MultiCLIPModel, MultiImage, MultiCLIPGradCAM, \
    draw_ellipse_on_hd_image, prepare_workspace, fix_seed
from run import arg_parser, tuning_an_ellipse

args = arg_parser()
args.workspace = 'workspace/demo'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ===================== Launch CLIP Models =====================
prepare_workspace(args.workspace)
clipModel = MultiCLIPModel(args.sim_clip, device=device)
GradCAM = MultiCLIPGradCAM(args.cam_clip, device=device)


def main(img_path, caption):

    Img = MultiImage(img_path, [224, 336, 384], device)
    img_text_for_similarity = args.sim_text_prompt.format(caption)
    img_text_for_cam = args.cam_text_prompt.format(caption)

    # ===================== Tuning =====================
    (center_x, center_y, major_axis, minor_axis, angle, similarity, model,
     ellipse_img, ellipse_mask, grad_cam_highres,
     init_grid_info, pred_ellipse_list) = (
        tuning_an_ellipse(args,
                          clipModel,
                          GradCAM, Img,
                          img_text_for_similarity, img_text_for_cam,
                          device))

    # ===================== Draw =====================
    # init ellipse
    init_elp_params = init_grid_info['init_elpse']
    hd_init_img = draw_ellipse_on_hd_image(img_path, init_elp_params)

    # result ellipse
    result_elp_params = (center_x, center_y, major_axis, minor_axis, angle)
    hd_result_img = draw_ellipse_on_hd_image(img_path, result_elp_params)

    # processing
    procs_imgs = []
    for step_i in [1, 5, 10, 20, 40, 60, 80, 120, 160, 200]:
        step_i = step_i - 1
        proc_elp_params = pred_ellipse_list[step_i]
        proc_img = draw_ellipse_on_hd_image(img_path, proc_elp_params)
        procs_imgs.append(proc_img)
    procs_imgs_a = np.concatenate(procs_imgs[:5], axis=1)
    procs_imgs_b = np.concatenate(procs_imgs[5:], axis=1)
    procs_imgs = np.concatenate([procs_imgs_a, procs_imgs_b], axis=0)

    hd_init_img = cv2.cvtColor(hd_init_img, cv2.COLOR_BGR2RGB)
    hd_result_img = cv2.cvtColor(hd_result_img, cv2.COLOR_BGR2RGB)
    procs_imgs = cv2.cvtColor(procs_imgs, cv2.COLOR_BGR2RGB)

    procs_imgs = cv2.resize(procs_imgs, (procs_imgs.shape[1] // 2, procs_imgs.shape[0] // 2),
                            interpolation=cv2.INTER_AREA)

    return hd_init_img, hd_result_img, procs_imgs


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Tune-An-Ellipse Demo

        """
    )

    with gr.Row():
        with gr.Column():
            in_image = gr.Image(label="Input Image", type="filepath")
            in_refer_text = gr.Textbox(label="Caption")

            click_btn = gr.Button(value="Tune")
        with gr.Column():
            with gr.Row():
                out_init_ellipse_image = gr.Image(label="Initial Ellipse")
                out_result_image = gr.Image(label="Result Image")
            out_tuning_process_images = gr.Image(label="Tuning Process Images")

            click_btn.click(fn=main,
                            inputs=[in_image, in_refer_text],
                            outputs=[out_init_ellipse_image, out_result_image, out_tuning_process_images]
                            )

    gr.Examples(
        examples=[[os.path.join(os.path.dirname(__file__), "source/frog.png"), "the frog in the middle"],
                  [os.path.join(os.path.dirname(__file__), "source/cat.png"), "jumping cat"],
                  ],
        inputs=[in_image, in_refer_text],
        outputs=[out_init_ellipse_image, out_result_image, out_tuning_process_images],
        fn=main,
        cache_examples=True,
    )

if __name__ == '__main__':
    fix_seed(0)
    demo.launch(server_port=55888)
