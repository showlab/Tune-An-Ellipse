import numpy as np
from matplotlib import pyplot as plt
import torch
import logging


class CircleLogger:
    def __init__(self,
                 title,
                 log_path):

        self._initialize_logger(title, log_path)

    def _initialize_logger(self, title, log_path):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        th = logging.StreamHandler()
        th.setLevel(logging.INFO)
        th.setFormatter(formatter)
        logger.addHandler(th)
        self.logger = logger

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)


class CircleBoard:
    def __init__(self, board_path, board_per_row=5, log_board: bool = True):
        self.board_path = board_path
        self.num_per_row = board_per_row
        self.board_subfigs = []
        self.log_board = log_board

    def _initialize_board(self, subplot_num, wspace=0.1, hspace=0.1):
        num_per_row = self.num_per_row
        n_rows = (subplot_num - 1) // num_per_row + 1
        n_cols = min(subplot_num, num_per_row)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        fig.subplots_adjust(wspace=wspace, hspace=hspace)
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)
        self.board_fig = fig
        self.board_axes = axes
        self.board_subplot_num = n_rows * n_cols

    def _draw_subfis_on_board(self):
        axes = self.board_axes
        num_per_row = self.num_per_row
        for i, subfig in enumerate(self.board_subfigs):
            ax = axes[i // num_per_row, i % num_per_row]
            if 'img' in subfig:
                ax.imshow(subfig['img'])
            elif 'plots' in subfig:
                ax.plot(subfig['plots'])
            if 'title' in subfig:
                ax.set_title(subfig['title'], fontsize=subfig['title_font_size'])
            if 'axis' in subfig:
                ax.axis('on' if subfig['axis'] else 'off')

    def _clear_unused_axes(self):
        for i in range(len(self.board_subfigs), self.board_subplot_num-1):
            ax = self.board_axes[i // self.num_per_row, i % self.num_per_row]
            ax.axis('off')

    def show_image_on_board(self, img, title=None):
        if not self.log_board:
            return
        if type(img) is torch.Tensor:
            img = img.detach().permute(1, 2, 0).cpu().numpy()
        elif type(img) is np.ndarray:
            img = img
        else:
            raise TypeError(f'img type {type(img)} is not supported.')

        subfig = {'img': img, 'title': title, 'title_font_size': 15, 'axis': False}
        self.board_subfigs.append(subfig)

    def show_plot_on_board(self, plots, title=None):
        if not self.log_board:
            return
        subfig = {'plots': plots, 'title': title, 'title_font_size': 15}
        self.board_subfigs.append(subfig)

    def save_board(self, **kwargs):
        if not self.log_board:
            return
        self._initialize_board(len(self.board_subfigs))
        self._draw_subfis_on_board()
        self._clear_unused_axes()

        save_path = self.board_path
        self.board_fig.tight_layout()
        self.board_fig.savefig(save_path, **kwargs)
        plt.close(self.board_fig)
