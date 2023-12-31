#!/usr/bin/env python
 
import argparse
import os
import os.path as osp
 
import numpy as np
import scipy.sparse
import torch
import torch.utils.data as tdata
from PIL import Image
 
import hw_4
 
# Constants for drawing
BORDER = 10
COLORS_CLAZZ = (
    np.array(
        (
            (128, 128, 128, 100),
            (245, 130, 48, 100),
            (255, 255, 25, 100),
            (240, 50, 230, 100),
            (0, 130, 200, 100),
            (60, 180, 75, 100),
        )
    )
    / 255
)
 
COLORS_OK = np.array(((255, 0, 0, 100), (0, 255, 0, 100))) / 255
 
# Constants about problem
CLAZZ = ['Background & Buildings', 'Car', 'Humans & Bikes', 'Interest', 'Sky', 'Nature']
WEIGHTS = np.array([0, 1, 1, 1, 0, 0])
NUM_CLAZZ = len(CLAZZ)
 
 
class Dataset(tdata.Dataset):
    def __init__(self, rgb_file, label_file):
        super().__init__()
        self.rgbs = np.load(rgb_file, mmap_mode='r')  # mmap is way faster for these large data
        self.labels = np.load(label_file, mmap_mode='r')  # mmap is way faster for these large data
 
    def __len__(self):
        return self.rgbs.shape[0]
 
    def __getitem__(self, i):
        return {
            'labels': np.asarray(self.labels[i]).astype('i8'),  # torch wants labels to be of type LongTensor, in order to compute losses
            'rgbs': np.asarray(self.rgbs[i]).astype('f4').transpose((2, 0, 1)) / 255,
            'key': i,  # for saving of the data
            # due to mmap, it is necessary to wrap your data in np.asarray. It does not add almost any overhead as it does not copy anything
        }
 
 
def blend_img(background, overlay_rgba, gamma=2.2):
    alpha = overlay_rgba[:, :, 3]
    over_corr = np.float_power(overlay_rgba[:, :, :3], gamma)
    bg_corr = np.float_power(background, gamma)
    return np.float_power(over_corr * alpha[..., None] + (1 - alpha)[..., None] * bg_corr, 1 / gamma)  # dark magic
    # partially taken from https://en.wikipedia.org/wiki/Alpha_compositing#Composing_alpha_blending_with_gamma_correction
 
 
def create_vis(rgb, label, prediction):
    if rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    if len(prediction.shape) == 3:
        prediction = np.argmax(prediction, 0)
 
    h, w, _ = rgb.shape
 
    gt_map = blend_img(rgb, COLORS_CLAZZ[label])  # we can index colors, wohoo!
    pred_map = blend_img(rgb, COLORS_CLAZZ[prediction])
    ok_map = blend_img(rgb, COLORS_OK[(label == prediction).astype('u1')])  # but we cannot do it by boolean, otherwise it won't work
    canvas = np.ones((h * 2 + BORDER, w * 2 + BORDER, 3))
    canvas[:h, :w] = rgb
    canvas[:h, -w:] = gt_map
    canvas[-h:, :w] = pred_map
    canvas[-h:, -w:] = ok_map
 
    canvas = (np.clip(canvas, 0, 1) * 255).astype('u1')
    return Image.fromarray(canvas)
 
 
class Metrics:
    def __init__(self, num_classes, weights=None, clazz_names=None):
        self.num_classes = num_classes
        self.cm = np.zeros((num_classes, num_classes), 'u8')  # confusion matrix
        self.tps = np.zeros(num_classes, dtype='u8')  # true positives
        self.fps = np.zeros(num_classes, dtype='u8')  # false positives
        self.fns = np.zeros(num_classes, dtype='u8')  # false negatives
        self.weights = weights if weights is not None else np.ones(num_classes)  # Weights of each class for mean IOU
        self.clazz_names = clazz_names if clazz_names is not None else np.arange(num_classes)  # for nicer printing
 
    def update(self, labels, predictions, verbose=True):
        labels = labels.cpu().numpy()
        predictions = predictions.cpu().numpy()
 
        predictions = np.argmax(predictions, 1)  # first dimension are probabilities/scores
 
        tmp_cm = scipy.sparse.coo_matrix(
            (np.ones(np.prod(labels.shape), 'u8'), (labels.flatten(), predictions.flatten())), shape=(self.num_classes, self.num_classes)
        ).toarray()  # Fastest possible way to create confusion matrix. Speed is the necessity here, even then it takes quite too much
 
        tps = np.diag(tmp_cm)
        fps = tmp_cm.sum(0) - tps
        fns = tmp_cm.sum(1) - tps
        self.cm += tmp_cm
        self.tps += tps
        self.fps += fps
        self.fns += fns
 
        precisions, recalls, ious, weights, miou = self._compute_stats(tps, fps, fns)
 
        if verbose:
            self._print_stats(tmp_cm, precisions, recalls, ious, weights, miou)
 
    def _compute_stats(self, tps, fps, fns):
        with np.errstate(all='ignore'):  # any division could be by zero, we don't really care about these errors, we know about these
            precisions = tps / (tps + fps)
            recalls = tps / (tps + fns)
            ious = tps / (tps + fps + fns)
            weights = np.copy(self.weights)
            weights[np.isnan(ious)] = 0
            miou = np.ma.average(ious, weights=weights)
        return precisions, recalls, ious, weights, miou
 
    def _print_stats(self, cm, precisions, recalls, ious, weights, miou):
        print('Confusion matrix:')
        print(cm)
        print('\n---\n')
        for c in range(self.num_classes):
            print(
                f'Class: {str(self.clazz_names[c]):20s}\t'
                f'Precision: {precisions[c]:.3f}\t'
                f'Recall {recalls[c]:.3f}\t'
                f'IOU: {ious[c]:.3f}\t'
                f'mIOU weight: {weights[c]:.1f}'
            )
        print(f'Mean IOU: {miou}')
        print('\n---\n')
 
    def print_final(self):
        precisions, recalls, ious, weights, miou = self._compute_stats(self.tps, self.fps, self.fns)
        self._print_stats(self.cm, precisions, recalls, ious, weights, miou)
 
    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), 'u8')
        self.tps = np.zeros(self.num_classes, dtype='u8')
        self.fps = np.zeros(self.num_classes, dtype='u8')
        self.fns = np.zeros(self.num_classes, dtype='u8')
 
 
def evaluate(model, metrics, dataset, device, batch_size=8, verbose=True, create_imgs=False, save_dir='.'):
    model = model.eval().to(device)
    loader = tdata.DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
    with torch.no_grad():  # disable gradient computation
        for i, batch in enumerate(loader):
            data = batch['rgbs'].to(device)
 
            predictions = model(data)
            metrics.update(batch['labels'], predictions, verbose)
            if create_imgs:
                for j, img_id in enumerate(batch['key']):
                    img = create_vis(data[j].cpu().numpy(), batch['labels'][j].numpy(), predictions[j].cpu().numpy())
                    os.makedirs(save_dir, exist_ok=True)
                    img.save(osp.join(save_dir, f'{img_id:04d}.png'))
            print(f'Processed {i+1:02d}th batch')
 
    metrics.print_final()
    return metrics
 
 
def prepare(args, model=None):
    dataset = Dataset(args.dataset_rgbs, args.dataset_labels)
    if model is None:
        model = hw_4.load_model()
    metrics = Metrics(NUM_CLAZZ, WEIGHTS, CLAZZ)
    return model, metrics, dataset
 
 
def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, metrics, dataset = prepare(args)
    evaluate(model, metrics, dataset, device, args.batch_size, args.verbose, args.create_imgs, args.store_dir)
 
 
def parse_args():
    parser = argparse.ArgumentParser('Evaluation demo for HW03')
    parser.add_argument('dataset_rgbs', help='NPY file, where dataset RGB data is stored')
    parser.add_argument('dataset_labels', help='NPY file, where dataset labels are stored')
    parser.add_argument(
        '-ci', '--create_imgs', default=False, action='store_true', help='Whether to create images. Warning! It will take significantly longer!'
    )
    parser.add_argument('-sd', '--store_dir', default='.', help='Where to store images. Only valid, if create_imgs is set to True')
    parser.add_argument('-bs', '--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Whether to print stats of each minibatch')
 
    return parser.parse_args()
 
 
def main():
    args = parse_args()
    print(args)
    run(args)
 
 
if __name__ == '__main__':
    main()