import argparse
import cv2
import datetime
import matplotlib.cm as cm
import numpy as np
import os
import sys
import time
import torch
sys.path.append('../neural-wrappers')

from neural_wrappers.utilities import minMaxNormalizeData
from neural_wrappers.pytorch import maybeCuda
from unet_tiny_sum import ModelUNetTinySum


def minMaxNormalizeFrame(frame):
    Min, Max = np.min(frame), np.max(frame)
    frame -= Min
    frame /= (Max - Min)
    frame *= 255
    return np.uint8(frame)


def get_opts():
    p = argparse.ArgumentParser()
    p.add_argument('--dstdir', type=str, default='results')
    p.add_argument('--task', type=str, default='classification')
    p.add_argument('--weights_file', type=str, default=None)
    p.add_argument('--cam_id', type=int, default=0)
    p.add_argument('--cam_width', type=int, default=320)
    p.add_argument('--cam_height', type=int, default=240)
    p.add_argument('--cam_fps', type=int, default=30)
    p.add_argument('--cam_format', type=str, default='YUYV')
    p.add_argument('--num_frames', type=int, default=100)
    p.add_argument('--no_save', action='store_true')
    return p.parse_args()
                                

def set_params(opt):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, opt.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.cam_height)
    cap.set(cv2.CAP_PROP_FPS, opt.cam_fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*opt.cam_format))


def show_params(opt):
    print(f'------------------- parameters -------------------')
    print(f'         task: {opt.task}')
    print(f' weights file: {opt.weights_file}')
    if not opt.no_save:
        print(f'  destination: {opt.dstdir}')
    print(f'       device: {opt.cam_id}')
    print(f'         size: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}')
    print(f'          fps: {cap.get(cv2.CAP_PROP_FPS):.0f}')
    print(f'       format: {int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("utf-8")}')
    print(f'-'*50)

    
if __name__ == '__main__':
    opt = get_opts()
    
    opt.dstdir = os.path.join(opt.dstdir, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(opt.dstdir) and not opt.no_save:
        os.makedirs(opt.dstdir)

    if opt.task == 'classification':
        dIn, dOut = 3, 3
    elif opt.task == 'regression':
        dIn, dOut = 3, 1
    else:
        assert False, f'invalid task: {opt.task}'

    if opt.weights_file is None:
        if opt.task == 'classification':
            opt.weigths_file = 'weights/small-hvo.pkl'
        elif opt.task == 'regression':
            opt.weights_file = 'weights/small-depth.pkl'
        else:
            assert False, f'invalid task: {opt.task}'
        
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(opt.cam_id, cv2.CAP_V4L)
    assert cap.isOpened(), 'camera not found'

    set_params(opt)
    show_params(opt)

    # load model
    model = ModelUNetTinySum(dIn=dIn, dOut=dOut, numFilters=16)
    model = maybeCuda(model)
    model.loadWeights(opt.weights_file)

    i = 0
    t_begin = t = time.time()
    fps_list, t_list = [], []
    while True:
        try:
            _, frame = cap.read()
            if frame is None:
                continue

            # inference proc
            img = minMaxNormalizeData(np.float32(frame), np.min(frame), np.max(frame))
            img = np.expand_dims(img, axis=0)

            img_ = maybeCuda(torch.from_numpy(img))
            img_ = model.forward(img_)
        
            res_ = img_.detach().cpu().numpy()[0]
            res_ = minMaxNormalizeFrame(res_)

            if opt.task == 'classification':
                hvn = np.argmax(res_, axis=-1)
                tmp = np.zeros((*hvn.shape, 3), dtype=np.float32)
                tmp[np.where(hvn == 0)] = 0, 255, 0
                tmp[np.where(hvn == 1)] = 255, 0, 0
                tmp[np.where(hvn == 2)] = 0, 0, 255
            elif opt.task == 'regression':
                tmp = cm.hot(res_)[...,0:3]
            else:
                assert False, f'invalid task: {opt.task}'

            res = minMaxNormalizeFrame(tmp)

            if not opt.no_save:
                dst = os.path.join(opt.dstdir, f'{i:05d}.png')
                cv2.imwrite(dst, cv2.hconcat([frame, res]))

            t_ = time.time() - t
            fps = 1 / t_
            t = time.time()
            fps_list.append(fps)
            t_list.append(t_)

            # progress bar
            pbar = '=' * (30 * (i+1) // opt.num_frames)
            t_up = time.time() - t_begin
            t_eta = (i+1) / t_up * (opt.num_frames-i-1)
            up = time.strftime('%M:%S', time.gmtime(t_up))
            eta = time.strftime('%M:%S', time.gmtime(t_eta))

            if -1 < opt.num_frames-1 < i:
                break

            if opt.num_frames < 0:
                print(f'frame: {i+1}, uptime: {up}, {fps:.1f} fps, {1e3*t_:.0f} ms', end='\r')
            else:
                print(f'{i+1:4d}/{opt.num_frames} [{pbar:30s}] {(i+1)/opt.num_frames:4.0%} [{up}<{eta}, {fps:.1f} fps, {1e3*t_:.0f} ms] ', end='\r')

            # if -1 < opt.num_frames-2 < i:
            #     break

            i += 1

        except KeyboardInterrupt:
            break

    cap.release()
    cv2.destroyAllWindows()

    if fps_list and t_list:
        fps_list.pop(0)
        t_list.pop(0)
        print()
        print(f'-------------------- results ---------------------')
        print(f' # of frames: {len(t_list)}')
        print(f'         fps: {np.mean(fps_list):.3f}+-{np.std(fps_list, ddof=1):.4f}')
        print(f'      t_iter: {np.mean(t_list)*1e3:.0f}+-{np.std(t_list, ddof=1)*1e3:.1f} ms')
        print(f'-'*50)
    
