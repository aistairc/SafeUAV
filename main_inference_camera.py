import argparse
import cv2
import os
from matplotlib.cm import hot
from neural_wrappers.utilities import minMaxNormalizeData
from neural_wrappers.pytorch import maybeCuda
from unet_tiny_sum import ModelUNetTinySum

DEVICE_ID = 0
WIDTH = 320
HEIGHT = 240


def minMaxNormalizeFrame(frame):
    Min, Max = np.min(frame), np.max(frame)
    frame -= Min
    frame /= (Max - Min)
    frame *= 255
    return np.uint8(frame)


if __name__ == '__main__':
    dstdir = 'camera'
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
        
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(DEVICE_ID, cv2.CAP_V4L)
    assert cap.isOpened(), 'camera not found'

    # set camera params
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # load model
    model = ModelUNetTinySum(dIn=3, dOut=3, numFilters=16)
    model = maybeCuda(model)

    print(f'     device: {DEVICE_ID}')
    print(f' resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')
    print(f'        fps: {cap.get(cv2.CAP_PROP_FPS)}')

    for i in range(32):
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

        if False:
            tmp = hot(res_)[...,0:3]
        else:
            hvn = np.argmax(res_, axis=-1)
            tmp = np.zeros((*hvn.shape, 3), dtype=npfloat32)
            tmp[np.where(hvn == 0)] = 0, 255, 0
            tmp[np.where(hvn == 1)] = 255, 0, 0
            tmp[np.where(hvn == 2)] = 0, 0, 255

        res = minMaxNormalizeFrame(tmp)

        dst = os.path.join(dstdir, f'{i:05d}.jpg')
        cv2.imwrite(dst, cv2.hconcat([frame, res]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
