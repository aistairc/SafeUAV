import cv2
import os

WIDTH = 320
HEIGHT = 240


if __name__ == '__main__':
    dstdir = 'camera'
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
        
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L)
    assert cap.isOpened(), 'camera not found'

    # camera params
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    print(f' WxH: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')
    print(f' fps: {cap.get(cv2.CAP_PROP_FPS)}')

    for i in range(32):
        _, frame = cap.read()
        if frame is None:
            continue

        dst = os.path.join(dstdir, f'{i:05d}.jpg')
        cv2.imwrite(dst, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
