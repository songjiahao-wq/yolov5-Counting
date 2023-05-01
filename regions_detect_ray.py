# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
通过向右射线相交的数量进行判断行人是否在区域内
"""

import argparse
import os
import sys
from pathlib import Path

import imutils
import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    ray_classes = 2 #选择安全区域类型，1为一个安全区域，2为两个
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()

        # mask for certain region
        # # 1,2,3,4 分别对应左上，右上，右下，左下四个点
        # hl1 = 4.2 / 10  # 监测区域高度距离图片顶部比例
        # wl1 = 1.6 / 10  # 监测区域高度距离图片左部比例
        # hl2 = 2.5 / 10  # 监测区域高度距离图片顶部比例
        # wl2 = 6.8 / 10  # 监测区域高度距离图片左部比例
        # hl4 = 9.9 / 10  # 监测区域高度距离图片顶部比例
        # wl4 = 2.5 / 10  # 监测区域高度距离图片左部比例
        # hl3 = 7.4 / 10  # 监测区域高度距离图片顶部比例
        # wl3 = 9.4 / 10  # 监测区域高度距离图片左部比例
        poly1 = [[933, 103], [209, 446], [1788, 977], [1878, 232], [935, 103]] #四边形区域1
        poly2 = [[1121, 217], [751, 512], [1372, 773], [1624, 335], [1122, 218]] #四边形区域2

        pts = [poly1, poly2]
        if ray_classes ==1:
            ima_w = 480
            ima_h = 270

            # 1,2,3,4 分别对应左上，右上，右下，左下四个点
            x11, y11, x12, y12, x13, y13, x14, y14 = 947, 108, 235, 457, 818, 856, 1750, 179

            hl1 = round(y11 / ima_h, 2)  # 监测区域高度距离图片顶部比例
            wl1 = round(x11 / ima_w, 2)  # 监测区域高度距离图片左部比例
            hl2 = round(y12 / ima_h, 2)  # 监测区域高度距离图片顶部比例
            wl2 = round(x12 / ima_w, 2)  # 监测区域高度距离图片左部比例
            hl3 = round(y13 / ima_h, 2)  # 监测区域高度距离图片顶部比例
            wl3 = round(x13 / ima_w, 2)  # 监测区域高度距离图片左部比例
            hl4 = round(y14 / ima_h, 2)  # 监测区域高度距离图片顶部比例
            wl4 = round(x14 / ima_w, 2)  # 监测区域高度距离图片左部比例
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            # if webcam:  # batch_size >= 1
            #     p, im0, frame = path[i], im0s[i].copy(), dataset.count
            #     s += f'{i}: '
            # else:
            #     p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            # *************************************************************
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                if ray_classes==1:
                    cv2.putText(im0, "Detection_Region", (int(im0.shape[1] * wl1 - 5), int(im0.shape[0] * hl1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (255, 255, 0), 2, cv2.LINE_AA)

                    pts = np.array([[int(im0.shape[1] * wl1), int(im0.shape[0] * hl1)],  # pts1
                                    [int(im0.shape[1] * wl2), int(im0.shape[0] * hl2)],  # pts2
                                    [int(im0.shape[1] * wl3), int(im0.shape[0] * hl3)],  # pts3
                                    [int(im0.shape[1] * wl4), int(im0.shape[0] * hl4)]], np.int32)  # pts4

                    # pts = pts.reshape((-1, 1, 2))
                    zeros = np.zeros((im0.shape), dtype=np.uint8)
                    mask = cv2.fillPoly(zeros, [pts], color=(0, 165, 255))
                    im0 = cv2.addWeighted(im0, 1, mask, 0.2, 0)
                    cv2.polylines(im0, [pts], True, (255, 255, 0), 3)
                # plot_one_box(dr, im0, label='Detection_Region', color=(0, 255, 0), line_thickness=2)
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                if ray_classes == 1:
                    cv2.putText(im0, "Detection_Region", (int(im0.shape[1] * wl1 - 5), int(im0.shape[0] * hl1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (255, 255, 0), 2, cv2.LINE_AA)
                    pts = np.array([[int(im0.shape[1] * wl1), int(im0.shape[0] * hl1)],  # pts1
                                    [int(im0.shape[1] * wl2), int(im0.shape[0] * hl2)],  # pts2
                                    [int(im0.shape[1] * wl3), int(im0.shape[0] * hl3)],  # pts3
                                    [int(im0.shape[1] * wl4), int(im0.shape[0] * hl4)]], np.int32)  # pts4
                    # pts = pts.reshape((-1, 1, 2))
                    zeros = np.zeros((im0.shape), dtype=np.uint8)
                    mask = cv2.fillPoly(zeros, [pts], color=(0, 165, 255))
                    im0 = cv2.addWeighted(im0, 1, mask, 0.2, 0)

                    cv2.polylines(im0, [pts], True, (255, 255, 0), 3)
            #     ********************************************************

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        print(torch.tensor(xyxy).view(1, 4))
                        pos = torch.tensor(xyxy).view(1, 4).tolist()[0]#脚底左下角做定位
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()#中心点做定位


                        people_pos = [pos[0],pos[3]]
                        cv2.circle(im0, (int(pos[0]),int(pos[3])), 3, (0, 0, 255),15)
                        print(people_pos)
                        print('***************')
                        annotator.box_label(xyxy, label, color=(56, 56, 56))
                        print(colors(c,True))

                        is_in = is_in_poly(people_pos, pts)
                        if is_in:
                            annotator.box_label(xyxy, label, color=(0, 0, 255))

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            # 在帧上绘制多边形
            for poly in pts:
                if len(poly) > 1:
                    cv2.polylines(im0, [np.array(poly)], isClosed=False, color=(0, 255, 0), thickness=2)
            # Stream results
            im0 = annotator.result()
            if view_img:
                # if p not in windows:
                #     windows.append(p)
                #     # cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                #     cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # im0 = imutils.resize(im0, width=720)
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

# def is_in_poly(p, poly):
#     """
#     :param p: [x, y]
#     :param poly: [[], [], [], [], ...]
#     :return:
#     """
#     px, py = p
#     is_in = False
#     for i, corner in enumerate(poly):
#         next_i = i + 1 if i + 1 < len(poly) else 0
#         x1, y1 = corner
#         x2, y2 = poly[next_i]
#         if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex 如果点在顶点上
#             is_in = True
#             break
#         if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon 找到多边形的水平边缘
#             x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
#             if x == px:  # if point is on edge 如果点在边缘上
#                 is_in = True
#                 break
#             elif x > px:  # if point is on left-side of line 如果点在线的左侧
#                 is_in = not is_in
#     return is_in
def is_in_poly(p, polys):
    """
    :param p: [x, y]
    :param polys: [[[x1, y1], [x2, y2], ...], [[x1, y1], [x2, y2], ...]]
    :return:
    """
    px, py = p
    is_in = False

    for poly in polys:
        for i, corner in enumerate(poly):
            next_i = i + 1 if i + 1 < len(poly) else 0
            x1, y1 = corner
            x2, y2 = poly[next_i]
            if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
                is_in = True
                break
            if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
                x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                if x == px:  # if point is on edge
                    is_in = True
                    break
                elif x > px:  # if point is on left-side of line
                    is_in = not is_in



    return is_in
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / r'D:\my_job\DATA\data/test.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True,action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', default='0',nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', default=True,action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
