import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, plot_polylines
from utils.torch_utils import select_device, load_classifier, time_synchronized
from calculate_IoU import bb_intersection_over_union
from calculate_IoU_polygon import polygon_intersection_over_union

import numpy as np

def detect(opt, save_img=False):
    # source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, True, opt.img_size
    save_img = not opt.nosave and not source.endswith(
        '.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                                   exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                ), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + \
                ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                plist = []
                clist = []
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        # label中加入推理时间
                        infer_time = t2 - t1
                        # label format
                        line = (
                            # cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                            cls, *xywh, conf, infer_time) if True else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        # !!!change labels format
                        # label = f'{names[int(cls)]} {conf:.2f}'
                        label = f'{names[int(cls)][:1]}'
                        # print('!!!label', label)
                        # print('!!!xyxy', xyxy)
                        if label != 'c' and opt.weights == 'weights/bestexp7.pt':
                            # 不画斑马线
                            plot_one_box(xyxy, im0, label=label,
                                        color=colors[int(cls)], line_thickness=3)
                        # points = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        points = [xyxy[0], xyxy[1], xyxy[2], xyxy[3]]
                        # print('!!!points', points)
                        if label == 'c':
                            clist.append(points)
                        elif label == 'p':
                            plist.append(points)
                print('!!!clist', clist)
                print('!!!plist', plist)
                # crosswalk (225,194) (1032,194) (1032,262) (5,238)
                # crosswalk [225,194,1032,194,1032,262,5,238]
                # driveway  (813,0) (1065,0) (1032,194) (225,194)
                # driveway  [813,0,1065,0,1032,194,225,194]
                # 确保仪表识别时不绘制斑马线区域和车道区域
                if opt.weights == 'weights/bestexp7.pt':
                    points_crosswalk = np.array([[228,194],[1032,194],[1032,262],[5,238]], np.int32)
                    points_crosswalk = points_crosswalk.reshape((-1,1,2))
                    plot_polylines(points_crosswalk, im0, label='crosswalk', color=[255, 0, 0], line_thickness=3)
                    points_driveway = np.array([[813,0],[1065,0],[1032,194],[228,194]], np.int32)
                    points_driveway = points_driveway.reshape((-1,1,2))
                    plot_polylines(points_driveway, im0, label='driveway', color=[0, 255, 0], line_thickness=3)
                    for p in plist:
                    # iou = bb_intersection_over_union(clist[0], p)
                    line_crosswalk = [228,194,1032,194,1032,262,5,238] # crosswalk
                    line_driveway  = [813,0,1065,0,1032,194,228,194]   # driveway
                    line2 = [int(p[0]),int(p[1]),int(p[2]),int(p[1]),
                             int(p[0]),int(p[3]),int(p[2]),int(p[3])]
                    p1 = p[1]
                    # 只看bbox最下端即脚部分p_feet
                    p_feet = p
                    p_feet[1] = p_feet[3] - 20
                    line2_feet = [int(p_feet[0]),int(p_feet[1]),int(p_feet[2]),int(p_feet[1]),
                             int(p_feet[0]),int(p_feet[3]),int(p_feet[2]),int(p_feet[3])]
                    iou_crosswalk = polygon_intersection_over_union(line_crosswalk, line2)
                    iou_crosswalk_feet = polygon_intersection_over_union(line_crosswalk, line2_feet)
                    iou_driveway = polygon_intersection_over_union(line_driveway, line2)
                    iou_driveway_feet = polygon_intersection_over_union(line_driveway, line2_feet)
                    print('iou_crosswalk', iou_crosswalk)
                    print('iou_crosswalk_feet', iou_crosswalk_feet)
                    print('iou_driveway', iou_driveway)
                    print('iou_driveway_feet', iou_driveway_feet)
                    if iou_crosswalk > 0:
                        print('!!!p', p)
                        plot_one_box(p_feet, im0, label='feet',
                                     color=[0, 0, 0], line_thickness=3)
                        p[1] = p1
                        print('!!!after p', p)
                        if iou_crosswalk_feet > 0 and iou_driveway_feet == 0:
                            # 只有在与crosswalk有交集和driveway无交集才绘制
                            plot_one_box(p, im0, label='crossing',
                                         color=[0, 0, 255], line_thickness=3)
                    if iou_driveway > 0:
                        print('!!!p', p)
                        p_feet[1] = p_feet[3] - 20
                        plot_one_box(p_feet, im0, label='feet',
                                     color=[0, 0, 0], line_thickness=3)
                        p[1] = p1
                        print('!!!after p', p)
                        if iou_driveway_feet > 0:
                            plot_one_box(p, im0, label='warning',
                                         color=[255, 0, 255], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}time inference + NMS Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            # save_path += '.mp4'
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        print('!!!vid save_path', save_path)
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
                        # vid_writer = cv2.VideoWriter(
                        #     # save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        #     # 参考 https://xugaoxiang.com/2021/08/20/opencv-h264-videowrite
                        #     save_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
                    # print('vid_writer', vid_writer)
                    vid_writer.write(im0)
                    # print('vid_writer done')

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'All done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='weights/yolov5s.pt', help='model.pt path(s)')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()