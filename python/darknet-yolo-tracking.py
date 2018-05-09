#!/usr/bin/python

from ctypes import *
import math
import random
import sys
from sort import *
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.offsetbox as offsetbox
import matplotlib
from skimage import io
import cv2


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
print(sys.argv[0])
lib = CDLL(os.path.join(os.path.dirname(sys.argv[0]), "../libdarknet.so"), RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE


def np_array_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)

    return image

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, np_image, thresh=.5, hier_thresh=.5, nms=.45):
    im = np_array_to_image(np_image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x - b.w/2, b.y - b.w/2, b.x + b.w/2, b.y + b.h/2)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    if len(sys.argv) < 5:
        print("Usage: darknet <data cfg> <net cfg> <weights> <input video> [<framerate divisor>]")
        sys.exit(1)

    data_cfg = sys.argv[1]
    net_cfg = sys.argv[2]
    net_weights = sys.argv[3]
    input_filename = sys.argv[4]
    frame_div = 1
    if len(sys.argv) > 5:
        frame_div = int(sys.argv[5])

#    fig = plt.figure()
    colours = np.random.random_integers(0, 255, (32,3))

    mot_tracker = Sort(max_age=4, min_hits=1)

    net = load_net(net_cfg, net_weights, 0)
    meta = load_meta(data_cfg)

    output_template = "output/frame-%05d.jpg"


    cap = cv2.VideoCapture(input_filename)

    seq = 1

    print(input_filename)

    last_matched = {}
    track_history = {}

    red = (0, 0, 255)

    while cap.isOpened():
        ret, orig_frame = cap.read()

        if seq % frame_div != 0:
            seq += 1
            continue

        out_file = output_template % (seq)

        print('Processing frame %s...' % (seq))

        h, w = orig_frame.shape[:2]

        if h > 540:
            factor = 540.0 / h
            h = int(factor * h)
            w = int(factor * w)
            frame = cv2.resize(orig_frame, (w, h))
        else:
            frame = orig_frame


        print(orig_frame.shape[:2])
        print(frame.shape[:2])

        r = detect(net, meta, frame)

        dets = []
        for detection in r:
            name, prob, bbox = detection
            dets.append([bbox[0], bbox[1], bbox[2], bbox[3], prob])

        np_dets = np.array(dets)
        print("Detected %d objects" % (len(np_dets)))
#        print np.array2string(np_dets, precision=4)
        trackers, unmatched = mot_tracker.update(np_dets)
        print("Tracked %d objects" % (len(trackers)))
#        print np.array2string(trackers, precision=4)
        print("%d tracked objects are not detected" % (len(unmatched)))
        print(unmatched)

        det_frame = frame.copy()
#        cv2.imshow('frame', frame)

#        for d in np_dets:
#            d = d.astype(np.int32)
#            cv2.rectangle(det_frame,(d[0],d[1]), (d[2], d[3]), red, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for d in trackers:
            d = d.astype(np.int32)

            color = colours[d[4] % 32]

            if track_history.get(d[4]) is None:
                track_history[d[4]] = []
            track_history[d[4]].append((int(d[0] + (d[2] - d[0])/2), int(d[1] + (d[3] - d[1])/2)))

            cv2.rectangle(det_frame,(d[0],d[1]), (d[2], d[3]), colours[d[4] % 32], 1)
            cv2.putText(det_frame, str(d[4]), (d[0], d[1] - 2), font, 0.5, colours[d[4] % 32], 1)

        for t in track_history:
            for p in track_history[t]:
                cv2.circle(det_frame, (p[0], p[1]), 1, color, -1)



#        cv2.imshow('det_frame', det_frame)

#        cv2.waitKey(100)

        print("Saving to %s" % (out_file))
        cv2.imwrite(out_file, det_frame)



#        ax1 = fig.add_subplot(2, 2, 2)
#        ax2 = fig.add_subplot(2, 2, 1)
#        ax3 = fig.add_subplot(2, 2, 3)
##        im = io.imread(filename)
#        im = frame
#        ax1.imshow(im)
#        ax2.imshow(im)
#        ax3.imshow(im)
#
#        for d in np_dets:
#            d = d.astype(np.int32)
#            ax2.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,ec='pink',lw=1))
##            ax2.set_adjustable('box')
#
#        for d in trackers:
#            d = d.astype(np.int32)
#            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=0.5,ec=colours[d[4]%32,:]))
#            ax1.text(d[0], d[1]-1, str(d[4]), size='xx-small', color=colours[d[4]%32,:])
##            ax1.set_adjustable('box')
#            if last_matched.get(d[4]) is not None:
#                last_d = last_matched[d[4]]
#                x1 = last_d[0] + (last_d[2] - last_d[0])/2
#                y1 = last_d[1] + (last_d[3] - last_d[1])/2
#                x2 = d[0] + (d[2] - d[0])/2
#                y2 = d[1] + (d[3] - d[1])/2
#                ax3.add_line(matplotlib.lines.Line2D((x1, x2), (y1, y2), linewidth=0.5, c=colours[d[4]%32,:]))
#            last_matched[d[4]] = d
#
#        for d in unmatched:
#            d = d.astype(np.int32)
#            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=0.5,ec='gray'))
#            ax1.text(d[0], d[1]-1, str(d[4]), size='xx-small', color=colours[d[4]%32,:])
#
#
#        print("Saving to %s" % (out_file))
#        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#        plt.savefig(out_file, dpi=300)
#        ax1.cla()
#        ax2.cla()
#
        seq += 1
        latest_matched = trackers



