import os
import cv2
import time
import argparse

import torch
import model.detector
import utils.utils
#python3 test.py --data coco.data --weights model.pth --img IMG_3739.JPG
if __name__ == '__main__':
    # Specify the training configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='',
                        help='Specify the training profile *.data')
    parser.add_argument('--weights', type=str, default='',
                        help='The path of the .pth model to be transformed')
    parser.add_argument('--img', type=str, default='',
                        help='The path of the test image')

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    assert os.path.exists(opt.weights), "Please specify the correct model path"
    assert os.path.exists(opt.img), "Please specify the correct test image path"

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))

    # Set the module in evaluation mode
    model.eval()

    # Data preprocessing
    ori_img = cv2.imread(opt.img)
    res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
    img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0

    # Model inference
    start = time.perf_counter()
    preds = model(img)
    end = time.perf_counter()
    time_taken = (end - start) * 1000.  # in milliseconds
    print("Forward time: %f ms" % time_taken)

    # Post-process the feature maps
    output = utils.utils.handel_preds(preds, cfg, device)
    output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)

    # Load label names
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())

    h, w, _ = ori_img.shape
    scale_h, scale_w = h / cfg["height"], w / cfg["width"]

    # Draw predicted bounding boxes
    for box in output_boxes[0]:
        box = box.tolist()

        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

    cv2.imwrite("test_result.png", ori_img)
