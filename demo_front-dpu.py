"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Demonstration script for the front view only
"""

import argparse
import sys
import os
import time
import warnings
import zipfile

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import torch
import numpy as np

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.demo_dataset import Demo_KittiDataset
from models.model_utils import create_model
from utils.evaluation_utils import draw_predictions, convert_det_to_real_values
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration
from utils.demo_utils import parse_demo_configs, download_and_unzip, write_credit
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing
from utils.torch_utils import _sigmoid


from pynq_dpu import DpuOverlay


def do_detect(dpu, shapeIn, image, input_data, output_data, configs, bevmap, is_front):
    
    if not is_front:
        bevmap = torch.flip(bevmap, [1, 2])
    input_bev_maps = bevmap.unsqueeze(0).to("cpu", non_blocking=True).float()
    
   
    
    input_bev_maps = input_bev_maps.permute(0, 2, 3, 1)
    
    image[0,...] = input_bev_maps[0,...] #.reshape(shapeIn[1:])

    #print("shapeIn[1:]", shapeIn[1:])
    #print("input_bev_maps.shape", input_bev_maps.shape)
    #print("input_bev_maps.reshape", input_bev_maps.reshape(shapeIn[1:]).shape)
    
    t1 = time_synchronized() 
    
    job_id = dpu.execute_async(input_data, output_data)
    
    dpu.wait(job_id)

    t2 = time_synchronized() 

    fps = 1 / (t2 - t1)
    
    #print(output_data[0])
    
    outputs0 = torch.tensor(output_data[0])
    outputs1 = torch.tensor(output_data[1])
    outputs2 = torch.tensor(output_data[2])
    outputs3 = torch.tensor(output_data[3])
    outputs4 = torch.tensor(output_data[4])
    
    outputs0 = outputs0.permute(0, 3, 1, 2)
    outputs1 = outputs1.permute(0, 3, 1, 2)
    outputs2 = outputs2.permute(0, 3, 1, 2)
    outputs3 = outputs3.permute(0, 3, 1, 2)
    outputs4 = outputs4.permute(0, 3, 1, 2)

    outputs0 = _sigmoid(outputs0)
    outputs1 = _sigmoid(outputs1)
    
    detections = decode(
                        outputs0,
                        outputs1,
                        outputs2,
                        outputs3,
                        outputs4, K=configs.K)
        
    detections = detections.cpu().numpy().astype(np.float32)
    detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
    t2 = time_synchronized()
    # Inference speed

    return detections[0], bevmap, fps





if __name__ == '__main__':
    configs = parse_demo_configs()

    # Try to download the dataset for demonstration
    server_url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data'
    download_url = '{}/{}/{}.zip'.format(server_url, configs.foldername[:-5], configs.foldername)
    download_and_unzip(configs.dataset_dir, download_url)

    out_cap = None
    demo_dataset = Demo_KittiDataset(configs)
    
    
    overlay = DpuOverlay("dpu.bit")
    overlay.load_model("./CNN_zcu102.xmodel")
    dpu = overlay.runner
    
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()

    shapeIn = tuple(inputTensors[0].dims)
    
    #print("shapein", shapeIn)
    
    outputSize = int(outputTensors[0].get_data_size() / shapeIn[0])

    shapeOut = tuple(outputTensors[0].dims)
    shapeOut1 = tuple(outputTensors[1].dims)
    shapeOut2 = tuple(outputTensors[2].dims)
    shapeOut3 = tuple(outputTensors[3].dims)
    shapeOut4 = tuple(outputTensors[4].dims)

    #print(shapeOut,shapeOut1,shapeOut2,shapeOut3,shapeOut4)
    
    output_data = [np.empty(shapeOut, dtype=np.float32, order="C"),
                  np.empty(shapeOut1, dtype=np.float32, order="C"),
                  np.empty(shapeOut2, dtype=np.float32, order="C"),
                  np.empty(shapeOut3, dtype=np.float32, order="C"),
                  np.empty(shapeOut4, dtype=np.float32, order="C")]
    
    input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
    image = input_data[0]
    
    for sample_idx in range(len(demo_dataset)):
        metadatas, bev_map, img_rgb = demo_dataset.load_bevmap_front(sample_idx)
        detections, bev_map, fps = do_detect(dpu, shapeIn, image, input_data, output_data, configs, bev_map, is_front=True)

        # Draw prediction in the image
        bev_map = (bev_map.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        bev_map = draw_predictions(bev_map, detections, configs.num_classes)

        # Rotate the bev_map
        bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

        img_path = metadatas['img_path'][0]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        calib = Calibration(configs.calib_path)
        kitti_dets = convert_det_to_real_values(detections)
        if len(kitti_dets) > 0:
            kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
            img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)

        out_img = merge_rgb_to_bev(img_bgr, bev_map, output_width=configs.output_width)
        write_credit(out_img, (80, 210), text_author='3D Object Detection on KV260', org_fps=(80, 250), fps=fps)
        if out_cap is None:
            out_cap_h, out_cap_w = out_img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out_path = os.path.join(configs.results_dir, '{}_front.avi'.format(configs.foldername))
            print('Create video writer at {}'.format(out_path))
            out_cap = cv2.VideoWriter(out_path, fourcc, 30, (out_cap_w, out_cap_h))

        out_cap.write(out_img)

    if out_cap:
        out_cap.release()
