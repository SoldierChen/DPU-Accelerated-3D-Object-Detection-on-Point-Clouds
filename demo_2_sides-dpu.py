import sys
import os
import warnings

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
from utils.visualization_utils import show_rgb_image_with_boxes
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
    #for sample_idx in range(10):  
        print(sample_idx)
        
        metadatas, front_bevmap, back_bevmap, img_rgb = demo_dataset.load_bevmap_front_vs_back(sample_idx)

        front_detections, front_bevmap, fps = do_detect(dpu, shapeIn, image, input_data, output_data, configs, front_bevmap, is_front=True)
        back_detections, back_bevmap, _ = do_detect(dpu, shapeIn, image, input_data, output_data, configs, back_bevmap, is_front=False)

        # Draw prediction in the image
        front_bevmap = (front_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        front_bevmap = cv2.resize(front_bevmap, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        front_bevmap = draw_predictions(front_bevmap, front_detections, configs.num_classes)
        # Rotate the front_bevmap
        front_bevmap = cv2.rotate(front_bevmap, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Draw prediction in the image
        back_bevmap = (back_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        back_bevmap = cv2.resize(back_bevmap, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        back_bevmap = draw_predictions(back_bevmap, back_detections, configs.num_classes)
        # Rotate the back_bevmap
        back_bevmap = cv2.rotate(back_bevmap, cv2.ROTATE_90_CLOCKWISE)

        # merge front and back bevmap
        full_bev = np.concatenate((back_bevmap, front_bevmap), axis=1)

        img_path = metadatas['img_path'][0]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        calib = Calibration(configs.calib_path)
        kitti_dets = convert_det_to_real_values(front_detections)
        if len(kitti_dets) > 0:
            kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
            img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)
        img_bgr = cv2.resize(img_bgr, (cnf.BEV_WIDTH * 2, 375))

        out_img = np.concatenate((img_bgr, full_bev), axis=0)
        write_credit(out_img, (50, 410), text_author='3D Object Detection on KV260', org_fps=(900, 410), fps=fps*2)

        if out_cap is None:
            out_cap_h, out_cap_w = out_img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out_path = os.path.join(configs.results_dir, '{}_both_2_sides_final.avi'.format(configs.foldername))
            print('Create video writer at {}'.format(out_path))
            out_cap = cv2.VideoWriter(out_path, fourcc, 30, (out_cap_w, out_cap_h))

        out_cap.write(out_img)

    if out_cap:
        out_cap.release()
