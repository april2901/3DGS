import json
import os
import numpy as np
from cameras_bin_parser import CameraLoader
from images_bin_parser import ImageLoader
import config


def save_camera_json(cameras, images, output_path=config.CAMERA_JSON_PATH):
    json_data = []
    
    print(f"save camera.json")
    
    for img_id in images:
        img = images[img_id] # {'name': '....JPG', 'qvec': ( , , , ), 'tvec': ( , , ), 'camera_id': 1}
        cam = cameras[img["camera_id"]] # cameras[1] = {'model_id': 1, 'width': , 'height': , 'params': (, , , )}
        
        # 3DGS 학습에 필요한 최소한의 카메라 파라미터 구성
        cam_dict = {
            "id": img["camera_id"],
            "img_name": img["name"],
            "width": cam["width"],
            "height": cam["height"],
            
            "fx": float(cam["params"][0]),
            "fy": float(cam["params"][0]),
            "cx": float(cam["params"][1]),
            "cy": float(cam["params"][2]),
            
            "rotation": list(img["qvec"]),
            "position": list(img["tvec"])
        }
        json_data.append(cam_dict)
    # JSON 파일로 저장
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)
        
    print(f"{len(json_data)}개의 카메라 정보가 {output_path}에 저장되었습니다.")


cameras = CameraLoader(config.CAMERAS_BIN_PATH).load()
images = ImageLoader(config.IMAGES_BIN_PATH).load()

save_camera_json(cameras, images)