# config.py
import os

OBJECT_NAME = "bicycle" 

IMAGE_PATH = f"./360_v2/{OBJECT_NAME}"
OUTPUT_PATH = f"./360_v2/{OBJECT_NAME}_output"
CAMERA_JSON_PATH = OUTPUT_PATH+"cameras.json"

CAMERAS_BIN_PATH=f"./360_v2/{OBJECT_NAME}/sparse/0/cameras.bin"
POINTS3D_BIN_PATH=f"./360_v2/{OBJECT_NAME}/sparse/0/points3D.bin"
IMAGES_BIN_PATH=f"./360_v2/{OBJECT_NAME}/sparse/0/images.bin"