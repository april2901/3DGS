import struct
import os
import config


class CameraLoader:
    def __init__(self, path):
        self.path = path
        self.cameras = {}

    def load(self):
        if not os.path.exists(self.path):
            print(f"{self.path} 파일을 찾을 수 없습니다.")
            return {}

        with open(self.path, "rb") as fid:
            #전체 카메라 개수 (8B)
            num_cameras = struct.unpack("<Q", fid.read(8))[0]
            
            for _ in range(num_cameras):
                # 고정 헤더 파싱 (ID:4, Model:4, W:8, H:8) = 24B
                camera_id, model_id, width, height = struct.unpack("<iiQQ", fid.read(24))
                
                # 파라미터 개수 결정 (PINHOLE 모델 기준 fx, fy, cx, cy)
                params = struct.unpack("<dddd", fid.read(32))
                
                self.cameras[camera_id] = {
                    "model_id": model_id,
                    "width": width,
                    "height": height,
                    "params": params
                }
        
        print(f"카메라 모델 {len(self.cameras)}개 로드 완료.")
        return self.cameras

if __name__ == "__main__":
    path = config.CAMERAS_BIN_PATH
    loader = CameraLoader(path)
    cameras = loader.load()
    print(cameras)