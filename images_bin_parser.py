import struct
import os
import config

class ImageLoader:
    def __init__(self, path):
        self.path = path
        self.images = {}

    def load(self):
        if not os.path.exists(self.path):
            print(f" {self.path} 파일 없음.")
            return {}

        images = {}
        with open(self.path, "rb") as fid:
            
            num_reg_images = struct.unpack("<Q", fid.read(8))[0] #전체 이미지 수
            print(f"총 등록된 이미지 개수: {num_reg_images}")

            for _ in range(num_reg_images):
                #ID:4, Q:32, T:24, CamID:4
                binary_header = fid.read(64)
                data = struct.unpack("<idddddddi", binary_header)
                
                image_id = data[0]
                qvec = (data[1], data[2], data[3], data[4]) # qw, qx, qy, qz
                tvec = (data[5], data[6], data[7])          # tx, ty, tz
                camera_id = data[8]

                #파일 이름 읽기, 널까지
                image_name = ""
                while True:
                    char = fid.read(1).decode("utf-8")
                    if char == "\0":
                        break
                    image_name += char

                # 2d 점 부분 뛰어넘기
                # 각 점은 (x, y, point3D_id)
                num_points2d = struct.unpack("<Q", fid.read(8))[0]
                fid.read(num_points2d * 24)

                self.images[image_id] = { #필요할 것만 저장
                    "name": image_name,
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id
                }


        return self.images

if __name__ == "__main__":
    path = config.IMAGES_BIN_PATH
    loader = ImageLoader(path)
    images = loader.load()
    print(images)
