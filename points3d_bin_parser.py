import struct
import config

class Point3DLoader:
    def __init__(self, path):
        self.path = path
        self.points = {}
    

    def load(self):
        with open(self.path, "rb") as fid:
            # 전체 점 개수
            num_points = struct.unpack("<Q", fid.read(8))[0]
            print(f"총 점 개수: {num_points}")

            
            for _ in range(num_points):
                # 고정 43바이트 읽기
                binary_data = fid.read(43)
                p_id, x, y, z, r, g, b, error = struct.unpack("<QdddBBBd", binary_data)
                
                track_len = struct.unpack("<Q", fid.read(8))[0]
                fid.read(track_len * 8) 
                
                self.points[p_id] = {
                    "xyz": (x, y, z),
                    "rgb": (r, g, b),
                    "error": error
                }
                
        return self.points

if __name__ == "__main__":
    path = config.POINTS3D_BIN_PATH
    loader = Point3DLoader(path)
    points = loader.load()
    print(points)