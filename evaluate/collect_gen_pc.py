import os
import json
import glob
import numpy as np
import h5py
from joblib import Parallel, delayed
import argparse
import sys
sys.path.append("..")
from plyfile import PlyData, PlyElement
from cadlib.visualize import vec2CADsolid, CADsolid2pc


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=None, required=True)
parser.add_argument('--n_points', type=int, default=2000)
args = parser.parse_args()

SAVE_DIR = args.src + '_pc'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def read_ply(path):
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertex = np.stack([x, y, z], axis=1)
    return vertex

def write_ply(points, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)

def process_one(path):
    data_id = path.split("/")[-1]

    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    if os.path.exists(save_path):
        return

    # print("[processing] {}".format(data_id))
    with h5py.File(path, 'r') as fp:
        out_vec = fp["vec"][:].astype(np.float64)

    try:
        shape = vec2CADsolid(out_vec)
    except Exception as e:
        print("create_CAD failed", data_id)
        return None

    try:
        out_pc = CADsolid2pc(shape, args.n_points, data_id)
    except Exception as e:
        print("convert pc failed:", data_id)
        return None
    
    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    write_ply(out_pc, save_path)

def get_test_paths(json_path, cad_vec_root):
    """
    train_val_test_split.json을 읽어 test 셋의 cad_vec 전체 경로 리스트를 반환합니다.
    """
    # JSON 파일 로드
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 'test' 스플릿 데이터 가져오기
    test_items = data.get("test", [])
    
    paths = []
    for item in test_items:
        # 항목 형식 (예: "0078/00780135")을 실제 파일 경로로 변환
        # .h5 확장자를 붙여야 실제 파일과 매칭됩니다.
        full_path = os.path.join(cad_vec_root, item + '.h5')
        paths.append(full_path)
        
    return paths

# all_paths = glob.glob(os.path.join(args.src, "*.h5"))
all_paths = get_test_paths(
    "/workspace/Drawing2CAD/data/train_val_test_split.json",
    args.src
)
Parallel(n_jobs=8, verbose=2)(delayed(process_one)(x) for x in all_paths)