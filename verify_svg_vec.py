import numpy as np
import os
import glob
from config.macro import *

def verify_npy(file_path):
    print(f"Verifying {file_path}...")
    try:
        data = np.load(file_path)
        print(f"Shape: {data.shape}")
        
        # Check shape: (400, 10)
        # 4 views * 100 len = 400
        # 1 view + 1 cmd + 8 args = 10
        expected_shape = (400, 10)
        if data.shape != expected_shape:
            print(f"ERROR: Shape mismatch. Expected {expected_shape}, got {data.shape}")
            return False
            
        # Check content
        # View indices: 0, 1, 2, 3
        views = data[:, 0]
        unique_views = np.unique(views)
        print(f"Unique views: {unique_views}")
        
        # Command indices
        cmds = data[:, 1]
        unique_cmds = np.unique(cmds)
        print(f"Unique commands: {unique_cmds}")
        
        # Check for new commands
        has_arc = SVG_A_IDX in unique_cmds
        has_circle = SVG_O_IDX in unique_cmds
        print(f"Has Arc: {has_arc}")
        print(f"Has Circle: {has_circle}")
        
        return True
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return False

files = glob.glob("/workspace/Drawing2CAD/data/svg_vec/*.npy")
if files:
    verify_npy(files[0])
else:
    print("No .npy files found.")
