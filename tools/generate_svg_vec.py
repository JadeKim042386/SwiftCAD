import os
import json
import glob
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from config.macro import *

# Constants from macro.py
# SVG_COMMANDS = ['SOS', 'EOS', 'L', 'C', 'A', 'O']
# SVG_N_ARGS = 8

def parse_svg_file(file_path):
    """Parses an SVG file and returns a list of commands and arguments."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        commands = []
        args = []
        
        # Add SOS
        commands.append(SVG_SOS_IDX)
        args.append([PAD_VAL] * SVG_N_ARGS)
        
        for elem in root.iter():
            tag = elem.tag.replace(f"{{{ns['svg']}}}", "")
            
            if tag == 'path':
                d = elem.get('d')
                if not d: continue
                
                # Simple parsing of d attribute
                # Assuming format: M x y L x y ... or A ...
                # But convert_to_primitives.py output format is:
                # L x y
                # A rx ry rot large sweep x y
                # And Circle element: <circle cx cy r />
                
                parts = d.strip().split()
                i = 0
                while i < len(parts):
                    cmd = parts[i]
                    if cmd == 'M':
                        # Move to is start of path, usually we treat it as start point for next command
                        # But in our vector format, 'L' needs start and end?
                        # Wait, macro says: SVG_N_ARGS_LINE = 4 # start=(x1, y1), end=(x2, y2)
                        # So we need to track current point.
                        current_x = float(parts[i+1].split(',')[0])
                        current_y = float(parts[i+1].split(',')[1])
                        i += 2
                    elif cmd == 'L':
                        # L x,y
                        end_pt = parts[i+1].split(',')
                        end_x = float(end_pt[0])
                        end_y = float(end_pt[1])
                        
                        commands.append(SVG_L_IDX)
                        # L args: x1, y1, x2, y2, 0, 0, 0, 0
                        cmd_args = [current_x, current_y, end_x, end_y, 0, 0, 0, 0]
                        args.append(cmd_args)
                        
                        current_x = end_x
                        current_y = end_y
                        i += 2
                    elif cmd == 'A':
                        # A rx,ry rot large sweep x,y
                        rx = float(parts[i+1].split(',')[0])
                        ry = float(parts[i+1].split(',')[1])
                        rot = float(parts[i+2])
                        large = float(parts[i+3])
                        sweep = float(parts[i+4])
                        end_pt = parts[i+5].split(',')
                        end_x = float(end_pt[0])
                        end_y = float(end_pt[1])
                        
                        commands.append(SVG_A_IDX)
                        # A args: rx, ry, rot, large, sweep, x, y, 0
                        cmd_args = [rx, ry, rot, large, sweep, end_x, end_y, 0]
                        args.append(cmd_args)
                        
                        current_x = end_x
                        current_y = end_y
                        i += 6
                    elif cmd == 'Z':
                        i += 1
                    else:
                        i += 1

            elif tag == 'line':
                x1 = float(elem.get('x1'))
                y1 = float(elem.get('y1'))
                x2 = float(elem.get('x2'))
                y2 = float(elem.get('y2'))
                
                commands.append(SVG_L_IDX)
                cmd_args = [x1, y1, x2, y2, 0, 0, 0, 0]
                args.append(cmd_args)

            elif tag == 'circle':
                cx = float(elem.get('cx'))
                cy = float(elem.get('cy'))
                r = float(elem.get('r'))
                
                commands.append(SVG_O_IDX)
                # O args: cx, cy, r, 0, 0, 0, 0, 0
                cmd_args = [cx, cy, r, 0, 0, 0, 0, 0]
                args.append(cmd_args)
        
        # Add EOS
        commands.append(SVG_EOS_IDX)
        args.append([PAD_VAL] * SVG_N_ARGS)
        
        return commands, args

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return [], []

def quantize_args(args):
    """Quantizes arguments to 0-255 range."""
    # Assuming coordinates are in some range, need to normalize?
    # macro.py says ARGS_DIM = 256
    # bi_sequence_dataset.py loads them as long tensors, suggesting they are indices.
    # But wait, SVG coordinates are floats.
    # The model embeds them: self.args_embed = nn.Embedding(args_dim, 64, padding_idx=0)
    # So they MUST be integers.
    # How were they quantized originally?
    # Looking at bi_sequence_dataset.py:
    # data = np.load(npy_path) ... args_vec = data[..., 2:]
    # It seems the .npy file already contains quantized integers.
    # We need to know the quantization scale.
    # Let's assume 0-255 for now, mapping 0.0-1.0 to 0-255?
    # Or maybe the coordinates are already integers?
    # The converted SVGs have floats like 0.1234.
    # We need to check how the original data was quantized.
    # Since we don't have that info, let's look at a sample .npy file if possible.
    # But I can't read .npy directly.
    # Let's assume standard quantization: val * 255.
    # But we need to know the range.
    # Let's assume the SVG viewbox is 0-1 or similar.
    # If not, we might need to normalize per file or globally.
    
    # For now, let's just cast to int if they are large, or scale if small.
    # Let's assume they are in [0, 255] range effectively?
    # Wait, macro.py: CAD_NORM_FACTOR = 0.75
    # Let's check a converted SVG file content to see coordinate range.
    pass
    return args

def process_data(data_root, output_root):
    split_path = os.path.join(data_root, "train_val_test_split.json")
    with open(split_path, "r") as f:
        split_data = json.load(f)
    
    all_ids = []
    for phase in ['train', 'validation', 'test']:
        all_ids.extend(split_data[phase])
    
    # Remove duplicates
    all_ids = sorted(list(set(all_ids)))
    
    # DEBUG: Limit to 10 items
    all_ids = all_ids[:10]
    
    print(f"Found {len(all_ids)} data IDs.")
    
    os.makedirs(output_root, exist_ok=True)
    
    views = ['Front', 'Top', 'Right', 'FrontTopRight']
    
    for data_id in tqdm(all_ids):
        # Construct file paths
        # ID format: 00000007 -> 0000/00000007
        subdir = data_id[:4]
        svg_dir = os.path.join(data_root, "svg_raw_convertion", subdir, data_id)
        
        stacked_data = []
        
        for view_idx, view_name in enumerate(views):
            svg_file = os.path.join(svg_dir, f"{data_id}_{view_name}.svg")
            
            if not os.path.exists(svg_file):
                # print(f"Warning: {svg_file} not found. Padding with EOS.")
                cmds = [SVG_SOS_IDX, SVG_EOS_IDX]
                args_list = [[PAD_VAL]*SVG_N_ARGS, [PAD_VAL]*SVG_N_ARGS]
            else:
                cmds, args_list = parse_svg_file(svg_file)
            
            # Quantization (Placeholder logic - needs verification)
            # Assuming coordinates are 0-255 compatible integers for now
            # or we scale them.
            # Let's check one file content first.
            
            # Pad to SVG_MAX_TOTAL_LEN (100)
            pad_len = SVG_MAX_TOTAL_LEN - len(cmds)
            if pad_len > 0:
                cmds.extend([PAD_VAL] * pad_len)
                args_list.extend([[PAD_VAL]*SVG_N_ARGS] * pad_len)
            else:
                cmds = cmds[:SVG_MAX_TOTAL_LEN]
                args_list = args_list[:SVG_MAX_TOTAL_LEN]
            
            # Create (100, 2 + 8) array
            # Col 0: View Index (0-3)
            # Col 1: Command Index
            # Col 2-9: Args
            
            view_col = np.full((SVG_MAX_TOTAL_LEN, 1), view_idx)
            cmd_col = np.array(cmds).reshape(-1, 1)
            args_col = np.array(args_list)
            
            # Quantize args to int
            # Based on inspection, coordinates are in 0-256 range (e.g. 216.0000)
            # So we can directly cast to int.
            args_col = np.round(args_col).astype(int)
            
            # Clip to [0, 255] just in case
            args_col = np.clip(args_col, 0, 255)
            
            view_data = np.hstack([view_col, cmd_col, args_col])
            stacked_data.append(view_data)
            
        # Stack 4 views: (400, 10)
        final_data = np.vstack(stacked_data)
        
        save_path = os.path.join(output_root, f"{data_id}.npy")
        print(f"Saving to {save_path}")
        np.save(save_path, final_data)

if __name__ == "__main__":
    data_root = "/workspace/Drawing2CAD/data"
    output_root = "/workspace/Drawing2CAD/data/svg_vec_conversion"
    process_data(data_root, output_root)
