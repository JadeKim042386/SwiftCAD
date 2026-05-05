import os
import xml.etree.ElementTree as ET
import numpy as np
import glob
import random
from svgpathtools import parse_path
from convert_to_primitives import process_file

def get_path_points(path_string, num_points=100):
    try:
        path = parse_path(path_string)
        return [path.point(t) for t in np.linspace(0, 1, num_points)]
    except:
        return []

def min_dist_to_path(point, path):
    if not path: return float('inf')
    return min([abs(point - p) for p in path])

def verify_geometric_accuracy(original_file, converted_file):
    print(f"Verifying {os.path.basename(original_file)}...")
    
    tree_orig = ET.parse(original_file)
    tree_conv = ET.parse(converted_file)
    
    paths_orig = [elem.get('d') for elem in tree_orig.findall('.//{http://www.w3.org/2000/svg}path') if elem.get('d')]
    paths_conv = [elem.get('d') for elem in tree_conv.findall('.//{http://www.w3.org/2000/svg}path') if elem.get('d')]
    
    # Collect all converted points
    all_points_conv = []
    for d in paths_conv:
        all_points_conv.extend(get_path_points(d, num_points=200))
    
    # Add points from lines
    lines_conv = tree_conv.findall('.//{http://www.w3.org/2000/svg}line')
    for line in lines_conv:
        p1 = complex(float(line.get('x1')), float(line.get('y1')))
        p2 = complex(float(line.get('x2')), float(line.get('y2')))
        for t in np.linspace(0, 1, 50):
            all_points_conv.append(p1 + (p2-p1)*t)
            
    # Add points from circles
    circles_conv = tree_conv.findall('.//{http://www.w3.org/2000/svg}circle')
    for circle in circles_conv:
        cx = float(circle.get('cx'))
        cy = float(circle.get('cy'))
        r = float(circle.get('r'))
        # Sample circle
        for t in np.linspace(0, 2*np.pi, 100):
            all_points_conv.append(complex(cx + r*np.cos(t), cy + r*np.sin(t)))

    if not all_points_conv:
        print("  No converted geometry found.")
        return False

    total_max_error = 0
    all_errors = []
    
    for i, d_orig in enumerate(paths_orig):
        points_orig = get_path_points(d_orig, num_points=50)
        if not points_orig: continue
        
        errors = [min_dist_to_path(p, all_points_conv) for p in points_orig]
        max_error = max(errors) if errors else 0
        all_errors.extend(errors)
        total_max_error = max(total_max_error, max_error)
        
    global_mean_error = np.mean(all_errors) if all_errors else 0
    print(f"  Global Max Error: {total_max_error:.4f}")
    print(f"  Global Mean Error: {global_mean_error:.4f}")
    
    if total_max_error < 1.0:
        print("  ✅ Passed")
        return True
    else:
        print("  ⚠️ Failed (Error > 1.0)")
        return False

def verify_multiple_files():
    input_dir = "/workspace/Drawing2CAD/data/svg_raw"
    output_dir = "/workspace/Drawing2CAD/verification_output"
    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(input_dir, "**/*.svg"), recursive=True)
    if len(files) > 10:
        files = random.sample(files, 10)
    
    print(f"Verifying {len(files)} files...")
    
    passed_count = 0
    for i, input_file in enumerate(files):
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"verified_{i}_{filename}")
        
        success = process_file(input_file, output_file)
        if success:
            if verify_geometric_accuracy(input_file, output_file):
                passed_count += 1
        else:
            print(f"  Failed to process {filename}")
            
    print(f"\nSummary: {passed_count}/{len(files)} passed.")

if __name__ == "__main__":
    verify_multiple_files()
