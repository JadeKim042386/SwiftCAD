import os
import xml.etree.ElementTree as ET
import numpy as np
from svgpathtools import parse_path, CubicBezier, Line, Arc, Path
import glob
from tqdm import tqdm
import traceback

def is_collinear(p0, p1, p2, tolerance=1e-5):
    """Check if three points are collinear."""
    val = (p1.imag - p0.imag) * (p2.real - p1.real) - (p1.real - p0.real) * (p2.imag - p1.imag)
    return abs(val) < tolerance

def get_arc_error(segment, center, radius, tolerance):
    d1 = abs(abs(segment.control1 - center) - radius)
    d2 = abs(abs(segment.control2 - center) - radius)
    mid = segment.point(0.5)
    dm = abs(abs(mid - center) - radius)
    return max(d1, d2, dm)

def fit_arc(p0, p3, mid_curve):
    x1, y1 = p0.real, p0.imag
    x2, y2 = mid_curve.real, mid_curve.imag
    x3, y3 = p3.real, p3.imag
    
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-6: return None

    cx = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    cy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    
    center = complex(cx, cy)
    radius = abs(p0 - center)
    return center, radius

def bezier_to_arcs_recursive(segment, tolerance=0.1):
    p0 = segment.start
    p3 = segment.end
    
    if is_collinear(p0, segment.control1, p3) and is_collinear(p0, segment.control2, p3):
        return [('L', p3)]

    mid_curve = segment.point(0.5)
    fit = fit_arc(p0, p3, mid_curve)
    
    if fit:
        center, radius = fit
        error = get_arc_error(segment, center, radius, tolerance)
        
        if error < tolerance:
            x1, y1 = p0.real, p0.imag
            x2, y2 = mid_curve.real, mid_curve.imag
            x3, y3 = p3.real, p3.imag
            area = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
            sweep_flag = 1 if area > 0 else 0 
            large_arc_flag = 0
            return [('A', radius, radius, 0, large_arc_flag, sweep_flag, p3)]
            
    seg1, seg2 = segment.split(0.5)
    return bezier_to_arcs_recursive(seg1, tolerance) + bezier_to_arcs_recursive(seg2, tolerance)

def is_path_closed(subpath, tolerance=1e-6):
    """Check if a path is closed with tolerance, avoiding svgpathtools assertion."""
    if len(subpath) == 0: return False
    return abs(subpath[0].start - subpath[-1].end) < tolerance

def convert_subpath_to_element(subpath, tolerance=0.1):
    """Convert a continuous subpath to an element."""
    # 1. Check for Line (single segment or multiple collinear)
    is_line = True
    for seg in subpath:
        if isinstance(seg, Line): continue
        if isinstance(seg, CubicBezier):
            if not (is_collinear(seg.start, seg.control1, seg.end) and is_collinear(seg.start, seg.control2, seg.end)):
                is_line = False; break
        else:
            is_line = False; break
            
    if is_line:
        # It's a line or polyline. If it's a single segment, <line>
        # If multiple, <polyline> or multiple <line>?
        # User asked for "Line" element. <line> is standard.
        # If it's a polyline, we can break it into multiple <line>s or keep as <path> with L.
        # Let's try to make <line> if it's just one segment.
        if len(subpath) == 1:
            elem = ET.Element('line')
            elem.set('x1', f"{subpath[0].start.real:.4f}")
            elem.set('y1', f"{subpath[0].start.imag:.4f}")
            elem.set('x2', f"{subpath[0].end.real:.4f}")
            elem.set('y2', f"{subpath[0].end.imag:.4f}")
            return elem, "line"
        else:
            # Check if it's a straight line composed of multiple segments (collinear)
            # If so, merge them.
            # But if it's a polyline (corners), keep as path or split?
            # Let's keep as path with L for now, unless user insists on <line>s.
            pass

    # 2. Check for Circle
    if is_path_closed(subpath):
        points = [seg.start for seg in subpath]
        avg_center = np.mean(points)
        avg_radius = np.mean([abs(p - avg_center) for p in points])
        max_err = 0
        for seg in subpath:
            for t in [0, 0.5, 1]:
                p = seg.point(t)
                err = abs(abs(p - avg_center) - avg_radius)
                if err > max_err: max_err = err
        
        if max_err < tolerance:
             elem = ET.Element('circle')
             elem.set('cx', f"{avg_center.real:.4f}")
             elem.set('cy', f"{avg_center.imag:.4f}")
             elem.set('r', f"{avg_radius:.4f}")
             return elem, "circle"

    # 3. Convert to Path with A/L
    current_path_d = ""
    start_point = subpath[0].start
    current_path_d += f"M {start_point.real:.4f},{start_point.imag:.4f} "
    
    for segment in subpath:
        p3 = segment.end
        if isinstance(segment, Line):
            current_path_d += f"L {p3.real:.4f},{p3.imag:.4f} "
        elif isinstance(segment, CubicBezier):
            commands = bezier_to_arcs_recursive(segment, tolerance)
            for cmd in commands:
                if cmd[0] == 'L':
                    pt = cmd[1]
                    current_path_d += f"L {pt.real:.4f},{pt.imag:.4f} "
                elif cmd[0] == 'A':
                    _, rx, ry, rot, large, sweep, pt = cmd
                    current_path_d += f"A {rx:.4f},{ry:.4f} {rot} {large} {sweep} {pt.real:.4f},{pt.imag:.4f} "
        elif isinstance(segment, Arc):
             current_path_d += f"A {segment.radius.real:.4f},{segment.radius.imag:.4f} {segment.rotation} {int(segment.large_arc)} {int(segment.sweep)} {p3.real:.4f},{p3.imag:.4f} "
    
    if is_path_closed(subpath):
        current_path_d += "Z"
        
    elem = ET.Element('path')
    elem.set('d', current_path_d.strip())
    return elem, "path"

def convert_path_to_elements(path_string, tolerance=0.1):
    try:
        # svgpathtools parse_path returns a single Path object
        # If there are multiple M commands, it might handle it or not.
        # Actually, parse_path splits by M if we look at source, but returns one Path object?
        # No, parse_path returns a Path object which is a list of segments.
        # If there is a Move, the segment list just continues, but the start of next segment != end of previous.
        path = parse_path(path_string)
    except:
        return [], "error"

    elements = []
    
    # Split into continuous subpaths
    subpaths = []
    if len(path) > 0:
        current_subpath = Path()
        current_subpath.append(path[0])
        for i in range(1, len(path)):
            prev_end = path[i-1].end
            curr_start = path[i].start
            if abs(prev_end - curr_start) > 1e-6:
                # Discontinuous
                subpaths.append(current_subpath)
                current_subpath = Path()
            current_subpath.append(path[i])
        subpaths.append(current_subpath)
    
    for subpath in subpaths:
        elem, type_ = convert_subpath_to_element(subpath, tolerance)
        elements.append(elem)
        
    return elements, "mixed"

def process_file(input_path, output_path):
    try:
        tree = ET.parse(input_path)
        root = tree.getroot()
        ET.register_namespace('', "http://www.w3.org/2000/svg")
        
        parent_map = {c: p for p in tree.iter() for c in p}
        replacements = []
        
        for elem in root.findall('.//{http://www.w3.org/2000/svg}path'):
            d = elem.get('d')
            if not d: continue
            
            new_elements, type_ = convert_path_to_elements(d)
            if new_elements:
                attribs = elem.attrib.copy()
                if 'd' in attribs: del attribs['d']
                for new_elem in new_elements:
                    for k, v in attribs.items():
                        if k not in new_elem.attrib:
                            new_elem.set(k, v)
                replacements.append((elem, new_elements))
        
        for old_elem, new_elems in replacements:
            parent = parent_map[old_elem]
            children = list(parent)
            try:
                idx = children.index(old_elem)
                parent.remove(old_elem)
                for i, new_elem in enumerate(new_elems):
                    parent.insert(idx + i, new_elem)
            except ValueError:
                pass
                
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"Error processing {input_path}: {e}")
        return False

def main():
    input_dir = "/workspace/Drawing2CAD/data/svg_raw"
    output_dir = "/workspace/Drawing2CAD/data/svg_raw_convertion"
    files = glob.glob(os.path.join(input_dir, "**/*.svg"), recursive=True)
    print(f"Found {len(files)} SVG files.")
    for file_path in tqdm(files):
        rel_path = os.path.relpath(file_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        process_file(file_path, out_path)

if __name__ == "__main__":
    main()
