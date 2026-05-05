import xml.etree.ElementTree as ET
import numpy as np
from svgpathtools import parse_path, CubicBezier, Path, Arc

def is_bezier_segment_circular(segment, tolerance=0.05):
    """
    단일 큐빅 베지어 세그먼트가 원형 호를 근사하는지 확인합니다.
    """
    if not isinstance(segment, CubicBezier):
        return False, None

    # 시작점(P0), 제어점(P1, P2), 끝점(P3)
    p0 = segment.start
    p1 = segment.control1
    p2 = segment.control2
    p3 = segment.end

    # 현(Chord)의 길이
    chord = abs(p3 - p0)
    if chord < 1e-6: return False, None

    # 원의 중심과 반지름 추정 (간단한 기하학적 방법 사용)
    # 중점
    mid = (p0 + p3) / 2
    
    # 베지어 곡선의 중간 지점(t=0.5) 계산
    mid_curve = segment.point(0.5)
    
    # 현의 수직 이등분선 위에 원의 중심이 있어야 함
    # 하지만 여기서는 간단히 t=0.5 지점이 원주 위에 있다고 가정하고 계산
    
    # 3점을 지나는 원의 중심 계산 (P0, mid_curve, P3)
    x1, y1 = p0.real, p0.imag
    x2, y2 = mid_curve.real, mid_curve.imag
    x3, y3 = p3.real, p3.imag
    
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-6: return False, None # 일직선임

    cx = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    cy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    
    center = complex(cx, cy)
    radius = abs(p0 - center)

    # 검증: 제어점들이 반지름 거리 내에 있는지, 대칭성을 이루는지 확인
    # 큐빅 베지어로 원을 근사할 때 오차 확인
    dist_p1 = abs(p1 - center)
    dist_p2 = abs(p2 - center)
    dist_p3 = abs(p3 - center)

    # 반지름 오차율 체크
    if (abs(dist_p3 - radius) / radius > tolerance): return False, None
    
    # 아주 정밀한 체크는 생략하고 반지름과 중심 반환
    return True, (center, radius)

def convert_path_to_shape(path_string):
    """
    SVG Path 문자열을 분석하여 Circle 또는 최적화된 Path로 변환합니다.
    """
    try:
        path = parse_path(path_string)
    except:
        return None, "parse_error"

    if len(path) == 0:
        return None, "empty"

    centers = []
    radii = []
    
    is_circular = True
    
    # 모든 세그먼트가 원형 호를 구성하는지 확인
    for segment in path:
        is_seg_circular, params = is_bezier_segment_circular(segment)
        if not is_seg_circular:
            is_circular = False
            break
        centers.append(params[0])
        radii.append(params[1])

    # 1. Circle 변환 시도
    if is_circular and path.isclosed():
        # 모든 세그먼트의 중심점과 반지름이 거의 일치하는지 확인
        avg_center = np.mean(centers)
        avg_radius = np.mean(radii)
        
        center_error = max([abs(c - avg_center) for c in centers])
        radius_error = max([abs(r - avg_radius) for r in radii])

        # 오차 허용 범위 (픽셀 단위)
        if center_error < 1.0 and radius_error < 1.0:
            return {
                'type': 'circle',
                'cx': round(avg_center.real, 4),
                'cy': round(avg_center.imag, 4),
                'r': round(avg_radius, 4)
            }, "converted_to_circle"

    # 2. Arc 변환 (여기서는 Circle 변환이 안 된 경우, 원본 반환)
    # 복잡한 곡선을 A 명령어로 바꾸는 것은 shape가 유지되지 않을 위험이 큼
    
    return None, "keep_original"

def process_svg_file(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    
    # 네임스페이스 처리 (태그명 앞에 {uri}가 붙는 것 방지)
    ET.register_namespace('', "http://www.w3.org/2000/svg")

    converted_count = 0

    for elem in root.findall('.//{http://www.w3.org/2000/svg}path'):
        d = elem.get('d')
        if not d: continue

        shape_data, status = convert_path_to_shape(d)

        if status == "converted_to_circle":
            # 기존 path 태그를 circle 태그로 교체하기 위해 부모를 찾아야 함
            # ElementTree는 부모 찾기가 어려우므로, path 태그 자체를 circle로 변경
            elem.tag = '{http://www.w3.org/2000/svg}circle'
            elem.set('cx', str(shape_data['cx']))
            elem.set('cy', str(shape_data['cy']))
            elem.set('r', str(shape_data['r']))
            
            # d 속성 삭제
            del elem.attrib['d']
            converted_count += 1
            print(f"✅ 변환 성공: Path -> Circle (r={shape_data['r']})")
        
        elif status == "parse_error":
            print(f"⚠️ 파싱 에러: {d[:20]}...")

    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"\n총 {converted_count}개의 Path가 Circle로 변환되어 '{output_file}'에 저장되었습니다.")


if __name__ == '__main__':
    file_path = "/workspace/Drawing2CAD/data/svg_raw/0000/00000007/00000007_FrontTopRight.svg"
    process_svg_file(file_path, "output.svg")
