import cv2
import numpy as np

# 전역 변수 설정
points = []
roi_selected = False

def mouse_callback(event, x, y, flags, param):
    global points, roi_selected
    
    # 왼쪽 버튼 클릭 시 좌표 저장
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            print(f"점 {len(points)} 선택됨: {x}, {y}")
            
            # 점 시각화 (작은 원)
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", img_display)
            
            # 두 점이 다 선택되면
            if len(points) == 2:
                roi_selected = True
                print("두 점 선택 완료. ROI 처리를 시작합니다.")

def find_intersection(line1, line2):
    """두 직선의 교차점 (x, y)를 계산합니다."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: return None
    
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    
    return [int(px), int(py)]

def get_corners_advanced(roi_img, debug=True):
    """
    [고급] CLAHE 전처리 + LSD(Line Segment Detector)를 이용한 정밀 코너 검출
    """
    
    # =========================================================
    # [Step 1] 전처리: 대비 향상 (CLAHE) & 노이즈 제거
    # =========================================================
    if len(roi_img.shape) == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_img

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Bilateral Filter: 엣지는 보존하면서 노이즈만 제거
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    if debug:
        cv2.imshow("Debug 1: Enhanced (CLAHE + Bilateral)", filtered)

    # =========================================================
    # [Step 2] 엣지 검출 알고리즘 교체: LSD (Line Segment Detector)
    # =========================================================
    # OpenCV 4.x 이상에서 사용 가능
    lsd = cv2.createLineSegmentDetector(0) # 0: LSD_REFINE_STD
    
    # 직선 검출 (lines: N x 1 x 4 배열 [x1, y1, x2, y2])
    lines, width, prec, nfa = lsd.detect(filtered)
    
    if lines is None:
        print("❌ No lines detected by LSD.")
        return None

    # 라인 필터링 (너무 짧은 선 제거)
    min_len = min(roi_img.shape[:2]) * 0.2
    valid_lines = []
    
    # 시각화용 이미지 (디버깅)
    if debug:
        debug_lsd_img = roi_img.copy()
        lsd.drawSegments(debug_lsd_img, lines)
        cv2.imshow("Debug 2: All LSD Lines", debug_lsd_img)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length > min_len:
            valid_lines.append(line[0])

    if not valid_lines:
        print("❌ No valid long lines found.")
        return None

    # =========================================================
    # [Step 3] 4개의 대표 직선(상/하/좌/우) 분류 및 교차점 계산
    # =========================================================
    horizontals = []
    verticals = []
    
    for line in valid_lines:
        x1, y1, x2, y2 = line
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if -45 < angle < 45: horizontals.append(line)
        elif abs(angle) > 45: verticals.append(line)
    
    if not horizontals or not verticals:
        print("❌ Failed to classify lines into H/V.")
        return None

    # 가장 외곽에 있는 선 선택 (Top, Bottom, Left, Right)
    top_line = min(horizontals, key=lambda l: (l[1] + l[3]) / 2)
    bottom_line = max(horizontals, key=lambda l: (l[1] + l[3]) / 2)
    left_line = min(verticals, key=lambda l: (l[0] + l[2]) / 2)
    right_line = max(verticals, key=lambda l: (l[0] + l[2]) / 2)

    if debug:
        debug_final_img = roi_img.copy()
        # 상단(노랑), 하단(파랑), 좌측(보라), 우측(빨강)
        cv2.line(debug_final_img, (int(top_line[0]), int(top_line[1])), (int(top_line[2]), int(top_line[3])), (0, 255, 255), 2)
        cv2.line(debug_final_img, (int(bottom_line[0]), int(bottom_line[1])), (int(bottom_line[2]), int(bottom_line[3])), (255, 0, 0), 2)
        cv2.line(debug_final_img, (int(left_line[0]), int(left_line[1])), (int(left_line[2]), int(left_line[3])), (255, 0, 255), 2)
        cv2.line(debug_final_img, (int(right_line[0]), int(right_line[1])), (int(right_line[2]), int(right_line[3])), (0, 0, 255), 2)
        cv2.imshow("Debug 3: Selected 4 Lines", debug_final_img)

    # 교차점 계산
    corners = []
    corners.append(find_intersection(top_line, left_line))     # TL
    corners.append(find_intersection(top_line, right_line))    # TR
    corners.append(find_intersection(bottom_line, right_line)) # BR
    corners.append(find_intersection(bottom_line, left_line))  # BL
    
    if any(c is None for c in corners):
        print("❌ Intersection calculation failed.")
        return None

    return np.array(corners)

# ==========================================
# 메인 실행 루프
# ==========================================
image_path = "/workspace/sequences_sample/202511171343454627/image/0000.jpg" 
original_img = cv2.imread(image_path)

if original_img is None:
    print("이미지를 불러올 수 없습니다.")
    original_img = np.zeros((600, 800, 3), dtype=np.uint8) + 255
    cv2.rectangle(original_img, (200, 150), (600, 450), (0, 0, 0), -1)

img_display = original_img.copy()

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

print("이미지에서 ROI의 좌측상단, 우측하단 두 점을 클릭하세요.")

while True:
    cv2.imshow("Image", img_display)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    
    if roi_selected:
        p1, p2 = points
        x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
        y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
        
        roi = original_img[y1:y2, x1:x2]
        
        if roi.size == 0:
            print("ROI 영역이 유효하지 않습니다.")
            points = []
            roi_selected = False
            continue

        cv2.imshow("Cropped ROI", roi)
        
        # [핵심 변경] LSD 기반 코너 검출 호출
        corners = get_corners_advanced(roi.copy(), debug=True)
        
        if corners is not None:
            global_corners = []
            for point in corners:
                gx = point[0] + x1
                gy = point[1] + y1
                global_corners.append((gx, gy))
            
            res_img = original_img.copy()
            cv2.rectangle(res_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            for idx, (gx, gy) in enumerate(global_corners):
                cv2.circle(res_img, (gx, gy), 8, (0, 255, 0), -1)
                cv2.putText(res_img, str(idx+1), (gx+10, gy-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            pts = np.array(global_corners, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(res_img, [pts], True, (0, 255, 0), 2)

            cv2.imshow("Result", res_img)
            print("✅ LSD 기반 코너 검출 완료.")
        
        roi_selected = False 
        points = [] 

cv2.destroyAllWindows()