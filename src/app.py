# app.py 
import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import math
import folium
from streamlit_folium import st_folium
from PIL import Image
from io import BytesIO
import pytesseract
import re
import sqlite3
from datetime import datetime
import pandas as pd
import os
import sys
import base64
import random
import json

# Add parent directory to path to find components module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import custom components
try:
    from components.brochure_viewer import brochure_viewer
except ImportError as e:
    raise ImportError(
        f"Failed to import brochure_viewer. "
        f"Make sure the components directory is in the project root. Error: {e}"
    )

try:
    from components.editable_plot_viewer import editable_plot_viewer
except ImportError as e:
    st.warning(f"⚠️ Failed to import editable_plot_viewer. Editing functionality may not work. Error: {e}")
    editable_plot_viewer = None

# --- CORE DETECTION FUNCTIONS ---

def detect_all_numbers_in_image(image):
    """
    Enhanced OCR to detect all plot numbers with multiple preprocessing methods.
    Returns list of (number, center_x, center_y) tuples.
    """
    try:
        detected_numbers = []
        detected_positions = set()
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Multiple preprocessing approaches
        preprocessing_methods = []
        
        # Method 1: Inverted Otsu (best for black text on white)
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessing_methods.append(('otsu_inv', thresh1))
        
        # Method 2: Enhanced contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        _, thresh2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessing_methods.append(('enhanced', thresh2))
        
        # Method 3: High threshold
        _, thresh3 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        preprocessing_methods.append(('thresh_180', thresh3))
        
        scale_factor = 4
        psm_modes = [7, 11]  # PSM 7: Single line. PSM 11: Sparse text.
        
        for method_name, processed_image in preprocessing_methods:
            height, width = processed_image.shape
            resized = cv2.resize(processed_image, (width * scale_factor, height * scale_factor),
                               interpolation=cv2.INTER_CUBIC)
            
            for psm in psm_modes:
                custom_config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789'
                
                try:
                    data = pytesseract.image_to_data(resized, config=custom_config,
                                                    output_type=pytesseract.Output.DICT)
                    
                    for i in range(len(data['text'])):
                        if int(data['conf'][i]) > 15:  # Low threshold
                            text_content = re.sub(r'[^0-9]', '', data['text'][i])
                            if text_content.isdigit():
                                number = int(text_content)
                                if 1 <= number <= 200:
                                    x = int(data['left'][i] / scale_factor)
                                    y = int(data['top'][i] / scale_factor)
                                    w = int(data['width'][i] / scale_factor)
                                    h = int(data['height'][i] / scale_factor)
                                    
                                    center_x = x + w // 2
                                    center_y = y + h // 2
                                    
                                    position_key = (center_x // 25, center_y // 25, number)
                                    if position_key not in detected_positions:
                                        detected_numbers.append((number, center_x, center_y))
                                        detected_positions.add(position_key)
                except:
                    continue
        
        # Deduplicate close numbers
        final_numbers = []
        used_positions = set()
        
        for number, x, y in sorted(detected_numbers, key=lambda item: (item[0], item[1], item[2])):
            is_duplicate = False
            for used_num, used_x, used_y in used_positions:
                if number == used_num and abs(x - used_x) < 35 and abs(y - used_y) < 35:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_numbers.append((number, x, y))
                used_positions.add((number, x, y))
        
        print(f"OCR detected {len(final_numbers)} numbers: {sorted([n for n, x, y in final_numbers])}")
        return final_numbers
        
    except Exception as e:
        print(f"OCR error: {e}")
        return []


def point_in_polygon(px, py, corners):
    """
    Checks if point is inside polygon using ray casting algorithm.
    """
    polygon = [
        (corners['A']['x'], corners['A']['y']),
        (corners['B']['x'], corners['B']['y']),
        (corners['C']['x'], corners['C']['y']),
        (corners['D']['x'], corners['D']['y'])
    ]
    
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def detect_plot_shapes_enhanced(image_bytes):
    """
    Detects plot boundaries using contour detection on green lines.
    Enhanced for complex layouts with irregular/angled plots.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original_img = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Multiple thresholding methods
    thresh_adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
    
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh_binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
    
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    kernel = np.ones((2, 2), np.uint8)
    thresh_canny = cv2.dilate(edges, kernel, iterations=1)
    
    # Combine all thresholds
    thresh = cv2.bitwise_or(thresh_adaptive, thresh_otsu)
    thresh = cv2.bitwise_or(thresh, thresh_binary)
    thresh = cv2.bitwise_or(thresh, thresh_canny)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_plots_raw = []
    img_area = img.shape[0] * img.shape[1]
    min_plot_area = img_area * 0.0003  # Filter small noise
    max_plot_area = img_area * 0.25     # Filter large regions
    
    print(f"Image: {img.shape[1]}x{img.shape[0]}, Total contours: {len(contours)}")
    
    # First pass: collect potential plots
    temp_plots = []
    for i, contour in enumerate(contours):
        parent = hierarchy[0][i][3]
        area = cv2.contourArea(contour)
        
        if area < min_plot_area or area > max_plot_area:
            continue
        if parent == -1 and area > (img_area * 0.8):
            continue
        
        peri = cv2.arcLength(contour, True)
        for tolerance in [0.04, 0.03, 0.05, 0.02, 0.06]:
            test_approx = cv2.approxPolyDP(contour, tolerance * peri, True)
            if len(test_approx) == 4:
                temp_plots.append({'approx': test_approx, 'area': area})
                break
    
    # Calculate median area for filtering
    if temp_plots:
        areas = [p['area'] for p in temp_plots]
        median_area = np.median(areas)
        area_min_threshold = median_area * 0.08  # Very permissive
        area_max_threshold = median_area * 20.0   # Allow large plots
        print(f"Median area: {median_area:.0f}, Range: {area_min_threshold:.0f} - {area_max_threshold:.0f}")
    else:
        area_min_threshold = min_plot_area
        area_max_threshold = max_plot_area
    
    # Second pass: apply filters
    for i, contour in enumerate(contours):
        parent = hierarchy[0][i][3]
        area = cv2.contourArea(contour)
        
        if area < min_plot_area:
            continue
        if parent == -1 and area > (img_area * 0.8):
            continue
        
        peri = cv2.arcLength(contour, True)
        approx = None
        for tolerance in [0.04, 0.03, 0.05, 0.02, 0.06]:
            test_approx = cv2.approxPolyDP(contour, tolerance * peri, True)
            if len(test_approx) == 4:
                approx = test_approx
                break
        
        if approx is None or len(approx) != 4:
            continue
        
        if area < area_min_threshold or area > area_max_threshold:
            continue
        
        pts = approx.reshape(4, 2)
        
        # Check dimensions
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)
        
        if width > 0 and height > 0:
            min_dimension = min(width, height)
            if min_dimension < 10:  # Too small
                continue
            
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 10.0:  # Too elongated
                continue
        
        # Check rectangularity
        rect_area = width * height
        if rect_area > 0:
            rectangularity = area / rect_area
            if rectangularity < 0.4:  # Too irregular
                continue
        
        # Use bounding rectangle for perfectly straight plots
        # This ensures plots are not rotated or crooked
        x, y, w, h = cv2.boundingRect(contour)
        
        # Create perfectly aligned rectangular corners
        corners = {
            "A": {"x": int(x), "y": int(y + h)},        # Bottom-left
            "B": {"x": int(x + w), "y": int(y + h)},    # Bottom-right
            "C": {"x": int(x + w), "y": int(y)},        # Top-right
            "D": {"x": int(x), "y": int(y)},            # Top-left
        }
        
        detected_plots_raw.append({
            "corners": corners,
            "center_x": int(x + w / 2),
            "center_y": int(y + h / 2)
        })
    
    print(f"✅ Detected {len(detected_plots_raw)} plot boundaries")
    
    # Detect numbers
    detected_numbers = detect_all_numbers_in_image(img)
    
    # Match numbers to plots
    plot_number_map = {}  # plot_index -> number
    
    for num, nx, ny in detected_numbers:
        best_plot_idx = None
        best_distance = float('inf')
        
        for idx, plot in enumerate(detected_plots_raw):
            # Check if number is inside plot
            is_inside = point_in_polygon(nx, ny, plot['corners'])
            
            cx = plot['center_x']
            cy = plot['center_y']
            distance = math.sqrt((nx - cx)**2 + (ny - cy)**2)
            
            if is_inside and distance < best_distance:
                best_distance = distance
                best_plot_idx = idx
        
        if best_plot_idx is not None:
            plot_number_map[best_plot_idx] = num
    
    print(f"✅ Matched {len(plot_number_map)} plots to OCR numbers")
    
    # Sort plots spatially and apply sequential numbering
    for idx, plot in enumerate(detected_plots_raw):
        plot['temp_idx'] = idx
    
    detected_plots_raw.sort(key=lambda p: (p['center_y'], p['center_x']))
    
    # Group into rows using improved clustering
    rows = []
    if detected_plots_raw:
        y_tolerance = 80  # Increased tolerance
        
        for plot in detected_plots_raw:
            assigned = False
            # Try to add to existing row if Y is within tolerance of row's average
            for row in rows:
                if row:
                    avg_y = sum(p['center_y'] for p in row) / len(row)
                    if abs(plot['center_y'] - avg_y) < y_tolerance:
                        row.append(plot)
                        assigned = True
                        break
            
            if not assigned:
                rows.append([plot])
        
        # Sort rows by average Y coordinate
        rows.sort(key=lambda r: sum(p['center_y'] for p in r) / len(r) if r else 0)
    
    print(f"✅ Organized into {len(rows)} rows")
    
    # Debug: Show row sizes and Y coordinates
    for idx, row in enumerate(rows):
        row_y_min = min(p['center_y'] for p in row)
        row_y_max = max(p['center_y'] for p in row)
        print(f"   Row {idx+1}: {len(row)} plots, Y range: {row_y_min}-{row_y_max}")
    
    # Apply sequential numbering per row AND regularize grid
    final_plots = []
    
    # Step 1: Collect all unique X and Y coordinates for grid alignment
    all_x_coords = set()
    all_y_coords = set()
    
    for row in rows:
        for plot in row:
            corners = plot['corners']
            for corner in corners.values():
                all_x_coords.add(corner['x'])
                all_y_coords.add(corner['y'])
    
    # Sort coordinates
    sorted_x = sorted(all_x_coords)
    sorted_y = sorted(all_y_coords)
    
    # Create grid lines by clustering nearby coordinates
    def cluster_coordinates(coords, tolerance=15):
        """Groups nearby coordinates into clusters and returns average."""
        if not coords:
            return []
        clusters = []
        current_cluster = [coords[0]]
        
        for coord in coords[1:]:
            if coord - current_cluster[-1] < tolerance:
                current_cluster.append(coord)
            else:
                clusters.append(int(sum(current_cluster) / len(current_cluster)))
                current_cluster = [coord]
        clusters.append(int(sum(current_cluster) / len(current_cluster)))
        return clusters
    
    # Use tighter tolerance for better alignment
    grid_x = cluster_coordinates(sorted_x, tolerance=15)
    grid_y = cluster_coordinates(sorted_y, tolerance=15)
    
    print(f"✅ Grid structure: {len(grid_x)} columns, {len(grid_y)} rows")
    
    # Step 2: Snap corners to nearest grid lines
    def snap_to_grid(value, grid_lines):
        """Snaps a coordinate to the nearest grid line."""
        return min(grid_lines, key=lambda x: abs(x - value))
    
    # NEW LOGIC: Detect pattern from Row 1 - where does plot 1 start?
    base_direction_pattern = None  # Will be set after Row 1 processing
    
    for row_idx, row in enumerate(rows):
        # Detect if this row is arranged vertically (especially last row with few plots)
        is_last_row = (row_idx == len(rows) - 1)
        is_small_row = len(row) <= 4
        
        is_vertical = False
        if is_small_row and len(row) > 1:
            x_coords = [p['center_x'] for p in row]
            y_coords = [p['center_y'] for p in row]
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            is_vertical = y_range > x_range * 1.5  # More vertical than horizontal
        
        # Sort by Y if vertical (top to bottom), by X if horizontal (left to right)
        if is_vertical:
            row.sort(key=lambda p: p['center_y'])  # Sort top to bottom
            print(f"   Row {row_idx+1}: Vertical arrangement detected, sorting by Y coordinate")
        else:
            row.sort(key=lambda p: p['center_x'])  # Sort left to right by X coordinate
        
        # Map OCR numbers to their column positions (after sorting)
        row_numbers = {}  # col_idx -> ocr_number
        for col_idx, plot in enumerate(row):
            if plot['temp_idx'] in plot_number_map:
                row_numbers[col_idx] = plot_number_map[plot['temp_idx']]
        
        # Debug: Show OCR numbers found in this row
        if row_numbers:
            print(f"   Row {row_idx+1}: {len(row)} plots, OCR numbers found: {sorted(row_numbers.values())} at cols {sorted(row_numbers.keys())}")
        else:
            print(f"   Row {row_idx+1}: {len(row)} plots, No OCR numbers detected")
        
        # NEW ALTERNATING LOGIC: For Row 1, detect where plot 1 is. For subsequent rows, alternate.
        direction = None
        start_num = -1
        
        # For Row 1: Find plot number 1 in OCR to determine starting direction
        if row_idx == 0:
            start_num = 1  # Row 1 always starts from 1
            plot1_found = False
            plot1_col = None
            row_midpoint = len(row) / 2
            
            # First, try to find plot number 1 directly
            for col_idx, ocr_num in row_numbers.items():
                if ocr_num == 1:
                    plot1_col = col_idx
                    plot1_found = True
                    break
            
            # If plot 1 not found, check smallest detected number position
            if not plot1_found and row_numbers:
                min_ocr_num = min(row_numbers.values())
                min_num_cols = [col for col, num in row_numbers.items() if num == min_ocr_num]
                min_num_col = min(min_num_cols)
                
                # If smallest number is on RIGHT side, likely plot 1 is also on RIGHT (RTL sequence)
                # If smallest number is on LEFT side, likely plot 1 is also on LEFT (LTR sequence)
                if min_ocr_num <= 5:  # Only trust if min number is very small (likely near plot 1)
                    plot1_col = min_num_col
                    print(f"   Row 1: Plot 1 not in OCR, but min number {min_ocr_num} at col {min_num_col} → Inferring plot 1 position")
                else:
                    plot1_col = None
            
            if plot1_col is not None:
                if plot1_col < row_midpoint:
                    # Plot 1 is on LEFT → Row 1 goes LTR, then alternate RTL, LTR, RTL...
                    base_direction_pattern = 'left'
                    direction = 'left_to_right'
                    print(f"   Row 1: Plot 1 at LEFT (col {plot1_col}) → Base pattern: LTR, RTL, LTR, RTL...")
                else:
                    # Plot 1 is on RIGHT → Row 1 goes RTL, then alternate LTR, RTL, LTR...
                    base_direction_pattern = 'right'
                    direction = 'right_to_left'
                    print(f"   Row 1: Plot 1 at RIGHT (col {plot1_col}) → Base pattern: RTL, LTR, RTL, LTR...")
            else:
                # Plot 1 position unknown, check overall pattern
                if row_numbers:
                    # Check if sequence increases or decreases
                    sorted_cols = sorted(row_numbers.keys())
                    sorted_nums = [row_numbers[col] for col in sorted_cols]
                    if len(sorted_nums) >= 2:
                        # Check if numbers are decreasing (e.g., 10, 9, 8, 7...) which means RTL
                        # or increasing (e.g., 1, 2, 3...) which means LTR
                        increasing = sorted_nums[0] < sorted_nums[-1]
                        decreasing = sorted_nums[0] > sorted_nums[-1]
                        
                        # Also check if largest number is at rightmost position (indicates RTL)
                        max_num = max(row_numbers.values())
                        max_num_cols = [col for col, num in row_numbers.items() if num == max_num]
                        max_num_col = max(max_num_cols)
                        
                        # If largest number is at rightmost position and numbers are decreasing,
                        # it's likely RTL (e.g., 10 on right, 9, 8, 7 going left)
                        if decreasing and max_num_col == max(sorted_cols):
                            base_direction_pattern = 'right'
                            direction = 'right_to_left'
                            print(f"   Row 1: Sequence decreases L→R with max at right → Base pattern: RTL, LTR, RTL, LTR...")
                        elif increasing:
                            # Increasing left to right → LTR
                            base_direction_pattern = 'left'
                            direction = 'left_to_right'
                            print(f"   Row 1: Sequence increases L→R → Base pattern: LTR, RTL, LTR, RTL...")
                        else:
                            # Decreasing left to right → RTL
                            base_direction_pattern = 'right'
                            direction = 'right_to_left'
                            print(f"   Row 1: Sequence decreases L→R → Base pattern: RTL, LTR, RTL, LTR...")
                    else:
                        # Single number - default to left
                        base_direction_pattern = 'left'
                        direction = 'left_to_right'
                        print(f"   Row 1: Single OCR number → Default: LTR, RTL, LTR, RTL...")
                else:
                    # No OCR at all - default to left
                    base_direction_pattern = 'left'
                    direction = 'left_to_right'
                    print(f"   Row 1: No OCR numbers → Default: LTR, RTL, LTR, RTL...")
            
            print(f"   Row 1: Forced start_num=1, direction={direction.upper()}")
        
        # For Row 2+: Alternate direction based on base pattern
        elif base_direction_pattern:
            # Use the is_vertical flag already detected during sorting
            
            if base_direction_pattern == 'left':
                # Pattern: Row 1=LTR, Row 2=RTL, Row 3=LTR, Row 4=RTL...
                direction = 'right_to_left' if row_idx % 2 == 1 else 'left_to_right'
            else:  # base_direction_pattern == 'right'
                # Pattern: Row 1=RTL, Row 2=LTR, Row 3=RTL, Row 4=LTR...
                direction = 'left_to_right' if row_idx % 2 == 1 else 'right_to_left'
            
            # SPECIAL CASE: Vertical arrangement - use OCR to determine direction
            if is_vertical and row_numbers:
                sorted_cols = sorted(row_numbers.keys())
                sorted_nums = [row_numbers[col] for col in sorted_cols]
                if len(sorted_nums) >= 2:
                    if sorted_nums[0] < sorted_nums[-1]:
                        # Increasing sequence → LTR
                        direction = 'left_to_right'
                        print(f"   Row {row_idx+1}: Last row (vertical), OCR shows INCREASING → Override to LTR")
                    else:
                        # Decreasing sequence → RTL
                        direction = 'right_to_left'
                        print(f"   Row {row_idx+1}: Last row (vertical), OCR shows DECREASING → Override to RTL")
                elif len(sorted_nums) == 1:
                    # Single number - check if it's small (likely start) or large (likely end)
                    single_num = sorted_nums[0]
                    if single_num >= 75:  # Likely near the end
                        direction = 'right_to_left'
                        print(f"   Row {row_idx+1}: Last row, single large number {single_num} → RTL")
                    else:
                        direction = 'left_to_right'
                        print(f"   Row {row_idx+1}: Last row, single small number {single_num} → LTR")
            else:
                print(f"   Row {row_idx+1}: Using alternating pattern → {direction.upper()}")
            
            # Calculate start_num for alternating rows using OCR or sequential
            if row_numbers:
                min_num = min(row_numbers.values())
                min_num_cols = [col for col, num in row_numbers.items() if num == min_num]
                min_num_col = min(min_num_cols)
                
                if direction == 'left_to_right':
                    start_num = min_num - min_num_col
                else:  # right_to_left
                    offset = len(row) - 1 - min_num_col
                    start_num = min_num - offset
                    if start_num < 1:
                        start_num = max(1, min_num - (len(row) - 1))
                
                print(f"   Row {row_idx+1}: Calculated start_num={start_num} from min={min_num} at col {min_num_col}")
            else:
                # No OCR in this row - continue from previous
                if final_plots:
                    numbered_plots = [p['plot_number'] for p in final_plots if p['plot_number'] is not None]
                    if numbered_plots:
                        start_num = max(numbered_plots) + 1
                    else:
                        start_num = len(final_plots) + 1
                else:
                    start_num = 1
                print(f"   Row {row_idx+1}: No OCR, using sequential start={start_num}")
        
        # Fallback: If direction not set yet (shouldn't happen if base_pattern is set), use old logic
        if direction is None and row_numbers:
            # Get leftmost and rightmost by POSITION (column index)
            leftmost_col = min(row_numbers.keys())
            rightmost_col = max(row_numbers.keys())
            leftmost_num = row_numbers[leftmost_col]
            rightmost_num = row_numbers[rightmost_col]
            
            # Also get min/max by VALUE for start_num calculation
            min_num = min(row_numbers.values())
            max_num = max(row_numbers.values())
            min_num_cols = [col for col, num in row_numbers.items() if num == min_num]
            min_num_col = min(min_num_cols)
            
            print(f"   Row {row_idx+1}: Leftmost={leftmost_num} at col {leftmost_col}, Rightmost={rightmost_num} at col {rightmost_col}")
            
            # Determine direction using multiple signals
            if len(row_numbers) >= 2:
                # METHOD 1: Compare leftmost vs rightmost values
                # If leftmost < rightmost: sequence increases L→R → number LTR
                # If leftmost > rightmost: sequence decreases L→R → number RTL
                if leftmost_num < rightmost_num:
                    direction = 'left_to_right'
                    print(f"   Row {row_idx+1}: {leftmost_num} < {rightmost_num} → Sequence INCREASES L→R → LTR")
                elif leftmost_num > rightmost_num:
                    direction = 'right_to_left'
                    print(f"   Row {row_idx+1}: {leftmost_num} > {rightmost_num} → Sequence DECREASES L→R → RTL")
                else:
                    # Leftmost == Rightmost: Use position of minimum value
                    row_midpoint = len(row) / 2
                    if min_num_col < row_midpoint:
                        direction = 'left_to_right'
                        print(f"   Row {row_idx+1}: {leftmost_num} == {rightmost_num}, min at left → LTR")
                    else:
                        direction = 'right_to_left'
                        print(f"   Row {row_idx+1}: {leftmost_num} == {rightmost_num}, min at right → RTL")
                
                # METHOD 2: Verify with pattern if we have 3+ numbers
                if len(row_numbers) >= 3:
                    # Sort by column to see trend
                    sorted_by_col = sorted(row_numbers.items(), key=lambda x: x[0])
                    nums_only = [num for col, num in sorted_by_col]
                    
                    # Check if numbers generally increase or decrease
                    increasing_count = sum(1 for i in range(len(nums_only)-1) if nums_only[i+1] > nums_only[i])
                    decreasing_count = sum(1 for i in range(len(nums_only)-1) if nums_only[i+1] < nums_only[i])
                    
                    # If pattern contradicts leftmost/rightmost, trust the pattern more
                    if increasing_count > decreasing_count and leftmost_num > rightmost_num:
                        # Pattern shows increasing but ends are reversed (might be OCR error)
                        direction = 'left_to_right'
                        print(f"   Row {row_idx+1}: Pattern shows INCREASING trend → Override to LTR")
                    elif decreasing_count > increasing_count and leftmost_num < rightmost_num:
                        # Pattern shows decreasing but ends are reversed (might be OCR error)
                        direction = 'right_to_left'
                        print(f"   Row {row_idx+1}: Pattern shows DECREASING trend → Override to RTL")
            
            else:
                # Only one OCR number detected - use its position
                single_col = list(row_numbers.keys())[0]
                single_num = list(row_numbers.values())[0]
                row_midpoint = len(row) / 2
                if single_col < row_midpoint:
                    direction = 'left_to_right'
                    print(f"   Row {row_idx+1}: Single number {single_num} at left → LTR")
                else:
                    direction = 'right_to_left'
                    print(f"   Row {row_idx+1}: Single number {single_num} at right → RTL")
                min_num = single_num
                min_num_col = single_col
            
            # Calculate start_num based on direction and minimum value position
            if direction == 'left_to_right':
                start_num = min_num - min_num_col
            else:  # right_to_left
                offset = len(row) - 1 - min_num_col
                start_num = min_num - offset
                if start_num < 1:
                    start_num = max(1, min_num - (len(row) - 1))
            
            print(f"   Row {row_idx+1}: Final → {direction.upper()}, start={start_num} (min={min_num} at col {min_num_col})")
                
        else:
            # No OCR numbers - use alternating pattern or sequential
            if row_idx == 0:
                # Row 1 with no OCR - default to LTR pattern
                start_num = 1
                if direction is None:
                    base_direction_pattern = 'left'
                    direction = 'left_to_right'
                print(f"   Row 1: No OCR numbers. Using start: {start_num}, direction: {direction}")
            elif base_direction_pattern:
                # Use alternating pattern
                if base_direction_pattern == 'left':
                    direction = 'right_to_left' if row_idx % 2 == 1 else 'left_to_right'
                else:
                    direction = 'left_to_right' if row_idx % 2 == 1 else 'right_to_left'
                
                # Continue from previous row
                if final_plots:
                    numbered_plots = [p['plot_number'] for p in final_plots if p['plot_number'] is not None]
                    if numbered_plots:
                        start_num = max(numbered_plots) + 1
                    else:
                        start_num = len(final_plots) + 1
                else:
                    start_num = 1
                print(f"   Row {row_idx+1}: No OCR, using alternating pattern → {direction.upper()}, start: {start_num}")
            else:
                # Fallback if base_pattern not set
                direction = 'left_to_right'
                if final_plots:
                    numbered_plots = [p['plot_number'] for p in final_plots if p['plot_number'] is not None]
                    if numbered_plots:
                        start_num = max(numbered_plots) + 1
                    else:
                        start_num = len(final_plots) + 1
                else:
                    start_num = 1
                print(f"   Row {row_idx+1}: No OCR numbers. Using start: {start_num}")
        
        # FINAL CHECK: Row 1 MUST always start from 1
        if row_idx == 0:
            if start_num != 1:
                print(f"   ⚠️ CRITICAL: Row 1 start_num was {start_num}, forcing to 1!")
                start_num = 1
            print(f"   ✅ Row 1 FINAL CHECK: start_num={start_num}, direction={direction.upper()}")
        
        # Assign numbers based on detected direction
        if row_idx == 0:
            print(f"   🎯 Row 1 ASSIGNMENT START: start_num={start_num}, direction={direction}, row_length={len(row)}")
        
        for col_idx, plot in enumerate(row):
            if direction == 'left_to_right':
                plot_num = start_num + col_idx
            else:  # right_to_left
                plot_num = start_num + (len(row) - 1 - col_idx)
            
            # Debug output for Row 1
            if row_idx == 0 and col_idx < 3:
                print(f"      Row 1 Plot[{col_idx}]: plot_num = {start_num} + {col_idx} = {plot_num}")
            
            # Regularize corners by snapping to grid
            corners = plot['corners']
            regularized_corners = {
                'A': {'x': snap_to_grid(corners['A']['x'], grid_x), 
                      'y': snap_to_grid(corners['A']['y'], grid_y)},
                'B': {'x': snap_to_grid(corners['B']['x'], grid_x), 
                      'y': snap_to_grid(corners['B']['y'], grid_y)},
                'C': {'x': snap_to_grid(corners['C']['x'], grid_x), 
                      'y': snap_to_grid(corners['C']['y'], grid_y)},
                'D': {'x': snap_to_grid(corners['D']['x'], grid_x), 
                      'y': snap_to_grid(corners['D']['y'], grid_y)},
            }
            
            # FIX: Ensure corners don't collapse - check for duplicate corner coordinates
            corner_points = [
                (regularized_corners['A']['x'], regularized_corners['A']['y']),
                (regularized_corners['B']['x'], regularized_corners['B']['y']),
                (regularized_corners['C']['x'], regularized_corners['C']['y']),
                (regularized_corners['D']['x'], regularized_corners['D']['y']),
            ]
            unique_points = set(corner_points)
            
            # Get min/max coordinates for bounding box
            x_coords = [c['x'] for c in regularized_corners.values()]
            y_coords = [c['y'] for c in regularized_corners.values()]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # ENFORCE PERFECT RECTANGULAR ALIGNMENT
            # Force corners to perfect rectangle boundaries (no rotation)
            regularized_corners = {
                'A': {'x': min_x, 'y': max_y},      # Bottom-left
                'B': {'x': max_x, 'y': max_y},      # Bottom-right
                'C': {'x': max_x, 'y': min_y},      # Top-right
                'D': {'x': min_x, 'y': min_y},      # Top-left
            }
            
            # Validate: ensure minimum width/height
            width = max_x - min_x
            height = max_y - min_y
            
            # If width or height is zero/too small, use original corners
            if width < 5 or height < 5:
                print(f"⚠️ Plot {plot_num}: Regularized polygon too small (w={width}, h={height}), using original corners")
                regularized_corners = corners.copy()
            
            final_plots.append({
                'plot_id': f"Plot {plot_num}",
                'plot_number': plot_num,
                'corners': regularized_corners,
                'row': row_idx,
                'col': col_idx
            })
    
    print(f"✅ Final: {len(final_plots)} plots with sequential numbering")
    
    # Debug: Check for plots 75, 79, 80
    all_plot_nums = sorted([p['plot_number'] for p in final_plots])
    print(f"📍 All plot numbers: {all_plot_nums}")
    for check_num in [75, 79, 80]:
        matching = [p for p in final_plots if p['plot_number'] == check_num]
        if matching:
            print(f"   ✅ Plot {check_num}: Found at row {matching[0]['row']}, col {matching[0]['col']}")
        else:
            print(f"   ⚠️ Plot {check_num}: NOT FOUND!")
    
    # Verify Row 1 numbering
    row1_plots = [p for p in final_plots if p.get('row') == 0]
    if row1_plots:
        row1_nums = sorted([p['plot_number'] for p in row1_plots])
        print(f"📍 Row 1 verification: First 10 plot numbers = {row1_nums[:10]}{'...' if len(row1_nums) > 10 else ''}")
        if row1_nums and row1_nums[0] != 1:
            print(f"   ⚠️ ERROR: Row 1 starts from {row1_nums[0]} instead of 1!")
        else:
            print(f"   ✅ Row 1 correctly starts from 1")
    
    # Create visualization - black and white wireframe with red lines and red dots
    display_img = original_img.copy()
    # Convert to grayscale for black and white wireframe
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    # Convert back to BGR for colored annotations (red lines and dots)
    display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    
    for plot in final_plots:
        corners = plot['corners']
        pts = np.array([
            [corners['A']['x'], corners['A']['y']],
            [corners['B']['x'], corners['B']['y']],
            [corners['C']['x'], corners['C']['y']],
            [corners['D']['x'], corners['D']['y']]
        ], np.int32)
        # Red lines for plot boundaries (BGR format: red = (0, 0, 255))
        cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
        
        # Draw red dots at each corner (matching wireframe style)
        cv2.circle(display_img, (corners['A']['x'], corners['A']['y']), 4, (0, 0, 255), -1)  # Red dot at A
        cv2.circle(display_img, (corners['B']['x'], corners['B']['y']), 4, (0, 0, 255), -1)  # Red dot at B
        cv2.circle(display_img, (corners['C']['x'], corners['C']['y']), 4, (0, 0, 255), -1)  # Red dot at C
        cv2.circle(display_img, (corners['D']['x'], corners['D']['y']), 4, (0, 0, 255), -1)  # Red dot at D
        
        # Draw plot number in black for better visibility on grayscale background
        plot_number = plot.get('plot_number')
        if plot_number is not None:
            cx = sum([corners[c]['x'] for c in corners]) // 4
            cy = sum([corners[c]['y'] for c in corners]) // 4
            cv2.putText(display_img, str(plot_number),
                       (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return final_plots, display_img, thresh, detected_numbers


def calculate_geocoordinates(plots, ref_plot_id, ref_corner, ref_lat, ref_lon, px_to_ft):
    """
    Converts pixel coordinates to lat/lon.
    """
    ref_x_px, ref_y_px = None, None
    
    for plot in plots:
        if plot['plot_id'] == ref_plot_id:
            # CRITICAL: Use 'points' array if available, otherwise use corners
            points = plot.get('points', [])
            if points:
                # Find the point with the matching label
                corner_idx = ord(ref_corner) - ord('A')
                if corner_idx < len(points):
                    ref_x_px = points[corner_idx].get('x', 0)
                    ref_y_px = points[corner_idx].get('y', 0)
                else:
                    # Fallback to corners if point not found in points array
                    if ref_corner in plot.get('corners', {}):
                        ref_x_px = plot['corners'][ref_corner]['x']
                        ref_y_px = plot['corners'][ref_corner]['y']
            else:
                # Fallback: Use corners
                if ref_corner in plot.get('corners', {}):
                    ref_x_px = plot['corners'][ref_corner]['x']
                    ref_y_px = plot['corners'][ref_corner]['y']
            break
    
    if ref_x_px is None:
        st.error(f"Reference plot '{ref_plot_id}' with corner '{ref_corner}' not found!")
        return []
    
    FT_TO_M = 0.3048
    EARTH_RADIUS_M = 6371000
    ref_lat_rad = math.radians(ref_lat)
    m_per_deg_lat = 111132
    m_per_deg_lon = (math.pi / 180) * EARTH_RADIUS_M * math.cos(ref_lat_rad)
    
    lat_deg_per_px = (1 / m_per_deg_lat) * FT_TO_M * px_to_ft
    lon_deg_per_px = (1 / m_per_deg_lon) * FT_TO_M * px_to_ft
    
    origin_lat = ref_lat + (ref_y_px * lat_deg_per_px)
    origin_lon = ref_lon - (ref_x_px * lon_deg_per_px)
    
    plots_with_latlon = []
    for plot in plots:
        new_corners = {}
        has_invalid_coords = False
        
        # CRITICAL: Use 'points' array if available (supports more than 4 points), 
        # otherwise use corners (for backward compatibility)
        points = plot.get('points', [])
        if not points:
            # Fallback: Convert corners to points
            corners = plot.get('corners', {})
            corner_labels = ['A', 'B', 'C', 'D']
            points = []
            for label in corner_labels:
                if label in corners:
                    points.append({
                        'x': corners[label].get('x', 0),
                        'y': corners[label].get('y', 0)
                    })
        
        # Generate column labels (A, B, C, D, E, F, ...)
        corner_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
        
        # Process all points and assign labels
        for idx, point in enumerate(points):
            # Always use index-based assignment (points array doesn't have labels)
            if idx < len(corner_labels):
                corner_label = corner_labels[idx]
            else:
                # For points beyond P, use extended labels
                corner_label = chr(ord('Q') + (idx - len(corner_labels)))
            
            x = point.get('x', 0)
            y = point.get('y', 0)
            
            # Debug: Log for plot 71
            if plot.get('plot_number') == 71:
                print(f"   Point {idx} ({corner_label}): x={x}, y={y}")
            
            # Calculate lat/lon for all points (even if x=0, y=0, as they might be valid)
            lat = origin_lat - (y * lat_deg_per_px)
            lon = origin_lon + (x * lon_deg_per_px)
            
            # Warn if coordinates are 0, but still calculate lat/lon
            if x == 0 and y == 0:
                print(f"⚠️ Plot {plot.get('plot_number')}: Point {corner_label} has zero coordinates, but calculating lat/lon anyway")
            
            # Check for invalid coordinates
            if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
                has_invalid_coords = True
                print(f"⚠️ Plot {plot.get('plot_number')}: Invalid coordinates at point {corner_label}")
                continue
            
            new_corners[corner_label] = {"lat": lat, "lon": lon}
            
            # Debug: Log calculated lat/lon for plot 71
            if plot.get('plot_number') == 71:
                print(f"   Calculated {corner_label}: lat={lat:.6f}, lon={lon:.6f}")
        
        # Always include plot, even if coordinates seem invalid (let map handle it)
        plots_with_latlon.append({
            "plot_id": plot['plot_id'],
            "plot_number": plot['plot_number'],
            "corners": new_corners
        })
    
    return plots_with_latlon


def recalculate_pixel_from_coordinates(ref_lat, ref_lon, ref_x, ref_y, new_lat, new_lon, px_to_ft):
    """
    Recalculates pixel coordinates from lat/lon using the same formula as calculate_geocoordinates.
    """
    FT_TO_M = 0.3048
    EARTH_RADIUS_M = 6371000
    ref_lat_rad = math.radians(ref_lat)
    m_per_deg_lat = 111132
    m_per_deg_lon = (math.pi / 180) * EARTH_RADIUS_M * math.cos(ref_lat_rad)
    
    lat_deg_per_px = (1 / m_per_deg_lat) * FT_TO_M * px_to_ft
    lon_deg_per_px = (1 / m_per_deg_lon) * FT_TO_M * px_to_ft
    
    origin_lat = ref_lat + (ref_y * lat_deg_per_px)
    origin_lon = ref_lon - (ref_x * lon_deg_per_px)
    
    # Reverse calculation: from lat/lon back to pixel
    calculated_y = (origin_lat - new_lat) / lat_deg_per_px
    calculated_x = (new_lon - origin_lon) / lon_deg_per_px
    
    return calculated_x, calculated_y


def renumber_plots_sequentially(plots, selected_plot_id, direction, numbering_mode='row', second_plot_id=None):
    """
    Renumbers all plots sequentially based on selected plot and direction.
    Uses snake/alternating pattern for natural flow.
    
    Args:
        plots: List of plot dictionaries with 'plot_id', 'corners', 'plot_number'
        selected_plot_id: ID of the plot that should be Plot 1
        direction: Direction to number:
            - Row mode: 'left' = left-to-right, 'right' = right-to-left
            - Column mode: 'left' = top-to-bottom, 'right' = bottom-to-top
        numbering_mode: 'row' for row-wise, 'column' for column-wise
        second_plot_id: Optional ID of plot that should be Plot 2 (for handling spanning plots)
    
    Returns:
        Updated plots list with new plot numbers
    """
    if not plots or not selected_plot_id:
        return plots
    
    # Calculate center coordinates for all plots
    plots_with_centers = []
    for plot in plots:
        corners = plot.get('corners', {})
        if not corners:
            continue
        
        # Calculate center from corners
        center_x = sum([corners[c].get('x', 0) for c in ['A', 'B', 'C', 'D']]) / 4
        center_y = sum([corners[c].get('y', 0) for c in ['A', 'B', 'C', 'D']]) / 4
        
        plots_with_centers.append({
            'plot': plot,
            'center_x': center_x,
            'center_y': center_y
        })
    
    if not plots_with_centers:
        return plots
    
    # Find the selected plot
    selected_plot_data = None
    for plot_data in plots_with_centers:
        if plot_data['plot']['plot_id'] == selected_plot_id:
            selected_plot_data = plot_data
            break
    
    if not selected_plot_data:
        print(f"⚠️ Selected plot {selected_plot_id} not found")
        return plots
    
    # Find the second plot if manually specified
    second_plot_data = None
    if second_plot_id:
        for plot_data in plots_with_centers:
            if plot_data['plot']['plot_id'] == second_plot_id:
                second_plot_data = plot_data
                break
        
        if not second_plot_data:
            print(f"⚠️ Second plot {second_plot_id} not found")
            return plots
    
    if numbering_mode == 'row':
        # ========== ROW-WISE NUMBERING (SNAKE PATTERN: LTR→RTL→LTR) ==========
        # Group plots into rows based on Y coordinate
        rows = []
        y_tolerance = 50
        
        for plot_data in plots_with_centers:
            assigned = False
            for row in rows:
                if row:
                    avg_y = sum(p['center_y'] for p in row) / len(row)
                    if abs(plot_data['center_y'] - avg_y) < y_tolerance:
                        row.append(plot_data)
                        assigned = True
                        break
            if not assigned:
                rows.append([plot_data])
        
        # Sort rows top to bottom
        rows.sort(key=lambda r: sum(p['center_y'] for p in r) / len(r))
        
        # Sort each row left to right
        for row in rows:
            row.sort(key=lambda p: p['center_x'])
        
        # Find selected plot's row
        selected_row_idx = None
        for row_idx, row in enumerate(rows):
            for plot_data in row:
                if plot_data['plot']['plot_id'] == selected_plot_id:
                    selected_row_idx = row_idx
                    break
            if selected_row_idx is not None:
                break
        
        if selected_row_idx is None:
            print(f"⚠️ Selected plot not found")
            return plots
        
        # Handle manual Plot 2 selection
        if second_plot_data:
            # Manually assign Plot 1 and Plot 2
            selected_plot_data['plot']['plot_number'] = 1
            selected_plot_data['plot']['plot_id'] = "Plot 1"
            second_plot_data['plot']['plot_number'] = 2
            second_plot_data['plot']['plot_id'] = "Plot 2"
            
            # Find Plot 2's position in the grid
            second_row_idx = None
            second_col_idx = None
            for row_idx, row in enumerate(rows):
                for col_idx, plot_data in enumerate(row):
                    if plot_data['plot']['plot_id'] == second_plot_id:
                        second_row_idx = row_idx
                        second_col_idx = col_idx
                        break
                if second_row_idx is not None:
                    break
            
            # Create ordered list starting from Plot 2's row
            ordered_rows = rows[second_row_idx:] + rows[:second_row_idx]
            
            # Number from Plot 3 onwards, starting from Plot 2's position
            current_number = 3
            for idx, row in enumerate(ordered_rows):
                # Determine direction for this row
                if idx == 0:
                    # First row (Plot 2's row): use selected direction
                    row_direction = direction if direction else 'left'
                else:
                    # Alternate for subsequent rows
                    if direction == 'left':
                        row_direction = 'left' if (idx % 2 == 0) else 'right'
                    else:
                        row_direction = 'right' if (idx % 2 == 0) else 'left'
                
                # Number this row, skipping already numbered plots
                if row_direction == 'left':
                    for col_idx, plot_data in enumerate(row):
                        if plot_data['plot']['plot_number'] in [1, 2]:
                            continue
                        # For Plot 2's row, start after Plot 2
                        if idx == 0 and col_idx <= second_col_idx:
                            continue
                        plot_data['plot']['plot_number'] = current_number
                        plot_data['plot']['plot_id'] = f"Plot {current_number}"
                        current_number += 1
                else:
                    for col_idx, plot_data in enumerate(reversed(row)):
                        if plot_data['plot']['plot_number'] in [1, 2]:
                            continue
                        rev_col_idx = len(row) - 1 - col_idx
                        if idx == 0 and rev_col_idx <= second_col_idx:
                            continue
                        plot_data['plot']['plot_number'] = current_number
                        plot_data['plot']['plot_id'] = f"Plot {current_number}"
                        current_number += 1
            
            print(f"✅ Renumbered {len(plots)} plots row-wise with manual Plot 2")
            print(f"   Plot 1: {selected_plot_id}")
            print(f"   Plot 2: {second_plot_id} (manually selected)")
            return plots
        
        # Create ordered list of rows starting from selected row
        ordered_rows = rows[selected_row_idx:] + rows[:selected_row_idx]
        
        # Number with alternating direction (SNAKE pattern)
        current_number = 1
        for idx, row in enumerate(ordered_rows):
            # Determine direction for this row
            if direction == 'left':
                # First row: LTR, Second: RTL, Third: LTR...
                row_direction = 'left' if (idx % 2 == 0) else 'right'
            else:
                # First row: RTL, Second: LTR, Third: RTL...
                row_direction = 'right' if (idx % 2 == 0) else 'left'
            
            if row_direction == 'left':
                # Left to right
                for plot_data in row:
                    plot_data['plot']['plot_number'] = current_number
                    plot_data['plot']['plot_id'] = f"Plot {current_number}"
                    current_number += 1
            else:
                # Right to left
                for plot_data in reversed(row):
                    plot_data['plot']['plot_number'] = current_number
                    plot_data['plot']['plot_id'] = f"Plot {current_number}"
                    current_number += 1
        
        print(f" Renumbered {len(plots)} plots row-wise (SNAKE pattern)")
        print(f"   Starting from: {selected_plot_id} → Plot 1")
        print(f"   Direction: {'LTR→RTL→LTR...' if direction == 'left' else 'RTL→LTR→RTL...'}")
        
    else:
        # ========== COLUMN-WISE NUMBERING (SNAKE PATTERN: Top→Bottom→Top) ==========
        # Group plots into columns based on X coordinate
        columns = []
        x_tolerance = 50
        
        for plot_data in plots_with_centers:
            assigned = False
            for col in columns:
                if col:
                    avg_x = sum(p['center_x'] for p in col) / len(col)
                    if abs(plot_data['center_x'] - avg_x) < x_tolerance:
                        col.append(plot_data)
                        assigned = True
                        break
            if not assigned:
                columns.append([plot_data])
        
        # Sort columns left to right
        columns.sort(key=lambda c: sum(p['center_x'] for p in c) / len(c))
        
        # Sort each column top to bottom
        for col in columns:
            col.sort(key=lambda p: p['center_y'])
        
        # Find selected plot's column
        selected_col_idx = None
        for col_idx, col in enumerate(columns):
            for plot_data in col:
                if plot_data['plot']['plot_id'] == selected_plot_id:
                    selected_col_idx = col_idx
                    break
            if selected_col_idx is not None:
                break
        
        if selected_col_idx is None:
            print(f"⚠️ Selected plot not found")
            return plots
        
        # Handle manual Plot 2 selection
        if second_plot_data:
            print(f"🔍 MANUAL PLOT 2 MODE ACTIVATED")
            
            # STEP 1: Clear all existing numbers and assign Plot 1 and Plot 2
            for plot in plots:
                plot['plot_number'] = None  # Clear all numbers first
            
            # Now assign Plot 1 and Plot 2
            selected_plot_data['plot']['plot_number'] = 1
            selected_plot_data['plot']['plot_id'] = "Plot 1"
            second_plot_data['plot']['plot_number'] = 2
            second_plot_data['plot']['plot_id'] = "Plot 2"
            
            # Find positions
            plot1_col_idx = selected_col_idx
            plot1_row_idx = None
            for row_idx, plot_data in enumerate(columns[plot1_col_idx]):
                if plot_data['plot']['plot_id'] == selected_plot_id:
                    plot1_row_idx = row_idx
                    break
            
            second_col_idx = None
            second_row_idx = None
            for col_idx, col in enumerate(columns):
                for row_idx, plot_data in enumerate(col):
                    if plot_data['plot']['plot_id'] == second_plot_id:
                        second_col_idx = col_idx
                        second_row_idx = row_idx
                        break
                if second_col_idx is not None:
                    break
            
            print(f"   Plot 1: {selected_plot_id} at column {plot1_col_idx}, row {plot1_row_idx}")
            print(f"   Plot 2: {second_plot_id} at column {second_col_idx}, row {second_row_idx}")
            print(f"   Total columns: {len(columns)}, Direction: {'Move Right' if direction == 'left' else 'Move Left'}")
            
            # STEP 2: Create column order starting from Plot 2's column
            # Plot 2's column is the starting point
            current_number = 3
            
            # Number Plot 2's column going DOWN (top to bottom)
            print(f"\n   📍 COLUMN {second_col_idx} (Plot 2's column) - Going DOWN:")
            for row_idx in range(len(columns[second_col_idx])):
                plot_data = columns[second_col_idx][row_idx]
                if plot_data['plot'].get('plot_number') in [1, 2]:
                    continue  # Skip Plot 1 and Plot 2
                plot_data['plot']['plot_number'] = current_number
                plot_data['plot']['plot_id'] = f"Plot {current_number}"
                print(f"      Row {row_idx}: {plot_data['plot']['plot_id']} → {current_number}")
                current_number += 1
            
            
            if direction == 'left':
                # "Move Right" was selected, but Plot 2 is already at the rightmost column
                # So continue by moving LEFT through remaining columns
                if second_col_idx == len(columns) - 1:
                    # Plot 2 is at rightmost, go LEFT
                    remaining_cols = list(range(second_col_idx - 1, -1, -1))
                    print(f"   Plot 2 at rightmost column, continuing LEFT: {remaining_cols}")
                else:
                    # Plot 2 not at edge, continue RIGHT then wrap
                    remaining_cols = list(range(second_col_idx + 1, len(columns))) + list(range(0, second_col_idx))
                    print(f"   Continuing RIGHT from Plot 2: {remaining_cols}")
            else:
                # "Move Left" was selected
                if second_col_idx == 0:
                    # Plot 2 is at leftmost, go RIGHT
                    remaining_cols = list(range(1, len(columns)))
                    print(f"   Plot 2 at leftmost column, continuing RIGHT: {remaining_cols}")
                else:
                    # Plot 2 not at edge, continue LEFT then wrap
                    remaining_cols = list(range(second_col_idx - 1, -1, -1)) + list(range(len(columns) - 1, second_col_idx, -1))
                    print(f"   Continuing LEFT from Plot 2: {remaining_cols}")
            
            print(f"\n   Remaining columns order: {remaining_cols}")
            
            # STEP 4: Number remaining columns with SNAKE pattern (alternate UP/DOWN)
            for idx, col_idx in enumerate(remaining_cols):
                col = columns[col_idx]
                
                # Determine direction: Plot 2's col went DOWN, so first remaining col goes UP, then DOWN, then UP...
                if idx % 2 == 0:
                    # Even index: go UP (bottom to top)
                    direction_str = "UP (bottom→top)"
                    print(f"\n   📍 COLUMN {col_idx} - Going {direction_str}:")
                    for row_idx in range(len(col) - 1, -1, -1):
                        plot_data = col[row_idx]
                        if plot_data['plot'].get('plot_number') in [1, 2]:
                            print(f"      Row {row_idx}: {plot_data['plot']['plot_id']} → SKIP (already {plot_data['plot'].get('plot_number')})")
                            continue
                        plot_data['plot']['plot_number'] = current_number
                        plot_data['plot']['plot_id'] = f"Plot {current_number}"
                        print(f"      Row {row_idx}: {plot_data['plot']['plot_id']} → {current_number}")
                        current_number += 1
                else:
                    # Odd index: go DOWN (top to bottom)
                    direction_str = "DOWN (top→bottom)"
                    print(f"\n   📍 COLUMN {col_idx} - Going {direction_str}:")
                    for row_idx in range(len(col)):
                        plot_data = col[row_idx]
                        if plot_data['plot'].get('plot_number') in [1, 2]:
                            print(f"      Row {row_idx}: {plot_data['plot']['plot_id']} → SKIP (already {plot_data['plot'].get('plot_number')})")
                            continue
                        plot_data['plot']['plot_number'] = current_number
                        plot_data['plot']['plot_id'] = f"Plot {current_number}"
                        print(f"      Row {row_idx}: {plot_data['plot']['plot_id']} → {current_number}")
                        current_number += 1
            
            print(f"\n Renumbered ALL plots: 1 to {current_number - 1}")
            return plots
        
        # Create ordered list of columns starting from selected column
        # Direction determines which way we move between columns
        if direction == 'left':
            # Move right through columns
            ordered_columns = columns[selected_col_idx:] + columns[:selected_col_idx]
        else:
            # Move left through columns
            ordered_columns = list(reversed(columns[:selected_col_idx+1])) + list(reversed(columns[selected_col_idx+1:]))
        
        # Number with alternating direction (SNAKE pattern)
        # direction='left' means Top-to-Bottom first
        # direction='right' means Bottom-to-Top first
        current_number = 1
        
        print(f"🔍 AUTOMATIC MODE (no manual Plot 2)")
        print(f"   Starting column: {selected_col_idx}")
        print(f"   Direction: {'Top→Bottom (then alternating)' if direction == 'left' else 'Bottom→Top (then alternating)'}")
        
        for idx, col in enumerate(ordered_columns):
            # Determine direction for this column
            if direction == 'left':
                # direction='left' means start Top-to-Bottom
                # First column: Top→Bottom, Second: Bottom→Top, Third: Top→Bottom...
                col_direction = 'top' if (idx % 2 == 0) else 'bottom'
            else:
                # direction='right' means start Bottom-to-Top
                # First column: Bottom→Top, Second: Top→Bottom, Third: Bottom→Top...
                col_direction = 'bottom' if (idx % 2 == 0) else 'top'
            
            direction_str = "DOWN (top→bottom)" if col_direction == 'top' else "UP (bottom→top)"
            print(f"\n   📍 COLUMN {columns.index(col)} (idx={idx}) - Going {direction_str}:")
            
            if col_direction == 'top':
                # Top to bottom
                for row_idx, plot_data in enumerate(col):
                    plot_data['plot']['plot_number'] = current_number
                    plot_data['plot']['plot_id'] = f"Plot {current_number}"
                    print(f"      Row {row_idx}: {plot_data['plot']['plot_id']} → {current_number}")
                    current_number += 1
            else:
                # Bottom to top
                for row_idx in range(len(col) - 1, -1, -1):
                    plot_data = col[row_idx]
                    plot_data['plot']['plot_number'] = current_number
                    plot_data['plot']['plot_id'] = f"Plot {current_number}"
                    print(f"      Row {row_idx}: {plot_data['plot']['plot_id']} → {current_number}")
                    current_number += 1
        
        print(f"\n Renumbered ALL plots: 1 to {current_number - 1}")
    
    return plots


def pil_image_to_base64(image):
    """
    Converts a PIL Image to a base64 data URL string.
    This is a workaround for streamlit-drawable-canvas compatibility issues.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def ndarray_to_data_url(image_array):
    """
    Converts an OpenCV BGR numpy array to a PNG data URL.
    """
    if image_array is None:
        return ""
    pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    return pil_image_to_base64(pil_image)


# --- STREAMLIT UI ---

# Get the directory of the current script and construct path to favicon and CSS
script_dir = os.path.dirname(os.path.abspath(__file__))
favicon_path = os.path.join(os.path.dirname(script_dir), "favicon", "plot_icon.ico")
css_path = os.path.join(script_dir, "styles.css")

st.set_page_config(layout="wide", page_title="Geo Plot Mapper", page_icon=favicon_path)

# Load external CSS file
with open(css_path, 'r') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Additional inline CSS for navigation buttons with maximum specificity
st.markdown("""
<style>
/* MAXIMUM SPECIFICITY FOR NEXT BUTTONS */
div[data-testid="column"] div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button[key*="next_step"][kind="primary"],
div[data-testid="column"] button[key*="next_step"],
button[key*="next_step"] {
    background-color: #1e40af !important;
    background-image: none !important;
    background: #1e40af !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 14px 28px 14px 20px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 12px !important;
    box-shadow: 0 8px 20px rgba(30, 64, 175, 0.35) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    min-height: 48px !important;
    width: 100% !important;
}

button[key*="next_step"]::before {
    content: "›" !important;
    order: 1 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 32px !important;
    height: 32px !important;
    border-radius: 50% !important;
    background: #ffffff !important;
    color: #1e40af !important;
    font-size: 24px !important;
    font-weight: 900 !important;
    line-height: 1 !important;
    flex-shrink: 0 !important;
    margin-left: 8px !important;
}

div[data-testid="column"] button[key*="next_step"]:hover,
button[key*="next_step"]:hover {
    background-color: #2563eb !important;
    background: #2563eb !important;
    box-shadow: 0 12px 28px rgba(37, 99, 235, 0.4) !important;
    transform: translateY(-2px) scale(1.02) !important;
}

/* MAXIMUM SPECIFICITY FOR PREVIOUS BUTTONS */
div[data-testid="column"] div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button[key*="prev_step"],
div[data-testid="column"] button[key*="prev_step"],
button[key*="prev_step"] {
    background-color: #ffffff !important;
    background-image: none !important;
    background: #ffffff !important;
    color: #1f2937 !important;
    border: 2px solid #d1d5db !important;
    border-radius: 50px !important;
    padding: 14px 20px 14px 28px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 12px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    min-height: 48px !important;
    width: 100% !important;
}

button[key*="prev_step"]::after {
    content: "‹" !important;
    order: -1 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 32px !important;
    height: 32px !important;
    border-radius: 50% !important;
    background: #1e40af !important;
    color: #ffffff !important;
    font-size: 24px !important;
    font-weight: 900 !important;
    line-height: 1 !important;
    flex-shrink: 0 !important;
    margin-right: 8px !important;
}

div[data-testid="column"] button[key*="prev_step"]:hover,
button[key*="prev_step"]:hover {
    background-color: #f9fafb !important;
    background: #f9fafb !important;
    border-color: #1e40af !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12) !important;
    transform: translateY(-2px) scale(1.02) !important;
}
</style>
""", unsafe_allow_html=True)

PLOT_STATUS_OPTIONS = ["available", "booked", "sold"]
# Session state
if 'plots' not in st.session_state:
    st.session_state.plots = []
if 'geo_plots' not in st.session_state:
    st.session_state.geo_plots = []
if 'detection_image' not in st.session_state:
    st.session_state.detection_image = None
if 'px_to_ft' not in st.session_state:
    st.session_state.px_to_ft = 0.5
# Initialize current_step
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'coordinates_detected' not in st.session_state:
    st.session_state.coordinates_detected = False
if 'show_detected_coords' not in st.session_state:
    st.session_state.show_detected_coords = False
if 'show_editable_grid' not in st.session_state:
    st.session_state.show_editable_grid = False
if 'brochure_shapes' not in st.session_state:
    st.session_state.brochure_shapes = None
if 'plot_statuses' not in st.session_state:
    st.session_state.plot_statuses = {}
if 'detected_overlay_url' not in st.session_state:
    st.session_state.detected_overlay_url = ""
if 'brochure_bg_url' not in st.session_state:
    st.session_state.brochure_bg_url = ""
if 'brochure_overlay_url' not in st.session_state:
    st.session_state.brochure_overlay_url = ""
if 'edited_plot_coordinates' not in st.session_state:
    st.session_state.edited_plot_coordinates = None
if 'original_image_base64' not in st.session_state:
    st.session_state.original_image_base64 = None
if 'original_uploaded_image_base64' not in st.session_state:
    st.session_state.original_uploaded_image_base64 = None
if 'edits_made_in_step2' not in st.session_state:
    st.session_state.edits_made_in_step2 = False
if 'show_next_warning_step2' not in st.session_state:
    st.session_state.show_next_warning_step2 = False

# Workflow Sidebar - Modern UI with Streamlit buttons (working click navigation)
with st.sidebar:
    # Inject sidebar styling
    st.markdown("""
    <style>
    .sidebar-nav-item {
        position: relative;
        margin-bottom: 4px;
        cursor: pointer;
    }
    .sidebar-nav-display {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        min-height: 40px;
        background-color: #ffffff;
        color: #333;
        transition: all 0.2s ease;
    }
    .sidebar-nav-display:hover {
        background-color: #f4f7ff;
    }
    .sidebar-nav-display.active {
        background-color: #1f3c88;
        color: white;
    }
    .sidebar-nav-display.active svg {
        stroke: white;
        color: white;
    }
    .sidebar-nav-icon {
        flex-shrink: 0;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .sidebar-nav-icon svg {
        width: 20px;
        height: 20px;
        stroke: #333;
        color: #333;
        fill: none;
    }
    .sidebar-nav-display.active .sidebar-nav-icon svg {
        stroke: white;
        color: white;
    }
    .sidebar-nav-text {
        flex: 1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .sidebar-nav-item button {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: 0;
        z-index: 1;
        cursor: pointer;
        margin: 0;
        padding: 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-header">Workflow Steps</div>
    """, unsafe_allow_html=True)
    
    steps = [
        ("1 - Upload Layout Image", '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>'),
        ("2 - Regenerated Image", '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>'),
        ("3 - Detect Coordinates", '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>'),
        ("4 - Preview Polygons", '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>'),
        ("5 - Preview in Brochure", '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>'),
        ("6 - Configure Map Settings", '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M12 1v6m0 6v6M5.64 5.64l4.24 4.24m8.24 8.24l-4.24-4.24M1 12h6m6 0h6M5.64 18.36l4.24-4.24m8.24-8.24l-4.24 4.24"></path></svg>'),
        ("7 - Update Lat and Long", '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>'),
        ("8 - Preview in Google Map", '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"></polygon><line x1="8" y1="2" x2="8" y2="18"></line><line x1="16" y1="6" x2="16" y2="22"></line></svg>')
    ]
    
    # Get current step
    current_step = st.session_state.get("current_step", 1)
    
    # Build all sidebar items HTML (no extra whitespace)
    sidebar_items_html = ""
    for i, (step_label, svg_icon) in enumerate(steps, 1):
        is_active = current_step == i
        active_class = "active" if is_active else ""
        
        sidebar_items_html += f'<div class="sidebar-nav-item" onclick="navigateToStep({i})"><div class="sidebar-nav-display {active_class}"><div class="sidebar-nav-icon">{svg_icon}</div><div class="sidebar-nav-text">{step_label}</div></div></div>'
    
    # Render all items in a single markdown block (compact HTML)
    st.markdown(f'<div class="sidebar-nav-container">{sidebar_items_html}</div><script>function navigateToStep(step){{const url=new URL(window.location.href);url.searchParams.set("step",step);window.location.href=url.toString();}}</script>', unsafe_allow_html=True)
    
    # Handle navigation from query parameters
    step_from_query = st.query_params.get("step", None)
    if step_from_query:
        try:
            step_num = int(step_from_query)
            if 1 <= step_num <= 8:
                if st.session_state.get("current_step", 1) != step_num:
                    st.session_state.current_step = step_num
                    st.query_params.pop("step", None)
                    st.rerun()
        except (ValueError, TypeError):
            pass
    
    # Database section removed from sidebar - now appears below Publish button in Step 8

# Get current step for rendering
current_step = st.session_state.get("current_step", 1)

# Main content area - Title only shows on Step 1
if current_step == 1:
    # Add class to body for Step 1 specific styling
    st.markdown("""
    <script>
        document.body.classList.add('step-1-active');
    </script>
    """, unsafe_allow_html=True)
    # Styled header - centered perfectly
    st.markdown("""
    <div class="geo-header-container">
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            margin: 0 auto;
            padding: 0;
            text-align: center;
        ">
            <h1 style="
                color: white;
                font-weight: bold;
                font-size: 2.2rem;
                margin: 0 auto;
                padding: 0;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                letter-spacing: 1px;
                text-align: center;
            ">Geo Plot Mapper</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)

# STEP 1: Upload Layout Image
if current_step == 1:
    # Add spacing after header
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Custom label for upload section
    st.markdown("""
    <div style="
        color: #4a5568;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 10px;
        margin-left: 0;
    "></div>
    """, unsafe_allow_html=True)
    
    # Add wrapper div with class for easier targeting
    st.markdown("""
    <div class="custom-upload-wrapper">
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Layout Image", type=["jpg", "jpeg", "png"], label_visibility="hidden")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Store image data when uploaded
        if st.session_state.get('last_uploaded_file_name') != uploaded_file.name:
            st.session_state.uploaded_image_bytes = image_bytes  # Store for use in Step 5
            # Store original uploaded image as base64 (before detection/regeneration)
            # This will be used for background toggle in Step 2
            nparr = np.frombuffer(image_bytes, np.uint8)
            original_cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if original_cv_image is not None:
                st.session_state.original_uploaded_image_base64 = ndarray_to_data_url(original_cv_image)
            st.session_state.last_uploaded_file_name = uploaded_file.name
        
        # Image Preview Expander - inline with file name
        with st.expander(" Image Preview", expanded=False):
            st.image(pil_image, caption="Uploaded Image", use_container_width=True)
        
        # Detect Plots button
        if st.button("Detect Plots", type="primary"):
                with st.spinner("Analyzing... Please wait"):
                    plots, display_img, thresh, numbers = detect_plot_shapes_enhanced(image_bytes)
                    st.session_state.plots = plots
                    st.session_state.detection_image = display_img
                    st.session_state.detected_numbers = numbers
                    st.session_state.detected_overlay_url = ndarray_to_data_url(display_img)
                    st.session_state.brochure_overlay_url = st.session_state.detected_overlay_url
                    # Store original image as base64 for editable viewer
                    st.session_state.original_image_base64 = ndarray_to_data_url(display_img)
                    
                    existing_statuses = st.session_state.get('plot_statuses', {})
                    updated_statuses = {}
                    for plot in plots:
                        plot_id = plot.get('plot_id')
                        if plot_id:
                            updated_statuses[plot_id] = existing_statuses.get(
                                plot_id, random.choice(PLOT_STATUS_OPTIONS)
                            )
                    st.session_state.plot_statuses = updated_statuses
                    
                    if plots:
                        st.success(f"Detected {len(plots)} plots!")
                    else:
                        st.error("No plots detected")
        
        # Next button for Step 1 - goes to Step 2 (Regenerated Image)
        if uploaded_file is not None and st.session_state.plots:
            col_btn1, col_btn2 = st.columns([10, 1])
            with col_btn1:
                st.write("")  # Spacer
            with col_btn2:
                if st.button("Next", type="primary", key="next_step1"):
                    st.session_state.current_step = 2  # Go to Step 2 (Regenerated Image)
                    st.rerun()

# STEP 2: Regenerated Image 
elif current_step == 2:
    if st.session_state.detection_image is not None and st.session_state.plots:
        # Initialize show_background_step2 if not already set
        if 'show_background_step2' not in st.session_state:
            st.session_state.show_background_step2 = True
        # Store original image as base64 for the editable viewer
        if st.session_state.original_image_base64 is None:
            # Convert detection image to base64
            st.session_state.original_image_base64 = ndarray_to_data_url(st.session_state.detection_image)
        
        # Add toggle switch for background image in a visible container
        st.markdown("<div style='margin-top: 10px; margin-bottom: 15px;'>", unsafe_allow_html=True)
        show_background = st.toggle(
            "Show Background",
            value=st.session_state.get('show_background_step2', True),
            key="toggle_background_step2",
            help="Toggle to show/hide the original background image. When off, only plot lines and dots are visible."
        )
        st.session_state.show_background_step2 = show_background
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Create tabs for "Plot with lines" first, then "Plot with points"
        tab1, tab2 = st.tabs(["Plot with Lines", "Plot with Points"])
        
        # Prepare plots data for the editable viewer
        def prepare_plots_for_viewer():
            plots_data = []
            for plot in st.session_state.plots:
                # CRITICAL: Use 'points' array if available (preserves all points), 
                # otherwise fall back to converting corners (for backward compatibility)
                if 'points' in plot and plot['points']:
                    # Use stored points array (preserves all points, not just 4)
                    points = [{'x': int(p.get('x', 0)), 'y': int(p.get('y', 0))} for p in plot['points']]
                else:
                    # Fallback: Convert corners to points (for backward compatibility with old data)
                    corners = plot.get('corners', {})
                    points = []
                    for corner_label in ['A', 'B', 'C', 'D']:
                        corner = corners.get(corner_label, {})
                        if corner:
                            points.append({'x': corner.get('x', 0), 'y': corner.get('y', 0)})
                
                plots_data.append({
                    'id': plot.get('plot_id', 'unknown'),
                    'plot_number': plot.get('plot_number', 0),
                    'points': points  # CRITICAL: This now contains ALL points, not just 4
                })
            return plots_data
        
        plots_for_viewer = prepare_plots_for_viewer()
        
        # Get image dimensions for blank image creation
        height, width = st.session_state.detection_image.shape[:2]
        
        # Use original uploaded image if available, otherwise fall back to original_image_base64
        original_bg_image = st.session_state.get('original_uploaded_image_base64') or st.session_state.original_image_base64
        background_url = original_bg_image
        
        # Helper function to add background toggle JavaScript
        def add_background_toggle_script():
            show_bg = st.session_state.get('show_background_step2', True)
            st.components.v1.html(f"""
            <script>
                (function() {{
                    // Wait for iframe to load, then send toggle message
                    setTimeout(function() {{
                        try {{
                            // Find all iframes that might contain the canvas
                            const iframes = parent.document.querySelectorAll('iframe');
                            
                            for (let iframe of iframes) {{
                                try {{
                                    // Send message to iframe to toggle background
                                    iframe.contentWindow.postMessage({{
                                        type: 'toggle_background',
                                        show: {str(show_bg).lower()}
                                    }}, '*');
                                }} catch(e) {{
                                    // Iframe might not be ready or cross-origin
                                }}
                            }}
                        }} catch(e) {{
                            console.error('Error sending background toggle message:', e);
                        }}
                    }}, 300);
                }})();
            </script>
            """, height=0)
        
        with tab1:
            if editable_plot_viewer and background_url:
                editable_plot_viewer(
                    background_image_url=background_url,
                    plots=plots_for_viewer,
                    mode="lines"
                )
                add_background_toggle_script()
            else:
                st.warning("⚠️ Editable viewer not available. Showing static image.")
                if st.session_state.get('show_background_step2', True):
                    st.image(st.session_state.detection_image, channels="BGR",
                            caption=f"{len(st.session_state.plots)} plots detected with red lines and dots",
                            use_container_width=True)
                else:
                    # Show blank image with plot overlays
                    blank_display = np.ones((height, width, 3), dtype=np.uint8) * 255
                    
                    # Draw plots on blank image
                    for plot in st.session_state.plots:
                        corners = plot.get('corners', {})
                        if not corners or len(corners) < 4:
                            continue
                        
                        pts = np.array([
                            [corners['A']['x'], corners['A']['y']],
                            [corners['B']['x'], corners['B']['y']],
                            [corners['C']['x'], corners['C']['y']],
                            [corners['D']['x'], corners['D']['y']]
                        ], np.int32)
                        
                        cv2.polylines(blank_display, [pts], True, (0, 0, 255), 2)
                        cv2.circle(blank_display, (corners['A']['x'], corners['A']['y']), 5, (0, 0, 255), -1)
                        cv2.circle(blank_display, (corners['B']['x'], corners['B']['y']), 5, (0, 0, 255), -1)
                        cv2.circle(blank_display, (corners['C']['x'], corners['C']['y']), 5, (0, 0, 255), -1)
                        cv2.circle(blank_display, (corners['D']['x'], corners['D']['y']), 5, (0, 0, 255), -1)
                        
                        plot_number = plot.get('plot_number')
                        if plot_number is not None:
                            cx = sum([corners[c]['x'] for c in corners]) // 4
                            cy = sum([corners[c]['y'] for c in corners]) // 4
                            cv2.putText(blank_display, str(plot_number),
                                       (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    st.image(blank_display, channels="RGB",
                            caption=f"{len(st.session_state.plots)} plots (no background)",
                            use_container_width=True)
        
        with tab2:
            if editable_plot_viewer and background_url:
                editable_plot_viewer(
                    background_image_url=background_url,
                    plots=plots_for_viewer,
                    mode="points"
                )
                add_background_toggle_script()
            else:
                st.warning("⚠️ Editable viewer not available. Showing static image.")
                if st.session_state.get('show_background_step2', True):
                    st.image(st.session_state.detection_image, channels="BGR",
                            caption=f"{len(st.session_state.plots)} plots detected with red lines and dots",
                            use_container_width=True)
                else:
                    # Show blank image with plot overlays
                    blank_display = np.ones((height, width, 3), dtype=np.uint8) * 255
                    
                    # Draw plots on blank image (points mode)
                    for plot in st.session_state.plots:
                        corners = plot.get('corners', {})
                        if not corners or len(corners) < 4:
                            continue
                        
                        # Draw only red dots at corners (no lines)
                        cv2.circle(blank_display, (corners['A']['x'], corners['A']['y']), 5, (0, 0, 255), -1)
                        cv2.circle(blank_display, (corners['B']['x'], corners['B']['y']), 5, (0, 0, 255), -1)
                        cv2.circle(blank_display, (corners['C']['x'], corners['C']['y']), 5, (0, 0, 255), -1)
                        cv2.circle(blank_display, (corners['D']['x'], corners['D']['y']), 5, (0, 0, 255), -1)
                        
                        plot_number = plot.get('plot_number')
                        if plot_number is not None:
                            cx = sum([corners[c]['x'] for c in corners]) // 4
                            cy = sum([corners[c]['y'] for c in corners]) // 4
                            cv2.putText(blank_display, str(plot_number),
                                       (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    st.image(blank_display, channels="RGB",
                            caption=f"{len(st.session_state.plots)} plots (no background)",
                            use_container_width=True)
        
        # Button to apply changes from the editable viewer
        st.divider()
        
        # Show instruction
        st.markdown(
            """
            <div style="
                background:#fee2e2;
                color:#b91c1c;
                border:1px solid #fecaca;
                border-radius:8px;
                padding:12px 14px;
                font-weight:700;
            ">
                ⚠️ <strong>How to use :</strong> Edit plots above, click <em>Save Changes</em> — the yellow box already contains the copied JSON. Paste it below, then click save button and then click <em>Detect Coordinates</em>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Renumber Plot Section
        with st.expander("Renumber Plots", expanded=False):
            st.write("**Sequential Renumbering with Snake Pattern:**")
            st.write("1. Select the plot that should be **Plot 1**")
            st.write("2. Choose numbering mode (Row-wise or Column-wise)")
            st.write("3. Choose the starting direction")
            st.write("4. All plots will be renumbered in a snake/zigzag pattern")
            
            st.info(" **Tip:** Uses snake pattern (alternating directions) for efficient numbering!\n\n" + 
                   "• **Row-wise:** Numbers across row, then alternates direction each row\n\n" +
                   "• **Column-wise:** Numbers down/up column, then alternates direction each column")
            
            if st.session_state.plots:
                # Get list of plots for selection
                plot_options = {f"{p.get('plot_id', 'Unknown')} (Current: Plot {p.get('plot_number', 'N/A')})": p.get('plot_id') 
                               for p in st.session_state.plots}
                plot_options_list = list(plot_options.keys())
                
                col_renumber1, col_renumber2, col_renumber3 = st.columns(3)
                
                with col_renumber1:
                    selected_plot_display = st.selectbox(
                        "📍 Select Plot 1:",
                        options=plot_options_list,
                        key="renumber_plot_select",
                        help="Choose which plot should be numbered as Plot 1"
                    )
                    selected_plot_id = plot_options.get(selected_plot_display)
                
                with col_renumber2:
                    numbering_mode = st.radio(
                        "🔄 Numbering Mode:",
                        options=['row', 'column'],
                        format_func=lambda x: '📊 Row-wise (→)' if x == 'row' else '📋 Column-wise (↓)',
                        key="renumber_mode",
                        help="Row-wise: number across rows, then move to next row\nColumn-wise: number down columns, then move to next column"
                    )
                
                with col_renumber3:
                    if numbering_mode == 'row':
                        direction = st.radio(
                            "🔄 Starting Direction:",
                            options=['left', 'right'],
                            format_func=lambda x: '▶️ Left To Right' if x == 'left' else '◀️ Right To Left',
                            key="renumber_direction",
                            help="LTR: First row Left→Right, second row Right→Left (snake pattern)\nRTL: First row Right→Left, second row Left→Right"
                        )
                    else:
                        direction = st.radio(
                            "🔄 Starting Direction:",
                            options=['left', 'right'],
                            format_func=lambda x: '⬇️ Top to Bottom ↑↓' if x == 'left' else '⬆️ Bottom to Top ↓↑',
                            key="renumber_direction",
                            help="Top to Bottom: First column goes DOWN, next goes UP, then DOWN, UP...\nBottom to Top: First column goes UP, next goes DOWN, then UP, DOWN..."
                        )
                
                # Optional: Manual Plot 2 selection
                st.divider()
                with st.expander("Manual Plot #2 Selection (Optional)", expanded=False):
                    st.markdown("""
                    **When to use:** Plot 1 spans multiple columns/rows with multiple plots below/beside it
                    
                    **Examples:**
                    - Plot 1 is wider and has 2+ plots directly below it
                    - Plot 1 covers multiple sub-columns
                    - You want to control which plot becomes Plot 2
                    
                    **Workflow:** ✓ Select Plot 1 above → ✓ Enable this → ✓ Select Plot 2 → Click Renumber
                    """)
                    
                    enable_manual_plot2 = st.checkbox(
                        "✓ Enable manual Plot #2 selection",
                        value=False,
                        key="enable_manual_plot2",
                        help="Check this BEFORE renumbering if Plot 1 spans multiple plots below/beside it"
                    )
                    
                    second_plot_id = None
                    if enable_manual_plot2:
                        st.success("Great! Now select which plot should be Plot #2:")
                        # Filter out Plot 1 from options
                        plot_options_for_plot2 = {k: v for k, v in plot_options.items() if v != selected_plot_id}
                        plot_options_list_plot2 = list(plot_options_for_plot2.keys())
                        
                        if plot_options_list_plot2:
                            selected_plot2_display = st.selectbox(
                                "📍 Select Plot #2:",
                                options=plot_options_list_plot2,
                                key="renumber_plot2_select",
                                help="Choose the plot that should be numbered as Plot 2. Works for both row-wise and column-wise modes!"
                            )
                            second_plot_id = plot_options_for_plot2.get(selected_plot2_display)
                            
                            st.info(f"Ready to renumber: {selected_plot_id} → Plot 1, {second_plot_id} → Plot 2")
                        else:
                            st.warning("No other plots available.")
                
                if st.button("Renumber All Plots", type="primary", key="btn_renumber_plots"):
                    if selected_plot_id:
                        # Store original numbers for potential rollback
                        original_numbers = {p.get('plot_id'): p.get('plot_number') for p in st.session_state.plots}
                        
                        # Apply renumbering
                        st.session_state.plots = renumber_plots_sequentially(
                            st.session_state.plots.copy(),
                            selected_plot_id,
                            direction,
                            numbering_mode,
                            second_plot_id
                        )
                        
                        # Update geo_plots with new plot numbers (match by plot_id)
                        if st.session_state.geo_plots:
                            # Create a mapping of plot_id to new plot_number
                            plot_id_to_number = {p.get('plot_id'): p.get('plot_number') for p in st.session_state.plots}
                            
                            # Update geo_plots
                            for geo_plot in st.session_state.geo_plots:
                                plot_id = geo_plot.get('plot_id')
                                if plot_id in plot_id_to_number:
                                    geo_plot['plot_number'] = plot_id_to_number[plot_id]
                            
                            print(f"Updated {len(st.session_state.geo_plots)} geo_plots with new plot numbers")
                        
                        # Regenerate image with new numbers
                        if st.session_state.detection_image is not None:
                            original_img = st.session_state.detection_image.copy()
                            if len(original_img.shape) == 3:
                                height, width = original_img.shape[:2]
                                display_img = np.ones((height, width, 3), dtype=np.uint8) * 255
                            else:
                                height, width = original_img.shape
                                display_img = np.ones((height, width, 3), dtype=np.uint8) * 255
                            
                            for plot in st.session_state.plots:
                                corners = plot.get('corners', {})
                                if not corners:
                                    continue
                                pts = np.array([
                                    [corners['A']['x'], corners['A']['y']],
                                    [corners['B']['x'], corners['B']['y']],
                                    [corners['C']['x'], corners['C']['y']],
                                    [corners['D']['x'], corners['D']['y']]
                                ], np.int32)
                                cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
                                cv2.circle(display_img, (corners['A']['x'], corners['A']['y']), 4, (0, 0, 255), -1)
                                cv2.circle(display_img, (corners['B']['x'], corners['B']['y']), 4, (0, 0, 255), -1)
                                cv2.circle(display_img, (corners['C']['x'], corners['C']['y']), 4, (0, 0, 255), -1)
                                cv2.circle(display_img, (corners['D']['x'], corners['D']['y']), 4, (0, 0, 255), -1)
                                
                                plot_number = plot.get('plot_number')
                                if plot_number is not None:
                                    cx = sum([corners[c]['x'] for c in corners]) // 4
                                    cy = sum([corners[c]['y'] for c in corners]) // 4
                                    cv2.putText(display_img, str(plot_number),
                                               (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            
                            st.session_state.detection_image = display_img
                            st.session_state.detected_overlay_url = ndarray_to_data_url(display_img)
                            # Don't overwrite original_image_base64 - preserve for background toggle
                            if st.session_state.original_image_base64 is None:
                                st.session_state.original_image_base64 = ndarray_to_data_url(display_img)
                            st.session_state.brochure_overlay_url = st.session_state.detected_overlay_url
                        
                        # Success message
                        mode_text = "row-wise" if numbering_mode == 'row' else "column-wise"
                        if numbering_mode == 'row':
                            direction_text = "LTR→RTL→LTR (snake)" if direction == 'left' else "RTL→LTR→RTL (snake)"
                        else:
                            direction_text = "Top→Bottom→Top (↓↑↓)" if direction == 'left' else "Bottom→Top→Bottom (↑↓↑)"
                        
                        success_msg = f"Successfully renumbered {len(st.session_state.plots)} plots!"
                        success_msg += f"\n\n🎯 **Starting Plot:** {selected_plot_id} → Plot 1"
                        if second_plot_id:
                            success_msg += f"\n🎯 **Plot #2:** {second_plot_id} → Plot 2 (manually selected)"
                        success_msg += f"\n📊 **Mode:** {mode_text.title()} - {direction_text}"
                        success_msg += f"\n🔢 **Pattern:** Snake pattern with synchronized plot IDs"
                        if st.session_state.geo_plots:
                            success_msg += f"\n\n📍 **Updated in:**\n-  Step 2 (Regenerated Image)\n-  Step 5 (Brochure Preview)\n- ✅ Step 8 (Google Map Preview)"
                        else:
                            success_msg += f"\n\n📍 **Updated in:**\n-  Step 2 (Regenerated Image)\n-  Step 5 (Brochure Preview)\n- ℹ️ Step 8 (Map Preview) - will update when you generate the map"
                        
                        st.success(success_msg)
                        st.rerun()
                    else:
                        st.error("Please select a plot to be Plot 1.")
            else:
                st.warning("No plots available for renumbering.")
        
        # Combined Save + Detect action
        col_text, _ = st.columns([0.6, 0.4])
        with col_text:
            coord_json = st.text_area(
                "Paste coordinates JSON here:",
                height=110,
                key="coord_json_input_step2",
                placeholder='Click "Save Changes" in the viewer above, then copy the JSON from the yellow box and paste it here.',
                help="After editing plots, click 'Save Changes' to get the JSON coordinates"
            )
        
        # Left-aligned Save & Detect button (compact)
        if st.button("Detect Coordinates", type="primary", use_container_width=False):
            if coord_json and coord_json.strip():
                try:
                    import json
                    data = json.loads(coord_json)
                    if isinstance(data, dict) and 'plots' in data:
                        plots_data = data['plots']
                    elif isinstance(data, list):
                        plots_data = data
                    else:
                        plots_data = []
                    
                    if plots_data:
                        # Create a mapping of plot IDs from the saved data
                        plot_dict = {p['id']: p for p in plots_data}
                        updated_plot_ids = set(plot_dict.keys())
                        
                        # Create a mapping of plots by their corners (for matching after renumbering)
                        def get_corner_key(corners):
                            """Create a unique key from plot corners for matching"""
                            if not corners:
                                return None
                            corner_list = []
                            for label in ['A', 'B', 'C', 'D']:
                                if label in corners:
                                    corner_list.append((corners[label].get('x', 0), corners[label].get('y', 0)))
                            return tuple(sorted(corner_list)) if corner_list else None
                        
                        # Build corner mapping from saved data
                        saved_corner_map = {}
                        for plot_data in plots_data:
                            points = plot_data.get('points', [])
                            if len(points) >= 3:
                                corners = {}
                                for i, label in enumerate(['A', 'B', 'C', 'D']):
                                    if i < len(points):
                                        corners[label] = {'x': int(points[i]['x']), 'y': int(points[i]['y'])}
                                    else:
                                        last_point = points[-1]
                                        corners[label] = {'x': int(last_point['x']), 'y': int(last_point['y'])}
                                corner_key = get_corner_key(corners)
                                if corner_key:
                                    saved_corner_map[corner_key] = plot_data
                        
                        existing_plot_ids = {p.get('plot_id') for p in st.session_state.plots}
                        
                        updated_count = 0
                        added_count = 0
                        deleted_count = 0
                        
                        # Track which plots were matched (by ID or by corners)
                        matched_plot_ids = set()
                        
                        # Update existing plots - try to match by ID first, then by corners
                        for plot in st.session_state.plots[:]:
                            plot_id = plot.get('plot_id')
                            matched = False
                            
                            if plot_id in plot_dict:
                                updated_plot = plot_dict[plot_id]
                                matched_plot_ids.add(plot_id)
                                matched = True
                            else:
                                plot_corners = plot.get('corners', {})
                                corner_key = get_corner_key(plot_corners)
                                if corner_key and corner_key in saved_corner_map:
                                    updated_plot = saved_corner_map[corner_key]
                                    matched_plot_ids.add(updated_plot.get('id'))
                                    matched = True
                            
                            if matched:
                                points = updated_plot.get('points', [])
                                
                                if len(points) >= 3:
                                    # CRITICAL: Store ALL points, not just first 4
                                    # Store points array for plots with more than 4 points
                                    plot['points'] = [{'x': int(p['x']), 'y': int(p['y'])} for p in points]
                                    
                                    # Also create corners for backward compatibility (use first 4 points)
                                    corners = {}
                                    corner_labels = ['A', 'B', 'C', 'D']
                                    for i, label in enumerate(corner_labels):
                                        if i < len(points):
                                            corners[label] = {'x': int(points[i]['x']), 'y': int(points[i]['y'])}
                                        else:
                                            # If less than 4 points, duplicate last point
                                            last_point = points[-1]
                                            corners[label] = {'x': int(last_point['x']), 'y': int(last_point['y'])}
                                    plot['corners'] = corners
                                    
                                    if 'plot_number' in updated_plot:
                                        new_plot_number = updated_plot.get('plot_number')
                                        if new_plot_number is not None:
                                            plot['plot_number'] = int(new_plot_number)
                                    
                                    new_plot_id = updated_plot.get('id')
                                    if new_plot_id and new_plot_id != plot_id:
                                        plot['plot_id'] = new_plot_id
                                        if plot_id in st.session_state.plot_statuses:
                                            status = st.session_state.plot_statuses[plot_id]
                                            st.session_state.plot_statuses[new_plot_id] = status
                                            del st.session_state.plot_statuses[plot_id]
                                    
                                    updated_count += 1
                        
                        # Remove deleted plots
                        plots_to_remove = existing_plot_ids - matched_plot_ids
                        if plots_to_remove:
                            for removed_id in plots_to_remove:
                                if removed_id in st.session_state.plot_statuses:
                                    del st.session_state.plot_statuses[removed_id]
                            
                            st.session_state.plots = [p for p in st.session_state.plots 
                                                     if p.get('plot_id') not in plots_to_remove]
                            deleted_count = len(plots_to_remove)
                        
                        # Add new plots
                        plots_to_add = updated_plot_ids - matched_plot_ids
                        for plot_id in plots_to_add:
                            new_plot_data = plot_dict[plot_id]
                            points = new_plot_data.get('points', [])
                            
                            if len(points) >= 3:
                                # CRITICAL: Store ALL points, not just first 4
                                all_points = [{'x': int(p['x']), 'y': int(p['y'])} for p in points]
                                
                                # Also create corners for backward compatibility (use first 4 points)
                                corners = {}
                                corner_labels = ['A', 'B', 'C', 'D']
                                for i, label in enumerate(corner_labels):
                                    if i < len(points):
                                        corners[label] = {'x': int(points[i]['x']), 'y': int(points[i]['y'])}
                                    else:
                                        # If less than 4 points, duplicate last point
                                        last_point = points[-1]
                                        corners[label] = {'x': int(last_point['x']), 'y': int(last_point['y'])}
                                
                                new_plot = {
                                    'plot_id': new_plot_data.get('id', plot_id),
                                    'points': all_points,  # CRITICAL: Store all points
                                    'plot_number': new_plot_data.get('plot_number', len(st.session_state.plots) + 1),
                                    'corners': corners
                                }
                                st.session_state.plots.append(new_plot)
                                added_count += 1
                        
                        # Regenerate image if any changes
                        if updated_count > 0 or added_count > 0 or deleted_count > 0:
                            if st.session_state.detection_image is not None:
                                original_img = st.session_state.detection_image.copy()
                                if len(original_img.shape) == 3:
                                    height, width = original_img.shape[:2]
                                    display_img = np.ones((height, width, 3), dtype=np.uint8) * 255
                                else:
                                    height, width = original_img.shape
                                    display_img = np.ones((height, width, 3), dtype=np.uint8) * 255
                                
                                for plot in st.session_state.plots:
                                    corners = plot.get('corners', {})
                                    if not corners:
                                        continue
                                    required_corners = ['A', 'B', 'C', 'D']
                                    if not all(corner in corners for corner in required_corners):
                                        continue
                                    try:
                                        pts = np.array([
                                            [int(corners['A']['x']), int(corners['A']['y'])],
                                            [int(corners['B']['x']), int(corners['B']['y'])],
                                            [int(corners['C']['x']), int(corners['C']['y'])],
                                            [int(corners['D']['x']), int(corners['D']['y'])]
                                        ], np.int32)
                                    except (KeyError, ValueError, TypeError):
                                        continue
                                    cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
                                    cv2.circle(display_img, (corners['A']['x'], corners['A']['y']), 4, (0, 0, 255), -1)
                                    cv2.circle(display_img, (corners['B']['x'], corners['B']['y']), 4, (0, 0, 255), -1)
                                    cv2.circle(display_img, (corners['C']['x'], corners['C']['y']), 4, (0, 0, 255), -1)
                                    cv2.circle(display_img, (corners['D']['x'], corners['D']['y']), 4, (0, 0, 255), -1)
                                    plot_number = plot.get('plot_number')
                                    if plot_number is not None:
                                        cx = sum([corners[c]['x'] for c in corners]) // 4
                                        cy = sum([corners[c]['y'] for c in corners]) // 4
                                        cv2.putText(display_img, str(plot_number),
                                                   (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                
                                st.session_state.detection_image = display_img
                                st.session_state.detected_overlay_url = ndarray_to_data_url(display_img)
                                if st.session_state.original_image_base64 is None:
                                    st.session_state.original_image_base64 = ndarray_to_data_url(display_img)
                                st.session_state.brochure_overlay_url = st.session_state.detected_overlay_url
                            
                            msg_parts = []
                            if updated_count > 0:
                                msg_parts.append(f"updated {updated_count}")
                            if added_count > 0:
                                msg_parts.append(f"added {added_count}")
                            if deleted_count > 0:
                                msg_parts.append(f"deleted {deleted_count}")
                            msg = f"Applied changes: {', '.join(msg_parts)} plot(s)! The image has been regenerated."
                            st.success(msg)
                            st.session_state.coordinates_detected = True
                            st.session_state.edits_made_in_step2 = True
                            st.session_state.show_next_warning_step2 = False  # Reset warning
                            st.rerun()
                        else:
                            st.warning("No matching plots found to update. Make sure the plot IDs match.")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON format: {e}")
                except Exception as e:
                    st.error(f"Error applying coordinates: {e}")
                    st.exception(e)
            else:
                # No JSON pasted; just mark as ready using current data
                st.session_state.coordinates_detected = True
                st.session_state.show_next_warning_step2 = False  # Reset warning
        
        # Show success message if coordinates are detected
        if st.session_state.get('coordinates_detected', False):
            st.success("Coordinates ready! You can proceed to the next step.")
        
        # Show warning if Next was clicked without coordinates detected
        if st.session_state.get('show_next_warning_step2', False):
            col_warn1, col_warn2, col_warn3 = st.columns([8, 1, 1])
            with col_warn1:
                st.write("")  # Spacer
            with col_warn2:
                st.write("")  # Spacer
            with col_warn3:
                st.markdown("""
                <div style="
                    background-color: #fff3cd;
                    border: 1px solid #ffc107;
                    border-radius: 8px;
                    padding: 12px 16px;
                    margin-bottom: 12px;
                    text-align: center;
                    font-size: 12px;
                    color: #856404;
                    line-height: 1.5;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    min-width: 200px;
                ">
                    <div style="font-weight: 600; margin-bottom: 4px;">⚠️ Please ensure</div>
                    <div>coordinates are detected first!</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Navigation buttons for Step 2 - in same row
        col_btn1, col_btn2, col_btn3 = st.columns([8, 1, 1])
        with col_btn1:
            st.write("")  # Spacer
        with col_btn2:
            if st.button("Prev", type="primary", key="prev_step2"):
                st.session_state.current_step = 1  # Go back to Step 1
                st.session_state.show_next_warning_step2 = False  # Reset warning
                st.rerun()
        with col_btn3:
            if st.button("Next", type="primary", key="next_step2"):
                if st.session_state.get('coordinates_detected', False):
                    # If edits were made, try to apply them
                    if st.session_state.get('edits_made_in_step2', False):
                        # Note: We can't directly read localStorage from Python
                        # The user should use Step 3's grid to make precise edits
                        st.info("💡 Tip: Use Step 3's grid table to make precise coordinate adjustments if needed.")
                    st.session_state.current_step = 3  # Go to Step 3 (Detect Coordinates)
                    st.session_state.show_next_warning_step2 = False  # Reset warning
                    st.rerun()
                else:
                    # Show warning when Next is clicked without coordinates
                    st.session_state.show_next_warning_step2 = True
                    st.rerun()
    else:
        st.info("Please upload and detect plots first in Step 1.")

# STEP 3: Detect Coordinates (shows editable grid table)
elif current_step == 3:
    if st.session_state.plots:
        st.subheader("Edit Plot Numbers & Coordinates")
        st.write("Use the table below to correct plot numbers and point coordinates. The table dynamically shows all points (A, B, C, D, E, F, ...) based on the maximum number of points across all plots. All columns except Plot ID are editable.")
        
        # Build DataFrame for editable table
        # CRITICAL: Dynamically determine max points across all plots to show all columns
        sorted_plots = sorted(st.session_state.plots, key=lambda x: (x.get('plot_number') is None, x.get('plot_number') if x.get('plot_number') is not None else 0, x.get('plot_id')))
        
        # Find maximum number of points across all plots
        max_points = 0
        for p in sorted_plots:
            points = p.get('points', [])
            if points:
                max_points = max(max_points, len(points))
            else:
                # Fallback: count corners
                corners = p.get('corners', {})
                max_points = max(max_points, len(corners))
        
        # Generate column labels (A, B, C, D, E, F, ...)
        corner_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
        point_labels = corner_labels[:max_points] if max_points <= len(corner_labels) else corner_labels + [chr(ord('Q') + i) for i in range(max_points - len(corner_labels))]
        
        # Build DataFrame with dynamic columns for all points
        plots_data = []
        for p in sorted_plots:
            row = {
                "plot_id": p.get('plot_id'),
                "plot_number": p.get('plot_number'),
            }
            
            # Get points array if available, otherwise convert corners to points
            points = p.get('points', [])
            if not points:
                # Fallback: Convert corners to points
                corners = p.get('corners', {})
                points = []
                for label in ['A', 'B', 'C', 'D']:
                    if label in corners:
                        points.append({'x': corners[label].get('x', 0), 'y': corners[label].get('y', 0)})
            
            # Add columns for all points (up to max_points)
            for i, label in enumerate(point_labels):
                if i < len(points):
                    row[f"{label}_x"] = points[i].get('x', 0)
                    row[f"{label}_y"] = points[i].get('y', 0)
                else:
                    row[f"{label}_x"] = None
                    row[f"{label}_y"] = None
            
            plots_data.append(row)
        
        plots_df_flat = pd.DataFrame(plots_data)
        
        # Build column config dynamically
        column_config = {
            "plot_id": st.column_config.TextColumn("Plot ID", disabled=True),
            "plot_number": st.column_config.NumberColumn("Plot Number", min_value=1, max_value=9999, step=1),
        }
        
        # Add columns for all points
        for label in point_labels:
            column_config[f"{label}_x"] = st.column_config.NumberColumn(f"{label} → x", min_value=0, step=1, help=f"Point {label}, x coordinate")
            column_config[f"{label}_y"] = st.column_config.NumberColumn(f"{label} → y", min_value=0, step=1, help=f"Point {label}, y coordinate")
        
        edited_df = st.data_editor(
            plots_df_flat,
            hide_index=True,
            column_config=column_config,
            use_container_width=True,
            num_rows="fixed",
            height=300
        )
        
        # Use edited_df directly for processing
        edited_df_processed = edited_df
        
        # Check for duplicate plot numbers in edited data
        duplicate_mask = edited_df_processed['plot_number'].duplicated(keep=False) & edited_df_processed['plot_number'].notna()
        duplicate_numbers = edited_df_processed[duplicate_mask]
        if not duplicate_numbers.empty:
            duplicate_nums = sorted(duplicate_numbers['plot_number'].unique())
            duplicate_plot_ids = duplicate_numbers['plot_id'].tolist()
            st.error(f"⚠️ **Duplicate plot numbers detected:** {', '.join(map(str, duplicate_nums))} in plots: {', '.join(duplicate_plot_ids)}")
            
            # Show a styled view with red highlighting for duplicates
            def highlight_duplicates(row):
                """Highlight rows with duplicate plot numbers in red."""
                if pd.notna(row['plot_number']) and row['plot_number'] in duplicate_nums:
                    return ['background-color: #ffcccc'] * len(row)  # Light red background
                return [''] * len(row)
            
            with st.expander(" View duplicates highlighted in red", expanded=True):
                st.markdown("*Rows with duplicate plot numbers are highlighted in red:*")
                st.dataframe(edited_df_processed.style.apply(highlight_duplicates, axis=1), use_container_width=True)
        else:
            st.success("✓ No duplicate plot numbers found")
        
        col_save, col_revert, col_spacer = st.columns([1.5, 1.5, 5])
        with col_save:
            if st.button("Save Changes", type="primary"):
                # Apply edited numbers and coordinates back to session_state.plots
                plot_updates = {}
                for _, row in edited_df_processed.iterrows():
                    plot_id = row['plot_id']
                    # Find original plot to preserve coordinates if needed
                    original_plot = next((p for p in st.session_state.plots if p.get('plot_id') == plot_id), None)
                    original_points = original_plot.get('points', []) if original_plot else []
                    original_corners = original_plot.get('corners', {}) if original_plot else {}
                    
                    # Update plot number
                    plot_number = int(row['plot_number']) if pd.notna(row['plot_number']) else None
                    
                    # Update points, preserving original if new value is invalid/NaN
                    def get_coord(row, coord_key, original_value):
                        """Get coordinate value, using original if new value is invalid."""
                        if coord_key in row and pd.notna(row[coord_key]):
                            try:
                                return int(row[coord_key])
                            except (ValueError, TypeError):
                                return original_value
                        return original_value
                    
                    # CRITICAL: Collect all points from the grid (A, B, C, D, E, F, ...)
                    points = []
                    corners = {}
                    
                    for label in point_labels:
                        x_key = f'{label}_x'
                        y_key = f'{label}_y'
                        
                        # Get coordinates from grid, or use original if not available
                        if x_key in row and y_key in row:
                            x_val = get_coord(row, x_key, None)
                            y_val = get_coord(row, y_key, None)
                            
                            if x_val is not None and y_val is not None:
                                points.append({'x': x_val, 'y': y_val})
                                
                                # Also update corners for backward compatibility (first 4 points)
                                if label in ['A', 'B', 'C', 'D']:
                                    corners[label] = {'x': x_val, 'y': y_val}
                    
                    # If no points from grid, fall back to original
                    if not points:
                        if original_points:
                            points = original_points.copy()
                        else:
                            # Convert corners to points
                            for label in ['A', 'B', 'C', 'D']:
                                if label in original_corners:
                                    points.append({
                                        'x': original_corners[label].get('x', 0),
                                        'y': original_corners[label].get('y', 0)
                                    })
                                    corners[label] = original_corners[label]
                    
                    # Ensure we have at least 3 points
                    if len(points) < 3:
                        st.warning(f"⚠️ Plot {plot_id} has less than 3 points. Skipping update.")
                        continue
                    
                    plot_updates[plot_id] = {
                        'plot_number': plot_number,
                        'points': points,  # CRITICAL: Store all points
                        'corners': corners  # Also store corners for backward compatibility
                    }
                
                # Apply updates
                for p in st.session_state.plots:
                    if p.get('plot_id') in plot_updates:
                        update = plot_updates[p['plot_id']]
                        p['plot_number'] = update['plot_number']
                        p['points'] = update['points']  # CRITICAL: Store all points
                        p['corners'] = update['corners']  # Also update corners
                
                # Regenerate detection image with updated coordinates
                if st.session_state.detection_image is not None:
                    original_img = st.session_state.detection_image.copy()
                    # Redraw the image with updated coordinates
                    display_img = original_img.copy()
                    
                    for plot in st.session_state.plots:
                        # CRITICAL: Use 'points' array if available (supports more than 4 points)
                        points = plot.get('points', [])
                        if not points:
                            # Fallback: Convert corners to points
                            corners = plot.get('corners', {})
                            if not corners:
                                continue
                            corner_labels = ['A', 'B', 'C', 'D']
                            points = []
                            for label in corner_labels:
                                if label in corners:
                                    points.append({'x': corners[label].get('x', 0), 'y': corners[label].get('y', 0)})
                        
                        if len(points) < 3:
                            continue
                        
                        # Create points array with all points (not just 4)
                        pts = np.array([
                            [int(p['x']), int(p['y'])] for p in points
                        ], np.int32)
                        
                        # Red lines for plot boundaries
                        cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
                        
                        # Draw red dots at each point (all points, not just 4)
                        for point in points:
                            cv2.circle(display_img, (int(point['x']), int(point['y'])), 4, (0, 0, 255), -1)
                        
                        # Draw plot number - calculate center from all points
                        cx = sum([int(p['x']) for p in points]) // len(points)
                        cy = sum([int(p['y']) for p in points]) // len(points)
                        cv2.putText(display_img, str(plot['plot_number']),
                                   (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    st.session_state.detection_image = display_img
                    st.session_state.detected_overlay_url = ndarray_to_data_url(display_img)
                    # Don't overwrite original_image_base64 - preserve for background toggle
                    if st.session_state.original_image_base64 is None:
                        st.session_state.original_image_base64 = ndarray_to_data_url(display_img)
                
                # Reset geo_plots so user regenerates with consistent numbers
                st.session_state.geo_plots = []
                st.success("Applied changes. The image will be updated in Step 4.")
                st.rerun()
        with col_revert:
            if st.button("Revert Changes", type="primary"):
                st.info("Reverted UI edits. The table reflects current values from detection.")
                st.rerun()
        
        # Navigation buttons for Step 3
        col_btn1, col_btn2, col_btn3 = st.columns([8, 1, 1])
        with col_btn2:
            if st.button("Prev", type="primary", key="prev_step3"):
                st.session_state.current_step = 2  # Go back to Step 2
                st.rerun()
        with col_btn3:
            if st.button("Next", type="primary", key="next_step3"):
                st.session_state.current_step = 4  # Go to Step 4 (Preview Polygons)
                st.rerun()
    else:
        st.info("Please upload an image and detect plots first.")

# STEP 5: Preview in Brochure
elif current_step == 5:
    if st.session_state.plots:

        # Initialize plot statuses if not set
        for plot in st.session_state.plots:
            plot_id = plot.get('plot_id')
            if plot_id and plot_id not in st.session_state.plot_statuses:
                st.session_state.plot_statuses[plot_id] = random.choice(PLOT_STATUS_OPTIONS)
        
        # Always use uploaded image as background (no default background)
        background_image_url = ""
        uploaded_image_bytes = st.session_state.get('uploaded_image_bytes')
        
        if uploaded_image_bytes:
            try:
                uploaded_pil_image = Image.open(BytesIO(uploaded_image_bytes)).convert("RGB")
                background_image_url = pil_image_to_base64(uploaded_pil_image)
            except Exception as e:
                st.error(f"Failed to load uploaded image as background: {e}")
        else:
            st.warning("⚠️ Uploaded image not found. Please upload an image in Step 1 first.")
        
        # CRITICAL: Do NOT use detected overlay in brochure preview - only show background image
        # The brochure viewer will draw colored plot polygons directly, not from an overlay image
        # plot_overlay_url = None  # Explicitly set to None to show only background
        
        # Plot Status Controls - visible note only
        st.markdown(
            '<div style="margin: 6px 0 12px 0; font-weight:600; color:#1f2937;">'
            'Statuses are auto-assigned. Click below to randomize them anytime.'
            '</div>',
            unsafe_allow_html=True
        )
        
        if st.button("Randomize plot statuses", type="primary", key="randomize_statuses"):
            for plot_id in st.session_state.plot_statuses:
                st.session_state.plot_statuses[plot_id] = random.choice(PLOT_STATUS_OPTIONS)
            st.success("Plot statuses randomized.")
        
        # Build plot payload with current statuses
        def build_plot_payload(plot):
            # CRITICAL: Use 'points' array if available (supports more than 4 points)
            points = plot.get('points', [])
            if not points:
                # Fallback: Convert corners to points for backward compatibility
                corners = plot.get('corners', {})
                ordered_labels = ['A', 'B', 'C', 'D']
                points = []
                for label in ordered_labels:
                    corner = corners.get(label, {})
                    if corner:
                        points.append({'x': corner.get('x', 0), 'y': corner.get('y', 0)})
            geo_lat = None
            geo_lon = None
            geo_plot = next((gp for gp in st.session_state.geo_plots if gp.get('plot_id') == plot.get('plot_id')), None) if st.session_state.geo_plots else None
            if geo_plot:
                lat_vals = [corner.get('lat') for corner in geo_plot['corners'].values() if corner.get('lat') is not None]
                lon_vals = [corner.get('lon') for corner in geo_plot['corners'].values() if corner.get('lon') is not None]
                if lat_vals and lon_vals:
                    geo_lat = sum(lat_vals) / len(lat_vals)
                    geo_lon = sum(lon_vals) / len(lon_vals)
            
            # Get plot_number - ensure it's always set correctly
            plot_number = plot.get('plot_number')
            plot_id = plot.get('plot_id', 'unknown')
            
            # If plot_number is None or 0, try to extract it from plot_id
            if plot_number is None or plot_number == 0:
                import re
                id_match = re.search(r'(\d+)', str(plot_id))
                if id_match:
                    plot_number = int(id_match.group(1))
                else:
                    plot_number = None
            
            return {
                "id": plot_id,
                "plot_number": plot_number,  # Include plot number for display (always set correctly)
                "points": points,
                "lat": geo_lat,
                "lon": geo_lon,
                "status": st.session_state.plot_statuses.get(plot_id, "available")
            }
        
        # Filter plots to only include those with valid corners
        def has_valid_corners(plot):
            corners = plot.get('corners', {})
            if not corners:
                return False
            required_corners = ['A', 'B', 'C', 'D']
            if not all(corner in corners for corner in required_corners):
                return False
            # Validate coordinates are numbers
            try:
                for corner in required_corners:
                    int(corners[corner].get('x', 0))
                    int(corners[corner].get('y', 0))
                return True
            except (KeyError, ValueError, TypeError):
                return False
        
        plots_payload = [build_plot_payload(plot) for plot in st.session_state.plots if has_valid_corners(plot)]
        
        # Auto-render interactive Fabric Canvas
        # CRITICAL: Pass None for plot_overlay_url to show only background image (no Step 4 polygon overlay)
        if background_image_url and plots_payload:
            brochure_viewer(
                background_image_url=background_image_url,
                plots=plots_payload,
                plot_overlay_url=None  # Explicitly None - only show background, not Step 4 overlay
            )
            
            # Initialize status updates storage
            if 'pending_status_updates' not in st.session_state:
                st.session_state.pending_status_updates = {}
            
            # JavaScript to listen for status update messages and store them
            status_updates_container = st.empty()
            
            # Add message listener script
            st.components.v1.html("""
            <div id="status-updates-container"></div>
            <script>
                (function() {
                    // Store for pending updates
                    window.pendingStatusUpdates = window.pendingStatusUpdates || {};
                    
                    // Listen for messages from the brochure viewer iframe
                    window.addEventListener('message', function(event) {
                        if (event.data && event.data.type === 'plot_status_update') {
                            console.log('Received status update:', event.data);
                            
                            // Store the update
                            window.pendingStatusUpdates[event.data.plotId] = {
                                plotId: event.data.plotId,
                                newStatus: event.data.newStatus,
                                oldStatus: event.data.oldStatus,
                                timestamp: event.data.timestamp
                            };
                            
                            // Update the container with current updates count
                            let container = document.getElementById('status-updates-container');
                            if (container) {
                                let count = Object.keys(window.pendingStatusUpdates).length;
                                container.innerHTML = '<input type="hidden" id="status-updates-count" value="' + count + '">';
                            }
                            
                            // Show notification
                            showStatusUpdateNotification(event.data.plotId, event.data.newStatus);
                            
                            // Apply update immediately by updating URL query parameter
                            applyStatusUpdateImmediately(event.data.plotId, event.data.newStatus);
                        }
                    });
                    
                    function showStatusUpdateNotification(plotId, newStatus) {
                        let notification = document.getElementById('status-update-notification');
                        if (!notification) {
                            notification = document.createElement('div');
                            notification.id = 'status-update-notification';
                            notification.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #4CAF50; color: white; padding: 12px 20px; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); z-index: 10000; font-family: Arial, sans-serif; font-size: 14px; max-width: 300px;';
                            document.body.appendChild(notification);
                        }
                        notification.innerHTML = '✅ Plot ' + plotId + ' status changed to <strong>' + newStatus + '</strong><br><small>Changes saved automatically</small>';
                        notification.style.display = 'block';
                        
                        setTimeout(function() {
                            notification.style.display = 'none';
                        }, 4000);
                    }
                    
                    function applyStatusUpdateImmediately(plotId, newStatus) {
                        // Trigger Streamlit to read and apply by setting a query parameter
                        const url = new URL(window.location);
                        url.searchParams.set('status_update', plotId + ':' + newStatus);
                        url.searchParams.set('_timestamp', Date.now());
                        // Use history API to update URL and trigger Streamlit rerun
                        window.history.replaceState({}, '', url);
                        
                        // Trigger a small delay then reload to apply
                        setTimeout(function() {
                            window.location.reload();
                        }, 500);
                    }
                    
                    // Make updates accessible to Streamlit
                    window.getPendingStatusUpdates = function() {
                        return window.pendingStatusUpdates;
                    };
                    
                    window.clearPendingStatusUpdates = function() {
                        window.pendingStatusUpdates = {};
                        let container = document.getElementById('status-updates-container');
                        if (container) {
                            container.innerHTML = '<input type="hidden" id="status-updates-count" value="0">';
                        }
                    };
                })();
            </script>
            """, height=0)
            
            # Check for status updates from query parameters
            query_params = st.query_params
            if 'status_update' in query_params:
                status_update = query_params['status_update']
                try:
                    plot_id, new_status = status_update.split(':', 1)
                    if plot_id in st.session_state.plot_statuses and new_status in PLOT_STATUS_OPTIONS:
                        old_status = st.session_state.plot_statuses[plot_id]
                        st.session_state.plot_statuses[plot_id] = new_status
                        # Remove the query parameter to avoid reprocessing
                        st.query_params.pop('status_update', None)
                        st.query_params.pop('_timestamp', None)
                        st.rerun()
                except (ValueError, KeyError):
                    pass
            
            # Add UI info
            st.caption("**Note:** Click on any plot in the preview to change its status. Colors: 🟢 Available (Green), 🔵 Booked (Blue), 🔴 Sold (Red). Changes are saved automatically.")
            
            # Add controls panel for brochure viewer
            st.components.v1.html("""
            <div id="brochure-controls-panel" style="
                background: rgba(255, 255, 255, 0.98);
                padding: 20px 25px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                display: flex;
                gap: 35px;
                align-items: center;
                justify-content: center;
                margin: 25px auto;
                border: 1px solid #d0d0d0;
                font-family: Arial, sans-serif;
                max-width: 95%;
                box-sizing: border-box;
                overflow: visible;
            ">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-weight: bold; color: #333; margin-right: 5px;">Scale</span>
                    <button id="brochure-scale-up" style="
                        width: 40px; height: 40px; border-radius: 50%; border: none;
                        background: #6366f1; color: white; font-size: 18px; cursor: pointer;
                    ">+</button>
                    <button id="brochure-scale-down" style="
                        width: 40px; height: 40px; border-radius: 50%; border: none;
                        background: #6366f1; color: white; font-size: 18px; cursor: pointer;
                    ">-</button>
                    <input type="number" id="brochure-scale-input" value="0.1" step="0.01" min="0.01" max="1.0" placeholder="Scale Step" style="
                        width: 80px; height: 35px; border: 1px solid #ccc; border-radius: 4px;
                        padding: 5px; text-align: center; font-size: 13px; margin-left: 5px;
                    " title="Enter scale increment value (e.g., 0.25). Click +/- to scale by this amount.">
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-weight: bold; color: #333; margin-right: 5px;">Move</span>
                    <div style="display: grid; grid-template-columns: repeat(3, 40px); grid-template-rows: repeat(3, 40px); gap: 2px;">
                        <div></div>
                        <button id="brochure-move-up" style="
                            width: 40px; height: 40px; border-radius: 50%; border: none;
                            background: #6366f1; color: white; font-size: 16px; cursor: pointer;
                        ">▲</button>
                        <div></div>
                        <button id="brochure-move-left" style="
                            width: 40px; height: 40px; border-radius: 50%; border: none;
                            background: #6366f1; color: white; font-size: 16px; cursor: pointer;
                        ">◄</button>
                        <div></div>
                        <button id="brochure-move-right" style="
                            width: 40px; height: 40px; border-radius: 50%; border: none;
                            background: #6366f1; color: white; font-size: 16px; cursor: pointer;
                        ">►</button>
                        <div></div>
                        <button id="brochure-move-down" style="
                            width: 40px; height: 40px; border-radius: 50%; border: none;
                            background: #6366f1; color: white; font-size: 16px; cursor: pointer;
                        ">▼</button>
                        <div></div>
                    </div>
                    <input type="number" id="brochure-move-x-input" value="10" step="1" placeholder="X Step" style="
                        width: 70px; height: 35px; border: 1px solid #ccc; border-radius: 4px;
                        padding: 5px; text-align: center; font-size: 13px; margin-left: 5px;
                    " title="Enter move increment for X (pixels). Click arrow buttons to move by this amount.">
                    <input type="number" id="brochure-move-y-input" value="10" step="1" placeholder="Y Step" style="
                        width: 70px; height: 35px; border: 1px solid #ccc; border-radius: 4px;
                        padding: 5px; text-align: center; font-size: 13px;
                    " title="Enter move increment for Y (pixels). Click arrow buttons to move by this amount.">
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-weight: bold; color: #333; margin-right: 5px;">Rotate</span>
                    <input type="number" id="brochure-rotate-input" value="0" step="1" min="0" max="360" style="
                        width: 60px; height: 40px; border: 1px solid #ccc; border-radius: 4px;
                        padding: 5px; text-align: center; font-size: 14px;
                    ">
                </div>
            </div>
            <script>
                (function() {
                    let brochureIframe = null;
                    let brochureCanvas = null;
                    let plotPolygons = [];
                    let controlsConnected = false;
                    
                    // Wait for iframe to load and connect to Fabric.js canvas
                    function connectToBrochureCanvas() {
                        // Access parent window to find the brochure viewer iframe
                        let searchContext = window;
                        try {
                            // Try to access parent window (if we're in an iframe)
                            if (window.parent && window.parent !== window) {
                                searchContext = window.parent;
                            }
                        } catch(e) {
                            // Can't access parent, use current window
                            console.log('Cannot access parent window, using current window');
                        }
                        
                        const iframes = searchContext.document.querySelectorAll('iframe');
                        console.log('Searching', iframes.length, 'iframes for brochure canvas...');
                        
                        for (let iframe of iframes) {
                            try {
                                // Wait for iframe to be loaded
                                if (!iframe.contentWindow) {
                                    continue;
                                }
                                
                                const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                                const iframeWindow = iframe.contentWindow;
                                
                                // Check if this is the brochure viewer iframe by looking for fabric-canvas
                                const canvas = iframeDoc.getElementById('fabric-canvas');
                                
                                if (canvas) {
                                    console.log('Found fabric-canvas element!');
                                    
                                    // Wait a bit for Fabric.js to initialize
                                    if (iframeWindow.fabric && iframeWindow.fabricCanvas) {
                                        brochureIframe = iframe;
                                        brochureCanvas = iframeWindow.fabricCanvas;
                                        
                                        // Initialize transform state in iframe window
                                        if (!iframeWindow.currentScale) iframeWindow.currentScale = 1.0;
                                        if (!iframeWindow.currentOffsetX) iframeWindow.currentOffsetX = 0;
                                        if (!iframeWindow.currentOffsetY) iframeWindow.currentOffsetY = 0;
                                        if (!iframeWindow.currentRotation) iframeWindow.currentRotation = 0;
                                        
                                        // Get all plot polygons
                                        const allObjects = brochureCanvas.getObjects();
                                        plotPolygons = allObjects.filter(obj => obj.plotId);
                                        
                                        // Store original points ONCE - never update them
                                        plotPolygons.forEach(polygon => {
                                            if (!polygon.originalPoints) {
                                                // Deep copy to ensure we store the true original
                                                polygon.originalPoints = JSON.parse(JSON.stringify(polygon.points.map(p => ({x: p.x, y: p.y}))));
                                                polygon.originalLeft = polygon.left;
                                                polygon.originalTop = polygon.top;
                                                // Mark as set to prevent overwriting
                                                polygon.originalPointsSet = true;
                                            }
                                        });
                                        
                                        console.log('Brochure canvas found with', plotPolygons.length, 'plots!');
                                        return true;
                                    } else {
                                        // Canvas element exists but Fabric.js not ready yet
                                        console.log('Canvas element found but Fabric.js not ready, will retry...');
                                    }
                                }
                            } catch(e) {
                                // Cross-origin or other error - continue searching
                                // Only log if it's not a common cross-origin error
                                if (!e.message.includes('Blocked a frame') && !e.message.includes('cross-origin')) {
                                    console.log('Error accessing iframe:', e.message);
                                }
                            }
                        }
                        return false;
                    }
                    
                    // Apply transform function
                    function applyTransform() {
                        console.log('applyTransform called');
                        if (!brochureIframe) {
                            console.log('Iframe not ready');
                            return;
                        }
                        
                        try {
                            const iframeWindow = brochureIframe.contentWindow;
                            
                            // Try using the global function first (if available)
                            if (iframeWindow.applyBrochureTransform && typeof iframeWindow.applyBrochureTransform === 'function') {
                                console.log('Using global applyBrochureTransform function');
                                iframeWindow.applyBrochureTransform();
                                return;
                            }
                            
                            console.log('Using manual transform');
                            
                            // Fallback to manual transform
                            if (!brochureCanvas) {
                                brochureCanvas = iframeWindow.fabricCanvas;
                                if (!brochureCanvas) {
                                    console.log('Canvas not ready');
                                    return;
                                }
                            }
                            
                            const scale = iframeWindow.currentScale || 1.0;
                            const offsetX = iframeWindow.currentOffsetX || 0;
                            const offsetY = iframeWindow.currentOffsetY || 0;
                            const rotation = iframeWindow.currentRotation || 0;
                            
                            // Refresh polygon list in case it changed
                            const allObjects = brochureCanvas.getObjects();
                            const currentPolygons = allObjects.filter(obj => obj.plotId);
                            
                            currentPolygons.forEach(polygon => {
                                // Ensure originalPoints exists - set once from true original
                                if (!polygon.originalPoints || !polygon.originalPointsSet) {
                                    // Deep copy to ensure we store the true original
                                    polygon.originalPoints = JSON.parse(JSON.stringify(polygon.points.map(p => ({x: p.x, y: p.y}))));
                                    polygon.originalLeft = polygon.left;
                                    polygon.originalTop = polygon.top;
                                    polygon.originalPointsSet = true;
                                }
                                
                                // Always use the stored originalPoints for transformations
                                const origCenterX = polygon.originalPoints.reduce((sum, p) => sum + p.x, 0) / polygon.originalPoints.length;
                                const origCenterY = polygon.originalPoints.reduce((sum, p) => sum + p.y, 0) / polygon.originalPoints.length;
                                
                                // Transform from ORIGINAL points using absolute scale and offset values
                                const transformedPoints = polygon.originalPoints.map(p => {
                                    let x = p.x - origCenterX;
                                    let y = p.y - origCenterY;
                                    
                                    // Apply absolute scale (not cumulative)
                                    x *= scale;
                                    y *= scale;
                                    
                                    const angleRad = (rotation * Math.PI) / 180;
                                    const cos = Math.cos(angleRad);
                                    const sin = Math.sin(angleRad);
                                    const newX = x * cos - y * sin;
                                    const newY = x * sin + y * cos;
                                    
                                    // Apply absolute offset (not cumulative)
                                    return {
                                        x: newX + origCenterX + offsetX,
                                        y: newY + origCenterY + offsetY
                                    };
                                });
                                
                                polygon.set({
                                    points: transformedPoints,
                                    left: polygon.originalLeft + offsetX,
                                    top: polygon.originalTop + offsetY,
                                    angle: rotation
                                });
                            });
                            
                            brochureCanvas.renderAll();
                        } catch(err) {
                            console.error('Error applying transform:', err);
                        }
                    }
                    
                    // Connect controls after canvas is found
                    function connectControls() {
                        if (controlsConnected) {
                            return true; // Already connected
                        }
                        
                        // Buttons are in the same document as this script
                        const scaleUpBtn = document.getElementById('brochure-scale-up');
                        const scaleDownBtn = document.getElementById('brochure-scale-down');
                        const scaleInput = document.getElementById('brochure-scale-input');
                        const moveUpBtn = document.getElementById('brochure-move-up');
                        const moveDownBtn = document.getElementById('brochure-move-down');
                        const moveLeftBtn = document.getElementById('brochure-move-left');
                        const moveRightBtn = document.getElementById('brochure-move-right');
                        const moveXInput = document.getElementById('brochure-move-x-input');
                        const moveYInput = document.getElementById('brochure-move-y-input');
                        const rotateInput = document.getElementById('brochure-rotate-input');
                        
                        console.log('Looking for buttons:', {
                            scaleUp: !!scaleUpBtn,
                            scaleDown: !!scaleDownBtn,
                            moveUp: !!moveUpBtn,
                            moveDown: !!moveDownBtn,
                            moveLeft: !!moveLeftBtn,
                            moveRight: !!moveRightBtn,
                            rotate: !!rotateInput,
                            iframe: !!brochureIframe,
                            canvas: !!brochureCanvas
                        });
                        
                        if (!brochureIframe) {
                            console.log('Cannot connect controls: iframe not ready');
                            return false;
                        }
                        
                        const iframeWindow = brochureIframe.contentWindow;
                        
                        // Scale input stores the increment/decrement value (not the current scale)
                        // Initialize scale display separately
                        if (!iframeWindow.scaleDisplay) {
                            iframeWindow.scaleDisplay = document.createElement('span');
                            iframeWindow.scaleDisplay.id = 'brochure-scale-display';
                            iframeWindow.scaleDisplay.style.cssText = 'margin-left: 5px; font-size: 12px; color: #666; min-width: 50px; display: inline-block;';
                            const currentScale = iframeWindow.currentScale || 1.0;
                            iframeWindow.scaleDisplay.textContent = 'Current: ' + currentScale.toFixed(2) + 'x';
                            if (scaleInput && scaleInput.parentNode) {
                                scaleInput.parentNode.appendChild(iframeWindow.scaleDisplay);
                            }
                        }
                        
                        // Scale input handler - stores the step value for +/- buttons
                        if (scaleInput) {
                            scaleInput.onchange = function() {
                                // Just validate the step value, don't apply transform
                                const stepValue = parseFloat(this.value) || 0.1;
                                if (stepValue < 0.01 || stepValue > 1.0) {
                                    this.value = 0.1;
                                }
                            };
                        }
                        
                        if (scaleUpBtn) {
                            scaleUpBtn.onclick = function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                try {
                                    const currentScale = iframeWindow.currentScale || 1.0;
                                    const stepValue = parseFloat(scaleInput?.value || 0.1) || 0.1;
                                    // Add step value to current scale
                                    const newScale = currentScale + stepValue;
                                    iframeWindow.currentScale = newScale;
                                    // Update display
                                    if (iframeWindow.scaleDisplay) {
                                        iframeWindow.scaleDisplay.textContent = 'Current: ' + newScale.toFixed(2) + 'x';
                                    }
                                    applyTransform();
                                } catch(err) {
                                    console.error('Error in scale up:', err);
                                }
                                return false;
                            };
                        }
                        
                        if (scaleDownBtn) {
                            scaleDownBtn.onclick = function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                try {
                                    const currentScale = iframeWindow.currentScale || 1.0;
                                    const stepValue = parseFloat(scaleInput?.value || 0.1) || 0.1;
                                    // Subtract step value from current scale
                                    const newScale = Math.max(0.1, currentScale - stepValue);
                                    iframeWindow.currentScale = newScale;
                                    // Update display
                                    if (iframeWindow.scaleDisplay) {
                                        iframeWindow.scaleDisplay.textContent = 'Current: ' + newScale.toFixed(2) + 'x';
                                    }
                                    applyTransform();
                                } catch(err) {
                                    console.error('Error in scale down:', err);
                                }
                                return false;
                            };
                        }
                        
                        // Move input handlers - stores the step value for arrow buttons
                        // Initialize move display separately
                        if (!iframeWindow.moveDisplay) {
                            iframeWindow.moveDisplay = document.createElement('div');
                            iframeWindow.moveDisplay.id = 'brochure-move-display';
                            iframeWindow.moveDisplay.style.cssText = 'margin-left: 5px; font-size: 11px; color: #666;';
                            const currentX = iframeWindow.currentOffsetX || 0;
                            const currentY = iframeWindow.currentOffsetY || 0;
                            iframeWindow.moveDisplay.innerHTML = 'Current: X=' + currentX + ', Y=' + currentY;
                            if (moveYInput && moveYInput.parentNode) {
                                moveYInput.parentNode.appendChild(iframeWindow.moveDisplay);
                            }
                        }
                        
                        if (moveXInput) {
                            moveXInput.onchange = function() {
                                // Just validate the step value, don't apply transform
                                const stepValue = parseFloat(this.value) || 10;
                                if (stepValue < 1) {
                                    this.value = 10;
                                }
                            };
                        }
                        
                        if (moveYInput) {
                            moveYInput.onchange = function() {
                                // Just validate the step value, don't apply transform
                                const stepValue = parseFloat(this.value) || 10;
                                if (stepValue < 1) {
                                    this.value = 10;
                                }
                            };
                        }
                        
                        if (moveUpBtn) {
                            moveUpBtn.onclick = function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                try {
                                    const stepValue = parseFloat(moveYInput?.value || 10) || 10;
                                    iframeWindow.currentOffsetY = (iframeWindow.currentOffsetY || 0) - stepValue;
                                    // Update display
                                    if (iframeWindow.moveDisplay) {
                                        iframeWindow.moveDisplay.innerHTML = 'Current: X=' + (iframeWindow.currentOffsetX || 0) + ', Y=' + (iframeWindow.currentOffsetY || 0);
                                    }
                                    applyTransform();
                                } catch(err) {
                                    console.error('Error in move up:', err);
                                }
                                return false;
                            };
                        }
                        
                        if (moveDownBtn) {
                            moveDownBtn.onclick = function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                try {
                                    const stepValue = parseFloat(moveYInput?.value || 10) || 10;
                                    iframeWindow.currentOffsetY = (iframeWindow.currentOffsetY || 0) + stepValue;
                                    // Update display
                                    if (iframeWindow.moveDisplay) {
                                        iframeWindow.moveDisplay.innerHTML = 'Current: X=' + (iframeWindow.currentOffsetX || 0) + ', Y=' + (iframeWindow.currentOffsetY || 0);
                                    }
                                    applyTransform();
                                } catch(err) {
                                    console.error('Error in move down:', err);
                                }
                                return false;
                            };
                        }
                        
                        if (moveLeftBtn) {
                            moveLeftBtn.onclick = function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                try {
                                    const stepValue = parseFloat(moveXInput?.value || 10) || 10;
                                    iframeWindow.currentOffsetX = (iframeWindow.currentOffsetX || 0) - stepValue;
                                    // Update display
                                    if (iframeWindow.moveDisplay) {
                                        iframeWindow.moveDisplay.innerHTML = 'Current: X=' + (iframeWindow.currentOffsetX || 0) + ', Y=' + (iframeWindow.currentOffsetY || 0);
                                    }
                                    applyTransform();
                                } catch(err) {
                                    console.error('Error in move left:', err);
                                }
                                return false;
                            };
                        }
                        
                        if (moveRightBtn) {
                            moveRightBtn.onclick = function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                try {
                                    const stepValue = parseFloat(moveXInput?.value || 10) || 10;
                                    iframeWindow.currentOffsetX = (iframeWindow.currentOffsetX || 0) + stepValue;
                                    // Update display
                                    if (iframeWindow.moveDisplay) {
                                        iframeWindow.moveDisplay.innerHTML = 'Current: X=' + (iframeWindow.currentOffsetX || 0) + ', Y=' + (iframeWindow.currentOffsetY || 0);
                                    }
                                    applyTransform();
                                } catch(err) {
                                    console.error('Error in move right:', err);
                                }
                                return false;
                            };
                        }
                        
                        if (rotateInput) {
                            rotateInput.onchange = function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                try {
                                    iframeWindow.currentRotation = parseFloat(this.value) || 0;
                                    applyTransform();
                                } catch(err) {
                                    console.error('Error in rotate:', err);
                                }
                            };
                        }
                        
                        controlsConnected = true;
                        console.log('Brochure controls connected successfully!');
                        return true;
                    }
                    
                    // Try to connect, retry if needed
                    let retries = 0;
                    const maxRetries = 50; // Increased retries
                    function tryConnect() {
                        if (retries >= maxRetries) {
                            console.log('Max retries reached for brochure canvas. Controls may not work.');
                            // Try to connect controls anyway in case canvas exists but wasn't detected
                            connectControls();
                            return;
                        }
                        retries++;
                        
                        if (connectToBrochureCanvas()) {
                            // Wait a bit more for everything to be ready
                            setTimeout(function() {
                                connectControls();
                            }, 200);
                        } else {
                            setTimeout(tryConnect, 300); // Reduced delay for faster retries
                        }
                    }
                    
                    // Start trying to connect after a short delay
                    setTimeout(tryConnect, 1500);
                    
                    // Also try when DOM is ready
                    if (document.readyState === 'loading') {
                        document.addEventListener('DOMContentLoaded', function() {
                            setTimeout(tryConnect, 500);
                        });
                    }
                })();
            </script>
            """, height=200)
        elif not background_image_url:
            st.error("❌ Background image not found. Please upload an image in Step 1 first.")
        elif not plots_payload:
            st.warning("⚠️ No plots with valid coordinates found.")
        
        # Navigation buttons for Step 5
        col_btn1, col_btn2, col_btn3 = st.columns([8, 1, 1])
        with col_btn1:
            st.write("")  # Spacer for alignment
        with col_btn2:
            if st.button("Prev", type="primary", key="prev_step5"):
                st.session_state.current_step = 4  # Go back to Step 4
                st.rerun()
        with col_btn3:
            if st.button("Next", type="primary", key="next_step5"):
                st.session_state.current_step = 6  # Go to Step 6 (Configure Map Settings)
                st.rerun()
    else:
        st.info("Please complete previous steps first.")

# STEP 6: Configure Map Settings (moved from Step 5)
elif current_step == 6:
    st.header("Configure Map Settings")
    
    if st.session_state.plots:
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.px_to_ft = st.number_input("Feet per Pixel", 
                                                         value=st.session_state.px_to_ft, 
                                                         min_value=0.01,
                                                         max_value=10.0, 
                                                         step=0.05, 
                                                         format="%.3f",
                                                         key="config_px_to_ft")
        
        with col2:
            plot_ids = [p['plot_id'] for p in st.session_state.plots]
            ref_plot_id = st.selectbox("Reference Plot", plot_ids, key="config_ref_plot")
            ref_corner = st.selectbox("Corner", ["A", "B", "C", "D"], key="config_ref_corner")
            
            col2a, col2b = st.columns(2)
            with col2a:
                ref_lat = st.number_input("Latitude", value=13.0003, format="%.6f", key="config_lat")
            with col2b:
                ref_lon = st.number_input("Longitude", value=77.0001, format="%.6f", key="config_lon")
        
        if st.button("Generate Map", type="primary", key="generate_map_config"):
            with st.spinner("Calculating..."):
                st.session_state.geo_plots = calculate_geocoordinates(
                    st.session_state.plots, ref_plot_id, ref_corner,
                    ref_lat, ref_lon, st.session_state.px_to_ft
                )
                if st.session_state.geo_plots:
                    st.success(f"Generated map for {len(st.session_state.geo_plots)} plots!")
        
        # Navigation buttons for Step 6
        if st.session_state.geo_plots:
            col_btn1, col_btn2, col_btn3 = st.columns([8, 1, 1])
            with col_btn2:
                if st.button("Prev", type="primary", key="prev_step6"):
                    st.session_state.current_step = 5  # Go back to Step 5
                    st.rerun()
            with col_btn3:
                if st.button("Next", type="primary", key="next_step6"):
                    st.session_state.current_step = 7  # Go to Step 7 (Update Lat and Long)
                    st.rerun()
    else:
        st.info("Please complete previous steps first.")

# STEP 4: Preview Polygons (shows regenerated image based on edits)
elif current_step == 4:


    if st.session_state.plots:
        st.markdown(
            '<div style="margin: 10px 0 12px 0; font-weight:600; color:#1f2937; font-size: 14px; visibility: visible !important; display: block !important;">'
            'This preview shows the updated image based on any edits made in previous steps.'
            '</div>',
            unsafe_allow_html=True
        )
        
        # Button to manually regenerate image from current plot coordinates
        if st.button("Regenerate Image", type="primary", use_container_width=False):
            st.rerun()
        
        st.markdown('<div style="margin-top: 5px;"></div>', unsafe_allow_html=True)
        
        # Always regenerate the image to ensure it's up to date with current plot data
        if st.session_state.detection_image is not None:
            # Get the original image (before detection) if available, otherwise use detection_image
            # For now, we'll regenerate from the current plots
            original_img = st.session_state.detection_image.copy()
            
            # Get dimensions from the original image
            if len(original_img.shape) == 3:
                orig_height, orig_width = original_img.shape[:2]
            else:
                orig_height, orig_width = original_img.shape
            
            # CRITICAL: Calculate bounding box of all plots to ensure all are visible
            # This includes plots created below or outside the original image bounds
            min_x, min_y, max_x, max_y = None, None, None, None
            
            # First pass: calculate bounding box from all valid plots
            # CRITICAL: Use 'points' array if available (supports more than 4 points), 
            # otherwise fall back to corners (for backward compatibility)
            valid_plots = []
            for plot in st.session_state.plots:
                # Get points array if available, otherwise convert corners to points
                points = plot.get('points', [])
                if not points:
                    # Fallback: Convert corners to points for backward compatibility
                    corners = plot.get('corners', {})
                    if not corners:
                        continue
                    # Convert corners to points array
                    corner_labels = ['A', 'B', 'C', 'D']
                    points = []
                    for label in corner_labels:
                        if label in corners:
                            points.append({'x': corners[label].get('x', 0), 'y': corners[label].get('y', 0)})
                
                if len(points) < 3:
                    continue
                
                # Validate coordinates are numbers
                try:
                    # Get all point coordinates
                    x_coords = [int(p['x']) for p in points]
                    y_coords = [int(p['y']) for p in points]
                    
                    # Update bounding box
                    plot_min_x, plot_max_x = min(x_coords), max(x_coords)
                    plot_min_y, plot_max_y = min(y_coords), max(y_coords)
                    
                    if min_x is None:
                        min_x, min_y = plot_min_x, plot_min_y
                        max_x, max_y = plot_max_x, plot_max_y
                    else:
                        min_x = min(min_x, plot_min_x)
                        min_y = min(min_y, plot_min_y)
                        max_x = max(max_x, plot_max_x)
                        max_y = max(max_y, plot_max_y)
                    
                    valid_plots.append(plot)
                except (KeyError, ValueError, TypeError):
                    # Skip plots with invalid coordinates
                    continue
            
            # Calculate canvas size with padding to fit all plots
            padding = 50  # Padding around all plots
            if min_x is not None and max_x is not None:
                # Calculate the bounding box dimensions
                plot_width = max_x - min_x
                plot_height = max_y - min_y
                
                # Calculate required canvas size: bounding box + padding on all sides
                # If min_x/min_y are negative, plots extend left/up, so we need extra space
                required_width = plot_width + 2 * padding
                required_height = plot_height + 2 * padding
                
                # Calculate offset to shift all plots so the leftmost/topmost point
                # (after accounting for negative coordinates) appears at position (padding, padding)
                # If min_x is negative (e.g., -100), we shift right by padding - min_x (e.g., 50 - (-100) = 150)
                # If min_x is positive (e.g., 100), we shift left by min_x - padding (e.g., 100 - 50 = 50)
                # But we always want plots to start at padding, so we use: padding - min_x
                final_offset_x = padding - min_x  # Shift so leftmost plot is at x=padding
                final_offset_y = padding - min_y  # Shift so topmost plot is at y=padding
                
                # Calculate canvas dimensions to fit all shifted plots
                # After shifting, the rightmost plot will be at: max_x + final_offset_x
                # The bottommost plot will be at: max_y + final_offset_y
                canvas_width = int(max_x + final_offset_x + padding)
                canvas_height = int(max_y + final_offset_y + padding)
                
                # Ensure canvas is at least as large as original image
                canvas_width = max(canvas_width, orig_width)
                canvas_height = max(canvas_height, orig_height)
            else:
                # No valid plots, use original dimensions
                canvas_width = orig_width
                canvas_height = orig_height
                final_offset_x = 0
                final_offset_y = 0
            
            # Create white background with calculated dimensions
            display_img = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
            
            # Draw original image if it fits within canvas (optional - can be commented out)
            # If original image should be shown, uncomment this:
            # if offset_x >= 0 and offset_y >= 0:
            #     display_img[offset_y:offset_y+orig_height, offset_x:offset_x+orig_width] = original_img
            
            # Redraw all plots with current coordinates and updated plot numbers
            # CRITICAL: Use 'points' array if available (supports more than 4 points)
            for plot in valid_plots:
                # Get points array if available, otherwise convert corners to points
                points = plot.get('points', [])
                if not points:
                    # Fallback: Convert corners to points for backward compatibility
                    corners = plot.get('corners', {})
                    if not corners:
                        continue
                    corner_labels = ['A', 'B', 'C', 'D']
                    points = []
                    for label in corner_labels:
                        if label in corners:
                            points.append({'x': corners[label].get('x', 0), 'y': corners[label].get('y', 0)})
                
                if len(points) < 3:
                    continue
                
                # Adjust coordinates by offset to fit in canvas
                try:
                    # Create points array with all points (not just 4)
                    pts = np.array([
                        [int(p['x']) + final_offset_x, int(p['y']) + final_offset_y] 
                        for p in points
                    ], np.int32)
                except (KeyError, ValueError, TypeError):
                    continue
                
                # Red lines for plot boundaries (BGR format: red = (0, 0, 255))
                cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
                
                # Draw red dots at each point (all points, not just 4)
                for point in points:
                    cv2.circle(display_img, 
                              (int(point['x']) + final_offset_x, int(point['y']) + final_offset_y), 
                              4, (0, 0, 255), -1)
                
                # Draw plot number in black - use current plot number from session state
                plot_number = plot.get('plot_number')
                if plot_number is not None:
                    # Calculate center from all points
                    cx = sum([int(p['x']) + final_offset_x for p in points]) // len(points)
                    cy = sum([int(p['y']) + final_offset_y for p in points]) // len(points)
                    cv2.putText(display_img, str(plot_number),
                               (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Update session state with regenerated image
            st.session_state.detection_image = display_img
            st.session_state.detected_overlay_url = ndarray_to_data_url(display_img)
            st.session_state.brochure_overlay_url = st.session_state.detected_overlay_url
            
            # Display the regenerated image - centered and sized for clear visibility
            st.markdown("""
                <style>
                    div[data-testid="stImage"] img {
                        max-height: 400px !important;
                        object-fit: contain !important;
                    }
                </style>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns([0.3, 3.4, 0.3])
            with col2:
                st.image(display_img, channels="BGR",
                        caption=f"{len(st.session_state.plots)} plots with updated coordinates (red lines and dots)",
                        use_container_width=True)
        else:
            st.warning("⚠️ No detection image available. Please go back to Step 1 and detect plots.")
        
        # Navigation buttons for Step 4
        col_btn1, col_btn2, col_btn3 = st.columns([8, 1, 1])
        with col_btn2:
            if st.button("Prev", type="primary", key="prev_step4"):
                st.session_state.current_step = 3  # Go back to Step 3
                st.rerun()
        with col_btn3:
            if st.button("Next", type="primary", key="next_step4"):
                st.session_state.current_step = 5  # Go to Step 5 (Preview in Brochure)
                st.rerun()
    else:
        st.info("Please detect plots first.")

# STEP 7: Update Lat and Long (update geo coordinates)
elif current_step == 7:
    st.header("Update Lat and Long")
   
    if st.session_state.plots:
        if not st.session_state.geo_plots:
            st.warning("⚠️ Please configure map settings in Step 6 (Configure Map Settings) first.")
        else:
            # Create table matching wireframe: First row shows Coordinates, second row shows Lat Long
            sorted_geo_plots = sorted(st.session_state.geo_plots, 
                                     key=lambda p: p['plot_number'] if p['plot_number'] is not None else 9999)
            sorted_plots = sorted(st.session_state.plots, 
                                 key=lambda p: p['plot_number'] if p['plot_number'] is not None else 9999)
            
            # CRITICAL: Create a mapping of plot_number to pixel coordinates
            # Use 'points' array if available (supports more than 4 points), otherwise use corners
            plot_coords_map = {}
            for p in sorted_plots:
                plot_num = p.get('plot_number')
                # Get points array if available, otherwise convert corners to points
                points = p.get('points', [])
                if not points:
                    corners = p.get('corners', {})
                    points = []
                    for label in ['A', 'B', 'C', 'D']:
                        if label in corners:
                            points.append({'x': corners[label].get('x', 0), 'y': corners[label].get('y', 0)})
                plot_coords_map[plot_num] = points
            
            # Find maximum number of points across all plots to determine table columns
            max_points = 0
            for plot_num, points in plot_coords_map.items():
                max_points = max(max_points, len(points))
            # Also check geo_plots for lat/lon points
            for plot in sorted_geo_plots:
                corners = plot.get('corners', {})
                max_points = max(max_points, len(corners))
            
            # Generate column labels (A, B, C, D, E, F, ...)
            corner_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
            point_labels = corner_labels[:max_points] if max_points <= len(corner_labels) else corner_labels + [chr(ord('Q') + i) for i in range(max_points - len(corner_labels))]
            
            table_data = []
            for idx, plot in enumerate(sorted_geo_plots):
                plot_num = plot['plot_number']
                plot_id = f"P-{plot_num:02d}" if plot_num else "Unknown"
                pixel_points = plot_coords_map.get(plot_num, [])
                geo_corners = plot.get('corners', {})
                
                # First row: Coordinates (X,Y) - build dynamically for all points
                coord_row = {
                    "Sl. No": idx + 1,
                    "Plot No.": plot_id,
                    "Value": "Coordinates",
                }
                for label in point_labels:
                    label_idx = ord(label) - ord('A')
                    if label_idx < len(pixel_points):
                        coord_row[f"Side {label} (X,Y)"] = f"({pixel_points[label_idx].get('x', 0)}, {pixel_points[label_idx].get('y', 0)})"
                    else:
                        coord_row[f"Side {label} (X,Y)"] = "-"
                table_data.append(coord_row)
                
                # Second row: Lat Long - build dynamically for all points
                latlon_row = {
                    "Sl. No": "",
                    "Plot No.": plot_id,
                    "Value": "Lat Long",
                }
                for label in point_labels:
                    if label in geo_corners:
                        lat = geo_corners[label].get('lat', 0)
                        lon = geo_corners[label].get('lon', 0)
                        latlon_row[f"Side {label} (X,Y)"] = f"({lat:.6f}, {lon:.6f})"
                    else:
                        latlon_row[f"Side {label} (X,Y)"] = "-"
                table_data.append(latlon_row)
            
            # Create editable DataFrame with separate columns for lat/lon
            # CRITICAL: Dynamically build columns for all points (A, B, C, D, E, F, ...)
            editable_data = []
            for idx, plot in enumerate(sorted_geo_plots):
                plot_num = plot['plot_number']
                plot_id = f"P-{plot_num:02d}" if plot_num else "Unknown"
                pixel_points = plot_coords_map.get(plot_num, [])
                geo_corners = plot.get('corners', {})
                
                row = {
                    "Sl. No": idx + 1,
                    "Plot No.": plot_id,
                }
                
                # Add columns for all points dynamically
                for label in point_labels:
                    label_idx = ord(label) - ord('A')
                    # Pixel coordinates
                    if label_idx < len(pixel_points):
                        row[f"{label}_x"] = pixel_points[label_idx].get('x', 0)
                        row[f"{label}_y"] = pixel_points[label_idx].get('y', 0)
                    else:
                        row[f"{label}_x"] = 0
                        row[f"{label}_y"] = 0
                    
                    # Lat/Lon coordinates
                    if label in geo_corners:
                        row[f"{label}_lat"] = geo_corners[label].get('lat', 0.0)
                        row[f"{label}_lon"] = geo_corners[label].get('lon', 0.0)
                    else:
                        row[f"{label}_lat"] = 0.0
                        row[f"{label}_lon"] = 0.0
                
                editable_data.append(row)
            
            editable_df = pd.DataFrame(editable_data)
            
            # Initialize grid refresh counter
            if 'latlon_grid_refresh' not in st.session_state:
                st.session_state.latlon_grid_refresh = 0
            
            # Store original geo_plots data for change detection
            if 'original_geo_plots_for_edit' not in st.session_state or st.session_state.get('geo_plots_snapshot_saved', False) == False:
                import copy
                st.session_state.original_geo_plots_for_edit = copy.deepcopy(st.session_state.geo_plots)
                st.session_state.geo_plots_snapshot_saved = True
            
            # Build column config dynamically for all points
            column_config = {
                "Sl. No": st.column_config.NumberColumn("Sl. No", disabled=True),
                "Plot No.": st.column_config.TextColumn("Plot No.", disabled=True),
            }
            
            # Add columns for all points
            for label in point_labels:
                column_config[f"{label}_x"] = st.column_config.NumberColumn(f"{label} → x", min_value=0, step=1, help=f"Point {label}, x coordinate")
                column_config[f"{label}_y"] = st.column_config.NumberColumn(f"{label} → y", min_value=0, step=1, help=f"Point {label}, y coordinate")
                column_config[f"{label}_lat"] = st.column_config.NumberColumn(f"{label} → lat", min_value=-90.0, max_value=90.0, step=0.000001, format="%.6f", help=f"Point {label}, latitude")
                column_config[f"{label}_lon"] = st.column_config.NumberColumn(f"{label} → lon", min_value=-180.0, max_value=180.0, step=0.000001, format="%.6f", help=f"Point {label}, longitude")
            
            # Display editable table
            edited_df = st.data_editor(
                editable_df,
                hide_index=True,
                key=f"latlon_editor_{st.session_state.latlon_grid_refresh}",
                column_config=column_config,
                use_container_width=True,
                num_rows="fixed"
            )
            
            # Apply changes button
            if st.button("Save Changes", type="primary", use_container_width=False):
                import copy
                
                changes_made = False
                updated_count = 0
                
                # Get original geo_plots data for comparison
                original_geo_plots = st.session_state.get('original_geo_plots_for_edit', [])
                
                # Step 1: Detect which plot(s) had lat/lon changes and calculate offsets
                changed_plots = []
                offsets = {}  # Store offsets per plot and corner
                
                for idx, row in edited_df.iterrows():
                    plot_num = sorted_geo_plots[idx]['plot_number']
                    
                    # Find original plot
                    original_geo_plot = next((gp for gp in original_geo_plots if gp['plot_number'] == plot_num), None)
                    
                    if original_geo_plot:
                        plot_changed = False
                        plot_offsets = {}
                        
                        # CRITICAL: Check all points dynamically (A, B, C, D, E, F, ...)
                        for corner in point_labels:
                            # Check if this corner exists in the original geo_plot
                            if corner not in original_geo_plot.get('corners', {}):
                                continue
                            
                            try:
                                new_lat = float(row[f'{corner}_lat'])
                                new_lon = float(row[f'{corner}_lon'])
                                
                                orig_lat = original_geo_plot['corners'][corner]['lat']
                                orig_lon = original_geo_plot['corners'][corner]['lon']
                                
                                # Calculate offset
                                delta_lat = new_lat - orig_lat
                                delta_lon = new_lon - orig_lon
                                
                                # Check if this corner changed significantly
                                if abs(delta_lat) > 0.0000001 or abs(delta_lon) > 0.0000001:
                                    plot_changed = True
                                    plot_offsets[corner] = {
                                        'delta_lat': delta_lat,
                                        'delta_lon': delta_lon
                                    }
                            except (KeyError, ValueError, TypeError):
                                # Skip if this corner doesn't exist in the row
                                continue
                        
                        if plot_changed:
                            changed_plots.append({
                                'plot_num': plot_num,
                                'idx': idx,
                                'offsets': plot_offsets
                            })
                
                # Step 2: If any plot changed, apply the same offset to ALL plots
                if changed_plots:
                    with st.spinner("🔄 Applying changes to all plots proportionally..."):
                        # Use the first changed plot as reference for the transformation
                        reference_change = changed_plots[0]
                        ref_offsets = reference_change['offsets']
                        
                        # Calculate average offset across all changed corners
                        total_delta_lat = 0
                        total_delta_lon = 0
                        corner_count = 0
                        
                        for corner, offset_data in ref_offsets.items():
                            total_delta_lat += offset_data['delta_lat']
                            total_delta_lon += offset_data['delta_lon']
                            corner_count += 1
                        
                        if corner_count > 0:
                            avg_delta_lat = total_delta_lat / corner_count
                            avg_delta_lon = total_delta_lon / corner_count
                            
                            st.info(f"📍 Reference change detected: Δlat={avg_delta_lat:.8f}, Δlon={avg_delta_lon:.8f}")
                            st.info(f"🔄 Applying this offset to all {len(st.session_state.geo_plots)} plots...")
                            
                            # Apply offset to ALL plots (including the one that was changed)
                            for geo_plot in st.session_state.geo_plots:
                                plot_num = geo_plot['plot_number']
                                
                                # Find the corresponding row in edited_df
                                plot_idx = next((i for i, gp in enumerate(sorted_geo_plots) if gp['plot_number'] == plot_num), None)
                                
                                if plot_idx is not None:
                                    # Get the edited values for this plot
                                    edited_row = edited_df.iloc[plot_idx]
                                    
                                    # CRITICAL: Apply offset to all points dynamically (A, B, C, D, E, F, ...)
                                    for corner in point_labels:
                                        # Check if this corner exists in the geo_plot
                                        if corner not in geo_plot.get('corners', {}):
                                            continue
                                        
                                        try:
                                            # Get the edited value (which may already include changes)
                                            edited_lat = float(edited_row[f'{corner}_lat'])
                                            edited_lon = float(edited_row[f'{corner}_lon'])
                                            
                                            # Get original value
                                            original_plot = next((gp for gp in original_geo_plots if gp['plot_number'] == plot_num), None)
                                            
                                            if original_plot and corner in original_plot.get('corners', {}):
                                                orig_lat = original_plot['corners'][corner]['lat']
                                                orig_lon = original_plot['corners'][corner]['lon']
                                                
                                                # Apply the reference offset to the original values
                                                new_lat = orig_lat + avg_delta_lat
                                                new_lon = orig_lon + avg_delta_lon
                                                
                                                # Update geo_plot
                                                geo_plot['corners'][corner]['lat'] = new_lat
                                                geo_plot['corners'][corner]['lon'] = new_lon
                                                
                                                # Also update pixel coordinates if reference settings are available
                                                has_ref_settings = ('px_to_ft' in st.session_state and 
                                                                  'ref_plot_id' in st.session_state and
                                                                  st.session_state.ref_plot_id)
                                                
                                                if has_ref_settings:
                                                    ref_plot = next((p for p in st.session_state.plots if p.get('plot_id') == st.session_state.ref_plot_id), None)
                                                    ref_geo_plot = next((gp for gp in st.session_state.geo_plots if gp.get('plot_id') == st.session_state.ref_plot_id), None)
                                                    
                                                    if ref_plot and ref_geo_plot:
                                                        ref_corner = st.session_state.get('ref_corner', 'A')
                                                        ref_pixel_corner = ref_plot['corners'].get(ref_corner, {})
                                                        ref_x = ref_pixel_corner.get('x')
                                                        ref_y = ref_pixel_corner.get('y')
                                                        ref_geo_corner = ref_geo_plot['corners'].get(ref_corner, {})
                                                        ref_lat = ref_geo_corner.get('lat')
                                                        ref_lon = ref_geo_corner.get('lon')
                                                        
                                                        if all(v is not None for v in [ref_lat, ref_lon, ref_x, ref_y]):
                                                            px_to_ft = st.session_state.px_to_ft
                                                            
                                                            # Find pixel plot and update pixel coordinates
                                                            pixel_plot = next((p for p in st.session_state.plots if p.get('plot_number') == plot_num), None)
                                                            if pixel_plot:
                                                                # CRITICAL: Update points array if available, otherwise update corners
                                                                calculated_x, calculated_y = recalculate_pixel_from_coordinates(
                                                                    ref_lat, ref_lon, ref_x, ref_y, new_lat, new_lon, px_to_ft
                                                                )
                                                                
                                                                # Update in corners (for backward compatibility)
                                                                if 'corners' in pixel_plot and corner in pixel_plot['corners']:
                                                                    pixel_plot['corners'][corner]['x'] = calculated_x
                                                                    pixel_plot['corners'][corner]['y'] = calculated_y
                                                                
                                                                # Also update in points array if it exists
                                                                if 'points' in pixel_plot:
                                                                    corner_idx = ord(corner) - ord('A')
                                                                    if corner_idx < len(pixel_plot['points']):
                                                                        pixel_plot['points'][corner_idx]['x'] = calculated_x
                                                                        pixel_plot['points'][corner_idx]['y'] = calculated_y
                                        except (KeyError, ValueError, TypeError):
                                            # Skip if this corner doesn't exist
                                            continue
                            
                            changes_made = True
                            updated_count = len(st.session_state.geo_plots)
                
                if changes_made:
                    # Clear the snapshot so next time we start fresh
                    st.session_state.geo_plots_snapshot_saved = False
                    
                    # Increment refresh counter to force grid refresh
                    st.session_state.latlon_grid_refresh += 1
                    
                    # Force update session state to ensure changes are saved
                    st.session_state.geo_plots = st.session_state.geo_plots.copy()
                    st.session_state.plots = st.session_state.plots.copy()
                    
                    st.success(f"✅ Coordinates updated successfully! All {updated_count} plots have been adjusted proportionally. Grid and map will refresh.")
                    st.rerun()
                else:
                    st.info("ℹ️ No changes detected. Values are already up to date.")
        
        # Navigation buttons for Step 7
        if st.session_state.geo_plots:
            col_btn1, col_btn2, col_btn3 = st.columns([8, 1, 1])
            with col_btn2:
                if st.button("Prev", type="primary", key="prev_step7"):
                    st.session_state.current_step = 6  # Go back to Step 6
                    st.rerun()
            with col_btn3:
                if st.button("Next", type="primary", key="next_step7"):
                    st.session_state.current_step = 8  # Go to Step 8 (Preview in Google Map)
                    st.rerun()
    else:
        st.info("Please complete previous steps first.")

# STEP 8: Preview in Google Map (shows map view)
elif current_step == 8:
    
    
    if not st.session_state.plots:
        st.info("⚠️ Please complete the configuration steps in the 'Detection & Configuration' tab and generate the map first.")
    else:
        # Initialize session state for showing config panel
        if 'show_config_map_step8' not in st.session_state:
            st.session_state.show_config_map_step8 = False
        
        # Show the configure map settings panel if button was clicked
        if st.session_state.get('show_config_map_step8', False):
            with st.expander("🗺️ Configure Map Settings", expanded=True):
                st.write("**Adjust map settings below.** Click 'Generate Map' to apply changes and regenerate the map.")
                
                # Initialize default values if not set
                if 'px_to_ft' not in st.session_state:
                    st.session_state.px_to_ft = 0.5
                if 'ref_plot_id' not in st.session_state:
                    if st.session_state.plots:
                        st.session_state.ref_plot_id = st.session_state.plots[0]['plot_id']
                if 'ref_corner' not in st.session_state:
                    st.session_state.ref_corner = 'A'
                
                col1, col2 = st.columns(2)
                
                with col1:
                    px_to_ft = st.number_input("Feet per Pixel", 
                                             value=st.session_state.px_to_ft, 
                                             min_value=0.01,
                                             max_value=10.0, 
                                             step=0.05, 
                                             format="%.3f",
                                             key="config_px_to_ft_step8")
                    st.session_state.px_to_ft = px_to_ft
                
                with col2:
                    plot_ids = [p['plot_id'] for p in st.session_state.plots]
                    if st.session_state.ref_plot_id not in plot_ids:
                        st.session_state.ref_plot_id = plot_ids[0] if plot_ids else None
                    
                    ref_plot_id = st.selectbox("Reference Plot", plot_ids, 
                                             index=plot_ids.index(st.session_state.ref_plot_id) if st.session_state.ref_plot_id in plot_ids else 0,
                                             key="config_ref_plot_step8")
                    st.session_state.ref_plot_id = ref_plot_id
                    
                    ref_corner = st.selectbox("Corner", ["A", "B", "C", "D"], 
                                             index=["A", "B", "C", "D"].index(st.session_state.ref_corner) if st.session_state.ref_corner in ["A", "B", "C", "D"] else 0,
                                             key="config_ref_corner_step8")
                    st.session_state.ref_corner = ref_corner
                    
                    col2a, col2b = st.columns(2)
                    with col2a:
                        ref_lat = st.number_input("Latitude", value=13.0003, format="%.6f", key="config_lat_step8")
                    with col2b:
                        ref_lon = st.number_input("Longitude", value=77.0001, format="%.6f", key="config_lon_step8")
                
                col_spacer, col_close, col_apply = st.columns([4, 1, 1])
                with col_apply:
                    if st.button("🗺️ Generate Map", type="primary", use_container_width=True, key="generate_map_step8"):
                        with st.spinner("Calculating coordinates and regenerating map..."):
                            st.session_state.geo_plots = calculate_geocoordinates(
                                st.session_state.plots, ref_plot_id, ref_corner,
                                ref_lat, ref_lon, px_to_ft
                            )
                            if st.session_state.geo_plots:
                                st.success(f"✅ Generated map for {len(st.session_state.geo_plots)} plots! Map will update below.")
                                st.session_state.show_config_map_step8 = False
                                st.rerun()
                            else:
                                st.error("❌ Failed to generate map. Please check your settings.")
                
                with col_close:
                    if st.button("❌ Close", use_container_width=True, key="close_config_step8"):
                        st.session_state.show_config_map_step8 = False
                        st.rerun()
                
                st.divider()
        
        # Handle coordinate updates from JavaScript
        query_params = st.query_params
        if 'plot_move_update' in query_params:
            try:
                update_data_str = query_params['plot_move_update']
                update_data = json.loads(update_data_str)
                
                if update_data.get('type') == 'plot_coordinates_update' and 'updated_plots' in update_data:
                    updated_plots = update_data['updated_plots']
                    
                    print(f"🔄 Processing plot move update: {len(updated_plots)} plots")
                    
                    # Update session state geo_plots with new coordinates
                    updated_count = 0
                    for updated_plot in updated_plots:
                        plot_id = updated_plot.get('plot_id')
                        if plot_id:
                            # Find matching plot in session state
                            for geo_plot in st.session_state.geo_plots:
                                if geo_plot.get('plot_id') == plot_id:
                                    # CRITICAL: Update ALL corners (A, B, C, D, E, F, ...) not just first 4
                                    updated_corners = updated_plot.get('corners', {})
                                    for corner_label, corner_data in updated_corners.items():
                                        if corner_label in geo_plot.get('corners', {}):
                                            old_lat = geo_plot['corners'][corner_label].get('lat')
                                            old_lon = geo_plot['corners'][corner_label].get('lon')
                                            new_lat = corner_data.get('lat')
                                            new_lon = corner_data.get('lon')
                                            
                                            geo_plot['corners'][corner_label]['lat'] = new_lat
                                            geo_plot['corners'][corner_label]['lon'] = new_lon
                                            
                                            if abs(old_lat - new_lat) > 0.000001 or abs(old_lon - new_lon) > 0.000001:
                                                updated_count += 1
                                    break
                    
                    print(f"✅ Updated {updated_count} coordinate pairs")
                    
                    # Update pixel coordinates based on the offset
                    offset_lat = update_data.get('offset', {}).get('lat', 0)
                    offset_lng = update_data.get('offset', {}).get('lng', 0)
                    
                    # Recalculate pixel coordinates for all plots
                    if st.session_state.plots and st.session_state.geo_plots and 'px_to_ft' in st.session_state:
                        # Find a reference plot that exists in both plots and geo_plots
                        ref_plot = None
                        ref_geo_plot = None
                        ref_plot_id = None
                        
                        # Try to find first plot that exists in both
                        for plot in st.session_state.plots:
                            plot_id = plot.get('plot_id')
                            geo_plot = next((gp for gp in st.session_state.geo_plots if gp.get('plot_id') == plot_id), None)
                            if geo_plot:
                                ref_plot = plot
                                ref_geo_plot = geo_plot
                                ref_plot_id = plot_id
                                break
                        
                        if ref_plot and ref_geo_plot:
                            # Get reference corner (use corner A of reference plot) - use UPDATED coordinates
                            ref_geo_corner = ref_geo_plot.get('corners', {}).get('A', {})
                            ref_lat = ref_geo_corner.get('lat', 0)
                            ref_lon = ref_geo_corner.get('lon', 0)
                            
                            # Get original pixel coordinates of reference corner (before move)
                            ref_pixel_corner = ref_plot.get('corners', {}).get('A', {})
                            ref_x = ref_pixel_corner.get('x', 0)
                            ref_y = ref_pixel_corner.get('y', 0)
                            
                            # Apply offset to pixel coordinates as well (simple approach)
                            # Calculate pixel offset based on geo offset
                            # Approximate: 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(lat)
                            # Convert to pixels using px_to_ft
                            ft_per_deg_lat = 111000 * 3.28084  # feet per degree latitude
                            ft_per_deg_lon = 111000 * 3.28084 * math.cos(math.radians(ref_lat))
                            
                            px_per_deg_lat = ft_per_deg_lat / st.session_state.px_to_ft
                            px_per_deg_lon = ft_per_deg_lon / st.session_state.px_to_ft
                            
                            pixel_offset_x = offset_lng * px_per_deg_lon
                            pixel_offset_y = -offset_lat * px_per_deg_lat  # Negative because Y increases downward
                            
                            # Apply pixel offset to all plots
                            for plot in st.session_state.plots:
                                for corner in ['A', 'B', 'C', 'D']:
                                    if corner in plot.get('corners', {}):
                                        plot['corners'][corner]['x'] = int(plot['corners'][corner].get('x', 0) + pixel_offset_x)
                                        plot['corners'][corner]['y'] = int(plot['corners'][corner].get('y', 0) + pixel_offset_y)
                    
                    # Regenerate detection image with updated coordinates
                    if st.session_state.detection_image is not None and st.session_state.plots:
                        original_img = st.session_state.detection_image.copy()
                        if len(original_img.shape) == 3:
                            height, width = original_img.shape[:2]
                            display_img = np.ones((height, width, 3), dtype=np.uint8) * 255
                        else:
                            height, width = original_img.shape
                            display_img = np.ones((height, width, 3), dtype=np.uint8) * 255
                        
                        # Redraw all plots with updated pixel coordinates
                        for plot in st.session_state.plots:
                            corners = plot.get('corners', {})
                            if not corners:
                                continue
                            pts = np.array([
                                [corners['A']['x'], corners['A']['y']],
                                [corners['B']['x'], corners['B']['y']],
                                [corners['C']['x'], corners['C']['y']],
                                [corners['D']['x'], corners['D']['y']]
                            ], np.int32)
                            cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
                            cv2.circle(display_img, (corners['A']['x'], corners['A']['y']), 4, (0, 0, 255), -1)
                            cv2.circle(display_img, (corners['B']['x'], corners['B']['y']), 4, (0, 0, 255), -1)
                            cv2.circle(display_img, (corners['C']['x'], corners['C']['y']), 4, (0, 0, 255), -1)
                            cv2.circle(display_img, (corners['D']['x'], corners['D']['y']), 4, (0, 0, 255), -1)
                            
                            plot_number = plot.get('plot_number')
                            if plot_number is not None:
                                cx = sum([corners[c]['x'] for c in corners]) // 4
                                cy = sum([corners[c]['y'] for c in corners]) // 4
                                cv2.putText(display_img, str(plot_number),
                                           (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        st.session_state.detection_image = display_img
                        st.session_state.detected_overlay_url = ndarray_to_data_url(display_img)
                        # Don't overwrite original_image_base64 - preserve for background toggle
                        if st.session_state.original_image_base64 is None:
                            st.session_state.original_image_base64 = ndarray_to_data_url(display_img)
                        st.session_state.brochure_overlay_url = st.session_state.detected_overlay_url
                    
                    # Remove query parameters to avoid reprocessing
                    st.query_params.pop('plot_move_update', None)
                    st.query_params.pop('_timestamp', None)
                    
                    target_pos = update_data.get('target_position', {})
                    st.success(f"✅ All plots moved successfully! Plot layout centroid is now at ({target_pos.get('lat', 0):.6f}, {target_pos.get('lng', 0):.6f})")
                    st.rerun()
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                st.error(f"Error processing coordinate update: {e}")
                # Remove invalid parameter
                st.query_params.pop('plot_move_update', None)
        
        # Map display section - only show if geo_plots exist
        if not st.session_state.geo_plots:
            st.info("Click '⚙️ Configure Map Settings' above to generate the map.")
        else:
            st.subheader("Map View")
            
            # Top row: total count, move toggle, configure button (dark blue)
            col_count, col_toggle, col_cfg = st.columns([1.2, 1.4, 1.0])
            with col_count:
                st.markdown(f"**Total geo_plots:** {len(st.session_state.geo_plots)}")
            with col_toggle:
                move_mode_enabled = st.toggle(
                    "Enable Move Mode",
                    value=st.session_state.get('move_plots_enabled', False),
                    key="toggle_move_mode",
                    help="Enable to move all plots by clicking on the map. When enabled, click anywhere on the map to move all plots to that position."
                )
                st.session_state.move_plots_enabled = move_mode_enabled
            with col_cfg:
                if st.button("Configure Map Settings", type="primary", use_container_width=True, key="btn_config_map_step8"):
                    st.session_state.show_config_map_step8 = not st.session_state.show_config_map_step8
                    st.rerun()
            
            # Collect all valid coordinates from ALL plots for proper map bounds
            # CRITICAL: Collect ALL corners (A, B, C, D, E, F, ...) not just first 4
            all_lats = []
            all_lons = []
            invalid_coords_count = 0
            
            for plot in st.session_state.geo_plots:
                corners = plot.get('corners', {})
                # Process all corners, not just A, B, C, D
                for corner_label, corner_data in corners.items():
                    lat = corner_data.get('lat')
                    lon = corner_data.get('lon')
                    
                    # Validate coordinates
                    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            all_lats.append(lat)
                            all_lons.append(lon)
                        else:
                            invalid_coords_count += 1
                    else:
                        invalid_coords_count += 1
            
            # Determine map center and zoom (after collecting all coordinates)
            if all_lats and all_lons:
                center_lat = sum(all_lats) / len(all_lats)
                center_lon = sum(all_lons) / len(all_lons)
                zoom_start = 18
            else:
                center_lat = 13.0003
                center_lon = 77.0001
                zoom_start = 18
            
            # Create map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
            
            plots_added = 0
            plots_failed = 0
            failed_plots = []
            
            # Render each plot as a polygon
            # CRITICAL: Use ALL corners (A, B, C, D, E, F, ...) not just first 4
            for plot in st.session_state.geo_plots:
                try:
                    corners = plot.get('corners', {})
                    
                    # Sort corners by label (A, B, C, D, E, F, ...) to maintain order
                    sorted_corner_labels = sorted(corners.keys())
                    
                    # Validate coordinates and build coordinate list
                    coords = []
                    for corner_label in sorted_corner_labels:
                        corner_data = corners[corner_label]
                        lat = corner_data.get('lat')
                        lon = corner_data.get('lon')
                        
                        # Check if coordinates are valid numbers
                        if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
                            raise ValueError(f"Invalid coordinates for corner {corner_label}: lat={lat} (type: {type(lat).__name__}), lon={lon} (type: {type(lon).__name__})")
                        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                            raise ValueError(f"Coordinates out of range for corner {corner_label}: lat={lat}, lon={lon}")
                        
                        coords.append([lat, lon])  # Use list instead of tuple for Folium
                    
                    # Ensure we have at least 3 valid corners (minimum for a polygon)
                    if len(coords) < 3:
                        raise ValueError(f"Expected at least 3 corners, got {len(coords)}")
                    
                    # Create popup HTML with all corners
                    popup_lines = [f"<h4>Plot {plot['plot_number']}</h4><hr/>"]
                    for corner_label in sorted_corner_labels:
                        corner_data = corners[corner_label]
                        lat = corner_data.get('lat', 0)
                        lon = corner_data.get('lon', 0)
                        popup_lines.append(f"{corner_label}: {lat:.6f}, {lon:.6f}<br>")
                    popup_html = f"""
                    <div style="font-family: Arial;">
                        {''.join(popup_lines)}
                    </div>
                    """
                    
                    # Create tooltip HTML with all corners
                    tooltip_lines = [f"<b>Plot {plot['plot_number']}</b><br>"]
                    for corner_label in sorted_corner_labels:
                        corner_data = corners[corner_label]
                        lat = corner_data.get('lat', 0)
                        lon = corner_data.get('lon', 0)
                        tooltip_lines.append(f"{corner_label}: {lat:.6f}, {lon:.6f}<br>")
                    tooltip_html = ''.join(tooltip_lines)
                    
                    # Add polygon to map with black outlines matching wireframe (image 8)
                    # Use alternating colors: green, red, blue to match wireframe style
                    plot_num = plot.get('plot_number', 0)
                    color_index = plot_num % 3
                    fill_colors = ['#81C784', '#EF5350', '#42A5F5']  # Green, Red, Blue
                    fill_color = fill_colors[color_index]
                    
                    folium.Polygon(
                        locations=coords,
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=tooltip_html,
                        color='#000000',  # Black outlines matching wireframe
                        fill=True,
                        fill_color=fill_color,
                        fill_opacity=0.5,
                        weight=2,
                        draggable=False  # Disable dragging by default
                    ).add_to(m)
                    plots_added += 1
                    
                except Exception as e:
                    plots_failed += 1
                    plot_num = plot.get('plot_number', 'Unknown')
                    failed_plots.append((plot_num, str(e)))
                    print(f"⚠️ Failed to render Plot {plot_num}: {e}")

            
            # Display error summary if any plots failed
            if plots_failed > 0:
                st.warning(f"⚠️ {plots_failed} plot(s) failed to render.")
                with st.expander("View failed plots", expanded=True):
                    for plot_num, error in failed_plots[:10]:  # Show first 10 errors
                        st.text(f"Plot {plot_num}: {error}")
                    if len(failed_plots) > 10:
                        st.text(f"... and {len(failed_plots) - 10} more errors")
            
            # Fit bounds to show all plots (only if we have valid coordinates)
            if all_lats and all_lons and plots_added > 0:
                try:
                    m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]], padding=(50, 50))
                except Exception as e:
                    st.warning(f"Could not fit bounds: {e}")
            
            # Display the map
            map_data = st_folium(m, width=1400, height=700, returned_objects=[])
            
            # Pass plot data to JavaScript
            # CRITICAL: Include ALL corners (A, B, C, D, E, F, ...) not just first 4
            plots_data_json = json.dumps([
                {
                    'plot_id': p.get('plot_id', ''),
                    'plot_number': p.get('plot_number', 0),
                    'corners': {
                        corner_label: {
                            'lat': corner_data.get('lat', 0),
                            'lon': corner_data.get('lon', 0)
                        }
                        for corner_label, corner_data in p.get('corners', {}).items()
                    }
                }
                for p in st.session_state.geo_plots
            ])
            
            # Add script to pass plot data and move mode state to JavaScript - MUST be before map script
            move_mode_enabled = st.session_state.get('move_plots_enabled', False)
            st.markdown(f"""
            <script id="move-mode-state-script">
                // Store plots data for JavaScript access
                window.plotsDataForMove = {plots_data_json};
                
                // Set move mode state and convert to proper boolean
                window.movePlotsEnabled = {str(move_mode_enabled).lower()};
                if (window.movePlotsEnabled === 'true') window.movePlotsEnabled = true;
                if (window.movePlotsEnabled === 'false') window.movePlotsEnabled = false;
                
                console.log('✅ Move mode state script: Set to', window.movePlotsEnabled, 'Type:', typeof window.movePlotsEnabled);
                
                // Trigger handler update if map is already initialized
                if (window.leafletMapReady && typeof window.setupMovePlotsClickHandler === 'function') {{
                    console.log('🔄 Map already ready, updating click handler immediately');
                    window.setupMovePlotsClickHandler();
                }}
                
                // Also trigger update after delays to ensure it runs
                setTimeout(function() {{
                    if (window.leafletMapReady && typeof window.setupMovePlotsClickHandler === 'function') {{
                        console.log('🔄 Delayed update (100ms): updating click handler');
                        window.setupMovePlotsClickHandler();
                    }}
                }}, 100);
                
                setTimeout(function() {{
                    if (window.leafletMapReady && typeof window.setupMovePlotsClickHandler === 'function') {{
                        console.log('🔄 Delayed update (500ms): updating click handler');
                        window.setupMovePlotsClickHandler();
                    }}
                }}, 500);
            </script>
            """, unsafe_allow_html=True)
        
        # Add custom controls for scale, move, and rotate - positioned below map
        # Pass move mode state as data attribute
        move_mode_enabled = st.session_state.get('move_plots_enabled', False)
        st.components.v1.html("""
        <!-- Simple Confirmation Dialog -->
        <div id="move-plots-dialog-overlay" style="
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.4);
            z-index: 10000;
        ">
            <div id="move-plots-dialog" style="
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                max-width: 400px;
                width: 90%;
                animation: slideIn 0.2s ease-out;
                text-align: center;
            ">
                <div style="font-size: 48px; margin-bottom: 15px;">📍</div>
                <h3 style="margin: 0 0 10px 0; color: #333; font-size: 20px;">Move Plot Layout?</h3>
                <p style="color: #666; margin: 0 0 20px 0; font-size: 14px; line-height: 1.5;">
                    Do you want to move all plots to this position?<br>
                    <strong id="clicked-lat-lng" style="color: #4CAF50; font-size: 13px;"></strong>
                </p>
                <div style="display: flex; gap: 10px; margin-top: 20px;">
                    <button id="move-dialog-confirm" style="
                        flex: 1;
                        padding: 12px;
                        background: #4CAF50;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        font-size: 15px;
                        font-weight: 600;
                        cursor: pointer;
                        transition: background 0.2s;
                    " onmouseover="this.style.background='#45a049'" onmouseout="this.style.background='#4CAF50'">✅ Yes, Move</button>
                    <button id="move-dialog-cancel" style="
                        flex: 1;
                        padding: 12px;
                        background: #f5f5f5;
                        color: #666;
                        border: 1px solid #ddd;
                        border-radius: 6px;
                        font-size: 15px;
                        font-weight: 600;
                        cursor: pointer;
                        transition: background 0.2s;
                    " onmouseover="this.style.background='#e0e0e0'" onmouseout="this.style.background='#f5f5f5'">❌ Cancel</button>
                </div>
            </div>
        </div>
        <style>
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(-10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            @keyframes slideInRight {
                from {
                    opacity: 0;
                    transform: translateX(100px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
        </style>
        
        <div id="map-controls-panel" style="
            background: rgba(255, 255, 255, 0.98);
            padding: 20px 25px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            display: flex;
            gap: 35px;
            align-items: center;
            justify-content: center;
            margin: 25px auto;
            border: 1px solid #d0d0d0;
            font-family: Arial, sans-serif;
            max-width: 95%;
            box-sizing: border-box;
        ">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-weight: bold; color: #333; margin-right: 5px;">Scale</span>
                <button id="map-scale-up" style="
                    width: 40px; height: 40px; border-radius: 50%; border: none;
                    background: #6366f1; color: white; font-size: 18px; cursor: pointer;
                ">+</button>
                <button id="map-scale-down" style="
                    width: 40px; height: 40px; border-radius: 50%; border: none;
                    background: #6366f1; color: white; font-size: 18px; cursor: pointer;
                ">-</button>
                <input type="number" id="map-scale-input" value="0.1" step="0.01" min="0.01" max="1.0" placeholder="Scale Step" style="
                    width: 80px; height: 35px; border: 1px solid #ccc; border-radius: 4px;
                    padding: 5px; text-align: center; font-size: 13px; margin-left: 5px;
                " title="Enter scale increment value (e.g., 0.25). Click +/- to scale by this amount.">
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-weight: bold; color: #333; margin-right: 5px;">Move</span>
                <div style="display: grid; grid-template-columns: repeat(3, 40px); grid-template-rows: repeat(3, 40px); gap: 2px;">
                    <div></div>
                    <button id="map-move-up" style="
                        width: 40px; height: 40px; border-radius: 50%; border: none;
                        background: #6366f1; color: white; font-size: 16px; cursor: pointer;
                    ">▲</button>
                    <div></div>
                    <button id="map-move-left" style="
                        width: 40px; height: 40px; border-radius: 50%; border: none;
                        background: #6366f1; color: white; font-size: 16px; cursor: pointer;
                    ">◄</button>
                    <div></div>
                    <button id="map-move-right" style="
                        width: 40px; height: 40px; border-radius: 50%; border: none;
                        background: #6366f1; color: white; font-size: 16px; cursor: pointer;
                    ">►</button>
                    <div></div>
                    <button id="map-move-down" style="
                        width: 40px; height: 40px; border-radius: 50%; border: none;
                        background: #6366f1; color: white; font-size: 16px; cursor: pointer;
                    ">▼</button>
                    <div></div>
                </div>
                <input type="number" id="map-move-x-input" value="10" step="1" placeholder="X Step" style="
                    width: 70px; height: 35px; border: 1px solid #ccc; border-radius: 4px;
                    padding: 5px; text-align: center; font-size: 13px; margin-left: 5px;
                " title="Enter move increment for X (pixels). Click arrow buttons to move by this amount.">
                <input type="number" id="map-move-y-input" value="10" step="1" placeholder="Y Step" style="
                    width: 70px; height: 35px; border: 1px solid #ccc; border-radius: 4px;
                    padding: 5px; text-align: center; font-size: 13px;
                " title="Enter move increment for Y (pixels). Click arrow buttons to move by this amount.">
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-weight: bold; color: #333; margin-right: 5px;">Rotate</span>
                <input type="number" id="map-rotate-input" value="0" step="1" min="0" max="360" style="
                    width: 60px; height: 40px; border: 1px solid #ccc; border-radius: 4px;
                    padding: 5px; text-align: center; font-size: 14px;
                ">
            </div>
        </div>
        <script>
            (function() {
                // Set plots data immediately from Python
                window.plotsDataForMove = """ + plots_data_json + """;
                console.log('📊 Plots data set:', window.plotsDataForMove ? window.plotsDataForMove.length + ' plots' : 'null/undefined');
                
                // Set move mode state immediately from Python - convert to proper boolean
                window.movePlotsEnabled = """ + str(move_mode_enabled).lower() + """;
                // Ensure it's a proper boolean
                if (window.movePlotsEnabled === 'true') window.movePlotsEnabled = true;
                if (window.movePlotsEnabled === 'false') window.movePlotsEnabled = false;
                console.log('🎯 Map script: Move mode state initialized to:', window.movePlotsEnabled, 'Type:', typeof window.movePlotsEnabled);
                
                let leafletMap = null;
                let mapIframe = null;
                let mapIframeWindow = null; // Store iframe window to access Leaflet
                let Leaflet = null; // Store Leaflet instance from iframe
                let plotPolygons = [];
                let retryCount = 0;
                const maxRetries = 30;
                
                // Move plots state
                let clickedPosition = null;
                let mapClickHandler = null;
                
                // Transform state for plots
                let plotScale = 1.0;
                let plotOffsetLat = 0; // Offset in degrees latitude
                let plotOffsetLng = 0; // Offset in degrees longitude
                let plotRotation = 0;
                
                // Function to find and initialize the Leaflet map
                function findAndInitMap() {
                    if (retryCount >= maxRetries) {
                        console.log('Max retries reached, map not found');
                        return;
                    }
                    retryCount++;
                    
                    // Access parent window to find the map iframe (if we're in an iframe)
                    let searchContext = window;
                    try {
                        if (window.parent && window.parent !== window) {
                            searchContext = window.parent;
                        }
                    } catch(e) {
                        // Can't access parent, use current window
                    }
                    
                    // Try to find the map in iframes (st_folium renders in an iframe)
                    const iframes = searchContext.document.querySelectorAll('iframe');
                    
                    for (let iframe of iframes) {
                        try {
                            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                            const iframeWindow = iframe.contentWindow;
                            
                            // Check if Leaflet is available in iframe
                            if (iframeWindow.L && iframeWindow.L.Map) {
                                // Store references to iframe window and Leaflet
                                mapIframeWindow = iframeWindow;
                                Leaflet = iframeWindow.L;
                                
                                // Method 1: Try to get from _instances
                                if (iframeWindow.L.map && iframeWindow.L.map._instances) {
                                    const instances = Object.values(iframeWindow.L.map._instances);
                                    if (instances.length > 0) {
                                        leafletMap = instances[0];
                                        mapIframe = iframe;
                                        console.log('Found Leaflet map from instances!');
                                        initMapControls();
                                        return;
                                    }
                                }
                                
                                // Method 2: Find by container
                                const mapContainers = iframeDoc.querySelectorAll('.leaflet-container');
                                for (let container of mapContainers) {
                                    if (container._leaflet_id && iframeWindow.L.map._instances) {
                                        leafletMap = iframeWindow.L.map._instances[container._leaflet_id];
                                        if (leafletMap) {
                                            mapIframe = iframe;
                                            console.log('Found Leaflet map from container!');
                                            initMapControls();
                                            return;
                                        }
                                    }
                                }
                                
                                // Method 3: Try iframe window map property
                                if (iframeWindow.map) {
                                    leafletMap = iframeWindow.map;
                                    mapIframe = iframe;
                                    console.log('Found map from iframe window!');
                                    initMapControls();
                                    return;
                                }
                            }
                        } catch(e) {
                            // Cross-origin or other error - continue searching
                            console.log('Could not access iframe:', e.message);
                        }
                    }
                    
                    // If not found, try again after a delay
                    setTimeout(findAndInitMap, 500);
                }
                
                    // Function to get all plot polygons from the map
                function getPlotPolygons() {
                    if (!leafletMap || !Leaflet) return [];
                    
                    const polygons = [];
                    leafletMap.eachLayer(function(layer) {
                        if (layer instanceof Leaflet.Polygon) {
                            // Store original coordinates ONCE - never update them
                            if (!layer.originalLatLngs || !layer.originalLatLngsSet) {
                                const latLngs = layer.getLatLngs()[0];
                                // Deep copy to ensure we store the true original
                                layer.originalLatLngs = JSON.parse(JSON.stringify(latLngs.map(function(ll) {
                                    return [ll.lat, ll.lng];
                                })));
                                // Mark as set to prevent overwriting
                                layer.originalLatLngsSet = true;
                            }
                            polygons.push(layer);
                        }
                    });
                    return polygons;
                }
                
                // Function to reset transformations to original
                function resetPlotTransform() {
                    plotScale = 1.0;
                    plotOffsetLat = 0;
                    plotOffsetLng = 0;
                    plotRotation = 0;
                    
                    // Update input fields to show reset values
                    const scaleInput = document.getElementById('map-scale-input');
                    const moveXInput = document.getElementById('map-move-x-input');
                    const moveYInput = document.getElementById('map-move-y-input');
                    const rotateInput = document.getElementById('map-rotate-input');
                    if (scaleInput) scaleInput.value = '1.0';
                    if (moveXInput) moveXInput.value = '0';
                    if (moveYInput) moveYInput.value = '0';
                    if (rotateInput) rotateInput.value = '0';
                    
                    // Reset all polygons to original coordinates
                    if (!leafletMap || !Leaflet) return;
                    const polygons = getPlotPolygons();
                    polygons.forEach(function(polygon) {
                        if (polygon.originalLatLngs) {
                            const latLngs = polygon.originalLatLngs.map(function(ll) {
                                return Leaflet.latLng(ll[0], ll[1]);
                            });
                            polygon.setLatLngs([latLngs]);
                        }
                    });
                    leafletMap.invalidateSize();
                }
                
                // Function to calculate centroid of all plots
                function getPlotsCentroid(polygons) {
                    if (polygons.length === 0) return null;
                    
                    let totalLat = 0;
                    let totalLng = 0;
                    let pointCount = 0;
                    
                    polygons.forEach(function(polygon) {
                        const latLngs = polygon.originalLatLngs || polygon.getLatLngs()[0];
                        latLngs.forEach(function(ll) {
                            const lat = Array.isArray(ll) ? ll[0] : ll.lat;
                            const lng = Array.isArray(ll) ? ll[1] : ll.lng;
                            totalLat += lat;
                            totalLng += lng;
                            pointCount++;
                        });
                    });
                    
                    return {
                        lat: totalLat / pointCount,
                        lng: totalLng / pointCount
                    };
                }
                
                // Function to apply transformations to plot polygons
                function applyPlotTransform() {
                    if (!leafletMap || !Leaflet) return;
                    
                    const polygons = getPlotPolygons();
                    if (polygons.length === 0) {
                        console.log('No plot polygons found');
                        return;
                    }
                    
                    const centroid = getPlotsCentroid(polygons);
                    if (!centroid) return;
                    
                    polygons.forEach(function(polygon) {
                        // Always use the stored originalLatLngs (true original, never updated)
                        if (!polygon.originalLatLngs || !polygon.originalLatLngsSet) {
                            const latLngs = polygon.getLatLngs()[0];
                            polygon.originalLatLngs = JSON.parse(JSON.stringify(latLngs.map(function(ll) {
                                return [ll.lat, ll.lng];
                            })));
                            polygon.originalLatLngsSet = true;
                        }
                        
                        const originalLatLngs = polygon.originalLatLngs;
                        const transformedLatLngs = originalLatLngs.map(function(ll) {
                            let lat = Array.isArray(ll) ? ll[0] : ll.lat;
                            let lng = Array.isArray(ll) ? ll[1] : ll.lng;
                            
                            // Translate to origin (centroid)
                            let x = lng - centroid.lng;
                            let y = lat - centroid.lat;
                            
                            // Apply rotation
                            if (plotRotation !== 0) {
                                const angleRad = (plotRotation * Math.PI) / 180;
                                const cos = Math.cos(angleRad);
                                const sin = Math.sin(angleRad);
                                const newX = x * cos - y * sin;
                                const newY = x * sin + y * cos;
                                x = newX;
                                y = newY;
                            }
                            
                            // Apply absolute scale (from original, not cumulative)
                            x *= plotScale;
                            y *= plotScale;
                            
                            // Translate back and apply absolute offset (from original, not cumulative)
                            lng = centroid.lng + x + plotOffsetLng;
                            lat = centroid.lat + y + plotOffsetLat;
                            
                            return Leaflet.latLng(lat, lng);
                        });
                        
                        polygon.setLatLngs([transformedLatLngs]);
                    });
                    
                    leafletMap.invalidateSize();
                }
                
                function initMapControls() {
                    if (!leafletMap || !Leaflet) {
                        console.log('Map or Leaflet not found, retrying...');
                        setTimeout(findAndInitMap, 500);
                        return;
                    }
                    
                    console.log('Initializing plot transform controls...');
                    
                    // Wait a bit for polygons to load, then initialize
                    setTimeout(function() {
                        // Initialize plot polygons and store original coordinates
                        plotPolygons = getPlotPolygons();
                        console.log('Found', plotPolygons.length, 'plot polygons');
                        
                        // Store original coordinates and disable dragging by default for all polygons
                        plotPolygons.forEach(function(polygon) {
                            if (!polygon.originalLatLngs) {
                                const latLngs = polygon.getLatLngs()[0];
                                polygon.originalLatLngs = latLngs.map(function(ll) {
                                    return [ll.lat, ll.lng];
                                });
                            }
                            
                            // Disable dragging by default (will be enabled only when move mode is on)
                            if (polygon.dragging) {
                                polygon.dragging.disable();
                            }
                            polygon.options.draggable = false;
                        });
                        
                        console.log('Disabled dragging for all polygons by default');
                        
                        // Connect controls after polygons are ready
                        connectControls();
                        
                        // Setup move plots click handler
                        setupMovePlotsClickHandler();
                        
                        // Mark map as ready and expose handler function globally
                        window.leafletMapReady = true;
                        window.setupMovePlotsClickHandler = setupMovePlotsClickHandler;
                    }, 500);
                }
                
                function connectControls() {
                    // Buttons are in the same document as this script
                    const scaleUpBtn = document.getElementById('map-scale-up');
                    const scaleDownBtn = document.getElementById('map-scale-down');
                    const scaleInput = document.getElementById('map-scale-input');
                    
                    // Scale input stores the increment/decrement value (not the current scale)
                    // Initialize scale display separately
                    let scaleDisplay = null;
                    if (scaleInput && scaleInput.parentNode) {
                        scaleDisplay = document.createElement('span');
                        scaleDisplay.id = 'map-scale-display';
                        scaleDisplay.style.cssText = 'margin-left: 5px; font-size: 12px; color: #666; min-width: 50px; display: inline-block;';
                        scaleDisplay.textContent = 'Current: ' + plotScale.toFixed(2) + 'x';
                        scaleInput.parentNode.appendChild(scaleDisplay);
                    }
                    
                    // Scale input handler - stores the step value for +/- buttons
                    if (scaleInput) {
                        scaleInput.onchange = function() {
                            // Just validate the step value, don't apply transform
                            const stepValue = parseFloat(this.value) || 0.1;
                            if (stepValue < 0.01 || stepValue > 1.0) {
                                this.value = 0.1;
                            }
                        };
                    }
                    
                    if (scaleUpBtn) {
                        scaleUpBtn.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            console.log('Plot scale up clicked!');
                            const stepValue = parseFloat(scaleInput?.value || 0.1) || 0.1;
                            // Add step value to current scale
                            plotScale = plotScale + stepValue;
                            // Update display
                            if (scaleDisplay) {
                                scaleDisplay.textContent = 'Current: ' + plotScale.toFixed(2) + 'x';
                            }
                            applyPlotTransform();
                            return false;
                        };
                        console.log('Plot scale up button connected');
                    }
                    
                    if (scaleDownBtn) {
                        scaleDownBtn.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            console.log('Plot scale down clicked!');
                            const stepValue = parseFloat(scaleInput?.value || 0.1) || 0.1;
                            // Subtract step value from current scale
                            plotScale = Math.max(0.1, plotScale - stepValue);
                            // Update display
                            if (scaleDisplay) {
                                scaleDisplay.textContent = 'Current: ' + plotScale.toFixed(2) + 'x';
                            }
                            applyPlotTransform();
                            return false;
                        };
                        console.log('Plot scale down button connected');
                    }
                    
                    // Move controls - move plots, not map
                    const moveUpBtn = document.getElementById('map-move-up');
                    const moveDownBtn = document.getElementById('map-move-down');
                    const moveLeftBtn = document.getElementById('map-move-left');
                    const moveRightBtn = document.getElementById('map-move-right');
                    const moveXInput = document.getElementById('map-move-x-input');
                    const moveYInput = document.getElementById('map-move-y-input');
                    
                    // Move input handlers - stores the step value for arrow buttons
                    // Rough conversion: 1 pixel ≈ 0.0001 degrees at typical zoom levels
                    // Initialize move display separately
                    let moveDisplay = null;
                    if (moveYInput && moveYInput.parentNode) {
                        moveDisplay = document.createElement('div');
                        moveDisplay.id = 'map-move-display';
                        moveDisplay.style.cssText = 'margin-left: 5px; font-size: 11px; color: #666;';
                        const currentXPx = Math.round(plotOffsetLng / 0.0001);
                        const currentYPx = Math.round(-plotOffsetLat / 0.0001);
                        moveDisplay.innerHTML = 'Current: X=' + currentXPx + 'px, Y=' + currentYPx + 'px';
                        moveYInput.parentNode.appendChild(moveDisplay);
                    }
                    
                    if (moveXInput) {
                        moveXInput.onchange = function() {
                            // Just validate the step value, don't apply transform
                            const stepValue = parseFloat(this.value) || 10;
                            if (stepValue < 1) {
                                this.value = 10;
                            }
                        };
                    }
                    
                    if (moveYInput) {
                        moveYInput.onchange = function() {
                            // Just validate the step value, don't apply transform
                            const stepValue = parseFloat(this.value) || 10;
                            if (stepValue < 1) {
                                this.value = 10;
                            }
                        };
                    }
                    
                    if (moveUpBtn) {
                        moveUpBtn.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            console.log('Plot move up clicked!');
                            const stepValue = parseFloat(moveYInput?.value || 10) || 10;
                            // Convert pixels to degrees and subtract (move north)
                            plotOffsetLat += stepValue * 0.0001;
                            // Update display
                            if (moveDisplay) {
                                const currentXPx = Math.round(plotOffsetLng / 0.0001);
                                const currentYPx = Math.round(-plotOffsetLat / 0.0001);
                                moveDisplay.innerHTML = 'Current: X=' + currentXPx + 'px, Y=' + currentYPx + 'px';
                            }
                            applyPlotTransform();
                            return false;
                        };
                        console.log('Plot move up button connected');
                    }
                    
                    if (moveDownBtn) {
                        moveDownBtn.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            console.log('Plot move down clicked!');
                            const stepValue = parseFloat(moveYInput?.value || 10) || 10;
                            // Convert pixels to degrees and add (move south)
                            plotOffsetLat -= stepValue * 0.0001;
                            // Update display
                            if (moveDisplay) {
                                const currentXPx = Math.round(plotOffsetLng / 0.0001);
                                const currentYPx = Math.round(-plotOffsetLat / 0.0001);
                                moveDisplay.innerHTML = 'Current: X=' + currentXPx + 'px, Y=' + currentYPx + 'px';
                            }
                            applyPlotTransform();
                            return false;
                        };
                        console.log('Plot move down button connected');
                    }
                    
                    if (moveLeftBtn) {
                        moveLeftBtn.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            console.log('Plot move left clicked!');
                            const stepValue = parseFloat(moveXInput?.value || 10) || 10;
                            // Convert pixels to degrees and subtract (move west)
                            plotOffsetLng -= stepValue * 0.0001;
                            // Update display
                            if (moveDisplay) {
                                const currentXPx = Math.round(plotOffsetLng / 0.0001);
                                const currentYPx = Math.round(-plotOffsetLat / 0.0001);
                                moveDisplay.innerHTML = 'Current: X=' + currentXPx + 'px, Y=' + currentYPx + 'px';
                            }
                            applyPlotTransform();
                            return false;
                        };
                        console.log('Plot move left button connected');
                    }
                    
                    if (moveRightBtn) {
                        moveRightBtn.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            console.log('Plot move right clicked!');
                            const stepValue = parseFloat(moveXInput?.value || 10) || 10;
                            // Convert pixels to degrees and add (move east)
                            plotOffsetLng += stepValue * 0.0001;
                            // Update display
                            if (moveDisplay) {
                                const currentXPx = Math.round(plotOffsetLng / 0.0001);
                                const currentYPx = Math.round(-plotOffsetLat / 0.0001);
                                moveDisplay.innerHTML = 'Current: X=' + currentXPx + 'px, Y=' + currentYPx + 'px';
                            }
                            applyPlotTransform();
                            return false;
                        };
                        console.log('Plot move right button connected');
                    }
                    
                    
                    // Rotate control - rotate plots
                    const rotateInput = document.getElementById('map-rotate-input');
                    if (rotateInput) {
                        rotateInput.disabled = false;
                        rotateInput.style.backgroundColor = '#ffffff';
                        rotateInput.style.cursor = 'text';
                        rotateInput.title = 'Rotate plots (degrees)';
                        rotateInput.onchange = function() {
                            plotRotation = parseFloat(this.value) || 0;
                            applyPlotTransform();
                        };
                        console.log('Plot rotate input connected');
                    }
                    
                    console.log('Plot transform controls initialized!');
                }
                
                // Function to setup map click handler for move plots
                function setupMovePlotsClickHandler() {
                    if (!leafletMap || !Leaflet) {
                        console.log('Cannot setup move click handler: map or Leaflet not ready');
                        return;
                    }
                    
                    // Remove existing handler if any
                    if (mapClickHandler) {
                        leafletMap.off('click', mapClickHandler);
                        mapClickHandler = null;
                        console.log('Removed existing click handler');
                    }
                    
                    // Check if move mode is enabled - check multiple ways and normalize
                    let moveModeValue = window.movePlotsEnabled;
                    
                    // Normalize to boolean
                    if (moveModeValue === 'true' || moveModeValue === true) {
                        moveModeValue = true;
                    } else if (moveModeValue === 'false' || moveModeValue === false || moveModeValue === null || moveModeValue === undefined) {
                        moveModeValue = false;
                    } else {
                        moveModeValue = !!moveModeValue; // Convert to boolean
                    }
                    
                    const moveModeEnabled = moveModeValue === true;
                    
                    console.log('Move mode check:', {
                        windowValue: window.movePlotsEnabled,
                        normalizedValue: moveModeValue,
                        type: typeof window.movePlotsEnabled,
                        enabled: moveModeEnabled
                    });
                    
                    // Get all polygons and set their draggable state
                    const polygons = getPlotPolygons();
                    polygons.forEach(function(polygon) {
                        if (moveModeEnabled) {
                            // Enable dragging for polygons
                            if (polygon.dragging) {
                                polygon.dragging.enable();
                            }
                            polygon.options.draggable = true;
                            console.log('Enabled dragging for polygon');
                        } else {
                            // Disable dragging for polygons
                            if (polygon.dragging) {
                                polygon.dragging.disable();
                            }
                            polygon.options.draggable = false;
                            console.log('Disabled dragging for polygon');
                        }
                    });
                    
                    if (moveModeEnabled) {
                        // Remove any existing handlers first
                        if (mapClickHandler) {
                            leafletMap.off('click', mapClickHandler);
                        }
                        
                        // Enable click handler for moving plots
                        mapClickHandler = function(e) {
                            // Check current move mode state dynamically
                            let currentMoveMode = window.movePlotsEnabled;
                            if (currentMoveMode === 'true' || currentMoveMode === true) {
                                currentMoveMode = true;
                            } else {
                                currentMoveMode = false;
                            }
                            
                            if (!currentMoveMode) {
                                console.log('Map click ignored - move mode is disabled');
                                return true; // Allow normal map behavior
                            }
                            
                            console.log('🔵🔵🔵 CLICK HANDLER TRIGGERED!', e);
                            console.log('Event details:', {
                                type: e.type,
                                latlng: e.latlng,
                                originalEvent: e.originalEvent
                            });
                            
                            // Prevent default map behavior
                            if (e.originalEvent) {
                                e.originalEvent.preventDefault();
                                e.originalEvent.stopPropagation();
                            }
                            
                            if (e && e.latlng) {
                                clickedPosition = {
                                    lat: e.latlng.lat,
                                    lng: e.latlng.lng
                                };
                                
                                console.log('✅ Map clicked at:', clickedPosition);
                                
                                // Show confirmation dialog
                                try {
                                    console.log('Calling showMovePlotsDialog...');
                                    showMovePlotsDialog(clickedPosition);
                                    console.log('showMovePlotsDialog called successfully');
                                } catch(err) {
                                    console.error('❌ Error showing dialog:', err);
                                    console.error('Error stack:', err.stack);
                                    alert('Error showing dialog: ' + err.message);
                                }
                            } else {
                                console.error('❌ Invalid click event - no latlng:', e);
                            }
                            
                            return false;
                        };
                        
                        // Add click handler with high priority
                        leafletMap.on('click', mapClickHandler);
                        
                        // Verify handler was added
                        const hasClickHandler = leafletMap.listens('click');
                        console.log('Leaflet click handler registered:', hasClickHandler);
                        
                        // Also add to map container directly as backup (without cloning)
                        const mapContainer = leafletMap.getContainer();
                        if (mapContainer) {
                            // Remove old listener if exists
                            if (mapContainer._movePlotsClickHandler) {
                                mapContainer.removeEventListener('click', mapContainer._movePlotsClickHandler, true);
                            }
                            
                            // Create new handler that checks current move mode state dynamically
                            mapContainer._movePlotsClickHandler = function(domEvent) {
                                // Check current move mode state (not the captured variable)
                                let currentMoveMode = window.movePlotsEnabled;
                                if (currentMoveMode === 'true' || currentMoveMode === true) {
                                    currentMoveMode = true;
                                } else {
                                    currentMoveMode = false;
                                }
                                
                                if (currentMoveMode) {
                                    console.log('🔵🔵🔵 DIRECT CONTAINER CLICK DETECTED!', domEvent);
                                    // Get lat/lng from Leaflet
                                    try {
                                        const point = leafletMap.mouseEventToLatLng(domEvent);
                                        console.log('Point from mouseEventToLatLng:', point);
                                        if (point) {
                                            clickedPosition = {
                                                lat: point.lat,
                                                lng: point.lng
                                            };
                                            console.log('✅ Container click at:', clickedPosition);
                                            showMovePlotsDialog(clickedPosition);
                                            domEvent.preventDefault();
                                            domEvent.stopPropagation();
                                            return false;
                                        }
                                    } catch(err) {
                                        console.error('Error getting lat/lng from click:', err);
                                    }
                                } else {
                                    console.log('Container click ignored - move mode is disabled');
                                }
                            };
                            
                            // Add listener with capture phase
                            mapContainer.addEventListener('click', mapContainer._movePlotsClickHandler, true);
                            console.log('✅ Direct container click listener added');
                            
                            mapContainer.style.cursor = 'crosshair';
                            console.log('Cursor set to crosshair');
                        } else {
                            console.error('❌ Map container not found!');
                        }
                        
                        console.log('✅ Move plots click handler ENABLED - ready to receive clicks');
                        console.log('Handler function:', mapClickHandler);
                        
                        // Test: Try to manually trigger to verify dialog works
                        console.log('Testing dialog availability...');
                        try {
                            const testOverlay = document.getElementById('move-plots-dialog-overlay');
                            const parentOverlay = window.parent ? window.parent.document.getElementById('move-plots-dialog-overlay') : null;
                            console.log('Dialog overlay exists (current):', !!testOverlay);
                            console.log('Dialog overlay exists (parent):', !!parentOverlay);
                            
                            // Test if we can trigger a simple click test
                            if (mapContainer) {
                                console.log('Adding test click listener to verify clicks work...');
                                mapContainer.addEventListener('click', function testClick(e) {
                                    console.log('🧪 TEST CLICK DETECTED on container!', e);
                                }, {once: true, capture: true});
                            }
                        } catch(err) {
                            console.error('Error checking dialog:', err);
                        }
                    } else {
                        // Disable click handler - remove all handlers when move mode is disabled
                        console.log('Move mode is disabled - removing all click handlers');
                        
                        // Remove Leaflet map click handler
                        if (mapClickHandler) {
                            leafletMap.off('click', mapClickHandler);
                            mapClickHandler = null;
                            console.log('Removed Leaflet map click handler');
                        }
                        
                        // Remove container click handler
                        const mapContainer = leafletMap.getContainer();
                        if (mapContainer && mapContainer._movePlotsClickHandler) {
                            mapContainer.removeEventListener('click', mapContainer._movePlotsClickHandler, true);
                            mapContainer._movePlotsClickHandler = null;
                            console.log('Removed container click handler');
                        }
                        
                        // Reset cursor
                        if (mapContainer) {
                            mapContainer.style.cursor = '';
                            console.log('Reset cursor to default');
                        }
                        
                        console.log('✅ Move plots click handler DISABLED - all handlers removed');
                    }
                }
                
                // Function to create dialog fallback if not found
                function createMoveDialogFallback(doc, position) {
                    console.log('Creating fallback dialog...');
                    
                    // Create overlay
                    const overlay = doc.createElement('div');
                    overlay.id = 'move-plots-dialog-overlay-fallback';
                    overlay.style.cssText = 'display: flex; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.4); z-index: 10000; justify-content: center; align-items: center;';
                    
                    // Create dialog
                    const dialog = doc.createElement('div');
                    dialog.style.cssText = 'background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); max-width: 400px; width: 90%; text-align: center;';
                    
                    dialog.innerHTML = `
                        <div style="font-size: 48px; margin-bottom: 15px;">📍</div>
                        <h3 style="margin: 0 0 10px 0; color: #333; font-size: 20px;">Move Plot Layout?</h3>
                        <p style="color: #666; margin: 0 0 20px 0; font-size: 14px;">
                            Do you want to move all plots to this position?<br>
                            <strong style="color: #4CAF50;">${position.lat.toFixed(6)}, ${position.lng.toFixed(6)}</strong>
                        </p>
                        <div style="display: flex; gap: 10px; margin-top: 20px;">
                            <button id="fallback-confirm" style="flex: 1; padding: 12px; background: #4CAF50; color: white; border: none; border-radius: 6px; font-size: 15px; font-weight: 600; cursor: pointer;">✅ Yes, Move</button>
                            <button id="fallback-cancel" style="flex: 1; padding: 12px; background: #f5f5f5; color: #666; border: 1px solid #ddd; border-radius: 6px; font-size: 15px; font-weight: 600; cursor: pointer;">❌ Cancel</button>
                        </div>
                    `;
                    
                    overlay.appendChild(dialog);
                    doc.body.appendChild(overlay);
                    
                    // Setup buttons
                    const confirmBtn = dialog.querySelector('#fallback-confirm');
                    const cancelBtn = dialog.querySelector('#fallback-cancel');
                    
                    confirmBtn.onclick = function() {
                        if (applyPlotMove(position)) {
                            overlay.remove();
                            showMoveNotification('✅ All plots moved! Updating coordinates...');
                            sendCoordinateUpdateToStreamlit(position);
                        }
                    };
                    
                    cancelBtn.onclick = function() {
                        overlay.remove();
                    };
                    
                    overlay.onclick = function(e) {
                        if (e.target === overlay) {
                            overlay.remove();
                        }
                    };
                    
                    console.log('✅ Fallback dialog created and shown');
                }
                
                // Function to show move plots confirmation dialog - using native JavaScript confirm
                function showMovePlotsDialog(position) {
                    console.log('🔵🔵🔵 showMovePlotsDialog CALLED with position:', position);
                    
                    // Check if plots data is available
                    let plotsData = window.plotsDataForMove;
                    if (!plotsData || (Array.isArray(plotsData) && plotsData.length === 0)) {
                        console.warn('⚠️ Plots data not available when dialog opened');
                        // Try to get from parent
                        try {
                            if (window.parent && window.parent.plotsDataForMove) {
                                plotsData = window.parent.plotsDataForMove;
                                window.plotsDataForMove = plotsData;
                                console.log('✅ Got plots data from parent window');
                            }
                        } catch(e) {
                            console.error('Cannot access parent:', e);
                        }
                    }
                    
                    if (!plotsData || (Array.isArray(plotsData) && plotsData.length === 0)) {
                        alert('⚠️ Plots data not available. Please ensure you are on Step 8 (Google Map Preview) with plots loaded. Refresh the page if needed.');
                        return;
                    }
                    
                    // Use native JavaScript confirm dialog - always centered and visible
                    const message = `📍 Move Plot Layout?\n\nDo you want to move all plots to this position?\n\nCoordinates: ${position.lat.toFixed(6)}, ${position.lng.toFixed(6)}`;
                    
                    const userConfirmed = confirm(message);
                    
                    if (userConfirmed) {
                        console.log('User confirmed plot move');
                        if (applyPlotMove(position)) {
                            console.log('Plot move applied successfully, sending coordinates to Streamlit...');
                            // Show notification
                            showMoveNotification('✅ All plots moved! Updating coordinates...');
                            // Send update to Streamlit
                            sendCoordinateUpdateToStreamlit(position);
                        } else {
                            console.error('Failed to apply plot move');
                        }
                    } else {
                        console.log('User cancelled plot move');
                    }
                }
                
                // Function to apply plot move transformation - moves all plots so their centroid aligns with target
                function applyPlotMove(targetPosition) {
                    if (!leafletMap || !Leaflet) {
                        console.error('Cannot apply plot move: missing dependencies');
                        alert('Map not ready. Please wait a moment and try again.');
                        return false;
                    }
                    
                    // Check for plots data in multiple ways
                    let plotsData = window.plotsDataForMove;
                    
                    // Try to get from parent window if not available
                    if (!plotsData || plotsData.length === 0) {
                        try {
                            if (window.parent && window.parent.plotsDataForMove) {
                                plotsData = window.parent.plotsDataForMove;
                                window.plotsDataForMove = plotsData; // Sync to current window
                            }
                        } catch(e) {
                            console.log('Cannot access parent window:', e);
                        }
                    }
                    
                    // Check if it's an array or needs parsing
                    if (typeof plotsData === 'string') {
                        try {
                            plotsData = JSON.parse(plotsData);
                        } catch(e) {
                            console.error('Failed to parse plots data:', e);
                        }
                    }
                    
                    plotsData = plotsData || [];
                    
                    console.log('Plots data check:', {
                        available: !!plotsData,
                        length: plotsData.length,
                        type: typeof plotsData,
                        isArray: Array.isArray(plotsData)
                    });
                    
                    if (!plotsData || plotsData.length === 0) {
                        console.error('No plots data available. Available window properties:', Object.keys(window).filter(k => k.includes('plot')));
                        alert('No plots data available. Please ensure plots are loaded on the map. If the issue persists, refresh the page.');
                        return false;
                    }
                    
                    // Calculate current centroid of all plots
                    let totalLat = 0;
                    let totalLng = 0;
                    let pointCount = 0;
                    
                    plotsData.forEach(function(plot) {
                        for (let corner in plot.corners) {
                            totalLat += plot.corners[corner].lat;
                            totalLng += plot.corners[corner].lon;
                            pointCount++;
                        }
                    });
                    
                    if (pointCount === 0) {
                        alert('No plot coordinates found');
                        return false;
                    }
                    
                    const currentCentroidLat = totalLat / pointCount;
                    const currentCentroidLng = totalLng / pointCount;
                    
                    // Calculate offset to move centroid to target position
                    const offsetLat = targetPosition.lat - currentCentroidLat;
                    const offsetLng = targetPosition.lng - currentCentroidLng;
                    
                    console.log('Current centroid:', currentCentroidLat, currentCentroidLng);
                    console.log('Target position:', targetPosition.lat, targetPosition.lng);
                    console.log('Applying offset:', offsetLat, offsetLng);
                    
                    // Get polygons and apply offset
                    const polygons = getPlotPolygons();
                    if (polygons.length === 0) {
                        alert('No plots found on map');
                        return false;
                    }
                    
                    // Apply offset to all plots
                    polygons.forEach(function(polygon) {
                        if (polygon.originalLatLngs) {
                            const newLatLngs = polygon.originalLatLngs.map(function(ll) {
                                const newLat = ll[0] + offsetLat;
                                const newLng = ll[1] + offsetLng;
                                return Leaflet.latLng(newLat, newLng);
                            });
                            
                            // Update polygon
                            polygon.setLatLngs([newLatLngs]);
                            
                            // Update original coordinates for future transformations
                            polygon.originalLatLngs = newLatLngs.map(function(ll) {
                                return [ll.lat, ll.lng];
                            });
                        }
                    });
                    
                    leafletMap.invalidateSize();
                    return true;
                }
                
                // Function to send coordinate update to Streamlit
                function sendCoordinateUpdateToStreamlit(targetPosition) {
                    console.log('sendCoordinateUpdateToStreamlit called with position:', targetPosition);
                    
                    // Check for plots data in multiple ways
                    let plotsData = window.plotsDataForMove;
                    
                    // Try to get from parent window if not available
                    if (!plotsData || plotsData.length === 0) {
                        try {
                            if (window.parent && window.parent.plotsDataForMove) {
                                plotsData = window.parent.plotsDataForMove;
                                window.plotsDataForMove = plotsData; // Sync to current window
                            }
                        } catch(e) {
                            console.log('Cannot access parent window:', e);
                        }
                    }
                    
                    // Check if it's an array or needs parsing
                    if (typeof plotsData === 'string') {
                        try {
                            plotsData = JSON.parse(plotsData);
                        } catch(e) {
                            console.error('Failed to parse plots data:', e);
                        }
                    }
                    
                    plotsData = plotsData || [];
                    
                    console.log('Plots data for update:', {
                        available: !!plotsData,
                        length: plotsData.length,
                        type: typeof plotsData
                    });
                    
                    if (!plotsData || plotsData.length === 0) {
                        console.error('No plots data available for coordinate update');
                        alert('No plots data available. Please refresh the page and try again.');
                        return;
                    }
                    
                    console.log('Processing', plotsData.length, 'plots');
                    
                    // Calculate current centroid from ORIGINAL data (before visual move)
                    let totalLat = 0;
                    let totalLng = 0;
                    let pointCount = 0;
                    
                    plotsData.forEach(function(plot) {
                        for (let corner in plot.corners) {
                            totalLat += plot.corners[corner].lat;
                            totalLng += plot.corners[corner].lon;
                            pointCount++;
                        }
                    });
                    
                    const currentCentroidLat = totalLat / pointCount;
                    const currentCentroidLng = totalLng / pointCount;
                    
                    const offsetLat = targetPosition.lat - currentCentroidLat;
                    const offsetLng = targetPosition.lng - currentCentroidLng;
                    
                    console.log('Current centroid:', currentCentroidLat, currentCentroidLng);
                    console.log('Target position:', targetPosition.lat, targetPosition.lng);
                    console.log('Offset:', offsetLat, offsetLng);
                    
                    // Create update data with all plot coordinates
                    const updatedPlots = plotsData.map(function(plot) {
                        const updatedCorners = {};
                        for (let corner in plot.corners) {
                            updatedCorners[corner] = {
                                lat: plot.corners[corner].lat + offsetLat,
                                lon: plot.corners[corner].lon + offsetLng
                            };
                        }
                        return {
                            plot_id: plot.plot_id,
                            plot_number: plot.plot_number,
                            corners: updatedCorners
                        };
                    });
                    
                    const updateData = {
                        type: 'plot_coordinates_update',
                        target_position: targetPosition,
                        current_centroid: {
                            lat: currentCentroidLat,
                            lng: currentCentroidLng
                        },
                        offset: {
                            lat: offsetLat,
                            lng: offsetLng
                        },
                        updated_plots: updatedPlots,
                        timestamp: Date.now()
                    };
                    
                    console.log('Sending update data:', updateData);
                    console.log('Updated plots count:', updatedPlots.length);
                    
                    // Send to Streamlit via URL parameter
                    const url = new URL(window.location);
                    url.searchParams.set('plot_move_update', JSON.stringify(updateData));
                    url.searchParams.set('_timestamp', Date.now());
                    console.log('Redirecting to:', url.toString());
                    window.location.href = url.toString();
                }
                
                // Function to show move notification
                function showMoveNotification(message) {
                    // Create or update notification element
                    let notification = document.getElementById('move-notification');
                    if (!notification) {
                        notification = document.createElement('div');
                        notification.id = 'move-notification';
                        notification.style.cssText = `
                            position: fixed;
                            top: 20px;
                            right: 20px;
                            background: #4CAF50;
                            color: white;
                            padding: 12px 20px;
                            border-radius: 6px;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                            z-index: 10001;
                            font-family: Arial, sans-serif;
                            font-size: 14px;
                            font-weight: 500;
                            animation: slideInRight 0.3s ease-out;
                        `;
                        document.body.appendChild(notification);
                    }
                    
                    notification.textContent = message;
                    notification.style.display = 'block';
                    
                    // Auto-hide after 3 seconds
                    setTimeout(function() {
                        notification.style.display = 'none';
                    }, 3000);
                }
                
                // Function to check move mode state and update handler
                let lastMoveModeState = null;
                let handlerSetupCount = 0;
                function checkMoveModeState() {
                    // Check multiple ways to get the state - also check parent window
                    let currentState = window.movePlotsEnabled;
                    
                    // Try to get from parent window if available
                    try {
                        if (window.parent && window.parent.movePlotsEnabled !== undefined) {
                            currentState = window.parent.movePlotsEnabled;
                            window.movePlotsEnabled = currentState; // Sync to current window
                        }
                    } catch(e) {
                        // Can't access parent, use current window
                    }
                    
                    const moveModeEnabled = currentState === true || 
                                          currentState === 'true' || 
                                          (typeof currentState !== 'undefined' && currentState && currentState !== 'false' && currentState !== false);
                    
                    // Only update handler if state actually changed (to avoid constant removal/re-adding)
                    if (leafletMap && Leaflet) {
                        if (lastMoveModeState !== moveModeEnabled) {
                            console.log('🔄 Move mode state changed:', lastMoveModeState, '->', moveModeEnabled, '(raw value:', currentState, 'type:', typeof currentState, ')');
                            lastMoveModeState = moveModeEnabled;
                            handlerSetupCount++;
                            console.log('Setting up handler (count:', handlerSetupCount, ')');
                            setupMovePlotsClickHandler();
                        }
                    } else {
                        // Log state even if map not ready
                        if (lastMoveModeState !== moveModeEnabled) {
                            console.log('Move mode state:', moveModeEnabled, '(raw:', currentState, 'map not ready yet)');
                            lastMoveModeState = moveModeEnabled;
                        }
                    }
                }
                
                // Start searching for the map
                setTimeout(findAndInitMap, 1000);
                
                // Check move mode state more frequently
                setInterval(checkMoveModeState, 200);
                
                // Also check immediately and after delays
                checkMoveModeState();
                setTimeout(checkMoveModeState, 500);
                setTimeout(checkMoveModeState, 1500);
                setTimeout(checkMoveModeState, 3000);
            })();
        </script>
        """, height=200)
        
        # Navigation buttons for Step 8 (final step)
        col_db, col_btn1, col_btn2, col_btn3 = st.columns([2, 4, 1, 1])
        with col_db:
            # Small collapsible expander matching Publish button size
            with st.expander("Save to DB", expanded=False):
                profile_name = st.text_input("Profile name", value="default", key="modal_profile_name_step8")
                
                def ensure_db():
                    conn = sqlite3.connect("plot_numbers.db")
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS profiles (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            profile_name TEXT NOT NULL,
                            plot_id TEXT NOT NULL,
                            plot_number INTEGER,
                            created_at TEXT NOT NULL
                        )
                        """
                    )
                    return conn
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 Save", key="modal_save_step8", use_container_width=True):
                        if not st.session_state.get('plots'):
                            st.error("No plots to save.")
                        else:
                            conn = ensure_db()
                            now = datetime.utcnow().isoformat()
                            rows = [(profile_name, p.get('plot_id'), p.get('plot_number'), now) for p in st.session_state.plots]
                            with conn:
                                conn.executemany(
                                    "INSERT INTO profiles (profile_name, plot_id, plot_number, created_at) VALUES (?, ?, ?, ?)",
                                    rows
                                )
                            conn.close()
                            st.success(f"Saved {len(rows)} entries to profile '{profile_name}'.")
                
                with col2:
                    if st.button("📥 Load", key="modal_load_step8", use_container_width=True):
                        if not st.session_state.get('plots'):
                            st.warning("No plots available to load into.")
                        else:
                            conn = ensure_db()
                            cur = conn.cursor()
                            cur.execute(
                                "SELECT plot_id, plot_number FROM profiles WHERE profile_name = ? ORDER BY id DESC",
                                (profile_name,)
                            )
                            rows = cur.fetchall()
                            conn.close()
                            if not rows:
                                st.warning(f"No saved data for profile '{profile_name}'.")
                            else:
                                # Use most recent occurrence per plot_id
                                seen = {}
                                for plot_id, plot_number in rows:
                                    if plot_id not in seen:
                                        seen[plot_id] = plot_number
                                applied = 0
                                for p in st.session_state.plots:
                                    if p.get('plot_id') in seen:
                                        p['plot_number'] = seen[p['plot_id']]
                                        applied += 1
                                st.session_state.geo_plots = []
                                st.success(f"Applied {applied} plot numbers from profile '{profile_name}'.")
                                st.rerun()
                
                if st.button("📄 List profiles", key="modal_list_step8", use_container_width=True):
                    conn = ensure_db()
                    df = pd.read_sql_query(
                        "SELECT profile_name, COUNT(*) as entries, MIN(created_at) as first_saved, MAX(created_at) as last_saved FROM profiles GROUP BY profile_name ORDER BY last_saved DESC",
                        conn
                    )
                    conn.close()
                    if df.empty:
                        st.info("No profiles saved yet.")
                    else:
                        st.dataframe(df, use_container_width=True)
        
        with col_btn1:
            st.write("")  # Spacer to align with controls panel above
        with col_btn2:
            if st.button("Prev", type="primary", key="prev_step8"):
                st.session_state.current_step = 7  # Go back to Step 7
                st.rerun()
        with col_btn3:
            # Keep Publish button as standard primary button
            if st.button("Publish", type="primary", use_container_width=True, key="publish_step8"):
                st.success("✅ Map published successfully!")
                # Stay on Step 8 as it's the final step

st.divider()
st.caption("🔧 Geo Plot Mapper ")



