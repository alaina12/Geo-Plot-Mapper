# app2.py - Contour-Based Plot Detection with Sequential Numbering
import streamlit as st
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
        
        print(f"✅ OCR detected {len(final_numbers)} numbers: {sorted([n for n, x, y in final_numbers])}")
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
    
    # Create visualization
    display_img = original_img.copy()
    for plot in final_plots:
        corners = plot['corners']
        pts = np.array([
            [corners['A']['x'], corners['A']['y']],
            [corners['B']['x'], corners['B']['y']],
            [corners['C']['x'], corners['C']['y']],
            [corners['D']['x'], corners['D']['y']]
        ], np.int32)
        cv2.polylines(display_img, [pts], True, (0, 255, 255), 2)
        
        # Draw plot number
        cx = sum([corners[c]['x'] for c in corners]) // 4
        cy = sum([corners[c]['y'] for c in corners]) // 4
        cv2.putText(display_img, str(plot['plot_number']),
                   (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return final_plots, display_img, thresh, detected_numbers


def calculate_geocoordinates(plots, ref_plot_id, ref_corner, ref_lat, ref_lon, px_to_ft):
    """
    Converts pixel coordinates to lat/lon.
    """
    ref_x_px, ref_y_px = None, None
    
    for plot in plots:
        if plot['plot_id'] == ref_plot_id:
            ref_x_px = plot['corners'][ref_corner]['x']
            ref_y_px = plot['corners'][ref_corner]['y']
            break
    
    if ref_x_px is None:
        st.error(f"Reference plot '{ref_plot_id}' not found!")
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
        for corner_label, coords in plot['corners'].items():
            lat = origin_lat - (coords['y'] * lat_deg_per_px)
            lon = origin_lon + (coords['x'] * lon_deg_per_px)
            
            # Check for invalid coordinates
            if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
                has_invalid_coords = True
                print(f"⚠️ Plot {plot['plot_number']}: Invalid coordinates at corner {corner_label}")
            
            new_corners[corner_label] = {"lat": lat, "lon": lon}
        
        # Always include plot, even if coordinates seem invalid (let map handle it)
        plots_with_latlon.append({
            "plot_id": plot['plot_id'],
            "plot_number": plot['plot_number'],
            "corners": new_corners
        })
        
        # Debug: Check plot 75 specifically
        if plot['plot_number'] == 75:
            print(f"🔍 Plot 75 DEBUG: plot_id={plot['plot_id']}, corners={plot['corners']}")
            print(f"   Geo corners: {new_corners}")
            print(f"   Has invalid coords: {has_invalid_coords}")
    
    # Debug: List all plot numbers in geo_plots
    plot_nums = sorted([p['plot_number'] for p in plots_with_latlon])
    print(f"📍 Geo plots created: {len(plots_with_latlon)} plots with numbers: {plot_nums}")
    plot_75_in_list = any(p['plot_number'] == 75 for p in plots_with_latlon)
    print(f"   Plot 75 in geo_plots: {plot_75_in_list}")
    
    return plots_with_latlon


# --- STREAMLIT UI ---

# Get the directory of the current script and construct path to favicon
script_dir = os.path.dirname(os.path.abspath(__file__))
favicon_path = os.path.join(os.path.dirname(script_dir), "favicon", "plot-icon.ico")

st.set_page_config(layout="wide", page_title="Geo Plot Mapper", page_icon=favicon_path)
st.title("🗺️ Geo Plot Mapper ")

# Session state
if 'plots' not in st.session_state:
    st.session_state.plots = []
if 'geo_plots' not in st.session_state:
    st.session_state.geo_plots = []
if 'detection_image' not in st.session_state:
    st.session_state.detection_image = None
if 'px_to_ft' not in st.session_state:
    st.session_state.px_to_ft = 0.5

# STEP 1: Upload
st.header("📤 Upload Layout Image")

uploaded_file = st.file_uploader("Upload plot layout", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        with st.expander("📷 Image Preview", expanded=False):
            st.image(pil_image, caption="Uploaded Image", width='stretch')
    
    with col2:
        st.info("""
        **Detection Method:**
        - Finds plot boundaries (green lines)
        - Uses OCR for plot numbers
        - Applies sequential numbering
        - Handles irregular/angled plots
        """)
        
        if st.button("🔍 Detect Plots", type="primary"):
            with st.spinner("Analyzing... Please wait 30-60 seconds"):
                plots, display_img, thresh, numbers = detect_plot_shapes_enhanced(image_bytes)
                st.session_state.plots = plots
                st.session_state.detection_image = display_img
                st.session_state.detected_numbers = numbers
                
                if plots:
                    st.success(f"🎉 Found {len(plots)} plots!")
                else:
                    st.error("No plots detected")

# SIDEBAR: Plot Information and SQLite
if st.session_state.plots:
    with st.sidebar:
        st.header("📊 Plot Information")
        
        # Plot Numbers Section
        with st.expander("📋 Plot Numbers", expanded=True):
            nums = sorted([p['plot_number'] for p in st.session_state.plots if p['plot_number'] is not None])
            if nums:
                st.write(f"**Total Plots:** {len(nums)}")
                st.write(f"**Range:** {min(nums)} - {max(nums)}")
                st.write(f"**All Numbers:**")
                # Display in a more compact format
                num_text = ', '.join(map(str, nums))
                st.text(num_text)
            else:
                st.info("No plot numbers assigned yet")
        
        # Plot Pixel Coordinates Section
        with st.expander("📍 Plot Pixel Coordinates", expanded=False):
            st.write("**Corner coordinates (in pixels):**")
            # Sort plots by plot number for easier viewing
            sorted_plots = sorted(st.session_state.plots, key=lambda p: p['plot_number'] if p['plot_number'] is not None else 9999)
            
            # Use a selectbox to choose which plot to view details
            plot_options = [f"{p['plot_id']} (Plot {p['plot_number']})" if p['plot_number'] else p['plot_id'] 
                          for p in sorted_plots]
            if plot_options:
                selected_plot_idx = st.selectbox("Select plot to view coordinates:", 
                                                range(len(plot_options)), 
                                                format_func=lambda x: plot_options[x],
                                                index=0)
                selected_plot = sorted_plots[selected_plot_idx]
                corners = selected_plot['corners']
                st.write(f"**{selected_plot['plot_id']}**")
                st.text(f"A (Bottom-Left):  x={corners['A']['x']}, y={corners['A']['y']}")
                st.text(f"B (Bottom-Right): x={corners['B']['x']}, y={corners['B']['y']}")
                st.text(f"C (Top-Right):    x={corners['C']['x']}, y={corners['C']['y']}")
                st.text(f"D (Top-Left):     x={corners['D']['x']}, y={corners['D']['y']}")
        
        # SQLite / Numbering Profiles Section
        st.divider()
        st.header("💾 Numbering Profiles")
        with st.expander("SQLite Database", expanded=True):
            st.write("Save or load numbering profiles from SQLite database.")
            profile_name = st.text_input("Profile name", value="default", key="sidebar_profile_name")

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

            if st.button("💾 Save current numbering", key="sidebar_save"):
                if not st.session_state.plots:
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
            
            if st.button("📥 Load profile and apply", key="sidebar_load"):
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
            
            if st.button("📄 List profiles", key="sidebar_list"):
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

# Main page: Detection Results
if st.session_state.detection_image is not None:
    st.divider()
    with st.expander("📊 Detection Results", expanded=False):
        st.image(st.session_state.detection_image, channels="BGR",
                caption=f"{len(st.session_state.plots)} plots detected",
                use_container_width=True)

    # STEP 1.5: Manual Edit - Plot Numbers & Coordinates
    if st.session_state.plots:
        st.divider()
        with st.expander("✏️ Edit & Fix Plot Numbers & Coordinates", expanded=True):
            st.write("Use the table below to correct plot numbers and corner coordinates (A, B, C, D). All columns except Plot ID are editable.")
            
            # Build DataFrame with nested x/y columns under A, B, C, D
            sorted_plots = sorted(st.session_state.plots, key=lambda x: (x.get('plot_number') is None, x.get('plot_number') if x.get('plot_number') is not None else 0, x.get('plot_id')))
            
            # Create DataFrame with column names that show the nested structure
            plots_df_flat = pd.DataFrame([
                {
                    "plot_id": p.get('plot_id'),
                    "plot_number": p.get('plot_number'),
                    "A_x": p.get('corners', {}).get('A', {}).get('x'),
                    "A_y": p.get('corners', {}).get('A', {}).get('y'),
                    "B_x": p.get('corners', {}).get('B', {}).get('x'),
                    "B_y": p.get('corners', {}).get('B', {}).get('y'),
                    "C_x": p.get('corners', {}).get('C', {}).get('x'),
                    "C_y": p.get('corners', {}).get('C', {}).get('y'),
                    "D_x": p.get('corners', {}).get('D', {}).get('x'),
                    "D_y": p.get('corners', {}).get('D', {}).get('y'),
                }
                for p in sorted_plots
            ])
            
            # Add visual grouping indicator
            st.info("📊 **Column Structure:** Plot Info | **A** (x, y) | **B** (x, y) | **C** (x, y) | **D** (x, y)")
            
            edited_df = st.data_editor(
                plots_df_flat,
                hide_index=True,
                column_config={
                    "plot_id": st.column_config.TextColumn("Plot ID", disabled=True),
                    "plot_number": st.column_config.NumberColumn("Plot Number", min_value=1, max_value=9999, step=1),
                    "A_x": st.column_config.NumberColumn("A → x", min_value=0, step=1, help="Corner A, x coordinate"),
                    "A_y": st.column_config.NumberColumn("A → y", min_value=0, step=1, help="Corner A, y coordinate"),
                    "B_x": st.column_config.NumberColumn("B → x", min_value=0, step=1, help="Corner B, x coordinate"),
                    "B_y": st.column_config.NumberColumn("B → y", min_value=0, step=1, help="Corner B, y coordinate"),
                    "C_x": st.column_config.NumberColumn("C → x", min_value=0, step=1, help="Corner C, x coordinate"),
                    "C_y": st.column_config.NumberColumn("C → y", min_value=0, step=1, help="Corner C, y coordinate"),
                    "D_x": st.column_config.NumberColumn("D → x", min_value=0, step=1, help="Corner D, x coordinate"),
                    "D_y": st.column_config.NumberColumn("D → y", min_value=0, step=1, help="Corner D, y coordinate"),
                },
                use_container_width=True,
                num_rows="fixed"
            )
            
            # Use edited_df directly for processing (columns are already A_x, A_y, etc.)
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
                
                with st.expander("🔍 View duplicates highlighted in red", expanded=True):
                    st.markdown("*Rows with duplicate plot numbers are highlighted in red:*")
                    st.dataframe(edited_df_processed.style.apply(highlight_duplicates, axis=1), use_container_width=True)
            else:
                st.success("✓ No duplicate plot numbers found")
            
            col_apply, col_reset = st.columns([1, 1])
            with col_apply:
                if st.button("✅ Apply Changes", type="primary"):
                    # Apply edited numbers and coordinates back to session_state.plots
                    plot_updates = {}
                    for _, row in edited_df_processed.iterrows():
                        plot_id = row['plot_id']
                        # Find original plot to preserve coordinates if needed
                        original_plot = next((p for p in st.session_state.plots if p.get('plot_id') == plot_id), None)
                        original_corners = original_plot.get('corners', {}) if original_plot else {}
                        
                        # Update plot number
                        plot_number = int(row['plot_number']) if pd.notna(row['plot_number']) else None
                        
                        # Update corners, preserving original if new value is invalid/NaN
                        def get_coord(row, coord_key, original_value):
                            """Get coordinate value, using original if new value is invalid."""
                            if coord_key in row and pd.notna(row[coord_key]):
                                try:
                                    return int(row[coord_key])
                                except (ValueError, TypeError):
                                    return original_value
                            return original_value
                        
                        corners = {}
                        for corner in ['A', 'B', 'C', 'D']:
                            orig_x = original_corners.get(corner, {}).get('x', 0)
                            orig_y = original_corners.get(corner, {}).get('y', 0)
                            corners[corner] = {
                                'x': get_coord(row, f'{corner}_x', orig_x),
                                'y': get_coord(row, f'{corner}_y', orig_y)
                            }
                        
                        plot_updates[plot_id] = {
                            'plot_number': plot_number,
                            'corners': corners
                        }
                    
                    # Apply updates
                    for p in st.session_state.plots:
                        if p.get('plot_id') in plot_updates:
                            update = plot_updates[p['plot_id']]
                            p['plot_number'] = update['plot_number']
                            p['corners'] = update['corners']
                    
                    # Reset geo_plots so user regenerates with consistent numbers
                    st.session_state.geo_plots = []
                    st.success("✅ Applied changes. Please regenerate the map in the 'Map View' tab after configuring scale.")
                    st.rerun()
            with col_reset:
                if st.button("↩️ Revert Edits (reload from detection)"):
                    st.info("Reverted UI edits. The table reflects current values from detection.")
                    st.rerun()

st.divider()

# Create tabs for better organization (like Redux state management)
# Tabs allow separation of configuration and map viewing, similar to Redux state management
tab1, tab2 = st.tabs(["🔧 Detection & Configuration", "🗺️ Map View"])

with tab1:
    # STEP 2: Configuration
    if st.session_state.plots:
        st.header("📍 Step 2: Configure Scale & Reference")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.px_to_ft = st.number_input("Feet per Pixel", 
                                                         value=st.session_state.px_to_ft, 
                                                         min_value=0.01,
                                                         max_value=10.0, 
                                                         step=0.05, 
                                                         format="%.3f")
        
        with col2:
            plot_ids = [p['plot_id'] for p in st.session_state.plots]
            ref_plot_id = st.selectbox("Reference Plot", plot_ids)
            ref_corner = st.selectbox("Corner", ["A", "B", "C", "D"])
            
            col2a, col2b = st.columns(2)
            with col2a:
                ref_lat = st.number_input("Latitude", value=13.0003, format="%.6f")
            with col2b:
                ref_lon = st.number_input("Longitude", value=77.0001, format="%.6f")
        
        if st.button("🗺️ Generate Map", type="primary"):
            with st.spinner("Calculating..."):
                st.session_state.geo_plots = calculate_geocoordinates(
                    st.session_state.plots, ref_plot_id, ref_corner,
                    ref_lat, ref_lon, st.session_state.px_to_ft
                )
                if st.session_state.geo_plots:
                    st.success(f"✅ Mapped {len(st.session_state.geo_plots)} plots! Switch to the 'Map View' tab to see the interactive map.")
    else:
        st.info("Please upload an image and detect plots first.")

with tab2:
    # STEP 3: Map View
    st.header("🗺️ Interactive Map")
    
    if not st.session_state.geo_plots:
        st.info("⚠️ Please complete the configuration steps in the 'Detection & Configuration' tab and generate the map first.")
    else:
        # Diagnostic information
        st.write(f"**Total geo_plots:** {len(st.session_state.geo_plots)}")
        
        # Collect all valid coordinates and diagnose issues
        all_lats = []
        all_lons = []
        invalid_coords_count = 0
        sample_coords = []
        
        for plot_idx, plot in enumerate(st.session_state.geo_plots[:3]):  # Check first 3 plots
            plot_coords = {}
            for corner in ['A', 'B', 'C', 'D']:
                lat = plot['corners'][corner].get('lat')
                lon = plot['corners'][corner].get('lon')
                plot_coords[corner] = {'lat': lat, 'lon': lon}
                
                # Validate coordinates
                if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        all_lats.append(lat)
                        all_lons.append(lon)
                    else:
                        invalid_coords_count += 1
                else:
                    invalid_coords_count += 1
            
            if plot_idx == 0:  # Store first plot's coordinates for display
                sample_coords = plot_coords
        
            center_lat = 13.0003
            center_lon = 77.0001
            zoom_start = 18
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
        
        plots_added = 0
        plots_failed = 0
        failed_plots = []
        
        # Render each plot as a polygon
        for plot in st.session_state.geo_plots:
            try:
                # Validate coordinates
                coords = []
                for corner in ['A', 'B', 'C', 'D']:
                    lat = plot['corners'][corner].get('lat')
                    lon = plot['corners'][corner].get('lon')
                    
                    # Check if coordinates are valid numbers
                    if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
                        raise ValueError(f"Invalid coordinates for corner {corner}: lat={lat} (type: {type(lat).__name__}), lon={lon} (type: {type(lon).__name__})")
                    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                        raise ValueError(f"Coordinates out of range: lat={lat}, lon={lon}")
                    
                    coords.append([lat, lon])  # Use list instead of tuple for Folium
                
                # Ensure we have 4 valid corners
                if len(coords) != 4:
                    raise ValueError(f"Expected 4 corners, got {len(coords)}")
                
                # Create popup HTML
                popup_html = f"""
                <div style="font-family: Arial;">
                    <h4>Plot {plot['plot_number']}</h4>
                    <hr/>
                    A: {plot['corners']['A']['lat']:.6f}, {plot['corners']['A']['lon']:.6f}<br>
                    B: {plot['corners']['B']['lat']:.6f}, {plot['corners']['B']['lon']:.6f}<br>
                    C: {plot['corners']['C']['lat']:.6f}, {plot['corners']['C']['lon']:.6f}<br>
                    D: {plot['corners']['D']['lat']:.6f}, {plot['corners']['D']['lon']:.6f}
                </div>
                """
                
                # Create tooltip HTML with coordinates
                tooltip_html = f"""
                <b>Plot {plot['plot_number']}</b><br>
                A: {plot['corners']['A']['lat']:.6f}, {plot['corners']['A']['lon']:.6f}<br>
                B: {plot['corners']['B']['lat']:.6f}, {plot['corners']['B']['lon']:.6f}<br>
                C: {plot['corners']['C']['lat']:.6f}, {plot['corners']['C']['lon']:.6f}<br>
                D: {plot['corners']['D']['lat']:.6f}, {plot['corners']['D']['lon']:.6f}
                """
                
                # Add polygon to map
                folium.Polygon(
                    locations=coords,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=tooltip_html,
                    color='#2E7D32',
                    fill=True,
                    fill_color='#81C784',
                    fill_opacity=0.5,
                    weight=2
                ).add_to(m)
                plots_added += 1
                
            except Exception as e:
                plots_failed += 1
                plot_num = plot.get('plot_number', 'Unknown')
                failed_plots.append((plot_num, str(e)))
                print(f"⚠️ Failed to render Plot {plot_num}: {e}")
        
        # Display rendering results
        if plots_added > 0:
            st.success(f"✅ Successfully rendered {plots_added} plot(s) on the map!")
        else:
            st.error(f"❌ **No plots were rendered!** All {len(st.session_state.geo_plots)} plots failed validation.")
        
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
        st_folium(m, width=1400, height=700, returned_objects=[])
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Plots", len(st.session_state.geo_plots))
        with col2:
            st.metric("Rendered", plots_added)
        with col3:
            nums = [p['plot_number'] for p in st.session_state.geo_plots if p['plot_number'] is not None]
            if nums:
                st.metric("Range", f"{min(nums)}-{max(nums)}")
            else:
                st.metric("Range", "N/A")
        with col4:
            st.metric("Scale", f"{st.session_state.px_to_ft:.3f} ft/px")
        
        # Add Geo-Coordinates Display
        with st.expander("🌍 Plot Geo-Coordinates (Latitude/Longitude)", expanded=False):
            st.write("**Geographic coordinates for each plot corner:**")
            # Sort plots by plot number for easier viewing
            sorted_geo_plots = sorted(st.session_state.geo_plots, 
                                     key=lambda p: p['plot_number'] if p['plot_number'] is not None else 9999)
            
            for plot in sorted_geo_plots:
                plot_label = f"**{plot['plot_id']}**"
                if plot['plot_number'] is None:
                    plot_label = f"**{plot['plot_id']} (Unlabeled)**"
                
                st.markdown(plot_label)
                corners = plot['corners']
                st.text(f"  A (Bottom-Left):  Lat: {corners['A']['lat']:.6f}, Lon: {corners['A']['lon']:.6f}")
                st.text(f"  B (Bottom-Right): Lat: {corners['B']['lat']:.6f}, Lon: {corners['B']['lon']:.6f}")
                st.text(f"  C (Top-Right):    Lat: {corners['C']['lat']:.6f}, Lon: {corners['C']['lon']:.6f}")
                st.text(f"  D (Top-Left):     Lat: {corners['D']['lat']:.6f}, Lon: {corners['D']['lon']:.6f}")
                st.text("")  # Add spacing between plots

st.divider()
st.caption("🔧 Geo Plot Mapper v2.2 - Contour Detection with Sequential Logic")





