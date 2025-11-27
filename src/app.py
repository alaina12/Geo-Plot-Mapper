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
import sys
import base64
import random

# Add parent directory to path to find components module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import brochure_viewer function - robust import with multiple fallbacks
import inspect
import importlib
import importlib.util

brochure_viewer = None
editable_plot_viewer = None

# Method 1: Try package import (preferred)
try:
    from components.brochure_viewer import brochure_viewer
    if inspect.isfunction(brochure_viewer):
        pass  # Success!
    elif inspect.ismodule(brochure_viewer):
        # Got module instead, try to get function from it
        brochure_viewer = getattr(brochure_viewer, 'brochure_viewer', None)
except (ImportError, AttributeError) as e:
    pass

# Method 2: If Method 1 failed, try direct module import using importlib
if not inspect.isfunction(brochure_viewer):
    try:
        mod = importlib.import_module('components.brochure_viewer.brochure_viewer')
        brochure_viewer = getattr(mod, 'brochure_viewer', None)
    except (ImportError, AttributeError) as e:
        pass

# Method 3: Last resort - try importing the module file directly
if not inspect.isfunction(brochure_viewer):
    try:
        import sys
        import os
        mod_path = os.path.join(parent_dir, 'components', 'brochure_viewer', 'brochure_viewer.py')
        if os.path.exists(mod_path):
            spec = importlib.util.spec_from_file_location("brochure_viewer_module", mod_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                brochure_viewer = getattr(mod, 'brochure_viewer', None)
    except Exception as e:
        pass

# Final verification
if not inspect.isfunction(brochure_viewer):
    raise ImportError(
        f"Failed to import brochure_viewer function. "
        f"Tried all import methods. Got: {type(brochure_viewer)}"
    )

# Import editable_plot_viewer function - robust import with multiple fallbacks
try:
    from components.editable_plot_viewer import editable_plot_viewer
    if inspect.isfunction(editable_plot_viewer):
        pass  # Success!
    elif inspect.ismodule(editable_plot_viewer):
        editable_plot_viewer = getattr(editable_plot_viewer, 'editable_plot_viewer', None)
except (ImportError, AttributeError) as e:
    pass

if not inspect.isfunction(editable_plot_viewer):
    try:
        mod = importlib.import_module('components.editable_plot_viewer.editable_plot_viewer')
        editable_plot_viewer = getattr(mod, 'editable_plot_viewer', None)
    except (ImportError, AttributeError) as e:
        pass

if not inspect.isfunction(editable_plot_viewer):
    try:
        mod_path = os.path.join(parent_dir, 'components', 'editable_plot_viewer', 'editable_plot_viewer.py')
        if os.path.exists(mod_path):
            spec = importlib.util.spec_from_file_location("editable_plot_viewer_module", mod_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                editable_plot_viewer = getattr(mod, 'editable_plot_viewer', None)
    except Exception as e:
        pass

if not inspect.isfunction(editable_plot_viewer):
    st.warning("⚠️ Failed to import editable_plot_viewer. Editing functionality may not work.")
try:
    from streamlit_drawable_canvas import st_canvas  # type: ignore
    # Patch image_to_url function early to avoid compatibility issues
    # Always override to ensure it accepts any number of arguments
    try:
        import streamlit.elements.image as st_image
        def image_to_url(image, *args, **kwargs):
            """Workaround for missing or incompatible image_to_url in newer Streamlit versions."""
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        st_image.image_to_url = image_to_url
    except:
        pass
except ImportError:
    st_canvas = None

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


def detect_background_type(image_bytes):
    """
    Detects if the uploaded image has a white background (even with plots on it)
    or a colored/non-white background.
    Returns: 'white' if white background, 'colored' if non-white background
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return 'white'  # Default to white if can't decode
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Sample pixels from edges (background is usually visible at edges)
        h, w = img_rgb.shape[:2]
        edge_samples = []
        
        # Sample top edge (wider sample for better detection)
        edge_samples.extend(img_rgb[0, :].reshape(-1, 3))
        # Sample bottom edge
        edge_samples.extend(img_rgb[h-1, :].reshape(-1, 3))
        # Sample left edge
        edge_samples.extend(img_rgb[:, 0].reshape(-1, 3))
        # Sample right edge
        edge_samples.extend(img_rgb[:, w-1].reshape(-1, 3))
        
        # Also sample a few pixels inward from edges (to avoid plot boundaries)
        if h > 20 and w > 20:
            edge_samples.extend(img_rgb[5, :].reshape(-1, 3))  # 5px from top
            edge_samples.extend(img_rgb[h-6, :].reshape(-1, 3))  # 5px from bottom
            edge_samples.extend(img_rgb[:, 5].reshape(-1, 3))  # 5px from left
            edge_samples.extend(img_rgb[:, w-6].reshape(-1, 3))  # 5px from right
        
        edge_samples = np.array(edge_samples)
        
        # Calculate average RGB values of edge pixels
        avg_r = np.mean(edge_samples[:, 0])
        avg_g = np.mean(edge_samples[:, 1])
        avg_b = np.mean(edge_samples[:, 2])
        
        # Check if average is close to white (all channels > 240)
        is_white = (avg_r > 240) and (avg_g > 240) and (avg_b > 240)
        
        # Also check variance - white backgrounds have low variance
        variance = np.var(edge_samples.flatten())
        
        # Check percentage of very bright pixels (> 250 in all channels)
        bright_pixels = np.sum((edge_samples[:, 0] > 250) & (edge_samples[:, 1] > 250) & (edge_samples[:, 2] > 250))
        bright_percentage = (bright_pixels / len(edge_samples)) * 100
        
        # If average is white AND variance is low AND most pixels are bright, it's a white background
        if is_white and variance < 500 and bright_percentage > 70:
            return 'white'
        else:
            return 'colored'
            
    except Exception as e:
        print(f"Error detecting background: {e}")
        return 'white'  # Default to white on error


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


def recalculate_coordinates_from_pixel(ref_lat, ref_lon, ref_x, ref_y, new_x, new_y, px_to_ft):
    """
    Recalculates lat/lon from pixel coordinates using the same formula as calculate_geocoordinates.
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
    
    calculated_lat = origin_lat - (new_y * lat_deg_per_px)
    calculated_lon = origin_lon + (new_x * lon_deg_per_px)
    
    return calculated_lat, calculated_lon


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


def create_brochure_canvas_image():
    """
    Builds a colored brochure-style preview image using detected plots.
    """
    if not st.session_state.plots or st.session_state.detection_image is None:
        return None

    base_img = st.session_state.detection_image.copy()
    palette = [
        (60, 155, 60),   # deep green
        (70, 90, 200),   # blue
        (50, 80, 180),   # teal/blue
        (60, 60, 200),   # purple/blue
        (40, 40, 160),   # darker blue
        (50, 140, 210),  # cyan
        (60, 170, 120),  # greenish
        (60, 70, 230),   # violet
    ]
    # Add warm overlay to mimic brochure texture
    warm_overlay = np.full_like(base_img, (50, 110, 60))
    base_img = cv2.addWeighted(base_img, 0.2, warm_overlay, 0.8, 0)

    sorted_plots = sorted(
        st.session_state.plots,
        key=lambda p: p['plot_number'] if p['plot_number'] is not None else 9999
    )

    for idx, plot in enumerate(sorted_plots):
        if not plot.get('corners'):
            continue
        corners = plot['corners']
        pts = np.array([
            [corners['A']['x'], corners['A']['y']],
            [corners['B']['x'], corners['B']['y']],
            [corners['C']['x'], corners['C']['y']],
            [corners['D']['x'], corners['D']['y']]
        ], np.int32)
        fill_color = palette[idx % len(palette)]
        cv2.fillPoly(base_img, [pts], fill_color)
        cv2.polylines(base_img, [pts], True, (255, 255, 255), 2)

    return Image.fromarray(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))


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

# Get the directory of the current script and construct path to favicon
script_dir = os.path.dirname(os.path.abspath(__file__))
favicon_path = os.path.join(os.path.dirname(script_dir), "favicon", "plot-icon.ico")

st.set_page_config(layout="wide", page_title="Geo Plot Mapper", page_icon=favicon_path)

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
if 'edits_made_in_step2' not in st.session_state:
    st.session_state.edits_made_in_step2 = False

# Workflow Sidebar - matching wireframe
with st.sidebar:
    st.header("Workflow Steps")
    steps = [
        "1 - Upload Layout Image",
        "2 - Regenerated Image",
        "3 - Detect Coordinates",
        "4 - Preview Polygons",
        "5 - Preview in Brochure",
        "6 - Configure Map Settings",
        "7 - Update Lat and Long",
        "8 - Preview in Google Map"
    ]
    
    # Check if all steps are loaded (plots exist means step 1 is done)
    steps_loaded = st.session_state.get('plots', []) != []
    
    for i, step in enumerate(steps, 1):
        if st.session_state.current_step == i:
            st.markdown(f"<p style='color: #0066CC; font-weight: bold;'>{step}</p>", unsafe_allow_html=True)
        else:
            # Make steps clickable only after steps are loaded
            if steps_loaded:
                # Create clickable step link with styled button
                step_key = f"nav_step_{i}"
                if st.button(step, key=step_key, use_container_width=True, 
                            help=f"Navigate to {step}"):
                    st.session_state.current_step = i
                    st.rerun()
            else:
                st.markdown(f"<p style='color: #666666;'>{step}</p>", unsafe_allow_html=True)
    
    st.divider()

# Main content area
st.title("🗺️ Geo Plot Mapper")

# Global CSS for Previous buttons styling
st.markdown("""
    <style>
        /* Style Previous buttons with gray background - target by key pattern and column position */
        button[key*="prev_step"]:not([kind="primary"]) {
            background-color: #6c757d !important;
            color: white !important;
            border: 1px solid #6c757d !important;
            font-weight: 500 !important;
        }
        button[key*="prev_step"]:not([kind="primary"]):hover {
            background-color: #5a6268 !important;
            border-color: #5a6268 !important;
        }
        button[key*="prev_step"]:not([kind="primary"]):focus {
            box-shadow: 0 0 0 0.2rem rgba(108, 117, 125, 0.5) !important;
        }
        /* Fallback: target buttons in second column of navigation */
        div[data-testid="column"]:nth-child(2) button:not([kind="primary"]) {
            background-color: #6c757d !important;
            color: white !important;
            border: 1px solid #6c757d !important;
            font-weight: 500 !important;
        }
        div[data-testid="column"]:nth-child(2) button:not([kind="primary"]):hover {
            background-color: #5a6268 !important;
            border-color: #5a6268 !important;
        }
        div[data-testid="column"]:nth-child(2) button:not([kind="primary"]):focus {
            box-shadow: 0 0 0 0.2rem rgba(108, 117, 125, 0.5) !important;
        }
    </style>
""", unsafe_allow_html=True)

# SIDEBAR: Only SQLite (removed coordinates - moved to main page)
if st.session_state.plots:
    with st.sidebar:
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

# PAGE FLOW BASED ON CURRENT STEP
# STEP 1: Upload Layout Image (only shows upload section, no tabs, no config)
if st.session_state.current_step == 1:
    st.header("1 - Upload Layout Image")
    
    uploaded_file = st.file_uploader("Upload plot layout", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Detect background type when image is uploaded
        if 'uploaded_image_background_type' not in st.session_state or st.session_state.get('last_uploaded_file_name') != uploaded_file.name:
            with st.spinner("🔍 Detecting background type..."):
                background_type = detect_background_type(image_bytes)
                st.session_state.uploaded_image_background_type = background_type
                st.session_state.uploaded_image_bytes = image_bytes  # Store for use in Step 5
                st.session_state.last_uploaded_file_name = uploaded_file.name
        
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            with st.expander("📷 Image Preview", expanded=False):
                st.image(pil_image, caption="Uploaded Image", width='stretch')
        
        # with col2:
        #     st.info("""
        #     **Detection Method:**
        #     - Finds plot boundaries
        #     - Uses OCR for plot numbers
        #     - Applies sequential numbering
        #     - Handles irregular/angled plots
        #     """)
            
            if st.button("🔍 Detect Plots", type="primary"):
                with st.spinner("Analyzing... Please wait 30-60 seconds"):
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
                        st.success(f"🎉 Found {len(plots)} plots!")
                    else:
                        st.error("No plots detected")
        
        # Next button for Step 1 - goes to Step 2 (Regenerated Image)
        if uploaded_file is not None and st.session_state.plots:
            col_btn1, col_btn2 = st.columns([10, 1])
            with col_btn2:
                if st.button("Next", type="primary", use_container_width=True, key="next_step1"):
                    st.session_state.current_step = 2  # Go to Step 2 (Regenerated Image)
                    st.rerun()

# STEP 2: Regenerated Image (shows detection image with red lines and dots)
elif st.session_state.current_step == 2:
    st.header("2 - Regenerated Image")
    
    if st.session_state.detection_image is not None and st.session_state.plots:
        # Store original image as base64 for the editable viewer
        if st.session_state.original_image_base64 is None:
            # Convert detection image to base64
            st.session_state.original_image_base64 = ndarray_to_data_url(st.session_state.detection_image)
        
        # Create tabs for "Plot with lines" first, then "Plot with points"
        tab1, tab2 = st.tabs(["Plot with Lines", "Plot with Points"])
        
        # Prepare plots data for the editable viewer
        def prepare_plots_for_viewer():
            plots_data = []
            for plot in st.session_state.plots:
                corners = plot.get('corners', {})
                points = []
                for corner_label in ['A', 'B', 'C', 'D']:
                    corner = corners.get(corner_label, {})
                    if corner:
                        points.append({'x': corner.get('x', 0), 'y': corner.get('y', 0)})
                
                plots_data.append({
                    'id': plot.get('plot_id', 'unknown'),
                    'plot_number': plot.get('plot_number', 0),
                    'points': points
                })
            return plots_data
        
        plots_for_viewer = prepare_plots_for_viewer()
        
        with tab1:
            st.subheader("Plot with Lines")
            st.write("**Edit the red dots to adjust plot boundaries. Lines will update automatically.**")
            if editable_plot_viewer and st.session_state.original_image_base64:
                editable_plot_viewer(
                    background_image_url=st.session_state.original_image_base64,
                    plots=plots_for_viewer,
                    mode="lines"
                )
            else:
                st.warning("⚠️ Editable viewer not available. Showing static image.")
                st.image(st.session_state.detection_image, channels="BGR",
                        caption=f"{len(st.session_state.plots)} plots detected with red lines and dots",
                        use_container_width=True)
        
        with tab2:
            st.subheader("Plot with Points (Red Dots)")
            st.write("**Edit the red dots by dragging them to correct positions.**")
            if editable_plot_viewer and st.session_state.original_image_base64:
                editable_plot_viewer(
                    background_image_url=st.session_state.original_image_base64,
                    plots=plots_for_viewer,
                    mode="points"
                )
            else:
                st.warning("⚠️ Editable viewer not available. Showing static image.")
                st.image(st.session_state.detection_image, channels="BGR",
                        caption=f"{len(st.session_state.plots)} plots detected with red lines and dots",
                        use_container_width=True)
        
        # Button to apply changes from the editable viewer
        st.divider()
        
        # Show instruction
        st.info("💡 **How to use:** Edit plots in the viewer above, then click 'Save Changes'. The yellow box will appear with coordinates - copy and paste them below.")
        
        col_apply, col_detect = st.columns(2)
        with col_apply:
            coord_json = st.text_area("Paste coordinates JSON here:", 
                                     height=150, key="coord_json_input_step2",
                                     placeholder='Click "Save Changes" in the viewer above, then copy the JSON from the yellow box and paste it here.',
                                     help="After editing plots, click 'Save Changes' to get the JSON coordinates")
            
            if st.button("✅ Apply Coordinates", type="primary", use_container_width=True):
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
                            existing_plot_ids = {p.get('plot_id') for p in st.session_state.plots}
                            
                            updated_count = 0
                            added_count = 0
                            deleted_count = 0
                            
                            # Update existing plots
                            for plot in st.session_state.plots[:]:  # Use slice to allow modification
                                plot_id = plot.get('plot_id')
                                if plot_id in plot_dict:
                                    updated_plot = plot_dict[plot_id]
                                    points = updated_plot.get('points', [])
                                    
                                    if len(points) >= 3:  # Allow 3+ points for any shape
                                        # Update corners - ensure proper order (A, B, C, D)
                                        # Handle plots with 3 or more points
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
                                        updated_count += 1
                            
                            # Remove deleted plots (plots that exist in session but not in saved data)
                            plots_to_remove = existing_plot_ids - updated_plot_ids
                            if plots_to_remove:
                                st.session_state.plots = [p for p in st.session_state.plots 
                                                         if p.get('plot_id') not in plots_to_remove]
                                deleted_count = len(plots_to_remove)
                            
                            # Add new plots (plots that exist in saved data but not in session)
                            plots_to_add = updated_plot_ids - existing_plot_ids
                            for plot_id in plots_to_add:
                                new_plot_data = plot_dict[plot_id]
                                points = new_plot_data.get('points', [])
                                
                                if len(points) >= 3:
                                    corners = {}
                                    corner_labels = ['A', 'B', 'C', 'D']
                                    for i, label in enumerate(corner_labels):
                                        if i < len(points):
                                            corners[label] = {'x': int(points[i]['x']), 'y': int(points[i]['y'])}
                                        else:
                                            last_point = points[-1]
                                            corners[label] = {'x': int(last_point['x']), 'y': int(last_point['y'])}
                                    
                                    new_plot = {
                                        'plot_id': new_plot_data.get('id', plot_id),
                                        'plot_number': new_plot_data.get('plot_number', len(st.session_state.plots) + 1),
                                        'corners': corners
                                    }
                                    st.session_state.plots.append(new_plot)
                                    added_count += 1
                            
                            if updated_count > 0 or added_count > 0 or deleted_count > 0:
                                # Regenerate image immediately
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
                                        # Draw plot number - use current plot number from session state
                                        plot_number = plot.get('plot_number')
                                        if plot_number is not None:
                                            cx = sum([corners[c]['x'] for c in corners]) // 4
                                            cy = sum([corners[c]['y'] for c in corners]) // 4
                                            cv2.putText(display_img, str(plot_number),
                                                       (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                    
                                    st.session_state.detection_image = display_img
                                    st.session_state.detected_overlay_url = ndarray_to_data_url(display_img)
                                    st.session_state.original_image_base64 = ndarray_to_data_url(display_img)
                                    st.session_state.brochure_overlay_url = st.session_state.detected_overlay_url
                                
                                # Create success message with all changes
                                msg_parts = []
                                if updated_count > 0:
                                    msg_parts.append(f"updated {updated_count}")
                                if added_count > 0:
                                    msg_parts.append(f"added {added_count}")
                                if deleted_count > 0:
                                    msg_parts.append(f"deleted {deleted_count}")
                                
                                msg = f"✅ Applied changes: {', '.join(msg_parts)} plot(s)! The image has been regenerated."
                                st.success(msg)
                                st.session_state.coordinates_detected = True
                                st.session_state.edits_made_in_step2 = True
                                st.rerun()
                            else:
                                st.warning("No matching plots found to update. Make sure the plot IDs match.")
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON format: {e}")
                    except Exception as e:
                        st.error(f"Error applying coordinates: {e}")
                        st.exception(e)
                else:
                    st.warning("⚠️ Please paste the coordinates JSON first. Click 'Save Changes' in the viewer to get it.")
        
        with col_detect:
            if st.button("🔍 Detect Coordinates Using AI", use_container_width=True, 
                        help="Click to mark coordinates as ready and proceed to next step"):
                st.session_state.coordinates_detected = True
                st.success("✅ Coordinates marked as ready! You can proceed to the next step.")
        
        # Show success message if coordinates are detected
        if st.session_state.get('coordinates_detected', False):
            st.success("✅ Coordinates ready! You can proceed to the next step.")
        
        # Navigation buttons for Step 2
        col_btn1, col_btn2, col_btn3 = st.columns([6, 2, 2])
        with col_btn2:
            if st.button("Previous", use_container_width=True, key="prev_step2"):
                st.session_state.current_step = 1  # Go back to Step 1
                st.rerun()
        with col_btn3:
            if st.button("Next", type="primary", use_container_width=True, key="next_step2"):
                if st.session_state.get('coordinates_detected', False):
                    # If edits were made, try to apply them
                    if st.session_state.get('edits_made_in_step2', False):
                        # Note: We can't directly read localStorage from Python
                        # The user should use Step 3's grid to make precise edits
                        # Or we could add a JavaScript component to read and send the data
                        st.info("💡 Tip: Use Step 3's grid table to make precise coordinate adjustments if needed.")
                    st.session_state.current_step = 3  # Go to Step 3 (Detect Coordinates)
                    st.rerun()
                else:
                    st.warning("⚠️ Please ensure coordinates are detected first!")
    else:
        st.info("Please upload and detect plots first in Step 1.")

# STEP 3: Detect Coordinates (shows editable grid table)
elif st.session_state.current_step == 3:
    st.header("3 - Detected Coordinates")

    if st.session_state.plots:
        st.subheader("✏️ Edit Plot Numbers & Coordinates")
        st.write("**Note:** Editable Data Grid with Runtime Save option")
        st.write("Use the table below to correct plot numbers and corner coordinates (A, B, C, D). All columns except Plot ID are editable.")
        
        # Build DataFrame for editable table
        sorted_plots = sorted(st.session_state.plots, key=lambda x: (x.get('plot_number') is None, x.get('plot_number') if x.get('plot_number') is not None else 0, x.get('plot_id')))
        
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
                
                # Regenerate detection image with updated coordinates
                if st.session_state.detection_image is not None:
                    original_img = st.session_state.detection_image.copy()
                    # Redraw the image with updated coordinates
                    display_img = original_img.copy()
                    
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
                        # Red lines for plot boundaries
                        cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
                        
                        # Draw red dots at each corner
                        cv2.circle(display_img, (corners['A']['x'], corners['A']['y']), 4, (0, 0, 255), -1)
                        cv2.circle(display_img, (corners['B']['x'], corners['B']['y']), 4, (0, 0, 255), -1)
                        cv2.circle(display_img, (corners['C']['x'], corners['C']['y']), 4, (0, 0, 255), -1)
                        cv2.circle(display_img, (corners['D']['x'], corners['D']['y']), 4, (0, 0, 255), -1)
                        
                        # Draw plot number
                        cx = sum([corners[c]['x'] for c in corners]) // 4
                        cy = sum([corners[c]['y'] for c in corners]) // 4
                        cv2.putText(display_img, str(plot['plot_number']),
                                   (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    st.session_state.detection_image = display_img
                    st.session_state.detected_overlay_url = ndarray_to_data_url(display_img)
                    st.session_state.original_image_base64 = ndarray_to_data_url(display_img)
                
                # Reset geo_plots so user regenerates with consistent numbers
                st.session_state.geo_plots = []
                st.success("✅ Applied changes. The image will be updated in Step 4.")
                st.rerun()
        with col_reset:
            if st.button("↩️ Revert Edits (reload from detection)"):
                st.info("Reverted UI edits. The table reflects current values from detection.")
                st.rerun()
        
        # Navigation buttons for Step 3
        col_btn1, col_btn2, col_btn3 = st.columns([6, 2, 2])
        with col_btn2:
            if st.button("Previous", use_container_width=True, key="prev_step3"):
                st.session_state.current_step = 2  # Go back to Step 2
                st.rerun()
        with col_btn3:
            if st.button("Next", type="primary", use_container_width=True, key="next_step3"):
                st.session_state.current_step = 4  # Go to Step 4 (Preview Polygons)
                st.rerun()
    else:
        st.info("Please upload an image and detect plots first.")

# STEP 4.5: Edit Coordinates (editable grid page)
elif st.session_state.current_step == 4.5:
    st.header("4.5 - Edit Coordinates")

    if st.session_state.plots:
        st.subheader("✏️ Edit Plot Numbers & Coordinates")
        st.write("**Note:** Editable Data Grid with Runtime Save option")
        st.write("Use the table below to correct plot numbers and corner coordinates (A, B, C, D). All columns except Plot ID are editable.")
        
        # Build DataFrame for editable table
        sorted_plots = sorted(st.session_state.plots, key=lambda x: (x.get('plot_number') is None, x.get('plot_number') if x.get('plot_number') is not None else 0, x.get('plot_id')))
        
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
                
                # Regenerate detection image with updated coordinates
                if st.session_state.detection_image is not None:
                    original_img = st.session_state.detection_image.copy()
                    # Create a white background with the same dimensions
                    if len(original_img.shape) == 3:
                        height, width = original_img.shape[:2]
                        display_img = np.ones((height, width, 3), dtype=np.uint8) * 255
                    else:
                        height, width = original_img.shape
                        display_img = np.ones((height, width, 3), dtype=np.uint8) * 255
                    
                    # Redraw all plots with updated coordinates
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
                        # Red lines for plot boundaries
                        cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
                        
                        # Draw red dots at each corner
                        cv2.circle(display_img, (corners['A']['x'], corners['A']['y']), 4, (0, 0, 255), -1)
                        cv2.circle(display_img, (corners['B']['x'], corners['B']['y']), 4, (0, 0, 255), -1)
                        cv2.circle(display_img, (corners['C']['x'], corners['C']['y']), 4, (0, 0, 255), -1)
                        cv2.circle(display_img, (corners['D']['x'], corners['D']['y']), 4, (0, 0, 255), -1)
                        
                        # Draw plot number - use current plot number from session state
                        plot_number = plot.get('plot_number')
                        if plot_number is not None:
                            cx = sum([corners[c]['x'] for c in corners]) // 4
                            cy = sum([corners[c]['y'] for c in corners]) // 4
                            cv2.putText(display_img, str(plot_number),
                                       (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    st.session_state.detection_image = display_img
                    st.session_state.detected_overlay_url = ndarray_to_data_url(display_img)
                    st.session_state.original_image_base64 = ndarray_to_data_url(display_img)
                    st.session_state.brochure_overlay_url = st.session_state.detected_overlay_url
                
                # Reset geo_plots so user regenerates with consistent numbers
                st.session_state.geo_plots = []
                st.success("✅ Applied changes. The image has been regenerated and will be visible in Step 4.")
                st.rerun()
        with col_reset:
            if st.button("↩️ Revert Edits (reload from detection)"):
                st.info("Reverted UI edits. The table reflects current values from detection.")
                st.rerun()
        
        # Navigation buttons for Step 4.5
        col_btn1, col_btn2, col_btn3 = st.columns([6, 2, 2])
        with col_btn2:
            if st.button("Previous", use_container_width=True, key="prev_step4_5"):
                st.session_state.current_step = 4  # Go back to Step 4
                st.rerun()
        with col_btn3:
            if st.button("Next", type="primary", use_container_width=True, key="next_step4_5"):
                st.session_state.current_step = 5  # Go to Step 5 (Preview in Brochure)
                st.rerun()
    else:
        st.info("Please detect plots first.")

# STEP 5: Preview in Brochure
elif st.session_state.current_step == 5:
    st.header("5 - Preview in Brochure")
    
    if st.session_state.plots:
        st.write("**Interactive brochure preview with plot status controls:**")

        # Initialize plot statuses if not set
        for plot in st.session_state.plots:
            plot_id = plot.get('plot_id')
            if plot_id and plot_id not in st.session_state.plot_statuses:
                st.session_state.plot_statuses[plot_id] = random.choice(PLOT_STATUS_OPTIONS)
        
        # Determine which background to use based on detection in Step 1
        background_image_url = ""
        background_type = st.session_state.get('uploaded_image_background_type', 'white')
        
        if background_type == 'white':
            # Use default brochure background
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            background_image_path = os.path.join(project_root, "assets", "brochure_bg.png")
            
            if os.path.exists(background_image_path):
                try:
                    bg_image = Image.open(background_image_path).convert("RGB")
                    background_image_url = pil_image_to_base64(bg_image)
                    # st.info("📄 Using default brochure background (white background detected in uploaded image).")
                except Exception as e:
                    st.error(f"Failed to load background image: {e}")
            else:
                st.warning(f"Background image not found at: {background_image_path}")
        else:
            # Use uploaded image as background
            uploaded_image_bytes = st.session_state.get('uploaded_image_bytes')
            if uploaded_image_bytes:
                try:
                    uploaded_pil_image = Image.open(BytesIO(uploaded_image_bytes)).convert("RGB")
                    background_image_url = pil_image_to_base64(uploaded_pil_image)
                    # st.success("🎨 Using uploaded image as brochure background (colored background detected).")
                except Exception as e:
                    st.error(f"Failed to load uploaded image as background: {e}")
                    # Fallback to default
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(script_dir)
                    background_image_path = os.path.join(project_root, "assets", "brochure_bg.png")
                    if os.path.exists(background_image_path):
                        bg_image = Image.open(background_image_path).convert("RGB")
                        background_image_url = pil_image_to_base64(bg_image)
            else:
                st.warning("⚠️ Uploaded image not found. Using default background.")
                # Fallback to default
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                background_image_path = os.path.join(project_root, "assets", "brochure_bg.png")
                if os.path.exists(background_image_path):
                    bg_image = Image.open(background_image_path).convert("RGB")
                    background_image_url = pil_image_to_base64(bg_image)
        
        # Use detected overlay only for colored backgrounds (to overlay on existing plots)
        plot_overlay_url = None
        if background_type == 'colored':
            # For colored backgrounds, overlay detected plots on top of existing plots
            plot_overlay_url = st.session_state.get('detected_overlay_url', "")
            if not plot_overlay_url:
                st.warning("⚠️ No detected overlay available. Please run detection in Step 1 first.")
        # For white backgrounds, we don't need overlay (colored plots are enough)
        
        # Plot Status Controls
        st.subheader("Plot Status Controls")
        st.caption("Statuses are auto-assigned. Click below to randomize them anytime.")
        
        col_status_btn, col_status_summary = st.columns([1.5, 2])
        with col_status_btn:
            if st.button("🔀 Randomize plot statuses", key="randomize_statuses"):
                for plot_id in st.session_state.plot_statuses:
                    st.session_state.plot_statuses[plot_id] = random.choice(PLOT_STATUS_OPTIONS)
                st.success("Plot statuses randomized.")
        
        with col_status_summary:
            status_counts = {status: 0 for status in PLOT_STATUS_OPTIONS}
            for status in st.session_state.plot_statuses.values():
                if status in status_counts:
                    status_counts[status] += 1
            for status, count in status_counts.items():
                st.write(f"- `{status}`: {count}")
        
        # Build plot payload with current statuses
        def build_plot_payload(plot):
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
            return {
                "id": plot.get('plot_id', 'unknown'),
                "points": points,
                "lat": geo_lat,
                "lon": geo_lon,
                "status": st.session_state.plot_statuses.get(plot.get('plot_id'), "available")
            }
        
        plots_payload = [build_plot_payload(plot) for plot in st.session_state.plots if plot.get('corners')]
        
        # Auto-render interactive Fabric Canvas
        if background_image_url and plots_payload:
            brochure_viewer(
                background_image_url=background_image_url,
                plots=plots_payload,
                plot_overlay_url=plot_overlay_url if plot_overlay_url else None
            )
            
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
                                        
                                        // Store original points
                                        plotPolygons.forEach(polygon => {
                                            if (!polygon.originalPoints) {
                                                polygon.originalPoints = polygon.points.map(p => ({x: p.x, y: p.y}));
                                                polygon.originalLeft = polygon.left;
                                                polygon.originalTop = polygon.top;
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
                                if (!polygon.originalPoints) {
                                    polygon.originalPoints = polygon.points.map(p => ({x: p.x, y: p.y}));
                                    polygon.originalLeft = polygon.left;
                                    polygon.originalTop = polygon.top;
                                }
                                
                                const origCenterX = polygon.originalPoints.reduce((sum, p) => sum + p.x, 0) / polygon.originalPoints.length;
                                const origCenterY = polygon.originalPoints.reduce((sum, p) => sum + p.y, 0) / polygon.originalPoints.length;
                                
                                const transformedPoints = polygon.originalPoints.map(p => {
                                    let x = p.x - origCenterX;
                                    let y = p.y - origCenterY;
                                    
                                    x *= scale;
                                    y *= scale;
                                    
                                    const angleRad = (rotation * Math.PI) / 180;
                                    const cos = Math.cos(angleRad);
                                    const sin = Math.sin(angleRad);
                                    const newX = x * cos - y * sin;
                                    const newY = x * sin + y * cos;
                                    
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
                        const moveUpBtn = document.getElementById('brochure-move-up');
                        const moveDownBtn = document.getElementById('brochure-move-down');
                        const moveLeftBtn = document.getElementById('brochure-move-left');
                        const moveRightBtn = document.getElementById('brochure-move-right');
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
                        
                        if (scaleUpBtn) {
                            scaleUpBtn.onclick = function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                console.log('Scale up clicked!');
                                try {
                                    const currentScale = iframeWindow.currentScale || 1.0;
                                    iframeWindow.currentScale = currentScale * 1.1;
                                    console.log('New scale:', iframeWindow.currentScale);
                                    applyTransform();
                                } catch(err) {
                                    console.error('Error in scale up:', err);
                                }
                                return false;
                            };
                            console.log('Scale up button connected');
                        } else {
                            console.log('Scale up button NOT found!');
                        }
                        
                        if (scaleDownBtn) {
                            scaleDownBtn.onclick = function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                console.log('Scale down clicked!');
                                try {
                                    const currentScale = iframeWindow.currentScale || 1.0;
                                    iframeWindow.currentScale = currentScale / 1.1;
                                    console.log('New scale:', iframeWindow.currentScale);
                                    applyTransform();
                                } catch(err) {
                                    console.error('Error in scale down:', err);
                                }
                                return false;
                            };
                            console.log('Scale down button connected');
                        } else {
                            console.log('Scale down button NOT found!');
                        }
                        
                        if (moveUpBtn) {
                            moveUpBtn.onclick = function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                console.log('Move up clicked!');
                                try {
                                    iframeWindow.currentOffsetY = (iframeWindow.currentOffsetY || 0) - 10;
                                    console.log('New offsetY:', iframeWindow.currentOffsetY);
                                    applyTransform();
                                } catch(err) {
                                    console.error('Error in move up:', err);
                                }
                                return false;
                            };
                            console.log('Move up button connected');
                        }
                        
                        if (moveDownBtn) {
                            moveDownBtn.onclick = function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                console.log('Move down clicked!');
                                try {
                                    iframeWindow.currentOffsetY = (iframeWindow.currentOffsetY || 0) + 10;
                                    console.log('New offsetY:', iframeWindow.currentOffsetY);
                                    applyTransform();
                                } catch(err) {
                                    console.error('Error in move down:', err);
                                }
                                return false;
                            };
                            console.log('Move down button connected');
                        } else {
                            console.log('Move down button NOT found!');
                        }
                        
                        if (moveLeftBtn) {
                            moveLeftBtn.onclick = function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                console.log('Move left clicked!');
                                try {
                                    iframeWindow.currentOffsetX = (iframeWindow.currentOffsetX || 0) - 10;
                                    console.log('New offsetX:', iframeWindow.currentOffsetX);
                                    applyTransform();
                                } catch(err) {
                                    console.error('Error in move left:', err);
                                }
                                return false;
                            };
                            console.log('Move left button connected');
                        }
                        
                        if (moveRightBtn) {
                            moveRightBtn.onclick = function(e) {
                                e.preventDefault();
                                e.stopPropagation();
                                console.log('Move right clicked!');
                                try {
                                    iframeWindow.currentOffsetX = (iframeWindow.currentOffsetX || 0) + 10;
                                    console.log('New offsetX:', iframeWindow.currentOffsetX);
                                    applyTransform();
                                } catch(err) {
                                    console.error('Error in move right:', err);
                                }
                                return false;
                            };
                            console.log('Move right button connected');
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
            st.error("❌ Background image not found. Please ensure assets/brochure_bg.png exists.")
        elif not plots_payload:
            st.warning("⚠️ No plots with valid coordinates found.")
        
        # Navigation buttons for Step 5
        col_btn1, col_btn2, col_btn3 = st.columns([6, 2, 2])
        with col_btn1:
            st.write("")  # Spacer for alignment
        with col_btn2:
            if st.button("Previous", use_container_width=True, key="prev_step5"):
                st.session_state.current_step = 4  # Go back to Step 4
                st.rerun()
        with col_btn3:
            if st.button("Next", type="primary", use_container_width=True, key="next_step5"):
                st.session_state.current_step = 6  # Go to Step 6 (Configure Map Settings)
                st.rerun()
    else:
        st.info("Please complete previous steps first.")

# STEP 6: Configure Map Settings (moved from Step 5)
elif st.session_state.current_step == 6:
    st.header("6 - Configure Map Settings")
    
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
        
        if st.button("🗺️ Generate Map", type="primary", key="generate_map_config"):
            with st.spinner("Calculating..."):
                st.session_state.geo_plots = calculate_geocoordinates(
                    st.session_state.plots, ref_plot_id, ref_corner,
                    ref_lat, ref_lon, st.session_state.px_to_ft
                )
                if st.session_state.geo_plots:
                    st.success(f"✅ Generated map for {len(st.session_state.geo_plots)} plots!")
        
        # Navigation buttons for Step 6
        if st.session_state.geo_plots:
            col_btn1, col_btn2, col_btn3 = st.columns([6, 2, 2])
            with col_btn2:
                if st.button("Previous", use_container_width=True, key="prev_step6"):
                    st.session_state.current_step = 5  # Go back to Step 5
                    st.rerun()
            with col_btn3:
                if st.button("Next", type="primary", use_container_width=True, key="next_step6"):
                    st.session_state.current_step = 7  # Go to Step 7 (Update Lat and Long)
                    st.rerun()
    else:
        st.info("Please complete previous steps first.")

# STEP 4: Preview Polygons (shows regenerated image based on edits)
elif st.session_state.current_step == 4:
    st.header("4 - Preview Polygons")

    if st.session_state.plots:
        st.subheader("📸 Regenerated Image Preview")
        st.write("**This preview shows the updated image based on any edits made in previous steps.**")
        
        # Button to manually regenerate image from current plot coordinates
        if st.button("🔄 Regenerate Image", type="primary", use_container_width=False):
            st.rerun()
        
        # Always regenerate the image to ensure it's up to date with current plot data
        if st.session_state.detection_image is not None:
            # Get the original image (before detection) if available, otherwise use detection_image
            # For now, we'll regenerate from the current plots
            original_img = st.session_state.detection_image.copy()
            
            # Get dimensions from the original image
            if len(original_img.shape) == 3:
                height, width = original_img.shape[:2]
            else:
                height, width = original_img.shape
            
            # Create a white background with the same dimensions
            display_img = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Redraw all plots with current coordinates and updated plot numbers
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
                # Red lines for plot boundaries (BGR format: red = (0, 0, 255))
                cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
                
                # Draw red dots at each corner
                cv2.circle(display_img, (corners['A']['x'], corners['A']['y']), 4, (0, 0, 255), -1)
                cv2.circle(display_img, (corners['B']['x'], corners['B']['y']), 4, (0, 0, 255), -1)
                cv2.circle(display_img, (corners['C']['x'], corners['C']['y']), 4, (0, 0, 255), -1)
                cv2.circle(display_img, (corners['D']['x'], corners['D']['y']), 4, (0, 0, 255), -1)
                
                # Draw plot number in black - use current plot number from session state
                plot_number = plot.get('plot_number')
                if plot_number is not None:
                    cx = sum([corners[c]['x'] for c in corners]) // 4
                    cy = sum([corners[c]['y'] for c in corners]) // 4
                    cv2.putText(display_img, str(plot_number),
                               (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Update session state with regenerated image
            st.session_state.detection_image = display_img
            st.session_state.detected_overlay_url = ndarray_to_data_url(display_img)
            st.session_state.brochure_overlay_url = st.session_state.detected_overlay_url
            
            # Display the regenerated image
            st.image(display_img, channels="BGR",
                    caption=f"{len(st.session_state.plots)} plots with updated coordinates (red lines and dots)",
                    use_container_width=True)
        else:
            st.warning("⚠️ No detection image available. Please go back to Step 1 and detect plots.")
        
        # Navigation buttons for Step 4
        col_btn1, col_btn2, col_btn3 = st.columns([6, 2, 2])
        with col_btn2:
            if st.button("Previous", use_container_width=True, key="prev_step4"):
                st.session_state.current_step = 3  # Go back to Step 3
                st.rerun()
        with col_btn3:
            if st.button("Next", type="primary", use_container_width=True, key="next_step4"):
                st.session_state.current_step = 5  # Go to Step 5 (Preview in Brochure)
                st.rerun()
    else:
        st.info("Please detect plots first.")

# STEP 7: Update Lat and Long (update geo coordinates)
elif st.session_state.current_step == 7:
    st.header("7 - Update Lat and Long")
    
    if st.session_state.plots:
        if not st.session_state.geo_plots:
            st.warning("⚠️ Please configure map settings in Step 6 (Configure Map Settings) first.")
        else:
            # Create table matching wireframe: First row shows Coordinates, second row shows Lat Long
            sorted_geo_plots = sorted(st.session_state.geo_plots, 
                                     key=lambda p: p['plot_number'] if p['plot_number'] is not None else 9999)
            sorted_plots = sorted(st.session_state.plots, 
                                 key=lambda p: p['plot_number'] if p['plot_number'] is not None else 9999)
            
            # Create a mapping of plot_number to pixel coordinates
            plot_coords_map = {p.get('plot_number'): p.get('corners', {}) for p in sorted_plots}
            
            table_data = []
            for idx, plot in enumerate(sorted_geo_plots):
                plot_num = plot['plot_number']
                plot_id = f"P-{plot_num:02d}" if plot_num else "Unknown"
                pixel_corners = plot_coords_map.get(plot_num, {})
                
                # First row: Coordinates (X,Y)
                table_data.append({
                    "Sl. No": idx + 1,
                    "Plot No.": plot_id,
                    "Value": "Coordinates",
                    "Side A (X,Y)": f"({pixel_corners.get('A', {}).get('x', 0)}, {pixel_corners.get('A', {}).get('y', 0)})" if pixel_corners.get('A') else "-",
                    "Side B (X,Y)": f"({pixel_corners.get('B', {}).get('x', 0)}, {pixel_corners.get('B', {}).get('y', 0)})" if pixel_corners.get('B') else "-",
                    "Side C (X,Y)": f"({pixel_corners.get('C', {}).get('x', 0)}, {pixel_corners.get('C', {}).get('y', 0)})" if pixel_corners.get('C') else "-",
                    "Side D (X,Y)": f"({pixel_corners.get('D', {}).get('x', 0)}, {pixel_corners.get('D', {}).get('y', 0)})" if pixel_corners.get('D') else "-",
                    "Side E (X,Y)": "-",  # Most plots don't have Side E
                })
                
                # Second row: Lat Long
                table_data.append({
                    "Sl. No": "",
                    "Plot No.": plot_id,
                    "Value": "Lat Long",
                    "Side A (X,Y)": f"({plot['corners']['A']['lat']:.6f}, {plot['corners']['A']['lon']:.6f})",
                    "Side B (X,Y)": f"({plot['corners']['B']['lat']:.6f}, {plot['corners']['B']['lon']:.6f})",
                    "Side C (X,Y)": f"({plot['corners']['C']['lat']:.6f}, {plot['corners']['C']['lon']:.6f})",
                    "Side D (X,Y)": f"({plot['corners']['D']['lat']:.6f}, {plot['corners']['D']['lon']:.6f})",
                    "Side E (X,Y)": "-",
                })
            
            # Create editable DataFrame with separate columns for lat/lon
            editable_data = []
            for idx, plot in enumerate(sorted_geo_plots):
                plot_num = plot['plot_number']
                plot_id = f"P-{plot_num:02d}" if plot_num else "Unknown"
                pixel_corners = plot_coords_map.get(plot_num, {})
                
                editable_data.append({
                    "Sl. No": idx + 1,
                    "Plot No.": plot_id,
                    "A_x": pixel_corners.get('A', {}).get('x', 0) if pixel_corners.get('A') else 0,
                    "A_y": pixel_corners.get('A', {}).get('y', 0) if pixel_corners.get('A') else 0,
                    "A_lat": plot['corners']['A']['lat'],
                    "A_lon": plot['corners']['A']['lon'],
                    "B_x": pixel_corners.get('B', {}).get('x', 0) if pixel_corners.get('B') else 0,
                    "B_y": pixel_corners.get('B', {}).get('y', 0) if pixel_corners.get('B') else 0,
                    "B_lat": plot['corners']['B']['lat'],
                    "B_lon": plot['corners']['B']['lon'],
                    "C_x": pixel_corners.get('C', {}).get('x', 0) if pixel_corners.get('C') else 0,
                    "C_y": pixel_corners.get('C', {}).get('y', 0) if pixel_corners.get('C') else 0,
                    "C_lat": plot['corners']['C']['lat'],
                    "C_lon": plot['corners']['C']['lon'],
                    "D_x": pixel_corners.get('D', {}).get('x', 0) if pixel_corners.get('D') else 0,
                    "D_y": pixel_corners.get('D', {}).get('y', 0) if pixel_corners.get('D') else 0,
                    "D_lat": plot['corners']['D']['lat'],
                    "D_lon": plot['corners']['D']['lon'],
                })
            
            editable_df = pd.DataFrame(editable_data)
            
            # Initialize grid refresh counter
            if 'latlon_grid_refresh' not in st.session_state:
                st.session_state.latlon_grid_refresh = 0
            
            # Display editable table
            edited_df = st.data_editor(
                editable_df,
                hide_index=True,
                key=f"latlon_editor_{st.session_state.latlon_grid_refresh}",
                column_config={
                    "Sl. No": st.column_config.NumberColumn("Sl. No", disabled=True),
                    "Plot No.": st.column_config.TextColumn("Plot No.", disabled=True),
                    "A_x": st.column_config.NumberColumn("A → x", min_value=0, step=1, help="Corner A, x coordinate"),
                    "A_y": st.column_config.NumberColumn("A → y", min_value=0, step=1, help="Corner A, y coordinate"),
                    "A_lat": st.column_config.NumberColumn("A → lat", min_value=-90.0, max_value=90.0, step=0.000001, format="%.6f", help="Corner A, latitude"),
                    "A_lon": st.column_config.NumberColumn("A → lon", min_value=-180.0, max_value=180.0, step=0.000001, format="%.6f", help="Corner A, longitude"),
                    "B_x": st.column_config.NumberColumn("B → x", min_value=0, step=1, help="Corner B, x coordinate"),
                    "B_y": st.column_config.NumberColumn("B → y", min_value=0, step=1, help="Corner B, y coordinate"),
                    "B_lat": st.column_config.NumberColumn("B → lat", min_value=-90.0, max_value=90.0, step=0.000001, format="%.6f", help="Corner B, latitude"),
                    "B_lon": st.column_config.NumberColumn("B → lon", min_value=-180.0, max_value=180.0, step=0.000001, format="%.6f", help="Corner B, longitude"),
                    "C_x": st.column_config.NumberColumn("C → x", min_value=0, step=1, help="Corner C, x coordinate"),
                    "C_y": st.column_config.NumberColumn("C → y", min_value=0, step=1, help="Corner C, y coordinate"),
                    "C_lat": st.column_config.NumberColumn("C → lat", min_value=-90.0, max_value=90.0, step=0.000001, format="%.6f", help="Corner C, latitude"),
                    "C_lon": st.column_config.NumberColumn("C → lon", min_value=-180.0, max_value=180.0, step=0.000001, format="%.6f", help="Corner C, longitude"),
                    "D_x": st.column_config.NumberColumn("D → x", min_value=0, step=1, help="Corner D, x coordinate"),
                    "D_y": st.column_config.NumberColumn("D → y", min_value=0, step=1, help="Corner D, y coordinate"),
                    "D_lat": st.column_config.NumberColumn("D → lat", min_value=-90.0, max_value=90.0, step=0.000001, format="%.6f", help="Corner D, latitude"),
                    "D_lon": st.column_config.NumberColumn("D → lon", min_value=-180.0, max_value=180.0, step=0.000001, format="%.6f", help="Corner D, longitude"),
                },
                use_container_width=True,
                num_rows="fixed"
            )
            
            # Apply changes button
            if st.button("💾 Apply Changes", type="primary", use_container_width=False):
                changes_made = False
                updated_count = 0
                any_plot_changed = False
                
                # Check if reference plot settings are available (outside loop)
                has_ref_settings = ('px_to_ft' in st.session_state and 
                                  'ref_plot_id' in st.session_state and
                                  st.session_state.ref_plot_id)
                
                # First pass: Update the plots that were edited
                for idx, row in edited_df.iterrows():
                    plot_num = sorted_geo_plots[idx]['plot_number']
                    geo_plot = next((gp for gp in st.session_state.geo_plots if gp['plot_number'] == plot_num), None)
                    pixel_plot = next((p for p in st.session_state.plots if p.get('plot_number') == plot_num), None)
                    
                    if geo_plot and pixel_plot:
                        # Store original values to detect changes
                        original_pixel_corners = pixel_plot.get('corners', {}).copy()
                        original_geo_corners = geo_plot.get('corners', {}).copy()
                        
                        if has_ref_settings:
                            ref_plot = next((p for p in st.session_state.plots if p.get('plot_id') == st.session_state.ref_plot_id), None)
                            ref_geo_plot = next((gp for gp in st.session_state.geo_plots if gp.get('plot_id') == st.session_state.ref_plot_id), None)
                            
                            if ref_plot and ref_plot.get('corners') and ref_geo_plot and ref_geo_plot.get('corners'):
                                ref_corner = st.session_state.get('ref_corner', 'A')
                                # Get pixel coordinates from pixel plot
                                ref_pixel_corner = ref_plot['corners'].get(ref_corner, {})
                                ref_x = ref_pixel_corner.get('x')
                                ref_y = ref_pixel_corner.get('y')
                                # Get lat/lon from geo plot (this is the correct source)
                                ref_geo_corner = ref_geo_plot['corners'].get(ref_corner, {})
                                ref_lat = ref_geo_corner.get('lat')
                                ref_lon = ref_geo_corner.get('lon')
                                
                                if all(v is not None for v in [ref_lat, ref_lon, ref_x, ref_y]):
                                    px_to_ft = st.session_state.px_to_ft
                                    
                                    # First, collect all new values and check what changed for the entire plot
                                    new_values = {}
                                    any_pixel_changed = False
                                    any_geo_changed = False
                                    
                                    for corner in ['A', 'B', 'C', 'D']:
                                        new_x = float(row[f'{corner}_x'])
                                        new_y = float(row[f'{corner}_y'])
                                        new_lat = float(row[f'{corner}_lat'])
                                        new_lon = float(row[f'{corner}_lon'])
                                        
                                        # Get original values
                                        orig_x = original_pixel_corners.get(corner, {}).get('x', 0)
                                        orig_y = original_pixel_corners.get(corner, {}).get('y', 0)
                                        orig_lat = original_geo_corners.get(corner, {}).get('lat', 0)
                                        orig_lon = original_geo_corners.get(corner, {}).get('lon', 0)
                                        
                                        # Check if this corner changed
                                        x_changed = abs(new_x - orig_x) > 0.01
                                        y_changed = abs(new_y - orig_y) > 0.01
                                        lat_changed = abs(new_lat - orig_lat) > 0.0000001
                                        lon_changed = abs(new_lon - orig_lon) > 0.0000001
                                        
                                        if x_changed or y_changed:
                                            any_pixel_changed = True
                                        if lat_changed or lon_changed:
                                            any_geo_changed = True
                                        
                                        new_values[corner] = {
                                            'x': new_x,
                                            'y': new_y,
                                            'lat': new_lat,
                                            'lon': new_lon
                                        }
                                    
                                    # Now recalculate all corners based on what changed
                                    if any_geo_changed:
                                        # If ANY lat/lon changed, recalculate ALL pixel coordinates
                                        for corner in ['A', 'B', 'C', 'D']:
                                            new_lat = new_values[corner]['lat']
                                            new_lon = new_values[corner]['lon']
                                            calculated_x, calculated_y = recalculate_pixel_from_coordinates(
                                                ref_lat, ref_lon, ref_x, ref_y, new_lat, new_lon, px_to_ft
                                            )
                                            pixel_plot['corners'][corner]['x'] = calculated_x
                                            pixel_plot['corners'][corner]['y'] = calculated_y
                                            geo_plot['corners'][corner]['lat'] = new_lat
                                            geo_plot['corners'][corner]['lon'] = new_lon
                                        changes_made = True
                                    
                                    elif any_pixel_changed:
                                        # If ANY pixel x/y changed, recalculate ALL lat/lon
                                        for corner in ['A', 'B', 'C', 'D']:
                                            new_x = new_values[corner]['x']
                                            new_y = new_values[corner]['y']
                                            calculated_lat, calculated_lon = recalculate_coordinates_from_pixel(
                                                ref_lat, ref_lon, ref_x, ref_y, new_x, new_y, px_to_ft
                                            )
                                            geo_plot['corners'][corner]['lat'] = calculated_lat
                                            geo_plot['corners'][corner]['lon'] = calculated_lon
                                            pixel_plot['corners'][corner]['x'] = new_x
                                            pixel_plot['corners'][corner]['y'] = new_y
                                        changes_made = True
                                    
                                    else:
                                        # No changes detected, but still update to ensure consistency
                                        for corner in ['A', 'B', 'C', 'D']:
                                            pixel_plot['corners'][corner]['x'] = new_values[corner]['x']
                                            pixel_plot['corners'][corner]['y'] = new_values[corner]['y']
                                            geo_plot['corners'][corner]['lat'] = new_values[corner]['lat']
                                            geo_plot['corners'][corner]['lon'] = new_values[corner]['lon']
                                    
                                    if changes_made:
                                        updated_count += 1
                                        any_plot_changed = True
                                else:
                                    st.error(f"❌ Reference corner data incomplete for plot {plot_num}. Missing values.")
                            else:
                                st.error(f"❌ Reference plot not found or missing corners.")
                        else:
                            # No reference settings, just update values directly
                            for corner in ['A', 'B', 'C', 'D']:
                                pixel_plot['corners'][corner]['x'] = float(row[f'{corner}_x'])
                                pixel_plot['corners'][corner]['y'] = float(row[f'{corner}_y'])
                                geo_plot['corners'][corner]['lat'] = float(row[f'{corner}_lat'])
                                geo_plot['corners'][corner]['lon'] = float(row[f'{corner}_lon'])
                            changes_made = True
                            updated_count += 1
                            any_plot_changed = True
                
                # Second pass: If any plot changed, recalculate ALL plots to maintain consistency
                if any_plot_changed and has_ref_settings:
                    with st.spinner("🔄 Recalculating all plots based on changes..."):
                        # Get current reference settings
                        ref_plot_id = st.session_state.ref_plot_id
                        ref_corner = st.session_state.get('ref_corner', 'A')
                        px_to_ft = st.session_state.px_to_ft
                        
                        # Get updated reference plot's geo coordinates
                        ref_geo_plot = next((gp for gp in st.session_state.geo_plots if gp.get('plot_id') == ref_plot_id), None)
                        ref_plot = next((p for p in st.session_state.plots if p.get('plot_id') == ref_plot_id), None)
                        
                        if ref_geo_plot and ref_plot:
                            ref_geo_corner = ref_geo_plot['corners'].get(ref_corner, {})
                            ref_lat = ref_geo_corner.get('lat')
                            ref_lon = ref_geo_corner.get('lon')
                            
                            if ref_lat is not None and ref_lon is not None:
                                # Recalculate ALL plots using the updated coordinates
                                st.session_state.geo_plots = calculate_geocoordinates(
                                    st.session_state.plots, ref_plot_id, ref_corner,
                                    ref_lat, ref_lon, px_to_ft
                                )
                                changes_made = True
                
                if changes_made:
                    # Increment refresh counter to force grid refresh
                    st.session_state.latlon_grid_refresh += 1
                    
                    # Force update session state to ensure changes are saved
                    st.session_state.geo_plots = st.session_state.geo_plots.copy()
                    st.session_state.plots = st.session_state.plots.copy()
                    
                    st.success(f"✅ Coordinates updated successfully! All {len(st.session_state.geo_plots)} plots have been recalculated. Grid and map will refresh.")
                    st.rerun()
                else:
                    st.info("ℹ️ No changes detected. Values are already up to date.")
        
        # Navigation buttons for Step 7
        if st.session_state.geo_plots:
            col_btn1, col_btn2, col_btn3 = st.columns([6, 2, 2])
            with col_btn2:
                if st.button("Previous", use_container_width=True, key="prev_step7"):
                    st.session_state.current_step = 6  # Go back to Step 6
                    st.rerun()
            with col_btn3:
                if st.button("Next", type="primary", use_container_width=True, key="next_step7"):
                    st.session_state.current_step = 8  # Go to Step 8 (Preview in Google Map)
                    st.rerun()
    else:
        st.info("Please complete previous steps first.")

# STEP 8: Preview in Google Map (shows map view)
elif st.session_state.current_step == 8:
    st.header("8 - Preview in Google Map")
    
    if not st.session_state.plots:
        st.info("⚠️ Please complete the configuration steps in the 'Detection & Configuration' tab and generate the map first.")
    else:
        # Header row with Total geo_plots and Configure Map Settings button
        col_header1, col_header2 = st.columns([3, 1])
        with col_header1:
            if st.session_state.geo_plots:
                st.write(f"**Total geo_plots:** {len(st.session_state.geo_plots)}")
            else:
                st.write("**Total geo_plots:** 0 (Configure map settings to generate)")
        with col_header2:
            # Initialize session state for showing config panel
            if 'show_config_map_step8' not in st.session_state:
                st.session_state.show_config_map_step8 = False
            
            if st.button("⚙️ Configure Map Settings", type="secondary", use_container_width=True, key="btn_config_map_step8"):
                st.session_state.show_config_map_step8 = not st.session_state.show_config_map_step8
                st.rerun()
        
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
                
                col_apply, col_close = st.columns([1, 1])
                with col_apply:
                    if st.button("🗺️ Generate Map", type="primary", use_container_width=False, key="generate_map_step8"):
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
                    if st.button("❌ Close", use_container_width=False, key="close_config_step8"):
                        st.session_state.show_config_map_step8 = False
                        st.rerun()
                
                st.divider()
        
        # Map display section - only show if geo_plots exist
        if not st.session_state.geo_plots:
            st.info("💡 Click '⚙️ Configure Map Settings' above to generate the map.")
        else:
            st.subheader("📍 Interactive Map View")
            
            # Collect all valid coordinates from ALL plots for proper map bounds
            all_lats = []
            all_lons = []
            invalid_coords_count = 0
            
            for plot in st.session_state.geo_plots:
                for corner in ['A', 'B', 'C', 'D']:
                    lat = plot['corners'][corner].get('lat')
                    lon = plot['corners'][corner].get('lon')
                    
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
        
        # Add custom controls for scale, move, and rotate - positioned below map
        st.components.v1.html("""
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
                let leafletMap = null;
                let mapIframe = null;
                let retryCount = 0;
                const maxRetries = 30;
                
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
                
                function initMapControls() {
                    if (!leafletMap) {
                        console.log('Map not found, retrying...');
                        setTimeout(findAndInitMap, 500);
                        return;
                    }
                    
                    console.log('Initializing map controls...');
                    
                    // Buttons are in the same document as this script
                    const scaleUpBtn = document.getElementById('map-scale-up');
                    const scaleDownBtn = document.getElementById('map-scale-down');
                    
                    if (scaleUpBtn) {
                        scaleUpBtn.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            console.log('Map scale up clicked!');
                            try {
                                if (leafletMap && leafletMap.getZoom) {
                                    const currentZoom = leafletMap.getZoom();
                                    leafletMap.setZoom(Math.min(leafletMap.getMaxZoom(), currentZoom + 1));
                                    console.log('Map zoomed in to:', leafletMap.getZoom());
                                }
                            } catch(err) {
                                console.error('Error zooming in:', err);
                            }
                            return false;
                        };
                        console.log('Map scale up button connected');
                    }
                    
                    if (scaleDownBtn) {
                        scaleDownBtn.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            console.log('Map scale down clicked!');
                            try {
                                if (leafletMap && leafletMap.getZoom) {
                                    const currentZoom = leafletMap.getZoom();
                                    leafletMap.setZoom(Math.max(leafletMap.getMinZoom(), currentZoom - 1));
                                    console.log('Map zoomed out to:', leafletMap.getZoom());
                                }
                            } catch(err) {
                                console.error('Error zooming out:', err);
                            }
                            return false;
                        };
                        console.log('Map scale down button connected');
                    }
                    
                    // Move controls
                    const moveUpBtn = document.getElementById('map-move-up');
                    const moveDownBtn = document.getElementById('map-move-down');
                    const moveLeftBtn = document.getElementById('map-move-left');
                    const moveRightBtn = document.getElementById('map-move-right');
                    
                    if (moveUpBtn) {
                        moveUpBtn.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            console.log('Map move up clicked!');
                            try {
                                if (leafletMap && leafletMap.panBy) {
                                    leafletMap.panBy([0, -50]);
                                }
                            } catch(err) {
                                console.error('Error moving up:', err);
                            }
                            return false;
                        };
                        console.log('Map move up button connected');
                    }
                    
                    if (moveDownBtn) {
                        moveDownBtn.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            console.log('Map move down clicked!');
                            try {
                                if (leafletMap && leafletMap.panBy) {
                                    leafletMap.panBy([0, 50]);
                                }
                            } catch(err) {
                                console.error('Error moving down:', err);
                            }
                            return false;
                        };
                        console.log('Map move down button connected');
                    }
                    
                    if (moveLeftBtn) {
                        moveLeftBtn.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            console.log('Map move left clicked!');
                            try {
                                if (leafletMap && leafletMap.panBy) {
                                    leafletMap.panBy([-50, 0]);
                                }
                            } catch(err) {
                                console.error('Error moving left:', err);
                            }
                            return false;
                        };
                        console.log('Map move left button connected');
                    }
                    
                    if (moveRightBtn) {
                        moveRightBtn.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            console.log('Map move right clicked!');
                            try {
                                if (leafletMap && leafletMap.panBy) {
                                    leafletMap.panBy([50, 0]);
                                }
                            } catch(err) {
                                console.error('Error moving right:', err);
                            }
                            return false;
                        };
                        console.log('Map move right button connected');
                    }
                    
                    // Rotate control (Leaflet doesn't natively support rotation)
                    const rotateInput = document.getElementById('map-rotate-input');
                    if (rotateInput) {
                        rotateInput.disabled = true;
                        rotateInput.style.backgroundColor = '#f0f0f0';
                        rotateInput.style.cursor = 'not-allowed';
                        rotateInput.title = 'Map rotation is not supported by Leaflet';
                        rotateInput.onchange = function() {
                            const rotation = parseFloat(this.value) || 0;
                            console.log('Rotation requested:', rotation, 'but not supported by Leaflet maps');
                        };
                    }
                    
                    console.log('Map controls initialized!');
                }
                
                // Start searching for the map
                setTimeout(findAndInitMap, 1000);
            })();
        </script>
        """, height=200)
        
        # Navigation buttons for Step 8 (final step)
        col_btn1, col_btn2, col_btn3 = st.columns([6, 2, 2])
        with col_btn1:
            st.write("")  # Spacer to align with controls panel above
        with col_btn2:
            if st.button("Previous", use_container_width=True, key="prev_step8"):
                st.session_state.current_step = 7  # Go back to Step 7
                st.rerun()
        with col_btn3:
            if st.button("Publish", type="primary", use_container_width=True, key="publish_step8"):
                st.success("✅ Map published successfully!")
                # Stay on Step 8 as it's the final step

st.divider()
st.caption("🔧 Geo Plot Mapper v2.2 - Contour Detection with Sequential Logic")