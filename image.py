"""
Robust extraction of wavelength vs transmission from the provided plot image.

Usage:
 - Replace input_image_path with the path to your image file.
 - Replace output_csv_path with where you want the CSV saved.
 - Requires: numpy, opencv-python (cv2), scipy, pandas, matplotlib
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d

# ---------- USER PATHS ----------
input_image_path = "Screenshot 2025-12-08 131642.png"   # <-- replace
output_csv_path = r"C:\Jonathan\Local Lab Y2\data.csv"  # <-- replace

# ---------- Helpers ----------
def detect_plot_frame(gray):
    # Detect thick dark border by thresholding and contour area
    mask = (gray < 80).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    h,w = gray.shape
    for cnt in contours:
        x,y,ww,hh = cv2.boundingRect(cnt)
        area = ww*hh
        if area > best_area and ww > 0.3*w and hh > 0.3*h:
            best_area = area
            best = (x,y,ww,hh)
    if best is None:
        # fallback guess
        best = (int(0.12*w), int(0.08*h), int(0.76*w), int(0.8*h))
    return best

def detect_grid_peaks(gray_crop, axis='vertical', min_height=0.35, min_dist=8):
    # axis 'vertical' -> sum columns; 'horizontal' -> sum rows
    if axis == 'vertical':
        prof = cv2.GaussianBlur(gray_crop, (9,9), 0).mean(axis=0)
    else:
        prof = cv2.GaussianBlur(gray_crop, (9,9), 0).mean(axis=1)
    prof_norm = (prof - prof.min()) / (np.ptp(prof) + 1e-9)
    peaks, props = find_peaks(prof_norm, height=min_height, distance=min_dist)
    return peaks, prof_norm

def choose_even_spaced(peaks, count):
    if len(peaks) >= count:
        idxs = np.linspace(0, len(peaks)-1, count).astype(int)
        return peaks[idxs]
    else:
        # fallback spread across full crop width/height
        return np.linspace(0, (peaks.max() if len(peaks)>0 else 100), count).astype(int)

def detect_blue_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([100, 100, 50])
    upper = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return mask

def extract_median_per_x(mask_blue, x_min, x_max, y_min, y_max):
    xs, ys = [], []
    for x in range(x_min, x_max+1):
        col = mask_blue[y_min:y_max+1, x]
        ids = np.where(col>0)[0]
        if ids.size>0:
            median_rel = int(np.median(ids))
            ys.append(y_min + median_rel)
            xs.append(x)
    return np.array(xs), np.array(ys)

# ---------- Main ----------
if not os.path.exists(input_image_path):
    raise FileNotFoundError(f"Image not found: {input_image_path}")

img = cv2.imread(input_image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h,w = gray.shape

# detect frame
bx,by,bw,bh = detect_plot_frame(gray)
pad = 6
x0 = bx + pad; x1 = bx + bw - pad
y0 = by + pad; y1 = by + bh - pad
crop = gray[y0:y1, x0:x1]

# detect grid peaks
v_peaks, v_prof = detect_grid_peaks(crop, axis='vertical', min_height=0.35, min_dist=8)
h_peaks, h_prof = detect_grid_peaks(crop, axis='horizontal', min_height=0.35, min_dist=6)

# ensure enough peaks by lowering thresholds if needed
if len(v_peaks) < 6:
    v_peaks, _ = detect_grid_peaks(crop, axis='vertical', min_height=0.25, min_dist=6)
if len(h_peaks) < 9:
    h_peaks, _ = detect_grid_peaks(crop, axis='horizontal', min_height=0.25, min_dist=4)

# choose evenly spaced ticks
v_choice = choose_even_spaced(v_peaks, 6)  # positions inside crop
h_choice = choose_even_spaced(h_peaks, 9)

# convert to image coordinates
vlines_px = (v_choice + x0).astype(int)
hlines_px = (h_choice + y0).astype(int)

# assign known grid values (from the chart)
x_grid_vals = np.array([500,520,540,560,580,600])
y_grid_vals = np.array([80,70,60,50,40,30,20,10,0])  # top->bottom

# detect blue curve
mask_blue = detect_blue_mask(img)

# extract curve inside detected plot box
xs_px, ys_px = extract_median_per_x(mask_blue, x0, x1, y0, y1)
if len(xs_px)==0:
    raise RuntimeError("No blue pixels found; may need to adjust HSV thresholds for detect_blue_mask()")

# map pixels -> data (linear least squares)
A = np.vstack([vlines_px, np.ones(len(vlines_px))]).T
a_x, b_x = np.linalg.lstsq(A, x_grid_vals, rcond=None)[0]
B = np.vstack([hlines_px, np.ones(len(hlines_px))]).T
a_y, b_y = np.linalg.lstsq(B, y_grid_vals, rcond=None)[0]
data_x = a_x * xs_px + b_x
data_y = a_y * ys_px + b_y

# sort and smooth
order = np.argsort(data_x)
data_x = data_x[order]; data_y = data_y[order]; xs_px = xs_px[order]; ys_px = ys_px[order]
if len(data_y) >= 11:
    y_smooth = savgol_filter(data_y, 11, 3)
else:
    y_smooth = gaussian_filter1d(data_y, sigma=2)

# remove large step-like jumps (artifact)
dy = np.diff(y_smooth)
jumps = np.where(np.abs(dy) > 8)[0]
y_final = y_smooth.copy()
if jumps.size > 0:
    for ji in jumps:
        left = max(0, ji-1); right = min(len(y_final)-1, ji+2)
        y_final[left:right+1] = np.nan
    nans = np.isnan(y_final)
    if np.any(nans):
        y_final[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], y_final[~nans])

# save CSV
df = pd.DataFrame({
    'wavelength_nm': data_x,
    'transmission_percent': data_y,
    'transmission_smoothed_percent': y_final,
    'pixel_x': xs_px,
    'pixel_y': ys_px
})
df.to_csv(output_csv_path, index=False)
print("Saved CSV to:", output_csv_path)

# Optional: plot overlay and show summary metrics
peak_idx = int(np.nanargmax(y_final))
peak_wl = data_x[peak_idx]; peak_val = y_final[peak_idx]
half = peak_val / 2.0
def find_crossings(x,y,half):
    inds = np.where(y>=half)[0]
    if inds.size==0: return None,None
    iL,iR = inds[0], inds[-1]
    # left interp
    if iL == 0:
        L = x[0]
    else:
        x1,x2 = x[iL-1], x[iL]; y1,y2 = y[iL-1], y[iL]
        L = x1 + (half - y1) * (x2-x1) / (y2-y1) if (y2-y1)!=0 else x1
    if iR == len(x)-1:
        R = x[-1]
    else:
        x1,x2 = x[iR], x[iR+1]; y1,y2 = y[iR], y[iR+1]
        R = x1 + (half - y1) * (x2-x1) / (y2-y1) if (y2-y1)!=0 else x1
    return L,R

L,R = find_crossings(data_x, y_final, half)
fwhm = (R-L) if (L is not None and R is not None) else None
print(f"Peak: {peak_wl:.3f} nm, {peak_val:.3f}% ; FWHM est: {fwhm:.3f} nm (L {L}, R {R})")

# Visual overlay
plt.figure(figsize=(10,6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
pxs = (data_x - b_x) / a_x
pys = (y_final - b_y) / a_y
plt.plot(pxs, pys, '-r', linewidth=2)
plt.scatter(pxs, pys, s=4, c='red')
for xv in vlines_px: plt.axvline(x=xv, color='yellow', alpha=0.6)
for yv in hlines_px: plt.axhline(y=yv, color='yellow', alpha=0.6)
plt.axis('off'); plt.title('Final extraction overlay')
plt.show()
