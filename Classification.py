import os
import argparse
import sys
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont 
from tqdm import tqdm
import rasterio
import rasterio.features
from rasterio.windows import Window
from rasterio.enums import Resampling

# Excel Library Check
try:
    import xlsxwriter
    HAVE_XLSX = True
except ImportError:
    HAVE_XLSX = False
    print("WARNING: 'xlsxwriter' not found. Excel chart will be skipped.")

# GUI Imports
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, Toplevel, Label, Button
    HAVE_TK = True
except Exception:
    HAVE_TK = False

NDVI_VEG_THRESHOLD = 0.20 
NDWI_WATER_THRESHOLD = 0.05 
BRIGHTNESS_SAT_THRESHOLD_REFLECT = 0.80
KMEANS_K = 2
DEFAULT_KMEANS_SAMPLE = 150_000
DEFAULT_TILE_SIZE = 2048
SMOOTH_ITERS = 1
DOWNSAMPLE_PREVIEW = 10
INVALID_PIXEL_VALUE = 255
CANDIDATE_SCALES = [1.0, 100.0, 1000.0, 10000.0]
PERCENTILE_FOR_SCALE = 98.0
MIN_WATER_BLOB_SIZE = 50 

# ---------------- helper IO funcs ----------------
def find_band_file(folder, patterns):
    if not os.path.exists(folder): return None
    for fname in os.listdir(folder):
        ln = fname.lower()
        for p in patterns:
            if p in ln:
                return os.path.join(folder, fname)
    return None

def detect_bands(folder):
    return {
        'B3': find_band_file(folder, ['b3','b03','_03','band3']),
        'B4': find_band_file(folder, ['b4','b04','_04','band4']),
        'B8': find_band_file(folder, ['b8','b08','_08','band8']),
        'B11': find_band_file(folder, ['b11','_11','band11']),
    }

def read_window_resample_ds(ds, win, out_h, out_w):
    try:
        arr = ds.read(window=win, out_shape=(1, out_h, out_w), resampling=Resampling.bilinear)[0]
        return arr.astype(np.float32)
    except Exception:
        return None

# ---------------- statistical helpers ----------------
def percentiles_of_sample_open_ds(b3_ds, b4_ds, b8_ds, tiles_to_sample=6, tile_size=512):
    rng = np.random.default_rng(seed=0)
    height, width = b4_ds.height, b4_ds.width
    collected = {'B3': [], 'B4': [], 'B8': []}
    tiles_to_sample = min(tiles_to_sample, (height // tile_size) * (width // tile_size) + 1)
    for _ in range(tiles_to_sample):
        if width <= tile_size or height <= tile_size:
            x = y = 0; h = height; w = width
        else:
            y = int(rng.integers(0, max(1, height - tile_size)))
            x = int(rng.integers(0, max(1, width - tile_size)))
            h = min(tile_size, height - y)
            w = min(tile_size, width - x)
        win = Window(x, y, w, h)
        for key, ds in (('B3', b3_ds), ('B4', b4_ds), ('B8', b8_ds)):
            v = read_window_resample_ds(ds, win, h, w)
            if v is not None and v.size > 0:
                p = float(np.nanpercentile(v, PERCENTILE_FOR_SCALE))
                if not np.isnan(p) and p > 0:
                    collected[key].append(p)
    med = {}
    for k, arr in collected.items():
        med[k] = float(np.median(arr)) if len(arr) > 0 else 0.0
    return med

def detect_scale_for_open_ds(b3_ds, b4_ds, b8_ds):
    p98 = percentiles_of_sample_open_ds(b3_ds, b4_ds, b8_ds)
    vals = [v for v in p98.values() if v > 0]
    if not vals: return 1.0, 0.0
    median_val = float(np.median(vals))
    chosen = CANDIDATE_SCALES[-1]
    for sc in sorted(CANDIDATE_SCALES):
        if (median_val / sc) <= 1.5:
            chosen = float(sc)
            break
    return chosen, median_val

# ---------------- math helpers ----------------
def auto_scale(arr, scale):
    if arr is None: return None
    return arr.astype(np.float32) / float(scale)

def compute_ndvi(nir, red, eps=1e-8):
    return (nir - red) / (nir + red + eps)

def compute_ndwi(green, nir, eps=1e-8):
    return (green - nir) / (green + nir + eps)

def kmeans_from_scratch(data, K=2, max_iter=200, tol=1e-5):
    N = data.shape[0]
    rng = np.random.default_rng(seed=42)
    if N == 0: return np.zeros((K, data.shape[1]), dtype=float)
    if N <= K: return data.copy()
    idx = rng.choice(N, size=K, replace=False)
    centers = data[idx].astype(np.float64)
    for _ in range(max_iter):
        d2 = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        new_centers = np.zeros_like(centers)
        for k in range(K):
            mask = (labels == k)
            if np.any(mask):
                new_centers[k] = data[mask].mean(axis=0)
            else:
                new_centers[k] = data[rng.integers(0, N)]
        if np.allclose(centers, new_centers, atol=tol):
            break
        centers = new_centers
    return centers

def majority_filter_optimized(arr, iterations=1, invalid_value=INVALID_PIXEL_VALUE):
    if iterations == 0: return arr
    H, W = arr.shape
    if (H * W) < (800 * 1024 * 1024): 
        a = arr.copy()
        for _ in range(iterations):
            pad = np.pad(a, 1, mode='constant', constant_values=invalid_value)
            counts = np.zeros((3, H, W), dtype=np.int8)
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    y0, y1 = 1+dy, 1+dy+H
                    x0, x1 = 1+dx, 1+dx+W
                    slice_view = pad[y0:y1, x0:x1]
                    mask_v = (slice_view != invalid_value)
                    counts[0] += (slice_view == 0) & mask_v
                    counts[1] += (slice_view == 1) & mask_v
                    counts[2] += (slice_view == 2) & mask_v
            sums = np.sum(counts, axis=0)
            winners = np.argmax(counts, axis=0)
            has_neighbors = (sums > 0)
            out = np.full_like(a, invalid_value)
            out[~has_neighbors] = a[~has_neighbors]
            out[has_neighbors] = winners[has_neighbors]
            a = out
        return a
    else:
        print("Warning: Image too large for in-memory smoothing. Skipping.")
        return arr

def save_preview_from_array(arr, out_png, downsample_step=10):
    preview = arr[::downsample_step, ::downsample_step]
    rgb = np.zeros((preview.shape[0], preview.shape[1], 3), dtype=np.uint8)
    rgb[preview == 0] = (0, 0, 255)       # Water = Blue
    rgb[preview == 1] = (0, 255, 0)       # Veg = Green
    rgb[preview == 2] = (180, 180, 180)   # Other = Grey
    try:
        Image.fromarray(rgb).save(out_png)
    except Exception as e:
        print(f"Error saving PNG: {e}")

# ---------------- THEMATIC MAP WITH EMBEDDED LEGEND ----------------
def save_thematic_change_map(arr_old, arr_new, out_png, downsample_step=10):
    """
    Creates a thematic map and appends a legend at the bottom.
    """
    old_s = arr_old[::downsample_step, ::downsample_step]
    new_s = arr_new[::downsample_step, ::downsample_step]
    
    rows, cols = old_s.shape
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
    
    rgb[new_s == 0] = (0, 0, 255)
    rgb[new_s == 1] = (0, 255, 0)
    rgb[new_s == 2] = (180, 180, 180)
    
    # Highlights
    # Water Gain (Cyan), Water Loss (Red)
    rgb[(old_s != 0) & (new_s == 0)] = (0, 255, 255) 
    rgb[(old_s == 0) & (new_s != 0)] = (255, 0, 0)
    
    # Veg Gain (Purple), Veg Loss (Yellow)
    rgb[(old_s == 2) & (new_s == 1)] = (128, 0, 128)
    rgb[(old_s == 1) & (new_s == 2)] = (255, 255, 0)

    try:
        map_img = Image.fromarray(rgb)
        
        # Legend Settings
        legend_height = 60
        width, height = map_img.size
        
        final_img = Image.new("RGB", (width, height + legend_height), (255, 255, 255))
        final_img.paste(map_img, (0, 0))
        
        draw = ImageDraw.Draw(final_img)
        
        try:
            font = ImageFont.load_default() 
        except:
            font = None 

        items = [
            ((0, 255, 255), "Water Gain"),
            ((255, 0, 0),   "Water Loss"),
            ((128, 0, 128), "Veg Gain"),
            ((255, 255, 0), "Veg Loss"),
            ((180, 180, 180), "Stable")
        ]
        
        box_size = 15
        start_x = 20
        y = height + 20 
        
        for color, label in items:
           
            draw.rectangle([start_x, y, start_x + box_size, y + box_size], fill=color, outline="black")
            
            draw.text((start_x + box_size + 5, y), label, fill="black", font=font)
            
            start_x += box_size + len(label)*7 + 25

        final_img.save(out_png)
        
    except Exception as e:
        print(f"Error saving Thematic Map with Legend: {e}")
        try:
            Image.fromarray(rgb).save(out_png)
        except: pass

# ---------------- EXCEL CHART GENERATION ----------------
def generate_excel_report(out_dir, summary_list):
    if not HAVE_XLSX: return
    excel_path = os.path.join(out_dir, "Summary_Report_With_Charts.xlsx")
    workbook = xlsxwriter.Workbook(excel_path)
    worksheet = workbook.add_worksheet("Stats")
    bold = workbook.add_format({'bold': True})
    percent_fmt = workbook.add_format({'num_format': '0.00%'})
    headers = ['Image Name', 'Water (%)', 'Vegetation (%)', 'Other (%)']
    worksheet.write_row('A1', headers, bold)
    row = 1
    for s in summary_list:
        if 'pct' not in s: continue
        worksheet.write(row, 0, s['name'])
        worksheet.write(row, 1, s['pct'][0], percent_fmt)
        worksheet.write(row, 2, s['pct'][1], percent_fmt)
        worksheet.write(row, 3, s['pct'][2], percent_fmt)
        row += 1
    chart = workbook.add_chart({'type': 'column'})
    num_items = row - 1
    chart.add_series({'name': ['Stats', 0, 1], 'categories': ['Stats', 1, 0, num_items, 0], 'values': ['Stats', 1, 1, num_items, 1], 'fill': {'color': '#0000FF'}, 'data_labels': {'value': True, 'num_format': '0.00%'}})
    chart.add_series({'name': ['Stats', 0, 2], 'categories': ['Stats', 1, 0, num_items, 0], 'values': ['Stats', 1, 2, num_items, 2], 'fill': {'color': '#00FF00'}, 'data_labels': {'value': True, 'num_format': '0.00%'}})
    chart.add_series({'name': ['Stats', 0, 3], 'categories': ['Stats', 1, 0, num_items, 0], 'values': ['Stats', 1, 3, num_items, 3], 'fill': {'color': '#B4B4B4'}, 'data_labels': {'value': True, 'num_format': '0.00%'}})
    chart.set_title ({'name': 'Land Cover Change Over Time'})
    chart.set_x_axis({'name': 'Year / Image'})
    chart.set_y_axis({'name': 'Percentage Coverage'})
    chart.set_style(11)
    worksheet.insert_chart('E2', chart, {'x_scale': 1.5, 'y_scale': 1.5})
    workbook.close()
    print(f"  Generated Excel Chart: {excel_path}")

# ---------------- GUI Results Display ----------------
def show_results_window(summary_list):
    if not HAVE_TK: return
    
    def on_close():
        top.destroy()
        try:
            root = tk._default_root
            if root: root.destroy()
        except: pass
        sys.exit(0)

    top = Toplevel()
    top.title("Classification Results Gallery")
    top.protocol("WM_DELETE_WINDOW", on_close)
    top.images = [] 
    
    canvas = tk.Canvas(top, height=550)
    scrollbar = tk.Scrollbar(top, orient="horizontal", command=canvas.xview)
    scroll_frame = tk.Frame(canvas)
    
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(xscrollcommand=scrollbar.set)
    
    canvas.pack(side="top", fill="both", expand=True)
    scrollbar.pack(side="bottom", fill="x")

    for item in summary_list:
        path = item['preview']
        name = item['name']
        stats = item.get('pct', {0:0, 1:0, 2:0})
        diffs = item.get('diffs', {0:0, 1:0, 2:0}) 
        
        if os.path.exists(path):
            try:
                img_obj = Image.open(path)
                target_h = 350
                aspect = img_obj.width / img_obj.height
                target_w = int(target_h * aspect)
                img_obj = img_obj.resize((target_w, target_h), Image.Resampling.BILINEAR)
                photo = ImageTk.PhotoImage(img_obj)
                top.images.append(photo)
                
                frame = tk.Frame(scroll_frame, bd=2, relief=tk.RIDGE, padx=5, pady=5, bg="white")
                frame.pack(side=tk.LEFT, padx=10, pady=10, fill="y")
                
                lbl_img = Label(frame, image=photo, bg="white")
                lbl_img.pack()
                
                lbl_txt = Label(frame, text=name, font=("Arial", 11, "bold"), bg="white")
                lbl_txt.pack(pady=(5,0))
                
                if "Thematic Change" in name:
                    pass
                else:
                    def fmt_change(val):
                        if val == 0: return "-"
                        sym = "▲" if val > 0 else "▼"
                        return f"{sym} {abs(val):.2f}%"

                    stat_frame = tk.Frame(frame, bg="white")
                    stat_frame.pack(pady=5, fill="x")
                    
                    w_txt = f"Water: {stats[0]*100:.2f}%  ({fmt_change(diffs[0])})"
                    Label(stat_frame, text=w_txt, fg="blue", bg="white", font=("Consolas", 9)).pack(anchor="w")
                    v_txt = f"Veg:   {stats[1]*100:.2f}%  ({fmt_change(diffs[1])})"
                    Label(stat_frame, text=v_txt, fg="green", bg="white", font=("Consolas", 9)).pack(anchor="w")
                    o_txt = f"Other: {stats[2]*100:.2f}%  ({fmt_change(diffs[2])})"
                    Label(stat_frame, text=o_txt, fg="gray", bg="white", font=("Consolas", 9)).pack(anchor="w")

            except Exception as e:
                print(f"Failed to display {name}: {e}")

    btn_exit = Button(top, text="CLOSE AND EXIT", command=on_close, bg="red", fg="white", font=("Arial", 12, "bold"))
    btn_exit.pack(side="bottom", pady=10, fill="x")

# ---------------- Main pipeline ----------------
def process_folders(folders, out_dir, forced_resolution=None, tile_size=DEFAULT_TILE_SIZE, kmeans_sample=DEFAULT_KMEANS_SAMPLE, no_gui=False):
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    memory_maps = []
    rng = np.random.default_rng(seed=42)
    folders.sort()
    
    baseline_pct = None 

    for idx, folder in enumerate(folders):
        folder_name = os.path.basename(os.path.normpath(folder))
        print(f"\nProcessing {idx+1}/{len(folders)}: {folder_name}")
        
        bands = detect_bands(folder)
        if not (bands['B3'] and bands['B4'] and bands['B8']):
            print(f"Skipping {folder}: Missing B3/B4/B8")
            continue

        b3_ds = rasterio.open(bands['B3'])
        b4_ds = rasterio.open(bands['B4'])
        b8_ds = rasterio.open(bands['B8'])
        b11_ds = rasterio.open(bands['B11']) if bands.get('B11') else None
        
        H, W = b4_ds.height, b4_ds.width
        
        if forced_resolution is not None:
            pixel_area_m2 = forced_resolution * forced_resolution
        else:
            if b4_ds.transform.a != 1.0:
                pixel_area_m2 = abs(b4_ds.transform.a * b4_ds.transform.e)
            else:
                print("  Warning: No metadata found. Defaulting to 10m/px.")
                pixel_area_m2 = 100.0

        d_scale, median_val = detect_scale_for_open_ds(b3_ds, b4_ds, b8_ds)
        print(f"  Scale Factor: {d_scale}")

        full_mask = np.full((H, W), INVALID_PIXEL_VALUE, dtype=np.uint8)
        sample_reservoir = np.zeros((0, 2), dtype=np.float32)
        total_tiles = ((H // tile_size) + 1) * ((W // tile_size) + 1)
        
        pbar = tqdm(total=total_tiles, desc="  Calc")
        for y in range(0, H, tile_size):
            h_win = min(tile_size, H - y)
            for x in range(0, W, tile_size):
                w_win = min(tile_size, W - x)
                win = Window(x, y, w_win, h_win)
                red = read_window_resample_ds(b4_ds, win, h_win, w_win)
                green = read_window_resample_ds(b3_ds, win, h_win, w_win)
                nir = read_window_resample_ds(b8_ds, win, h_win, w_win)
                if red is None: 
                    pbar.update(1); continue
                r = auto_scale(red, d_scale)
                g = auto_scale(green, d_scale)
                n = auto_scale(nir, d_scale)
                swir_val = None
                if b11_ds: swir_val = read_window_resample_ds(b11_ds, win, h_win, w_win)
                s = auto_scale(swir_val, d_scale) if swir_val is not None else None
                bright = (r + g + n) / 3.0
                if (s is not None) and (s.shape == bright.shape):
                    mask = ((bright > 0.8) & (s > 0.4))
                else:
                    mask = (bright > BRIGHTNESS_SAT_THRESHOLD_REFLECT)
                ndvi = compute_ndvi(n, r)
                ndwi = compute_ndwi(g, n)
                tile_out = np.full((h_win, w_win), INVALID_PIXEL_VALUE, dtype=np.uint8)
                is_veg = (ndvi > NDVI_VEG_THRESHOLD) & (~mask)
                is_water = (ndwi > NDWI_WATER_THRESHOLD) & (ndvi < 0.0) & (~mask)
                is_amb = (~is_veg) & (~is_water) & (~mask)
                tile_out[is_veg] = 1
                tile_out[is_water] = 0
                tile_out[is_amb] = 2
                full_mask[y:y+h_win, x:x+w_win] = tile_out
                coords = np.nonzero(is_amb)
                if coords[0].size > 0:
                    feats = np.column_stack((ndvi[coords], ndwi[coords]))
                    n_new = feats.shape[0]
                    curr_size = sample_reservoir.shape[0]
                    if curr_size + n_new <= kmeans_sample:
                        sample_reservoir = np.vstack((sample_reservoir, feats))
                    else:
                        space = kmeans_sample - curr_size
                        if space > 0:
                            sample_reservoir = np.vstack((sample_reservoir, feats[:space]))
                            feats = feats[space:]
                        if feats.shape[0] > 0:
                            replace_idx = rng.integers(0, kmeans_sample, size=feats.shape[0])
                            sample_reservoir[replace_idx] = feats
                pbar.update(1)
        pbar.close()

        if sample_reservoir.shape[0] > 100:
            k_centers = kmeans_from_scratch(sample_reservoir, K=KMEANS_K)
            pbar = tqdm(total=total_tiles, desc="  Refine")
            for y in range(0, H, tile_size):
                h_win = min(tile_size, H - y)
                for x in range(0, W, tile_size):
                    w_win = min(tile_size, W - x)
                    tile_slice = full_mask[y:y+h_win, x:x+w_win]
                    if not np.any(tile_slice == 2):
                        pbar.update(1); continue
                    win = Window(x, y, w_win, h_win)
                    red = read_window_resample_ds(b4_ds, win, h_win, w_win)
                    green = read_window_resample_ds(b3_ds, win, h_win, w_win)
                    nir = read_window_resample_ds(b8_ds, win, h_win, w_win)
                    if red is None: pbar.update(1); continue
                    r = auto_scale(red, d_scale)
                    g = auto_scale(green, d_scale)
                    n = auto_scale(nir, d_scale)
                    ndvi = compute_ndvi(n, r)
                    ndwi = compute_ndwi(g, n)
                    amb_mask = (tile_slice == 2)
                    coords = np.nonzero(amb_mask)
                    if coords[0].size > 0:
                        pix_feats = np.column_stack((ndvi[coords], ndwi[coords]))
                        dists = np.sum((pix_feats[:, None, :] - k_centers[None, :, :])**2, axis=2)
                        labels = np.argmin(dists, axis=1)
                        new_vals = tile_slice[coords] 
                        for k_id in range(KMEANS_K):
                            c_ndvi = k_centers[k_id, 0]
                            c_ndwi = k_centers[k_id, 1]
                            is_water_cluster = (c_ndwi > c_ndvi) and (c_ndwi > 0.0)
                            mask_k = (labels == k_id)
                            if is_water_cluster: new_vals[mask_k] = 0
                            else: new_vals[mask_k] = 2
                        tile_slice[coords] = new_vals
                        full_mask[y:y+h_win, x:x+w_win] = tile_slice
                    pbar.update(1)
            pbar.close()

        b3_ds.close(); b4_ds.close(); b8_ds.close()
        if b11_ds: b11_ds.close()
        
        print("  Smoothing...")
        full_mask = majority_filter_optimized(full_mask, iterations=SMOOTH_ITERS)
        print("  Despeckling Water...")
        try:
            full_mask = rasterio.features.sieve(full_mask, size=MIN_WATER_BLOB_SIZE, connectivity=4)
        except Exception: pass

        memory_maps.append(full_mask)
        valid = (full_mask != INVALID_PIXEL_VALUE)
        tot_valid = np.sum(valid)
        counts = np.bincount(full_mask[valid].ravel(), minlength=3)
        
        pct = {
            0: counts[0]/tot_valid if tot_valid else 0,
            1: counts[1]/tot_valid if tot_valid else 0,
            2: counts[2]/tot_valid if tot_valid else 0
        }
        areas = {
            0: (counts[0] * pixel_area_m2) / 1e6,
            1: (counts[1] * pixel_area_m2) / 1e6,
            2: (counts[2] * pixel_area_m2) / 1e6
        }
        
        diffs = {0:0, 1:0, 2:0}
        if baseline_pct is None:
            baseline_pct = pct
        else:
            diffs[0] = (pct[0] - baseline_pct[0]) * 100 
            diffs[1] = (pct[1] - baseline_pct[1]) * 100
            diffs[2] = (pct[2] - baseline_pct[2]) * 100
        
        preview_filename = f"{folder_name}_classified.png"
        preview_png = os.path.join(out_dir, preview_filename)
        save_preview_from_array(full_mask, preview_png, DOWNSAMPLE_PREVIEW)
        
        summary.append({
            'name': folder_name,
            'preview': preview_png,
            'areas': areas,
            'pct': pct,
            'diffs': diffs 
        })

    if len(memory_maps) >= 2:
        print("\nGenerating Thematic Change Map (First vs Last)...")
        m_old = memory_maps[0]
        m_new = memory_maps[-1]
        name_old = summary[0]['name']
        name_new = summary[-1]['name']
        
        print(f"  Comparing: {name_old} -> {name_new}")
        
        thematic_name = f"thematic_change_{name_old}_vs_{name_new}.png"
        thematic_path = os.path.join(out_dir, thematic_name)
        save_thematic_change_map(m_old, m_new, thematic_path, DOWNSAMPLE_PREVIEW)
        
        summary.append({
            'name': f"Thematic Change\n({name_old} vs {name_new})",
            'preview': thematic_path
        })

    print("\nGenerating Excel Report...")
    generate_excel_report(out_dir, summary)
            
    print(f"\nDone! Saved to: {out_dir}")
    if HAVE_TK and not no_gui:
        show_results_window(summary)
        tk.mainloop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gui', action='store_true')
    parser.add_argument('--input', nargs='*')
    parser.add_argument('--out')
    args = parser.parse_args()

    if not args.no_gui and HAVE_TK:
        root = tk.Tk(); root.withdraw()
        forced_res = None
        if messagebox.askyesno("Resolution Mode", "Do you want to manually enforce a resolution (e.g., 10m) for ALL images?"):
            val = simpledialog.askfloat("Set Resolution", "Enter resolution in meters/pixel:", initialvalue=10.0)
            if val is not None:
                forced_res = float(val)

        folders = []
        messagebox.showinfo("Select Input", "Select Folder 1 (Oldest), then Folder 2, etc.")
        while True:
            d = filedialog.askdirectory(title=f"Select Folder {len(folders)+1}")
            if not d: break
            folders.append(d)
            if len(folders) >= 5: break
            if not messagebox.askyesno("More?", "Add another folder?"): break
        
        if len(folders) < 2: return
        out = filedialog.askdirectory(title="Select Output Folder")
        if not out: return
        process_folders(folders, out, forced_resolution=forced_res, no_gui=False)
    else:
        if args.input and args.out:
            process_folders(args.input, args.out, forced_resolution=10.0, no_gui=True)
        else:
            print("Usage: python script.py --no-gui --input f1 f2 --out output_dir")

if __name__ == "__main__":
    main()