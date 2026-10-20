import os
import re
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

def generate_banners():
    photo_path = 'c:/Users/Harish/OneDrive/Desktop/github_readme/photo.jpg'
    output_dir = 'c:/Users/Harish/OneDrive/Desktop/github_readme'
    
    print("[1/6] Loading and cropping photo...")
    im = Image.open(photo_path)
    w_orig, h_orig = im.size
    
    # Head + shoulders crop: centered horizontally around x=395, width=635, top=110
    crop_w = 635
    crop_h = int(crop_w * 340 / 300) # 719 px
    left = int(395 - crop_w / 2)
    top = 110
    right = left + crop_w
    bottom = top + crop_h
    
    im_crop = im.crop((left, top, right, bottom)).resize((300, 340), Image.Resampling.LANCZOS)
    
    # Preprocessing:
    # Contrast 1.3x only, autocontrast(cutoff=1), UnsharpMask(radius=3, percent=140)
    im_gray = im_crop.convert('L')
    im_auto = ImageOps.autocontrast(im_gray, cutoff=1)
    im_cont = ImageEnhance.Contrast(im_auto).enhance(1.3)
    im_sharp = im_cont.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    arr_sharp = np.array(im_sharp, dtype=np.float64)
    
    print("[2/6] Segmenting background for Dark Mode...")
    # GrabCut with color seeds on cropped RGB
    import cv2
    img_bgr = cv2.cvtColor(np.array(im_crop), cv2.COLOR_RGB2BGR)
    h, w = 340, 300
    
    mask_init = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
    # Background seeds: top border and upper corners
    mask_init[:20, :] = cv2.GC_BGD
    mask_init[:110, :20] = cv2.GC_BGD
    mask_init[:110, w-20:] = cv2.GC_BGD
    
    # Foreground seeds: face, shirt, lower body
    mask_init[70:160, 110:190] = cv2.GC_FGD
    mask_init[170:240, 120:180] = cv2.GC_FGD
    mask_init[280:, :] = cv2.GC_FGD
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, mask_init, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    
    fg_mask = np.where((mask_init == cv2.GC_FGD) | (mask_init == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    fg_mask = ndimage.binary_closing(fg_mask, structure=np.ones((7,7))).astype(np.uint8)
    fg_mask = ndimage.binary_fill_holes(fg_mask).astype(np.uint8)
    
    lbl, num = ndimage.label(fg_mask)
    sizes = ndimage.sum(fg_mask, lbl, range(num + 1))
    largest = (lbl == np.argmax(sizes))
    mask_subject = ndimage.binary_fill_holes(largest).astype(bool)
    
    print(f"Mask subject coverage: {mask_subject.mean():.4f}")
    
    # Shadow floor inside mask so suit outline is visible (~17k dots total)
    arr_dark_input = arr_sharp.copy()
    arr_dark_input[mask_subject] = 28.0 + (arr_sharp[mask_subject] / 255.0) * (255.0 - 28.0)
    arr_dark_input[~mask_subject] = 0.0
    
    print("[3/6] Running Serpentine Floyd-Steinberg Dither...")
    def dither_serpentine(input_arr, is_dark, mask):
        h_arr, w_arr = input_arr.shape
        if is_dark:
            buf = input_arr.copy().astype(np.float64)
            if mask is not None:
                buf[~mask] = 0.0
        else:
            # Light mode: dots draw dark parts of photo
            buf = 255.0 - input_arr.copy().astype(np.float64)
            
        dots = np.zeros((h_arr, w_arr), dtype=np.uint8)
        
        for y in range(h_arr):
            direction = 1 if y % 2 == 0 else -1
            x_range = range(w_arr) if direction == 1 else range(w_arr - 1, -1, -1)
            
            for x in x_range:
                if is_dark and mask is not None and not mask[y, x]:
                    dots[y, x] = 0
                    buf[y, x] = 0.0
                    continue
                    
                old_val = buf[y, x]
                new_val = 255.0 if old_val >= 128.0 else 0.0
                dots[y, x] = 1 if new_val == 255.0 else 0
                err = old_val - new_val
                
                # Diffuse error
                def add_err(ny, nx, weight):
                    if 0 <= ny < h_arr and 0 <= nx < w_arr:
                        if is_dark and mask is not None:
                            if mask[ny, nx]:
                                buf[ny, nx] += err * weight
                        else:
                            buf[ny, nx] += err * weight
                            
                add_err(y, x + direction, 7.0 / 16.0)
                add_err(y + 1, x - direction, 3.0 / 16.0)
                add_err(y + 1, x, 5.0 / 16.0)
                add_err(y + 1, x + direction, 1.0 / 16.0)
                
        if is_dark and mask is not None:
            dots[~mask] = 0
        return dots
        
    dots_dark = dither_serpentine(arr_dark_input, is_dark=True, mask=mask_subject)
    dots_light = dither_serpentine(arr_sharp, is_dark=False, mask=None)
    
    n_dark = np.sum(dots_dark)
    n_light = np.sum(dots_light)
    print(f"Total dots — Dark mode: {n_dark}, Light mode: {n_light}")
    
    print("[4/6] Sampling 3 Logos and Computing Optimal Transport...")
    # 900 points for each logo, centered at (150, 170)
    cx, cy = 150.0, 170.0
    N_TRAVELLERS = 900
    
    # 1. Python Logo: trace official vector contour
    python_svg_path = (
        'M 99.05 0 C 44.7 0 47.19 23.75 47.19 23.75 L 47.19 48.37 L 99.05 48.37 L 99.05 55.77 '
        'L 27.25 55.77 C 27.25 55.77 0 52.69 0 107.03 C 0 161.37 23.75 158.29 23.75 158.29 '
        'L 44.73 158.29 L 44.73 128.71 C 44.73 128.71 43.5 94.22 79.23 94.22 L 131.09 94.22 '
        'C 131.09 94.22 163.74 94.84 163.74 63.43 L 163.74 30.82 C 163.74 30.82 168.05 0 99.05 0 Z '
        'M 100.95 200 C 155.3 200 152.81 176.25 152.81 176.25 L 152.81 151.63 L 100.95 151.63 '
        'L 100.95 144.23 L 172.75 144.23 C 172.75 144.23 200 147.31 200 92.97 C 200 38.63 176.25 41.71 176.25 41.71 '
        'L 155.27 41.71 L 155.27 71.29 C 155.27 71.29 156.5 105.78 120.77 105.78 L 68.91 105.78 '
        'C 68.91 105.78 36.26 105.16 36.26 136.57 L 36.26 169.18 C 36.26 169.18 31.95 200 100.95 200 Z'
    )
    
    def parse_path_pts(path_str, n_points):
        tokens = re.findall(r'([A-Za-z]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', path_str)
        raw_pts = []
        curr = [0.0, 0.0]
        start_sub = [0.0, 0.0]
        i = 0
        cmd = ''
        while i < len(tokens):
            t = tokens[i]
            if t.isalpha():
                cmd = t
                i += 1
            if cmd == 'M':
                x, y = float(tokens[i]), float(tokens[i+1])
                curr = [x, y]
                start_sub = [x, y]
                raw_pts.append(curr.copy())
                i += 2
                cmd = 'L'
            elif cmd == 'L':
                x, y = float(tokens[i]), float(tokens[i+1])
                p0, p1 = np.array(curr), np.array([x, y])
                for s in np.linspace(0, 1, 15):
                    raw_pts.append((p0 * (1 - s) + p1 * s).tolist())
                curr = [x, y]
                i += 2
            elif cmd == 'C':
                x1, y1 = float(tokens[i]), float(tokens[i+1])
                x2, y2 = float(tokens[i+2]), float(tokens[i+3])
                x, y = float(tokens[i+4]), float(tokens[i+5])
                p0, p1, p2, p3 = np.array(curr), np.array([x1, y1]), np.array([x2, y2]), np.array([x, y])
                for s in np.linspace(0, 1, 20):
                    pt = (1-s)**3 * p0 + 3*(1-s)**2 * s * p1 + 3*(1-s) * s**2 * p2 + s**3 * p3
                    raw_pts.append(pt.tolist())
                curr = [x, y]
                i += 6
            elif cmd in ['Z', 'z']:
                p0, p1 = np.array(curr), np.array(start_sub)
                for s in np.linspace(0, 1, 15):
                    raw_pts.append((p0 * (1 - s) + p1 * s).tolist())
                curr = start_sub.copy()
            else:
                i += 1
        raw_pts = np.array(raw_pts)
        # Uniform arc-length resampling to exactly n_points
        dists = np.sqrt(np.sum(np.diff(raw_pts, axis=0)**2, axis=1))
        cum_dist = np.insert(np.cumsum(dists), 0, 0)
        total_len = cum_dist[-1]
        target_dists = np.linspace(0, total_len, n_points, endpoint=False)
        pts_resampled = np.zeros((n_points, 2))
        for dim in range(2):
            pts_resampled[:, dim] = np.interp(target_dists, cum_dist, raw_pts[:, dim])
        return pts_resampled
        
    pts_py_raw = parse_path_pts(python_svg_path, N_TRAVELLERS)
    # Center Python logo at (150, 170) and scale to ~160px
    py_min = pts_py_raw.min(axis=0)
    py_max = pts_py_raw.max(axis=0)
    py_center = (py_min + py_max) / 2
    scale_py = 160.0 / max(py_max - py_min)
    pts_logo1 = (pts_py_raw - py_center) * scale_py + np.array([cx, cy])
    
    # 2. </> Code Glyph:
    pts_code = []
    # Left chevron: (115, 120) -> (75, 170) -> (115, 220) [280 pts]
    for s in np.linspace(0, 1, 140, endpoint=False):
        pts_code.append([115 - 40*s, 120 + 50*s])
    for s in np.linspace(0, 1, 140, endpoint=False):
        pts_code.append([75 + 40*s, 170 + 50*s])
    # Slash: (135, 225) -> (165, 115) [340 pts]
    for s in np.linspace(0, 1, 340, endpoint=False):
        pts_code.append([135 + 30*s, 225 - 110*s])
    # Right chevron: (185, 120) -> (225, 170) -> (185, 220) [280 pts]
    for s in np.linspace(0, 1, 140, endpoint=False):
        pts_code.append([185 + 40*s, 120 + 50*s])
    for s in np.linspace(0, 1, 140, endpoint=False):
        pts_code.append([225 - 40*s, 170 + 50*s])
    pts_logo2 = np.array(pts_code)
    
    # 3. React Logo: Nucleus + 3 rotated ellipses
    pts_react = []
    # Center nucleus: 90 pts, r=14
    for th in np.linspace(0, 2*np.pi, 90, endpoint=False):
        pts_react.append([cx + 14 * np.cos(th), cy + 14 * np.sin(th)])
    # 3 ellipses: 270 pts each
    for ang in [0, 60, 120]:
        rad = np.radians(ang)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        for t in np.linspace(0, 2*np.pi, 270, endpoint=False):
            ex = 86 * np.cos(t)
            ey = 33 * np.sin(t)
            pts_react.append([cx + ex * cos_a - ey * sin_a, cy + ex * sin_a + ey * cos_a])
    pts_logo3 = np.array(pts_react)
    
    # Optimal Transport Matching:
    # Logo 1 -> Logo 2
    cost12 = cdist(pts_logo1, pts_logo2)
    _, p12 = linear_sum_assignment(cost12)
    pts_logo2_matched = pts_logo2[p12]
    
    # Logo 2 -> Logo 3
    cost23 = cdist(pts_logo2_matched, pts_logo3)
    _, p23 = linear_sum_assignment(cost23)
    pts_logo3_matched = pts_logo3[p23]
    
    print("Optimal transport matching between logos complete.")
    
    print("[5/6] Building SVG Paths and Animations...")
    # Helper to encode binary grid into path runs:
    def encode_path_runs(dots_matrix):
        # returns list of (y, x_start, length)
        runs = []
        h_m, w_m = dots_matrix.shape
        for y in range(h_m):
            in_run = False
            x_start = 0
            for x in range(w_m):
                if dots_matrix[y, x]:
                    if not in_run:
                        in_run = True
                        x_start = x
                else:
                    if in_run:
                        in_run = False
                        runs.append((y, x_start, x - x_start))
            if in_run:
                runs.append((y, x_start, w_m - x_start))
        return runs

    def runs_to_svg_d(runs_list):
        parts = []
        for y, x, length in runs_list:
            if length == 1:
                parts.append(f"M{x},{y}h1v1h-1Z")
            else:
                parts.append(f"M{x},{y}h{length}v1h-{length}Z")
        return "".join(parts)

    def generate_svg(is_dark):
        theme_name = "dark" if is_dark else "light"
        dots_matrix = dots_dark if is_dark else dots_light
        
        # Colors:
        # Dark: portrait #A78BFA, UI chrome #22D3EE, accent #10B981, bg #0A101F
        # Light: portrait #7C3AED, UI chrome #0891B2, accent #059669, bg #F8FAFC
        bg_col = "#0A101F" if is_dark else "#F8FAFC"
        card_bg = "#0D1527" if is_dark else "#FFFFFF"
        border_col = "#1E293B" if is_dark else "#E2E8F0"
        header_bar_bg = "#162036" if is_dark else "#F1F5F9"
        title_col = "#94A3B8" if is_dark else "#64748B"
        ui_chrome = "#22D3EE" if is_dark else "#0891B2"
        portrait_col = "#A78BFA" if is_dark else "#7C3AED"
        accent_col = "#10B981" if is_dark else "#059669"
        text_val = "#E2E8F0" if is_dark else "#0F172A"
        text_label = "#22D3EE" if is_dark else "#0891B2"
        leader_col = "#334155" if is_dark else "#CBD5E1"
        
        # Get all runs
        all_runs = encode_path_runs(dots_matrix)
        total_runs = len(all_runs)
        print(f"[{theme_name}] Total path runs: {total_runs}")
        
        # 1. INTRO LAYER (~3.2s once, 60 interleaved random groups):
        np.random.seed(42)
        perm = np.random.permutation(total_runs)
        group_assignments = perm % 60
        
        # Verify evenness metric (2-sample KS test):
        run_ys = np.array([r[0] for r in all_runs])
        run_xs = np.array([r[1] for r in all_runs])
        ks_scores = []
        for g in range(60):
            idx = (group_assignments == g)
            ks_x = ks_2samp(run_xs[idx], run_xs).statistic
            ks_y = ks_2samp(run_ys[idx], run_ys).statistic
            ks_scores.append((ks_x + ks_y) / 2)
        evenness_metric = np.mean(ks_scores)
        print(f"[{theme_name}] Intro evenness metric (KS test): {evenness_metric:.4f} (target: ~0.05)")
        
        intro_groups_svg = []
        for g in range(60):
            g_runs = [all_runs[i] for i in range(total_runs) if group_assignments[i] == g]
            d_str = runs_to_svg_d(g_runs)
            begin_t = g * (1.6 / 60.0)
            intro_groups_svg.append(
                f'<g opacity="0">\n'
                f'  <path d="{d_str}" fill="{portrait_col}" shape-rendering="crispEdges"/>\n'
                f'  <animate attributeName="opacity" values="0;1" dur="0.4s" begin="{begin_t:.2f}s" fill="freeze"/>\n'
                f'</g>'
            )
        
        intro_layer = (
            f'<g id="intro-layer">\n'
            f'  <set attributeName="visibility" to="hidden" begin="3.2s" fill="freeze"/>\n'
            + "\n".join(intro_groups_svg) +
            f'\n</g>'
        )
        
        # 2. LOOP LAYER (~94 drift bands with sigma=4 per-dot noise):
        noise = np.random.normal(0, 4.0, total_runs)
        dist_to_center = np.sqrt((run_xs - cx)**2 + (run_ys - cy)**2)
        sort_metric = dist_to_center + 0.3 * run_ys + noise
        band_bins = np.percentile(sort_metric, np.linspace(0, 100, 95))
        band_assignments = np.clip(np.digitize(sort_metric, band_bins) - 1, 0, 93)
        
        print(f"[{theme_name}] Assigned {len(np.unique(band_assignments))} drift bands with sigma=4.0 noise.")
        
        # Loop Keyframes:
        # Loop begins at 3.2s:
        # t=0..3.0s: hold portrait (opacity 1, translate 0 0)
        # t=3.0..4.3s: drift 42% toward (cx, cy) and fade (opacity 1 -> 0)
        # t=4.3..12.9s: hidden (opacity 0)
        # t=12.9..14.2s: return from 42% offset back to (0,0) and fade in (opacity 0 -> 1)
        key_times = "0; 0.2113; 0.3028; 0.9085; 1"
        op_values = "1; 1; 0; 0; 1"
        
        loop_bands_svg = []
        for b in range(94):
            b_runs = [all_runs[i] for i in range(total_runs) if band_assignments[i] == b]
            if not b_runs:
                continue
            d_str = runs_to_svg_d(b_runs)
            bx = np.mean([r[1] for r in b_runs])
            by = np.mean([r[0] for r in b_runs])
            
            dx = 0.42 * (cx - bx)
            dy = 0.42 * (cy - by)
            
            trans_values = f"0 0; 0 0; {dx:.1f} {dy:.1f}; {dx:.1f} {dy:.1f}; 0 0"
            
            loop_bands_svg.append(
                f'<g>\n'
                f'  <path d="{d_str}" fill="{portrait_col}" shape-rendering="crispEdges"/>\n'
                f'  <animate attributeName="opacity" values="{op_values}" keyTimes="{key_times}" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>\n'
                f'  <animateTransform attributeName="transform" type="translate" values="{trans_values}" keyTimes="{key_times}" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>\n'
                f'</g>'
            )
            
        loop_layer = (
            f'<g id="loop-layer" visibility="hidden">\n'
            f'  <set attributeName="visibility" to="visible" begin="3.2s" fill="freeze"/>\n'
            + "\n".join(loop_bands_svg) +
            f'\n</g>'
        )
        
        # 3. TRAVELLERS LAYER (~900 dots morphing across Python -> </> -> React):
        traveller_keytimes = "0; 0.2113; 0.3028; 0.4437; 0.5352; 0.6761; 0.7676; 0.9085; 1"
        traveller_op_vals = "0; 0; 1; 1; 1; 1; 1; 1; 0"
        
        travellers_svg = []
        for i in range(N_TRAVELLERS):
            p1 = pts_logo1[i]
            p2 = pts_logo2_matched[i]
            p3 = pts_logo3_matched[i]
            
            cx_vals = f"{p1[0]:.1f}; {p1[0]:.1f}; {p1[0]:.1f}; {p1[0]:.1f}; {p2[0]:.1f}; {p2[0]:.1f}; {p3[0]:.1f}; {p3[0]:.1f}; {p3[0]:.1f}"
            cy_vals = f"{p1[1]:.1f}; {p1[1]:.1f}; {p1[1]:.1f}; {p1[1]:.1f}; {p2[1]:.1f}; {p2[1]:.1f}; {p3[1]:.1f}; {p3[1]:.1f}; {p3[1]:.1f}"
            
            travellers_svg.append(
                f'<circle r="1.4" fill="{portrait_col}">\n'
                f'  <animate attributeName="cx" values="{cx_vals}" keyTimes="{traveller_keytimes}" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>\n'
                f'  <animate attributeName="cy" values="{cy_vals}" keyTimes="{traveller_keytimes}" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>\n'
                f'  <animate attributeName="opacity" values="{traveller_op_vals}" keyTimes="{traveller_keytimes}" dur="14.2s" begin="3.2s" repeatCount="indefinite"/>\n'
                f'</circle>'
            )
            
        travellers_group = (
            f'<g id="travellers-layer" visibility="hidden">\n'
            f'  <set attributeName="visibility" to="visible" begin="3.2s" fill="freeze"/>\n'
            + "\n".join(travellers_svg) +
            f'\n</g>'
        )
        
        # 4. INFO PANEL ROWS (SYSTEM.INFO readout):
        info_rows = [
            ("Subject", "Harish M"),
            ("Role", "Aspiring Full-Stack &amp; ML Engineer"),
            ("Origin", "Chennai, India"),
            ("Education", "B.Tech AIML"),
            ("Status", "Learning + Building + Shipping"),
            ("ToolChain", "VS Code, Git, GitHub"),
            ("Core.Lang", "Python, Java, SQL, JavaScript, C"),
            ("Core.Frontend", "HTML, CSS, React"),
            ("Core.Backend", "Python, Flask, Node.js, REST APIs"),
            ("Core.Database", "MySQL, PostgreSQL, SQL"),
            ("Core.Infra", "Git, GitHub, GitHub Actions"),
            ("Grid.Mail", "harish.m.2024.aiml@rajalakshmi.edu.in"),
            ("Grid.Portfolio", "coming soon"),
            ("Grid.LinkedIn", "in/harish-m-488273318"),
            ("Grid.Instagram", "@harizzz__19"),
            ("Grid.GitHub", "@HARISH241501066")
        ]
        
        rows_svg = []
        start_y = 158
        row_spacing = 25
        panel_x_start = 455
        panel_x_end = 1135
        
        # Monospace width approx 8.4px at font-size 14
        char_w = 8.4
        
        for idx, (label, val_clean) in enumerate(info_rows):
            cur_y = start_y + idx * row_spacing
            
            # Unescaped length for character measurement
            raw_val = val_clean.replace("&amp;", "&")
            label_len_px = len(label) * char_w
            val_len_px = len(raw_val) * char_w
            
            leader_x1 = panel_x_start + label_len_px + 12
            leader_x2 = panel_x_end - val_len_px - 12
            
            # Value locking textLength
            val_text_len = f'{val_len_px:.1f}'
            
            rows_svg.append(
                f'<g class="info-row">\n'
                f'  <text x="{panel_x_start}" y="{cur_y}" font-family="ui-monospace, SFMono-Regular, \'SF Mono\', Menlo, Consolas, monospace" font-size="14" fill="{text_label}" font-weight="600">{label}</text>\n'
                f'  <line x1="{leader_x1:.1f}" y1="{cur_y - 4}" x2="{leader_x2:.1f}" y2="{cur_y - 4}" stroke="{leader_col}" stroke-width="1.5" stroke-dasharray="2 5" stroke-linecap="round"/>\n'
                f'  <text x="{panel_x_end}" y="{cur_y}" text-anchor="end" font-family="ui-monospace, SFMono-Regular, \'SF Mono\', Menlo, Consolas, monospace" font-size="14" fill="{text_val}" textLength="{val_text_len}" lengthAdjust="spacingAndGlyphs">{val_clean}</text>\n'
                f'</g>'
            )
            
        all_rows_str = "\n".join(rows_svg)
        
        # 5. ASSEMBLE FULL SVG (1180x610)
        svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 610" width="1180" height="610">
  <defs>
    <style>
      .terminal-title {{ font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace; font-size: 13px; font-weight: 500; fill: {title_col}; }}
      .section-hdr {{ font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace; font-size: 13px; font-weight: 700; fill: {ui_chrome}; letter-spacing: 1.5px; }}
      .pill-text {{ font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace; font-size: 14px; font-weight: 700; fill: {accent_col}; }}
      .live-text {{ font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace; font-size: 12px; font-weight: 700; fill: #EF4444; letter-spacing: 1px; }}
    </style>
    <!-- Live pulse animation -->
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
  </defs>

  <!-- Terminal Window Background -->
  <rect x="0" y="0" width="1180" height="610" rx="14" fill="{bg_col}"/>
  <rect x="0.5" y="0.5" width="1179" height="609" rx="13.5" fill="none" stroke="{border_col}" stroke-width="1"/>

  <!-- Title Bar -->
  <path d="M0 14 C0 6.27 6.27 0 14 0 L1166 0 C1173.73 0 1180 6.27 1180 14 L1180 42 L0 42 Z" fill="{header_bar_bg}"/>
  <line x1="0" y1="42" x2="1180" y2="42" stroke="{border_col}" stroke-width="1"/>

  <!-- Window Controls -->
  <circle cx="24" cy="21" r="6" fill="#EF4444"/>
  <circle cx="44" cy="21" r="6" fill="#F59E0B"/>
  <circle cx="64" cy="21" r="6" fill="#10B981"/>

  <!-- Window Title -->
  <text x="590" y="26" text-anchor="middle" class="terminal-title">profile.sh --live</text>

  <!-- Left Frame: VISUAL.MAP (~38% of canvas) -->
  <g transform="translate(42, 60)">
    <!-- Header -->
    <text x="10" y="24" class="section-hdr">VISUAL.MAP</text>
    <line x1="120" y1="20" x2="350" y2="20" stroke="{border_col}" stroke-width="1"/>

    <!-- Reticle Frame surrounding portrait -->
    <rect x="5" y="38" width="345" height="485" rx="8" fill="{card_bg}" stroke="{border_col}" stroke-width="1"/>
    
    <!-- Corner brackets -->
    <path d="M5 58 L5 38 L25 38" fill="none" stroke="{ui_chrome}" stroke-width="2"/>
    <path d="M330 38 L350 38 L350 58" fill="none" stroke="{ui_chrome}" stroke-width="2"/>
    <path d="M5 503 L5 523 L25 523" fill="none" stroke="{ui_chrome}" stroke-width="2"/>
    <path d="M330 523 L350 523 L350 503" fill="none" stroke="{ui_chrome}" stroke-width="2"/>

    <!-- Portrait 300x340 positioned inside frame -->
    <g transform="translate(28, 80) scale(1.0, 1.0)">
      {intro_layer}
      {loop_layer}
      {travellers_group}
    </g>

    <!-- Subtitle badge below portrait -->
    <g transform="translate(28, 440)">
      <rect x="0" y="0" width="290" height="24" rx="4" fill="{header_bar_bg}" stroke="{border_col}" stroke-width="1"/>
      <text x="145" y="16" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, monospace" font-size="11" fill="{title_col}">1-BIT FLOYD-STEINBERG DITHER // SERPENTINE</text>
    </g>
  </g>

  <!-- Right Panel: SYSTEM.INFO Readout -->
  <g transform="translate(0, 0)">
    <!-- Section Header & Badges -->
    <g transform="translate(455, 96)">
      <text x="0" y="0" class="section-hdr">SYSTEM.INFO</text>
      
      <!-- Pulsing Red LIVE Badge -->
      <g transform="translate(130, -5)">
        <rect x="0" y="-12" width="65" height="20" rx="10" fill="#EF4444" fill-opacity="0.15" stroke="#EF4444" stroke-width="1"/>
        <circle cx="12" cy="-2" r="4" fill="#EF4444">
          <animate attributeName="r" values="3.5;5;3.5" dur="1.5s" repeatCount="indefinite"/>
          <animate attributeName="opacity" values="1;0.4;1" dur="1.5s" repeatCount="indefinite"/>
        </circle>
        <text x="24" y="2" class="live-text">LIVE</text>
      </g>

      <!-- Colored Pill with Handle -->
      <g transform="translate(480, -7)">
        <rect x="0" y="-12" width="195" height="24" rx="12" fill="{accent_col}" fill-opacity="0.15" stroke="{accent_col}" stroke-width="1.2"/>
        <text x="97" y="4" text-anchor="middle" class="pill-text">@HARISH241501066</text>
      </g>
    </g>

    <!-- Header Divider -->
    <line x1="455" y1="120" x2="1135" y2="120" stroke="{border_col}" stroke-width="1"/>

    <!-- 16 Data Rows with Dotted Leaders & Locked Right-Aligned Values -->
    {all_rows_str}
  </g>
</svg>'''
        
        out_path = os.path.join(output_dir, f"{theme_name}.svg")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        size_kb = os.path.getsize(out_path) / 1024.0
        print(f"Generated {theme_name}.svg successfully! File size: {size_kb:.1f} KB")

    print("[6/6] Writing dark.svg and light.svg...")
    generate_svg(is_dark=True)
    generate_svg(is_dark=False)
    print("All banners generated successfully!")

if __name__ == '__main__':
    generate_banners()
