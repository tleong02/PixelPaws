"""
render_skeleton_video.py — Render a DLC pose file as a skeleton video on a black background.

Paw body parts: bright pixels from the source video ROI are colourised and stamped
onto the canvas. Other body parts: filled circles. A fading ghost trail accumulates
over time. Stick-figure skeleton lines are drawn on the current frame.

Usage:
    python render_skeleton_video.py <h5_file> <video_file> [options]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SKELETON = [
    ('snout',     'neck'),
    ('neck',      'centroid'),
    ('centroid',  'tailbase'),
    ('tailbase',  'tailtip'),
    ('neck',      'frpaw'),
    ('neck',      'flpaw'),
    ('centroid',  'hrpaw'),
    ('centroid',  'hlpaw'),
]

# BGR colour palettes
COLORWAYS = {
    'default': {
        # gold hrpaw, magenta hlpaw, cyan frpaw, yellow-green flpaw
        'hrpaw':    (220, 210,   0),
        'hlpaw':    (200,   0, 200),
        'frpaw':    (  0, 155, 255),
        'flpaw':    (160, 210,   0),
        'snout':    (  0, 220, 220),
        'neck':     (180, 180, 180),
        'centroid': (120, 120, 120),
        'tailbase': ( 70, 120, 240),
        'tailtip':  ( 90, 150, 255),
    },
    'redblue': {
        # right paws = warm (red/orange), left paws = cool (blue/cyan)
        'hrpaw':    (  0,  80, 240),
        'hlpaw':    (200,  60,   0),
        'frpaw':    ( 20, 140, 255),
        'flpaw':    (220, 100,  20),
        'snout':    (200, 200, 200),
        'neck':     (160, 160, 160),
        'centroid': (100, 100, 100),
        'tailbase': (140, 140, 180),
        'tailtip':  (160, 160, 200),
    },
    'neon': {
        # fully saturated, high-contrast
        'hrpaw':    (  0, 255, 255),
        'hlpaw':    (255,   0, 255),
        'frpaw':    (255, 255,   0),
        'flpaw':    (  0, 255, 128),
        'snout':    (128, 255, 255),
        'neck':     (255, 255, 255),
        'centroid': (200, 200, 200),
        'tailbase': (200, 200, 255),
        'tailtip':  (220, 220, 255),
    },
    'pastel': {
        # soft, low-saturation tints
        'hrpaw':    (140, 180, 210),
        'hlpaw':    (180, 130, 200),
        'frpaw':    (210, 190, 130),
        'flpaw':    (130, 195, 155),
        'snout':    (170, 215, 215),
        'neck':     (195, 195, 195),
        'centroid': (155, 155, 155),
        'tailbase': (165, 175, 210),
        'tailtip':  (175, 185, 220),
    },
    'mono': {
        # greyscale — distinct shades per paw
        'hrpaw':    (220, 220, 220),
        'hlpaw':    (165, 165, 165),
        'frpaw':    (200, 200, 200),
        'flpaw':    (145, 145, 145),
        'snout':    (210, 210, 210),
        'neck':     (150, 150, 150),
        'centroid': (110, 110, 110),
        'tailbase': ( 90,  90,  90),
        'tailtip':  (120, 120, 120),
    },
}
DEFAULT_COLORS = COLORWAYS['default']   # backward-compat alias
FALLBACK_COLOR = (255, 255, 255)
SKELETON_COLOR = (100, 110, 100)  # dim green-grey


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Render DLC skeleton video on a black background with paw-pixel stamps.'
    )
    p.add_argument('h5_file',    help='DLC .h5 pose file')
    p.add_argument('video_file', help='Source video (.mp4 / .avi / …)')
    p.add_argument('--output',     metavar='PATH',   default=None,
                   help='Output .mp4 path (default: <h5_stem>_skeleton.mp4 beside h5 file)')
    p.add_argument('--decay',      type=float, default=0.82,
                   help='Trail decay factor 0–1 (default: 0.82)')
    p.add_argument('--likelihood',    type=float, default=0.3,
                   help='Min DLC likelihood to draw a body part (default: 0.3)')
    p.add_argument('--threshold',     type=float, default=0.40,
                   help='Pixel brightness threshold 0–1 (default: 0.40)')
    p.add_argument('--size',          type=int,   default=15,
                   help='Half-width of forepaw ROI in pixels (default: 15)')
    p.add_argument('--hindpaw-size',  type=int,   default=40,
                   help='Half-width of hindpaw ROI in pixels (default: 40)')
    p.add_argument('--glow',          type=float, default=0.2,
                   help='Glow blend strength after Gaussian blur (default: 0.2)')
    p.add_argument('--glow-sigma',    type=float, default=2.0,
                   help='Gaussian blur sigma for glow — smaller = crisper (default: 2.0)')
    p.add_argument('--trail-interval', type=int,   default=10,
                   help='Frames between hindpaw trail stamps (default: 10)')
    p.add_argument('--trail-decay',    type=float, default=0.993,
                   help='Per-frame fade of hindpaw trail (default: 0.993)')
    p.add_argument('--grey-paws',       action='store_true',
                   help='Render paw stamps in natural video colours instead of body-part tint')
    p.add_argument('--colorway',        choices=list(COLORWAYS), default='default',
                   help='Colour palette for body parts (default: default)')
    p.add_argument('--export-original', action='store_true',
                   help='Also write a cropped/trimmed original video alongside the skeleton output')
    p.add_argument('--no-glow',     action='store_true',
                   help='Disable the glow/bloom effect entirely')
    p.add_argument('--no-trail',    action='store_true',
                   help='Disable hindpaw footprint trail')
    p.add_argument('--no-skeleton', action='store_true',
                   help='Disable skeleton lines')
    for _bp in ('hrpaw', 'hlpaw', 'frpaw', 'flpaw'):
        p.add_argument(f'--color-{_bp}', type=str, default=None, metavar='B,G,R',
                       help=f'Override colour for {_bp} as B,G,R (0-255 each)')
    p.add_argument('--crop',        metavar='X1,Y1,X2,Y2', default=None,
                   help='Crop region (skips interactive selection)')
    p.add_argument('--start-frame', type=int,   default=None, metavar='N',
                   help='First frame to render (0-based, default: 0)')
    p.add_argument('--end-frame',   type=int,   default=None, metavar='N',
                   help='Last frame (exclusive, default: all)')
    p.add_argument('--start-time',  type=float, default=None, metavar='SEC',
                   help='Start time in seconds (overrides --start-frame)')
    p.add_argument('--end-time',    type=float, default=None, metavar='SEC',
                   help='End time in seconds (overrides --end-frame)')
    p.add_argument('--bout-file',    metavar='PATH', default=None,
                   help='Predictions CSV — renders one clip per behavior bout')
    p.add_argument('--bout-column',  metavar='COL',  default=None,
                   help='Column to use from --bout-file (default: auto-detect)')
    p.add_argument('--bout-padding', type=int, default=30,
                   help='Frames to include before/after each bout (default: 30)')
    p.add_argument('--min-bout-frames', type=int, default=10,
                   help='Skip bouts shorter than this many frames (default: 10)')
    p.add_argument('--label-bouts', action='store_true',
                   help='Overlay behavior label on frames during active bouts')
    p.add_argument('--label-text', metavar='TEXT', default=None,
                   help='Custom label text (default: "<column> detected")')
    return p.parse_args()


# ---------------------------------------------------------------------------
# DLC data loading  (matches pose_features.py:67-79)
# ---------------------------------------------------------------------------

def load_h5(h5_file: str):
    df = pd.read_hdf(h5_file)
    df.columns = pd.MultiIndex.from_tuples([(c[1], c[2]) for c in df.columns])
    data = {}
    for bp in df.columns.get_level_values(0).unique():
        data[bp] = (
            df[bp]['x'].values.astype(np.float32),
            df[bp]['y'].values.astype(np.float32),
            df[bp]['likelihood'].values.astype(np.float32),
        )
    return data


# ---------------------------------------------------------------------------
# Interactive crop selection
# ---------------------------------------------------------------------------

def select_crop(cap: cv2.VideoCapture, crop_arg: str | None):
    """
    Returns (x1, y1, x2, y2) or None (= full frame).
    If --crop is given, parse it; otherwise use the full frame.
    """
    if crop_arg is None:
        print('[crop] No --crop supplied — using full frame.')
        return None

    parts = [int(v.strip()) for v in crop_arg.split(',')]
    if len(parts) != 4:
        sys.exit('--crop must be X1,Y1,X2,Y2')
    x1, y1, x2, y2 = parts
    print(f'[crop] Using supplied crop: ({x1},{y1})→({x2},{y2})')
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Per-frame rendering helpers
# ---------------------------------------------------------------------------

def stamp_paw_pixels(canvas: np.ndarray, frame: np.ndarray,
                     x: int, y: int,
                     color_bgr: tuple, half: int, threshold: float,
                     crop_region, grey: bool = False):
    """
    Extract ROI from video frame, keep bright pixels, tint them, stamp onto canvas.
    Coordinates x,y are already in canvas space (crop-offset applied by caller).
    frame is the cropped video frame (same spatial extent as canvas).
    If grey=True, preserve natural video pixel colours instead of applying color_bgr tint.
    """
    H, W = canvas.shape[:2]
    x1 = max(0, x - half);  x2 = min(W, x + half)
    y1 = max(0, y - half);  y2 = min(H, y + half)
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    roi_f = roi.astype(np.float32) / 255.0          # normalise 0→1
    mask  = roi_f.max(axis=2) >= threshold            # bright pixels
    if grey:
        colored = roi_f * mask[:, :, None]             # natural video colours
    else:
        color_f = np.array(color_bgr, dtype=np.float32) / 255.0
        colored = roi_f * mask[:, :, None] * color_f  # body-part tint

    canvas[y1:y2, x1:x2] += colored * 255.0


def draw_skeleton(canvas: np.ndarray, positions: dict,
                  likelihoods: dict, likelihood_thresh: float):
    """Draw dim skeleton lines onto a uint8 canvas."""
    for bp_a, bp_b in DEFAULT_SKELETON:
        if bp_a not in positions or bp_b not in positions:
            continue
        if likelihoods.get(bp_a, 0) < likelihood_thresh:
            continue
        if likelihoods.get(bp_b, 0) < likelihood_thresh:
            continue
        xa, ya = positions[bp_a]
        xb, yb = positions[bp_b]
        cv2.line(canvas, (xa, ya), (xb, yb), SKELETON_COLOR, 1, cv2.LINE_AA)


def _draw_label(frame: np.ndarray, text: str):
    """Overlay white text with a dark shadow for readability on any background."""
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.0
    thick = 2
    x, y  = 20, 50
    cv2.putText(frame, text, (x+2, y+2), font, scale, (0, 0, 0),   thick+2, cv2.LINE_AA)
    cv2.putText(frame, text, (x,   y),   font, scale, (255, 255, 255), thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Bout parsing
# ---------------------------------------------------------------------------

def load_bouts(csv_path: str, column, min_frames: int = 1):
    """
    Parse a predictions CSV and return:
      ([(start_frame, end_frame), ...], col_name_used)
    Bouts shorter than min_frames are skipped.
    """
    df = pd.read_csv(csv_path)

    # Auto-detect column
    if column:
        col = column
    else:
        candidates = [c for c in df.columns if c not in ('frame', 'probability')]
        col = 'prediction' if 'prediction' in df.columns else (candidates[0] if candidates else None)
        if col is None:
            sys.exit(f'[bouts] No prediction column found in {csv_path}')
    if col not in df.columns:
        sys.exit(f'[bouts] Column "{col}" not found in {csv_path}')

    print(f'[bouts] Using column "{col}"  ({int(df[col].sum())} bout frames total)')

    # Build sorted frame → prediction dict
    if 'frame' in df.columns:
        pred = dict(zip(df['frame'].astype(int), df[col].astype(int)))
    else:
        pred = {i: int(v) for i, v in enumerate(df[col])}

    if not pred:
        return [], col

    max_f = max(pred)
    arr = np.array([pred.get(i, 0) for i in range(max_f + 1)], dtype=np.int8)

    # Find contiguous runs of 1s
    bouts = []
    in_bout = False
    for i, v in enumerate(arr):
        if v and not in_bout:
            start = i
            in_bout = True
        elif not v and in_bout:
            if i - start >= min_frames:
                bouts.append((start, i))
            in_bout = False
    if in_bout and max_f + 1 - start >= min_frames:
        bouts.append((start, max_f + 1))

    print(f'[bouts] {len(bouts)} bouts found (min_frames={min_frames})')
    return bouts, col


# ---------------------------------------------------------------------------
# Per-clip render helper
# ---------------------------------------------------------------------------

def _render_clip(cap, data, body_parts, colors, args,
                 start_frame, end_frame, out_path,
                 out_w, out_h, cx1, cy1, fps, fourcc, crop,
                 bout_idx, total_bouts, video_path=None,
                 writer=None, label=None, bout_active=(None, None),
                 label_mask=None):
    """Render one clip from start_frame to end_frame.

    If *writer* is supplied the frames are appended to it and it is NOT
    released on return (caller owns it).  Otherwise a new VideoWriter is
    created for *out_path* and released before returning.
    """
    n_render = end_frame - start_frame
    own_writer = writer is None
    if own_writer:
        print(f'[render] Output: {out_path}  ({out_w}x{out_h} @ {fps:.1f} fps)')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            sys.exit(f'Cannot open VideoWriter for: {out_path}')
    print(f'[render] Frames {start_frame}-{end_frame}  ({n_render} frames, '
          f'{start_frame/fps:.1f}s-{end_frame/fps:.1f}s)')
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Accumulator (float32, 0–255 range)
    accum = np.zeros((out_h, out_w, 3), dtype=np.float32)

    # Separate slow-decay accumulator for hindpaw footprint trail
    trail_accum   = np.zeros((out_h, out_w, 3), dtype=np.float32)
    trail_counter = 0

    print('[render] Rendering …')
    for frame_idx in range(start_frame, end_frame):
        ok, bgr_full = cap.read()
        if not ok:
            print(f'[warn] Video ended at frame {frame_idx}')
            break

        # Crop video frame to canvas size
        bgr = bgr_full[cy1:cy1+out_h, cx1:cx1+out_w]

        # Current-frame canvas (float32, 0–255 range)
        temp = np.zeros((out_h, out_w, 3), dtype=np.float32)

        # Collect positions and likelihoods for this frame
        positions   = {}   # bp -> (x_px, y_px) in canvas coords
        likelihoods = {}   # bp -> float

        for bp in body_parts:
            xs, ys, ls = data[bp]
            lk = float(ls[frame_idx])
            likelihoods[bp] = lk
            if lk < args.likelihood:
                continue
            # Raw coordinates → canvas coordinates (subtract crop offset)
            cx = int(round(float(xs[frame_idx]))) - cx1
            cy = int(round(float(ys[frame_idx]))) - cy1
            # Skip if outside canvas
            if not (0 <= cx < out_w and 0 <= cy < out_h):
                continue
            positions[bp] = (cx, cy)

        # Draw body parts
        for bp, (cx, cy) in positions.items():
            color_bgr = colors.get(bp, FALLBACK_COLOR)

            if bp.endswith('paw'):
                half = args.hindpaw_size if bp in ('hlpaw', 'hrpaw') else args.size
                stamp_paw_pixels(
                    temp, bgr, cx, cy,
                    color_bgr, half, args.threshold,
                    crop, grey=args.grey_paws
                )
            else:
                # Filled circle
                color_f = np.array(color_bgr, dtype=np.float32)
                cv2.circle(temp, (cx, cy), 5, color_f.tolist(), -1, cv2.LINE_AA)

        # Glow pass
        if not args.no_glow:
            _ks = max(1, int(args.glow_sigma * 3)) | 1
            glow = cv2.GaussianBlur(temp, (_ks, _ks), args.glow_sigma)
            temp = np.clip(temp + glow * args.glow, 0.0, 255.0)

        # Accumulate with decay
        accum = accum * args.decay + temp

        # Hindpaw trail — stamp every trail_interval frames, decay every frame
        if not args.no_trail:
            trail_accum *= args.trail_decay
            trail_counter += 1
            if trail_counter >= args.trail_interval:
                trail_counter = 0
                for bp in ('hlpaw', 'hrpaw'):
                    if bp in positions:
                        tcx, tcy = positions[bp]
                        color_bgr = colors.get(bp, FALLBACK_COLOR)
                        stamp_paw_pixels(
                            trail_accum, bgr, tcx, tcy,
                            color_bgr, args.hindpaw_size, args.threshold,
                            crop, grey=args.grey_paws
                        )

        # Write frame — trail behind, skeleton on top (neither accumulates)
        trail_layer = trail_accum if not args.no_trail else np.zeros_like(accum)
        out_frame = np.clip(trail_layer + accum, 0, 255).astype(np.uint8)
        if not args.no_skeleton:
            draw_skeleton(out_frame, positions, likelihoods, args.likelihood)
        if label:
            b0, b1 = bout_active
            if b0 is not None:
                show_label = (b0 <= frame_idx < b1)         # bout mode: range check
            elif label_mask is not None:
                show_label = (frame_idx in label_mask)       # clip mode: per-frame check
            else:
                show_label = True                            # no mask: always show
            if show_label:
                _draw_label(out_frame, label)
        writer.write(out_frame)

        done = frame_idx - start_frame + 1
        if done % 100 == 0 or frame_idx == end_frame - 1:
            bout_tag = f'bout {bout_idx}/{total_bouts}  ' if total_bouts > 1 else ''
            print(f'   {bout_tag}frame {done}/{n_render}  ({100*done/n_render:.0f}%)', flush=True)

    if own_writer:
        writer.release()
        print(f'\n[done] Wrote {out_path}')

    # ── Optional: export cropped/trimmed original video ───────────────────────
    if own_writer and args.export_original and video_path is not None:
        orig_path = out_path.with_name(out_path.stem.replace('_skeleton', '') + '_original.mp4')
        print(f'[orig] Writing original video to {orig_path} …')
        orig_cap = cv2.VideoCapture(str(video_path))
        orig_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        orig_writer = cv2.VideoWriter(str(orig_path), fourcc, fps, (out_w, out_h))
        for fi in range(start_frame, end_frame):
            ok, fr = orig_cap.read()
            if not ok:
                break
            orig_writer.write(fr[cy1:cy1+out_h, cx1:cx1+out_w])
        orig_cap.release()
        orig_writer.release()
        print(f'[orig] Wrote {orig_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    h5_path    = Path(args.h5_file)
    video_path = Path(args.video_file)

    if not h5_path.exists():
        sys.exit(f'h5 file not found: {h5_path}')
    if not video_path.exists():
        sys.exit(f'Video file not found: {video_path}')

    # Default output path
    if args.output is None:
        out_path = h5_path.with_name(h5_path.stem + '_skeleton.mp4')
    else:
        out_path = Path(args.output)

    # Load pose data
    print(f'[load] Reading pose data from {h5_path.name} …')
    data = load_h5(str(h5_path))
    body_parts = list(data.keys())
    n_frames_pose = len(next(iter(data.values()))[0])
    print(f'       Body parts: {body_parts}')
    print(f'       Pose frames: {n_frames_pose}')

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f'Cannot open video: {video_path}')

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resolve colour palette + apply per-paw overrides
    colors = dict(COLORWAYS[args.colorway])
    for _bp in ('hrpaw', 'hlpaw', 'frpaw', 'flpaw'):
        raw = getattr(args, f'color_{_bp}', None)
        if raw:
            try:
                vals = tuple(int(v) for v in raw.split(','))
                if len(vals) == 3:
                    colors[_bp] = vals
            except ValueError:
                print(f'[warn] Invalid --color-{_bp} value "{raw}" — ignored')

    # Interactive crop
    crop = select_crop(cap, args.crop)

    if crop is not None:
        cx1, cy1, cx2, cy2 = crop
        cx1 = max(0, cx1);  cy1 = max(0, cy1)
        cx2 = min(vid_w, cx2);  cy2 = min(vid_h, cy2)
        out_w = cx2 - cx1
        out_h = cy2 - cy1
    else:
        cx1, cy1 = 0, 0
        out_w, out_h = vid_w, vid_h

    # Compute start / end frames from time or frame args
    _total_avail = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), n_frames_pose)
    start_frame = int(args.start_time * fps) if args.start_time is not None \
                  else (args.start_frame or 0)
    end_frame   = int(args.end_time   * fps) if args.end_time   is not None \
                  else (args.end_frame if args.end_frame is not None else _total_avail)
    start_frame = max(0, min(start_frame, _total_avail))
    end_frame   = max(start_frame, min(end_frame, _total_avail))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Bout mode only when --bout-file is given AND no explicit clip range.
    # If start/end frames are also set (e.g. from a suggestion), treat as
    # single-clip but still use the bout-file to resolve the label text.
    _is_bout_mode = args.bout_file and not (
        args.start_frame is not None or args.start_time is not None or
        args.end_frame   is not None or args.end_time   is not None
    )

    if _is_bout_mode:
        # ── Bout mode: all bouts concatenated into one output video ───────────
        bouts, col_used = load_bouts(args.bout_file, args.bout_column,
                                     args.min_bout_frames)
        if not bouts:
            sys.exit('[bouts] No bouts found — nothing to render.')

        total_bouts = len(bouts)
        print(f'[bouts] Writing {total_bouts} bout clip(s) to: {out_path}')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            sys.exit(f'Cannot open VideoWriter for: {out_path}')

        label = (args.label_text or f'{col_used} detected') if args.label_bouts else None

        for bi, (b_start, b_end) in enumerate(bouts, 1):
            clip_start = max(0, b_start - args.bout_padding)
            clip_end   = min(_total_avail, b_end + args.bout_padding)
            print(f'\n[bouts] Bout {bi}/{total_bouts}: '
                  f'frames {b_start}-{b_end}  (clip {clip_start}-{clip_end})')
            _render_clip(cap, data, body_parts, colors, args,
                         clip_start, clip_end, out_path,
                         out_w, out_h, cx1, cy1, fps, fourcc, crop,
                         bi, total_bouts, video_path,
                         writer=writer,
                         label=label, bout_active=(b_start, b_end))

        writer.release()
        print(f'\n[bouts] Done -- {total_bouts} bout clip(s) written to {out_path}')
    else:
        # ── Normal mode (single-clip render) ──────────────────────────────────
        # Resolve label + per-frame mask from predictions CSV (if supplied)
        label = None
        label_mask = None
        if args.label_bouts:
            if args.label_text:
                label = args.label_text
            if args.bout_file:
                bouts_sc, col_used = load_bouts(args.bout_file, args.bout_column, 1)
                if not args.label_text:
                    label = f'{col_used} detected'
                # Build set of active frames that fall within this clip
                label_mask = set()
                for bs, be in bouts_sc:
                    for f in range(max(bs, start_frame), min(be, end_frame)):
                        label_mask.add(f)
                if not label_mask:
                    label_mask = None  # no active frames → don't draw at all
        _render_clip(cap, data, body_parts, colors, args,
                     start_frame, end_frame, out_path,
                     out_w, out_h, cx1, cy1, fps, fourcc, crop,
                     1, 1, video_path,
                     label=label, label_mask=label_mask)

    cap.release()


if __name__ == '__main__':
    main()
