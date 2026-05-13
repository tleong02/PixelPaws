"""
feature_cache.py — Centralised Feature Cache Management
========================================================
Single source of truth for:
  - Feature hash computation (config → 8-char hex)
  - Cache file search across the 7-directory fallback hierarchy
  - Atomic cache writes
  - Version upgrade (v2 → v3, adding kinematics without video re-read)

All other modules should import from here instead of reimplementing
the cache logic locally.
"""

import os
import glob
import hashlib
import pickle
import tempfile

import pandas as pd

# ---------------------------------------------------------------------------
# Pose feature version — authoritative constant
# ---------------------------------------------------------------------------
try:
    from pose_features import POSE_FEATURE_VERSION
except ImportError:
    POSE_FEATURE_VERSION = 3


# ---------------------------------------------------------------------------
# FeatureCacheManager
# ---------------------------------------------------------------------------

class FeatureCacheManager:
    """Static helpers for finding, loading, saving, and upgrading feature caches."""

    # Expose version so callers don't need their own import
    POSE_FEATURE_VERSION = POSE_FEATURE_VERSION

    # ------------------------------------------------------------------
    # Hash computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(cfg: dict) -> str:
        """MD5 hash of a normalised config dict → 8-char hex string.

        Uses explicit type coercion so the key is identical regardless of
        whether tk.BooleanVar.get() returns Python bool or int.
        """
        key_dict = {
            'bp_include_list':      cfg.get('bp_include_list'),
            'bp_pixbrt_list':       list(cfg.get('bp_pixbrt_list', [])),
            'square_size':          [int(x) for x in cfg.get('square_size', [])],
            'pix_threshold':        round(float(cfg.get('pix_threshold', 0.3)), 6),
            'pose_feature_version': int(POSE_FEATURE_VERSION),
            'include_optical_flow': bool(cfg.get('include_optical_flow', False)),
            'bp_optflow_list':      list(cfg.get('bp_optflow_list', [])),
        }
        return hashlib.md5(repr(key_dict).encode('utf-8')).hexdigest()[:8]

    # ------------------------------------------------------------------
    # Fallback directory search
    # ------------------------------------------------------------------

    @staticmethod
    def _build_search_dirs(canonical_dir: str, video_dir: str,
                           project_root: str = None) -> list:
        """Return an ordered, de-duplicated list of directories to search.

        Order:
          1. canonical_dir         (<project>/features/)
          2. video_dir             (same folder as the video)
          3. video_dir/features/
          4. video_dir/FeatureCache/
          5. video_dir/PredictionCache/
          6+ ancestor walk up to project_root:
               <ancestor>/features/  and  <ancestor>/FeatureCache/
        """
        dirs = [
            canonical_dir,
            video_dir,
            os.path.join(video_dir, 'features'),
            os.path.join(video_dir, 'FeatureCache'),
            os.path.join(video_dir, 'PredictionCache'),
        ]
        # Ancestor walk
        _stop = os.path.normpath(project_root) if project_root else None
        _ancestor = video_dir
        while True:
            _parent = os.path.dirname(_ancestor)
            if _parent == _ancestor:
                break
            _ancestor = _parent
            dirs.append(os.path.join(_ancestor, 'features'))
            dirs.append(os.path.join(_ancestor, 'FeatureCache'))
            if _stop and os.path.normpath(_ancestor) == _stop:
                break

        # De-duplicate while preserving order
        seen = set()
        unique = []
        for d in dirs:
            nd = os.path.normpath(d)
            if nd not in seen:
                seen.add(nd)
                unique.append(d)
        return unique

    @staticmethod
    def find_cache(session_name: str, cfg_hash: str,
                   canonical_dir: str, video_dir: str,
                   project_root: str = None) -> str | None:
        """Search canonical + fallback locations for an exact-hash cache file.

        Returns the full path if found, or ``None``.
        """
        fname = f"{session_name}_features_{cfg_hash}.pkl"
        for d in FeatureCacheManager._build_search_dirs(
                canonical_dir, video_dir, project_root):
            candidate = os.path.join(d, fname)
            if os.path.isfile(candidate):
                return candidate
        return None

    @staticmethod
    def find_any_cache(session_name: str, canonical_dir: str,
                       video_dir: str,
                       project_root: str = None) -> str | None:
        """Like *find_cache* but ignores the hash — returns the newest match.

        Useful as a last-resort fallback (caller should log a hash-mismatch
        warning).
        """
        pattern = f"{session_name}_features_*.pkl"
        for d in FeatureCacheManager._build_search_dirs(
                canonical_dir, video_dir, project_root):
            if not os.path.isdir(d):
                continue
            matches = glob.glob(os.path.join(d, pattern))
            if matches:
                plain = [m for m in matches if '_corrected' not in m]
                return sorted(plain or matches, key=os.path.getmtime)[-1]
        return None

    # ------------------------------------------------------------------
    # Atomic save
    # ------------------------------------------------------------------

    @staticmethod
    def save_cache(df, target_path: str) -> None:
        """Write a DataFrame (or any picklable object) atomically.

        Writes to a temp file in the same directory, then renames.
        """
        dir_path = os.path.dirname(target_path) or '.'
        os.makedirs(dir_path, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
        try:
            with os.fdopen(tmp_fd, 'wb') as f:
                pickle.dump(df, f)
            os.replace(tmp_path, target_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Version upgrade (v2 → v3)
    # ------------------------------------------------------------------

    @staticmethod
    def try_upgrade_v2_to_v3(old_path: str, target_path: str,
                             cfg: dict, pose_path: str,
                             log_fn=None) -> pd.DataFrame | None:
        """Attempt an incremental upgrade from a v2 cache (missing kinematics).

        If the old cache has velocity columns but no Jerk columns, extracts
        only the missing kinematics from the DLC file (no video re-read).

        Returns the upgraded DataFrame on success, or ``None`` on failure.
        The upgraded cache is saved atomically to *target_path*.
        """
        _log = log_fn or (lambda msg: None)
        try:
            with open(old_path, 'rb') as fh:
                old_X = pickle.load(fh)
            has_base = any('_Vel1' in c for c in old_X.columns)
            needs_v3 = not any('_Jerk1' in c for c in old_X.columns)
            if not (has_base and needs_v3):
                return None

            _log("  [Cache] Found v2 cache → upgrading to v3 (no video re-read)")
            from pose_features import PoseFeatureExtractor
            upg_extractor = PoseFeatureExtractor(
                bodyparts=cfg.get('bp_include_list') or [],
                likelihood_threshold=cfg.get('pix_threshold', 0.8),
                velocity_delta=cfg.get('dt_vel', 2),
            )
            new_feats = upg_extractor.extract_new_kinematics_only(pose_path)
            if len(new_feats) > len(old_X):
                new_feats = new_feats.iloc[:len(old_X)].reset_index(drop=True)

            if len(new_feats) == len(old_X):
                X_full = pd.concat(
                    [old_X, new_feats.reset_index(drop=True)], axis=1)
                FeatureCacheManager.save_cache(X_full, target_path)
                _log(f"  ✓ Cache upgraded: +{len(new_feats.columns)} new columns → {target_path}")
                return X_full
            else:
                _log(f"  [Cache] Row count mismatch ({len(old_X)} vs {len(new_feats)}) — full extraction needed")
                return None
        except Exception as e:
            _log(f"  [Cache] Upgrade attempt failed: {e}")
            return None
