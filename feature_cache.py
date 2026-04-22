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
    POSE_FEATURE_VERSION = 5

try:
    from brightness_features import BRIGHTNESS_FEATURE_VERSION
except ImportError:
    BRIGHTNESS_FEATURE_VERSION = 1


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
            'brightness_feature_version': int(BRIGHTNESS_FEATURE_VERSION),
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
        """Write a DataFrame (or any picklable object) atomically, plus a
        sidecar `.version.json` describing the feature versions that produced
        this cache.  The sidecar is best-effort — a save failure there never
        aborts the primary pickle write, and missing sidecars are tolerated
        by all readers (fall back to column-name heuristics).
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

        # Sidecar — records pose & brightness versions alongside the pkl.
        # Readers tolerate its absence.
        try:
            import json
            from datetime import datetime as _dt
            sidecar = {
                'pose_feature_version':       int(POSE_FEATURE_VERSION),
                'brightness_feature_version': int(BRIGHTNESS_FEATURE_VERSION),
                'saved_at':                   _dt.now().isoformat(timespec='seconds'),
                'column_count': (int(df.shape[1])
                                 if hasattr(df, 'shape') and len(df.shape) == 2 else None),
                'row_count':    (int(df.shape[0])
                                 if hasattr(df, 'shape') and len(df.shape) >= 1 else None),
            }
            sidecar_path = target_path + '.version.json'
            with open(sidecar_path, 'w', encoding='utf-8') as f:
                json.dump(sidecar, f, indent=2)
        except Exception:
            pass   # never abort on sidecar failure

    @staticmethod
    def load_cache_meta(target_path: str) -> dict:
        """Read the `.version.json` sidecar next to a feature cache pkl.

        Returns a dict with keys:
          pose_feature_version, brightness_feature_version, saved_at,
          column_count, row_count
        — all None when the sidecar is missing or unreadable.
        """
        sidecar_path = target_path + '.version.json'
        if not os.path.isfile(sidecar_path):
            return {
                'pose_feature_version':       None,
                'brightness_feature_version': None,
                'saved_at':                   None,
                'column_count': None, 'row_count': None,
            }
        try:
            import json
            with open(sidecar_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {
                'pose_feature_version':       None,
                'brightness_feature_version': None,
                'saved_at':                   None,
                'column_count': None, 'row_count': None,
            }

    @staticmethod
    def check_feature_versions(cache_path: str) -> dict:
        """Inspect a feature cache and report its version state.

        Prefers the sidecar `.version.json`.  Falls back to column-name
        heuristics when the sidecar is missing (older caches).  Returns a
        dict with: pose_version (int|None), brightness_version (int|None),
        current_pose_version, current_brightness_version, up_to_date (bool),
        source ('sidecar' | 'column-heuristic' | 'none').
        """
        meta = FeatureCacheManager.load_cache_meta(cache_path)
        pose_v = meta.get('pose_feature_version')
        brt_v  = meta.get('brightness_feature_version')
        source = 'sidecar' if pose_v is not None else 'none'

        if pose_v is None and os.path.isfile(cache_path):
            # Column-heuristic fallback — works for bare-DataFrame caches
            try:
                with open(cache_path, 'rb') as f:
                    X = pickle.load(f)
                if isinstance(X, dict) and 'X' in X:
                    X = X['X']
                cols = list(X.columns) if hasattr(X, 'columns') else []
                if any('_PreQuiescence' in c for c in cols):
                    pose_v = 5
                elif any('_VelAsymmetry' in c for c in cols):
                    pose_v = 4
                elif any('_Jerk1' in c for c in cols):
                    pose_v = 3
                elif any('_Vel1' in c for c in cols):
                    pose_v = 2
                else:
                    pose_v = None  # pre-v2 or unknown
                # Brightness versioning was introduced at v1; assume v1 if Pix_* cols exist
                if brt_v is None and any(c.startswith('Pix_') for c in cols):
                    brt_v = 1
                source = 'column-heuristic'
            except Exception:
                pass

        return {
            'pose_version':                 pose_v,
            'brightness_version':           brt_v,
            'current_pose_version':         int(POSE_FEATURE_VERSION),
            'current_brightness_version':   int(BRIGHTNESS_FEATURE_VERSION),
            'pose_up_to_date':       (pose_v == int(POSE_FEATURE_VERSION)) if pose_v is not None else False,
            'brightness_up_to_date': (brt_v  == int(BRIGHTNESS_FEATURE_VERSION)) if brt_v  is not None else False,
            'saved_at':                     meta.get('saved_at'),
            'source':                       source,
        }

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

    # ------------------------------------------------------------------
    # Version upgrade (v3 → v4)
    # ------------------------------------------------------------------

    @staticmethod
    def try_upgrade_v3_to_v4(old_path: str, target_path: str,
                             cfg: dict, pose_path: str,
                             log_fn=None) -> pd.DataFrame | None:
        """Attempt an incremental upgrade from a v3 cache (missing flinch features).

        If the old cache has Jerk columns but no VelAsymmetry columns, extracts
        only the missing v4 flinch features from the DLC file (no video re-read).

        Returns the upgraded DataFrame on success, or ``None`` on failure.
        The upgraded cache is saved atomically to *target_path*.
        """
        _log = log_fn or (lambda msg: None)
        try:
            with open(old_path, 'rb') as fh:
                old_X = pickle.load(fh)
            has_v3 = any('_Jerk1' in c for c in old_X.columns)
            needs_v4 = not any('_VelAsymmetry' in c for c in old_X.columns)
            if not (has_v3 and needs_v4):
                return None

            _log("  [Cache] Found v3 cache → upgrading to v4 (no video re-read)")
            from pose_features import PoseFeatureExtractor
            upg_extractor = PoseFeatureExtractor(
                bodyparts=cfg.get('bp_include_list') or [],
                likelihood_threshold=cfg.get('pix_threshold', 0.8),
                velocity_delta=cfg.get('dt_vel', 2),
            )
            new_feats = upg_extractor.extract_v4_features_only(pose_path)
            if len(new_feats) > len(old_X):
                new_feats = new_feats.iloc[:len(old_X)].reset_index(drop=True)

            if len(new_feats) == len(old_X):
                X_full = pd.concat(
                    [old_X, new_feats.reset_index(drop=True)], axis=1)
                FeatureCacheManager.save_cache(X_full, target_path)
                _log(f"  ✓ Cache upgraded v3→v4: +{len(new_feats.columns)} flinch columns → {target_path}")
                return X_full
            else:
                _log(f"  [Cache] Row count mismatch ({len(old_X)} vs {len(new_feats)}) — full extraction needed")
                return None
        except Exception as e:
            _log(f"  [Cache] v3→v4 upgrade attempt failed: {e}")
            return None

    @staticmethod
    def try_upgrade_v4_to_v5(old_path: str, target_path: str,
                             cfg: dict, pose_path: str,
                             log_fn=None) -> pd.DataFrame | None:
        """Attempt an incremental upgrade from a v4 cache (missing v5 temporal-context features).

        If the old cache has VelAsymmetry columns but no PreQuiescence columns,
        extracts only the four new v5 columns per body part (PreQuiescence,
        Jy_signed, OnsetSharpness, HFEnergy) from the DLC file — no video re-read.

        Returns the upgraded DataFrame on success, or ``None`` on failure.
        The upgraded cache is saved atomically to *target_path*.
        """
        _log = log_fn or (lambda msg: None)
        try:
            with open(old_path, 'rb') as fh:
                old_X = pickle.load(fh)
            has_v4 = any('_VelAsymmetry' in c for c in old_X.columns)
            needs_v5 = not any('_PreQuiescence' in c for c in old_X.columns)
            if not (has_v4 and needs_v5):
                return None

            _log("  [Cache] Found v4 cache → upgrading to v5 (no video re-read)")
            from pose_features import PoseFeatureExtractor
            upg_extractor = PoseFeatureExtractor(
                bodyparts=cfg.get('bp_include_list') or [],
                likelihood_threshold=cfg.get('pix_threshold', 0.8),
                velocity_delta=cfg.get('dt_vel', 2),
            )
            new_feats = upg_extractor.extract_v5_features_only(pose_path)
            if len(new_feats) > len(old_X):
                new_feats = new_feats.iloc[:len(old_X)].reset_index(drop=True)

            if len(new_feats) == len(old_X):
                X_full = pd.concat(
                    [old_X, new_feats.reset_index(drop=True)], axis=1)
                FeatureCacheManager.save_cache(X_full, target_path)
                _log(f"  ✓ Cache upgraded v4→v5: +{len(new_feats.columns)} temporal-context columns → {target_path}")
                return X_full
            else:
                _log(f"  [Cache] Row count mismatch ({len(old_X)} vs {len(new_feats)}) — full extraction needed")
                return None
        except Exception as e:
            _log(f"  [Cache] v4→v5 upgrade attempt failed: {e}")
            return None
