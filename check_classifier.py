"""
Script to inspect PixelPaws classifier file contents
"""

import pickle
import sys

def check_classifier(pkl_file):
    """Check what's in a PixelPaws classifier file"""
    
    print("=" * 60)
    print("PIXELPAWS CLASSIFIER INSPECTION")
    print("=" * 60)
    print(f"\nFile: {pkl_file}\n")
    
    try:
        with open(pkl_file, 'rb') as f:
            clf = pickle.load(f)
        
        if not isinstance(clf, dict):
            print(f"❌ ERROR: Expected dict, got {type(clf)}")
            return
        
        print("✓ Loaded successfully\n")
        print("-" * 60)
        print("ALL KEYS IN CLASSIFIER:")
        print("-" * 60)
        for key in clf.keys():
            print(f"  - {key}")
        
        print("\n" + "=" * 60)
        print("FEATURE EXTRACTION CONFIGURATION")
        print("=" * 60)
        
        # Brightness features
        bp_pixbrt = clf.get('bp_pixbrt_list')
        print(f"\nbp_pixbrt_list: {bp_pixbrt}")
        if bp_pixbrt:
            print(f"  → Type: {type(bp_pixbrt)}")
            print(f"  → Length: {len(bp_pixbrt)}")
            if len(bp_pixbrt) > 0:
                print(f"  → Body parts: {bp_pixbrt}")
        else:
            print(f"  → Type: {type(bp_pixbrt)}")
        
        # Square sizes
        square_size = clf.get('square_size')
        print(f"\nsquare_size: {square_size}")
        if square_size:
            print(f"  → Type: {type(square_size)}")
        
        # Pixel threshold
        pix_threshold = clf.get('pix_threshold')
        print(f"\npix_threshold: {pix_threshold}")
        if pix_threshold:
            print(f"  → Type: {type(pix_threshold)}")
        
        # Body parts for pose
        bp_include = clf.get('bp_include_list')
        print(f"\nbp_include_list: {bp_include}")
        if bp_include:
            print(f"  → Type: {type(bp_include)}")
            if isinstance(bp_include, list) and len(bp_include) > 0:
                print(f"  → Body parts: {bp_include}")
        
        print("\n" + "=" * 60)
        print("MODEL INFORMATION")
        print("=" * 60)
        
        if 'clf_model' in clf:
            model = clf['clf_model']
            print(f"\nModel type: {type(model).__name__}")
            
            if hasattr(model, 'n_features_in_'):
                n_features = model.n_features_in_
                print(f"Expected features (n_features_in_): {n_features}")
                
                # Estimate breakdown
                if bp_pixbrt and len(bp_pixbrt) > 0:
                    # Each body part adds: 1 individual + pairwise ratios
                    n_bp = len(bp_pixbrt)
                    brightness_features = n_bp + (n_bp * (n_bp - 1) // 2)
                    # Add velocity features (doubles)
                    brightness_features *= 2
                    
                    pose_features = n_features - brightness_features
                    
                    print(f"\nEstimated breakdown:")
                    print(f"  → Pose features: ~{pose_features}")
                    print(f"  → Brightness features: ~{brightness_features}")
                    print(f"     ({n_bp} body parts × 2 (base + velocity))")
                else:
                    print(f"\nAll {n_features} features are pose features (no brightness)")
        
        print("\n" + "=" * 60)
        print("DIAGNOSIS")
        print("=" * 60)
        
        if not bp_pixbrt or (isinstance(bp_pixbrt, list) and len(bp_pixbrt) == 0):
            print("\n❌ PROBLEM FOUND!")
            print("   bp_pixbrt_list is EMPTY or None")
            print("\n   This means:")
            print("   - Feature extraction will only extract POSE features")
            print("   - No brightness features will be extracted")
            print("   - Features will be 363 instead of 375")
            print("\n   Solution:")
            print("   - Your classifier was trained WITH brightness features")
            print("   - But it didn't save which body parts to use")
            print("   - You need to either:")
            print("     1. Re-train and make sure bp_pixbrt_list is saved")
            print("     2. Manually specify in Active Learning settings")
        else:
            print("\n✓ Configuration looks good!")
            print(f"   Brightness features will be extracted for: {bp_pixbrt}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR loading classifier: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_classifier.py <path_to_classifier.pkl>")
        sys.exit(1)
    
    check_classifier(sys.argv[1])
