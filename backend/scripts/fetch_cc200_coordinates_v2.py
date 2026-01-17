"""
Script to fetch real CC200 coordinates from nilearn and save to JSON file.
This version uses find_parcellation_cut_coords which is more reliable.
"""

import json
import numpy as np
from nilearn import datasets
from nilearn import plotting
from nilearn.image import load_img
import os

# #region agent log
log_path = r'd:\workplace\ASDModelPredict\.cursor\debug.log'
try:
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"location": "fetch_cc200_coordinates_v2.py:15", "message": "Script started", "data": {}, "timestamp": int(__import__('time').time() * 1000), "sessionId": "debug-session", "runId": "run1", "hypothesisId": "A"}) + '\n')
except:
    pass
# #endregion

def determine_lobe_and_hemisphere(x, y, z):
    """Determine lobe and hemisphere based on coordinates."""
    # Determine hemisphere
    if x < -5:
        hemisphere = 'left'
    elif x > 5:
        hemisphere = 'right'
    else:
        hemisphere = 'midline'
    
    # Determine lobe based on coordinates
    # Frontal: y > 0, z > 0 generally
    # Parietal: y < -20, z > 20 generally
    # Temporal: y > -50, z < 20 generally
    # Occipital: y < -60, z around 0
    # Subcortical: |x| < 30, |y| < 30, |z| < 30
    # Cerebellum: y < -40, z < -20
    
    if abs(x) < 30 and abs(y) < 30 and abs(z) < 30:
        lobe = 'subcortical'
    elif y < -40 and z < -20:
        lobe = 'cerebellum'
    elif y < -60:
        lobe = 'occipital'
    elif y > 0 and z > 0:
        lobe = 'frontal'
    elif y < -20 and z > 20:
        lobe = 'parietal'
    elif y > -50 and z < 20:
        lobe = 'temporal'
    else:
        # Default based on y coordinate
        if y > 0:
            lobe = 'frontal'
        elif y < -60:
            lobe = 'occipital'
        elif z < 0:
            lobe = 'temporal'
        else:
            lobe = 'parietal'
    
    return lobe, hemisphere

def main():
    # #region agent log
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"location": "fetch_cc200_coordinates_v2.py:65", "message": "Fetching atlas", "data": {}, "timestamp": int(__import__('time').time() * 1000), "sessionId": "debug-session", "runId": "run1", "hypothesisId": "A"}) + '\n')
    except:
        pass
    # #endregion
    
    print("Fetching Craddock 2012 atlas (CC200)...")
    
    # Disable SSL verification for development
    import ssl
    import urllib.request
    import os
    import warnings
    import requests
    
    # Suppress SSL warnings
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    
    # Disable SSL verification
    original_ssl_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Also set environment variable for urllib3
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    
    try:
        # Monkey patch requests to disable SSL verification
        original_request = requests.Session.request
        def patched_request(self, *args, **kwargs):
            kwargs.setdefault('verify', False)
            return original_request(self, *args, **kwargs)
        requests.Session.request = patched_request
        
        # Try to fetch the atlas
        try:
            atlas = datasets.fetch_atlas_craddock_2012()
        except Exception as e:
            print(f"Warning: Could not fetch atlas automatically: {e}")
            print("Please download the atlas manually or check your internet connection.")
            return
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"location": "fetch_cc200_coordinates_v2.py:105", "message": "Atlas fetched", "data": {"atlas_keys": list(atlas.keys()) if hasattr(atlas, 'keys') else "N/A"}, "timestamp": int(__import__('time').time() * 1000), "sessionId": "debug-session", "runId": "run1", "hypothesisId": "A"}) + '\n')
        except:
            pass
        # #endregion
        
        # Load the CC200 atlas image
        # Try different possible keys
        atlas_filename = None
        possible_keys = ['scorr_mean', 'scorr_05', 'scorr_10', 'scorr_15', 'scorr_20', 'scorr_25', 'tcorr_mean']
        
        for key in possible_keys:
            if hasattr(atlas, key) and atlas[key] is not None:
                atlas_filename = atlas[key]
                break
            elif isinstance(atlas, dict) and key in atlas and atlas[key] is not None:
                atlas_filename = atlas[key]
                break
        
        if atlas_filename is None:
            print("Error: Could not find CC200 atlas file in the downloaded data.")
            if hasattr(atlas, 'keys'):
                print(f"Available keys: {list(atlas.keys())}")
            return
        
        print(f"Loading atlas from: {atlas_filename}")
        labels_img = load_img(atlas_filename)
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"location": "fetch_cc200_coordinates_v2.py:130", "message": "Atlas loaded", "data": {"shape": labels_img.shape}, "timestamp": int(__import__('time').time() * 1000), "sessionId": "debug-session", "runId": "run1", "hypothesisId": "A"}) + '\n')
        except:
            pass
        # #endregion
        
        # Use find_parcellation_cut_coords to get center coordinates
        print("Calculating region centroids...")
        try:
            coords, labels = plotting.find_parcellation_cut_coords(
                labels_img, 
                return_label_names=True
            )
        except Exception as e:
            print(f"Error using find_parcellation_cut_coords: {e}")
            print("Falling back to manual calculation...")
            # Fallback: manual calculation
            data = labels_img.get_fdata()
            affine = labels_img.affine
            unique_labels = np.unique(data)
            unique_labels = unique_labels[unique_labels > 0]
            
            coords = []
            labels = []
            for label in sorted(unique_labels):
                mask = data == label
                if not np.any(mask):
                    continue
                inds = np.array(np.where(mask))
                # Convert voxel indices to MNI coordinates
                from nibabel.affines import apply_affine
                mni_coords = apply_affine(affine, inds.T)
                centroid = mni_coords.mean(axis=0)
                coords.append(centroid)
                labels.append(int(label))
            coords = np.array(coords)
            labels = np.array(labels)
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"location": "fetch_cc200_coordinates_v2.py:165", "message": "Coordinates calculated", "data": {"num_regions": len(coords), "first_coord": coords[0].tolist() if len(coords) > 0 else None}, "timestamp": int(__import__('time').time() * 1000), "sessionId": "debug-session", "runId": "run1", "hypothesisId": "A"}) + '\n')
        except:
            pass
        # #endregion
        
        print(f"Found {len(coords)} brain regions")
        
        # Convert to our format
        regions = []
        for idx, (label, coord) in enumerate(zip(labels, coords)):
            x, y, z = coord[0], coord[1], coord[2]
            lobe, hemisphere = determine_lobe_and_hemisphere(x, y, z)
            
            regions.append({
                "id": int(label) - 1,  # Convert to 0-based indexing
                "name": f"Region {int(label)}",
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "lobe": lobe,
                "hemisphere": hemisphere
            })
        
        # Sort by ID to ensure consistent ordering
        regions.sort(key=lambda r: r['id'])
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"location": "fetch_cc200_coordinates_v2.py:195", "message": "Regions processed", "data": {"total_regions": len(regions), "first_region": regions[0] if regions else None}, "timestamp": int(__import__('time').time() * 1000), "sessionId": "debug-session", "runId": "run1", "hypothesisId": "A"}) + '\n')
        except:
            pass
        # #endregion
        
        # Save to JSON file
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cc200_coordinates.json')
        frontend_public_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'web', 'public', 'cc200_coordinates.json')
        
        print(f"Saving coordinates to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(regions, f, indent=2, ensure_ascii=False)
        
        print(f"Saving coordinates to frontend public folder: {frontend_public_path}")
        os.makedirs(os.path.dirname(frontend_public_path), exist_ok=True)
        with open(frontend_public_path, 'w', encoding='utf-8') as f:
            json.dump(regions, f, indent=2, ensure_ascii=False)
        
        print(f"\nSuccessfully saved {len(regions)} CC200 regions to JSON files")
        print(f"  - Backend: {output_path}")
        print(f"  - Frontend: {frontend_public_path}")
        
        # Print sample coordinates
        print("\nSample regions:")
        for region in regions[:5]:
            print(f"  ID {region['id']}: {region['name']} at ({region['x']:.1f}, {region['y']:.1f}, {region['z']:.1f}) mm - {region['lobe']} ({region['hemisphere']})")
        
    finally:
        # Restore original SSL context
        ssl._create_default_https_context = original_ssl_context
        if 'PYTHONHTTPSVERIFY' in os.environ:
            del os.environ['PYTHONHTTPSVERIFY']

if __name__ == '__main__':
    main()
