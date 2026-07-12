import sys
import os
import argparse
import numpy as np
import torch
from scipy.signal import resample

# Ensure ecgtizer is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ECGTIZER_REPO = os.path.join(BASE_DIR, 'ecgtizer')
if os.path.isdir(ECGTIZER_REPO) and ECGTIZER_REPO not in sys.path:
    sys.path.insert(0, ECGTIZER_REPO)

try:
    from ecgtizer import ECGtizer
    from ecgtizer.PDF2XML import LEAD_TIME_3X4, LEAD_TIME_6X2
except ImportError as e:
    print(f"Error importing ECGtizer. Please ensure the ecgtizer package is in {ECGTIZER_REPO}")
    print(e)
    sys.exit(1)

def ecgtizer_to_ptbxl(extracted_leads, target_samples=1000, layout="3x4"):
    """
    Convert ECGtizer extracted leads to PTB-XL compatible numpy array.
    
    Uses Tiling with a Manual Phase Shift logic to construct a 10-second signal.
    """
    ecgtizer_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    signal = np.zeros((target_samples, 12), dtype=np.float32)
    lead_time_map = LEAD_TIME_3X4 if layout == "3x4" else LEAD_TIME_6X2
    
    for col_idx, lead_name in enumerate(ecgtizer_names):
        if lead_name not in extracted_leads:
            print(f'  Warning: Lead {lead_name} not found')
            continue
        
        raw = np.array(extracted_leads[lead_name])
        t_start, t_end = lead_time_map[lead_name]
        
        segment = raw[t_start:t_end]
        segment_mv = segment / 1000.0  # uV to mV
        
        orig_start = t_start // 5
        orig_end = t_end // 5
        target_len = orig_end - orig_start
        
        if len(segment_mv) > 0 and target_len > 0:
            resampled = resample(segment_mv, target_len)
        else:
            resampled = np.zeros(target_len)
            
        # Tiling with a Manual Phase Shift
        if len(resampled) > 0:
            tiled = np.zeros(target_samples, dtype=np.float32)
            
            # 1. Place the first tile normally
            tiled[0:target_len] = resampled
            
            # --- PHASE SHIFT SETTING ---
            shift_seconds = 0.3  # Push the second tile 0.3 seconds to the right
            shift_samples = int(shift_seconds * 100) # 100Hz = 100 samples/sec
            
            # 2. Place the second tile shifted to the right
            start_idx = target_len + shift_samples
            if start_idx < target_samples:
                space_left = target_samples - start_idx
                tiled[start_idx:] = resampled[:space_left]
                
                # 3. Fill the gap with a smooth flat baseline
                v_start = tiled[target_len - 1]
                v_end = tiled[start_idx]
                gap_size = (start_idx + 1) - (target_len - 1)
                tiled[target_len - 1 : start_idx + 1] = np.linspace(v_start, v_end, gap_size)
            
            signal[:, col_idx] = tiled
            
    return signal

def save_as_pt(signal, filepath):
    """Save signal as PyTorch tensor file in (12, 1000) format."""
    tensor = torch.tensor(signal, dtype=torch.float32)
    
    if tensor.shape == (1000, 12):
        tensor = tensor.T
        
    torch.save(tensor, filepath)
    print(f'Saved tensor to: {filepath}')
    print(f'Shape: {tensor.shape}, dtype: {tensor.dtype}')

def main():
    parser = argparse.ArgumentParser(description="Digitize an ECG image/PDF to a .pt tensor")
    parser.add_argument("input_file", help="Path to the input ECG image or PDF")
    parser.add_argument("--output", "-o", help="Path to save the output .pt file. Defaults to <input_file>.pt", default=None)
    parser.add_argument("--layout", "-l", choices=["3x4", "6x2"], default="3x4", help="ECG layout format (3x4 or 6x2)")
    parser.add_argument("--dpi", "-d", type=int, default=300, help="DPI for extraction")
    parser.add_argument("--method", "-m", choices=["full", "fragmented", "lazy"], default="full", help="Extraction method")
    
    args = parser.parse_args()
    
    input_path = args.input_file
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)
        
    output_path = args.output
    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}.pt"
        
    print(f"Processing '{input_path}'...")
    print(f"Layout: {args.layout}, DPI: {args.dpi}, Method: {args.method}")
    
    # Run ECGtizer
    ecg = ECGtizer(
        file=input_path,
        dpi=args.dpi,
        extraction_method=args.method,
        verbose=True,
    )
    
    if not ecg.good:
        print("Warning: ECGtizer reported extraction was not completely successful (IsGood: False)")
        
    leads = ecg.extracted_lead
    if not isinstance(leads, dict) or len(leads) == 0:
        print("Error: No leads were extracted. Ensure the image is high resolution and the layout matches.")
        sys.exit(1)
        
    print(f"Extracted {len(leads)} leads.")
    
    # Convert to PTB-XL format
    print("Converting to PTB-XL format with tiling...")
    signal = ecgtizer_to_ptbxl(leads, target_samples=1000, layout=args.layout)
    
    # Save as .pt
    save_as_pt(signal, output_path)

if __name__ == "__main__":
    main()
