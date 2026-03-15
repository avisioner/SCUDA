"""
process.py (Optimized)
──────────────────────
Takes raw CSV output from the XIAO and produces a fully labeled,
resampled dataset where every row is one segment with exactly 100 timesteps.

Improvements:
- Properly discards short segments (doesn't output frames from discarded segments)
- Distinguishes STABLE vs TRANSITIONING based on sensor data
- Handles "rest" as a special unlabeled state until explicitly labeled
- Filters consecutive rest frames into one "rest segment" for reference

Usage:
    python process.py raw_data.csv labeled_data.csv
"""

import sys
import csv
import numpy as np
from collections import defaultdict

# ─── Configuration ────────────────────────────────────────────────────────────
N_SAMPLES = 100  # resample every segment to this many timesteps

SENSOR_COLS = [
    "real", "i", "j", "k",
    "accel_x", "accel_y", "accel_z",
    "gyro_x",  "gyro_y",  "gyro_z",
    "linear_x", "linear_y", "linear_z",
    "stability"
]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def resample(values, n=N_SAMPLES):
    """Resample a list of floats to exactly n points via linear interpolation."""
    values = np.array(values, dtype=float)
    if len(values) == 1:
        return np.full(n, values[0])
    original = np.linspace(0, 1, len(values))
    target   = np.linspace(0, 1, n)
    return np.interp(target, original, values)

def resample_timestamps(timestamps, n=N_SAMPLES):
    """Resample timestamps to n evenly spaced points between first and last."""
    t = np.array(timestamps, dtype=float)
    if len(t) == 1:
        return np.full(n, t[0])
    return np.linspace(t[0], t[-1], n)

def load_raw(filepath):
    """
    Load raw CSV, skip comment lines (starting with #).
    Returns dict of seg_id -> list of row dicts.
    Also detect which segments were marked as discarded in serial output.
    """
    segments = defaultdict(list)
    headers  = []
    discarded_segments = set()

    with open(filepath, newline="") as f:
        for line in f:
            line = line.strip()
            
            # Track discarded segments from comments
            if line.startswith("# Segment") and "DISCARDED" in line:
                # Parse: "# Segment 5 DISCARDED (too short)"
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        seg_id = int(parts[2])
                        discarded_segments.add(seg_id)
                    except:
                        pass
            
            if not line or line.startswith("#"):
                continue
            if line.startswith("seg_id"):
                headers = line.split(",")
                continue
            if not headers:
                continue
            parts = line.split(",")
            if len(parts) != len(headers):
                continue
            row    = dict(zip(headers, parts))
            seg_id = int(row["seg_id"])
            segments[seg_id].append(row)

    return segments, discarded_segments

def build_output_header():
    """Build the full CSV header for the output file."""
    cols = ["seg_id", "seg_type", "from_position", "to_position", "label"]
    cols += [f"t_{n}" for n in range(N_SAMPLES)]
    for ch in SENSOR_COLS:
        cols += [f"{ch}_{n}" for n in range(N_SAMPLES)]
    return cols

def infer_segment_type(rows):
    """
    Infer segment type based on stability values in the rows.
    Returns 'S' for mostly STABLE or 'T' for mostly TRANSITIONING.
    """
    stabilities = [int(r["stability"]) for r in rows]
    
    # Count stability types: 0=unknown, 1=stable, 2=on_the_move
    stable_count = sum(1 for s in stabilities if s == 1)
    transitioning_count = sum(1 for s in stabilities if s == 2)
    
    total = len(stabilities)
    if total == 0:
        return 'S'
    
    # If more than 50% of frames are transitioning, mark as T
    if transitioning_count > (total * 0.5):
        return 'T'
    else:
        return 'S'

def process_segment(seg_id, seg_type, from_position, to_position, label, rows):
    """
    Given a list of raw rows for one segment, resample all channels
    to N_SAMPLES and return a flat dict ready to write as a CSV row.
    """
    timestamps = [float(r["timestamp"]) for r in rows]

    # Normalise timestamps so every segment starts at 0
    t0         = timestamps[0]
    timestamps = [t - t0 for t in timestamps]

    resampled_t = resample_timestamps(timestamps)

    result = {
        "seg_id":        seg_id,
        "seg_type":      seg_type,
        "from_position": from_position,
        "to_position":   to_position,
        "label":         label,
    }

    # Resample each sensor channel
    for ch in SENSOR_COLS:
        values   = [float(r[ch]) for r in rows]
        resampled = resample(values)
        for n, v in enumerate(resampled):
            result[f"{ch}_{n}"] = round(float(v), 6)

    # Resampled timestamps
    for n, v in enumerate(resampled_t):
        result[f"t_{n}"] = round(float(v), 2)

    return result

def print_segment_summary(seg_id, rows):
    """Print a brief summary of a segment to help the user label it."""
    duration = float(rows[-1]["timestamp"]) - float(rows[0]["timestamp"])
    n_frames = len(rows)

    energies = []
    for r in rows:
        gx = float(r["gyro_x"])
        gy = float(r["gyro_y"])
        gz = float(r["gyro_z"])
        energies.append(np.sqrt(gx**2 + gy**2 + gz**2))

    avg_energy = np.mean(energies)
    max_energy = np.max(energies)
    
    # Get positions from first row (constant during segment)
    from_position = rows[0].get("from_position", "NONE")
    to_position = rows[0].get("to_position", "NONE")
    
    # Infer type
    inferred_type = infer_segment_type(rows)
    stabilities = [int(r["stability"]) for r in rows]
    stable_count = sum(1 for s in stabilities if s == 1)
    transitioning_count = sum(1 for s in stabilities if s == 2)

    print(f"\n{'─'*60}")
    print(f"  Segment {seg_id}  |  {n_frames} frames  |  {duration:.0f}ms")
    
    # Show trajectory or position
    if inferred_type == 'T':
        print(f"  Trajectory: {from_position} → {to_position}")
    else:
        print(f"  Position: {to_position}")
    
    print(f"  Gyro energy — avg: {avg_energy:.3f}   max: {max_energy:.3f}")
    print(f"  Stability: {stable_count} STABLE  |  {transitioning_count} TRANSITIONING")
    print(f"  Inferred type: {inferred_type}")
    print(f"{'─'*60}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 3:
        print("Usage: python process.py raw_data.csv labeled_data.csv")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    print(f"\nLoading {input_path}...")
    segments, discarded_segments = load_raw(input_path)

    if not segments:
        print("No segments found. Make sure the CSV has data rows.")
        sys.exit(1)

    # Separate rest frames (seg_id=0) from actual segments
    seg_ids = sorted([sid for sid in segments.keys() if sid != 0])
    print(f"Found {len(seg_ids)} segments to label.")
    if discarded_segments:
        print(f"Discarded segments: {sorted(discarded_segments)}")
    print()

    print("Commands:")
    print("  <label> + Enter   → assign label (e.g. WAVE, OPEN_PALM, null)")
    print("  Enter alone       → label as 'null' (unlabeled)")
    print("  skip              → skip this segment (won't appear in output)")
    print("  quit              → save what's done and exit\n")

    processed = []

    for seg_id in seg_ids:
        # Skip discarded segments
        if seg_id in discarded_segments:
            print(f"⊘ Segment {seg_id} was discarded (too short) — skipping\n")
            continue

        rows = segments[seg_id]
        print_segment_summary(seg_id, rows)
        
        # Get positions from segment
        from_position = rows[0].get("from_position", "NONE")
        to_position = rows[0].get("to_position", "NONE")

        # Get seg_type from user (or auto-infer)
        while True:
            auto_inferred = infer_segment_type(rows)
            user_input = input(f"  Type (T=transition / S=stable, or Enter for auto={auto_inferred}): ").strip().upper()
            
            if user_input == "":
                seg_type = auto_inferred
                break
            elif user_input in ("T", "S"):
                seg_type = user_input
                break
            else:
                print("  Please enter T, S, or press Enter for auto-detection.")

        # Get label from user
        label_input = input("  Label (Enter = null, 'skip', 'quit'): ").strip()

        if label_input.lower() == "quit":
            print("\nSaving and exiting early...")
            break

        if label_input.lower() == "skip":
            print(f"  ✓ Skipped segment {seg_id}.\n")
            continue

        label = label_input if label_input else "null"

        result = process_segment(seg_id, seg_type, from_position, to_position, label, rows)
        processed.append(result)
        
        if seg_type == 'T':
            print(f"  ✓ Saved  seg_type=T  {from_position}→{to_position}  label='{label}'\n")
        else:
            print(f"  ✓ Saved  seg_type=S  @ {to_position}  label='{label}'\n")

    if not processed:
        print("\nNo segments were labeled. Exiting without saving.")
        sys.exit(0)

    # Write output CSV
    header = build_output_header()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in processed:
            writer.writerow(row)

    print(f"\n✓ Saved {len(processed)} segments to {output_path}")
    print(f"  Each segment: {N_SAMPLES} timesteps x {len(SENSOR_COLS)} channels")
    print(f"  Total columns per row: {len(header)}")

if __name__ == "__main__":
    main()