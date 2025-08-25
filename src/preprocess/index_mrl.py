"""
Index an 'open_eyes' / 'closed_eyes' layout of the MRL Eye Dataset into a CSV,
to record metadata for analysis / slice metrics and to validate the eye_state parameter against the folder structure

Filenames are in the format of: s0001_00001_0_0_0_0_0_01.png

MRL token meaning:
  s0001           -> subject_id = 0001
  00001           -> image_id
  0               -> gender   (0=man, 1=woman)
  0               -> glasses  (0=no, 1=yes)
  0               -> eye_state (0=closed, 1=open)
  0               -> reflections (0 none, 1 small, 2 big)
  0               -> lighting (0 bad, 1 good)
  01              -> sensor_id (01 RealSense, 02 IDS, 03 Aptina)
"""

import argparse, csv, os, re
from pathlib import Path
from typing import Optional, Tuple

import cv2

RAW_ROOT = Path("data/mrl/raw")
COPIES_DIR = Path("data/mrl/eye_crops")
META_DIR = Path("data/mrl/meta")
OUT_CSV = META_DIR / "mrl_index.csv"

EYE_SIZE = 64

# Token parsing from filename
def parse_filename_tokens(name: str) -> Tuple[Optional[str], Optional[str], list]:
    """
    Parse 's0001_00001_0_0_0_0_0_01.png' -> ('0001','00001',[0,0,0,0,0,1])
    Returns (subject_id, image_id, tokens[]) where tokens are ints (length expected: 6).
    """
    base = Path(name).stem.lower()  # strip extension
    parts = base.split('_')
    if len(parts) < 8:
        return None, None, []
    subj = parts[0]
    image_id = parts[1]
    # remaining parts should be 6 numeric tokens
    tail = parts[2:]
    # Some packs use sensor '01' which becomes '01' -> int 1
    try:
        tokens = [int(t) for t in tail]
    except ValueError:
        tokens = []
    # extract numeric subject id from 's0001'
    m = re.match(r"s(\d+)$", subj)
    subject_id = m.group(1) if m else None
    return subject_id, image_id, tokens

def main():
    ap = argparse.ArgumentParser()
    # Define optional command line flags to customize paths
    ap.add_argument("--raw_root", default=str(RAW_ROOT))
    ap.add_argument("--out_csv",  default=str(OUT_CSV))
    ap.add_argument("--write_copies", action="store_true", help="Write 64x64 grayscale copies into data/mrl/eye_crops")
    args = ap.parse_args()

    raw_root = Path(args.raw_root) # Convert from string format to a Path object
    META_DIR.mkdir(parents=True, exist_ok=True)
    # Only create copies dir if flag is set to true
    if args.write_copies:
        COPIES_DIR.mkdir(parents=True, exist_ok=True)

    open_dir = raw_root / "open_eyes"
    closed_dir = raw_root / "closed_eyes"
    if not open_dir.exists() or not closed_dir.exists():
        raise SystemExit(f"Expected directories {open_dir} and {closed_dir} do not exist.")

    rows = []
    problems = {"bad_name": 0, "state_mismatch": 0}

    def handle_folder(folder: Path, folder_state: str):
        # folder_state: "1" for open, "0" for closed (matches MRL encoding)
        # Iterate over all image files in the folder
        for fp in folder.rglob("*"):
            if not fp.is_file() or fp.suffix.lower() not in {".png",".jpg",".jpeg",".bmp",".pgm"}:
                continue # Skip non-image files
            subject_id, image_id, toks = parse_filename_tokens(fp.name)
            if subject_id is None or image_id is None or len(toks) < 6:
                problems["bad_name"] += 1 # File name does not match the expected format
                continue

            # Unpack tokens per MRL convention:
            gender, glasses, eye_state_tok, reflections, lighting, sensor_id_int = toks[:6]

            # Cross-check eye state (folder vs token). Folder is source of truth here.
            eye_state = int(folder_state)
            if eye_state_tok != eye_state:
                problems["state_mismatch"] += 1  # keep track; dataset variants can differ

            # Write a normalized copy if the Command Line Tool Flag is set
            if args.write_copies:
                img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (EYE_SIZE, EYE_SIZE), interpolation=cv2.INTER_AREA)
                # keep an organized layout in copies dir
                out_rel = Path(f"subj_{subject_id}") / ("open_eyes" if eye_state==1 else "closed_eyes") / fp.name
                out_abs = COPIES_DIR / out_rel
                out_abs.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_abs), img)
                img_relpath = (Path("eye_crops") / out_rel).as_posix()
            else:
                # reference original path relative to dataset root
                img_relpath = (Path("raw") / fp.relative_to(raw_root)).as_posix()

            # Store as strings to keep CSV tidy
            rows.append([
                subject_id, image_id, img_relpath, str(gender), str(glasses), str(eye_state),
                str(reflections), str(lighting), f"{sensor_id_int:02d}"
            ])

    # open=1, closed=0 per MRL docs
    handle_folder(open_dir, folder_state="1")
    handle_folder(closed_dir, folder_state="0")

    # write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "subject_id", "image_id", "img_relpath", "gender", "glasses", "eye_state",
            "reflections", "lighting", "sensor_id"
        ])
        w.writerows(rows)

    total = len(rows)
    print(f"[mrl] wrote {total} rows -> {args.out_csv}")
    print(f"[mrl] issues: {problems}")

if __name__ == "__main__":
    main()