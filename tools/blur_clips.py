"""Anonymize the release clips: blur faces, license plates, and text
(street signs, house numbers, business names) in every frame, strip all
video metadata, and write browser-safe mp4s into GITHUB/dataset/clips/.

READS source clips listed in release_clip_manifest.DO_NOT_PUBLISH.json
(original paths) and WRITES only into GITHUB/dataset/clips/. Never touches
data/iphone/ originals.

Detectors:
  - faces  : CenterFace (via the `deface` package)
  - text   : EasyOCR text-region detector — catches plate numbers, street
             signs, house numbers, business signage in one pass
Both boxes are dilated and mosaicked (pixelated). OCR runs every OCR_STRIDE
frames and the boxes are held across the gap (objects move little in ~1/15 s).

This is PROBABILISTIC. It will miss some small/blurry/angled text and
faces. A human must review the QA contact sheet before publishing, and the
dataset card must disclose residual risk. Resumable: existing outputs skip.

Usage:
  python tools/blur_clips.py            # all clips
  python tools/blur_clips.py --limit 5  # quick test on 5 clips
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "release_clip_manifest.DO_NOT_PUBLISH.json"
QA_DIR = ROOT / "GITHUB" / "dataset" / "PRIVACY_QA"

OCR_CANVAS = 640      # width OCR runs at (recall vs speed)
OCR_STRIDE = 2        # run OCR every N frames; hold boxes across the gap
DILATE = 0.30         # grow every detection box by this fraction each side
FACE_THRESH = 0.25
TEXT_THRESH = 0.20      # lower = more text caught (recall over precision)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def dilate_box(x1, y1, x2, y2, w, h):
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * DILATE; x2 += bw * DILATE
    y1 -= bh * DILATE; y2 += bh * DILATE
    return (max(0, int(x1)), max(0, int(y1)),
            min(w, int(x2)), min(h, int(y2)))


def mosaic(frame, box):
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    bh, bw = roi.shape[:2]
    blocks = max(1, min(bw, bh) // 6)
    small = cv2.resize(roi, (max(1, bw // blocks), max(1, bh // blocks)),
                       interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(pix, (0, 0), sigmaX=max(2, bw / 12))


def text_boxes(reader, frame):
    h, w = frame.shape[:2]
    scale = OCR_CANVAS / w
    small = cv2.resize(frame, (OCR_CANVAS, int(h * scale)))
    res = reader.detect(small, text_threshold=TEXT_THRESH)
    out = []
    raw = res[0][0] if res and res[0] else []
    for b in raw:
        x1, x2, y1, y2 = b[0] / scale, b[1] / scale, b[2] / scale, b[3] / scale
        out.append(dilate_box(x1, y1, x2, y2, w, h))
    return out


def face_boxes(cf, frame):
    h, w = frame.shape[:2]
    dets, _ = cf(frame, threshold=FACE_THRESH)
    out = []
    for d in dets:
        x1, y1, x2, y2 = d[:4]
        out.append(dilate_box(x1, y1, x2, y2, w, h))
    return out


def process(src, dst, cf, reader):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = Path(tempfile.mktemp(suffix=".mp4"))
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx, held_text, n_face, n_text = 0, [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = face_boxes(cf, frame)
        if idx % OCR_STRIDE == 0:
            held_text = text_boxes(reader, frame)
        n_face += len(faces); n_text += len(held_text)
        for box in faces + held_text:
            mosaic(frame, box)
        vw.write(frame)
        idx += 1
    cap.release(); vw.release()
    # Re-encode to browser-safe H.264 and STRIP ALL METADATA.
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp), "-map_metadata", "-1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-an", "-loglevel", "error", str(dst)], check=True)
    tmp.unlink(missing_ok=True)
    return idx, n_face, n_text


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    clips = json.loads(MANIFEST.read_text())
    if limit:
        clips = clips[:limit]
    log(f"loading detectors (faces + text)…")
    from deface.centerface import CenterFace
    import easyocr
    cf = CenterFace(in_shape=None, backend="auto")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    QA_DIR.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    qa_frames = []
    for i, c in enumerate(clips):
        src, dst = Path(c["src"]), Path(c["dst"])
        if dst.exists():
            skipped += 1
            continue
        if not src.exists():
            log(f"MISSING src {c['clip_id']}")
            continue
        t = time.time()
        nf, faces, texts = process(src, dst, cf, reader)
        done += 1
        log(f"[{i+1}/{len(clips)}] {c['clip_id']}: {nf} frames, "
            f"{faces} face-blurs, {texts} text-blurs, {time.time()-t:.0f}s")
        # grab one blurred mid-frame for the QA contact sheet
        if done % 8 == 1:
            cap = cv2.VideoCapture(str(dst))
            cap.set(cv2.CAP_PROP_POS_FRAMES, nf // 2)
            ok, fr = cap.read(); cap.release()
            if ok:
                fr = cv2.resize(fr, (180, 320))
                cv2.putText(fr, c["clip_id"], (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                qa_frames.append(fr)
    # build QA contact sheet (blurred mid-frames, for human review)
    if qa_frames:
        cols = 6
        rows = (len(qa_frames) + cols - 1) // cols
        sheet = np.zeros((rows * 320, cols * 180, 3), dtype=np.uint8)
        for k, fr in enumerate(qa_frames):
            r, cc = divmod(k, cols)
            sheet[r*320:(r+1)*320, cc*180:(cc+1)*180] = fr
        cv2.imwrite(str(QA_DIR / "contact_sheet.jpg"), sheet)
        log(f"QA contact sheet -> {QA_DIR/'contact_sheet.jpg'} ({len(qa_frames)} samples)")
    log(f"DONE: {done} blurred, {skipped} already done. "
        f"REVIEW {QA_DIR/'contact_sheet.jpg'} before publishing.")


if __name__ == "__main__":
    main()
