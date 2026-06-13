"""Data setup assistant.

Checks which datasets are present, downloads what can be fetched
automatically, and prints exact instructions for the rest. Run it any
time — it only reports and downloads; it never deletes anything.

Usage:
  python tools/download_data.py                 # status + instructions
  python tools/download_data.py --roboflow KEY  # also download the
                                                #   pothole image set from
                                                #   Roboflow (free API key)

Why not download everything automatically? Most research datasets
(nuScenes, Pothole-600) require you to create an account or accept a
license first — scripting around that would violate their terms. This
script gets you to each download page and verifies your folders after.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

CHECKS = [
    ("Image: any YOLO-format pothole set", DATA / "image",
     lambda p: any(d.joinpath("data.yaml").exists()
                   for d in p.rglob("*") if d.is_dir()) if p.exists() else False,
     """Get one from Roboflow Universe (free account):
       1. Sign up at roboflow.com and search Universe for "pothole".
       2. Pick a dataset, choose Download -> Format: YOLOv8.
       3. Unzip into data/image/<dataset-name>/
     OR rerun this script with your API key to fetch one automatically:
       python tools/download_data.py --roboflow YOUR_API_KEY"""),

    ("Image: Pothole-600 (real pixel masks, for U-Net)", DATA / "image" / "Pothole600_Segmentation",
     lambda p: (p / "train" / "rgb").exists(),
     """Search the web for "Pothole-600 dataset" (by Rui Fan et al.) and
     request/download it from the authors' page. Arrange as:
       data/image/Pothole600_Segmentation/{train,valid,test}/{rgb,label}/
     This one is optional — only the U-Net segmentation benchmark needs it."""),

    ("Sensor: per-pass labeled accelerometer recordings", DATA / "sensor",
     lambda p: any(f.suffix == ".csv" for d in p.iterdir() if d.is_dir()
                   for f in d.iterdir()) if p.exists() else False,
     """Search for public "road anomaly accelerometer dataset" collections
     (labeled passes over potholes, cracks, etc.), or record your own.
     Format (see docs/DATA_GUIDE.md):
       data/sensor/<name>/<Category>_NNN.csv with columns
       label, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, seconds_elapsed
     IMPORTANT: label = 1 only for real potholes. Check the labels — our
     main dataset shipped with every anomaly labeled as a pothole and had
     to be fixed (tools/relabel_sensor_data.py)."""),

    ("Real drives: your own recordings (recommended!)", DATA / "iphone",
     lambda p: any((d / "Accelerometer.csv").exists() or (d / "sensor.csv").exists()
                   for d in p.iterdir() if d.is_dir()) if p.exists() else False,
     """The fastest path and the most fun: record a drive yourself.
       1. Install the free "Sensor Logger" app (iOS/Android).
       2. Mount the phone, record sensors + video while driving.
       3. Export and arrange (see docs/DATA_GUIDE.md):
          data/iphone/<MyDrive-date>/{Accelerometer.csv,Gyroscope.csv,
          Location.csv,Camera/video.mp4}
     Then: python tools/run_transfer.py  (uses models/pretrained/)"""),

    ("Real drives: nuScenes scenes (optional)", DATA / "nuscenes",
     lambda p: any(d.name.startswith("scene-") for d in p.iterdir()) if p.exists() else False,
     """nuScenes requires a free research account (their license forbids
     redistribution, so we cannot fetch it for you):
       1. Register at nuscenes.org and download the "mini" split.
       2. Export front-camera frames + IMU per scene as:
          data/nuscenes/scene-XXXX/{frames_cam_front/*.jpg, imu/ms_imu.csv}
       (frame files named by their microsecond timestamp)"""),
]


def roboflow_download(api_key):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  The roboflow package is not installed. Run:")
        print("    pip install roboflow")
        return False
    # A widely-used public pothole detection dataset on Roboflow Universe.
    # Swap workspace/project for any other Universe pothole dataset.
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("brad-dwyer").project("pothole-voxrl")
        target = DATA / "image" / "Roboflow_Pothole"
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading to {target} (YOLOv8 format)...")
        project.version(1).download("yolov8", location=str(target))
        print("  Done. Validate it from the dashboard (Setup panel).")
        return True
    except Exception as e:
        print(f"  Roboflow download failed: {e}")
        print("  You can still download manually from universe.roboflow.com")
        return False


def main():
    print("=" * 70)
    print("Dataset status")
    print("=" * 70)
    missing = []
    for name, path, present_fn, instructions in CHECKS:
        try:
            ok = present_fn(path)
        except Exception:
            ok = False
        mark = "FOUND  " if ok else "MISSING"
        print(f"[{mark}] {name}")
        if not ok:
            missing.append((name, instructions))

    if "--roboflow" in sys.argv:
        key = sys.argv[sys.argv.index("--roboflow") + 1]
        print("\nRoboflow download:")
        roboflow_download(key)

    if missing:
        print("\n" + "=" * 70)
        print("How to get what's missing")
        print("=" * 70)
        for name, instructions in missing:
            print(f"\n--- {name} ---")
            for line in instructions.splitlines():
                print(" " + line.strip())
        print("\nNone of these are required to start: the pre-trained models in")
        print("models/pretrained/ work on your own recorded drives right away.")
    else:
        print("\nEverything found. You're ready to train and run experiments.")


if __name__ == "__main__":
    main()
