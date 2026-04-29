#!/usr/bin/env python3
"""
check_setup.py — Validates that the fullSystem environment is correctly set up.
Run this before using any of the pipeline scripts.
"""
import sys

def check(name, import_fn):
    try:
        result = import_fn()
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

def main():
    print("=" * 60)
    print("SafeRoadside fullSystem — Setup Checker")
    print("=" * 60)
    all_ok = True

    # 1) Python version
    print(f"\n1) Python: {sys.version}")

    # 2) Required packages
    print("\n2) Checking Python packages...")
    all_ok &= check("torch (PyTorch)", lambda: __import__("torch").__version__)
    all_ok &= check("torch CUDA", lambda: "Available (" + __import__("torch").cuda.get_device_name(0) + ")" if __import__("torch").cuda.is_available() else (_ for _ in ()).throw(RuntimeError("CUDA not available! GPU required.")))
    all_ok &= check("numpy", lambda: __import__("numpy").__version__)
    all_ok &= check("opencv (cv2)", lambda: __import__("cv2").__version__)
    all_ok &= check("ultralytics (YOLO)", lambda: __import__("ultralytics").__version__)
    all_ok &= check("deep_sort_realtime", lambda: (__import__("deep_sort_realtime"), "installed")[1])
    all_ok &= check("mss", lambda: __import__("mss").__version__)
    all_ok &= check("pyautogui", lambda: (__import__("pyautogui"), "installed")[1])
    all_ok &= check("pygame", lambda: __import__("pygame").__version__)

    # 3) Required files
    import os
    print("\n3) Checking required files...")
    files = {
        "models/yolo11n.pt": "YOLOv11 model weights",
        "road.png": "Empty road image",
        "tes.png": "Calibration image",
        "map.txt": "Homography matrix (from step 0)",
        "withguide.png": "Guide image (from step 2)",
        "trajetorias.txt": "Raw trajectories (from step 3)",
        "trajetoriasClean.txt": "Cleaned trajectories (from step 4)",
    }
    for f, desc in files.items():
        exists = os.path.isfile(f)
        size = os.path.getsize(f) if exists else 0
        status = f"OK ({size:,} bytes)" if exists else "MISSING"
        symbol = "[OK]" if exists else "[!!]"
        print(f"  {symbol} {f} — {desc}: {status}")
        if not exists and f in ("models/yolo11n.pt", "road.png", "tes.png"):
            all_ok = False  # These are mandatory

    # 4) shared/data.txt
    shared_data = os.path.join("..", "shared", "data.txt")
    if os.path.isfile(shared_data):
        print(f"  [OK] ../shared/data.txt — BLE output file: exists")
    else:
        print(f"  [!!] ../shared/data.txt — BLE output file: MISSING (will be created at runtime)")

    # 5) Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED — System is ready!")
        print("\nTo get started, run the pipeline in order:")
        print("  Step 0: python 0_createMap.py    (calibrate pixel<->GPS)")
        print("  Step 1: python 1_conv.py         (verify calibration)")
        print("  Step 2: python 2_guideDraw.py    (auto-detect trajectories)")
        print("  Step 3: python 3_drawTrajectories.py (draw reference paths)")
        print("  Step 4: python 4_cleanTrajectories.py (clean trajectories)")
        print("  Step 5: python 5_realTime.py     (real-time collision alerts)")
        print("\nSince you already have map.txt + trajetoriasClean.txt,")
        print("you can jump directly to:  python 5_realTime.py")
    else:
        print("SOME CHECKS FAILED — see above for details.")
        print("Fix the issues and run this script again.")
    print("=" * 60)

if __name__ == "__main__":
    main()

