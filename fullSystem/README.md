# fullSystem

The Road Sight Unit pipeline: it detects and tracks road users in a camera
image, places them on the ground and predicts collisions. It also publishes
its tracks to the SafeCorners gateway.

Install:

    pip install -r requirements.txt

Run on the bundled clip, from this folder:

    python 5_realTime.py --source demo_video.mp4

A scene needs map.txt, road.png and trajetoriasClean.txt in the working
folder. They are made in order with 0_createMap.py, 2_guideDraw.py,
2b_tracksToTrajectories.py or 3_drawTrajectories.py, and 4_cleanTrajectories.py.

- scenes/: the two recorded scenes
- gateway_glue.py: the link to the gateway
- trackers/: tracker configuration
- check_setup.py: checks that the needed files and packages are present

Tests:

    python -m pytest -q tests
