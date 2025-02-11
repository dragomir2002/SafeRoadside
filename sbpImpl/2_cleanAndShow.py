import cv2
import random
import re

def clean_and_show(
    sbp_file="sbp_ok.txt",
    background_image="EmptyRoad.png",
    output_image="emptyRoadWithTrajectories.png",
    min_points=20
):
    """
    Reads sbp_file lines, filters out trajectories with fewer than `min_points`, 
    and draws them on top of background_image. Saves the result to output_image.
    """

    # Load the background image
    img = cv2.imread(background_image)
    if img is None:
        print(f"Could not read image '{background_image}'!")
        return

    # Open the SBP file
    try:
        with open(sbp_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File '{sbp_file}' not found!")
        return

    # For each trajectory:
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Example line: "myvideo.mp4-3: (100,200), (110,210), (120,220)"
        # Split at the colon
        try:
            left_part, coords_part = line.split(":")
        except ValueError:
            # If line is malformed, skip it
            continue

        # left_part might be something like "myvideo.mp4-3"
        track_identifier = left_part.strip()

        # coords_part might be "(100,200), (110,210), (120,220)"
        # Use regex to extract (x,y) pairs
        matches = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", coords_part)
        # Convert the matches to integer tuples
        points = [(int(x), int(y)) for (x, y) in matches]

        # Filter out short trajectories
        if len(points) < min_points:
            continue

        # Generate a random color for this track
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

        # Draw each point of this trajectory on the image
        for (px, py) in points:
            cv2.circle(img, (px, py), 3, color, -1)

        # Optionally, you could connect the points with lines, e.g.:
        #   for i in range(len(points) - 1):
        #       cv2.line(img, points[i], points[i+1], color, 2)

    # Save the final image
    cv2.imwrite(output_image, img)
    print(f"Output image saved to '{output_image}'")

    # (Optional) Show the result
    cv2.imshow("Trajectories (Filtered)", img)
    print("Press any key (or 'q') to close the window...")
    if cv2.waitKey(0) & 0xFF == ord('q'):
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    clean_and_show(
        sbp_file="sbp_ok.txt",
        background_image="EmptyRoad.png",
        output_image="emptyRoadWithTrajectories.png",
        min_points=20
    )
