import re
import math

def clean_near_points(
    input_file="sbp.txt",
    output_file="sbp_ok.txt",
    NEAR_THRESHOLD=10
):
    """
    Reads 'input_file', filters out consecutive points that are too close 
    (distance < NEAR_THRESHOLD), and writes the cleaned data to 'output_file'.
    """

    try:
        with open(input_file, "r") as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        print(f"File '{input_file}' not found!")
        return

    with open(output_file, "w") as f_out:
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Example line:
            #   "myvideo.mp4-3: (100,200), (110,210), (120,220)"
            # We split by ":", extracting the identifier and the points string
            parts = line.split(":", maxsplit=1)
            if len(parts) < 2:
                # Malformed line, skip
                continue

            track_id = parts[0].strip()       # e.g. "myvideo.mp4-3"
            coords_str = parts[1].strip()     # e.g. "(100,200), (110,210), (120,220)"

            # Use a regex to find all "(x,y)" pairs
            matches = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", coords_str)
            points = [(int(x), int(y)) for (x, y) in matches]

            # Filter out near-duplicate points
            cleaned_points = []
            for p in points:
                if not cleaned_points:
                    # Always keep the first point
                    cleaned_points.append(p)
                else:
                    last_p = cleaned_points[-1]
                    dist = math.hypot(p[0] - last_p[0], p[1] - last_p[1])
                    if dist >= NEAR_THRESHOLD:
                        cleaned_points.append(p)

            # If after cleaning we still have points, write them out
            if cleaned_points:
                # Rebuild line in the original format: track_id: (x1,y1), (x2,y2), ...
                pts_str = ", ".join(f"({x},{y})" for (x, y) in cleaned_points)
                new_line = f"{track_id}: {pts_str}\n"
                f_out.write(new_line)

    print(f"Done! Cleaned data saved in '{output_file}'.")

if __name__ == "__main__":
    # Example usage:
    clean_near_points(
        input_file="sbp.txt",
        output_file="sbp_ok.txt",
        NEAR_THRESHOLD=10
    )
