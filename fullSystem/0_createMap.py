#!/usr/bin/env python3
# createMap.py

import argparse
import csv
import math

import cv2
import numpy as np

# Lista de 13 coordenadas lat/long em formato string.
# Exemplo (modifica ou adapta às tuas coords):
coords_str = [
    "38.736741°N 9.141845°W",
    "38.736654°N 9.141690°W",
    "38.736401°N 9.141675°W",
    "38.736663°N 9.141836°W",
    "38.736632°N 9.141947°W",
    "38.736564°N 9.142016°W",
    "38.736437°N 9.141992°W",
    "38.736471°N 9.142024°W",
    "38.736422°N 9.141906°W",
    "38.736405°N 9.141905°W",
    "38.736409°N 9.141815°W",
    "38.736446°N 9.141810°W",
    "38.736539°N 9.141778°W"
] # TODO abre uma imagem para por os pontos estas coornedas no mapa extamente e gerar map.txt, para ver as coorednadas extas poedmos ir ao google earth
    # TODO ArUco markers para facilitar a seleção dos pontos (mas tem de ser na mesma ordem das coords_str)

    # TODO trajetorias sao as vistas pelo sistemas/as desenhadas, ou seja caso um carro faca alguma coisa mal, o sistema nao o vai detetar
    
"""
    coords_str = [
    "38.736407°N 9.143549°W",
    "38.736401°N 9.143609°W",
    "38.736482°N 9.143597°W",
    "38.736467°N 9.143683°W",
    "38.736401°N 9.143743°W",
    "38.736392°N 9.143791°W",
    "38.736465°N 9.143808°W",
    "38.736386°N 9.143878°W",
    "38.736288°N 9.143773°W",
    "38.736242°N 9.143705°W",
    "38.736297°N 9.143675°W",
    "38.736290°N 9.143581°W",
    "38.736348°N 9.143535°W"
]
"""



def parse_latlon(coord_str):
    """
    Recebe algo como "38.877713°N 7.172653°W"
    Retorna (lat_dec, lon_dec) em graus decimais,
    em que W e S são negativos.
    """
    coord_str = coord_str.strip().upper().replace('°', '')
    # Ex: "38.877713N 7.172653W"
    parts = coord_str.split()
    lat_str = parts[0]  # ex: "38.877713N"
    lon_str = parts[1]  # ex: "7.172653W"

    def parse_part_lat(s):
        if 'N' in s:
            return float(s.replace('N', ''))
        elif 'S' in s:
            return -float(s.replace('S', ''))
        return float(s)  # fallback

    def parse_part_lon(s):
        if 'E' in s:
            return float(s.replace('E', ''))
        elif 'W' in s:
            return -float(s.replace('W', ''))
        return float(s)  # fallback

    lat_dec = parse_part_lat(lat_str)
    lon_dec = parse_part_lon(lon_str)
    return lat_dec, lon_dec

def latlon_to_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    """
    Converte (lat, lon) em graus decimais para
    coordenadas locais (X, Y) em metros, usando
    uma projeção planar simples em torno de (lat0, lon0).
    """
    R = 6371000.0  # Raio médio da Terra em metros (aprox)
    # Converte para radianos
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    # Projeção local
    X = R * (lon - lon0) * math.cos(lat0)
    Y = R * (lat - lat0)
    return X, Y


# --- calibration from a points file (--points) --------------------------------

def load_points(path):
    """Calibration points: CSV with header name,lat,lon,px,py.

    lat/lon in decimal degrees (south and west negative). px/py are pixels of
    the full-resolution calibration image, blank for points still to be clicked.
    """
    def num(s):
        s = (s or "").strip()
        return float(s) if s else None
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [{"name": r["name"].strip(), "lat": float(r["lat"]), "lon": float(r["lon"]),
                 "px": num(r.get("px")), "py": num(r.get("py"))}
                for r in csv.DictReader(fh)]


def save_points(path, points):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "lat", "lon", "px", "py"])
        for p in points:
            w.writerow([p["name"], p["lat"], p["lon"],
                        "" if p["px"] is None else round(p["px"], 1),
                        "" if p["py"] is None else round(p["py"], 1)])


def _project(H, pixels):
    p = np.hstack([np.asarray(pixels, dtype=np.float64),
                   np.ones((len(pixels), 1))]) @ np.asarray(H, dtype=np.float64).T
    return p[:, :2] / p[:, 2:3]


def fit_map(pixels, latlons):
    """Pixel -> local metres homography, fitted as the click flow fits it
    (RANSAC, 5 m), plus a quality figure per point.

    residual_m: the point's error under the fitted H. loo_m: its error under an
    H fitted without it (leave-one-out) -- the honest one, because a point
    always pulls a fit towards itself. inlier: kept by RANSAC.
    """
    origin = (float(latlons[0][0]), float(latlons[0][1]))
    src = np.asarray(pixels, dtype=np.float64)
    dst = np.array([latlon_to_xy(la, lo, *origin) for la, lo in latlons], dtype=np.float64)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("no homography fits these points")
    residual = np.linalg.norm(_project(H, src) - dst, axis=1)
    loo = []
    for i in range(len(src)):
        keep = np.arange(len(src)) != i
        Hi, _ = cv2.findHomography(src[keep], dst[keep], cv2.RANSAC, 5.0)
        loo.append(math.inf if Hi is None
                   else float(np.linalg.norm(_project(Hi, src[i:i + 1])[0] - dst[i])))
    return {"origin": origin, "H": H, "residual_m": [float(r) for r in residual],
            "loo_m": loo, "inlier": [bool(m) for m in mask.ravel()]}


def write_map(path, origin, H):
    """map.txt: origin lat lon, then the 3x3 homography, one row per line."""
    with open(path, "w") as f:
        f.write(f"{origin[0]} {origin[1]}\n")
        for row in H:
            f.write(" ".join(map(str, row)) + "\n")


def click_missing(img, points, win="calibration - click the named point, ESC stops"):
    """Fill blank px/py by clicking. The image is shown fitted to the screen and
    clicks are scaled back to full resolution; a 4x inset of the full image
    under the cursor helps aim. Precision is about one displayed pixel (~2-3 px
    of a 4K image), so reading px/py from an image viewer is better."""
    h, w = img.shape[:2]
    scale = min(1600.0 / w, 900.0 / h, 1.0)
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    state = {"cursor": (0, 0), "click": None}

    def on_mouse(event, x, y, flags, param):
        state["cursor"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    for p in points:
        if p["px"] is not None and p["py"] is not None:
            continue
        state["click"] = None
        while state["click"] is None:
            disp = small.copy()
            fx, fy = int(state["cursor"][0] / scale), int(state["cursor"][1] / scale)
            crop = img[max(fy - 50, 0):fy + 50, max(fx - 50, 0):fx + 50]
            if crop.size:
                inset = cv2.resize(crop, (crop.shape[1] * 4, crop.shape[0] * 4),
                                   interpolation=cv2.INTER_NEAREST)
                ih, iw = inset.shape[:2]
                disp[0:ih, 0:iw] = inset
                cv2.drawMarker(disp, (iw // 2, ih // 2), (0, 0, 255), cv2.MARKER_CROSS, 30, 1)
            cv2.putText(disp, f"click: {p['name']}", (10, disp.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(win, disp)
            if cv2.waitKey(15) & 0xFF == 27:
                cv2.destroyAllWindows()
                return False
        p["px"], p["py"] = state["click"][0] / scale, state["click"][1] / scale
        print(f"{p['name']}: pixel ({p['px']:.1f}, {p['py']:.1f})")
    cv2.destroyAllWindows()
    return True


def draw_check(img, points, fit, grid_m=5.0):
    """The calibration drawn over the image: each point where it was clicked
    (green circle) and where its lat/lon lands through the fit (red cross),
    labelled with its leave-one-out error, over a grid_m ground grid."""
    out = img.copy()
    Hinv = np.linalg.inv(np.asarray(fit["H"], dtype=np.float64))
    xy = np.array([latlon_to_xy(p["lat"], p["lon"], *fit["origin"]) for p in points])
    lo, hi = xy.min(axis=0) - 10.0, xy.max(axis=0) + 10.0

    def to_px(X, Y):
        u, v, s = Hinv @ np.array([X, Y, 1.0])
        return None if s <= 1e-9 else (int(round(u / s)), int(round(v / s)))

    for X in np.arange(math.floor(lo[0] / grid_m) * grid_m, hi[0] + grid_m, grid_m):
        for Y0, Y1 in zip(np.arange(lo[1], hi[1], 1.0), np.arange(lo[1] + 1.0, hi[1] + 1.0, 1.0)):
            a, b = to_px(X, Y0), to_px(X, Y1)
            if a and b:
                cv2.line(out, a, b, (255, 200, 0), 1)
    for Y in np.arange(math.floor(lo[1] / grid_m) * grid_m, hi[1] + grid_m, grid_m):
        for X0, X1 in zip(np.arange(lo[0], hi[0], 1.0), np.arange(lo[0] + 1.0, hi[0] + 1.0, 1.0)):
            a, b = to_px(X0, Y), to_px(X1, Y)
            if a and b:
                cv2.line(out, a, b, (255, 200, 0), 1)
    for i, (p, (X, Y)) in enumerate(zip(points, xy)):
        c = (int(round(p["px"])), int(round(p["py"])))
        cv2.circle(out, c, 14, (0, 255, 0), 3)
        q = to_px(X, Y)
        if q:
            cv2.drawMarker(out, q, (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.putText(out, f"{i + 1} {p['name']} loo {fit['loo_m'][i]:.2f} m", (c[0] + 18, c[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return out


def main_points(points_path, image_path, map_path, check_path):
    points = load_points(points_path)
    if len(points) < 4:
        print("Need at least 4 points (10-15 recommended).")
        return 1
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read {image_path}.")
        return 1
    if any(p["px"] is None or p["py"] is None for p in points):
        if not click_missing(img, points):
            print("Stopped before every point had a pixel; nothing written.")
            return 1
        save_points(points_path, points)
        print(f"Pixels saved into {points_path}.")
    fit = fit_map([(p["px"], p["py"]) for p in points], [(p["lat"], p["lon"]) for p in points])
    print(f"{'#':>2} {'point':24s} {'px':>7} {'py':>7} {'resid m':>8} {'loo m':>7}  inlier")
    for i, p in enumerate(points):
        print(f"{i + 1:2d} {p['name'][:24]:24s} {p['px']:7.1f} {p['py']:7.1f} "
              f"{fit['residual_m'][i]:8.2f} {fit['loo_m'][i]:7.2f}  {fit['inlier'][i]}")
    inl = [r for r, k in zip(fit["residual_m"], fit["inlier"]) if k]
    print(f"inliers {len(inl)}/{len(points)}, residual RMS (inliers) "
          f"{math.sqrt(sum(r * r for r in inl) / len(inl)):.2f} m, "
          f"leave-one-out max {max(fit['loo_m']):.2f} m")
    write_map(map_path, fit["origin"], fit["H"])
    cv2.imwrite(check_path, draw_check(img, points, fit))
    print(f"Wrote {map_path} and {check_path}.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Scene calibration: map.txt from ground points.")
    ap.add_argument("--points", help="CSV name,lat,lon,px,py (blank px/py are clicked). "
                                     "Without it: the coords_str list in this file, clicked in order.")
    ap.add_argument("--image", default="tes.png")
    ap.add_argument("--out", default="map.txt")
    ap.add_argument("--check", default="calib_check.png")
    args = ap.parse_args()
    if args.points:
        return main_points(args.points, args.image, args.out, args.check)
    return legacy_main(args.image, args.out, args.check)


def legacy_main(image_path="tes.png", map_path="map.txt", check_path="calib_check.png",
                save_points_path="points_clicked.csv"):
    """The original flow: the coordinates live in coords_str above, and you click
    them on the image in the same order.

    The clicking is the improved one (image fitted to the screen, a 4x inset to
    aim with, one named prompt per point), because the original showed a 4K image
    at 1:1 in an auto-sized window: on a smaller screen the far half of the scene
    was off-screen and could not be clicked at all. Same maths as before --
    RANSAC over pixel -> local metres -- plus a per-point error report and an
    overlay to check the fit by eye.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read {image_path}.")
        return 1
    points = []
    for i, coord in enumerate(coords_str, 1):
        lat, lon = parse_latlon(coord)
        points.append({"name": f"{i}: {coord}", "lat": lat, "lon": lon, "px": None, "py": None})

    print(f"Click the {len(points)} points in the order of coords_str. ESC aborts.")
    if not click_missing(img, points):
        print("Stopped before every point was clicked; nothing written.")
        return 1

    fit = fit_map([(p["px"], p["py"]) for p in points], [(p["lat"], p["lon"]) for p in points])
    print(f"{'#':>2} {'point':28s} {'px':>7} {'py':>7} {'resid m':>8} {'loo m':>7}  inlier")
    for i, p in enumerate(points):
        print(f"{i + 1:2d} {p['name'][:28]:28s} {p['px']:7.1f} {p['py']:7.1f} "
              f"{fit['residual_m'][i]:8.2f} {fit['loo_m'][i]:7.2f}  {fit['inlier'][i]}")
    inl = [r for r, k in zip(fit["residual_m"], fit["inlier"]) if k]
    print(f"inliers {len(inl)}/{len(points)}, residual RMS (inliers) "
          f"{math.sqrt(sum(r * r for r in inl) / len(inl)):.2f} m, "
          f"leave-one-out max {max(fit['loo_m']):.2f} m")

    write_map(map_path, fit["origin"], fit["H"])
    cv2.imwrite(check_path, draw_check(img, points, fit))
    save_points(save_points_path, points)
    print(f"Wrote {map_path}, {check_path} and {save_points_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
