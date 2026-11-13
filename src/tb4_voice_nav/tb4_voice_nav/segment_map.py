"""
Phân vùng bản đồ + gắn nhãn ngữ nghĩa -> config/zones.yaml (dùng cho node điều hướng bằng giọng nói).

Luồng xử lý:
  map.yaml + .pgm --segmentation.segment()--> ảnh nhãn vùng tự động
    -> (tùy chọn) vùng vẽ tay: đa giác trong zone_names.yaml đè lên vùng tự động
    -> gắn nhãn: mục trong zone_names.yaml chọn vùng chứa `point` -> đặt id/tên/bí danh;
       vùng còn lại nhận chữ cái kế tiếp theo thứ tự đọc (trên -> dưới, trái -> phải), tên "Khu vực X"
    -> điểm đích hợp lệ mỗi vùng (tâm đường tròn nội tiếp lớn nhất, kiểm tra khoảng cách tới tường)
    -> ghi zones.yaml (tọa độ thực trong frame map), zones_labels.png (ảnh nhãn 16 bit),
       zones_preview.png (ảnh trực quan có tên vùng)

Dùng:
  ros2 run tb4_voice_nav segment_map
  ros2 run tb4_voice_nav segment_map --map ~/tb4_ws/maps/map.yaml --names "" --out /tmp/zones.yaml
  ros2 run tb4_voice_nav segment_map --core-dist 0.9 --min-area 8 --show
"""

import argparse
import math
import os
import string
import sys

import cv2
import numpy as np
import yaml

from tb4_voice_nav import segmentation as seg

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def package_file(*parts):
    """Đường dẫn thật (theo symlink của --symlink-install -> ghi thẳng vào src) trong share của package."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('tb4_voice_nav')
    except Exception:  # chạy trực tiếp từ src
        share = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    path = os.path.join(share, *parts)
    return os.path.realpath(path) if os.path.exists(path) else path


def parse_args(argv):
    d = seg.SegmentParams()
    ap = argparse.ArgumentParser(description='Phân vùng bản đồ Occupancy Grid và gắn nhãn khu vực')
    ap.add_argument('--map', default=package_file('maps', 'maze.yaml'), help='map.yaml của map_server')
    ap.add_argument('--names', default=package_file('config', 'zone_names.yaml'),
                    help='YAML đặt tên / vẽ tay vùng ("" = chỉ dùng chữ cái)')
    ap.add_argument('--out', default=package_file('config', 'zones.yaml'), help='file zones.yaml xuất ra')
    ap.add_argument('--core-dist', type=float, default=d.core_dist,
                    help='m, nửa bề rộng tối thiểu của phòng (cửa/hành lang hẹp hơn 2 lần giá trị này)')
    ap.add_argument('--min-area', type=float, default=d.min_area, help='m², vùng nhỏ hơn bị gộp')
    ap.add_argument('--corridor-min-area', type=float, default=d.corridor_min_area)
    ap.add_argument('--door-margin', type=float, default=d.door_margin)
    ap.add_argument('--min-clearance', type=float, default=0.45,
                    help='m, điểm đích phải cách tường ít nhất (robot_radius 0.175 + đệm)')
    ap.add_argument('--manual-only', action='store_true',
                    help='bỏ phân vùng tự động, chỉ dùng các đa giác vẽ tay trong --names')
    ap.add_argument('--scale', type=int, default=2, help='hệ số phóng ảnh preview')
    ap.add_argument('--show', action='store_true', help='mở cửa sổ xem ảnh preview')
    return ap.parse_args(argv)


def load_names(path):
    if not path:
        return []
    with open(os.path.expanduser(path)) as f:
        data = yaml.safe_load(f) or {}
    return list(data.get('zones') or [])


def apply_manual(grid, labels, free, entries):
    """Vùng vẽ tay (polygon) đè lên vùng tự động. Trả về {chỉ số mục: nhãn}."""
    manual = {}
    for i, e in enumerate(entries):
        if not e.get('polygon'):
            continue
        mask = seg.rasterize_polygon(grid, e['polygon']) & free
        if not mask.any():
            raise ValueError(f"Đa giác của '{e.get('name', i)}' không chứa ô trống nào")
        lab = int(labels.max()) + 1
        labels[mask > 0] = lab
        manual[i] = lab
    return manual


def nearest_label(labels, col, row, radius_px):
    r0, c0 = max(row - radius_px, 0), max(col - radius_px, 0)
    win = labels[r0:row + radius_px + 1, c0:col + radius_px + 1]
    ys, xs = np.nonzero(win)
    if not ys.size:
        return 0
    k = int(np.argmin((ys + r0 - row) ** 2 + (xs + c0 - col) ** 2))
    return int(win[ys[k], xs[k]])


def assign_names(grid, labels, regions, entries, manual):
    """Ghép mục trong zone_names.yaml với vùng -> {nhãn: mục}."""
    by_label = {}
    for i, e in enumerate(entries):
        if i in manual:
            lab = manual[i]
        elif e.get('point'):
            col, row = grid.world_to_pixel(*e['point'][:2])
            if not grid.inside(col, row):
                raise ValueError(f"point của '{e.get('name')}' nằm ngoài bản đồ")
            lab = int(labels[row, col]) or nearest_label(labels, col, row, int(0.5 / grid.resolution))
            if not lab:
                raise ValueError(f"point {e['point']} của '{e.get('name')}' không thuộc vùng trống nào")
        else:
            raise ValueError(f"Mục '{e.get('name')}' cần 'point' hoặc 'polygon'")
        if lab in by_label:
            raise ValueError(f"'{e.get('name')}' và '{by_label[lab].get('name')}' rơi vào cùng một vùng; "
                             'chỉnh point hoặc --core-dist / --min-area')
        by_label[lab] = e
    return by_label


def reading_order(grid, regions, band=4.0):
    """Sắp vùng theo thứ tự đọc: dải ngang band m từ trên xuống, trong dải từ trái sang phải."""
    def key(r):
        x, y = grid.pixel_to_world(*r.centroid_px)
        return (-math.floor(y / band), x)
    return sorted(regions, key=key)


def build_zones(grid, labels, regions, by_label, min_clearance):
    used = {str(e['id']).upper() for e in by_label.values() if e.get('id')}
    letters = (c for c in string.ascii_uppercase if c not in used)
    zones, warnings = [], []
    for r in reading_order(grid, regions):
        e = by_label.get(r.label, {})
        zid = str(e['id']).upper() if e.get('id') else next(letters)
        name = e.get('name') or f'Khu vực {zid}'
        if e.get('goal'):
            gx, gy = float(e['goal'][0]), float(e['goal'][1])
            col, row = grid.world_to_pixel(gx, gy)
            if not grid.inside(col, row) or labels[row, col] != r.label:
                warnings.append(f'{zid}: goal tự đặt ({gx}, {gy}) không nằm trong vùng')
        else:
            gx, gy = grid.pixel_to_world(*r.goal_px)
        cx, cy = grid.pixel_to_world(*r.centroid_px)
        if r.clearance < min_clearance and not e.get('goal'):
            warnings.append(f'{zid}: điểm đích chỉ cách tường {r.clearance:.2f} m < {min_clearance} m')
        yaw = float(e['goal'][2]) if e.get('goal') and len(e['goal']) > 2 else 0.0
        zones.append({
            'id': zid,
            'name': name,
            'aliases': list(e.get('aliases') or []),
            'kind': e.get('kind') or r.kind,
            'label': r.label,
            'area': round(r.area, 2),
            'goal': [round(gx, 3), round(gy, 3), yaw],
            'clearance': round(r.clearance, 2),
            'center': [round(cx, 3), round(cy, 3)],
            'polygon': [[round(v, 3) for v in grid.pixel_to_world(c, rr)] for c, rr in r.contour_px],
        })
    zones.sort(key=lambda z: z['id'])
    return zones, warnings


def write_outputs(args, grid, labels, zones, prm, n_seed_rooms):
    out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    base = os.path.splitext(os.path.basename(out))[0]
    labels_png = f'{base}_labels.png'
    preview_png = os.path.join(os.path.dirname(out), f'{base}_preview.png')
    cv2.imwrite(os.path.join(os.path.dirname(out), labels_png), labels.astype(np.uint16))

    data = {
        'frame_id': 'map',
        'map': {'yaml': grid.yaml_path, 'resolution': grid.resolution,
                'origin': list(grid.origin), 'width': grid.width, 'height': grid.height},
        'labels_image': labels_png,
        'segmentation': {'method': 'manual' if args.manual_only else 'distance-transform + watershed',
                         'core_dist': prm.core_dist, 'min_area': prm.min_area,
                         'door_margin': prm.door_margin, 'corridor_min_area': prm.corridor_min_area,
                         'room_seeds': n_seed_rooms},
        'zones': zones,
    }
    with open(out, 'w') as f:
        f.write('# Sinh tự động bởi: ros2 run tb4_voice_nav segment_map (đừng sửa tay, sửa zone_names.yaml '
                'rồi chạy lại)\n# goal: [x, y, yaw_deg] điểm đích hợp lệ trong frame map; polygon: biên vùng\n')
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=None, width=120)
    preview = render_preview(grid, labels, zones, args.scale)
    cv2.imwrite(preview_png, preview)
    return out, preview_png, preview


def render_preview(grid, labels, zones, scale):
    """Ảnh màu: bản đồ + vùng tô màu + biên + điểm đích + nhãn tiếng Việt + bảng chú thích."""
    from PIL import Image, ImageDraw, ImageFont

    base = cv2.cvtColor(grid.image, cv2.COLOR_GRAY2BGR)
    cols = seg.palette(len(zones))
    color_of = {z['label']: cols[i] for i, z in enumerate(zones)}
    vis = base.copy()
    for lab, c in color_of.items():
        m = labels == lab
        vis[m] = (0.35 * base[m] + 0.65 * np.array(c)).astype(np.uint8)
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for z in zones:
        m = cv2.resize((labels == z['label']).astype(np.uint8), None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cs, -1, (60, 60, 60), 2)

    img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    f_id = ImageFont.truetype(FONT_BOLD, 15 * scale)
    f_name = ImageFont.truetype(FONT, 7 * scale)
    for z in zones:
        col, row = grid.world_to_pixel(*z['goal'][:2])
        x, y = (col + 0.5) * scale, (row + 0.5) * scale
        r = 4 * scale
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(220, 30, 30), outline=(255, 255, 255), width=2)
        draw.text((x, y - 6 * scale), z['id'], font=f_id, fill=(20, 20, 20), anchor='md',
                  stroke_width=2, stroke_fill=(255, 255, 255))
        if z['name'] != f"Khu vực {z['id']}":
            draw.text((x, y + 6 * scale), z['name'], font=f_name, fill=(20, 20, 20), anchor='ma',
                      stroke_width=2, stroke_fill=(255, 255, 255))

    # bảng chú thích bên phải
    f_leg = ImageFont.truetype(FONT, 8 * scale)
    f_head = ImageFont.truetype(FONT_BOLD, 9 * scale)
    line = 12 * scale
    panel_w = 250 * scale
    out = Image.new('RGB', (img.width + panel_w, max(img.height, (len(zones) + 4) * line)), 'white')
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    x0 = img.width + 10 * scale
    d.text((x0, 6 * scale), f'{len(zones)} khu vực  ({os.path.basename(grid.yaml_path)})', font=f_head,
           fill=(0, 0, 0))
    for i, z in enumerate(zones):
        y = 6 * scale + (i + 1.5) * line
        c = color_of[z['label']]
        d.rectangle([x0, y + 2, x0 + 9 * scale, y + 9 * scale], fill=(c[2], c[1], c[0]), outline=(60, 60, 60))
        kind = 'hành lang' if z['kind'] == 'corridor' else 'phòng'
        d.text((x0 + 13 * scale, y), f"{z['id']}  {z['name']}", font=f_leg, fill=(0, 0, 0))
        d.text((x0 + 130 * scale, y), f"{kind}, {z['area']:.0f} m²  ({z['goal'][0]:.1f}, {z['goal'][1]:.1f})",
               font=f_leg, fill=(70, 70, 70))
    y = 6 * scale + (len(zones) + 2) * line
    d.text((x0, y), '● điểm đích (goal) của mỗi khu vực', font=f_leg, fill=(200, 30, 30))
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    grid = seg.load_map(args.map)
    prm = seg.SegmentParams(core_dist=args.core_dist, min_area=args.min_area,
                            corridor_min_area=args.corridor_min_area, door_margin=args.door_margin)
    free = seg.clean_free(grid.free_mask())
    entries = load_names(args.names)

    if args.manual_only:
        labels, n_seed_rooms = np.zeros(free.shape, np.int32), 0
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * grid.resolution
    else:
        labels, dist, n_seed_rooms = seg.segment(free, grid.resolution, prm)
    manual = apply_manual(grid, labels, free, entries)
    if manual:
        labels = seg.keep_largest_parts(labels)
        labels = seg.merge_small(labels, grid.resolution, prm.min_area, protected=set(manual.values()))
    if labels.max() == 0:
        print('Không có vùng nào', file=sys.stderr)
        return 1
    regions = seg.describe(labels, dist, grid.resolution, prm)
    by_label = assign_names(grid, labels, regions, entries, manual)
    zones, warnings = build_zones(grid, labels, regions, by_label, args.min_clearance)
    out, preview_png, preview = write_outputs(args, grid, labels, zones, prm, n_seed_rooms)

    print(f'Bản đồ {grid.yaml_path}: {grid.width}x{grid.height} ô, {grid.resolution} m/ô')
    print(f"{'id':3s} {'tên':22s} {'loại':10s} {'m²':>6s} {'goal x':>7s} {'goal y':>7s} {'cách tường':>10s}")
    for z in zones:
        print(f"{z['id']:3s} {z['name']:22s} {z['kind']:10s} {z['area']:6.1f} {z['goal'][0]:7.2f} "
              f"{z['goal'][1]:7.2f} {z['clearance']:9.2f}m")
    for w in warnings:
        print('CẢNH BÁO:', w)
    print(f'Đã ghi {out}\n       {preview_png}')
    if args.show:
        cv2.imshow('zones', preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())
