"""
Phân vùng bản đồ Occupancy Grid 2D thành phòng / hành lang (không phụ thuộc ROS).

Thuật toán (ý tưởng "phòng = chỗ rộng, cửa / hành lang = chỗ hẹp"):
  1. Ảnh .pgm -> xác suất chiếm chỗ p = (255 - px) / 255 (negate = 0) -> ô TRỐNG nếu p < free_thresh.
     Làm sạch: mở hình thái (morphology open) 3x3 bỏ nhiễu, chỉ giữ thành phần trống lớn nhất
     (vùng robot thực sự đi tới được).
  2. Distance transform: d(p) = khoảng cách (m) từ ô trống tới ô không trống gần nhất.
  3. Hạt giống PHÒNG: các thành phần liên thông của {d > core_dist}. Cửa và hành lang hẹp hơn
     2 * core_dist nên bị cắt đứt -> mỗi phòng rộng một hạt giống.
  4. Hạt giống HÀNH LANG: phần trống KHÔNG nằm trong "phép mở" của các phòng (opening = dilate hạt
     giống bằng đĩa bán kính core_dist), bỏ thêm dải door_margin quanh phòng; thành phần đủ lớn
     (>= corridor_min_area) là một hành lang.
  5. Lan vùng (priority flood / watershed trên d): duyệt ô trống theo d giảm dần, mỗi ô nhận nhãn của
     ô lân cận đã gán trước nó. Hai vùng gặp nhau ở chỗ d nhỏ nhất giữa chúng = chỗ thắt (cửa).
     Tự viết thay cho cv2.watershed vì cv2 cho nhãn chảy dọc chân tường (d ~ 0) thành dải mỏng.
  6. Gộp vùng nhỏ (< min_area) vào vùng láng giềng có đường biên chung dài nhất.
  7. Với mỗi vùng: điểm đích = tâm đường tròn nội tiếp lớn nhất (max distance transform của riêng
     vùng) -> xa tường và xa ranh giới nhất; đa giác bao; loại phòng / hành lang theo độ thon dài.
"""

from dataclasses import dataclass, field
import heapq
import math
import os

import cv2
import numpy as np
import yaml


@dataclass
class GridMap:
    image: np.ndarray            # ảnh xám gốc (hàng 0 = phía trên = y lớn nhất)
    resolution: float            # m / ô
    origin: tuple                # (x, y, yaw) của góc dưới-trái ảnh trong frame map
    free_thresh: float
    occupied_thresh: float
    negate: int = 0
    yaml_path: str = ''
    mode: str = 'trinary'

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def width(self):
        return self.image.shape[1]

    def occupancy(self):
        img = self.image.astype(np.float32)
        return img / 255.0 if self.negate else (255.0 - img) / 255.0

    def free_mask(self):
        free = self.occupancy() < self.free_thresh
        if self.mode == 'trinary':
            # map_saver ghi ô chưa biết = 205 (p = 0.196); với free_thresh 0.25 (mặc định của Nav2 map_saver)
            # ô này bị tính là trống -> phân vùng tràn ra cả vùng chưa quét. Luôn coi 205 là chưa biết.
            free &= self.image != 205
        return free.astype(np.uint8)

    def pixel_to_world(self, col, row):
        """Tâm ô (col, row) của ảnh -> (x, y) trong frame map."""
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (self.height - row - 0.5) * self.resolution
        return x, y

    def world_to_pixel(self, x, y):
        col = int(math.floor((x - self.origin[0]) / self.resolution))
        row = self.height - 1 - int(math.floor((y - self.origin[1]) / self.resolution))
        return col, row

    def inside(self, col, row):
        return 0 <= col < self.width and 0 <= row < self.height


def load_map(yaml_path):
    """Đọc map.yaml của map_server (ROS) + ảnh .pgm đi kèm."""
    yaml_path = os.path.abspath(os.path.expanduser(yaml_path))
    with open(yaml_path) as f:
        info = yaml.safe_load(f)
    image_path = info['image']
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(yaml_path), image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'Không đọc được ảnh bản đồ {image_path}')
    origin = tuple(float(v) for v in info.get('origin', [0.0, 0.0, 0.0]))
    if abs(origin[2]) > 1e-6:
        raise ValueError('Chỉ hỗ trợ bản đồ có origin yaw = 0')
    return GridMap(image=image, resolution=float(info['resolution']), origin=origin,
                   free_thresh=float(info.get('free_thresh', 0.25)),
                   occupied_thresh=float(info.get('occupied_thresh', 0.65)),
                   negate=int(info.get('negate', 0)), yaml_path=yaml_path,
                   mode=str(info.get('mode', 'trinary')))


@dataclass
class SegmentParams:
    core_dist: float = 1.0          # m, nửa bề rộng tối thiểu của "phòng" (cửa/hành lang hẹp hơn 2x)
    door_margin: float = 0.6        # m, dải quanh phòng không dùng làm hạt giống hành lang
    min_seed_area: float = 0.2      # m², bỏ hạt giống phòng nhỏ hơn
    corridor_min_area: float = 1.0  # m², hạt giống hành lang tối thiểu
    min_area: float = 6.0           # m², vùng nhỏ hơn bị gộp vào láng giềng
    corridor_width: float = 2.4     # m, vùng hẹp hơn mức này và thon dài -> hành lang
    corridor_elongation: float = 3.0


@dataclass
class Region:
    label: int
    area: float                     # m²
    goal_px: tuple                  # (col, row) điểm đích
    clearance: float                # m, khoảng cách từ điểm đích tới tường
    inscribed: float                # m, bán kính đường tròn nội tiếp lớn nhất trong vùng
    centroid_px: tuple
    contour_px: list = field(default_factory=list)   # [(col, row), ...]
    kind: str = 'room'              # room | corridor


def clean_free(free, min_component=None):
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    if n <= 1:
        return free
    keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == keep).astype(np.uint8)


def disc(radius_px):
    r = max(int(round(radius_px)), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def make_seeds(free, dist, res, prm):
    """Bước 3 + 4: ảnh nhãn hạt giống (int32, 0 = chưa gán)."""
    px2 = res * res
    seeds = np.zeros(free.shape, np.int32)
    opened = np.zeros_like(free)
    n, lab, st, _ = cv2.connectedComponentsWithStats((dist > prm.core_dist).astype(np.uint8), 8)
    k = 0
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] * px2 < prm.min_seed_area:
            continue
        k += 1
        comp = (lab == i).astype(np.uint8)
        seeds[comp > 0] = k
        opened |= cv2.dilate(comp, disc(prm.core_dist / res))
    n_rooms = k
    near_rooms = cv2.dilate(opened, disc(prm.door_margin / res))
    narrow = free & (1 - near_rooms)
    n, lab, st, _ = cv2.connectedComponentsWithStats(narrow, 8)
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] * px2 >= prm.corridor_min_area:
            k += 1
            seeds[lab == j] = k
    return seeds, n_rooms


def priority_flood(seeds, dist, free):
    """Bước 5: watershed theo d giảm dần, gán nhãn khi đẩy vào hàng đợi (nhãn đầu tiên thắng)."""
    h, w = free.shape
    lab = seeds.ravel().copy()
    d = dist.ravel()
    fr = free.ravel().astype(bool)
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    heap = [(-float(d[i]), int(i)) for i in np.flatnonzero(lab > 0)]
    heapq.heapify(heap)
    while heap:
        _, i = heapq.heappop(heap)
        y, x = divmod(i, w)
        li = lab[i]
        for dy, dx in offsets:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                j = yy * w + xx
                if fr[j] and lab[j] == 0:
                    lab[j] = li
                    heapq.heappush(heap, (-float(d[j]), j))
    return lab.reshape(h, w)


def neighbour_counts(labels, label):
    m = (labels == label).astype(np.uint8)
    ring = cv2.dilate(m, np.ones((3, 3), np.uint8)) & (1 - m)
    nb = labels[(ring > 0) & (labels > 0)]
    vals, cnt = np.unique(nb, return_counts=True)
    return dict(zip(vals.tolist(), cnt.tolist()))


def merge_small(labels, res, min_area, protected=()):
    """Bước 6: gộp dần vùng nhỏ nhất vào láng giềng có biên chung dài nhất."""
    px2 = res * res
    while True:
        ids, cnt = np.unique(labels[labels > 0], return_counts=True)
        small = [(c * px2, int(i)) for i, c in zip(ids, cnt)
                 if c * px2 < min_area and int(i) not in protected]
        if not small:
            return labels
        _, lab = min(small)
        nb = neighbour_counts(labels, lab)
        labels[labels == lab] = max(nb, key=nb.get) if nb else 0


def keep_largest_parts(labels):
    """Mỗi nhãn chỉ giữ thành phần liên thông lớn nhất; mảnh rời gộp vào láng giềng."""
    for lab in np.unique(labels[labels > 0]):
        m = (labels == lab).astype(np.uint8)
        n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 2:
            continue
        keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        for i in range(1, n):
            if i == keep:
                continue
            piece = (cc == i)
            ring = cv2.dilate(piece.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~piece
            nb = labels[ring & (labels > 0) & (labels != lab)]
            labels[piece] = np.bincount(nb).argmax() if nb.size else 0
    return labels


def relabel(labels):
    out = np.zeros_like(labels)
    for k, lab in enumerate(np.unique(labels[labels > 0]), 1):
        out[labels == lab] = k
    return out


def segment(free, res, prm=None):
    """Bước 2-6. Trả về (ảnh nhãn int32 1..N, distance transform m, số hạt giống phòng)."""
    prm = prm or SegmentParams()
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * res
    seeds, n_rooms = make_seeds(free, dist, res, prm)
    if seeds.max() == 0:
        raise RuntimeError(f'Không có chỗ nào rộng hơn 2 x core_dist = {2 * prm.core_dist} m; '
                           'giảm core_dist')
    labels = priority_flood(seeds, dist, free)
    labels = merge_small(labels, res, prm.min_area)
    return relabel(labels), dist, n_rooms


def describe(labels, dist, res, prm=None):
    """Bước 7: thuộc tính từng vùng."""
    prm = prm or SegmentParams()
    regions = []
    for lab in np.unique(labels[labels > 0]):
        m = (labels == lab).astype(np.uint8)
        own = cv2.distanceTransform(m, cv2.DIST_L2, 5) * res
        row, col = np.unravel_index(int(np.argmax(own)), own.shape)
        ys, xs = np.nonzero(m)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        contour = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
        area = float(m.sum() * res * res)
        inscribed = float(own[row, col])
        width = 2.0 * inscribed
        kind = 'corridor' if (width < prm.corridor_width and
                              area / (width * width) > prm.corridor_elongation) else 'room'
        regions.append(Region(label=int(lab), area=area, goal_px=(int(col), int(row)),
                              clearance=float(dist[row, col]), inscribed=inscribed,
                              centroid_px=(float(xs.mean()), float(ys.mean())),
                              contour_px=[(int(c), int(r)) for c, r in contour], kind=kind))
    return regions


def rasterize_polygon(grid, polygon_xy):
    """Đa giác (x, y) trong frame map -> mặt nạ ảnh."""
    pts = np.array([grid.world_to_pixel(x, y) for x, y in polygon_xy], np.int32)
    mask = np.zeros((grid.height, grid.width), np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def palette(n, seed=7):
    """Màu BGR phân biệt cho n vùng (HSV xoay theo tỷ lệ vàng)."""
    cols = []
    for i in range(n):
        h = int((i * 0.618033988749895 % 1.0) * 180)
        c = cv2.cvtColor(np.uint8([[[h, 150, 235]]]), cv2.COLOR_HSV2BGR)[0, 0]
        cols.append(tuple(int(v) for v in c))
    return cols
