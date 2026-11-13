"""Đọc config/zones.yaml (kết quả segment_map) và tra cứu vùng theo tọa độ / tên (không phụ thuộc ROS)."""

from dataclasses import dataclass, field
import math
import os

import cv2
import numpy as np
import yaml

KIND_TEXT = {'room': 'phòng', 'corridor': 'hành lang'}


@dataclass
class Zone:
    id: str                       # chữ cái gọi bằng giọng nói: A, B, C ...
    name: str                     # tên có ý nghĩa: Phòng học, Hành lang ...
    label: int                    # giá trị pixel trong ảnh nhãn
    kind: str = 'room'
    aliases: list = field(default_factory=list)
    goal: tuple = (0.0, 0.0, 0.0)  # (x, y, yaw_deg) điểm đích hợp lệ trong frame map
    center: tuple = (0.0, 0.0)
    area: float = 0.0
    clearance: float = 0.0
    polygon: list = field(default_factory=list)

    @property
    def title(self):
        return self.name if self.name.lower() == f'khu vực {self.id}'.lower() \
            else f'{self.id} ({self.name})'


class ZoneMap:

    def __init__(self, path):
        self.path = os.path.abspath(os.path.expanduser(path))
        with open(self.path) as f:
            data = yaml.safe_load(f) or {}
        self.frame_id = data.get('frame_id', 'map')
        m = data['map']
        self.resolution = float(m['resolution'])
        self.origin = tuple(float(v) for v in m['origin'])
        self.width, self.height = int(m['width']), int(m['height'])
        self.zones = [Zone(id=str(z['id']), name=z['name'], label=int(z['label']),
                           kind=z.get('kind', 'room'), aliases=list(z.get('aliases') or []),
                           goal=tuple(float(v) for v in z['goal']),
                           center=tuple(float(v) for v in z.get('center', z['goal'][:2])),
                           area=float(z.get('area', 0.0)), clearance=float(z.get('clearance', 0.0)),
                           polygon=[tuple(p) for p in z.get('polygon') or []])
                      for z in data.get('zones') or []]
        self.labels = None
        img = data.get('labels_image')
        if img:
            img = img if os.path.isabs(img) else os.path.join(os.path.dirname(self.path), img)
            self.labels = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            if self.labels is None or self.labels.shape != (self.height, self.width):
                self.labels = None
        self._by_label = {z.label: z for z in self.zones}

    def __iter__(self):
        return iter(self.zones)

    def __len__(self):
        return len(self.zones)

    def by_id(self, zone_id):
        for z in self.zones:
            if z.id.upper() == str(zone_id).upper():
                return z
        return None

    def world_to_pixel(self, x, y):
        col = int(math.floor((x - self.origin[0]) / self.resolution))
        row = self.height - 1 - int(math.floor((y - self.origin[1]) / self.resolution))
        return col, row

    def pixel_to_world(self, col, row):
        return (self.origin[0] + (col + 0.5) * self.resolution,
                self.origin[1] + (self.height - row - 0.5) * self.resolution)

    def zone_at(self, x, y, search_radius=0.5):
        """Vùng chứa điểm (x, y). Điểm sát tường (ngoài ảnh nhãn) -> vùng gần nhất trong search_radius."""
        if self.labels is not None:
            col, row = self.world_to_pixel(x, y)
            r = int(math.ceil(search_radius / self.resolution))
            if 0 <= col < self.width and 0 <= row < self.height and self.labels[row, col] > 0:
                return self._by_label.get(int(self.labels[row, col]))
            r0, c0 = max(row - r, 0), max(col - r, 0)
            win = self.labels[r0:row + r + 1, c0:col + r + 1]
            ys, xs = np.nonzero(win)
            if ys.size:
                k = int(np.argmin((ys + r0 - row) ** 2 + (xs + c0 - col) ** 2))
                return self._by_label.get(int(win[ys[k], xs[k]]))
            return None
        for z in self.zones:
            if z.polygon and point_in_polygon(x, y, z.polygon):
                return z
        return None

    def summary(self):
        return ', '.join(z.title for z in self.zones)


def point_in_polygon(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside
