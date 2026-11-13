"""Kiểm thử phân vùng trên bản đồ maze của package và tính hợp lệ của config/zones.yaml (không cần ROS).

Chạy: python3 -m pytest src/tb4_voice_nav/test -q
"""

import os

import numpy as np
import pytest

from tb4_voice_nav import segmentation as seg
from tb4_voice_nav.zones import ZoneMap

PKG = os.path.join(os.path.dirname(__file__), '..')


@pytest.fixture(scope='module')
def maze():
    grid = seg.load_map(os.path.join(PKG, 'maps', 'maze.yaml'))
    free = seg.clean_free(grid.free_mask())
    labels, dist, _ = seg.segment(free, grid.resolution)
    return grid, free, labels, dist


def test_at_least_three_zones_cover_free_space(maze):
    grid, free, labels, _ = maze
    n = labels.max()
    assert n >= 3
    assert set(np.unique(labels[free > 0])) == set(range(1, n + 1))   # mọi ô trống thuộc một vùng
    assert not labels[free == 0].any()                                 # không gán nhãn cho tường


def test_goals_are_valid(maze):
    grid, _, labels, dist = maze
    for r in seg.describe(labels, dist, grid.resolution):
        col, row = r.goal_px
        assert labels[row, col] == r.label           # điểm đích nằm trong chính vùng đó
        assert r.clearance >= 0.45                   # đủ xa tường cho robot bán kính 0.175 m + inflation
        assert r.area >= seg.SegmentParams().min_area


def test_pixel_world_roundtrip(maze):
    grid = maze[0]
    for col, row in [(0, 0), (123, 456), (grid.width - 1, grid.height - 1)]:
        assert grid.world_to_pixel(*grid.pixel_to_world(col, row)) == (col, row)


def test_unknown_205_is_not_free():
    img = np.full((10, 10), 205, np.uint8)
    img[2:8, 2:8] = 254
    grid = seg.GridMap(image=img, resolution=0.05, origin=(0, 0, 0), free_thresh=0.25, occupied_thresh=0.65)
    assert grid.free_mask().sum() == 36


def test_zones_yaml_consistent():
    zm = ZoneMap(os.path.join(PKG, 'config', 'zones.yaml'))
    assert len(zm) >= 3
    assert len({z.id for z in zm}) == len(zm)
    for z in zm:
        assert zm.zone_at(z.goal[0], z.goal[1]) is z
    names = {z.name for z in zm}
    assert {'Phòng học', 'Phòng thí nghiệm', 'Hành lang'} <= names
