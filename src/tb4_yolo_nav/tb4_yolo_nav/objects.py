"""Đọc config/objects.yaml và tìm file model.sdf tương ứng (cache Fuel hoặc model tự dựng)."""

import os
import re

from ament_index_python.packages import get_package_share_directory
import yaml

FUEL_URL = re.compile(
    r'^https?://(?P<host>[^/]+)/1\.0/(?P<owner>[^/]+)/models/(?P<name>[^/]+)(?:/(?P<ver>\d+))?/?$')


def load_objects(path):
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get('world', 'maze'), data.get('objects') or []


def fuel_cache_dir():
    return os.environ.get('IGN_FUEL_CACHE_PATH',
                          os.path.join(os.path.expanduser('~'), '.ignition', 'fuel'))


def resolve_model(model):
    """
    Trả về đường dẫn để truyền cho `create -file`.

    - Tên model trong tb4_yolo_nav/models -> <share>/models/<tên>/model.sdf
    - URL Fuel đã có trong cache -> file model.sdf của phiên bản mới nhất trong cache
    - URL Fuel chưa có trong cache -> chính URL đó (Gazebo sẽ tự tải, cần mạng)
    """
    m = FUEL_URL.match(model.strip())
    if m is None:
        path = os.path.join(get_package_share_directory('tb4_yolo_nav'),
                            'models', model, 'model.sdf')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Không tìm thấy model cục bộ: {path}')
        return path

    # Cache của Fuel lưu owner/tên ở dạng chữ thường
    base = os.path.join(fuel_cache_dir(), m['host'], m['owner'].lower(), 'models',
                        m['name'].lower())
    versions = [m['ver']] if m['ver'] else sorted(
        (d for d in os.listdir(base) if d.isdigit()), key=int, reverse=True
    ) if os.path.isdir(base) else []
    for ver in versions:
        path = os.path.join(base, ver, 'model.sdf')
        if os.path.isfile(path):
            return path
    return model.strip()
