"""Kiểm thử bộ phân tích câu lệnh với danh sách khu vực thật trong config/zones.yaml (không cần ROS).

Chạy: python3 -m pytest src/tb4_voice_nav/test -q
"""

import os

import pytest

from tb4_voice_nav import command_parser as cp
from tb4_voice_nav.zones import ZoneMap

ZONES = os.path.join(os.path.dirname(__file__), '..', 'config', 'zones.yaml')


@pytest.fixture(scope='module')
def parser():
    return cp.CommandParser(ZoneMap(ZONES))


def ids(cmd):
    return [z.id for z in cmd.route]


@pytest.mark.parametrize('text, route, has_from', [
    # 3 câu mẫu của đề bài
    ('Đi đến khu vực A.', ['A'], False),
    ('Di chuyển từ khu vực A đến khu vực B.', ['A', 'B'], True),
    ('Đi đến phòng thí nghiệm.', ['H'], False),
    # biến thể
    ('đi tới khu vực C', ['C'], False),
    ('Robot ơi, hãy đi đến hành lang nhé', ['B'], False),
    ('Đi đến thư viện', ['J'], False),
    ('quay về sảnh chính', ['D'], False),
    ('đi vào phòng học giúp tôi', ['A'], False),
    ('từ phòng học sang phòng thí nghiệm', ['A', 'H'], True),
    ('đi đến khu vực A rồi đến khu vực D', ['A', 'D'], False),
    ('đi đến khu vực A rồi khu vực D sau đó về thư viện', ['A', 'D', 'J'], False),
    # lỗi nhận dạng / không dấu / cách đọc chữ cái
    ('di den phong thi nghiem', ['H'], False),
    ('đi đến phòng thí nghiệp', ['H'], False),
    ('đi đến khu vực bê', ['B'], False),
    ('đi đến khu vực xê', ['C'], False),
    ('đi đến khu vực đê', ['D'], False),
    ('đi đến khu vực đi', ['D'], False),
    ('đi đến khu vực a', ['A'], False),
    ('đi đến khu e', ['E'], False),
    ('di chuyển đến vùng ca', ['K'], False),
    ('Đi đến phòng lab', ['H'], False),
    ('đi đến khu vực A nhé', ['A'], False),
    ('từ khu vực A đến khu vực A', ['A'], True),
])
def test_goto(parser, text, route, has_from):
    cmd = parser.parse(text)
    assert cmd.kind == cp.GOTO, cp.describe(cmd)
    assert ids(cmd) == route
    assert cmd.has_from == has_from


@pytest.mark.parametrize('text, unknown', [
    ('Đi đến khu vực Z', 'khu vuc z'),
    ('Đi đến nhà bếp', 'nha bep'),
    ('đi đến phòng giám đốc', 'phong giam doc'),
    ('di chuyển từ khu vực A đến khu vực Y', 'khu vuc y'),
])
def test_unknown_zone(parser, text, unknown):
    cmd = parser.parse(text)
    assert cmd.kind == cp.UNKNOWN_ZONE
    assert cmd.unknown == [unknown]


@pytest.mark.parametrize('text, kind', [
    ('Dừng lại', cp.STOP),
    ('hủy', cp.STOP),
    ('stop', cp.STOP),
    ('Robot đang ở đâu?', cp.WHERE),
    ('Có những khu vực nào', cp.LIST),
    ('hôm nay trời đẹp quá', cp.NOT_UNDERSTOOD),
    ('', cp.NOT_UNDERSTOOD),
])
def test_other_intents(parser, text, kind):
    assert parser.parse(text).kind == kind


def test_fold():
    assert cp.fold('Đi đến Phòng Thí Nghiệm!') == 'di den phong thi nghiem'


@pytest.mark.parametrize('text', ['đi từ khu vực xê', 'đi đến khu vực gi ép', 'đi đến khu vực'])
def test_incomplete_or_bad_letter_is_not_guessed(parser, text):
    assert parser.parse(text).kind in (cp.UNKNOWN_ZONE, cp.NOT_UNDERSTOOD)
