"""
Phân tích câu lệnh tiếng Việt (văn bản từ bộ nhận dạng giọng nói) -> lệnh điều hướng. Không phụ thuộc ROS.

Các bước:
  1. Chuẩn hóa: chữ thường, bỏ dấu câu, gộp khoảng trắng; tạo thêm bản "bỏ dấu" (đ -> d) để so khớp
     chịu được lỗi dấu của bộ nhận dạng ("phong thi nghiem" == "phòng thí nghiệm").
  2. Nhận ý định theo từ khóa (trên bản bỏ dấu):
       dừng / hủy / stop                      -> STOP
       đang ở đâu / vị trí                    -> WHERE
       có những khu vực nào / danh sách       -> LIST
       "từ X đến|tới|sang|qua Y"              -> GOTO [X, Y]   (đi qua X trước rồi tới Y)
       "... đến|tới|về|sang|vào Y [rồi đến Z]" -> GOTO [Y, Z ...]
  3. Tra tên khu vực trong cụm X / Y:
       a. "khu vực|vùng|khu|điểm <chữ cái>": chữ cái viết thường hoặc cách đọc tiếng Việt
          (a, bê, xê, đê, e, ép, giê, hát, i, ca, e-lờ, em, en, o, pê ...)
       b. tên / bí danh khớp nguyên cụm từ (chọn tên dài nhất khớp được)
       c. so khớp mờ (difflib) với ngưỡng fuzzy_threshold
     Không tìm thấy -> UNKNOWN_ZONE (kèm cụm từ người dùng nói) để báo lỗi, robot không di chuyển.
"""

from dataclasses import dataclass, field
import difflib
import re
import unicodedata

GOTO, STOP, WHERE, LIST, UNKNOWN_ZONE, AMBIGUOUS, NOT_UNDERSTOOD = (
    'GOTO', 'STOP', 'WHERE', 'LIST', 'UNKNOWN_ZONE', 'AMBIGUOUS', 'NOT_UNDERSTOOD')

# Cách đọc chữ cái trong tiếng Việt (đã bỏ dấu) -> chữ cái
LETTER_NAMES = {
    'a': 'A', 'ay': 'A',
    'b': 'B', 'be': 'B', 'bo': 'B', 'bi': 'B',
    'c': 'C', 'xe': 'C', 'ce': 'C', 'co': 'C', 'xi': 'C', 'si': 'C',
    'd': 'D', 'de': 'D', 'di': 'D', 'do': 'D',
    'e': 'E',
    'f': 'F', 'ep': 'F', 'ep pho': 'F', 'phe': 'F',
    'g': 'G', 'gie': 'G', 'go': 'G', 'gi': 'G',
    'h': 'H', 'hat': 'H', 'hac': 'H', 'ach': 'H', 'hot': 'H',
    'i': 'I',
    'j': 'J', 'gi dai': 'J', 'zi': 'J',
    'k': 'K', 'ca': 'K', 'ka': 'K',
    'l': 'L', 'e lo': 'L', 'lo': 'L', 'eo': 'L', 'en lo': 'L',
    'm': 'M', 'em': 'M', 'e mo': 'M', 'mo': 'M', 'em mo': 'M',
    'n': 'N', 'en': 'N', 'e no': 'N', 'no': 'N', 'en no': 'N',
    'o': 'O',
    'p': 'P', 'pe': 'P', 'pi': 'P', 'po': 'P',
    'q': 'Q', 'quy': 'Q', 'cu': 'Q', 'kiu': 'Q',
    'r': 'R', 'e ro': 'R', 'ro': 'R',
    's': 'S', 'et': 'S', 'et si': 'S', 'ech': 'S', 'so': 'S',
    't': 'T', 'te': 'T', 'ti': 'T', 'to': 'T',
    'u': 'U',
    'v': 'V', 've': 'V', 'vi': 'V', 'vo': 'V',
    'w': 'W', 'vay kep': 'W', 've kep': 'W', 'dap liu': 'W',
    'x': 'X', 'ich': 'X', 'ich xi': 'X', 'ix': 'X',
    'y': 'Y', 'i dai': 'Y', 'i gret': 'Y',
    'z': 'Z', 'det': 'Z', 'zet': 'Z', 'giet': 'Z', 'dzet': 'Z',
}
ZONE_WORDS = r'(?:khu vuc|khu so|khu|vung|diem|zone|vi tri|phong)'
STOP_RE = re.compile(r'\b(?:dung lai|dung|huy|huy bo|stop|thoi|ngung|ngung lai|khong di nua)\b')
WHERE_RE = re.compile(r'\b(?:o dau|dang o|vi tri hien tai|o khu vuc nao|o khu nao|minh o dau)\b')
LIST_RE = re.compile(r'\b(?:nhung khu vuc nao|co khu vuc nao|danh sach|liet ke|cac khu vuc)\b')
SPLIT_RE = re.compile(r'\b(?:roi|sau do|tiep theo|xong roi|sau khi den do|va sau do)\b')
FROM_TO_RE = re.compile(r'\btu\s+(?P<a>.+?)\s+(?:di |chay |di chuyen )?(?:den|toi|sang|qua|ve)\s+(?P<b>.+)$')
TO_RE = re.compile(r'\b(?:den|toi|ve|sang|vao|qua)\s+(?P<b>.+)$')
VERB_ONLY_RE = re.compile(r'^(?:di|di chuyen|chay|dan duong|dua toi)\s+(?P<b>.+)$')
FILLER = ('nhe', 'nha', 'di', 'giup toi', 'giup minh', 'giup', 'ngay', 'ngay lap tuc', 'lam on',
          'nao', 'voi', 'nghe', 'luon', 'cai', 'thu', 'cho toi', 'cho minh', 'gium')
POLITE_PREFIX = ('robot oi', 'robot', 'turtlebot', 'ban oi', 'hay', 'lam on', 'xin', 'vui long', 'oi')


def normalize(text):
    """Chữ thường, chỉ giữ chữ / số / khoảng trắng."""
    text = unicodedata.normalize('NFC', text or '').lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def fold(text):
    """Bỏ dấu tiếng Việt: 'Phòng thí nghiệm' -> 'phong thi nghiem'."""
    text = normalize(text).replace('đ', 'd')
    text = unicodedata.normalize('NFD', text)
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')


@dataclass
class Command:
    kind: str
    text: str = ''
    route: list = field(default_factory=list)      # [Zone, ...] theo thứ tự cần đi qua
    has_from: bool = False                         # câu dạng "từ X đến Y"
    unknown: list = field(default_factory=list)    # cụm từ không khớp khu vực nào
    candidates: list = field(default_factory=list)  # khi AMBIGUOUS


class CommandParser:

    def __init__(self, zones, fuzzy_threshold=0.8):
        """zones: danh sách đối tượng có .id .name .aliases (tb4_voice_nav.zones.Zone)."""
        self.zones = list(zones)
        self.fuzzy_threshold = fuzzy_threshold
        self.names = []  # (cụm bỏ dấu, zone)
        for z in self.zones:
            for n in [z.name, *z.aliases]:
                f = fold(n)
                # tên mặc định "Khu vực X" đã do nhánh chữ cái xử lý; đưa vào so khớp mờ thì
                # "khu vực z" (không tồn tại) sẽ bị hiểu nhầm thành "khu vực e"
                if f and not re.fullmatch(ZONE_WORDS + r'\s+\w', f):
                    self.names.append((f, z))
        self.names.sort(key=lambda t: -len(t[0]))
        self.by_letter = {z.id.upper(): z for z in self.zones}

    # ---------- câu lệnh ----------
    def parse(self, text):
        raw = text
        t = fold(text)
        t = self._strip_prefix(t)
        if not t:
            return Command(NOT_UNDERSTOOD, raw)

        if FROM_TO_RE.search(t) or TO_RE.search(t) or VERB_ONLY_RE.search(t):
            cmd = self._parse_goto(t, raw)
            if cmd is not None:
                return cmd
        if STOP_RE.search(t):
            return Command(STOP, raw)
        if WHERE_RE.search(t):
            return Command(WHERE, raw)
        if LIST_RE.search(t):
            return Command(LIST, raw)
        return Command(NOT_UNDERSTOOD, raw)

    def _strip_prefix(self, t):
        changed = True
        while changed:
            changed = False
            for p in POLITE_PREFIX:
                if t.startswith(p + ' '):
                    t, changed = t[len(p) + 1:], True
        return t

    def _parse_goto(self, t, raw):
        phrases, has_from = [], False
        parts = [p.strip() for p in SPLIT_RE.split(t) if p.strip()]
        for i, part in enumerate(parts):
            m = FROM_TO_RE.search(part) if i == 0 else None
            if m:
                phrases += [m['a'], m['b']]
                has_from = True
                continue
            m = TO_RE.search(part) or VERB_ONLY_RE.search(part)
            if m:
                phrases.append(m['b'])
            elif i > 0 and not STOP_RE.search(part):   # "... rồi khu vực C" (thiếu động từ)
                phrases.append(part)
        if not phrases:
            return None

        route, unknown, ambiguous = [], [], []
        for ph in phrases:
            res = self.match(ph)
            if isinstance(res, list):
                ambiguous.append((ph, res))
            elif res is None:
                unknown.append(self._clean(ph))
            else:
                route.append(res)
        if unknown:
            return Command(UNKNOWN_ZONE, raw, unknown=unknown, has_from=has_from)
        if ambiguous:
            return Command(AMBIGUOUS, raw, unknown=[a[0] for a in ambiguous],
                           candidates=ambiguous[0][1], has_from=has_from)
        # bỏ điểm trùng liên tiếp ("từ A đến A")
        route = [z for i, z in enumerate(route) if i == 0 or z is not route[i - 1]]
        return Command(GOTO, raw, route=route, has_from=has_from)

    # ---------- tên khu vực ----------
    def _clean(self, phrase):
        words = phrase.split()
        changed = True
        while words and changed:
            changed = False
            for f in sorted(FILLER, key=len, reverse=True):
                fw = f.split()
                if len(words) > len(fw) and words[-len(fw):] == fw:
                    words, changed = words[:-len(fw)], True
                    break
        return ' '.join(words)

    def match(self, phrase):
        """Cụm từ -> Zone | None (không có) | [Zone, ...] (mơ hồ)."""
        p = fold(phrase)
        # a. "khu vực <chữ cái>": thử trước khi bỏ từ đệm vì "a", "đi" cũng là cách đọc chữ A, D
        for q in dict.fromkeys((p, self._clean(p))):
            m = re.fullmatch(ZONE_WORDS + r'\s+(?P<l>.+)', q)
            letter = self._letter(m['l']) if m else None
            if letter is not None:
                return self.by_letter.get(letter)
        p = self._clean(p)
        if not p:
            return None
        if len(p) == 1 and p.isalpha():
            return self.by_letter.get(p.upper())

        # b. tên / bí danh khớp nguyên cụm từ, ưu tiên tên dài nhất
        padded = f' {p} '
        hits = []
        for name, z in self.names:
            if f' {name} ' in padded or (len(p.split()) >= 2 and f' {p} ' in f' {name} '):
                hits.append((len(name), name == p, z))
        if hits:
            best = max(h[0] for h in hits)
            exact = [h[2] for h in hits if h[1]]
            if exact:
                return exact[0]
            top = list(dict.fromkeys(h[2].id for h in hits if h[0] == best))
            if len(top) == 1:
                return self.by_letter[top[0]]
            return [self.by_letter[i] for i in top]

        # c. so khớp mờ (lỗi nhận dạng: "phòng thí nghiệp", "thư viên" ...)
        scored = sorted(((difflib.SequenceMatcher(None, p, name).ratio(), name, z)
                         for name, z in self.names), key=lambda s: -s[0])
        if scored and scored[0][0] >= self.fuzzy_threshold:
            return scored[0][2]
        return None

    @staticmethod
    def _letter(token):
        # chỉ khớp nguyên cụm ("a nhé" đã được _clean bỏ từ đệm); "gi ép" không được đoán thành G
        return LETTER_NAMES.get(token.strip())


def zone_list_text(zones):
    return ', '.join(z.title for z in zones)


def describe(cmd):
    """Mô tả ngắn để log."""
    if cmd.kind == GOTO:
        return ('từ ' if cmd.has_from else '') + ' -> '.join(z.title for z in cmd.route)
    if cmd.kind == UNKNOWN_ZONE:
        return 'khu vực không tồn tại: ' + ', '.join(f'"{u}"' for u in cmd.unknown)
    if cmd.kind == AMBIGUOUS:
        return f'"{cmd.unknown[0]}" mơ hồ: ' + ', '.join(z.title for z in cmd.candidates)
    return cmd.kind
