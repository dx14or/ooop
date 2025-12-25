import io
import math
import random
import os
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

def _fibonacci_sphere(count: int) -> list[tuple[float, float, float]]:
    if count <= 1:
        return [(0.0, 0.0, 1.0)]
    points: list[tuple[float, float, float]] = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(count):
        y = 1.0 - (i / (count - 1)) * 2.0
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return points

def _spread_on_sphere(points: list[tuple[float, float, float]], pads: list[float], iterations: int = 140, strength: float = 0.35) -> list[tuple[float, float, float]]:
    if len(points) <= 1:
        return points
    vecs = [np.array(p, dtype=float) for p in points]
    count = len(vecs)
    for _ in range(iterations):
        forces = [np.zeros(3) for _ in range(count)]
        for i in range(count):
            for j in range(i + 1, count):
                delta = vecs[i] - vecs[j]
                dist = float(np.linalg.norm(delta))
                if dist < 1e-06:
                    delta = np.array([0.01, 0.0, 0.0])
                    dist = 0.01
                min_dist = pads[i] + pads[j]
                if dist < min_dist:
                    push = (min_dist - dist) * strength
                    direction = delta / dist
                    forces[i] += direction * push
                    forces[j] -= direction * push
        for i in range(count):
            vecs[i] += forces[i]
            norm = float(np.linalg.norm(vecs[i]))
            if norm > 0:
                vecs[i] /= norm
    return [(float(v[0]), float(v[1]), float(v[2])) for v in vecs]

def _rotate_x(x: float, y: float, z: float, angle: float) -> tuple[float, float, float]:
    ca = math.cos(angle)
    sa = math.sin(angle)
    return (x, y * ca - z * sa, y * sa + z * ca)

def _rotate_y(x: float, y: float, z: float, angle: float) -> tuple[float, float, float]:
    ca = math.cos(angle)
    sa = math.sin(angle)
    return (x * ca + z * sa, y, -x * sa + z * ca)

def _rotate_z(x: float, y: float, z: float, angle: float) -> tuple[float, float, float]:
    ca = math.cos(angle)
    sa = math.sin(angle)
    return (x * ca - y * sa, x * sa + y * ca, z)

def _rotate_2d(x: float, y: float, angle: float) -> tuple[float, float]:
    ca = math.cos(angle)
    sa = math.sin(angle)
    return (x * ca - y * sa, x * sa + y * ca)

def _resolve_font_path() -> str | None:
    candidates = ('DejaVu Sans Condensed Bold', 'DejaVu Sans Bold', 'DejaVu Sans')
    for name in candidates:
        try:
            return font_manager.findfont(name, fallback_to_default=False)
        except Exception:
            continue
    try:
        return font_manager.findfont('DejaVu Sans', fallback_to_default=True)
    except Exception:
        return None

_FONT_PATH = _resolve_font_path()

def _resolve_mono_font_path() -> str | None:
    candidates = ('DejaVu Sans Mono', 'DejaVu Sans Mono Bold', 'Courier New', 'Menlo', 'Monaco')
    for name in candidates:
        try:
            return font_manager.findfont(name, fallback_to_default=False)
        except Exception:
            continue
    try:
        return font_manager.findfont('DejaVu Sans Mono', fallback_to_default=True)
    except Exception:
        return None

_MONO_FONT_PATH = _resolve_mono_font_path()

def _normalize_style_override(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().lower().replace('-', '_').replace(' ', '_')
    if cleaned in ('random', 'rand', 'auto', 'none'):
        return None
    aliases = {
        'starwars': 'star_wars',
        'star_wars': 'star_wars',
        'matrix': 'matrix',
    }
    return aliases.get(cleaned)

def _get_font(size: int, cache: dict[int, ImageFont.ImageFont]) -> ImageFont.ImageFont:
    font = cache.get(size)
    if font:
        return font
    if _FONT_PATH:
        try:
            font = ImageFont.truetype(_FONT_PATH, size=size)
            cache[size] = font
            return font
        except Exception:
            pass
    font = ImageFont.load_default()
    cache[size] = font
    return font

def _get_mono_font(size: int, cache: dict[int, ImageFont.ImageFont]) -> ImageFont.ImageFont:
    font = cache.get(size)
    if font:
        return font
    if _MONO_FONT_PATH:
        try:
            font = ImageFont.truetype(_MONO_FONT_PATH, size=size)
            cache[size] = font
            return font
        except Exception:
            pass
    font = _get_font(size, cache)
    cache[size] = font
    return font

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        return font.getsize(text)

def _build_background(width: int, height: int) -> Image.Image:
    inner = np.array([18, 20, 38], dtype=float)
    outer = np.array([8, 10, 20], dtype=float)
    y, x = np.ogrid[:height, :width]
    cx = width / 2
    cy = height / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = math.hypot(cx, cy)
    t = np.clip(dist / max_dist, 0.0, 1.0)
    t = t ** 1.4
    rgb = inner + (outer - inner) * t[..., None]
    img = Image.fromarray(rgb.astype(np.uint8), mode='RGB')
    return img.convert('RGBA')

def _build_starfield_background(width: int, height: int, rng: random.Random) -> Image.Image:
    inner = np.array([8, 9, 16], dtype=float)
    outer = np.array([2, 2, 6], dtype=float)
    y, x = np.ogrid[:height, :width]
    cx = width / 2
    cy = height / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = math.hypot(cx, cy)
    t = np.clip(dist / max_dist, 0.0, 1.0)
    t = t ** 1.6
    rgb = inner + (outer - inner) * t[..., None]
    img = Image.fromarray(rgb.astype(np.uint8), mode='RGB').convert('RGBA')
    draw = ImageDraw.Draw(img)
    star_count = int(width * height * 0.0012)
    for _ in range(star_count):
        sx = rng.randrange(0, width)
        sy = rng.randrange(0, height)
        glow = rng.random()
        if glow > 0.97:
            color = (255, 255, 255, 220)
            draw.ellipse((sx - 1, sy - 1, sx + 1, sy + 1), fill=color)
        else:
            c = rng.randint(170, 255)
            draw.point((sx, sy), fill=(c, c, c, rng.randint(120, 200)))
    return img

def _build_matrix_base(width: int, height: int) -> Image.Image:
    inner = np.array([6, 16, 8], dtype=float)
    outer = np.array([2, 6, 3], dtype=float)
    y, x = np.ogrid[:height, :width]
    cx = width * 0.5
    cy = height * 0.45
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = math.hypot(cx, cy)
    t = np.clip(dist / max_dist, 0.0, 1.0)
    t = t ** 1.35
    rgb = inner + (outer - inner) * t[..., None]
    img = Image.fromarray(rgb.astype(np.uint8), mode='RGB').convert('RGBA')
    return img

def _build_matrix_columns(width: int, height: int, rng: random.Random, font_size: int) -> list[dict]:
    glyphs = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&*+"
    step = max(12, font_size + 3)
    col_gap = max(step + 2, int(width / 40))
    col_count = max(12, int(width / col_gap))
    columns = []
    for i in range(col_count):
        x = int(i * col_gap + rng.uniform(-0.2, 0.2) * col_gap)
        if x < 0:
            x = 0
        if x > width - 1:
            x = width - 1
        speed = rng.uniform(0.7, 1.6)
        length = rng.randint(int(height * 0.35), int(height * 0.8))
        offset = rng.uniform(0, height)
        glyph_count = max(8, int(length / step))
        text = "".join(rng.choice(glyphs) for _ in range(glyph_count))
        columns.append({
            'x': x,
            'speed': speed,
            'offset': offset,
            'step': step,
            'glyphs': text,
            'length': glyph_count * step,
        })
    return columns

def _layout_words(items: list[dict], width: int, height: int, rng: random.Random, font_cache: dict[int, ImageFont.ImageFont]) -> list[dict]:
    canvas = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(canvas)
    center_x = width / 2
    center_y = height / 2
    margin = int(min(width, height) * 0.08)
    placed: list[dict] = []
    boxes: list[tuple[float, float, float, float]] = []
    for item in items:
        size = item['size']
        font = _get_font(size, font_cache)
        text_w, text_h = _measure_text(draw, item['word'], font)
        pad_scale = item.get('pad', 1.0)
        pad = (max(10, int(size * 0.22)) + int(len(item['word']) * 0.4)) * pad_scale
        start_radius = (1.0 - item['weight']) * min(width, height) * 0.2
        angle = rng.random() * math.tau
        radius = start_radius
        radius_step = 0.6 + size * 0.02
        placed_item = None
        for _ in range(2000):
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            angle += 0.35
            radius += radius_step
            left = x - text_w / 2 - pad
            right = x + text_w / 2 + pad
            top = y - text_h / 2 - pad
            bottom = y + text_h / 2 + pad
            if left < margin or right > width - margin or top < margin or bottom > height - margin:
                continue
            overlap = False
            for ox1, oy1, ox2, oy2 in boxes:
                if left < ox2 and right > ox1 and top < oy2 and bottom > oy1:
                    overlap = True
                    break
            if overlap:
                continue
            placed_item = {**item, 'x': x, 'y': y, 'w': text_w, 'h': text_h}
            boxes.append((left, top, right, bottom))
            break
        if placed_item:
            dx = placed_item['x'] - center_x
            dy = placed_item['y'] - center_y
            placed_item['radius'] = math.hypot(dx, dy)
            placed_item['theta'] = math.atan2(dy, dx)
            placed_item['phase'] = rng.uniform(0.0, math.tau)
            placed_item['wobble'] = 6 + size * 0.08
            placed.append(placed_item)
    return placed

def generate_wordcloud(topic_terms: list[str], width: int = 1200, height: int = 1200, frames: int = 36) -> bytes:
    word_freq = {}
    for idx, terms in enumerate(topic_terms):
        weight = len(topic_terms) - idx + 1
        if isinstance(terms, str):
            for word in terms.split(', '):
                word = word.strip()
                if word:
                    word_freq[word] = word_freq.get(word, 0) + weight
        elif isinstance(terms, list):
            for word in terms:
                word = word.strip()
                if word:
                    word_freq[word] = word_freq.get(word, 0) + weight
    if not word_freq:
        return b''
    style_rng = random.Random()
    styles = ('star_wars', 'matrix')
    style_override = _normalize_style_override(os.environ.get('NTP_WORDCLOUD_STYLE'))
    style = style_override if style_override in styles else style_rng.choice(styles)
    if style == 'matrix':
        max_words = min(22, len(word_freq))
    else:
        max_words = min(20, len(word_freq))
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_words]
    max_freq = sorted_words[0][1] if sorted_words else 1
    min_freq = sorted_words[-1][1] if sorted_words else 1
    freq_range = max_freq - min_freq if max_freq > min_freq else 1
    if style == 'matrix':
        min_size = max(26, int(width * 0.022))
        max_size = max(min_size + 12, int(width * 0.06))
        base_color = (86, 255, 140)
        accent_color = (210, 255, 210)
        pad_scale = 1.5
    else:
        min_size = max(30, int(width * 0.026))
        max_size = max(min_size + 12, int(width * 0.07))
        base_color = (255, 214, 72)
        accent_color = None
        pad_scale = 1.6
    items: list[dict] = []
    for word, freq in sorted_words:
        importance = (freq - min_freq) / freq_range
        weight = 0.3 + 0.7 * (importance ** 0.6)
        size = int(min_size + weight * (max_size - min_size))
        depth = 0.35 + 0.65 * weight
        color = base_color
        items.append({
            'word': word.upper(),
            'size': size,
            'weight': weight,
            'depth': depth,
            'color': color,
            'pad': pad_scale,
        })
    items.sort(key=lambda item: item['size'], reverse=True)
    rng = random.Random(42)
    font_cache: dict[int, ImageFont.ImageFont] = {}
    placed = _layout_words(items, width, height, rng, font_cache)
    if not placed:
        return b''
    center_x = width / 2
    center_y = height / 2
    images = []
    if style == 'matrix':
        accent_count = max(1, len(placed) // 5)
        if accent_color:
            for item in sorted(placed, key=lambda w: w['size'], reverse=True)[:accent_count]:
                item['color'] = accent_color
                item['accent'] = True
        for item in placed:
            item['y'] = center_y + (item['y'] - center_y) * 0.72
            item['wobble'] = item['wobble'] * 0.5
        background = _build_matrix_base(width, height)
        mono_cache: dict[int, ImageFont.ImageFont] = {}
        matrix_font_size = max(14, int(width * 0.018))
        matrix_font = _get_mono_font(matrix_font_size, mono_cache)
        columns = _build_matrix_columns(width, height, rng, matrix_font_size)
        glow_steps = [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        for frame in range(frames):
            t = frame / max(frames, 1)
            frame_img = background.copy()
            draw = ImageDraw.Draw(frame_img)
            for col in columns:
                head_y = (col['offset'] + frame * col['speed'] * col['step']) % (height + col['length']) - col['length']
                glyphs = col['glyphs']
                tail_len = max(1, len(glyphs) - 1)
                for i, ch in enumerate(glyphs):
                    y = head_y + i * col['step']
                    if y < -col['step'] or y > height + col['step']:
                        continue
                    fade = i / tail_len
                    green = int(70 + 160 * fade)
                    alpha = int(30 + 150 * fade)
                    if i == len(glyphs) - 1:
                        color = (200, 255, 210, 220)
                    else:
                        color = (0, green, 0, alpha)
                    draw.text((col['x'], y), ch, font=matrix_font, fill=color)
            global_scale = 1.0 + 0.012 * math.sin(t * math.tau)
            for item in sorted(placed, key=lambda w: w['depth']):
                drift = math.sin(t * math.tau * 0.45 + item['phase']) * item['wobble']
                sway = math.cos(t * math.tau * 0.3 + item['phase']) * item['wobble'] * 0.5
                x = item['x'] + drift * 0.35
                y = item['y'] + sway * 0.22
                depth_scale = 0.95 + 0.2 * item['depth']
                font_size = max(12, int(item['size'] * depth_scale * global_scale))
                font = _get_font(font_size, font_cache)
                text_w, text_h = _measure_text(draw, item['word'], font)
                pos_x = x - text_w / 2
                pos_y = y - text_h / 2
                if pos_x + text_w < -5 or pos_x > width + 5 or pos_y + text_h < -5 or pos_y > height + 5:
                    continue
                base = item['color']
                alpha = int(180 + 60 * item['depth'])
                shadow = (0, 0, 0, int(alpha * 0.6))
                for ox, oy in glow_steps:
                    draw.text((pos_x + ox, pos_y + oy), item['word'], font=font, fill=shadow)
                glow = (64, 255, 140, int(alpha * 0.6))
                for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    draw.text((pos_x + ox, pos_y + oy), item['word'], font=font, fill=glow)
                draw.text((pos_x, pos_y), item['word'], font=font, fill=(*base, alpha))
            images.append(frame_img)
    else:
        background = _build_starfield_background(width, height, rng)
        vanish_y = height * 0.12
        bottom_y = height * 0.9
        plane_height = bottom_y - vanish_y
        crawl_base = height * 0.08
        glow_offsets = [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        for frame in range(frames):
            t = frame / max(frames, 1)
            frame_img = background.copy()
            draw = ImageDraw.Draw(frame_img)
            global_angle = t * math.tau * 0.04
            global_scale = 1.0 + 0.012 * math.sin(t * math.tau)
            crawl_shift = t * plane_height * 0.3
            for item in sorted(placed, key=lambda w: w['depth']):
                dx = item['x'] - center_x
                dy = item['y'] - center_y
                dx, dy = _rotate_2d(dx, dy, global_angle)
                x_raw = center_x + dx
                y_raw = center_y + dy + crawl_base - crawl_shift
                t_y = (y_raw - vanish_y) / plane_height
                if t_y < -0.2 or t_y > 1.2:
                    continue
                t_y = max(0.0, min(1.0, t_y))
                perspective = 0.35 + 0.85 * t_y
                x = center_x + (x_raw - center_x) * perspective
                y = vanish_y + (y_raw - vanish_y) * (0.65 + 0.35 * perspective)
                depth_scale = 0.85 + 0.25 * item['depth']
                font_size = max(10, int(item['size'] * perspective * depth_scale * global_scale))
                font = _get_font(font_size, font_cache)
                text_w, text_h = _measure_text(draw, item['word'], font)
                pos_x = x - text_w / 2
                pos_y = y - text_h / 2
                if pos_x + text_w < 0 or pos_x > width or pos_y + text_h < 0 or pos_y > height:
                    continue
                lift = 0.6 + 0.4 * item['depth']
                base = item['color']
                rgb = (
                    int(base[0] * lift),
                    int(base[1] * lift),
                    int(base[2] * lift),
                )
                alpha = int(170 + 70 * item['depth'])
                glow_alpha = int(90 + 70 * item['depth'])
                shadow = (0, 0, 0, int(alpha * 0.35))
                glow = (255, 190, 48, glow_alpha)
                draw.text((pos_x + 2, pos_y + 3), item['word'], font=font, fill=shadow)
                if font_size < 26:
                    glow_steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                else:
                    glow_steps = glow_offsets
                for ox, oy in glow_steps:
                    draw.text((pos_x + ox, pos_y + oy), item['word'], font=font, fill=glow)
                draw.text((pos_x, pos_y), item['word'], font=font, fill=(*rgb, alpha))
            images.append(frame_img)
    gif_buf = io.BytesIO()
    images[0].save(gif_buf, format='GIF', save_all=True, append_images=images[1:], duration=240, loop=0, optimize=True)
    gif_buf.seek(0)
    return gif_buf.read()

def analyze_trends(topics_seq: list[int], topic_labels: list[str], window_recent: int = 100, window_old: int = 100) -> list[dict]:
    if len(topics_seq) < window_recent + window_old:
        window_size = len(topics_seq) // 3
        if window_size < 10:
            return []
        window_recent = window_size
        window_old = window_size
    recent = topics_seq[-window_recent:]
    old = topics_seq[-(window_recent + window_old):-window_recent]
    recent_counts = Counter(recent)
    old_counts = Counter(old)
    recent_total = len(recent)
    old_total = len(old)
    trends = []
    all_topics = set(recent_counts.keys()) | set(old_counts.keys())
    for topic_id in all_topics:
        recent_pct = recent_counts.get(topic_id, 0) / recent_total if recent_total else 0
        old_pct = old_counts.get(topic_id, 0) / old_total if old_total else 0
        change = recent_pct - old_pct
        if abs(change) > 0.01:
            label = topic_labels[topic_id] if topic_id < len(topic_labels) else f'Тема {topic_id}'
            trends.append({
                'topic_id': topic_id,
                'label': label,
                'recent_pct': recent_pct,
                'old_pct': old_pct,
                'change': change,
                'direction': 'up' if change > 0 else 'down'
            })
    trends.sort(key=lambda x: abs(x['change']), reverse=True)
    return trends[:10]

def format_trends(trends: list[dict]) -> str:
    if not trends:
        return '📊 Недостаточно данных для анализа трендов'
    parts = ['📈 Тренды тем:\n']
    up_trends = [t for t in trends if t['direction'] == 'up']
    down_trends = [t for t in trends if t['direction'] == 'down']
    if up_trends:
        parts.append('🔥 Растут:')
        for t in up_trends[:5]:
            change_pct = abs(t['change']) * 100
            parts.append(f"  ↑ {t['label']} (+{change_pct:.0f}%)")
    if down_trends:
        parts.append('\n❄️ Падают:')
        for t in down_trends[:5]:
            change_pct = abs(t['change']) * 100
            parts.append(f"  ↓ {t['label']} (-{change_pct:.0f}%)")
    return '\n'.join(parts)
