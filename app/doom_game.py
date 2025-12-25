from __future__ import annotations

from dataclasses import dataclass
import html
import math
import random
from typing import Dict, List, Tuple


@dataclass
class DoomState:
    x: float = 15.5
    y: float = 6.5
    direction: float = -math.pi / 2
    hp: int = 100
    ammo: int = 6
    medkits: int = 2
    kills: int = 0
    steps: int = 0
    in_fight: bool = False
    enemy: str = ''
    enemy_hp: int = 0
    enemy_x: float = 0.0
    enemy_y: float = 0.0
    log: str = 'Ты в коридоре базы UAC. Впереди темно.'
    frame: int = 0
    flash: int = 0


class DoomGame:
    _ENEMIES = (
        ('Imp', 28, 50),
        ('Zombie', 24, 44),
        ('Demon', 38, 66),
    )
    _MAP = [
        '###############################',
        '#############.....#############',
        '#############.....#############',
        '#############.....#############',
        '#############.....#############',
        '#############.....#############',
        '#############.....#############',
        '#############.....#############',
        '#############.....#############',
        '#############.....#############',
        '#############.....#############',
        '#############.....#############',
        '#############.....#############',
        '#############.....#############',
        '#############.....#############',
        '#############.....#############',
        '###############################',
    ]
    _MAP_H = len(_MAP)
    _MAP_W = len(_MAP[0])
    _FOV = math.radians(52)
    _MAX_DEPTH = 8.0
    _MOVE_STEP = 0.6
    _TURN_STEP = 0.28
    _WALL_SCALE = 0.9

    def __init__(self, seed: int = 1337) -> None:
        self._rng = random.Random(seed)
        self._states: Dict[int, DoomState] = {}

    def is_active(self, user_id: int) -> bool:
        return user_id in self._states

    def start(self, user_id: int) -> str:
        state = DoomState()
        self._states[user_id] = state
        state.log = 'Ты проснулся в коридоре базы UAC. Двигайся.'
        return self._render(state)

    def handle_action(self, user_id: int, action: str) -> str:
        if action == 'restart':
            return self.start(user_id)

        state = self._states.get(user_id)
        if not state:
            state = DoomState()
            self._states[user_id] = state
            state.log = 'Новая игра. Используй кнопки.'
            return self._render(state)

        if action == 'exit':
            self._states.pop(user_id, None)
            return 'Вы вышли из DOOM. Введите /doom чтобы начать заново.'

        if state.hp <= 0:
            state.log = 'Ты мертв. Нажми "Новая игра".'
            return self._render(state)

        if action in ('up', 'down', 'left', 'right'):
            if state.in_fight and action in ('up', 'down'):
                state.log = 'В бою нельзя уходить. Стреляй или бей ножом.'
                return self._render(state)
            if action in ('left', 'right'):
                self._turn(state, action)
            else:
                self._move(state, action)
            self._maybe_spawn_enemy(state, chance=0.28)
            return self._render(state)

        if action == 'scan':
            state.steps += 1
            if self._rng.random() < 0.2:
                state.log = 'Слышны шаги... возможно рядом враг.'
            else:
                state.log = 'Тишина. Дальше темно.'
            self._maybe_spawn_enemy(state, chance=0.22)
            return self._render(state)

        if not state.in_fight:
            state.log = 'Вокруг пусто. Двигайся дальше.'
            return self._render(state)

        if action == 'shoot':
            if state.ammo <= 0:
                state.log = 'Патроны закончились. Нужна перезарядка.'
                return self._render(state)
            state.ammo -= 1
            if not self._enemy_visible(state):
                state.log = 'Выстрел мимо. Цели не видно.'
                state.flash = 2
                self._enemy_attack(state)
                return self._render(state)
            damage = self._rng.randint(14, 26)
            state.enemy_hp -= damage
            state.flash = 2
            if state.enemy_hp <= 0:
                state.kills += 1
                state.in_fight = False
                state.log = f'Выстрел! {state.enemy} убит.'
                state.enemy = ''
                state.enemy_hp = 0
                state.enemy_x = 0.0
                state.enemy_y = 0.0
                return self._render(state)
            state.log = f'Попадание по {state.enemy}. Урон {damage}.'
            self._enemy_attack(state)
            return self._render(state)

        if action == 'knife':
            damage = self._rng.randint(8, 16)
            state.enemy_hp -= damage
            if state.enemy_hp <= 0:
                state.kills += 1
                state.in_fight = False
                state.log = f'Нож в упор! {state.enemy} убит.'
                state.enemy = ''
                state.enemy_hp = 0
                state.enemy_x = 0.0
                state.enemy_y = 0.0
                return self._render(state)
            state.log = f'Ножом по {state.enemy}. Урон {damage}.'
            self._enemy_attack(state, bonus=4)
            return self._render(state)

        if action == 'reload':
            state.ammo = 6
            state.log = 'Перезарядка завершена.'
            self._enemy_attack(state)
            return self._render(state)

        if action == 'med':
            if state.medkits <= 0 or state.hp >= 100:
                state.log = 'Аптечек нет или здоровье полное.'
                return self._render(state)
            heal = self._rng.randint(18, 30)
            state.medkits -= 1
            state.hp = min(100, state.hp + heal)
            state.log = f'Аптечка использована. +{heal} HP.'
            self._enemy_attack(state)
            return self._render(state)

        if action == 'run':
            if self._rng.random() < 0.6:
                state.in_fight = False
                state.log = 'Ты вырвался и скрылся в коридорах.'
                state.enemy = ''
                state.enemy_hp = 0
                state.enemy_x = 0.0
                state.enemy_y = 0.0
            else:
                state.log = 'Побег не удался!'
                self._enemy_attack(state, bonus=6)
            return self._render(state)

        if action == 'tick':
            return self._render(state)

        state.log = 'Команда не распознана.'
        return self._render(state)

    def buttons(self, user_id: int) -> List[List[Tuple[str, str]]]:
        state = self._states.get(user_id)
        if not state:
            return [[('Новая игра', 'doom:restart')]]
        if state.hp <= 0:
            return [[('Новая игра', 'doom:restart')]]
        if state.in_fight:
            row1 = []
            if state.ammo > 0:
                row1.append(('Огонь', 'doom:shoot'))
            else:
                row1.append(('Перезарядка', 'doom:reload'))
            row1.append(('Нож', 'doom:knife'))
            row2 = [('Бежать', 'doom:run')]
            if state.medkits > 0 and state.hp < 100:
                row2.insert(0, ('Аптечка', 'doom:med'))
            row3 = [('⟳ Кадр', 'doom:tick'), ('Выход', 'doom:exit')]
            return [row1, row2, row3]
        row1 = [('Вперёд', 'doom:up'), ('Поворот ◀', 'doom:left'), ('Поворот ▶', 'doom:right'), ('Назад', 'doom:down')]
        row2 = [('Осмотреться', 'doom:scan'), ('⟳ Кадр', 'doom:tick')]
        if state.medkits > 0 and state.hp < 100:
            row2.append(('Аптечка', 'doom:med'))
        row3 = [('Выход', 'doom:exit')]
        return [row1, row2, row3]

    def _move(self, state: DoomState, direction: str) -> None:
        step = self._MOVE_STEP
        if direction == 'down':
            step = -step
        nx = state.x + math.cos(state.direction) * step
        ny = state.y + math.sin(state.direction) * step
        if self._is_wall(nx, ny):
            state.log = 'Стена впереди.'
            return
        state.x = nx
        state.y = ny
        state.steps += 1
        state.log = 'Ты продвигаешься по коридору.'

    def _turn(self, state: DoomState, direction: str) -> None:
        step = self._TURN_STEP if direction == 'right' else -self._TURN_STEP
        state.direction = (state.direction + step) % (math.tau)
        state.steps += 1
        state.log = 'Поворот в коридоре.'

    def _is_wall(self, x: float, y: float) -> bool:
        ix = int(x)
        iy = int(y)
        if ix < 0 or iy < 0 or iy >= self._MAP_H or ix >= self._MAP_W:
            return True
        return self._MAP[iy][ix] == '#'

    def _cast_ray(self, x: float, y: float, angle: float) -> float:
        step = 0.05
        dist = 0.0
        while dist < self._MAX_DEPTH:
            rx = x + math.cos(angle) * dist
            ry = y + math.sin(angle) * dist
            if self._is_wall(rx, ry):
                return dist
            dist += step
        return self._MAX_DEPTH

    def _wrap_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _scale_sprite(self, sprite: list[str], target_h: int) -> list[str]:
        target_h = max(3, target_h)
        src_h = len(sprite)
        src_w = max(len(row) for row in sprite)
        target_w = max(3, int(src_w * target_h / src_h))
        out: list[str] = []
        for y in range(target_h):
            src_y = int(y / target_h * src_h)
            row = sprite[src_y]
            line = []
            for x in range(target_w):
                src_x = int(x / target_w * src_w)
                ch = row[src_x] if src_x < len(row) else ' '
                line.append(ch)
            out.append(''.join(line))
        return out

    def _enemy_visible(self, state: DoomState) -> bool:
        if not state.in_fight:
            return False
        dx = state.enemy_x - state.x
        dy = state.enemy_y - state.y
        dist = math.hypot(dx, dy)
        if dist <= 0.3:
            return True
        angle = math.atan2(dy, dx)
        rel = self._wrap_angle(angle - state.direction)
        return abs(rel) < (self._FOV * 0.35)

    def _maybe_spawn_enemy(self, state: DoomState, chance: float) -> None:
        if state.in_fight:
            return
        if self._rng.random() > chance:
            return
        name, hp_min, hp_max = self._rng.choice(self._ENEMIES)
        for _ in range(12):
            dist = self._rng.uniform(3.0, 6.0)
            angle = state.direction + self._rng.uniform(-0.45, 0.45)
            ex = state.x + math.cos(angle) * dist
            ey = state.y + math.sin(angle) * dist
            if self._is_wall(ex, ey):
                continue
            state.in_fight = True
            state.enemy = name
            state.enemy_hp = self._rng.randint(hp_min, hp_max)
            state.enemy_x = ex
            state.enemy_y = ey
            state.log = f'Тревога! Перед тобой {name}.'
            return

    def _enemy_attack(self, state: DoomState, bonus: int = 0) -> None:
        if not state.in_fight or state.enemy_hp <= 0:
            return
        damage = self._rng.randint(6 + bonus, 16 + bonus)
        state.hp -= damage
        dx = state.x - state.enemy_x
        dy = state.y - state.enemy_y
        dist = math.hypot(dx, dy)
        if dist > 0.6:
            step = 0.15
            nx = state.enemy_x + (dx / dist) * step
            ny = state.enemy_y + (dy / dist) * step
            if not self._is_wall(nx, ny):
                state.enemy_x = nx
                state.enemy_y = ny
        if state.hp <= 0:
            state.hp = 0
            state.log += f' {state.enemy} добивает тебя ({damage}).'
        else:
            state.log += f' Ответный удар: -{damage} HP.'

    def _render(self, state: DoomState) -> str:
        raw_map = self._render_map(state)
        map_block = raw_map.replace('&', '&amp;').replace('<', '&lt;')
        heading = int((math.degrees(state.direction) % 360))
        status = f'HP {state.hp} | Ammo {state.ammo} | Med {state.medkits} | Kills {state.kills} | Steps {state.steps} | Dir {heading}°'
        pos = f'Position ({state.x:.1f}, {state.y:.1f})'
        lines = [
            'DOOM // UAC TEST',
            status,
            pos,
        ]
        if state.in_fight:
            lines.append(f'Enemy: {state.enemy} ({state.enemy_hp} HP)')
        lines.extend(['', 'View:', f'<pre>{map_block}</pre>', '', state.log])
        if state.flash > 0:
            state.flash = max(0, state.flash - 1)
        state.frame += 1
        return '\n'.join(lines)

    def _render_map(self, state: DoomState) -> str:
        width = 41
        height = 17
        buffer = [[' ' for _ in range(width)] for _ in range(height)]
        mid = height // 2
        front_dist = self._cast_ray(state.x, state.y, state.direction)
        depth_ratio = min(front_dist / max(self._MAX_DEPTH, 1e-6), 1.0)
        frames = [
            (0, width - 1, 0, height - 1),
            (3, width - 4, 2, height - 3),
            (6, width - 7, 4, height - 5),
            (9, width - 10, 6, height - 7),
        ]
        max_level = len(frames) - 1
        front_level = min(max_level, int(depth_ratio * max_level))

        for row in range(mid, height):
            for col in range(width):
                buffer[row][col] = '.'

        for lvl in range(front_level):
            l1, r1, t1, b1 = frames[lvl]
            l2, r2, t2, b2 = frames[lvl + 1]
            for row in range(t1, b1 + 1):
                if t1 == b1:
                    ratio = 0.0
                else:
                    ratio = (row - t1) / (b1 - t1)
                left = int(l1 + (l2 - l1) * ratio)
                right = int(r1 + (r2 - r1) * ratio)
                if 0 <= row < height:
                    if 0 <= left < width:
                        buffer[row][left] = '/'
                    if 0 <= right < width:
                        buffer[row][right] = '\\'
            for col in range(l1, r1 + 1):
                if 0 <= t1 < height:
                    buffer[t1][col] = '-'
                if 0 <= b1 < height:
                    buffer[b1][col] = '-'

        fl, fr, ft, fb = frames[front_level]
        for row in range(ft + 1, fb):
            for col in range(fl + 1, fr):
                buffer[row][col] = '#'
        for col in range(fl, fr + 1):
            buffer[ft][col] = '+'
            buffer[fb][col] = '+'
        for row in range(ft, fb + 1):
            buffer[row][fl] = '|'
            buffer[row][fr] = '|'

        if state.in_fight and state.enemy and self._enemy_visible(state):
            dx = state.enemy_x - state.x
            dy = state.enemy_y - state.y
            dist = math.hypot(dx, dy)
            size = max(3, int((height * 0.7) * (1 - min(dist / self._MAX_DEPTH, 1.0))))
            sprite_a = [
                r' /"""\\ ',
                r'| o o |',
                r'|  ^  |',
                r'| \\_/ |',
                r' \___/ ',
            ]
            sprite_b = [
                r' /"""\\ ',
                r'| - - |',
                r'|  o  |',
                r'| \\_/ |',
                r' \___/ ',
            ]
            sprite = sprite_a if state.frame % 2 == 0 else sprite_b
            scaled = self._scale_sprite(sprite, size)
            sprite_h = len(scaled)
            sprite_w = len(scaled[0])
            center_x = (fl + fr) // 2
            center_y = (ft + fb) // 2
            top = center_y - sprite_h // 2
            left = center_x - sprite_w // 2
            for sy, row in enumerate(scaled):
                ry = top + sy
                if 0 <= ry < height:
                    for sx, ch in enumerate(row):
                        cx = left + sx
                        if 0 <= cx < width and ch != ' ':
                            buffer[ry][cx] = ch

        cross_y = mid
        cross_x = width // 2
        if 0 <= cross_y < height:
            buffer[cross_y][cross_x] = '+' if state.frame % 6 else 'x'
            if cross_x - 1 >= 0:
                buffer[cross_y][cross_x - 1] = '-'
            if cross_x + 1 < width:
                buffer[cross_y][cross_x + 1] = '-'
            if cross_y - 1 >= 0:
                buffer[cross_y - 1][cross_x] = '|'
            if cross_y + 1 < height:
                buffer[cross_y + 1][cross_x] = '|'

        weapon_row = height - 2 - (state.frame % 2)
        weapon = '==[====]=>'
        if state.flash > 0:
            weapon = '==[====]=>*'
        weapon_left = width // 2 - len(weapon) // 2
        if 0 <= weapon_row < height:
            for i, ch in enumerate(weapon):
                cx = weapon_left + i
                if 0 <= cx < width:
                    buffer[weapon_row][cx] = ch

        return '\n'.join(''.join(row) for row in buffer)
