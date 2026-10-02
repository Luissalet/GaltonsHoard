"""Vision cases drawn with Pillow from a seed: counting shapes, naming a colour, reading text, a bar chart, a table cell and clocks.

``plan`` (cheap) gives the question, the checker and the reference answer; ``render`` draws the PNG. Both derive from the same
parameters, so a case is a few numbers in the database and the picture is rebuilt when a model needs it (or the UI shows it).
"""

from __future__ import annotations

import io
import math
import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ..util import stable_hash

W, H = 640, 480
COLORS = {"rojo": (214, 40, 40), "azul": (35, 80, 215), "verde": (30, 150, 70), "amarillo": (240, 200, 20), "morado": (135, 55, 175), "naranja": (240, 125, 15)}
SHAPES = {"círculo": "círculos", "cuadrado": "cuadrados", "triángulo": "triángulos"}
_PHRASES = ["Salida de emergencia", "Cerrado por vacaciones", "Biblioteca municipal", "Prohibido el paso", "Oferta de temporada", "Reunión en la sala azul", "Mercado de los jueves"]
_PRODUCTS = ["Manzanas", "Naranjas", "Peras", "Uvas", "Ciruelas", "Melones"]


def _rng(params: dict[str, Any], tag: str) -> random.Random:
    return random.Random(stable_hash(f"{tag}:{params.get('task')}:{params.get('seed', 0)}")[:12])


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError):  # very old Pillow: a fixed-size bitmap font
        return ImageFont.load_default()


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ----------------------------------------------------------------------------------------------- shapes
def _layout(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Shapes on a 4x3 grid (no overlaps). The target colour and shape appear exactly ``n_target`` times."""
    rng = _rng(params, "layout")
    cells = [(c, r) for c in range(4) for r in range(3)]
    rng.shuffle(cells)
    target_color, target_shape, n = params["color"], params["shape"], int(params["n_target"])
    items = [{"color": target_color, "shape": target_shape} for _ in range(n)]
    others = [(c, s) for c in COLORS for s in SHAPES if not (c == target_color and s == target_shape)]
    for _ in range(int(params.get("n_other", 5))):
        items.append(dict(zip(("color", "shape"), rng.choice(others))))
    rng.shuffle(items)
    out = []
    for item, (cx, cy) in zip(items, cells):
        out.append({**item, "x": 80 + cx * 160 + rng.randint(-18, 18), "y": 80 + cy * 150 + rng.randint(-18, 18), "size": rng.randint(46, 62)})
    return out


def _draw_shape(d: ImageDraw.ImageDraw, item: dict[str, Any]) -> None:
    x, y, s, fill = item["x"], item["y"], item["size"], COLORS[item["color"]]
    if item["shape"] == "círculo":
        d.ellipse([x - s // 2, y - s // 2, x + s // 2, y + s // 2], fill=fill)
    elif item["shape"] == "cuadrado":
        d.rectangle([x - s // 2, y - s // 2, x + s // 2, y + s // 2], fill=fill)
    else:
        d.polygon([(x, y - s // 2), (x - s // 2, y + s // 2), (x + s // 2, y + s // 2)], fill=fill)


def _render_shapes(params: dict[str, Any]) -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for item in _layout(params):
        _draw_shape(d, item)
    return img


def _plan_count(params: dict[str, Any]) -> dict[str, Any]:
    shape, color, n = params["shape"], params["color"], int(params["n_target"])
    plural = SHAPES[shape]
    return {"question": f"¿Cuántos {plural} de color {color} hay en la imagen? Responde solo con un número.", "checker": {"type": "number", "expected": n, "which": "marked"}, "reference": str(n)}


def _render_circle_color(params: dict[str, Any]) -> Image.Image:
    rng = _rng(params, "circle")
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    cells = [(c, r) for c in range(4) for r in range(3)]
    rng.shuffle(cells)
    others = [c for c in COLORS if c != params["color"]]
    kinds = ["circle"] + ["cuadrado"] * 2 + ["triángulo"] * 2
    for kind, (cx, cy) in zip(kinds, cells):
        x, y, s = 80 + cx * 160 + rng.randint(-15, 15), 80 + cy * 150 + rng.randint(-15, 15), rng.randint(50, 64)
        item = {"x": x, "y": y, "size": s, "shape": "círculo" if kind == "circle" else kind, "color": params["color"] if kind == "circle" else rng.choice(others)}
        _draw_shape(d, item)
    return img


def _plan_circle(params: dict[str, Any]) -> dict[str, Any]:
    color = params["color"]
    return {"question": "En la imagen hay un único círculo, rodeado de cuadrados y triángulos. ¿De qué color es el círculo? Responde con una sola palabra.",
            "checker": {"type": "contains", "all": [color], "none": [c for c in COLORS if c != color], "words": True, "ignore_accents": True}, "reference": color}


# ----------------------------------------------------------------------------------------------- text
def _text_value(params: dict[str, Any]) -> str:
    rng = _rng(params, "text")
    if params.get("style") == "code":
        return f"{rng.choice(['AZUL', 'NORTE', 'LIMA', 'ROBLE'])}-{rng.randint(1000, 9999)}"
    return rng.choice(_PHRASES)


def _render_text(params: dict[str, Any]) -> Image.Image:
    rng = _rng(params, "textstyle")
    bg = rng.choice([(255, 255, 255), (245, 240, 220), (225, 235, 250)])
    img = Image.new("RGB", (W, 240), bg)
    d = ImageDraw.Draw(img)
    f = font(46 if params.get("style") == "code" else 40)
    text = _text_value(params)
    box = d.textbbox((0, 0), text, font=f)
    d.text(((W - (box[2] - box[0])) / 2 - box[0], (240 - (box[3] - box[1])) / 2 - box[1]), text, fill=(20, 20, 20), font=f)
    return img


def _plan_text(params: dict[str, Any]) -> dict[str, Any]:
    text = _text_value(params)
    what = "el código" if params.get("style") == "code" else "el texto"
    return {"question": f"Transcribe exactamente {what} que aparece en la imagen. Responde solo con la transcripción.",
            "checker": {"type": "contains", "all": [text], "ignore_accents": True}, "reference": text}


# ----------------------------------------------------------------------------------------------- bar chart
def _bars(params: dict[str, Any]) -> dict[str, int]:
    rng = _rng(params, "bars")
    return {label: rng.randrange(10, 100, 5) for label in "ABCD"}


def _render_bars(params: dict[str, Any]) -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    left, right, top, bottom = 80, W - 30, 40, H - 60
    f = font(20)
    for v in range(0, 101, 20):
        y = bottom - (bottom - top) * v / 100
        d.line([(left, y), (right, y)], fill=(205, 205, 205), width=1)
        d.text((left - 12, y), str(v), fill=(40, 40, 40), font=f, anchor="rm")
    d.line([(left, top), (left, bottom)], fill=(40, 40, 40), width=2)
    d.line([(left, bottom), (right, bottom)], fill=(40, 40, 40), width=2)
    bars = _bars(params)
    slot = (right - left) / len(bars)
    palette = [COLORS["azul"], COLORS["naranja"], COLORS["verde"], COLORS["morado"]]
    for i, (label, value) in enumerate(bars.items()):
        x0 = left + slot * i + slot * 0.2
        x1 = left + slot * (i + 1) - slot * 0.2
        y = bottom - (bottom - top) * value / 100
        d.rectangle([x0, y, x1, bottom], fill=palette[i])
        d.text(((x0 + x1) / 2, bottom + 18), label, fill=(20, 20, 20), font=font(26), anchor="mm")
    return img


def _plan_bars(params: dict[str, Any]) -> dict[str, Any]:
    label = params["label"]
    value = _bars(params)[label]
    return {"question": f"En el gráfico de barras, ¿qué valor tiene la barra {label}? El eje vertical va de 0 a 100. Responde solo con un número (aproximado a la marca más cercana de 5).",
            "checker": {"type": "number", "expected": value, "tolerance": {"abs": 5}, "which": "marked"}, "reference": str(value)}


# ----------------------------------------------------------------------------------------------- table
def _table(params: dict[str, Any]) -> list[list[str]]:
    rng = _rng(params, "table")
    rows = rng.sample(_PRODUCTS, 4)
    return [["Producto", "Precio", "Unidades"]] + [[name, f"{rng.randint(1, 9)},{rng.choice(['20', '50', '80', '95'])}", str(rng.randint(12, 480))] for name in rows]


def _render_table(params: dict[str, Any]) -> Image.Image:
    data = _table(params)
    img = Image.new("RGB", (W, 360), "white")
    d = ImageDraw.Draw(img)
    f = font(26)
    col_w, row_h, x0, y0 = [230, 190, 160], 56, 30, 30
    for r, row in enumerate(data):
        x = x0
        for c, cell in enumerate(row):
            fill = (225, 232, 245) if r == 0 else "white"
            d.rectangle([x, y0 + r * row_h, x + col_w[c], y0 + (r + 1) * row_h], fill=fill, outline=(60, 60, 60), width=2)
            d.text((x + 14, y0 + r * row_h + row_h / 2), cell, fill=(20, 20, 20), font=f, anchor="lm")
            x += col_w[c]
    return img


def _plan_table(params: dict[str, Any]) -> dict[str, Any]:
    data = _table(params)
    r, c = int(params["row"]), int(params["col"])
    value = data[r][c]
    return {"question": f"En la tabla, ¿qué valor aparece en la columna «{data[0][c]}» de la fila de «{data[r][0]}»? Responde solo con el valor.",
            "checker": {"type": "contains", "any": [value, value.replace(",", ".")], "words": False}, "reference": value}


# ----------------------------------------------------------------------------------------------- clocks
def _digital_times(params: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int]]:
    rng = _rng(params, "clocks")
    a = rng.randrange(0, 24 * 60 - 300, 5)
    b = a + rng.randrange(20, 300, 5)
    return (a // 60, a % 60), (b // 60, b % 60)


def _render_digital(params: dict[str, Any]) -> Image.Image:
    (h1, m1), (h2, m2) = _digital_times(params)
    img = Image.new("RGB", (W, 280), (245, 245, 245))
    d = ImageDraw.Draw(img)
    f, small = font(80), font(26)
    for i, (label, h, m) in enumerate((("Reloj A", h1, m1), ("Reloj B", h2, m2))):
        x0 = 30 + i * 300
        d.rounded_rectangle([x0, 40, x0 + 280, 230], radius=14, fill=(15, 20, 15))
        d.text((x0 + 140, 85), f"{h:02d}:{m:02d}", fill=(80, 255, 120), font=f, anchor="mm")
        d.text((x0 + 140, 190), label, fill=(190, 190, 190), font=small, anchor="mm")
    return img


def _plan_digital(params: dict[str, Any]) -> dict[str, Any]:
    (h1, m1), (h2, m2) = _digital_times(params)
    diff = (h2 * 60 + m2) - (h1 * 60 + m1)
    return {"question": "La imagen muestra dos relojes digitales de 24 horas, A y B, del mismo día. ¿Cuántos minutos más tarde marca el reloj B que el reloj A? Responde solo con el número de minutos.",
            "checker": {"type": "number", "expected": diff, "which": "marked"}, "reference": str(diff)}


def _analog_time(params: dict[str, Any]) -> tuple[int, int]:
    rng = _rng(params, "analog")
    return rng.randint(1, 12), rng.choice([0, 30])


def _render_analog(params: dict[str, Any]) -> Image.Image:
    h, m = _analog_time(params)
    img = Image.new("RGB", (420, 420), "white")
    d = ImageDraw.Draw(img)
    cx = cy = 210
    d.ellipse([cx - 190, cy - 190, cx + 190, cy + 190], outline=(30, 30, 30), width=6, fill=(250, 250, 245))
    f = font(34)
    for n in range(1, 13):
        ang = math.radians(n * 30 - 90)
        d.text((cx + 155 * math.cos(ang), cy + 155 * math.sin(ang)), str(n), fill=(20, 20, 20), font=f, anchor="mm")
    for tick in range(60):
        ang = math.radians(tick * 6 - 90)
        r0 = 178 if tick % 5 else 172
        d.line([(cx + r0 * math.cos(ang), cy + r0 * math.sin(ang)), (cx + 186 * math.cos(ang), cy + 186 * math.sin(ang))], fill=(30, 30, 30), width=2)
    hour_ang = math.radians(((h % 12) + m / 60) * 30 - 90)
    min_ang = math.radians(m * 6 - 90)
    d.line([(cx, cy), (cx + 85 * math.cos(hour_ang), cy + 85 * math.sin(hour_ang))], fill=(20, 20, 20), width=11)
    d.line([(cx, cy), (cx + 135 * math.cos(min_ang), cy + 135 * math.sin(min_ang))], fill=(20, 20, 20), width=6)
    d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(20, 20, 20))
    return img


def _plan_analog(params: dict[str, Any]) -> dict[str, Any]:
    h, m = _analog_time(params)
    hours = [f"{h:02d}" if h > 9 else f"0?{h}"] + ([str(h + 12)] if h < 12 else ["0", "00"])
    pattern = rf"(?<!\d)(?:{'|'.join(hours)})\s?[:.h]?\s?{m:02d}(?!\d)" if m else rf"(?<!\d)(?:{'|'.join(hours)})\s?(?::00|\.00|h|\s?en punto)"
    return {"question": "¿Qué hora marca el reloj analógico? Responde en formato H:MM (reloj de 12 horas), por ejemplo 4:30.", "checker": {"type": "regex", "pattern": pattern, "extract": "last_line"},
            "reference": f"{h}:{m:02d}"}


TASKS = {
    "count": (_render_shapes, _plan_count), "circle_color": (_render_circle_color, _plan_circle), "text": (_render_text, _plan_text),
    "bars": (_render_bars, _plan_bars), "table": (_render_table, _plan_table), "digital": (_render_digital, _plan_digital), "analog": (_render_analog, _plan_analog),
}


def plan(params: dict[str, Any]) -> dict[str, Any]:
    task = TASKS.get(params.get("task", ""))
    if task is None:
        raise ValueError(f"unknown vision task {params.get('task')!r}")
    return task[1](params)


def render(params: dict[str, Any]) -> bytes:
    """The PNG of the case."""
    task = TASKS.get(params.get("task", ""))
    if task is None:
        raise ValueError(f"unknown vision task {params.get('task')!r}")
    return _png(task[0](params))


def suite_def(seed: int = 4242) -> dict[str, Any]:
    specs: list[tuple[str, dict[str, Any]]] = []
    for i, (color, shape, n) in enumerate([("rojo", "círculo", 3), ("azul", "cuadrado", 4), ("verde", "triángulo", 2), ("amarillo", "círculo", 5)]):
        specs.append((f"Contar {SHAPES[shape]} de color {color} ({n})", {"task": "count", "color": color, "shape": shape, "n_target": n, "n_other": 5, "seed": seed + i}))
    for i, color in enumerate(["morado", "naranja", "azul"]):
        specs.append((f"Color del círculo ({color})", {"task": "circle_color", "color": color, "seed": seed + 20 + i}))
    for i, style in enumerate(["code", "phrase", "code", "phrase"]):
        specs.append((f"Leer texto renderizado {i + 1}", {"task": "text", "style": style, "seed": seed + 40 + i}))
    for i, label in enumerate("BCD"):
        specs.append((f"Valor de la barra {label}", {"task": "bars", "label": label, "seed": seed + 60 + i}))
    for i, (row, col) in enumerate([(1, 1), (3, 2), (2, 0)]):
        specs.append((f"Celda de una tabla ({row},{col})", {"task": "table", "row": row, "col": col, "seed": seed + 80 + i}))
    for i in range(2):
        specs.append((f"Diferencia entre dos relojes digitales {i + 1}", {"task": "digital", "seed": seed + 100 + i}))
    specs.append(("Hora de un reloj analógico", {"task": "analog", "seed": seed + 120}))
    cases = []
    for title, params in specs:
        params = {"kind": "vision", **params}
        p = plan(params)
        cases.append({"title": title, "prompt": {"text": p["question"], "generate": params}, "checker": p["checker"], "reference": p["reference"], "tags": ["visión", params["task"]]})
    return {"id": "vision", "name": "Visión", "category": "vision", "version": 1, "max_tokens": 200, "position": 12,
            "description": "Imágenes dibujadas con una semilla fija: contar figuras, color, texto, un gráfico de barras, una tabla y relojes. Solo se ejecuta con modelos con visión.",
            "cases": cases}
