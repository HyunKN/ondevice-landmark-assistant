from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
METRICS = json.loads((ROOT / "data" / "metrics.json").read_text(encoding="utf-8"))

NAVY = "#0B1220"
NAVY_2 = "#111C31"
INK = "#E8EEF8"
MUTED = "#9FB0C8"
TEAL = "#3DD6C6"
BLUE = "#68A4FF"
AMBER = "#F6C453"
RED = "#FF7A7A"
GRID = "#273650"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_document(width: int, height: int, body: str, title: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{esc(title)}</title>
  <desc id="desc">{esc(title)}</desc>
  <style>
    text {{ font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  </style>
{body}
</svg>
'''


def write_svg(filename: str, width: int, height: int, body: str, title: str) -> None:
    path = ASSETS / filename
    path.write_text(svg_document(width, height, body, title), encoding="utf-8", newline="\n")
    print(path.relative_to(ROOT).as_posix())


def generate_hero() -> None:
    body = f'''  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{NAVY}"/>
      <stop offset="1" stop-color="#152844"/>
    </linearGradient>
    <linearGradient id="line" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{TEAL}"/>
      <stop offset="1" stop-color="{BLUE}"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="360" rx="28" fill="url(#bg)"/>
  <circle cx="1060" cy="68" r="180" fill="#1B355B" opacity="0.55"/>
  <circle cx="1120" cy="320" r="150" fill="#123D49" opacity="0.45"/>
  <path d="M830 84h190l70 70-70 70H830l-70-70z" fill="none" stroke="url(#line)" stroke-width="3" opacity="0.8"/>
  <circle cx="830" cy="84" r="8" fill="{TEAL}"/>
  <circle cx="1090" cy="154" r="8" fill="{BLUE}"/>
  <circle cx="830" cy="224" r="8" fill="{TEAL}"/>
  <text x="76" y="78" fill="{TEAL}" font-size="18" font-weight="700" letter-spacing="2">AI / COMPUTER VISION / MOBILE</text>
  <text x="76" y="154" fill="{INK}" font-size="52" font-weight="750">On-device Landmark Assistant</text>
  <text x="76" y="206" fill="{MUTED}" font-size="24">Model experiments to Android on-device inference</text>
  <rect x="76" y="260" width="480" height="48" rx="24" fill="#14243C" stroke="{GRID}"/>
  <text x="104" y="291" fill="{INK}" font-size="18">MobileCLIP2 · ONNX Runtime · Flutter</text>
'''
    write_svg("hero.svg", 1200, 360, body, "On-device Landmark Assistant")


def generate_experiment_chart() -> None:
    configurations = METRICS["configurations"]
    width, height = 1200, 660
    plot_x, plot_width = 300, 790
    baseline, ceiling = 94.0, 100.0
    parts = [
        f'  <rect width="{width}" height="{height}" rx="24" fill="{NAVY}"/>',
        f'  <text x="64" y="68" fill="{INK}" font-size="30" font-weight="700">Validation Top-1 across 8 configurations</text>',
        f'  <text x="64" y="102" fill="{MUTED}" font-size="17">Five-fold validation mean · axis starts at 94% to show the measured differences</text>',
    ]
    for tick in range(94, 101):
        x = plot_x + (tick - baseline) / (ceiling - baseline) * plot_width
        parts.append(f'  <line x1="{x:.1f}" y1="132" x2="{x:.1f}" y2="606" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'  <text x="{x:.1f}" y="630" fill="{MUTED}" font-size="14" text-anchor="middle">{tick}%</text>')

    for index, item in enumerate(configurations):
        y = 148 + index * 56
        value = float(item["val_top1"])
        bar_width = (value - baseline) / (ceiling - baseline) * plot_width
        color = TEAL if index == 0 else BLUE
        opacity = "1" if index == 0 else "0.78"
        label = esc(item["name"])
        parts.extend(
            [
                f'  <text x="278" y="{y + 24}" fill="{INK}" font-size="16" text-anchor="end">{label}</text>',
                f'  <rect x="{plot_x}" y="{y}" width="{bar_width:.1f}" height="32" rx="8" fill="{color}" opacity="{opacity}"/>',
                f'  <text x="{plot_x + bar_width + 12:.1f}" y="{y + 23}" fill="{INK}" font-size="16" font-weight="700">{value:.2f}%</text>',
            ]
        )
    write_svg("experiment-comparison.svg", width, height, "\n".join(parts), "Validation Top-1 comparison")


def generate_deployment_flow() -> None:
    nodes = [
        (54, 190, 164, "Image/Text", "Input"),
        (250, 190, 180, "MobileCLIP2", "Encoders"),
        (462, 190, 190, "Prototype/Text", "Index"),
        (684, 190, 180, "Confidence", "Policy"),
        (896, 190, 150, "Flutter", "UI"),
    ]
    parts = [
        f'  <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{TEAL}"/></marker></defs>',
        f'  <rect width="1200" height="430" rx="24" fill="{NAVY}"/>',
        f'  <text x="54" y="68" fill="{INK}" font-size="30" font-weight="700">Deployment flow</text>',
        f'  <text x="54" y="102" fill="{MUTED}" font-size="17">The app shares one embedding contract across image and natural-language search.</text>',
    ]
    for index in range(len(nodes) - 1):
        x, y, w, _, _ = nodes[index]
        nx = nodes[index + 1][0]
        parts.append(f'  <line x1="{x + w}" y1="245" x2="{nx - 16}" y2="245" stroke="{TEAL}" stroke-width="3" marker-end="url(#arrow)"/>')
    for x, y, w, top, bottom in nodes:
        parts.extend(
            [
                f'  <rect x="{x}" y="{y}" width="{w}" height="110" rx="18" fill="{NAVY_2}" stroke="{GRID}" stroke-width="2" aria-label="{esc(top)} {esc(bottom)}"/>',
                f'  <text x="{x + w / 2:.1f}" y="{y + 48}" fill="{INK}" font-size="17" font-weight="650" text-anchor="middle">{esc(top)}</text>',
                f'  <text x="{x + w / 2:.1f}" y="{y + 76}" fill="{MUTED}" font-size="16" text-anchor="middle">{esc(bottom)}</text>',
            ]
        )
    parts.extend(
        [
            f'  <line x1="971" y1="300" x2="971" y2="342" stroke="{BLUE}" stroke-width="3" marker-end="url(#arrow)"/>',
            f'  <rect x="860" y="350" width="222" height="52" rx="15" fill="#152744" stroke="{GRID}"/>',
            f'  <text x="971" y="383" fill="{INK}" font-size="16" text-anchor="middle">Local/Server Logs</text>',
        ]
    )
    write_svg("deployment-flow.svg", 1200, 430, "\n".join(parts), "Android deployment flow")


def generate_npu_evidence() -> None:
    latencies = METRICS["npu_latency_ms"]
    rows = (
        ("Snapdragon 8 Gen 2", latencies["snapdragon_8_gen_2"]),
        ("Snapdragon 8 Gen 3", latencies["snapdragon_8_gen_3"]),
        ("Snapdragon 8 Elite", latencies["snapdragon_8_elite"]),
    )
    parts = [
        f'  <rect width="1200" height="430" rx="24" fill="{NAVY}"/>',
        f'  <text x="58" y="68" fill="{INK}" font-size="30" font-weight="700">NPU feasibility measurements</text>',
        f'  <text x="58" y="102" fill="{MUTED}" font-size="17">Warm latency from the tested quantized path</text>',
    ]
    max_latency = max(value for _, value in rows)
    for index, (label, value) in enumerate(rows):
        y = 148 + index * 66
        bar_width = value / max_latency * 520
        parts.extend(
            [
                f'  <text x="58" y="{y + 23}" fill="{INK}" font-size="17">{esc(label)}</text>',
                f'  <rect x="256" y="{y}" width="{bar_width:.1f}" height="32" rx="8" fill="{BLUE}" opacity="0.82"/>',
                f'  <text x="{268 + bar_width:.1f}" y="{y + 23}" fill="{INK}" font-size="16" font-weight="700">{value:.2f} ms</text>',
            ]
        )
    parts.extend(
        [
            f'  <rect x="58" y="350" width="1084" height="54" rx="14" fill="#3B2D11" stroke="{AMBER}"/>',
            f'  <circle cx="88" cy="377" r="8" fill="{AMBER}"/>',
            f'  <text x="110" y="383" fill="{INK}" font-size="18" font-weight="650">Quantized accuracy collapsed — latency is feasibility evidence only</text>',
        ]
    )
    write_svg("npu-evidence.svg", 1200, 430, "\n".join(parts), "NPU latency and accuracy caveat")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    generate_hero()
    generate_experiment_chart()
    generate_deployment_flow()
    generate_npu_evidence()


if __name__ == "__main__":
    main()
