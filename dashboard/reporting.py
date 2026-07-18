"""PDF report generation for dashboard exports"""

import textwrap
import zlib
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from PIL import Image


LOGO_PATH = Path(__file__).resolve().parent / "assets" / "embio-black-logo.png"


def _pdf_escape(text: str) -> str:
    clean = str(text or "").replace("\u2013", "-").replace("\u2014", "-").replace("\u2022", "-")
    clean = clean.encode("latin-1", "replace").decode("latin-1")
    return clean.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_text_line(text: str, x: int, y: int, size: int = 10, font: str = "F1", color: str = "#121820") -> str:
    rgb = _pdf_rgb(color)
    return f"q {rgb} rg BT /{font} {size} Tf {x} {y} Td ({_pdf_escape(text)}) Tj ET Q\n"


def _pdf_rgb(hex_color: str) -> str:
    cleaned = hex_color.lstrip("#")
    r = int(cleaned[0:2], 16) / 255
    g = int(cleaned[2:4], 16) / 255
    b = int(cleaned[4:6], 16) / 255
    return f"{r:.3f} {g:.3f} {b:.3f}"


def _pdf_rect(x: int, y: int, width: int, height: int, fill: str, stroke: str | None = None) -> str:
    fill_rgb = _pdf_rgb(fill)
    if stroke:
        stroke_rgb = _pdf_rgb(stroke)
        return f"q {fill_rgb} rg {stroke_rgb} RG {x} {y} {width} {height} re B Q\n"
    return f"q {fill_rgb} rg {x} {y} {width} {height} re f Q\n"


def _pdf_line(x1: int, y1: int, x2: int, y2: int, color: str = "#E6E8F2") -> str:
    rgb = _pdf_rgb(color)
    return f"q {rgb} RG 1 w {x1} {y1} m {x2} {y2} l S Q\n"


def _pdf_image(name: str, x: int, y: int, width: int, height: int) -> str:
    return f"q {width} 0 0 {height} {x} {y} cm /{name} Do Q\n"


def _load_pdf_logo() -> tuple[bytes, bytes, int, int]:
    image = Image.open(LOGO_PATH).convert("RGBA")
    red, green, blue, alpha = image.split()
    rgb = Image.merge("RGB", (red, green, blue)).tobytes()
    return zlib.compress(rgb), zlib.compress(alpha.tobytes()), image.width, image.height


def _pdf_wrapped_lines(text: str, width: int = 94) -> list[str]:
    one_line = " ".join(str(text or "").split())
    return textwrap.wrap(one_line, width=width) or [""]


def build_weekly_pdf(results_df: pd.DataFrame, generated_on: date | None = None) -> bytes:
    """to build a concise supervisor-facing weekly insights PDF"""
    generated_on = generated_on or date.today()
    week_start = generated_on - timedelta(days=7)

    dated = results_df.copy()
    dated["_date"] = pd.to_datetime(dated["item_date"], errors="coerce").dt.date
    weekly = dated[dated["_date"].isna() | (dated["_date"] >= week_start)].copy()
    fallback = weekly.empty
    scored_df = dated if fallback else weekly
    report_df = scored_df.sort_values("score", ascending=False).head(10)

    all_keywords = [kw for kws in scored_df["matched_keywords"].dropna() for kw in kws]
    top_keywords = ", ".join(kw for kw, _ in Counter(all_keywords).most_common(8)) or "No dominant keyword themes"
    source_mix = scored_df["source_type"].value_counts().to_dict()
    high_count = int((scored_df["score"] >= 0.65).sum()) if not scored_df.empty else 0
    medium_count = int(((scored_df["score"] >= 0.50) & (scored_df["score"] < 0.65)).sum()) if not scored_df.empty else 0
    low_count = int((scored_df["score"] < 0.50).sum()) if not scored_df.empty else 0
    top_score = float(scored_df["score"].max()) if not scored_df.empty else 0.0
    avg_score = float(scored_df["score"].mean()) if not scored_df.empty else 0.0

    header = []
    header.append(_pdf_rect(0, 724, 612, 68, "#396070"))
    header.append(_pdf_text_line("Weekly Insights Brief", 48, 762, 20, "F2", "#FFFFFF"))
    header.append(_pdf_text_line(f"Generated {generated_on.isoformat()} | Window: {week_start.isoformat()} to {generated_on.isoformat()}", 48, 742, 9, "F1", "#FFFFFF"))
    header.append(_pdf_rect(434, 746, 130, 18, "#FFFFFF"))
    header.append(_pdf_image("Logo", 442, 750, 114, 14))

    card_y = 654
    cards = [
        ("Records", f"{len(scored_df)}", "#C5CAFB"),
        ("High relevance", f"{high_count}", "#E8F7EF"),
        ("Top score", f"{top_score:.2f}", "#FFF6D8"),
        ("Avg score", f"{avg_score:.2f}", "#F6F7FF"),
    ]
    for idx, (label, value, color) in enumerate(cards):
        x = 48 + idx * 126
        header.append(_pdf_rect(x, card_y, 112, 52, color, "#E6E8F2"))
        header.append(_pdf_text_line(label, x + 10, card_y + 32, 8, "F1"))
        header.append(_pdf_text_line(value, x + 10, card_y + 10, 18, "F2"))

    header.append(_pdf_text_line("Quantitative Signals", 48, 620, 13, "F2"))

    total = max(len(scored_df), 1)
    band_specs = [
        ("High", high_count, "#1D9E75"),
        ("Medium", medium_count, "#D9A600"),
        ("Low", low_count, "#D64545"),
    ]
    bar_x = 48
    for label, count, color in band_specs:
        width = int(180 * count / total)
        header.append(_pdf_text_line(f"{label}: {count}", bar_x, 596, 9, "F1"))
        header.append(_pdf_rect(bar_x, 578, 180, 10, "#F3F4F8", "#E6E8F2"))
        if width:
            header.append(_pdf_rect(bar_x, 578, width, 10, color))
        bar_x += 180

    header.append(_pdf_text_line("Source Mix", 48, 548, 11, "F2"))
    source_total = max(sum(source_mix.values()), 1)
    y_src = 526
    source_colors = ["#396070", "#9CA1FF", "#C5CAFB", "#6B7280"]
    for idx, (source, count) in enumerate(source_mix.items()):
        width = int(260 * count / source_total)
        header.append(_pdf_text_line(f"{source}: {count}", 48, y_src + 2, 9, "F1"))
        header.append(_pdf_rect(164, y_src, 260, 9, "#F3F4F8", "#E6E8F2"))
        header.append(_pdf_rect(164, y_src, width, 9, source_colors[idx % len(source_colors)]))
        y_src -= 18

    header.append(_pdf_text_line("Top Themes", 48, y_src - 8, 11, "F2"))
    header.append(_pdf_text_line(top_keywords, 48, y_src - 26, 9, "F1"))
    if fallback:
        header.append(_pdf_text_line("No dated records were found in the last 7 days, so this brief uses the top currently visible records.", 48, y_src - 44, 8, "F1"))

    lines = []
    lines.extend([("Top Insights", 13, "F2")])

    for idx, row in enumerate(report_df.itertuples(index=False), start=1):
        title = getattr(row, "title", "") or "Untitled"
        source_type = str(getattr(row, "source_type", "") or "record").upper()
        source_label = getattr(row, "source_label", "") or "Unknown source"
        item_date = getattr(row, "item_date", "") or "Unknown date"
        score = float(getattr(row, "score", 0) or 0)
        matched = getattr(row, "matched_keywords", None) or []
        note = getattr(row, "relevance_note", "") or getattr(row, "summary", "") or getattr(row, "body", "")

        lines.append(("", 5, "F1"))
        lines.append((f"{idx}. {title}", 11, "F2"))
        lines.append((f"{source_type} | {source_label} | {str(item_date)[:10]} | score {score:.2f}", 9, "F1"))
        if matched:
            lines.append((f"Matched themes: {', '.join(matched[:6])}", 9, "F1"))
        for wrapped in _pdf_wrapped_lines(note, width=96)[:3]:
            lines.append((wrapped, 9, "F1"))

    pages = []
    y = max(y_src - 72, 390)
    current = header + [_pdf_line(48, y + 16, 564, y + 16)]
    for text, size, font in lines:
        wrapped_lines = _pdf_wrapped_lines(text, width=76 if size >= 13 else 100)
        for line in wrapped_lines:
            if y < 48:
                pages.append("".join(current))
                current = []
                y = 760
            current.append(_pdf_text_line(line, 48, y, size=size, font=font))
            y -= size + 5
        if text == "":
            y -= 4
    pages.append("".join(current))

    logo_rgb, logo_alpha, logo_width, logo_height = _load_pdf_logo()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        None,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        (
            f"<< /Type /XObject /Subtype /Image /Width {logo_width} /Height {logo_height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
            f"/SMask 6 0 R /Length {len(logo_rgb)} >>\nstream\n"
        ).encode("latin-1") + logo_rgb + b"\nendstream",
        (
            f"<< /Type /XObject /Subtype /Image /Width {logo_width} /Height {logo_height} "
            f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
            f"/Length {len(logo_alpha)} >>\nstream\n"
        ).encode("latin-1") + logo_alpha + b"\nendstream",
    ]
    page_refs = []
    for page_content in pages:
        content_id = len(objects) + 2
        page_id = len(objects) + 1
        page_refs.append(f"{page_id} 0 R")
        content_bytes = page_content.encode("latin-1", "replace")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> /XObject << /Logo 5 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode("latin-1")
        )
        objects.append(b"<< /Length " + str(len(content_bytes)).encode("ascii") + b" >>\nstream\n" + content_bytes + b"endstream")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(page_refs)}] /Count {len(page_refs)} >>".encode("latin-1")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)
