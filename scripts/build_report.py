from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

MPL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "build" / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

from matplotlib import rc_context
from matplotlib.path import Path as MatplotlibPath
from matplotlib.textpath import TextPath
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT / "results" / "accuracy.csv"
DEFAULT_SOURCE_CSV_PATH = PROJECT / "results" / "source-statistics.csv"
DEFAULT_OUTPUT_PATH = PROJECT / "docs" / "frozen-noise-cortex-writeup.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2F6BFF")
TEAL = colors.HexColor("#167D78")
ORANGE = colors.HexColor("#D47A16")
MAGENTA = colors.HexColor("#B84A7A")
INK = colors.HexColor("#17202A")
MID = colors.HexColor("#596673")
LIGHT_TEXT = colors.HexColor("#DCE6EF")
LINE = colors.HexColor("#D8E0E8")
PALE_BLUE = colors.HexColor("#EAF1FF")
PALE_TEAL = colors.HexColor("#E8F5F3")
PALE_ORANGE = colors.HexColor("#FFF3E5")
PALE_GRAY = colors.HexColor("#F4F6F8")
WHITE = colors.white


def load_results(csv_path: Path) -> list[dict[str, float]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def load_source_results(csv_path: Path) -> list[dict[str, str | float]]:
    numeric_fields = {
        "duration_ms",
        "readout_noise_sd",
        "accuracy_mean",
        "accuracy_sem",
    }
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                key: float(value) if key in numeric_fields else value
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


class AccentRule(Flowable):
    def __init__(self, width: float, color=BLUE, thickness: float = 3):
        super().__init__()
        self.width = width
        self.height = thickness
        self.color = color
        self.thickness = thickness

    def draw(self) -> None:
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.thickness / 2, self.width, self.thickness / 2)


class PipelineFlowable(Flowable):
    def __init__(self, width: float, height: float = 106):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self) -> None:
        c = self.canv
        labels = [
            ("Frozen white", "noise sample"),
            ("ERB-spaced", "filterbank"),
            ("Envelope", "crossing spikes"),
            ("Whole-trace", "rate pooling"),
            ("Regularized", "LDA decoder"),
        ]
        fills = [PALE_GRAY, PALE_BLUE, PALE_TEAL, PALE_ORANGE, PALE_BLUE]
        gap = 13
        box_width = (self.width - gap * 4) / 5
        box_height = 54
        y = 30
        c.setFont("Helvetica", 7.8)
        for index, ((line_one, line_two), fill) in enumerate(zip(labels, fills)):
            x = index * (box_width + gap)
            c.setFillColor(fill)
            c.setStrokeColor(LINE)
            c.roundRect(x, y, box_width, box_height, 7, fill=1, stroke=1)
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 7.8)
            c.drawCentredString(x + box_width / 2, y + 32, line_one)
            c.setFont("Helvetica", 7.6)
            c.drawCentredString(x + box_width / 2, y + 19, line_two)
            if index < 4:
                arrow_x = x + box_width + 2
                arrow_y = y + box_height / 2
                c.setStrokeColor(MID)
                c.setFillColor(MID)
                c.setLineWidth(1.2)
                c.line(arrow_x, arrow_y, arrow_x + gap - 5, arrow_y)
                c.line(arrow_x + gap - 8, arrow_y + 3, arrow_x + gap - 5, arrow_y)
                c.line(arrow_x + gap - 8, arrow_y - 3, arrow_x + gap - 5, arrow_y)
        c.setFillColor(MID)
        c.setFont("Helvetica", 7.5)
        c.drawString(0, 10, "Physical stimulus")
        c.drawRightString(self.width, 10, "Learned two-class decision")


class AccuracyChart(Flowable):
    def __init__(self, results: list[dict[str, float]], width: float, height: float = 258):
        super().__init__()
        self.results = results
        self.width = width
        self.height = height

    def draw(self) -> None:
        c = self.canv
        left, right, bottom, top = 45, 10, 35, 28
        plot_w = self.width - left - right
        plot_h = self.height - bottom - top
        durations = sorted({row["duration_ms"] for row in self.results})
        noises = sorted({row["readout_noise_sd"] for row in self.results})
        palette = [BLUE, TEAL, ORANGE, MAGENTA]
        log_min = math.log10(min(durations))
        log_max = math.log10(max(durations))

        def x_pos(duration: float) -> float:
            fraction = (math.log10(duration) - log_min) / (log_max - log_min)
            return left + fraction * plot_w

        def y_pos(accuracy: float) -> float:
            return bottom + (accuracy - 0.5) / 0.5 * plot_h

        c.setFont("Helvetica", 7.5)
        for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            y = y_pos(tick)
            c.setStrokeColor(LINE)
            c.setLineWidth(0.6)
            c.line(left, y, left + plot_w, y)
            c.setFillColor(MID)
            c.drawRightString(left - 6, y - 2.5, f"{int(tick * 100)}%")

        for duration in durations:
            x = x_pos(duration)
            c.setStrokeColor(colors.HexColor("#EEF1F4"))
            c.line(x, bottom, x, bottom + plot_h)
            c.setFillColor(MID)
            label = f"{duration / 1000:g}s" if duration >= 1000 else f"{int(duration)}ms"
            c.drawCentredString(x, bottom - 13, label)

        c.setStrokeColor(MID)
        c.setLineWidth(0.8)
        c.rect(left, bottom, plot_w, plot_h, fill=0, stroke=1)
        c.setDash(4, 3)
        c.line(left, y_pos(0.5), left + plot_w, y_pos(0.5))
        c.setDash()

        for series_index, noise in enumerate(noises):
            color = palette[series_index % len(palette)]
            series = sorted(
                (row for row in self.results if row["readout_noise_sd"] == noise),
                key=lambda row: row["duration_ms"],
            )
            points = [(x_pos(row["duration_ms"]), y_pos(row["accuracy_mean"])) for row in series]
            c.setStrokeColor(color)
            c.setFillColor(WHITE)
            c.setLineWidth(1.8)
            for first, second in zip(points, points[1:]):
                c.line(first[0], first[1], second[0], second[1])
            for row, (x, y) in zip(series, points):
                sem = row["accuracy_sem"]
                y_low = y_pos(max(0.5, row["accuracy_mean"] - sem))
                y_high = y_pos(min(1.0, row["accuracy_mean"] + sem))
                c.line(x, y_low, x, y_high)
                c.line(x - 2.5, y_low, x + 2.5, y_low)
                c.line(x - 2.5, y_high, x + 2.5, y_high)
                c.circle(x, y, 2.5, fill=1, stroke=1)

        legend_x = left
        legend_y = self.height - 10
        c.setFont("Helvetica", 7.5)
        for index, noise in enumerate(noises):
            x = legend_x + index * 99
            c.setStrokeColor(palette[index % len(palette)])
            c.setLineWidth(2.2)
            c.line(x, legend_y, x + 15, legend_y)
            c.setFillColor(INK)
            c.drawString(x + 19, legend_y - 2.7, f"{noise:g} sp/s noise")

        c.setFillColor(INK)
        c.setFont("Helvetica", 8)
        c.drawCentredString(left + plot_w / 2, 3, "Frozen sample duration (log scale)")
        c.saveState()
        c.translate(9, bottom + plot_h / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, "Held-out accuracy")
        c.restoreState()


class FingerprintChart(Flowable):
    def __init__(self, results: list[dict[str, float]], width: float, height: float = 172):
        super().__init__()
        self.results = results
        self.width = width
        self.height = height

    def draw(self) -> None:
        c = self.canv
        unique = {}
        for row in self.results:
            unique[row["duration_ms"]] = row
        series = [unique[key] for key in sorted(unique)]
        left, right, bottom, top = 45, 10, 31, 20
        plot_w = self.width - left - right
        plot_h = self.height - bottom - top
        durations = [row["duration_ms"] for row in series]
        log_min = math.log10(min(durations))
        log_max = math.log10(max(durations))
        y_max = 80.0

        def x_pos(duration: float) -> float:
            return left + (math.log10(duration) - log_min) / (log_max - log_min) * plot_w

        def y_pos(distance: float) -> float:
            return bottom + distance / y_max * plot_h

        c.setFont("Helvetica", 7.5)
        for tick in (0, 20, 40, 60, 80):
            y = y_pos(tick)
            c.setStrokeColor(LINE)
            c.setLineWidth(0.6)
            c.line(left, y, left + plot_w, y)
            c.setFillColor(MID)
            c.drawRightString(left - 6, y - 2.5, str(tick))
        for duration in durations:
            x = x_pos(duration)
            label = f"{duration / 1000:g}s" if duration >= 1000 else f"{int(duration)}ms"
            c.setFillColor(MID)
            c.drawCentredString(x, bottom - 12, label)
        c.setStrokeColor(MID)
        c.rect(left, bottom, plot_w, plot_h, fill=0, stroke=1)

        points = [(x_pos(row["duration_ms"]), y_pos(row["clean_distance_mean"])) for row in series]
        c.setStrokeColor(BLUE)
        c.setLineWidth(2)
        for first, second in zip(points, points[1:]):
            c.line(first[0], first[1], second[0], second[1])
        c.setFillColor(WHITE)
        for row, (x, y) in zip(series, points):
            sem = row["clean_distance_sem"]
            c.line(x, y_pos(max(0, row["clean_distance_mean"] - sem)), x, y_pos(row["clean_distance_mean"] + sem))
            c.circle(x, y, 2.8, fill=1, stroke=1)

        c.setFillColor(INK)
        c.setFont("Helvetica", 8)
        c.drawCentredString(left + plot_w / 2, 2, "Frozen sample duration (log scale)")
        c.drawString(left + 3, self.height - 10, "Clean rate distance (spikes/s)")


class SourceStatisticsChart(Flowable):
    def __init__(
        self,
        results: list[dict[str, str | float]],
        width: float,
        height: float = 250,
        noise_sd: float = 20.0,
    ):
        super().__init__()
        self.results = results
        self.width = width
        self.height = height
        self.noise_sd = noise_sd

    def draw(self) -> None:
        c = self.canv
        rows = [
            row
            for row in self.results
            if float(row["readout_noise_sd"]) == self.noise_sd
            and row["task"] in {"same_source_exemplar_mean", "different_source"}
        ]
        left, right, bottom, top = 45, 10, 34, 42
        plot_w = self.width - left - right
        plot_h = self.height - bottom - top
        durations = sorted({float(row["duration_ms"]) for row in rows})
        log_min = math.log10(min(durations))
        log_max = math.log10(max(durations))

        def x_pos(duration: float) -> float:
            fraction = (math.log10(duration) - log_min) / (log_max - log_min)
            return left + fraction * plot_w

        def y_pos(accuracy: float) -> float:
            return bottom + (accuracy - 0.5) / 0.5 * plot_h

        c.setFont("Helvetica", 7.5)
        for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            y = y_pos(tick)
            c.setStrokeColor(LINE)
            c.setLineWidth(0.6)
            c.line(left, y, left + plot_w, y)
            c.setFillColor(MID)
            c.drawRightString(left - 6, y - 2.5, f"{int(tick * 100)}%")
        for duration in durations:
            x = x_pos(duration)
            c.setStrokeColor(colors.HexColor("#EEF1F4"))
            c.line(x, bottom, x, bottom + plot_h)
            c.setFillColor(MID)
            label = f"{duration / 1000:g}s" if duration >= 1000 else f"{int(duration)}ms"
            c.drawCentredString(x, bottom - 12, label)
        c.setStrokeColor(MID)
        c.rect(left, bottom, plot_w, plot_h, fill=0, stroke=1)

        series_specs = [
            (
                ("same_source_exemplar_mean", "all", ""),
                "same-source exemplars",
                MID,
                True,
            ),
            (
                ("different_source", "white", "light-pink"),
                "white vs light-pink",
                BLUE,
                False,
            ),
            (
                ("different_source", "white", "pink"),
                "white vs pink",
                TEAL,
                False,
            ),
            (
                ("different_source", "pink", "brown"),
                "pink vs brown",
                ORANGE,
                False,
            ),
        ]
        for series_index, (key, label, color, dashed) in enumerate(series_specs):
            task, source_a, source_b = key
            series = sorted(
                (
                    row
                    for row in rows
                    if row["task"] == task
                    and row["source_a"] == source_a
                    and row["source_b"] == source_b
                ),
                key=lambda row: float(row["duration_ms"]),
            )
            points = [
                (
                    x_pos(float(row["duration_ms"])),
                    y_pos(float(row["accuracy_mean"])),
                )
                for row in series
            ]
            c.setStrokeColor(color)
            c.setFillColor(WHITE)
            c.setLineWidth(1.8)
            if dashed:
                c.setDash(5, 3)
            for first, second in zip(points, points[1:]):
                c.line(first[0], first[1], second[0], second[1])
            c.setDash()
            for row, (x, y) in zip(series, points):
                mean = float(row["accuracy_mean"])
                sem = float(row["accuracy_sem"])
                y_low = y_pos(max(0.5, mean - sem))
                y_high = y_pos(min(1.0, mean + sem))
                c.line(x, y_low, x, y_high)
                c.circle(x, y, 2.5, fill=1, stroke=1)

            legend_x = left + (series_index % 2) * 177
            legend_y = self.height - 11 - (series_index // 2) * 14
            c.setStrokeColor(color)
            c.setLineWidth(2.2)
            if dashed:
                c.setDash(5, 3)
            c.line(legend_x, legend_y, legend_x + 14, legend_y)
            c.setDash()
            c.setFillColor(INK)
            c.drawString(legend_x + 18, legend_y - 2.5, label)

        c.setFillColor(INK)
        c.setFont("Helvetica", 8)
        c.drawCentredString(left + plot_w / 2, 3, "Excerpt duration (log scale)")
        c.saveState()
        c.translate(9, bottom + plot_h / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, "Held-out accuracy")
        c.restoreState()


class MathEquationBlock(Flowable):
    """Render Matplotlib mathtext as scalable vector paths on a PDF canvas."""

    def __init__(
        self,
        expressions: list[str],
        width: float,
        fill=PALE_GRAY,
        font_size: float = 13.0,
        padding: float = 9.0,
        line_gap: float = 7.0,
    ):
        super().__init__()
        if not expressions:
            raise ValueError("expressions cannot be empty")
        self.width = width
        self.fill = fill
        self.padding = padding
        self.line_gap = line_gap
        with rc_context({"mathtext.fontset": "stix", "mathtext.default": "it"}):
            self.paths = [
                TextPath((0.0, 0.0), f"${expression}$", size=font_size, usetex=False)
                for expression in expressions
            ]
        content_width = self.width - 2.0 * self.padding
        self.layout: list[tuple[TextPath, float, float, float]] = []
        for path in self.paths:
            bounds = path.get_extents()
            scale = min(1.0, content_width / max(bounds.width, 1e-9))
            self.layout.append((path, scale, bounds.width * scale, bounds.height * scale))
        self.height = (
            2.0 * self.padding
            + sum(item[3] for item in self.layout)
            + self.line_gap * (len(self.layout) - 1)
        )

    @staticmethod
    def _reportlab_path(canvas: Canvas, text_path: TextPath):
        """Convert Matplotlib glyph outlines to a ReportLab vector path."""

        output = canvas.beginPath()
        current = (0.0, 0.0)
        subpath_start = (0.0, 0.0)
        for vertices, code in text_path.iter_segments(curves=True, simplify=False):
            values = tuple(float(value) for value in vertices)
            if code == MatplotlibPath.MOVETO:
                current = (values[0], values[1])
                subpath_start = current
                output.moveTo(*current)
            elif code == MatplotlibPath.LINETO:
                current = (values[0], values[1])
                output.lineTo(*current)
            elif code == MatplotlibPath.CURVE3:
                control = (values[0], values[1])
                endpoint = (values[2], values[3])
                first_control = (
                    current[0] + 2.0 * (control[0] - current[0]) / 3.0,
                    current[1] + 2.0 * (control[1] - current[1]) / 3.0,
                )
                second_control = (
                    endpoint[0] + 2.0 * (control[0] - endpoint[0]) / 3.0,
                    endpoint[1] + 2.0 * (control[1] - endpoint[1]) / 3.0,
                )
                output.curveTo(*first_control, *second_control, *endpoint)
                current = endpoint
            elif code == MatplotlibPath.CURVE4:
                output.curveTo(*values)
                current = (values[4], values[5])
            elif code == MatplotlibPath.CLOSEPOLY:
                output.close()
                current = subpath_start
        return output

    def draw(self) -> None:
        canvas = self.canv
        canvas.saveState()
        canvas.setFillColor(self.fill)
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.6)
        canvas.roundRect(0, 0, self.width, self.height, 3, fill=1, stroke=1)

        content_width = self.width - 2.0 * self.padding
        top = self.height - self.padding
        canvas.setFillColor(INK)
        for path, scale, rendered_width, rendered_height in self.layout:
            bounds = path.get_extents()
            bottom = top - rendered_height
            x = self.padding + (content_width - rendered_width) / 2.0
            canvas.saveState()
            canvas.translate(x - bounds.x0 * scale, bottom - bounds.y0 * scale)
            canvas.scale(scale, scale)
            canvas.drawPath(
                self._reportlab_path(canvas, path),
                fill=1,
                stroke=0,
                fillMode=0,
            )
            canvas.restoreState()
            top = bottom - self.line_gap
        canvas.restoreState()


def make_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=31,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=11,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=sample["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=MID,
            spaceAfter=15,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=23,
            textColor=NAVY,
            spaceAfter=9,
        ),
        "subsection": ParagraphStyle(
            "Subsection",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=NAVY,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.2,
            textColor=INK,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10.5,
            textColor=MID,
            spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14.2,
            textColor=INK,
        ),
        "table": ParagraphStyle(
            "TableText",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=9.8,
            textColor=INK,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9.3,
            textColor=WHITE,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=sample["Code"],
            fontName="Courier",
            fontSize=7.4,
            leading=10,
            textColor=INK,
        ),
    }


def callout(text: str, styles: dict[str, ParagraphStyle], fill=PALE_BLUE) -> Table:
    table = Table([[Paragraph(text, styles["callout"])]], colWidths=[7.02 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), fill),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def equation_block(
    expressions: list[str],
    styles: dict[str, ParagraphStyle],
    fill=PALE_GRAY,
) -> MathEquationBlock:
    """Create a vector typeset equation block; ``styles`` keeps call sites uniform."""

    del styles
    return MathEquationBlock(
        expressions,
        width=7.02 * inch,
        fill=fill,
    )


def bullet_list(items: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    rows = [[Paragraph("-", styles["body"]), Paragraph(item, styles["body"])] for item in items]
    table = Table(rows, colWidths=[0.18 * inch, 6.82 * inch], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("TEXTCOLOR", (0, 0), (0, -1), BLUE),
            ]
        )
    )
    return table


def styled_table(
    rows: list[list[str | Paragraph]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
    header: bool = True,
    font_size: float = 7.4,
) -> Table:
    converted: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if header and row_index == 0 else styles["table"]
        converted.append(
            [cell if isinstance(cell, Paragraph) else Paragraph(str(cell), style) for cell in row]
        )
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    for row_index in range(1 if header else 0, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE_GRAY))
    table.setStyle(TableStyle(commands))
    return table


def on_first_page(canvas: Canvas, doc: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setTitle("Frozen-Noise Discrimination: Project Write-up and Initial Results")
    canvas.setAuthor("Codex project report")
    canvas.setSubject("Cochlea-like rate-code simulation of frozen white-noise discrimination")
    canvas.setFillColor(NAVY)
    canvas.rect(0, LETTER[1] - 18, LETTER[0], 18, fill=1, stroke=0)
    canvas.setFillColor(MID)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(50, 24, "Frozen-noise cortex - exploratory model v0.2")
    canvas.drawRightString(LETTER[0] - 50, 24, "14 August 2026")
    canvas.restoreState()


def on_later_pages(canvas: Canvas, doc: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.7)
    canvas.line(50, LETTER[1] - 39, LETTER[0] - 50, LETTER[1] - 39)
    canvas.setFillColor(MID)
    canvas.setFont("Helvetica", 7.2)
    canvas.drawString(50, LETTER[1] - 31, "FROZEN-NOISE DISCRIMINATION")
    canvas.drawRightString(LETTER[0] - 50, LETTER[1] - 31, "PROJECT WRITE-UP")
    canvas.line(50, 38, LETTER[0] - 50, 38)
    canvas.drawString(50, 26, "Exploratory computational model - not a validated physiological model")
    canvas.drawRightString(LETTER[0] - 50, 26, f"Page {doc.page}")
    canvas.restoreState()


def result_lookup(results: list[dict[str, float]]) -> dict[tuple[int, int], dict[str, float]]:
    return {
        (int(row["duration_ms"]), int(row["readout_noise_sd"])): row
        for row in results
    }


def source_result_lookup(
    results: list[dict[str, str | float]],
) -> dict[tuple[str, str, str, int, int], dict[str, str | float]]:
    return {
        (
            str(row["task"]),
            str(row["source_a"]),
            str(row["source_b"]),
            int(float(row["duration_ms"])),
            int(float(row["readout_noise_sd"])),
        ): row
        for row in results
    }


def build_pdf(
    csv_path: Path = DEFAULT_CSV_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    source_csv_path: Path = DEFAULT_SOURCE_CSV_PATH,
) -> Path:
    results = load_results(csv_path)
    source_results = load_source_results(source_csv_path)
    lookup = result_lookup(results)
    source_lookup = source_result_lookup(source_results)
    styles = make_styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=50,
        rightMargin=50,
        topMargin=52,
        bottomMargin=47,
        title="Frozen-Noise Discrimination: Project Write-up and Initial Results",
        author="Codex project report",
    )

    story: list[Flowable] = []

    # Cover and executive summary.
    story.extend(
        [
            Spacer(1, 0.28 * inch),
            Paragraph("Frozen-Noise Discrimination", styles["cover_title"]),
            Paragraph(
                "A cochlea-like rate-code simulation of why short frozen white-noise samples can remain perceptually distinct while longer samples become statistically alike",
                styles["cover_subtitle"],
            ),
            AccentRule(7.02 * inch),
            Spacer(1, 0.20 * inch),
            callout(
                "<b>Central results.</b> Fixed readout noise exposes the loss of frozen-exemplar identity with duration. At 20 spikes/s noise, same-source exemplar accuracy falls from 99.8% at 25 ms to 70.9% at 1.6 s, while a subtle white versus light-pink source contrast rises from 86.8% to 99.8%. Chance is 50%.",
                styles,
                PALE_BLUE,
            ),
            Spacer(1, 0.23 * inch),
            Paragraph("Model at a glance", styles["subsection"]),
            PipelineFlowable(7.02 * inch),
            Spacer(1, 0.10 * inch),
            Paragraph("Purpose", styles["subsection"]),
            Paragraph(
                "This project is a compact computational test of one explanatory idea: short random waveforms carry accidental, sample-specific patterns across auditory frequency channels; those fingerprints shrink as the sample is averaged for longer; and a noisy cortical readout eventually cannot learn the difference between two frozen samples.",
                styles["body"],
            ),
            Paragraph("Current scope", styles["subsection"]),
            bullet_list(
                [
                    "The encoder is cochlea-like rather than a detailed auditory-nerve model.",
                    "The decoder retains average channel rates and discards precise spike timing.",
                    "The reported sweep manipulates fixed Gaussian readout noise after spike-rate pooling.",
                    "All results are Monte Carlo estimates from 12 independently generated frozen-noise pairs.",
                ],
                styles,
            ),
            Spacer(1, 0.12 * inch),
            Paragraph(
                "Project version 0.2 | NumPy-only implementation | Reproducible seeds: 7 and 29",
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    # Scientific motivation, question, and mechanism.
    story.extend(
        [
            Paragraph("1. Introduction: the McDermott result", styles["section"]),
            Paragraph("Psychophysical starting point", styles["subsection"]),
            Paragraph(
                "McDermott, Schemitsch, and Simoncelli tested whether listeners represent sound textures through temporal detail or through statistics averaged over time. In their oddity tasks, discrimination of <i>different texture sources</i> improved as excerpts lengthened, whereas discrimination of <i>different exemplars of the same texture</i> declined. The crossover is evidence that local details dominate brief sounds, but longer textures are represented increasingly by summary statistics.",
                styles["body"],
            ),
            callout(
                "<b>Primary study.</b> McDermott, J. H., Schemitsch, M., &amp; Simoncelli, E. P. (2013). Summary statistics in auditory perception. <i>Nature Neuroscience, 16</i>(4), 493-498. <link href='https://doi.org/10.1038/nn.3347' color='#286B8E'>doi:10.1038/nn.3347</link>",
                styles,
                PALE_BLUE,
            ),
            Spacer(1, 0.08 * inch),
            Paragraph(
                "Their stimuli were naturalistic synthetic textures rather than raw white noise. The present project is therefore a deliberately minimal analogue, not a reproduction of that experiment: it asks whether frozen white and power-law noise, a cochlear filterbank, spike threshold crossings, and temporal rate averaging are already sufficient to produce the same qualitative opposition.",
                styles["body"],
            ),
            Paragraph("The proposed mechanism", styles["subsection"]),
            callout(
                "A finite white-noise sample is not perfectly flat in every auditory band. Short samples have large random deviations in bandwise envelope events. Longer samples average those deviations away. A discriminator operating on noisy, time-averaged channel rates therefore loses class information with duration.",
                styles,
                PALE_TEAL,
            ),
            Spacer(1, 0.11 * inch),
            Paragraph("Three linked hypotheses", styles["subsection"]),
            styled_table(
                [
                    ["Hypothesis", "Model implication", "Observable result"],
                    [
                        "H1 - finite-sample fingerprints",
                        "Two frozen samples create different threshold-crossing rates across cochlear channels.",
                        "The clean channel-rate distance is positive for every tested duration.",
                    ],
                    [
                        "H2 - convergence with duration",
                        "Whole-trace averaging makes accidental bandwise deviations smaller for long samples.",
                        "The clean rate distance falls as duration grows.",
                    ],
                    [
                        "H3 - noise-limited learning",
                        "Fixed post-pooling readout noise does not shrink with the sample fingerprint.",
                        "Accuracy approaches chance sooner at higher readout-noise levels.",
                    ],
                ],
                [1.45 * inch, 3.15 * inch, 2.42 * inch],
                styles,
            ),
            Spacer(1, 0.13 * inch),
            Paragraph("Why a rate code matters", styles["subsection"]),
            Paragraph(
                "The present decoder receives one feature per cochlear channel: the number of threshold-crossing spikes divided by the sample duration. This intentionally creates a low-dimensional summary of each sound. It is the central modeling assumption, not a neutral preprocessing choice.",
                styles["body"],
            ),
            Paragraph(
                "If the decoder instead receives the complete time-locked spike sequence, a longer frozen trace contains more identifying events and may become easier to classify. Comparing rate and temporal codes is therefore the most diagnostic next experiment.",
                styles["body"],
            ),
            Paragraph("What would count as support?", styles["subsection"]),
            bullet_list(
                [
                    "A monotonic reduction in sample-specific clean feature distance with duration.",
                    "A clear interaction in which readout noise has little effect on very short samples but progressively corrupts long-sample classification.",
                    "Results that replicate across independently generated pairs rather than depending on one unusually distinctive waveform.",
                ],
                styles,
            ),
            PageBreak(),
        ]
    )

    # Model details.
    story.extend(
        [
            Paragraph("2. Model architecture", styles["section"]),
            Paragraph(
                "The implementation uses transparent components and only NumPy. It is fast enough for parameter sweeps while keeping each transformation inspectable.",
                styles["body"],
            ),
            styled_table(
                [
                    ["Stage", "Implementation", "Function in the hypothesis"],
                    [
                        "Stimulus",
                        "Two independent Gaussian white-noise waveforms per pair; each waveform is demeaned and normalized to equal RMS.",
                        "Creates repeatable frozen exemplars while removing the trivial overall-level cue.",
                    ],
                    [
                        "Cochlear filterbank",
                        "16 overlapping fourth-order frequency responses with center frequencies equally spaced on the ERB-number scale from 200 to 7,000 Hz.",
                        "Turns accidental spectral fluctuations into a channel population code.",
                    ],
                    [
                        "Inner-hair-cell proxy",
                        "The analytic envelope is computed in every channel. Upward crossings of a fixed threshold generate Boolean spikes, with a 1 ms refractory interval.",
                        "Produces sparse events from changing subband envelopes.",
                    ],
                    [
                        "Trial noise",
                        "Three percent spike dropout and 0.2 spontaneous spikes/s are applied before pooling. Gaussian readout noise is added after pooling.",
                        "Separates noise that can average over time from noise that remains at the final decision representation.",
                    ],
                    [
                        "Feature code",
                        "Spike count per channel divided by sample duration, yielding a 16-dimensional rate vector in spikes/s.",
                        "Preserves the channel fingerprint but discards precise spike timing.",
                    ],
                    [
                        "Classifier",
                        "Two-class linear discriminant analysis with isotropic covariance regularization.",
                        "Learns a stable linear boundary from noisy repetitions.",
                    ],
                ],
                [1.25 * inch, 3.48 * inch, 2.29 * inch],
                styles,
            ),
            Spacer(1, 0.11 * inch),
            Paragraph("Signal processing details", styles["subsection"]),
            bullet_list(
                [
                    "The filterbank is applied in the Fourier domain. Each response is normalized to equal expected RMS for ideal white noise, so high-frequency channels do not win solely because their bandwidth is larger.",
                    "A NumPy Hilbert transform supplies the channel envelope. The default crossing threshold is 1.45 in normalized envelope units.",
                    "Stimulus RMS normalization means the classifier must use within-spectrum structure rather than total loudness.",
                    "The clean fingerprint distance is the root-mean-square Euclidean difference between the two class rate vectors across channels.",
                ],
                styles,
            ),
            Paragraph("Default cochlear settings", styles["subsection"]),
            styled_table(
                [
                    ["Parameter", "Value", "Parameter", "Value"],
                    ["Sample rate", "16,000 Hz", "Channels", "16"],
                    ["Frequency range", "200-7,000 Hz", "Bandwidth scale", "1.5 ERB"],
                    ["Envelope threshold", "1.45", "Refractory interval", "1.0 ms"],
                    ["Stimulus normalization", "Equal RMS", "Feature units", "Spikes/s"],
                ],
                [1.50 * inch, 1.36 * inch, 1.75 * inch, 1.35 * inch],
                styles,
            ),
            Spacer(1, 0.13 * inch),
            callout(
                "<b>Interpretive boundary.</b> ERB spacing, envelope crossings, and a refractory interval make the front end auditory-inspired. They do not make it a validated model of cochlear mechanics, auditory-nerve adaptation, or cortical population dynamics.",
                styles,
                PALE_ORANGE,
            ),
            PageBreak(),
        ]
    )

    # Experiment design.
    story.extend(
        [
            Paragraph("3. Experiment and evaluation", styles["section"]),
            Paragraph("Protocol", styles["subsection"]),
            styled_table(
                [
                    ["Step", "Operation"],
                    ["1", "Choose one of seven durations: 25, 50, 100, 200, 400, 800, or 1,600 ms."],
                    ["2", "Generate 12 independent pairs of frozen Gaussian white-noise samples at that duration."],
                    ["3", "Encode both samples once into clean cochlear threshold-crossing counts."],
                    ["4", "Generate 32 noisy training repetitions and 96 noisy test repetitions for each class."],
                    ["5", "Fit regularized LDA to the pooled rate vectors and score held-out two-alternative accuracy."],
                    ["6", "Repeat at four post-pooling readout-noise levels and average accuracy across frozen pairs."],
                ],
                [0.48 * inch, 6.54 * inch],
                styles,
            ),
            Spacer(1, 0.11 * inch),
            Paragraph("Noise manipulations", styles["subsection"]),
            styled_table(
                [
                    ["Noise source", "Default", "Where it enters", "Expected dependence on duration"],
                    [
                        "Spike dropout",
                        "3%",
                        "Before rate pooling",
                        "Partly averages over more events in longer samples.",
                    ],
                    [
                        "Spontaneous spikes",
                        "0.2 spikes/s/channel",
                        "Before rate pooling",
                        "Poisson variability in normalized rate decreases with duration.",
                    ],
                    [
                        "Readout noise",
                        "0, 10, 20, or 40 spikes/s SD",
                        "After rate pooling",
                        "Remains fixed while the frozen-sample fingerprint shrinks.",
                    ],
                ],
                [1.32 * inch, 1.22 * inch, 1.55 * inch, 2.93 * inch],
                styles,
            ),
            Spacer(1, 0.12 * inch),
            Paragraph("Classifier and uncertainty", styles["subsection"]),
            Paragraph(
                "For each pair, the classifier is trained and tested only on repetitions of the same two frozen samples. Accuracy therefore measures whether the model can learn sample identity, not whether it can generalize to a new category of noise. Curves show mean held-out accuracy across 12 frozen pairs; error bars are +/-1 standard error of the mean across pairs.",
                styles["body"],
            ),
            Paragraph("Controls built into the first version", styles["subsection"]),
            bullet_list(
                [
                    "Equal RMS removes overall loudness as a class cue.",
                    "The stimulus random-number stream is independent of the trial-noise stream, so adding or removing noise conditions does not change the underlying frozen samples.",
                    "Training and test trials are separate, preventing direct evaluation on the fitted examples.",
                    "A fixed random seed makes the supplied result files reproducible.",
                ],
                styles,
            ),
            Paragraph("What the experiment does not yet test", styles["subsection"]),
            bullet_list(
                [
                    "Acoustic noise mixed with the waveform before the cochlear filterbank.",
                    "Trial-to-trial filter or threshold jitter inside cochlear channels.",
                    "A leaky cortical integrator with an independently varied time constant.",
                    "Recognition after temporal shifts, onset uncertainty, or changes in presentation level.",
                ],
                styles,
            ),
            PageBreak(),
        ]
    )

    # Results.
    selected_durations = [25, 100, 400, 800, 1600]
    selected_noises = [0, 10, 20, 40]
    accuracy_rows: list[list[str]] = [
        ["Duration", "0 sp/s", "10 sp/s", "20 sp/s", "40 sp/s"]
    ]
    for duration in selected_durations:
        accuracy_rows.append(
            [
                f"{duration} ms" if duration < 1000 else "1.6 s",
                *[
                    f"{lookup[(duration, noise)]['accuracy_mean'] * 100:.1f}%"
                    for noise in selected_noises
                ],
            ]
        )

    story.extend(
        [
            Paragraph("4. Initial results", styles["section"]),
            Paragraph("Held-out classification accuracy", styles["subsection"]),
            AccuracyChart(results, 7.02 * inch),
            Paragraph(
                "Figure 1. Mean held-out accuracy across 12 frozen pairs. Error bars show +/-1 SEM. The no-readout-noise condition remains at ceiling, while stronger readout noise produces an increasingly steep duration-dependent decline.",
                styles["small"],
            ),
            styled_table(
                accuracy_rows,
                [1.18 * inch, 1.16 * inch, 1.16 * inch, 1.16 * inch, 1.16 * inch],
                styles,
            ),
            Spacer(1, 0.10 * inch),
            callout(
                "<b>Main pattern.</b> At 25 ms, all four conditions are at or above 99.6% accuracy. By 1.6 s, accuracy has separated sharply by readout noise: 100.0%, 88.0%, 72.3%, and 59.8% for 0, 10, 20, and 40 spikes/s respectively.",
                styles,
                PALE_BLUE,
            ),
            PageBreak(),
        ]
    )

    # Mechanistic readout and interpretation.
    story.extend(
        [
            Paragraph("5. Mechanistic interpretation", styles["section"]),
            Paragraph("The frozen-sample fingerprint shrinks", styles["subsection"]),
            FingerprintChart(results, 7.02 * inch),
            Paragraph(
                "Figure 2. Root-mean-square clean rate-vector distance between the two frozen samples, averaged across 12 pairs. Error bars show +/-1 SEM. The distance falls from 69.3 spikes/s at 25 ms to 8.2 spikes/s at 1.6 s.",
                styles["small"],
            ),
            Paragraph("Reading the two figures together", styles["subsection"]),
            bullet_list(
                [
                    "The encoder gives every finite frozen pair a distinct multichannel rate pattern.",
                    "Whole-trace averaging reduces that clean class separation by roughly a factor of eight across the tested duration range.",
                    "Pre-pooling spike variability can partly average out over time, but fixed post-pooling noise cannot.",
                    "Once the clean fingerprint is comparable to the fixed readout-noise scale, held-out classification approaches chance.",
                ],
                styles,
            ),
            Spacer(1, 0.09 * inch),
            callout(
                "The first simulation therefore supports the proposed mechanism <b>within a whole-trace rate-code model</b>. It does not yet establish that this is the mechanism used by human auditory cortex.",
                styles,
                PALE_TEAL,
            ),
            Spacer(1, 0.10 * inch),
            Paragraph("Why the clean condition stays perfect", styles["subsection"]),
            Paragraph(
                "Without post-pooling readout noise, repetitions from each class remain tightly clustered around a deterministic rate fingerprint. Even a small difference is sufficient for a linear classifier. The duration effect appears behaviorally only when the shrinking fingerprint must be read through irreducible noise or another source of finite precision.",
                styles["body"],
            ),
            Paragraph("What is already learned", styles["subsection"]),
            styled_table(
                [
                    ["Observation", "Implication"],
                    [
                        "Short samples remain robust under strong readout noise.",
                        "Their accidental cross-channel structure is large relative to the noise scale.",
                    ],
                    [
                        "Long-sample accuracy changes steeply with readout noise.",
                        "The predicted duration boundary is not fixed; it depends on internal precision.",
                    ],
                    [
                        "Clean distance declines smoothly with duration.",
                        "The effect is explained by convergence of finite-sample statistics rather than a hard duration switch.",
                    ],
                ],
                [2.92 * inch, 4.10 * inch],
                styles,
            ),
            PageBreak(),
        ]
    )

    # Source-statistics extension.
    source_accuracy_rows: list[list[str]] = [
        [
            "Duration",
            "Same-source exemplars",
            "White vs light-pink",
            "White vs pink",
            "Pink vs brown",
        ]
    ]
    for duration in (25, 100, 200, 400, 1600):
        keys = [
            ("same_source_exemplar_mean", "all", ""),
            ("different_source", "white", "light-pink"),
            ("different_source", "white", "pink"),
            ("different_source", "pink", "brown"),
        ]
        source_accuracy_rows.append(
            [
                f"{duration} ms" if duration < 1000 else "1.6 s",
                *[
                    f"{float(source_lookup[(task, first, second, duration, 20)]['accuracy_mean']) * 100:.1f}%"
                    for task, first, second in keys
                ],
            ]
        )

    story.extend(
        [
            Paragraph("6. Source-statistics extension", styles["section"]),
            Paragraph("Complementary tasks", styles["subsection"]),
            Paragraph(
                "The extension asks whether the same pooled rate representation can reproduce the opposing duration effects reported for auditory textures. In the exemplar task, the classifier learns two fixed waveforms from the same noise family. In the source task, it trains on independent waveforms from two families and must generalize to previously unseen exemplars.",
                styles["body"],
            ),
            Paragraph(
                "Four Gaussian power-law families were generated within the 200-7,000 Hz model band: white (1/f^0), light-pink (1/f^0.5), pink (1/f^1), and brown (1/f^2). The light-pink condition supplies a deliberately subtle source contrast; canonical white-pink and pink-brown contrasts test stronger spectral differences.",
                styles["body"],
            ),
            SourceStatisticsChart(source_results, 7.02 * inch),
            Paragraph(
                "Figure 3. Accuracy at 20 spikes/s post-pooling readout noise. The dashed curve averages frozen-exemplar discrimination over all four families. Solid curves classify noise family from independent training and test exemplars. Error bars show +/-1 SEM across frozen pairs or train/test splits.",
                styles["small"],
            ),
            styled_table(
                source_accuracy_rows,
                [0.83 * inch, 1.55 * inch, 1.55 * inch, 1.35 * inch, 1.35 * inch],
                styles,
            ),
            Spacer(1, 0.08 * inch),
            callout(
                "<b>Opposing duration effects.</b> Same-source exemplar accuracy falls from 99.8% at 25 ms to 70.9% at 1.6 s. The deliberately difficult white versus light-pink classification rises from 86.8% to 99.8%, crossing the exemplar curve between 100 and 200 ms.",
                styles,
                PALE_TEAL,
            ),
            Spacer(1, 0.08 * inch),
            Paragraph(
                "The mechanism is hierarchical: accidental exemplar deviations shrink with temporal averaging, but the stable mean spectral difference between sources remains. Canonical white-pink and pink-brown contrasts are nearly saturated even at 25 ms, showing that their source statistics are too different to reveal much duration dependence under these settings.",
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    # Limitations and next steps.
    story.extend(
        [
            Paragraph("7. Limitations and next experiments", styles["section"]),
            Paragraph("Important limitations", styles["subsection"]),
            bullet_list(
                [
                    "The Fourier-domain filterbank approximates auditory frequency analysis but is not a causal gammatone or transmission-line cochlear model.",
                    "Envelope threshold crossings omit adaptation, saturation, phase locking, multiple spontaneous-rate fiber classes, and correlated population noise.",
                    "The whole-trace mean-rate decoder has no explicit memory limit and no invariance to temporal shifts or onset uncertainty.",
                    "The readout-noise levels are expressed in model feature units and have not been calibrated to physiology or human psychophysics.",
                    "Ceiling performance in the clean condition shows that finite classifier capacity alone is not currently limiting learning.",
                ],
                styles,
            ),
            Paragraph("Highest-value next comparison", styles["subsection"]),
            callout(
                "Train two matched decoders on the same spikes: (A) the current whole-trace rate code and (B) a time-resolved code that preserves spike counts in short bins. If only the rate decoder shows declining performance with duration, the result directly links the phenomenon to temporal information loss.",
                styles,
                PALE_ORANGE,
            ),
            Spacer(1, 0.10 * inch),
            Paragraph("Suggested experiment sequence", styles["subsection"]),
            styled_table(
                [
                    ["Priority", "Extension", "Question answered"],
                    ["1", "Rate code vs time-resolved code", "Is temporal compression necessary for the duration effect?"],
                    ["2", "Independent sweeps of dropout, spontaneous spikes, threshold jitter, and readout noise", "Which noise location changes the duration boundary?"],
                    ["3", "Leaky integration with a tunable time constant", "Does a finite cortical memory window reproduce a sharper psychophysical transition?"],
                    ["4", "Add acoustic noise before the filterbank", "How does external SNR interact with frozen-sample duration?"],
                    ["5", "Fit to behavioral data", "Which parameters explain listener-level thresholds and individual differences?"],
                ],
                [0.62 * inch, 2.65 * inch, 3.75 * inch],
                styles,
            ),
            Spacer(1, 0.11 * inch),
            Paragraph("Reproduce the current sweep", styles["subsection"]),
            Table(
                [[Paragraph(
                    "python3 -m venv .venv<br/>"
                    ".venv/bin/pip install -e .<br/>"
                    "frozen-noise-demo<br/>"
                    "PYTHONPATH=src python -m unittest discover -s tests",
                    styles["code"],
                )]],
                colWidths=[7.02 * inch],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), PALE_GRAY),
                        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                        ("LEFTPADDING", (0, 0), (-1, -1), 9),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                ),
            ),
            Spacer(1, 0.12 * inch),
            Paragraph("Conclusion", styles["subsection"]),
            Paragraph(
                "The model now reproduces both sides of the qualitative psychophysical result. Short frozen noises generate large accidental fingerprints across cochlear channels, but those fingerprints contract as duration grows. Stable differences between noise-source spectra persist and become easier to estimate as within-source fluctuations average away. The result is mechanistically interpretable, while its most important alternatives - preserved temporal coding and discrimination based on nonspectral texture statistics - remain straightforward to test next.",
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    # Mathematical appendix: a plain-language map of every variable and noise source.
    story.extend(
        [
            Paragraph("Appendix A. A plain-language noise model", styles["section"]),
            Paragraph(
                "The classifier never sees the waveform directly. It sees one average spike rate for each cochlear channel. The equations below build that rate in the same order as the code, so each source of randomness has a clear location.",
                styles["body"],
            ),
            Paragraph("Symbols used throughout", styles["subsection"]),
            styled_table(
                [
                    ["Symbol", "Plain-language meaning"],
                    ["s", "Noise source or statistical family, such as white or pink noise"],
                    ["e", "One frozen exemplar: one random waveform drawn once and then reused"],
                    ["k", "One cochlear frequency channel; the model has 16 channels"],
                    ["j", "One noisy presentation of the same frozen waveform"],
                    ["T", "Sound duration in seconds"],
                    ["t", "Time within the waveform; f denotes acoustic frequency"],
                ],
                [0.82 * inch, 6.20 * inch],
                styles,
            ),
            Spacer(1, 0.08 * inch),
            callout(
                "<b>Signal versus noise.</b> The difference between two frozen waveforms is the information to be classified. Dropout, spontaneous spikes, and readout noise are fresh random changes on repeated presentations.",
                styles,
                PALE_ORANGE,
            ),
            Spacer(1, 0.08 * inch),
            Paragraph("A.1 Start with one frozen sound", styles["subsection"]),
            Paragraph(
                "The waveform is x. Its source family controls the average spectral slope: alpha = 0 is white, alpha = 1 is pink, and alpha = 2 is brown noise.",
                styles["body"],
            ),
            equation_block(
                [
                    r"S_s(f) \propto f^{-\alpha_s}",
                    r"\alpha_s \in \{0,\;0.5,\;1,\;2\}",
                ],
                styles,
                PALE_BLUE,
            ),
            Paragraph(
                "Here S is average power at frequency f. A particular exemplar e is one finite random draw from that source and is held fixed across trials.",
                styles["body"],
            ),
            Paragraph("A.2 Turn the frozen sound into clean spikes", styles["subsection"]),
            Paragraph(
                "First, filter the sound through cochlear channel k. The filter h_k selects a band of frequencies, and y is that channel's waveform:",
                styles["body"],
            ),
            equation_block(
                [r"y_{s,e,k}(t)=(h_k*x_{s,e})(t)"],
                styles,
            ),
            Paragraph(
                "Next, take the smooth amplitude envelope a. The symbol H means the Hilbert transform; the absolute-value bars mean envelope magnitude:",
                styles["body"],
            ),
            equation_block(
                [r"a_{s,e,k}(t)=\left|y_{s,e,k}(t)+i\mathcal{H}\{y_{s,e,k}\}(t)\right|"],
                styles,
            ),
            Paragraph(
                "Count an event whenever this envelope crosses the threshold upward. N is the resulting clean count during the T-second sound; the 1 ms refractory rule prevents immediate repeat events.",
                styles["body"],
            ),
            equation_block(
                [r"N_{s,e,k}(T)=\sum_{0\leq t<T}\mathbf{1}\{\mathrm{upward\ threshold\ crossing}\}"],
                styles,
                PALE_TEAL,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("Appendix A. From clean spikes to a noisy rate", styles["section"]),
            Paragraph("A.3 Add fresh noise on each presentation", styles["subsection"]),
            Paragraph(
                "There are two kinds of count noise before averaging. First, each clean spike survives independently with probability 0.97. D is the number that remains, and p = 0.03 is the dropout probability.",
                styles["body"],
            ),
            equation_block(
                [r"D_{s,e,k}^{(j)}\sim\mathrm{Binomial}\!\left(N_{s,e,k},1-p\right),\qquad p=0.03"],
                styles,
                PALE_BLUE,
            ),
            Paragraph(
                "Second, B extra events are added at random. The Poisson mean is lambda times T, where lambda = 0.2 spontaneous events per second per channel.",
                styles["body"],
            ),
            equation_block(
                [r"B_k^{(j)}\sim\mathrm{Poisson}(\lambda T),\qquad \lambda=0.2\;\mathrm{s^{-1}}"],
                styles,
            ),
            Paragraph(
                "Only now are events averaged over the whole sound. The tilde marks the rate before the final readout noise:",
                styles["body"],
            ),
            equation_block(
                [r"\widetilde r_{s,e,k}^{(j)}=\frac{D_{s,e,k}^{(j)}+B_k^{(j)}}{T}"],
                styles,
                PALE_TEAL,
            ),
            callout(
                "Because D and B are counts collected across the T-second window, their variance in rate units falls roughly as 1/T. Longer averaging makes these two noise sources smaller.",
                styles,
                PALE_ORANGE,
            ),
            Spacer(1, 0.08 * inch),
            Paragraph("A.4 Add fixed readout noise after averaging", styles["subsection"]),
            Paragraph(
                "After pooling, eta adds an independent Gaussian perturbation to each channel rate. Its standard deviation sigma_read is already measured in spikes per second, so it does not shrink when T grows. The max operation clips an impossible negative rate to zero.",
                styles["body"],
            ),
            equation_block(
                [
                    r"\eta_k^{(j)}\sim\mathcal{N}(0,\sigma_{\mathrm{read}}^2)",
                    r"r_{s,e,k}^{(j)}=\max\!\left[0,\widetilde r_{s,e,k}^{(j)}+\eta_k^{(j)}\right]",
                ],
                styles,
                PALE_BLUE,
            ),
            Paragraph(
                "In words: final rate = retained clean spikes divided by duration + spontaneous spikes divided by duration + fixed readout noise.",
                styles["body"],
            ),
            Paragraph("Complete stochastic feature equation", styles["subsection"]),
            equation_block(
                [r"r_{s,e,k}^{(j)}(T)=\max\!\left[0,\frac{\mathrm{Binomial}(N_{s,e,k}(T),0.97)+\mathrm{Poisson}(0.2T)}{T}+\mathcal{N}(0,\sigma_{\mathrm{read}}^2)\right]"],
                styles,
                PALE_TEAL,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("Appendix A. Why duration reverses the two tasks", styles["section"]),
            Paragraph("A.5 Separate the stable source from the accidental exemplar", styles["subsection"]),
            Paragraph(
                "The clean rate N/T can be viewed as two pieces. Mu is the long-run mean rate for source s in channel k. Delta is the accidental offset produced by this particular finite exemplar e.",
                styles["body"],
            ),
            equation_block(
                [
                    r"\frac{N_{s,e,k}(T)}{T}=\mu_{s,k}+\delta_{s,e,k}(T)",
                    r"\mathrm{SD}[\delta_{s,e,k}(T)]\propto T^{-1/2}",
                ],
                styles,
                PALE_BLUE,
            ),
            Paragraph(
                "The second line is the averaging law: across independently drawn frozen exemplars, the typical size of delta falls roughly like one divided by the square root of duration.",
                styles["body"],
            ),
            Paragraph("A.6 Combine the noise terms", styles["subsection"]),
            Paragraph(
                "Let rho_k be the clean spike rate in channel k. Ignoring the small effect of clipping rates at zero, repeated-trial variance is approximately:",
                styles["body"],
            ),
            equation_block(
                [r"\mathrm{Var}[r_k(T)]\approx\frac{\rho_kp(1-p)+\lambda}{T}+\sigma_{\mathrm{read}}^2"],
                styles,
                PALE_TEAL,
            ),
            Paragraph(
                "The fraction is pre-pooling count noise and falls as 1/T. The final sigma_read-squared term is post-pooling noise and remains fixed. This fixed floor is what eventually hides the shrinking exemplar fingerprint.",
                styles["body"],
            ),
            styled_table(
                [
                    ["Source of variation", "Where it enters", "What duration does"],
                    ["Frozen exemplar", "Waveform itself", "Typical offset shrinks about as 1/sqrt(T)"],
                    ["Spike dropout", "Before rate pooling", "Rate variance shrinks about as 1/T"],
                    ["Spontaneous spikes", "Before rate pooling", "Rate variance shrinks about as 1/T"],
                    ["Gaussian readout", "After rate pooling", "Variance stays fixed"],
                ],
                [1.70 * inch, 2.18 * inch, 3.14 * inch],
                styles,
            ),
            Spacer(1, 0.08 * inch),
            Paragraph("A.7 Same source versus different sources", styles["subsection"]),
            Paragraph(
                "For two exemplars of the same source, the two mu terms are identical and cancel. Only the difference between two shrinking delta terms remains:",
                styles["body"],
            ),
            equation_block(
                [r"\Delta r_{\mathrm{ex}}(T)=\delta_{s,e_1}(T)-\delta_{s,e_2}(T)"],
                styles,
            ),
            Paragraph(
                "For different sources, the stable difference between their mu terms remains even as the accidental delta terms shrink:",
                styles["body"],
            ),
            equation_block(
                [r"\Delta r_{\mathrm{source}}(T)=\mu_{s_1}-\mu_{s_2}+\delta_{s_1,e_1}(T)-\delta_{s_2,e_2}(T)"],
                styles,
                PALE_TEAL,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("Appendix A. Interpretation and model boundary", styles["section"]),
            callout(
                "<b>The crossover in one sentence.</b> Longer averaging erases the accidental details that distinguish same-source exemplars, but reveals the stable statistics that distinguish different sources.",
                styles,
                PALE_ORANGE,
            ),
            Spacer(1, 0.10 * inch),
            Paragraph("A.8 A compact discriminability sketch", styles["subsection"]),
            Paragraph(
                "The symbol d-prime means signal separation divided by trial-to-trial variability. The constant c collects the pre-pooling count-noise terms. Double bars mean the distance between two source-mean rate vectors.",
                styles["body"],
            ),
            equation_block(
                [
                    r"d_{\mathrm{ex}}^{\prime}(T)\propto\frac{T^{-1/2}}{\sqrt{c/T+\sigma_{\mathrm{read}}^2}}",
                    r"d_{\mathrm{source}}^{\prime}(T)\propto\frac{\|\mu_{s_1}-\mu_{s_2}\|}{\sqrt{c/T+\sigma_{\mathrm{read}}^2}}",
                ],
                styles,
                PALE_BLUE,
            ),
            Paragraph(
                "The exemplar numerator decreases with T, so fixed readout noise eventually wins. The source numerator stays nonzero while the count noise in the denominator decreases, so source discrimination improves and then reaches a readout-noise-limited plateau.",
                styles["body"],
            ),
            Paragraph("A.9 Acoustic masking noise is not yet included", styles["subsection"]),
            Paragraph(
                "The current model adds neural and readout variability but no new sound at the ear. A future masker condition would add an independently drawn noise waveform n before cochlear filtering. Beta would control masker strength:",
                styles["body"],
            ),
            equation_block(
                [r"x_{\mathrm{presented}}^{(j)}(t)=x_{\mathrm{frozen}}(t)+\beta n^{(j)}(t)"],
                styles,
                PALE_TEAL,
            ),
            Paragraph(
                "This would be qualitatively different from the existing readout noise: the masker would first pass through the filterbank, alter envelopes and threshold crossings, and only then affect the pooled rates.",
                styles["body"],
            ),
            Paragraph("Reference", styles["subsection"]),
            Paragraph(
                "McDermott, J. H., Schemitsch, M., &amp; Simoncelli, E. P. (2013). Summary statistics in auditory perception. <i>Nature Neuroscience, 16</i>(4), 493-498. <link href='https://doi.org/10.1038/nn.3347' color='#286B8E'>https://doi.org/10.1038/nn.3347</link>",
                styles["small"],
            ),
        ]
    )

    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the frozen-noise project report from an experiment CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="input accuracy CSV",
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=DEFAULT_SOURCE_CSV_PATH,
        help="input source-statistics CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="output PDF path",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    created = build_pdf(arguments.csv, arguments.output, arguments.source_csv)
    print(created)
