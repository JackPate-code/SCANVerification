"""
Component Inspection UI  (single file, native desktop app built with PySide6 / Qt)

Run:
    pip install PySide6
    python inspection_ui.py

HOW IT CONNECTS TO YOUR SCANNER
-------------------------------
The UI reads a folder tree. Have your analysis pipeline write each finished
batch like this (component type / part number / batch id):

    inspection_data/
      Bolts/
        BLT-M8-40/
          part.json                  <- nominal spec (optional, editable in the UI)
          BATCH-20261022-0001/
            batch.json               <- results summary (schema below)
            report.pdf               <- the PDF your pipeline generates

batch.json schema (all fields optional except batch_id):
    {
      "batch_id": "BATCH-20261022-0001",
      "timestamp": "2026-10-22T14:03:11",
      "item_count": 250, "pass_count": 246,
      "material": "Steel 304", "material_confidence": 0.97,    # hyperspectral
      "dimensions": {"Length (mm)": 40.02, "Diameter (mm)": 7.98},  # machine vision
      "Density (g/cm3)": 7.91,                                    # vibration + vision
      "vibration_peak_hz": 1523.4,                              # vibration
      "status": "CONFORM" | "NON-CONFORM"
    }

The UI refreshes automatically when files change. Set INSPECTION_DATA or use
File > Choose data folder... to point at a different root.
File > Load inspection data creates sample batches so you can see it working.
"""

from __future__ import annotations

import csv
import html
import json
import os
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from PySide6.QtCharts import (QBarCategoryAxis, QBarSeries, QBarSet, QChart, QChartView,
                              QDateTimeAxis, QLineSeries, QScatterSeries, QValueAxis)
from PySide6.QtCore import (QAbstractTableModel, QFileSystemWatcher, QMarginsF, QModelIndex, QSettings,
                            QSortFilterProxyModel, Qt, QTimer, QUrl)
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QFont, QKeySequence, QPageLayout, QPageSize,
                           QPainter, QPdfWriter, QTextDocument)
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QFrame, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPushButton, QSplitter, QStackedWidget, QTableView, QTableWidget,
                               QTableWidgetItem, QTabWidget, QTextBrowser, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

STATUS_COLORS = {"CONFORM": "#2e9e55", "NON-CONFORM": "#d64545"}


# ==========================================================================
# Data layer
# ==========================================================================
def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_")


def list_dirs(path: Path) -> list[str]:
    return sorted(p.name for p in path.iterdir()
                  if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__") if path.exists() else []


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_batches(root: Path, ctype: str, part: str) -> list[dict]:
    rows = []
    for bid in list_dirs(root / ctype / part):
        bdir = root / ctype / part / bid
        meta = read_json(bdir / "batch.json")
        try:
            ts = datetime.fromisoformat(meta["timestamp"])
        except (KeyError, ValueError, TypeError):
            ts = None
        n, ok = meta.get("item_count"), meta.get("pass_count")
        rejected = meta.get("suspect_count", meta.get("rejected_count"))
        if rejected is None and n is not None and ok is not None:
            rejected = max(0, int(n) - int(ok))
        raw_status = str(meta.get("status", "")).upper()
        # Present a binary disposition throughout the UI, including older batch records.
        conforms = (int(rejected or 0) == 0) if rejected is not None else raw_status in {"PASS", "CONFORM", "CONFORMING"}
        status = "CONFORM" if conforms else "NON-CONFORM"
        rows.append({
            "id": meta.get("batch_id", bid), "time": ts,
            "status": status,
            "items": n, "rejected": rejected,
            "rate": round(100 * ok / n, 1) if n and ok is not None else None,
            "material": meta.get("material", ""), "conf": meta.get("material_confidence"),
            "density": meta.get("Density (g/cm3)"), "vib": meta.get("vibration_peak_hz"),
            "dims": meta.get("dimensions") or {},
            "pdf": bdir / "report.pdf" if (bdir / "report.pdf").exists() else None,
            "dir": bdir, "meta": meta,
        })
    rows.sort(key=lambda r: r["time"] or datetime.min, reverse=True)
    return rows


class BatchModel(QAbstractTableModel):
    BASE = ["Batch ID", "Time", "Total Items", "Items Rejected", "Conform Rate %", "Material", "Material Confidence",
            "Density g/cm³", "Peak Vib. Hz"]
    TEXT_COLS = {0, 5}

    def __init__(self):
        super().__init__()
        self.rows: list[dict] = []
        self.dim_cols: list[str] = []

    def set_rows(self, rows: list[dict]):
        self.beginResetModel()
        self.rows = rows
        self.dim_cols = []
        for r in rows:
            for k in r["dims"]:
                if k not in self.dim_cols:
                    self.dim_cols.append(k)
        self.endResetModel()

    def columns(self) -> list[str]:
        return self.BASE + self.dim_cols

    def value(self, row: dict, col: int):
        if col < len(self.BASE):
            return [row["id"], row["time"], row["items"], row["rejected"], row["rate"], row["material"],
                    row["conf"], row["density"], row["vib"]][col]
        return row["dims"].get(self.dim_cols[col - len(self.BASE)])

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.columns())

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.columns()[section]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        col, v = index.column(), self.value(self.rows[index.row()], index.column())
        if role == Qt.DisplayRole:
            if v in (None, ""):
                return "—"
            if isinstance(v, datetime):
                return v.strftime("%Y-%m-%d %H:%M")
            if isinstance(v, float):
                return f"{v:.1f}" if col == 4 else f"{v:.3f}"
            return str(v)
        if role == Qt.UserRole:  # sort key
            if isinstance(v, datetime):
                return v.timestamp()
            if v is None:
                return "" if col in self.TEXT_COLS else -1e18
            return v
        if role == Qt.ForegroundRole and col == 3 and isinstance(v, (int, float)) and v > 0:
            return QColor(STATUS_COLORS["NON-CONFORM"])
        if role == Qt.FontRole and col == 3 and isinstance(v, (int, float)) and v > 0:
            f = QFont(); f.setBold(True); return f
        if role == Qt.TextAlignmentRole and col not in self.TEXT_COLS and col != 1:
            return int(Qt.AlignRight | Qt.AlignVCenter)


class BatchFilter(QSortFilterProxyModel):
    def __init__(self):
        super().__init__()
        self.status, self.text = "All", ""
        self.setSortRole(Qt.UserRole)

    def set_filters(self, status: str, text: str):
        self.status, self.text = status, text.lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        r = self.sourceModel().rows[row]
        return (self.status == "All" or r["status"] == self.status) and self.text in r["id"].lower()


# ==========================================================================
# Sample inspection catalog
# ==========================================================================
PART_CATALOG = {
    "Bolts": {
        "BLT-M6-20": dict(material="Zinc-plated steel", density=7.85, dims={"Length (mm)": 20.0, "Diameter (mm)": 6.0, "Head AF (mm)": 10.0}),
        "BLT-M8-40": dict(material="Stainless steel 304", density=7.93, dims={"Length (mm)": 40.0, "Diameter (mm)": 8.0, "Head AF (mm)": 13.0}),
        "BLT-M10-60": dict(material="Stainless steel 316", density=8.00, dims={"Length (mm)": 60.0, "Diameter (mm)": 10.0, "Head AF (mm)": 16.0}),
        "BLT-1-2-90": dict(material="SAE Grade 5 steel", density=7.85, dims={"Length (mm)": 76.2, "Diameter (mm)": 12.7, "Head AF (mm)": 19.05}),
    },
    "Nuts": {
        "NUT-M6-HEX": dict(material="Zinc-plated steel", density=7.85, dims={"width_mm": 10.0, "height_mm": 5.0, "thread_Diameter (mm)": 6.0}),
        "NUT-M8-HEX": dict(material="Brass", density=8.50, dims={"width_mm": 13.0, "height_mm": 6.5, "thread_Diameter (mm)": 8.0}),
        "NUT-M10-NYLON": dict(material="Steel / nylon insert", density=7.80, dims={"width_mm": 17.0, "height_mm": 10.0, "thread_Diameter (mm)": 10.0}),
    },
    "Chips": {
        "IC-QFN32-5X5": dict(material="Silicon / epoxy", density=2.30, dims={"Length (mm)": 5.0, "width_mm": 5.0, "height_mm": 0.9}),
        "IC-SOIC8-4P9X3P9": dict(material="Silicon / epoxy", density=2.33, dims={"Length (mm)": 4.9, "width_mm": 3.9, "height_mm": 1.5}),
        "IC-LQFP64-10X10": dict(material="Silicon / epoxy", density=2.32, dims={"Length (mm)": 10.0, "width_mm": 10.0, "height_mm": 1.4}),
    },
}


def write_report_pdf(path: Path, ctype: str, part: str, meta: dict) -> None:
    rows = [("Component Type", ctype), ("Part number", part), ("Batch", meta["batch_id"]),
            ("Timestamp", meta["timestamp"]), ("Items Scanned", meta["item_count"]),
            ("Items Conforming", meta["pass_count"]), ("Items Rejected", meta.get("rejected_count", 0)),
            ("Material", meta["material"]), ("Density (g/cm³)", meta["Density (g/cm3)"]),
            ("Peak Vibration (Hz)", meta.get("vibration_peak_hz", "—")),
            *meta["dimensions"].items(), ("Batch result", "Conform" if str(meta.get("status", "")).upper() in {"PASS", "CONFORM"} else "Non-conform")]
    body = "".join(f"<tr><td><b>{html.escape(str(k))}</b></td><td>{html.escape(str(v))}</td></tr>" for k, v in rows)
    html_doc = (f"<h1>Batch Summary: {html.escape(meta['batch_id'])}</h1>"
                f"<table border='1' cellpadding='6' cellspacing='0' width='100%'>{body}</table>")
    writer = QPdfWriter(str(path)); writer.setPageSize(QPageSize(QPageSize.PageSizeId.Letter))
    writer.setPageMargins(QMarginsF(15, 15, 15, 15), QPageLayout.Unit.Millimeter)
    doc = QTextDocument(); doc.setHtml(html_doc); doc.print_(writer)


def write_component_pdf(path: Path, ctype: str, part: str, batch_id: str, record: dict) -> None:
    rows = []
    for metric, expected in record.get("expected", {}).items():
        actual = record.get("actual", {}).get(metric)
        flags = record.get("flags", [])
        metric_flagged = any(flag in record.get("expected", {}) for flag in flags)
        flagged = metric in flags or (record.get("verdict") in {"REJECT", "NON-CONFORM"} and not metric_flagged)
        color = "#ffe0e0" if flagged else "#e8f6eb"
        verdict = "Non-conform" if flagged else "Conform"
        rows.append(f"<tr style='background:{color}'><td>{html.escape(str(metric))}</td>"
                    f"<td>{html.escape(str(expected))}</td><td>{html.escape(str(actual if actual is not None else '—'))}</td>"
                    f"<td><b>{verdict}</b></td></tr>")
    body = "".join(rows) or "<tr><td colspan='4'>No numeric measurements available</td></tr>"
    flag_text = ", ".join(record.get("flags", [])) or "None"
    doc_html = (f"<h1>Individual Component Inspection</h1><p><b>Component:</b> {html.escape(record['component_id'])}</p>"
                f"<p><b>Type / part:</b> {html.escape(ctype)} / {html.escape(part)}<br>"
                f"<b>Batch:</b> {html.escape(batch_id)}<br><b>Result:</b> {'Non-conform' if record.get('verdict') in {'REJECT', 'NON-CONFORM'} else 'Conform'}</p>"
                "<table border='1' cellpadding='6' cellspacing='0' width='100%'><tr style='background:#263b50;color:white'>"
                "<th>Metric</th><th>Expected value</th><th>Read value</th><th>Inspection</th></tr>" + body + "</table>"
                f"<p><b>Non-conforming measurements:</b> {html.escape(flag_text if flag_text != 'None' else 'None')}</p>")
    writer = QPdfWriter(str(path)); writer.setPageSize(QPageSize(QPageSize.PageSizeId.Letter))
    writer.setPageMargins(QMarginsF(12, 12, 12, 12), QPageLayout.Unit.Millimeter)
    doc = QTextDocument(); doc.setHtml(doc_html); doc.print_(writer)


def make_component_record(component_id: str, spec: dict, rng: random.Random, anomaly: bool = False) -> dict:
    expected = dict(spec.get("dimensions", {}) or {})
    if spec.get("Density (g/cm3)") is not None:
        expected["Density (g/cm3)"] = spec["Density (g/cm3)"]
    actual, flags = {}, []
    tol_pct = float(spec.get("tolerance_pct", 1.5) or 1.5)
    for metric, nominal in expected.items():
        try: nominal = float(nominal)
        except (ValueError, TypeError):
            actual[metric] = nominal
            continue
        deviation = rng.uniform(-tol_pct / 200, tol_pct / 200)
        if anomaly and rng.random() < 0.75:
            deviation = rng.choice([-1, 1]) * rng.uniform(tol_pct / 100 * 1.5, tol_pct / 100 * 4.0)
        value = nominal * (1 + deviation)
        actual[metric] = round(value, 4 if abs(value) < 10 else 3)
        if abs(value - nominal) > abs(nominal) * tol_pct / 100:
            flags.append(metric)
    return {"component_id": component_id, "expected": expected, "actual": actual,
            "flags": flags, "verdict": "NON-CONFORM" if flags else "CONFORM"}


def write_component_records(batch_dir: Path, ctype: str, part: str, batch_id: str,
                            records: list[dict], write_pdfs: bool = True) -> None:
    cdir = batch_dir / "components"; cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "components.json").write_text(json.dumps(records, indent=2))
    if write_pdfs:
        for record in records:
            write_component_pdf(cdir / f"{slug(record['component_id'])}.pdf", ctype, part, batch_id, record)


def seed_inspection_data(root: Path) -> None:
    """Create realistic sample batches for bolts, nuts, and IC packages."""
    rng, now = random.Random(42), datetime.now()
    for ctype, parts in PART_CATALOG.items():
        for part, base in parts.items():
            pdir = root / ctype / part
            pdir.mkdir(parents=True, exist_ok=True)
            spec = {"description": f"{ctype[:-1].lower().capitalize()} component, part number {part}", "material": base["material"],
                    "Density (g/cm3)": base["density"], "dimensions": base["dims"], "tolerance_pct": 1.5}
            (pdir / "part.json").write_text(json.dumps(spec, indent=2))
            for batch_idx in range(2):
                ts = now - timedelta(days=1 - batch_idx, hours=rng.randint(0, 5))
                bid = f"BATCH-{ts:%Y%m%d}-{batch_idx + 1:02d}"
                bdir = pdir / bid; bdir.mkdir(parents=True, exist_ok=True)
                records = []
                for i in range(1, 9):
                    anomaly = rng.random() < 0.13
                    records.append(make_component_record(f"{bid}-COMP-{i:03d}", spec, rng, anomaly))
                rejected = sum(r["verdict"] in {"REJECT", "NON-CONFORM"} for r in records)
                # Stable aggregate measurements support the summary table and cross-part charts.
                dims = {k: round(sum(float(r["actual"].get(k, v)) for r in records) / len(records), 4)
                        for k, v in base["dims"].items()}
                meta = {"batch_id": bid, "timestamp": ts.isoformat(timespec="seconds"), "item_count": len(records),
                        "pass_count": len(records) - rejected, "rejected_count": rejected, "suspect_count": rejected,
                        "material": base["material"], "material_confidence": round(rng.uniform(.94, .998), 3),
                        "dimensions": dims, "Density (g/cm3)": round(base["density"] * rng.uniform(.99, 1.01), 3),
                        "vibration_peak_hz": round(rng.uniform(1400, 1650), 1),
                        "status": "NON-CONFORM" if rejected else "CONFORM",
                        "sample_data": True}
                (bdir / "batch.json").write_text(json.dumps(meta, indent=2))
                write_report_pdf(bdir / "report.pdf", ctype, part, meta)
                write_component_records(bdir, ctype, part, bid, records)


# ==========================================================================
# UI pieces
# ==========================================================================
class MetricCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(0)
        t = QLabel(title)
        t.setStyleSheet("color: gray; font-size: 11px;")
        self.value = QLabel("—")
        f = self.value.font()
        f.setPointSize(f.pointSize() + 6)
        f.setBold(True)
        self.value.setFont(f)
        lay.addWidget(t)
        lay.addWidget(self.value)

    def set(self, text: str):
        self.value.setText(text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Component Inspection")
        self.resize(1450, 900)
        self.settings = QSettings("InspectionUI", "InspectionUI")
        self.root = Path(os.environ.get("INSPECTION_DATA") or
                         self.settings.value("root", str(Path(__file__).resolve().parent))).resolve()
        self.ctype: str | None = None
        self.part: str | None = None
        self.current: dict | None = None

        self.model = BatchModel()
        self.proxy = BatchFilter()
        self.proxy.setSourceModel(self.model)
        self.pdf_doc = QPdfDocument(self)
        self.watcher = QFileSystemWatcher(self)
        self.reload_timer = QTimer(self, singleShot=True, interval=500)
        self.reload_timer.timeout.connect(self.refresh_all)
        self.watcher.directoryChanged.connect(lambda _: self.reload_timer.start())

        self._build_menu()
        self._build_ui()
        self.refresh_all()

    # ---- construction ------------------------------------------------------
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, slot, key in [("Choose data folder…", self.choose_root, None),
                                ("Load inspection data (bolts, nuts, IC packages)", self.load_inspection_data, None),
                                ("Generate sample inspection + PDF…", self.generate_sample_batch, None),
                                ("Refresh", self.refresh_all, QKeySequence("F5")),
                                (None, None, None), ("Quit", self.close, QKeySequence.Quit)]:
            if text is None:
                m.addSeparator()
                continue
            a = QAction(text, self)
            a.triggered.connect(slot)
            if key:
                a.setShortcut(key)
            m.addAction(a)

    def _build_ui(self):
        # Left: component tree
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 4, 8)
        ll.addWidget(QLabel("<b>Components</b>"))
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self.on_tree_changed)
        ll.addWidget(self.tree, 1)
        row = QHBoxLayout()
        b1, b2 = QPushButton("+ Type"), QPushButton("+ Part number")
        b1.clicked.connect(self.add_type)
        b2.clicked.connect(self.add_part)
        row.addWidget(b1)
        row.addWidget(b2)
        ll.addLayout(row)
        self.root_label = QLabel()
        self.root_label.setStyleSheet("color: gray; font-size: 10px;")
        self.root_label.setWordWrap(True)
        ll.addWidget(self.root_label)

        # Right: header, metrics, filters, table + detail tabs
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 8, 8, 8)
        self.title = QLabel("Select a part number")
        tf = self.title.font()
        tf.setPointSize(tf.pointSize() + 8)
        tf.setBold(True)
        self.title.setFont(tf)
        self.subtitle = QLabel("")
        self.subtitle.setStyleSheet("color: gray;")
        rl.addWidget(self.title)
        rl.addWidget(self.subtitle)

        cards = QHBoxLayout()
        self.m_batches, self.m_items = MetricCard("Batches"), MetricCard("Total Items")
        self.m_rate, self.m_last = MetricCard("Items Rejected"), MetricCard("Latest Batch")
        for c in (self.m_batches, self.m_items, self.m_rate, self.m_last):
            cards.addWidget(c)
        rl.addLayout(cards)

        filt = QHBoxLayout()
        self.status_box = QComboBox()
        self.status_box.addItems(["All", "Conform", "Non-conform"])
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search batch ID…")
        self.search.setClearButtonEnabled(True)
        self.status_box.currentTextChanged.connect(self.apply_filters)
        self.search.textChanged.connect(self.apply_filters)
        filt.addWidget(QLabel("Status:"))
        filt.addWidget(self.status_box)
        filt.addWidget(self.search, 1)
        rl.addLayout(filt)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.selectionModel().currentRowChanged.connect(self.on_batch_changed)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_report_tab(), "Component Document")
        self.tabs.addTab(self._build_components_tab(), "Inspected Items")
        self.tabs.addTab(self._build_details_tab(), "Batch Measurements")
        self.tabs.addTab(self._build_trends_tab(), "Trends")
        self.tabs.addTab(self._build_spec_tab(), "Part Specifications")

        vs = QSplitter(Qt.Vertical)
        vs.addWidget(self.table)
        vs.addWidget(self.tabs)
        vs.setSizes([260, 560])
        rl.addWidget(vs, 1)

        hs = QSplitter(Qt.Horizontal)
        hs.addWidget(left)
        hs.addWidget(right)
        hs.setSizes([280, 1170])
        hs.setStretchFactor(1, 1)
        self.setCentralWidget(hs)
        self.statusBar().showMessage("Ready")

    def _build_report_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        self.btn_open = QPushButton("Open in PDF viewer")
        self.btn_save = QPushButton("Save a copy…")
        self.btn_open.clicked.connect(self.open_external)
        self.btn_save.clicked.connect(self.save_copy)
        bar.addWidget(self.btn_open)
        bar.addWidget(self.btn_save)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.pdf_view = QPdfView()
        self.pdf_view.setDocument(self.pdf_doc)
        self.pdf_view.setPageMode(QPdfView.PageMode.MultiPage)
        self.pdf_view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
        self.pdf_msg = QLabel("Select an inspected item to view its component PDF.")
        self.pdf_msg.setAlignment(Qt.AlignCenter)
        self.pdf_msg.setStyleSheet("color: gray;")
        self.pdf_stack = QStackedWidget()
        self.pdf_stack.addWidget(self.pdf_msg)
        self.pdf_stack.addWidget(self.pdf_view)
        lay.addWidget(self.pdf_stack, 1)
        return w

    def _build_components_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(QLabel("Select any component row to preview its report. Use View selected component document to open the full PDF. Red cells indicate non-conforming measurements."))
        self.components_table = QTableWidget(0, 5)
        self.components_table.setHorizontalHeaderLabels(["Component ID", "Metric", "Expected value", "Read value", "Result"])
        self.components_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.components_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.components_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.components_table.setAlternatingRowColors(True)
        self.components_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.components_table.horizontalHeader().setStretchLastSection(True)
        self.components_table.itemSelectionChanged.connect(self.on_component_selected)
        self.components_table.cellDoubleClicked.connect(lambda _r, _c: self.view_selected_component_document())
        lay.addWidget(self.components_table, 1)
        actions = QHBoxLayout()
        self.btn_component_document = QPushButton("View selected component document")
        self.btn_component_document.clicked.connect(self.view_selected_component_document)
        actions.addWidget(self.btn_component_document)
        actions.addStretch(1)
        lay.addLayout(actions)
        return w

    def _build_details_tab(self) -> QWidget:
        self.details = QTextBrowser()
        return self.details

    def _build_trends_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        top = QHBoxLayout()
        top.addWidget(QLabel("Metric:"))
        self.metric_box = QComboBox()
        self.metric_box.currentIndexChanged.connect(self.update_chart)
        top.addWidget(self.metric_box)
        top.addStretch(1)
        lay.addLayout(top)
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        lay.addWidget(self.chart_view, 1)
        return w

    def _build_spec_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.sp_desc, self.sp_mat = QLineEdit(), QLineEdit()
        self.sp_density = QDoubleSpinBox()
        self.sp_density.setDecimals(3)
        self.sp_density.setRange(0, 100)
        self.sp_density.setSuffix(" g/cm³")
        self.sp_tol = QDoubleSpinBox()
        self.sp_tol.setRange(0, 100)
        self.sp_tol.setSuffix(" %")
        form.addRow("Description", self.sp_desc)
        form.addRow("Material", self.sp_mat)
        form.addRow("Nominal density", self.sp_density)
        form.addRow("Tolerance", self.sp_tol)
        lay.addLayout(form)
        lay.addWidget(QLabel("<b>Nominal dimensions</b>"))
        self.sp_dims = QTableWidget(0, 2)
        self.sp_dims.setHorizontalHeaderLabels(["Dimension", "Nominal"])
        self.sp_dims.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self.sp_dims, 1)
        row = QHBoxLayout()
        add, rem, save = QPushButton("Add row"), QPushButton("Remove row"), QPushButton("Save spec")
        add.clicked.connect(lambda: self.sp_dims.insertRow(self.sp_dims.rowCount()))
        rem.clicked.connect(lambda: self.sp_dims.removeRow(self.sp_dims.currentRow()))
        save.clicked.connect(self.save_spec)
        for b in (add, rem):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(save)
        lay.addLayout(row)
        return w

    # ---- tree / navigation -------------------------------------------------
    def refresh_all(self):
        """Rebuild the tree (keeping selection) and reload the current part."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.root_label.setText(f"Data folder:\n{self.root}")
        keep = (self.ctype, self.part)
        self.tree.blockSignals(True)
        self.tree.clear()
        watch = [str(self.root)]
        target = None
        for t in list_dirs(self.root):
            ti = QTreeWidgetItem([t])
            ti.setData(0, Qt.UserRole, (t, None))
            f = ti.font(0)
            f.setBold(True)
            ti.setFont(0, f)
            self.tree.addTopLevelItem(ti)
            watch.append(str(self.root / t))
            if keep == (t, None):
                target = ti
            for p in list_dirs(self.root / t):
                pi = QTreeWidgetItem([p])
                pi.setData(0, Qt.UserRole, (t, p))
                ti.addChild(pi)
                watch.append(str(self.root / t / p))
                if (t, p) == keep:
                    target = pi
            ti.setExpanded(True)
        self.tree.blockSignals(False)
        if self.watcher.directories():
            self.watcher.removePaths(self.watcher.directories())
        self.watcher.addPaths(sorted(set(watch)))
        if target is None and self.tree.topLevelItemCount() and keep == (None, None):
            first = self.tree.topLevelItem(0)
            target = first.child(0) if first.childCount() else first
        if target:
            self.tree.setCurrentItem(target)
            self.on_tree_changed(target, None)
        else:
            self.on_tree_changed(None, None)

    def on_tree_changed(self, item, _prev):
        if item is None:
            self.ctype = self.part = None
            self.model.set_rows([])
            self.title.setText("Select a part number")
            self.subtitle.setText("Create a component type with “+ Type”, or use File > Load inspection data.")
            self.update_metrics([])
            self.show_batch(None)
            return
        t, p = item.data(0, Qt.UserRole)
        self.ctype, self.part = t, p
        if p:
            self.load_part(t, p)
        else:
            self.model.set_rows([])
            self.title.setText(t)
            self.subtitle.setText("Add or select a part number under this component type.")
            self.update_metrics([])
            self.show_batch(None)

    def add_type(self):
        name, ok = QInputDialog.getText(self, "New component type", "Type name (e.g. Washers):")
        if ok and slug(name):
            (self.root / slug(name)).mkdir(parents=True, exist_ok=True)
            self.ctype, self.part = slug(name), None
            self.refresh_all()

    def add_part(self):
        if not self.ctype:
            QMessageBox.information(self, "Add part number", "Select a component type first.")
            return
        name, ok = QInputDialog.getText(self, "New part number", f"Part number for {self.ctype}:")
        if ok and slug(name):
            (self.root / self.ctype / slug(name)).mkdir(parents=True, exist_ok=True)
            self.part = slug(name)
            self.refresh_all()

    def choose_root(self):
        d = QFileDialog.getExistingDirectory(self, "Choose inspection data folder", str(self.root))
        if d:
            self.root = Path(d)
            self.settings.setValue("root", d)
            self.ctype = self.part = None
            self.refresh_all()

    def load_inspection_data(self):
        seed_inspection_data(self.root)
        self.refresh_all()
        self.statusBar().showMessage("Inspection data loaded for bolts, nuts, and IC packages", 6000)

    def generate_sample_batch(self):
        """Generate a model-backed bolt batch; model-backed scans currently support the datasheet bolt only."""
        try:
            import pdf as pipeline
            result = pipeline.simulate_and_write(self.root, n=40, contamination=0.04)
            # Create an individual report for every inspected component from the real per-item CSV.
            pdir = Path(result["dir"])
            records = self.records_from_csv(pdir, result["ctype"], result["part"], result["batch_id"])
            write_component_records(pdir, result["ctype"], result["part"], result["batch_id"], records)
            self.refresh_all()
            self.statusBar().showMessage(f"Generated {len(records)} component PDFs in {pdir}", 8000)
        except Exception as exc:
            QMessageBox.critical(self, "Sample generation failed", str(exc))

    def records_from_csv(self, batch_dir: Path, ctype: str, part: str, batch_id: str) -> list[dict]:
        spec = read_json(batch_dir.parent / "part.json")
        records = []
        csv_path = batch_dir / "parts.csv"
        if not csv_path.exists():
            return records
        expected = dict(spec.get("dimensions", {}) or {})
        # Map model CSV fields to UI spec keys; actual CSV values are in inches for fastener dimensions.
        aliases = {"Length (mm)": ("length_in", 25.4), "Head AF (mm)": ("head_width_in", 25.4),
                   "head_h_mm": ("head_height_in", 25.4), "shank_mm": ("shank_in", 25.4),
                   "thread_mm": ("thread_in", 25.4), "tpi": ("tpi", 1.0)}
        tol = float(spec.get("tolerance_pct", 1.5) or 1.5)
        with csv_path.open(newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f), 1):
                actual, flags = {}, []
                for metric, nominal in expected.items():
                    alias = aliases.get(metric)
                    if alias and row.get(alias[0]):
                        try: value = float(row[alias[0]]) * alias[1]
                        except ValueError: value = None
                    elif metric == "Diameter (mm)" and row.get("shank_in"):
                        value = float(row["shank_in"]) * 25.4
                    else:
                        value = None
                    if value is not None:
                        value = round(value, 4)
                        actual[metric] = value
                        if nominal and abs(value - float(nominal)) > abs(float(nominal)) * tol / 100:
                            flags.append(metric)
                if row.get("verdict", "").upper() in {"SUSPECT", "NON-CONFORM", "REJECT"} and not flags:
                    flags.append("Model verification")
                records.append({"component_id": f"{batch_id}-ITEM-{i:04d}", "expected": expected,
                                "actual": actual, "flags": flags,
                                "verdict": "NON-CONFORM" if flags or row.get("verdict", "").upper() in {"SUSPECT", "NON-CONFORM", "REJECT"} else "CONFORM"})
        return records

    # ---- part / batch loading ---------------------------------------------
    def load_part(self, ctype: str, part: str):
        rows = load_batches(self.root, ctype, part)
        spec = read_json(self.root / ctype / part / "part.json")
        self.title.setText(f"{ctype}  ›  {part}")
        self.subtitle.setText(spec.get("description", ""))
        self.model.set_rows(rows)
        self.table.resizeColumnsToContents()
        self.update_metrics(rows)
        self.populate_metric_box()
        self.populate_spec(spec)
        self.apply_filters()
        if self.proxy.rowCount():
            self.table.sortByColumn(1, Qt.DescendingOrder)
            self.table.selectRow(0)
            self.on_batch_changed(self.proxy.index(0, 0), None)
        else:
            self.show_batch(None)

    def update_metrics(self, rows):
        self.m_batches.set(str(len(rows)))
        self.m_items.set(f"{sum(r['items'] or 0 for r in rows):,}")
        rejected = sum(int(r["rejected"] or 0) for r in rows)
        self.m_rate.set(f"{rejected:,}")
        times = [r["time"] for r in rows if r["time"]]
        self.m_last.set(max(times).strftime("%b %d, %H:%M") if times else "—")

    def apply_filters(self):
        selected = self.status_box.currentText()
        status = {"Conform": "CONFORM", "Non-conform": "NON-CONFORM"}.get(selected, "All")
        self.proxy.set_filters(status, self.search.text())

    def on_batch_changed(self, current, _prev):
        if not current.isValid():
            self.show_batch(None)
            return
        self.show_batch(self.model.rows[self.proxy.mapToSource(current).row()])

    def show_batch(self, batch: dict | None):
        self.current = batch
        self.current_component_pdf = None
        records = self.load_component_records(batch) if batch else []
        self.populate_components(records)
        self.show_details(batch)
        if records:
            self.components_table.selectRow(0)
            self.load_component_pdf(records[0])
        else:
            self.set_pdf_path(batch["pdf"] if batch else None, batch)

    def load_component_records(self, batch: dict) -> list[dict]:
        cdir = batch["dir"] / "components"
        data_file = cdir / "components.json"
        if data_file.exists():
            try:
                return json.loads(data_file.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        records = self.records_from_csv(batch["dir"], self.ctype or "", self.part or "", batch["id"])
        if not records:
            # Legacy aggregate-only batches cannot recover true per-item reads; label these as generated estimates.
            spec = read_json(self.root / (self.ctype or "") / (self.part or "") / "part.json")
            n = min(int(batch.get("items") or 0), 250)
            rng = random.Random(batch["id"])
            records = [make_component_record(f"{batch['id']}-ITEM-{i:04d}", spec, rng, False)
                       for i in range(1, n + 1)]
            for record in records:
                record["data_note"] = "Estimated from part specification; legacy batch had no per-item readings"
        if records:
            write_component_records(batch["dir"], self.ctype or "Component", self.part or "Unknown", batch["id"], records)
        return records

    def populate_components(self, records: list[dict]):
        self.components_table.blockSignals(True)
        self.components_table.setRowCount(0)
        for record in records:
            metrics = list(record.get("expected", {})) or ["Overall verdict"]
            for metric in metrics:
                r = self.components_table.rowCount()
                self.components_table.insertRow(r)
                actual = record.get("actual", {}).get(metric, "—")
                expected = record.get("expected", {}).get(metric, "—")
                flags = record.get("flags", [])
                metric_flagged = any(flag in record.get("expected", {}) for flag in flags)
                flagged = metric in flags or (record.get("verdict") in {"REJECT", "NON-CONFORM"} and not metric_flagged)
                result = "Non-conform" if flagged else "Conform"
                vals = [record.get("component_id", ""), metric, str(expected), str(actual), result]
                for c, value in enumerate(vals):
                    item = QTableWidgetItem(value)
                    item.setData(Qt.UserRole, record)
                    if flagged:
                        item.setBackground(QColor("#ffdede"))
                        item.setForeground(QColor("#8d1111"))
                    self.components_table.setItem(r, c, item)
        self.components_table.blockSignals(False)

    def on_component_selected(self):
        items = self.components_table.selectedItems()
        if items:
            record = items[0].data(Qt.UserRole)
            if record:
                self.load_component_pdf(record)

    def load_component_pdf(self, record: dict):
        if not self.current:
            return
        path = self.current["dir"] / "components" / f"{slug(record['component_id'])}.pdf"
        if not path.exists():
            write_component_pdf(path, self.ctype or "Component", self.part or "Unknown", self.current["id"], record)
        self.set_pdf_path(path, self.current)

    def view_selected_component_document(self):
        """Open the selected component's individual PDF in the built-in document tab."""
        items = self.components_table.selectedItems()
        if not items:
            QMessageBox.information(self, "Select a component", "Select a component row first.")
            return
        record = items[0].data(Qt.UserRole)
        if not record:
            return
        self.load_component_pdf(record)
        self.tabs.setCurrentIndex(0)

    def set_pdf_path(self, path: Path | None, batch: dict | None):
        self.current_component_pdf = path
        has_pdf = bool(path and Path(path).exists())
        self.btn_open.setEnabled(has_pdf)
        self.btn_save.setEnabled(has_pdf)
        if has_pdf:
            self.pdf_doc.load(str(path))
            self.pdf_stack.setCurrentWidget(self.pdf_view)
        else:
            self.pdf_doc.close()
            self.pdf_msg.setText("No component PDF available." if batch else "Select a batch and inspected item.")
            self.pdf_stack.setCurrentWidget(self.pdf_msg)

    def show_details(self, b: dict | None):
        if not b:
            self.details.setHtml("<p style='color:gray'>No batch selected.</p>")
            return
        col = STATUS_COLORS.get(b["status"], "gray")
        dims = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in b["dims"].items())
        conf = f"{b['conf']:.1%}" if isinstance(b["conf"], (int, float)) else "—"
        self.details.setHtml(f"""
            <h2>{b['id']} <span style='color:{col}'>{'Conform' if b['status'] == 'CONFORM' else 'Non-conform'}</span></h2>
            <p>{b['time'].strftime('%Y-%m-%d %H:%M:%S') if b['time'] else ''} &nbsp;|&nbsp;
               {b['items'] or '—'} total items, {b['rejected'] or 0} rejected, {b['rate'] if b['rate'] is not None else '—'}% conforming</p>
            <table cellpadding='4'>
              <tr><td colspan='2'><u>Hyperspectral</u></td></tr>
              <tr><td>Material</td><td><b>{b['material'] or '—'}</b> (confidence {conf})</td></tr>
              <tr><td colspan='2'><u>Machine vision</u></td></tr>{dims or "<tr><td>—</td></tr>"}
              <tr><td colspan='2'><u>Vibration / density</u></td></tr>
              <tr><td>Density</td><td><b>{b['density'] if b['density'] is not None else '—'}</b> g/cm³</td></tr>
              <tr><td>Peak vibration</td><td><b>{b['vib'] if b['vib'] is not None else '—'}</b> Hz</td></tr>
            </table>""")

    def open_external(self):
        if getattr(self, "current_component_pdf", None) and Path(self.current_component_pdf).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_component_pdf)))

    def save_copy(self):
        source = getattr(self, "current_component_pdf", None)
        if not (source and Path(source).exists()):
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Save component PDF", f"{Path(source).stem}.pdf", "PDF (*.pdf)")
        if dest:
            Path(dest).write_bytes(Path(source).read_bytes())
            self.statusBar().showMessage(f"Saved {dest}", 4000)

    # ---- cross-part comparison charts --------------------------------------
    def populate_metric_box(self):
        metrics = self.collect_comparison_metrics()
        current = self.metric_box.currentText()
        self.metric_box.blockSignals(True); self.metric_box.clear()
        for name in metrics:
            self.metric_box.addItem(name)
        if current and self.metric_box.findText(current) >= 0:
            self.metric_box.setCurrentText(current)
        self.metric_box.blockSignals(False)
        self.update_chart()

    def collect_comparison_metrics(self) -> dict[str, list[tuple[str, float]]]:
        result: dict[str, list[tuple[str, float]]] = {}
        for ctype in list_dirs(self.root):
            for part in list_dirs(self.root / ctype):
                rows = load_batches(self.root, ctype, part)
                if not rows: continue
                # Compare each part number using its latest batch, rather than plotting batch history.
                row = max(rows, key=lambda x: x["time"] or datetime.min)
                label = f"{part} ({ctype})"
                for key, value in (row.get("dims") or {}).items():
                    if isinstance(value, (int, float)):
                        result.setdefault(key, []).append((label, float(value)))
                for key, value in (("Density (g/cm³)", row.get("density")),
                                   ("Peak vibration (Hz)", row.get("vib")),
                                   ("Total items", row.get("items")),
                                   ("Items rejected", row.get("rejected")),
                                   ("Conform rate (%)", row.get("rate"))):
                    if isinstance(value, (int, float)):
                        result.setdefault(key, []).append((label, float(value)))
        return {k: sorted(v, key=lambda p: p[0]) for k, v in result.items() if v}

    def update_chart(self):
        chart = QChart(); chart.legend().hide()
        metric = self.metric_box.currentText()
        points = self.collect_comparison_metrics().get(metric, [])
        if points:
            barset = QBarSet(metric)
            categories = []
            for label, value in points:
                categories.append(label)
                barset.append(value)
            series = QBarSeries(); series.append(barset); chart.addSeries(series)
            ax = QBarCategoryAxis(); ax.append(categories); ax.setLabelsAngle(-35)
            ay = QValueAxis()
            vals = [v for _, v in points]; lo, hi = min(vals), max(vals)
            pad = (hi - lo) * 0.15 or abs(hi) * 0.03 or 1
            ay.setRange(min(0, lo - pad), hi + pad)
            chart.addAxis(ax, Qt.AlignBottom); chart.addAxis(ay, Qt.AlignLeft)
            series.attachAxis(ax); series.attachAxis(ay)
        chart.setTitle(f"{metric} compared across part numbers (latest batch)")
        self.chart_view.setChart(chart)

    # ---- part spec ---------------------------------------------------------
    def populate_spec(self, spec: dict):
        self.sp_desc.setText(spec.get("description", ""))
        self.sp_mat.setText(spec.get("material", ""))
        self.sp_density.setValue(float(spec.get("Density (g/cm3)", 0) or 0))
        self.sp_tol.setValue(float(spec.get("tolerance_pct", 1.0) or 0))
        dims = spec.get("dimensions", {}) or {}
        self.sp_dims.setRowCount(len(dims))
        for r, (k, v) in enumerate(dims.items()):
            self.sp_dims.setItem(r, 0, QTableWidgetItem(str(k)))
            self.sp_dims.setItem(r, 1, QTableWidgetItem(str(v)))

    def save_spec(self):
        if not (self.ctype and self.part):
            return
        dims = {}
        for r in range(self.sp_dims.rowCount()):
            k, v = self.sp_dims.item(r, 0), self.sp_dims.item(r, 1)
            if k and v and k.text().strip():
                try:
                    dims[k.text().strip()] = float(v.text())
                except ValueError:
                    QMessageBox.warning(self, "Invalid value", f"'{v.text()}' is not a number.")
                    return
        (self.root / self.ctype / self.part / "part.json").write_text(json.dumps({
            "description": self.sp_desc.text(), "material": self.sp_mat.text(),
            "Density (g/cm3)": self.sp_density.value(), "dimensions": dims,
            "tolerance_pct": self.sp_tol.value()}, indent=2))
        self.subtitle.setText(self.sp_desc.text())
        self.populate_metric_box()
        self.statusBar().showMessage("Spec Saved", 4000)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()