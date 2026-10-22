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
      "dimensions": {"length_mm": 40.02, "diameter_mm": 7.98},  # machine vision
      "density_g_cm3": 7.91,                                    # vibration + vision
      "vibration_peak_hz": 1523.4,                              # vibration
      "status": "PASS" | "REVIEW" | "FAIL"
    }

The UI refreshes automatically when files change. Set INSPECTION_DATA or use
File > Choose data folder... to point at a different root.
File > Load demo data populates sample parts so you can see it working.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from PySide6.QtCharts import QChart, QChartView, QDateTimeAxis, QLineSeries, QScatterSeries, QValueAxis
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

STATUS_COLORS = {"PASS": "#2e9e55", "REVIEW": "#d9930d", "FAIL": "#d64545"}


# ==========================================================================
# Data layer
# ==========================================================================
def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_")


def list_dirs(path: Path) -> list[str]:
    return sorted(p.name for p in path.iterdir() if p.is_dir()) if path.exists() else []


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
        rows.append({
            "id": meta.get("batch_id", bid), "time": ts,
            "status": str(meta.get("status", "")).upper(),
            "items": n, "rate": round(100 * ok / n, 1) if n and ok is not None else None,
            "material": meta.get("material", ""), "conf": meta.get("material_confidence"),
            "density": meta.get("density_g_cm3"), "vib": meta.get("vibration_peak_hz"),
            "dims": meta.get("dimensions") or {},
            "pdf": bdir / "report.pdf" if (bdir / "report.pdf").exists() else None,
            "dir": bdir, "meta": meta,
        })
    rows.sort(key=lambda r: r["time"] or datetime.min, reverse=True)
    return rows


class BatchModel(QAbstractTableModel):
    BASE = ["Batch ID", "Time", "Status", "Items", "Pass rate %", "Material", "Mat. conf.",
            "Density g/cm³", "Peak vib. Hz"]
    TEXT_COLS = {0, 2, 5}

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
            return [row["id"], row["time"], row["status"], row["items"], row["rate"], row["material"],
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
        if role == Qt.ForegroundRole and col == 2 and v in STATUS_COLORS:
            return QColor(STATUS_COLORS[v])
        if role == Qt.FontRole and col == 2:
            f = QFont()
            f.setBold(True)
            return f
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
# Demo data (only when you choose File > Load demo data)
# ==========================================================================
DEMO_CATALOG = {
    "Bolts": {
        "BLT-M8-40": dict(material="Steel 304", density=7.93, dims={"length_mm": 40.0, "diameter_mm": 8.0}),
        "BLT-M10-60": dict(material="Steel 316", density=8.00, dims={"length_mm": 60.0, "diameter_mm": 10.0}),
    },
    "Nuts": {"NUT-M8-HEX": dict(material="Brass", density=8.50, dims={"width_mm": 13.0, "height_mm": 6.5})},
    "Chips": {"CHP-QFN-32": dict(material="Silicon/Epoxy", density=2.30, dims={"length_mm": 5.0, "width_mm": 5.0})},
}


def write_report_pdf(path: Path, ctype: str, part: str, meta: dict) -> None:
    """Placeholder report. Your real pipeline writes its own report.pdf."""
    rows = [("Component type", ctype), ("Part number", part), ("Batch", meta["batch_id"]),
            ("Timestamp", meta["timestamp"]), ("Items scanned", meta["item_count"]),
            ("Items passed", meta["pass_count"]), ("Material", meta["material"]),
            ("Density (g/cm³)", meta["density_g_cm3"]), ("Peak vibration (Hz)", meta["vibration_peak_hz"]),
            *meta["dimensions"].items(), ("Status", meta["status"])]
    body = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows)
    html = (f"<h1>Batch Report: {meta['batch_id']}</h1>"
            "<p>Demo report generated by the UI. Replace with your pipeline's output.</p>"
            f"<table border='1' cellpadding='6' cellspacing='0' width='100%'>{body}</table>")
    writer = QPdfWriter(str(path))
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.Letter))
    writer.setPageMargins(QMarginsF(15, 15, 15, 15), QPageLayout.Unit.Millimeter)
    doc = QTextDocument()
    doc.setHtml(html)
    doc.print_(writer)


def seed_demo(root: Path) -> None:
    rng, now = random.Random(42), datetime.now()
    for ctype, parts in DEMO_CATALOG.items():
        for part, spec in parts.items():
            pdir = root / ctype / part
            pdir.mkdir(parents=True, exist_ok=True)
            (pdir / "part.json").write_text(json.dumps({
                "description": f"{ctype[:-1]} {part}", "material": spec["material"],
                "density_g_cm3": spec["density"], "dimensions": spec["dims"], "tolerance_pct": 1.5}, indent=2))
            for i in range(10):
                ts = now - timedelta(days=9 - i, hours=rng.randint(0, 5))
                bid = f"BATCH-{ts:%Y%m%d}-{i + 1:04d}"
                n = rng.choice([200, 250, 300])
                ok = n - rng.randint(0, 11)
                meta = {
                    "batch_id": bid, "timestamp": ts.isoformat(timespec="seconds"), "item_count": n,
                    "pass_count": ok, "material": spec["material"],
                    "material_confidence": round(rng.uniform(0.93, 0.995), 3),
                    "dimensions": {k: round(v * rng.uniform(0.992, 1.008), 3) for k, v in spec["dims"].items()},
                    "density_g_cm3": round(spec["density"] * rng.uniform(0.985, 1.015), 3),
                    "vibration_peak_hz": round(rng.uniform(1400, 1650), 1),
                    "status": "PASS" if ok / n > 0.985 else ("REVIEW" if ok / n > 0.96 else "FAIL"),
                }
                bdir = pdir / bid
                bdir.mkdir(exist_ok=True)
                (bdir / "batch.json").write_text(json.dumps(meta, indent=2))
                write_report_pdf(bdir / "report.pdf", ctype, part, meta)


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
                         self.settings.value("root", "inspection_data")).resolve()
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
                                ("Load demo data", self.load_demo, None),
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
        self.m_batches, self.m_items = MetricCard("Batches"), MetricCard("Items scanned")
        self.m_rate, self.m_last = MetricCard("Avg pass rate"), MetricCard("Latest batch")
        for c in (self.m_batches, self.m_items, self.m_rate, self.m_last):
            cards.addWidget(c)
        rl.addLayout(cards)

        filt = QHBoxLayout()
        self.status_box = QComboBox()
        self.status_box.addItems(["All", "PASS", "REVIEW", "FAIL"])
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
        self.tabs.addTab(self._build_report_tab(), "Report")
        self.tabs.addTab(self._build_details_tab(), "Measurements")
        self.tabs.addTab(self._build_trends_tab(), "Trends")
        self.tabs.addTab(self._build_spec_tab(), "Part spec")

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
        self.pdf_msg = QLabel("Select a batch to view its report.")
        self.pdf_msg.setAlignment(Qt.AlignCenter)
        self.pdf_msg.setStyleSheet("color: gray;")
        self.pdf_stack = QStackedWidget()
        self.pdf_stack.addWidget(self.pdf_msg)
        self.pdf_stack.addWidget(self.pdf_view)
        lay.addWidget(self.pdf_stack, 1)
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
            self.subtitle.setText("Create a component type with “+ Type”, or use File > Load demo data.")
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

    def load_demo(self):
        seed_demo(self.root)
        self.refresh_all()
        self.statusBar().showMessage("Demo data loaded", 4000)

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
        rates = [r["rate"] for r in rows if r["rate"] is not None]
        self.m_rate.set(f"{sum(rates) / len(rates):.1f}%" if rates else "—")
        times = [r["time"] for r in rows if r["time"]]
        self.m_last.set(max(times).strftime("%b %d, %H:%M") if times else "—")

    def apply_filters(self):
        self.proxy.set_filters(self.status_box.currentText(), self.search.text())

    def on_batch_changed(self, current, _prev):
        if not current.isValid():
            self.show_batch(None)
            return
        self.show_batch(self.model.rows[self.proxy.mapToSource(current).row()])

    def show_batch(self, batch: dict | None):
        self.current = batch
        has_pdf = bool(batch and batch["pdf"])
        self.btn_open.setEnabled(has_pdf)
        self.btn_save.setEnabled(has_pdf)
        if has_pdf:
            self.pdf_doc.load(str(batch["pdf"]))
            self.pdf_stack.setCurrentWidget(self.pdf_view)
        else:
            self.pdf_doc.close()
            self.pdf_msg.setText("No report.pdf found for this batch." if batch
                                 else "Select a batch to view its report.")
            self.pdf_stack.setCurrentWidget(self.pdf_msg)
        self.show_details(batch)

    def show_details(self, b: dict | None):
        if not b:
            self.details.setHtml("<p style='color:gray'>No batch selected.</p>")
            return
        col = STATUS_COLORS.get(b["status"], "gray")
        dims = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in b["dims"].items())
        conf = f"{b['conf']:.1%}" if isinstance(b["conf"], (int, float)) else "—"
        self.details.setHtml(f"""
            <h2>{b['id']} <span style='color:{col}'>{b['status'] or ''}</span></h2>
            <p>{b['time'].strftime('%Y-%m-%d %H:%M:%S') if b['time'] else ''} &nbsp;|&nbsp;
               {b['items'] or '—'} items, {b['rate'] if b['rate'] is not None else '—'}% passed</p>
            <table cellpadding='4'>
              <tr><td colspan='2'><u>Hyperspectral</u></td></tr>
              <tr><td>Material</td><td><b>{b['material'] or '—'}</b> (confidence {conf})</td></tr>
              <tr><td colspan='2'><u>Machine vision</u></td></tr>{dims or "<tr><td>—</td></tr>"}
              <tr><td colspan='2'><u>Vibration / density</u></td></tr>
              <tr><td>Density</td><td><b>{b['density'] if b['density'] is not None else '—'}</b> g/cm³</td></tr>
              <tr><td>Peak vibration</td><td><b>{b['vib'] if b['vib'] is not None else '—'}</b> Hz</td></tr>
            </table>""")

    def open_external(self):
        if self.current and self.current["pdf"]:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current["pdf"])))

    def save_copy(self):
        if not (self.current and self.current["pdf"]):
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Save report", f"{self.current['id']}.pdf", "PDF (*.pdf)")
        if dest:
            Path(dest).write_bytes(Path(self.current["pdf"]).read_bytes())
            self.statusBar().showMessage(f"Saved {dest}", 4000)

    # ---- trends ------------------------------------------------------------
    def populate_metric_box(self):
        cols = self.model.columns()
        self.metric_box.blockSignals(True)
        self.metric_box.clear()
        for i, name in enumerate(cols):
            if i not in (0, 1, 2, 5):
                self.metric_box.addItem(name, i)
        self.metric_box.setCurrentIndex(max(0, self.metric_box.findText("Pass rate %")))
        self.metric_box.blockSignals(False)
        self.update_chart()

    def update_chart(self):
        chart = QChart()
        chart.legend().hide()
        col = self.metric_box.currentData()
        pts = []
        if col is not None:
            pts = sorted(((r["time"], self.model.value(r, col)) for r in self.model.rows
                          if r["time"] and self.model.value(r, col) is not None), key=lambda p: p[0])
        if pts:
            line, dots = QLineSeries(), QScatterSeries()
            dots.setMarkerSize(8)
            for t, v in pts:
                ms = t.timestamp() * 1000
                line.append(ms, float(v))
                dots.append(ms, float(v))
            chart.addSeries(line)
            chart.addSeries(dots)
            ax = QDateTimeAxis()
            ax.setFormat("MMM d")
            ay = QValueAxis()
            lo, hi = min(float(v) for _, v in pts), max(float(v) for _, v in pts)
            pad = (hi - lo) * 0.15 or abs(hi) * 0.01 or 1
            ay.setRange(lo - pad, hi + pad)
            chart.addAxis(ax, Qt.AlignBottom)
            chart.addAxis(ay, Qt.AlignLeft)
            for s in (line, dots):
                s.attachAxis(ax)
                s.attachAxis(ay)
        chart.setTitle(self.metric_box.currentText() + " per batch")
        self.chart_view.setChart(chart)

    # ---- part spec ---------------------------------------------------------
    def populate_spec(self, spec: dict):
        self.sp_desc.setText(spec.get("description", ""))
        self.sp_mat.setText(spec.get("material", ""))
        self.sp_density.setValue(float(spec.get("density_g_cm3", 0) or 0))
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
            "density_g_cm3": self.sp_density.value(), "dimensions": dims,
            "tolerance_pct": self.sp_tol.value()}, indent=2))
        self.subtitle.setText(self.sp_desc.text())
        self.statusBar().showMessage("Spec saved", 4000)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()