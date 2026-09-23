"""Look of the whole app: colour tokens (same names as the design file design/ui-redesign.pen), the global style
sheet, and small line icons drawn from inline SVG (Qt ships no icon set)."""
from __future__ import annotations

import os
import tempfile

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer

BG = "#0E1116"
SURFACE = "#151920"
SURFACE_2 = "#1C212A"
BORDER = "#262C37"
BORDER_HOVER = "#343B49"
TEXT = "#E8ECF2"
TEXT_2 = "#9AA4B2"
TEXT_3 = "#6B7482"
ACCENT = "#4FA3FF"
ACCENT_SOFT = "rgba(79,163,255,0.14)"
ON_ACCENT = "#0B1220"
LIVE = "#3DD68C"
WARN = "#FFB454"
DANGER = "#FF6B6B"

UI_FONTS = ("Microsoft YaHei UI", "PingFang SC", "Noto Sans SC", "Segoe UI")
MONO_FONTS = ("Cascadia Mono", "JetBrains Mono", "SF Mono", "Menlo", "Consolas")

# 24×24 line icons (stroke = the requested colour). Paths are hand-drawn for this app.
_ICONS = {
    "play": '<path d="M7 5l12 7-12 7z" fill="C" stroke="none"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="2" fill="C" stroke="none"/>',
    "copy": '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/>',
    "download": '<path d="M12 3v12M7 10l5 5 5-5M5 21h14"/>',
    "trash": '<path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14M10 11v6M14 11v6"/>',
    "sliders": '<path d="M4 6h9M17 6h3M15 4v4M4 12h3M11 12h9M9 10v4M4 18h11M19 18h1M17 16v4"/>',
    "chevron-down": '<path d="M6 9l6 6 6-6"/>',
    "chevron-up": '<path d="M6 15l6-6 6 6"/>',
    "refresh": '<path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/>',
    "lock": '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    "overlay": '<rect x="3" y="5" width="18" height="14" rx="2"/><rect x="11" y="12" width="7" height="4" rx="1" fill="C"/>',
    "pointer-off": '<path d="M6 4l5 14 2-5.5L18.5 10z"/><path d="M3 21L21 3"/>',
    "learn": '<path d="M2 9l10-5 10 5-10 5zM6 11v5c3 2 9 2 12 0v-5M22 9v6"/>',
    "audio": '<path d="M3 12h1M7 8v8M11 5v14M15 9v6M19 11v2"/>',
    "languages": '<path d="M4 5h9M8.5 3v2M5.5 5c1 4 3.5 7 7 8.5M11.5 5c-1 4-3.5 7-7 8.5M13 21l4-9 4 9M14.3 18h5.4"/>',
    "mic": '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
    "sparkles": '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 17v4M17 19h4"/>',
    "gauge": '<path d="M12 14l4-4M3.3 18a9 9 0 1 1 17.4 0"/>',
    "monitor": '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>',
    "app": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18"/>',
    "volume": '<path d="M4 9h4l5-4v14l-5-4H4zM17 9a4 4 0 0 1 0 6"/>',
    "drive": '<path d="M3 13h18v6H3zM5 13l2-8h10l2 8M7 16h.01"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "reset": '<path d="M4 12a8 8 0 1 0 2.3-5.7M4 4v5h5"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    "check": '<path d="M5 12l5 5 9-10"/>',
    "alert": '<path d="M12 3l10 18H2zM12 10v4M12 17h.01"/>',
    "x": '<path d="M6 6l12 12M18 6L6 18"/>',
    "eye": '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "text-size": '<path d="M3 19l5-13 5 13M4.5 15h7M15 19l3-8 3 8M15.8 17h4.4"/>',
    "users": '<circle cx="9" cy="8" r="4"/><path d="M2 21c0-4 3-6 7-6s7 2 7 6M16 4a4 4 0 0 1 0 8M22 21c0-3-2-5-4-5.5"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "timer": '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l2 2M9 2h6"/>',
    "activity": '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    "palette": '<circle cx="12" cy="12" r="9"/><circle cx="8" cy="10" r="1" fill="C"/><circle cx="12" cy="7.5" r="1" fill="C"/><circle cx="16" cy="10" r="1" fill="C"/><path d="M12 21a2.5 2.5 0 0 1 0-5h2"/>',
    "scissors": '<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M8.1 8.1L20 20M8.1 15.9L20 4"/>',
    "shield": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
    "cloud": '<path d="M7 18a5 5 0 0 1-.5-10A6 6 0 0 1 18 9a4.5 4.5 0 0 1-.5 9zM12 11v6M9.5 14.5L12 17l2.5-2.5"/>',
    "file": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9zM14 3v6h6M8 13h8M8 17h5"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5zM3 13l9 5 9-5"/>',
    "restart": '<path d="M20 12a8 8 0 1 1-2.3-5.7M20 4v5h-5"/>',
    "minus": '<path d="M5 12h14"/>',
    "maximize": '<rect x="4" y="4" width="16" height="16" rx="1.5"/>',
    "restore": '<rect x="4" y="8" width="12" height="12" rx="1.5"/><path d="M8 8V5.5A1.5 1.5 0 0 1 9.5 4h9A1.5 1.5 0 0 1 20 5.5v9a1.5 1.5 0 0 1-1.5 1.5H16"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
}

_pixmaps: dict[tuple, QPixmap] = {}


def _svg(name: str, color: str, stroke: float) -> bytes:
    body = _ICONS[name].replace('"C"', f'"{color}"')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{body}</svg>').encode()


def pixmap(name: str, color: str = TEXT_2, size: int = 16, stroke: float = 2.0, dpr: float = 2.0) -> QPixmap:
    key = (name, color, size, stroke, dpr)
    pm = _pixmaps.get(key)
    if pm is None:
        pm = QPixmap(int(size * dpr), int(size * dpr))
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        QSvgRenderer(QByteArray(_svg(name, color, stroke))).render(p, QRectF(0, 0, pm.width(), pm.height()))
        p.end()
        pm.setDevicePixelRatio(dpr)
        _pixmaps[key] = pm
    return pm


APP_ICON = os.path.join(os.path.dirname(__file__), "assets", "icon.png")


def app_icon() -> QIcon:
    """The application logo (window / taskbar icon)."""
    return QIcon(APP_ICON)


def icon(name: str, color: str = TEXT_2, checked_color: str | None = None, size: int = 16,
         disabled_color: str = TEXT_3) -> QIcon:
    """An icon whose colour can differ when the button is checked or disabled."""
    ic = QIcon()
    ic.addPixmap(pixmap(name, color, size), QIcon.Mode.Normal, QIcon.State.Off)
    ic.addPixmap(pixmap(name, checked_color or color, size), QIcon.Mode.Normal, QIcon.State.On)
    ic.addPixmap(pixmap(name, disabled_color, size), QIcon.Mode.Disabled, QIcon.State.Off)
    ic.addPixmap(pixmap(name, disabled_color, size), QIcon.Mode.Disabled, QIcon.State.On)
    return ic


def _first_family(candidates: tuple[str, ...]) -> str:
    have = set(QFontDatabase.families())
    return next((f for f in candidates if f in have), "")


def ui_font_family() -> str:
    return _first_family(UI_FONTS)


def mono_font(px: int = 12, weight: int = 400) -> QFont:
    f = QFont()
    fam = _first_family(MONO_FONTS)
    if fam:
        f.setFamily(fam)
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPixelSize(px)
    f.setWeight(QFont.Weight(weight))
    return f


def _asset_dir() -> str:
    """Style sheets can only load images from files, so the few they need are written once to a temp folder."""
    d = os.path.join(tempfile.gettempdir(), "live_translator_theme")
    os.makedirs(d, exist_ok=True)
    for name, color in (("chevron-down", TEXT_3), ("chevron-up", TEXT_3), ("check", ON_ACCENT)):
        path = os.path.join(d, f"{name}-{color[1:]}.png")
        if not os.path.exists(path):
            pixmap(name, color, 12, 2.4, 2.0).save(path)
    return d.replace("\\", "/")


def style_sheet() -> str:
    a = _asset_dir()
    down, up, check = (f"{a}/chevron-down-{TEXT_3[1:]}.png", f"{a}/chevron-up-{TEXT_3[1:]}.png",
                       f"{a}/check-{ON_ACCENT[1:]}.png")
    return f"""
QWidget {{ color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px 8px; border-radius: 6px; }}

QMenuBar {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; padding: 2px 6px; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; border-radius: 5px; color: {TEXT_2}; }}
QMenuBar::item:selected {{ background: {SURFACE_2}; color: {TEXT}; }}
QFrame#titleBar {{ background: {BG}; border: none; border-bottom: 1px solid {BORDER}; }}
QLabel#winTitle {{ font-size: 12px; font-weight: 600; }}
QMenuBar#titleMenu {{ background: transparent; border: none; padding: 0; }}
QMenuBar#titleMenu::item {{ padding: 4px 9px; font-size: 12px; }}
QMenu {{ background: {SURFACE_2}; border: 1px solid {BORDER}; padding: 5px; border-radius: 8px; }}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 5px; }}
QMenu::item:selected {{ background: {ACCENT_SOFT}; }}
QMenu::item:disabled {{ color: {TEXT_3}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 8px; }}
QMenu::indicator {{ width: 14px; height: 14px; margin-left: 4px; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
    background: {BG}; border: 1px solid {BORDER}; border-radius: 7px; padding: 6px 10px;
    selection-background-color: {ACCENT}; selection-color: {ON_ACCENT}; min-height: 20px; }}
QPlainTextEdit {{ padding: 4px 6px; }}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QPlainTextEdit:hover {{ border-color: {BORDER_HOVER}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {{ border-color: {ACCENT}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QPlainTextEdit:disabled {{ color: {TEXT_3}; }}
QComboBox {{ padding-right: 26px; }}
QComboBox::drop-down {{ border: none; width: 26px; subcontrol-origin: padding; subcontrol-position: center right; }}
QComboBox::down-arrow {{ image: url({down}); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: {SURFACE_2}; border: 1px solid {BORDER}; outline: 0; padding: 4px;
    selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT}; }}
QSpinBox, QDoubleSpinBox {{ padding-right: 22px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 20px; border: none; background: transparent; margin-top: 2px; margin-right: 2px; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 20px; border: none; background: transparent; margin-bottom: 2px; margin-right: 2px; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({up}); width: 10px; height: 10px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({down}); width: 10px; height: 10px; }}

QPushButton {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 7px; padding: 7px 14px; }}
QPushButton:hover {{ border-color: {BORDER_HOVER}; background: #222835; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {TEXT_3}; }}
QPushButton#primary {{ background: {ACCENT}; color: {ON_ACCENT}; border: 1px solid {ACCENT}; font-weight: 600; }}
QPushButton#primary:hover {{ background: #6BB2FF; }}
QPushButton#danger {{ color: {DANGER}; background: rgba(255,107,107,0.08); border-color: rgba(255,107,107,0.28); }}
QPushButton#danger:disabled {{ color: {TEXT_3}; background: {SURFACE_2}; border-color: {BORDER}; }}
QPushButton#link {{ background: transparent; border: none; color: {TEXT_2}; padding: 4px 2px; }}
QPushButton#link:hover {{ color: {TEXT}; }}
QPushButton#startBtn {{ background: {ACCENT}; color: {ON_ACCENT}; border: 1px solid {ACCENT}; font-weight: 600; padding: 7px 16px; border-radius: 8px; }}
QPushButton#startBtn:hover {{ background: #6BB2FF; }}
QPushButton#startBtn[running="true"] {{ background: rgba(255,107,107,0.15); color: {DANGER}; border-color: rgba(255,107,107,0.35); }}
QPushButton#startBtn:disabled {{ background: {SURFACE_2}; color: {TEXT_3}; border-color: {BORDER}; }}
QPushButton#warnBtn {{ background: rgba(255,180,84,0.12); color: {WARN}; border: none; font-weight: 600; padding: 5px 12px; }}

QToolButton#iconBtn {{ background: transparent; border: none; border-radius: 7px; padding: 7px; }}
QToolButton#iconBtn:hover {{ background: {SURFACE_2}; }}
QToolButton#iconBtn:disabled {{ background: transparent; }}
QToolButton#fieldBtn {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 7px; padding: 8px; }}
QToolButton#fieldBtn:hover {{ border-color: {BORDER_HOVER}; }}
QFrame#pillGroup, QFrame#seg {{ background: {BG}; border-radius: 8px; }}
QToolButton#pill, QToolButton#segBtn {{ background: transparent; border: none; border-radius: 6px; padding: 5px 10px; color: {TEXT_3}; font-size: 12px; }}
QToolButton#pill:hover, QToolButton#segBtn:hover {{ color: {TEXT_2}; }}
QToolButton#pill:checked, QToolButton#segBtn:checked {{ background: {SURFACE_2}; color: {TEXT}; }}
QToolButton#segBtn:disabled {{ color: #3E4552; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border-radius: 4px; border: 1.5px solid {TEXT_3}; background: transparent; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url({check}); }}

QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {BORDER_HOVER}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 3px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #FFFFFF; border: 2px solid {ACCENT}; width: 10px; height: 10px; margin: -5px 0; border-radius: 7px; }}

QTreeWidget {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 8px; outline: 0; alternate-background-color: #11151B; }}
QTreeWidget::item {{ padding: 6px 4px; border: none; }}
QTreeWidget::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
QHeaderView::section {{ background: {SURFACE}; color: {TEXT_3}; border: none; border-bottom: 1px solid {BORDER}; padding: 6px 8px; font-size: 11px; }}

QStatusBar {{ background: {SURFACE}; border-top: 1px solid {BORDER}; color: {TEXT_3}; font-size: 11px; min-height: 26px; }}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ color: {TEXT_3}; font-size: 11px; padding: 0 6px; }}
QSplitter::handle {{ background: {BORDER}; }}

QLabel#hint, QLabel:disabled {{ color: {TEXT_3}; font-size: 12px; }}
QLabel#fieldLabel {{ color: {TEXT_3}; font-size: 11px; }}
QLabel#section {{ color: {TEXT_3}; font-size: 11px; font-weight: 600; }}
QLabel#muted {{ color: {TEXT_3}; font-size: 12px; }}
QLabel#pageTitle {{ font-size: 18px; font-weight: 600; }}
QLabel#ok {{ color: {LIVE}; }}
QLabel#bad {{ color: {DANGER}; }}

QFrame#topBar {{ background: {SURFACE}; border: none; border-bottom: 1px solid {BORDER}; }}
QFrame#sidebar {{ background: {SURFACE}; }}
QFrame#vline {{ background: {BORDER}; }}
QFrame#card {{ background: transparent; border: 1px solid transparent; border-radius: 10px; }}
QFrame#card[open="true"] {{ background: {SURFACE_2}; border-color: {BORDER}; }}
QFrame#row {{ background: transparent; border: 1px solid transparent; border-radius: 10px; }}
QFrame#row[current="true"] {{ background: {SURFACE}; border-color: {BORDER}; }}
QFrame#trHeader {{ border: none; border-bottom: 1px solid {BORDER}; }}
QFrame#footNote {{ background: {BG}; border-radius: 8px; }}
QFrame#warnBanner {{ background: rgba(255,180,84,0.10); border: 1px solid rgba(255,180,84,0.25); border-radius: 8px; }}
QFrame#warnBanner QLabel {{ color: {WARN}; font-size: 12px; }}
QFrame#infoBanner {{ background: {ACCENT_SOFT}; border-radius: 8px; }}
QFrame#infoBanner QLabel {{ color: {TEXT_2}; font-size: 12px; }}
QFrame#box {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#navPane {{ background: {BG}; border: none; border-right: 1px solid {BORDER}; }}
QFrame#dlgHeader {{ border: none; border-bottom: 1px solid {BORDER}; }}
QFrame#dlgFooter {{ border: none; border-top: 1px solid {BORDER}; }}
QFrame#settingRow {{ border: none; border-bottom: 1px solid {BORDER}; }}
"""


def apply(app) -> None:
    """Dark theme for the whole application (idempotent)."""
    if app is None or app.property("lt_themed"):
        return
    app.setStyle("Fusion")
    f = app.font()                               # a font-size in the style sheet would override every setFont()
    fam = ui_font_family()
    if fam:
        f.setFamily(fam)
    f.setPixelSize(13)
    app.setFont(f)
    pal = QPalette()
    for role, c in ((QPalette.ColorRole.Window, BG), (QPalette.ColorRole.Base, BG),
                    (QPalette.ColorRole.AlternateBase, SURFACE), (QPalette.ColorRole.Text, TEXT),
                    (QPalette.ColorRole.WindowText, TEXT), (QPalette.ColorRole.Button, SURFACE_2),
                    (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.Highlight, ACCENT),
                    (QPalette.ColorRole.HighlightedText, ON_ACCENT), (QPalette.ColorRole.ToolTipBase, SURFACE_2),
                    (QPalette.ColorRole.ToolTipText, TEXT), (QPalette.ColorRole.PlaceholderText, TEXT_3),
                    (QPalette.ColorRole.Link, ACCENT)):
        pal.setColor(role, QColor(c))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(TEXT_3))
    app.setPalette(pal)
    app.setStyleSheet(style_sheet())
    app.setProperty("lt_themed", True)


def repolish(w) -> None:
    """Re-apply the style sheet after changing a dynamic property used in a selector."""
    w.style().unpolish(w)
    w.style().polish(w)
    w.update()


ICON_SIZE = QSize(16, 16)
