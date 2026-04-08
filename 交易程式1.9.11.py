import os
import sys

# ==================== Python 版本檢查 ====================
if sys.version_info[:2] != (3, 10):
    _ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    _msg = f"本程式需要 Python 3.10 執行，目前版本為 {_ver}\n請安裝 Python 3.10 後重新執行。"
    if os.name == 'nt':
        import ctypes as _ct
        _ct.windll.user32.MessageBoxW(0, _msg, "Python 版本不符", 0x10)
    print(_msg)
    sys.exit(1)

# exe 打包模式：確保工作目錄在 exe 所在資料夾，並優先從 exe 目錄載入模組
if getattr(sys, 'frozen', False):
    _exe_dir = os.path.dirname(sys.executable)
    os.chdir(_exe_dir)
    sys.path.insert(0, _exe_dir)

os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--enable-webgl --ignore-gpu-blocklist"
import subprocess
import ctypes
import importlib

APP_VERSION = "1.9.11"
GITHUB_REPO = "OswallowO/Remora"

REQUIRED = [
    ("pandas",           "pandas"),
    ("numpy",            "numpy"),
    ("plotly",           "plotly"),
    ("colorama",         "colorama"),
    ("tabulate",         "tabulate"),
    ("openpyxl",         "openpyxl"),
    ("dateutil",         "python-dateutil"),
    ("matplotlib",       "matplotlib"),
    ("PyQt5",            "PyQt5"),
    ("PyQt5.QtWebEngineWidgets", "PyQtWebEngine"),
    ("scipy",            "scipy"),
    ("shioaji",          "shioaji"),
    ("touchprice",       "touchprice"),
    ("requests",         "requests"),
    ("bs4",              "beautifulsoup4"),
    ("lxml",             "lxml"),
    ("optuna",           "optuna"),
    ("packaging",        "packaging"),
]

def ensure_packages(pkgs):
    missing = []
    for mod, pkg in pkgs:
        try: 
            importlib.import_module(mod)
        except ImportError: 
            missing.append(pkg)
            
    if missing:
        msg = f"系統偵測到缺少必要套件：\n{', '.join(missing)}\n\n按下「確定」後將開始自動安裝，安裝期間會顯示黑畫面，請耐心稍候..."
        # 僅在 Windows 環境下彈出提示框
        if os.name == 'nt':
            ctypes.windll.user32.MessageBoxW(0, msg, "環境建置提示", 0x40)
            
        print("首次執行，正在安裝必要套件：", ", ".join(missing))
        
        # 升級 pip 並安裝缺失套件
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        for pkg in missing:
            print(f"正在安裝 {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            
        if os.name == 'nt':
            ctypes.windll.user32.MessageBoxW(0, "✅ 已成功安裝所有必要套件！\n為了確保系統穩定，程式即將自動重啟。", "安裝完成", 0x40)
        
        # 安裝完畢後自動重啟程式載入新套件
        os.execl(sys.executable, sys.executable, *sys.argv)

if not getattr(sys, 'frozen', False):
    ensure_packages(REQUIRED)

# 確保環境建置完畢後，再一次性、乾淨地載入所有套件
import json
import math
import io
import time as time_module
import warnings
import traceback
import csv
import threading
import re
import glob
import sqlite3
import html
import random
import logging
import logging.handlers
import winsound
import scipy.stats as st
from datetime import datetime, time, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from configparser import ConfigParser
from collections import deque

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==================== 結構化日誌系統 ====================
logger = logging.getLogger('remora')
logger.setLevel(logging.INFO)
_log_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
_file_handler = logging.handlers.RotatingFileHandler(
    'remora.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_log_fmt)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.WARNING)
_console_handler.setFormatter(_log_fmt)
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)
# ========================================================

def _play_alert(pattern: str):
    """背景執行緒播放提示音。pattern: 'entry' / 'stop_loss' / 'exit'"""
    def _beep():
        try:
            if pattern == 'entry':
                winsound.MessageBeep(winsound.MB_ICONASTERISK)    # 系統提示音
            elif pattern == 'stop_loss':
                winsound.MessageBeep(winsound.MB_ICONHAND)        # 系統錯誤音
            elif pattern == 'exit':
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)  # 系統警告音
        except Exception as e:
            logger.warning(f"音效播放失敗: {e}")
    if not getattr(sys_config, 'sound_enabled', True):
        return
    import threading as _th
    _th.Thread(target=_beep, daemon=True).start()

import contextlib
import io as _io
import pandas as pd
import numpy as np
import requests
import shioaji as sj
import touchprice as tp
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from bs4 import BeautifulSoup
from colorama import init, Fore, Style

def auto_install_esun_sdk():
    whl_filename = "esun_marketdata-2.2.0-cp37-abi3-win_amd64.whl" 
    try: importlib.import_module("esun_marketdata")
    except ImportError:
        whl_path = os.path.join(os.getcwd(), whl_filename)
        if os.path.exists(whl_path):
            try: subprocess.check_call([sys.executable, "-m", "pip", "install", whl_path])
            except Exception as e: sys.exit(f"❌ SDK 安裝失敗：{e}")
        else:
            sys.exit(f"❌ 找不到安裝檔：{whl_path}\n請確認 .whl 檔案是否放在同一個資料夾內。")

auto_install_esun_sdk()

import esun_marketdata
from esun_marketdata import EsunMarketdata
import shioaji_logic

# ==================== 自動檢查更新 ====================
def _close_msgbox(title):
    """透過視窗標題找到 MessageBox 並關閉"""
    hwnd = ctypes.windll.user32.FindWindowW(None, title)
    if hwnd:
        ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE

def _auto_download_and_install(installer_url, installer_name, fallback_url):
    """下載安裝檔到暫存目錄，完成後啟動安裝並關閉程式"""
    import tempfile
    temp_path = os.path.join(tempfile.gettempdir(), installer_name)
    _TITLE = "Remora — 正在下載更新"

    # 在另一個 thread 顯示「下載中」提示（主 thread 做實際下載）
    def _show_wait():
        ctypes.windll.user32.MessageBoxW(
            0,
            f"正在下載 {installer_name}…\n\n請稍候，下載完成後將自動開始安裝。\n（此視窗會自動關閉）",
            _TITLE, 0x00000040 | 0x00040000)  # MB_ICONINFO | MB_SETFOREGROUND

    wait_thread = threading.Thread(target=_show_wait, daemon=True)
    wait_thread.start()
    import time; time.sleep(0.3)  # 等待視窗出現

    try:
        r = requests.get(installer_url, stream=True, timeout=300)
        r.raise_for_status()
        with open(temp_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        # 下載成功 → 關閉等待視窗 → 啟動安裝檔 → 結束程式
        _close_msgbox(_TITLE)
        subprocess.Popen([temp_path])
        sys.exit(0)
    except Exception as e:
        _close_msgbox(_TITLE)
        result = ctypes.windll.user32.MessageBoxW(
            0, f"自動下載失敗：{e}\n\n按「確定」前往手動下載頁面。",
            "更新失敗", 0x00000010 | 0x00000001)  # MB_ICONERROR | MB_OKCANCEL
        if result == 1:
            os.startfile(fallback_url)

def check_for_update(silent=False):
    """檢查 GitHub Releases 是否有新版本，有則自動下載安裝"""
    if not GITHUB_REPO:
        return
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
            timeout=5, headers={'Accept': 'application/vnd.github.v3+json'}
        )
        if resp.status_code != 200:
            return
        data = resp.json()
        latest = data.get('tag_name', '').lstrip('vV')
        if not latest:
            return
        # 比較版本號
        from packaging.version import Version
        try:
            if Version(latest) <= Version(APP_VERSION):
                if not silent:
                    ctypes.windll.user32.MessageBoxW(0, f"目前已是最新版本 v{APP_VERSION}", "檢查更新", 0x40)
                return
        except Exception:
            if latest == APP_VERSION:
                return
        # 有新版 — 嘗試找到安裝檔 asset
        dl_url = data.get('html_url', f"https://github.com/{GITHUB_REPO}/releases/latest")
        assets = data.get('assets', [])
        installer = next((a for a in assets if a['name'].endswith('.exe') and 'Setup' in a['name']), None)

        if installer:
            msg = f"發現新版本 v{latest}（目前 v{APP_VERSION}）\n\n按「確定」自動下載並安裝更新。"
        else:
            msg = f"發現新版本 v{latest}（目前 v{APP_VERSION}）\n\n按「確定」開啟下載頁面。"

        if os.name == 'nt':
            result = ctypes.windll.user32.MessageBoxW(0, msg, "有新版本可用", 0x41)  # OK/Cancel
            if result == 1:  # OK
                if installer:
                    _auto_download_and_install(
                        installer['browser_download_url'],
                        installer['name'],
                        dl_url)
                else:
                    os.startfile(dl_url)
    except Exception:
        pass

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QTextEdit, 
    QInputDialog, QMessageBox, QDialog, QLineEdit, 
    QComboBox, QFormLayout, QScrollArea, QGridLayout,
    QFrame, QDialogButtonBox, QFileDialog,
    QGroupBox, QProgressBar, QSplitter, QListWidget,
    QAbstractItemView, QTableWidget, QTableWidgetItem, QHeaderView,
    QDoubleSpinBox, QMenu, QListView, QStyledItemDelegate,
    QCheckBox, QSpinBox, QStackedWidget, QTextBrowser,
    QTimeEdit, QTabWidget, QTabBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, pyqtSlot, QUrl, QTime, QTimer, QPoint
from PyQt5.QtGui import QFont, QColor, QTextCursor, QPalette, QLinearGradient, QGradient, QPainter, QBrush, QPen, QPixmap, QIcon
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
from PyQt5.QtWebChannel import QWebChannel

plt.rcParams['axes.unicode_minus'] = False
init(autoreset=True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3.connection")
np.seterr(divide='ignore', invalid='ignore')
pd.set_option('future.no_silent_downcasting', True)

RED = Fore.RED; GREEN = Fore.GREEN; YELLOW = Fore.YELLOW; BLUE = Fore.BLUE; RESET = Style.RESET_ALL
ESUN_LOGIN_PWD = None
ESUN_CERT_PWD = None

# ==================== Remora 專業深色主題系統 ====================
_TV_CLASSIC = {
    'bg':           '#131722',   # 主背景
    'panel':        '#1e222d',   # 側邊欄/頂欄面板
    'surface':      '#2a2e39',   # 卡片/元素底色
    'border':       '#2a2e39',   # 分隔線
    'border_light': '#363a45',   # 較亮分隔線
    'blue':         '#2962ff',   # 主要 Accent (TV 藍)
    'blue_hover':   '#1e53e5',
    'blue_dim':     '#1a3a8c',
    'green':        '#26a69a',   # 多頭/獲利
    'green_dim':    '#0d3531',
    'red':          '#ef5350',   # 空頭/虧損
    'red_dim':      '#3d1414',
    'yellow':       '#f7a600',   # 警告
    'orange':       '#ff9800',   # 次要 Accent
    'purple':       '#7c4dff',
    'text':         '#d1d4dc',   # 主要文字
    'text_dim':     '#787b86',   # 次要文字
    'text_bright':  '#ffffff',
    'console_bg':   '#0d1117',   # 終端機背景
    'console_text': '#a8b5c8',   # 終端機文字
}

# Matrix 終端風格主題（仿 WALLET HUNTER 掃描器）
_TV_MATRIX = {
    'bg':           '#000000',   # 純黑背景
    'panel':        '#080f08',   # 深綠黑面板
    'surface':      '#0d1a0d',   # 卡片底色（深綠）
    'border':       '#1a3a1a',   # 暗綠分隔線
    'border_light': '#2a5a2a',   # 亮綠分隔線
    'blue':         '#00e676',   # 主要 Accent（霓虹綠，替代藍）
    'blue_hover':   '#69f0ae',
    'blue_dim':     '#004d26',
    'green':        '#00ff41',   # 多頭/獲利（Matrix 亮綠）
    'green_dim':    '#003300',
    'red':          '#ff1744',   # 空頭/虧損
    'red_dim':      '#3d0000',
    'yellow':       '#ffff00',   # 警告（霓虹黃）
    'orange':       '#ff6d00',   # 次要 Accent
    'purple':       '#ea00ff',   # 紫色
    'text':         '#00cc44',   # 主要文字（中綠）
    'text_dim':     '#336b33',   # 次要文字（暗綠）
    'text_bright':  '#00ff41',   # 亮文字（Matrix 綠）
    'console_bg':   '#000000',   # 終端機背景
    'console_text': '#00ff41',   # 終端機文字
}

# 讀取 config.ini 中的主題設定（在 sys_config 載入前決定）
def _load_theme_early():
    try:
        _c = ConfigParser(); _c.read('config.ini', encoding='utf-8-sig')
        _t = _c.get('ui', 'theme', fallback='classic')
        return 'classic' if _t == 'tradingview' else _t
    except Exception:
        return 'classic'

_active_theme = _load_theme_early()
TV = _TV_MATRIX if _active_theme == 'matrix' else _TV_CLASSIC

# 專業風格 SVG 圖標（20x20, stroke-based）
TV_ICONS = {
    'play': '<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><polygon points="5,3 17,10 5,17" fill="{c}" stroke="none"/></svg>',
    'portfolio': '<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><rect x="2" y="7" width="5" height="11" rx="1" fill="none" stroke="{c}" stroke-width="1.5"/><rect x="7.5" y="3" width="5" height="15" rx="1" fill="none" stroke="{c}" stroke-width="1.5"/><rect x="13" y="5" width="5" height="13" rx="1" fill="none" stroke="{c}" stroke-width="1.5"/></svg>',
    'analytics': '<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><polyline points="2,15 6,9 10,12 14,5 18,8" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><circle cx="18" cy="8" r="2" fill="{c}"/></svg>',
    'crosshair': '<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><circle cx="10" cy="10" r="6" fill="none" stroke="{c}" stroke-width="1.5"/><line x1="10" y1="2" x2="10" y2="6" stroke="{c}" stroke-width="1.5" stroke-linecap="round"/><line x1="10" y1="14" x2="10" y2="18" stroke="{c}" stroke-width="1.5" stroke-linecap="round"/><line x1="2" y1="10" x2="6" y2="10" stroke="{c}" stroke-width="1.5" stroke-linecap="round"/><line x1="14" y1="10" x2="18" y2="10" stroke="{c}" stroke-width="1.5" stroke-linecap="round"/></svg>',
    'bar_chart': '<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><rect x="2" y="10" width="4" height="8" rx="1" fill="{c}"/><rect x="8" y="4" width="4" height="14" rx="1" fill="{c}"/><rect x="14" y="7" width="4" height="11" rx="1" fill="{c}"/></svg>',
    'grid': '<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><rect x="2" y="2" width="7" height="7" rx="1.5" fill="none" stroke="{c}" stroke-width="1.5"/><rect x="11" y="2" width="7" height="7" rx="1.5" fill="none" stroke="{c}" stroke-width="1.5"/><rect x="2" y="11" width="7" height="7" rx="1.5" fill="none" stroke="{c}" stroke-width="1.5"/><rect x="11" y="11" width="7" height="7" rx="1.5" fill="none" stroke="{c}" stroke-width="1.5"/></svg>',
    'key': '<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><circle cx="7" cy="8" r="4" fill="none" stroke="{c}" stroke-width="1.5"/><line x1="10.5" y1="10.5" x2="17" y2="17" stroke="{c}" stroke-width="1.5" stroke-linecap="round"/><line x1="14" y1="14" x2="17" y2="11" stroke="{c}" stroke-width="1.5" stroke-linecap="round"/></svg>',
    'download': '<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><line x1="10" y1="3" x2="10" y2="13" stroke="{c}" stroke-width="1.5" stroke-linecap="round"/><polyline points="6,10 10,14 14,10" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M3,15 L3,17 L17,17 L17,15" fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round"/></svg>',
    'list': '<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><line x1="3" y1="5" x2="17" y2="5" stroke="{c}" stroke-width="1.5" stroke-linecap="round"/><line x1="3" y1="10" x2="17" y2="10" stroke="{c}" stroke-width="1.5" stroke-linecap="round"/><line x1="3" y1="15" x2="17" y2="15" stroke="{c}" stroke-width="1.5" stroke-linecap="round"/></svg>',
    'settings': '<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><circle cx="10" cy="10" r="3" fill="none" stroke="{c}" stroke-width="1.5"/><path d="M10,2 L10,4 M10,16 L10,18 M2,10 L4,10 M16,10 L18,10 M4.3,4.3 L5.7,5.7 M14.3,14.3 L15.7,15.7 M15.7,4.3 L14.3,5.7 M5.7,14.3 L4.3,15.7" stroke="{c}" stroke-width="1.5" stroke-linecap="round"/></svg>',
    'warning': '<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path d="M10,2 L18,17 L2,17 Z" fill="none" stroke="{c}" stroke-width="1.5" stroke-linejoin="round"/><line x1="10" y1="8" x2="10" y2="12" stroke="{c}" stroke-width="1.5" stroke-linecap="round"/><circle cx="10" cy="14.5" r="0.8" fill="{c}"/></svg>',
}

def _svg_icon(name, size=20, color=None):
    """將 TV_ICONS 中的 SVG 轉為 QIcon"""
    from PyQt5.QtGui import QPixmap, QIcon
    if color is None: color = TV['text']
    svg_str = TV_ICONS.get(name, TV_ICONS['settings']).replace('{c}', color)
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    from PyQt5.QtSvg import QSvgRenderer
    from PyQt5.QtCore import QByteArray
    renderer = QSvgRenderer(QByteArray(svg_str.encode('utf-8')))
    from PyQt5.QtGui import QPainter
    painter = QPainter(pm)
    renderer.render(painter)
    painter.end()
    return QIcon(pm)

# 側邊欄按鈕通用樣式生成器
def _tv_nav_btn_style(accent='#2962ff', danger=False):
    if danger:
        return f"""
            QPushButton {{
                background-color: #1c1014;
                color: {TV['red']};
                border: 1px solid #2d1517;
                border-radius: 7px;
                text-align: left;
                font-size: 14px;
                font-weight: 600;
                padding: 0 10px;
            }}
            QPushButton:hover {{
                background-color: {TV['red_dim']};
                border-color: {TV['red']};
                color: #ff6b6b;
            }}
            QPushButton:pressed {{ background-color: {TV['red']}; color: white; }}
        """
    return f"""
        QPushButton {{
            background-color: transparent;
            color: {TV['text']};
            border: none;
            border-radius: 7px;
            text-align: left;
            font-size: 14px;
            padding: 0 10px;
        }}
        QPushButton:hover {{
            background-color: {TV['surface']};
            color: {TV['text_bright']};
            border-left: 3px solid {accent};
            padding-left: 7px;
        }}
        QPushButton:pressed {{ background-color: {TV['border_light']}; }}
    """

# 全域彈窗深色主題
TV_DIALOG_STYLE = f"""
    QDialog {{ background-color: {TV['bg']}; color: {TV['text']}; }}
    QWidget {{ background-color: {TV['bg']}; color: {TV['text']}; font-family: 'Segoe UI', '微軟正黑體', sans-serif; }}
    QLabel {{ font-size: 14px; color: {TV['text']}; }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTimeEdit {{
        background-color: {TV['surface']};
        color: {TV['text']};
        border: 1px solid {TV['border_light']};
        border-radius: 5px;
        padding: 6px 10px;
        font-size: 14px;
        selection-background-color: {TV['blue']};
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {TV['blue']};
    }}
    QPushButton {{
        background-color: {TV['surface']};
        color: {TV['text']};
        border: 1px solid {TV['border_light']};
        border-radius: 5px;
        font-size: 14px;
        font-weight: 600;
        padding: 7px 18px;
    }}
    QPushButton:hover {{ background-color: {TV['border_light']}; border-color: {TV['blue']}; color: white; }}
    QPushButton:pressed {{ background-color: {TV['blue_dim']}; }}
    QGroupBox {{
        color: {TV['text_dim']};
        font-weight: bold;
        font-size: 13px;
        border: 1px solid {TV['border_light']};
        border-radius: 7px;
        margin-top: 14px;
        padding-top: 8px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 6px;
        color: {TV['text_dim']};
    }}
    QTableWidget {{
        background-color: {TV['panel']};
        color: {TV['text']};
        gridline-color: {TV['border']};
        border: none;
    }}
    QTableWidget::item {{ color: {TV['text']}; padding: 4px; }}
    QTableWidget::item:selected {{ background-color: {TV['blue_dim']}; color: white; }}
    QHeaderView::section {{
        background-color: {TV['surface']};
        color: {TV['text_dim']};
        font-weight: bold;
        font-size: 13px;
        border: none;
        border-right: 1px solid {TV['border']};
        padding: 6px;
    }}
    QTextEdit, QTextBrowser {{
        background-color: {TV['console_bg']};
        color: {TV['console_text']};
        border: none;
        font-family: 'Cascadia Code', 'Consolas', '新細明體', monospace;
    }}
    QScrollBar:vertical {{
        background: {TV['panel']};
        width: 6px;
        border-radius: 3px;
    }}
    QScrollBar::handle:vertical {{
        background: {TV['surface']};
        border-radius: 3px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {TV['text_dim']}; }}
    QScrollBar:horizontal {{
        background: {TV['panel']};
        height: 6px;
    }}
    QScrollBar::handle:horizontal {{ background: {TV['surface']}; border-radius: 3px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QComboBox QAbstractItemView {{
        background-color: {TV['panel']};
        color: {TV['text']};
        selection-background-color: {TV['blue']};
        selection-color: white;
        border: 1px solid {TV['border_light']};
        outline: none;
    }}
    QCheckBox {{ color: {TV['text']}; font-size: 14px; }}
    QCheckBox::indicator {{
        width: 16px; height: 16px;
        background-color: {TV['surface']};
        border: 1px solid {TV['border_light']};
        border-radius: 3px;
    }}
    QCheckBox::indicator:checked {{ background-color: {TV['blue']}; border-color: {TV['blue']}; }}
    QProgressBar {{
        border: none;
        background-color: {TV['surface']};
        border-radius: 2px;
        height: 4px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {TV['blue']}, stop:1 {TV['green']});
        border-radius: 2px;
    }}
    QListWidget {{
        background-color: {TV['panel']};
        color: {TV['text']};
        border: 1px solid {TV['border_light']};
        border-radius: 5px;
    }}
    QListWidget::item:hover {{ background-color: {TV['surface']}; }}
    QListWidget::item:selected {{ background-color: {TV['blue_dim']}; color: white; }}
    QToolTip {{
        background-color: {TV['panel']};
        color: {TV['text']};
        border: 1px solid {TV['border_light']};
        font-size: 13px;
        padding: 4px 8px;
    }}
    QMessageBox {{ background-color: {TV['panel']}; }}
    QMessageBox QLabel {{ color: {TV['text']}; font-size: 14px; }}
    QMessageBox QPushButton {{ min-width: 80px; }}
"""

# ==================== 系統狀態與設定管理器 ====================

# 智慧股票輸入解析器
def resolve_stock_code(input_str):
    """解析輸入，支援代號(2330)、名稱(台積電)、或混合(2330台積電 / 台積電 2330)"""
    load_twse_name_map()
    input_str = input_str.strip()
    if not input_str: return None
    
    reverse_map = {}
    for mkt in ["TSE", "OTC"]:
        for code, name in STOCK_NAME_MAP.get(mkt, {}).items():
            reverse_map[name] = code
            
    # 完美命中名稱或代號
    if input_str in reverse_map: return reverse_map[input_str]
    if input_str in STOCK_NAME_MAP.get("TSE", {}) or input_str in STOCK_NAME_MAP.get("OTC", {}): return input_str
    
    # 萃取純數字與純文字
    digits = re.sub(r'\D', '', input_str)
    chars = re.sub(r'[\d\s\Wa-zA-Z]', '', input_str)
    
    # 判斷數字是否為有效代號
    if len(digits) >= 4 and (digits[:4] in STOCK_NAME_MAP.get("TSE", {}) or digits[:4] in STOCK_NAME_MAP.get("OTC", {})):
        return digits[:4]
        
    # 模糊名稱搜尋
    if chars in reverse_map: return reverse_map[chars]
    for name, code in reverse_map.items():
        if chars and (chars in name or name in chars): return code
        
    return None

class TradingConfig:
    def __init__(self):
        self.capital_per_stock = 15
        self.transaction_fee = 0.1425
        self.transaction_discount = 18.0
        self.trading_tax = 0.15
        self.below_50 = 500.0
        self.price_gap_50_to_100 = 1000.0
        self.price_gap_100_to_500 = 2000.0
        self.price_gap_500_to_1000 = 3000.0
        self.price_gap_above_1000 = 5000.0
        self.momentum_minutes = 1
        self.tg_bot_token = "8762583083:AAFm1B2-K6hzhAIvOoBrakJf3C1VPXtkE-4" 
        self.tg_chat_id = "" 
        self.tg_notify_enabled = True
        self.is_monitoring = False
        self.allow_reentry = False             # 是否開啟停損再進場
        self.max_reentry_times = 1             # 最多可再進場幾次
        self.reentry_lookback_candles = 3      # 停損後往前檢查幾根K棒
        
        # 進階策略配置：進階策略參數
        self.similarity_threshold = 0.35      # DTW 相似度門檻
        self.pull_up_pct_threshold = 1.2      # 領漲股拉高漲幅門檻 (%)
        self.follow_up_pct_threshold = 0.5    # 跟漲股追蹤漲幅門檻 (%)
        self.rise_lower_bound = -1.0          # 當日總漲幅下限 (%)
        self.rise_upper_bound = 7.5           # 當日總漲幅上限 (%)
        self.volume_multiplier = 1.1          # 等待期均量門檻 (倍數)
        self.min_volume_threshold = 800        # 等待期絕對數量門檻 (張)
        self.pullback_tolerance = 0.5         # 突破前高容錯率 (%)
        self.min_lag_pct = 1               # 選股：標的漲幅至少落後領漲股幾% (指標一：相對位置)
        self.min_height_pct = 3.5            # 選股：標的當日最高漲幅需達此值才可進場% (指標三：高度門檻)
        self.wait_min_avg_vol = 15          # 流動性：等待期均量需達此值 (張/分鐘)，OR邏輯：任一達標即放行
        self.wait_max_single_vol = 40       # 流動性：等待期單根最大量需達此值 (張)，OR邏輯：任一達標即放行
        self.volatility_min_range = 1.5     # 選股：當日漲跌幅活動範圍下限 (%)，過濾靜止不動的股票
        self.min_eligible_avg_vol = 1       # 選股：全日均量下限 (張/分)，0=不過濾，過濾委買賣稀薄的幽靈股
        self.min_close_price = 0            # 選股：股價下限 (元)，0=不過濾
        self.slippage_ticks = 3             # IOC委託向下穿價容許 (Tick)
        self.sl_cushion_pct = 0.0           # 停損緩衝空間 (%)，加在停損價上方，與回歸對齊
        self.cutoff_time_mins = 270         # 尾盤截止觸發 (分鐘，270=13:30=不截止)
        self.allow_leader_entry = False     # 無合格跟漲股時，是否允許做空領漲股
        self.live_trading_mode = False       # 正式下單模式（False=模擬，True=正式）

        # 🛡️ 風控模組
        self.total_capital = 249             # 總額度（萬元）
        self.max_daily_entries = 12          # 每日最大進場檔數
        self.max_daily_stops = 3             # 停損熔斷：當日累計停損N筆後停止進場
        self.risk_control_enabled = True     # 風控模組開關
        self.sound_enabled = True            # 音效提示開關

sys_config = TradingConfig()

class _NullLog:
    """headless 模式用：忽略所有 append，避免 f-string 格式化與 list 累積的開銷。"""
    __slots__ = ()
    def append(self, *a): pass

class TradingState:
    def __init__(self):
        # 1. 先從資料庫載入原始持倉
        raw_pos = sys_db.load_state('current_position', {})
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        valid_pos = {}
        purged_list = []
        
        # 2. 🚀 核心日期檢測邏輯
        for sym, info in raw_pos.items():
            entry_date = info.get('entry_date')
            
            # 如果這檔股票有存日期、日期就是今天、且已有實際成交量，才保留
            if entry_date == today_str and info.get('filled_shares', 0) > 0:
                valid_pos[sym] = info
            else:
                # 日期不符 (或者是舊版沒存日期的資料)，一律視為過期持倉並刪除
                purged_list.append(sym)
        
        if purged_list:
            print(f"[系統清理] 偵測到 {len(purged_list)} 筆過期持倉 ({', '.join(purged_list)})，已自動清除。")
            # 同步回資料庫，徹底洗白過期紀錄
            sys_db.save_state('current_position', valid_pos)
            
        self.open_positions = valid_pos
        
        self.triggered_limit_up = set()
        self.previous_stop_loss = set()
        self.risk_state = {'daily_entries': 0, 'daily_stops': 0, 'halted': False}  # 🛡️ 風控模組
        self.in_memory_intraday = {}
        self.lock = threading.RLock()
        self.quit_flag = False
        self.api = None
        self.to = None
        self.trading = False
        self.stop_trading_flag = False

# ==================== Shioaji 大數據專用採集引擎 ====================

class SlidingWindowLimiter:
    def __init__(self, max_calls, period_seconds):
        self.max_calls, self.period = max_calls, period_seconds
        self.timestamps, self.lock = deque(), threading.Lock()
    def wait_and_consume(self):
        with self.lock:
            now = time_module.time()
            while self.timestamps and now - self.timestamps[0] > self.period:
                self.timestamps.popleft()
            if len(self.timestamps) >= self.max_calls:
                time_module.sleep(self.period - (now - self.timestamps[0]))
                now = time_module.time()
                while self.timestamps and now - self.timestamps[0] > self.period:
                    self.timestamps.popleft()
            self.timestamps.append(now)

shioaji_limiter = SlidingWindowLimiter(max_calls=45, period_seconds=5.0)

def fill_zero_volume_kbars(df):
    """自動補齊 09:00 ~ 13:30 之間所有 Volume=0 的分鐘，並校正 Shioaji 的延遲時間戳"""
    df['ts'] = pd.to_datetime(df['ts'])
    
    # === 核心校正 1：時間平移 ===
    # 找出 13:30 的最後一盤 (不平移)
    mask_1330 = df['ts'].dt.time == pd.to_datetime('13:30:00').time()
    df_1330 = df[mask_1330].copy()
    
    # 盤中時間 (09:01 ~ 13:25) 往前平移 1 分鐘變成 (09:00 ~ 13:24)
    df_others = df[~mask_1330].copy()
    df_others['ts'] = df_others['ts'] - pd.Timedelta(minutes=1)
    
    # 合併並重新按時間排序
    df = pd.concat([df_others, df_1330]).sort_values('ts')
    
    # === 核心校正 2：標準補齊作業 (含 13:25~13:29 集合競價黑洞) ===
    df = df.set_index('ts')
    all_filled = []
    for day, group in df.groupby('day'):
        full_idx = pd.date_range(start=f"{day} 09:00:00", end=f"{day} 13:30:00", freq='1min')
        
        # 移除平移可能產生的極少數重複索引，保留最新狀態
        group = group[~group.index.duplicated(keep='last')]
        
        # reindex 會自動把 13:25~13:29 等空缺的時間生出來 (預設為 NaN)
        group = group.reindex(full_idx)
        
        # 價格向後填充：13:24 的收盤價會自動一路填滿 13:25~13:29 的空洞
        group['close'] = group['close'].ffill().bfill() 
        for c in ['open', 'high', 'low']: 
            group[c] = group[c].fillna(group['close'])
            
        # 沒資料的分鐘，成交量補 0 (確保 13:25~13:29 都是 0)
        group['volume'] = group['volume'].fillna(0)
        
        group['symbol'] = group['symbol'].ffill().bfill()
        group['day'] = day
        group['time'] = group.index.strftime('%H:%M:%S')
        all_filled.append(group)
        
    if not all_filled: return pd.DataFrame()
    return pd.concat(all_filled).reset_index().rename(columns={'index': 'ts'})

class ShioajiKLineFetchThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, selected_dates, symbols, twse_holidays):
        super().__init__()
        self.selected_dates = sorted(selected_dates)
        self.symbols = symbols
        self.twse_holidays = twse_holidays

    def get_contiguous_blocks(self):
        """🧠 智慧分組大腦：將使用者選擇的日期，依據真實交易日合併為連續區塊"""
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in self.selected_dates]
        blocks = []
        if not dates: return blocks
        
        current_block = [dates[0]]
        for i in range(1, len(dates)):
            prev = current_block[-1]
            curr = dates[i]
            is_contiguous = True
            
            # 檢查中間斷掉的日子，是不是剛好全是假日
            test_day = prev + timedelta(days=1)
            while test_day < curr:
                is_weekend = test_day.weekday() >= 5
                is_holiday = test_day.strftime("%Y%m%d") in self.twse_holidays
                if not (is_weekend or is_holiday):
                    is_contiguous = False
                    break
                test_day += timedelta(days=1)
                
            if is_contiguous:
                current_block.append(curr)
            else:
                blocks.append([d.strftime("%Y-%m-%d") for d in current_block])
                current_block = [curr]
        blocks.append([d.strftime("%Y-%m-%d") for d in current_block])
        return blocks

    def run(self):
        try:
            self.progress_signal.emit(5, f"啟動 Shioaji 連線...")
            _is_live = getattr(sys_config, 'live_trading_mode', False)
            _ak = shioaji_logic.LIVE_API_KEY if _is_live else shioaji_logic.TEST_API_KEY
            _as = shioaji_logic.LIVE_API_SECRET if _is_live else shioaji_logic.TEST_API_SECRET
            api = sj.Shioaji(simulation=not _is_live)
            api.login(api_key=_ak, secret_key=_as, contracts_timeout=10000)
            if _is_live and shioaji_logic.CA_CERT_PATH and shioaji_logic.CA_PASSWORD:
                api.activate_ca(ca_path=shioaji_logic.CA_CERT_PATH, ca_passwd=shioaji_logic.CA_PASSWORD)

            blocks = self.get_contiguous_blocks()
            
            # 1. 抓取處置股 (💡 神級省時優化：每個區塊只查詢最後一天，即可涵蓋前 15 天)
            self.progress_signal.emit(10, f"還原每日處置現場 (共 {len(self.selected_dates)} 天)...")
            dispo_dict = {}
            for d_idx, t_date in enumerate(self.selected_dates):
                self.progress_signal.emit(10 + int((d_idx / len(self.selected_dates)) * 5), f"還原 {t_date} 處置名單...")
                dispo_dict[t_date] = fetch_active_disposition_on_date(t_date)
                
            sys_db.save_state('disposition_stocks_dict', dispo_dict)
            
            # ⚠️ 關鍵修正：不再提前過濾 symbols_to_fetch！因為不同天有不同的處置狀態，全部都要抓 K 線
            symbols_to_fetch = self.symbols
            total_syms = len(symbols_to_fetch)
            all_dfs = []

            # 2. 抓取 K 線
            for i, sym in enumerate(symbols_to_fetch):
                shioaji_limiter.wait_and_consume()
                if (i+1) % 5 == 0:
                    self.progress_signal.emit(15 + int((i/total_syms)*80), f"採集 K 線 ({i+1}/{total_syms}) - {sym}")
                
                contract = api.Contracts.Stocks.get(sym)
                if not contract: continue
                
                sym_dfs = []
                for b_idx, block in enumerate(blocks):
                    start_dt = datetime.strptime(block[0], "%Y-%m-%d")
                    fetch_start = (start_dt - timedelta(days=5)).strftime("%Y-%m-%d") # 抓取基期
                    end_dt_str = block[-1]
                    
                    kbars = api.kbars(contract, start=fetch_start, end=end_dt_str)
                    df = pd.DataFrame({**kbars})
                    if df.empty: continue
                    
                    df.columns = [c.lower() for c in df.columns]
                    df['ts'] = pd.to_datetime(df['ts'])
                    df['day'] = df['ts'].dt.strftime('%Y-%m-%d')
                    df['symbol'] = sym
                    
                    df = fill_zero_volume_kbars(df)
                    
                    daily_last = df.groupby('day')['close'].last().shift(1)
                    df['yesterday_close'] = df['day'].map(daily_last)
                    
                    # 🔴 神聖過濾網：只保留該 Block 裡面「真正有被勾選」的日子！
                    df = df[df['day'].isin(block)].dropna(subset=['yesterday_close'])
                    if df.empty: continue
                    
                    df['limit_up'] = df['yesterday_close'].apply(calculate_limit_up_price)
                    df['limit_up'] = df['limit_up'].apply(lambda x: math.floor(x * 100) / 100.0 if pd.notnull(x) else x)
                    df['rise'] = round((df['close'] - df['yesterday_close']) / df['yesterday_close'] * 100, 2)
                    df['highest'] = df.groupby('day')['high'].cummax()
                    df['pct_increase'] = df.groupby('day')['rise'].diff().fillna(df['rise']).round(2)
                    
                    df.rename(columns={'yesterday_close': '昨日收盤價', 'limit_up': '漲停價'}, inplace=True)
                    cols = ['symbol', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', '昨日收盤價', '漲停價', 'rise', 'highest', 'pct_increase']
                    df['date'] = df['day']
                    sym_dfs.append(df[cols])
                    
                    # 💡 安全休息：如果有多個區塊，給 API 微小喘息空間防斷線
                    if len(blocks) > 1 and b_idx < len(blocks) - 1:
                        time_module.sleep(0.01)
                
                if sym_dfs:
                    all_dfs.extend(sym_dfs)

            # 3. 寫入資料庫
            self.progress_signal.emit(95, "正在寫入資料庫...")
            if all_dfs:
                final_df = pd.concat(all_dfs, ignore_index=True)
                intra_dict = {str(sym): group.to_dict('records') for sym, group in final_df.groupby('symbol')}
                
                # 自動洗掉舊資料庫，並換上新的陣列日期
                sys_db.save_kline('intraday_kline_history', intra_dict)
                sys_db.save_state('last_fetched_dates', self.selected_dates)
                
            api.logout()
            self.finished_signal.emit(True, f"✅ 採集完成，共 {len(self.selected_dates)} 個交易日")
        except Exception as e:
            _emsg = str(e)
            if "6999" in _emsg or "token_login" in _emsg:
                self.finished_signal.emit(False, "採集失敗：Shioaji 登入被拒（錯誤 6999）。可能原因：API 憑證錯誤/過期、非交易時段伺服器限制、或 live_trading_mode 設定與憑證類型不符。請確認後重試。")
            else:
                self.finished_signal.emit(False, f"採集崩潰: {_emsg}")

# ==========================================
# 📖 究極文件讀取引擎 (幫主程式徹底瘦身)
# ==========================================
def load_tutorial(doc_id):
    # 💡 這裡是讀取外部 docs 資料夾的 tutorials.json
    doc_path = os.path.join("docs", "tutorials.json")
    try:
        if not os.path.exists(doc_path):
            # 防呆：如果 json 不存在，回傳基礎說明
            return f"<h3>⚠️ 系統找不到代號為 '{doc_id}' 的教學內容</h3><p>請確認 docs/tutorials.json 檔案已正確建立。</p>"
        with open(doc_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        return docs.get(doc_id, f"<h3>⚠️ JSON 找不到 ID: {doc_id}</h3>")
    except Exception as e:
        return f"<h3>❌ 教學讀取失敗</h3><p>{e}</p>"

# ==========================================
# 📖 開放式系統百科全書 (Codex) - 究極美化版
# ==========================================
class SystemTutorialDialog(QDialog):
    def __init__(self, parent=None, start_page=None): # 💡 增加 start_page 參數
        super().__init__(parent)
        self.setWindowTitle("系統百科與操作指南")
        self.resize(800, 650)
        self.setStyleSheet(TV_DIALOG_STYLE)

        main_layout = QVBoxLayout(self)
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # --- 📄 第 0 頁：動態目錄頁 ---
        self.toc_page = QWidget()
        toc_layout = QVBoxLayout(self.toc_page)
        toc_layout.setContentsMargins(15, 15, 15, 15)

        title_lbl = QLabel("系統百科與操作指南")
        title_lbl.setStyleSheet(f"color: {TV['yellow']}; font-size: 26px; font-weight: bold; margin-bottom: 25px;")
        title_lbl.setAlignment(Qt.AlignCenter)
        toc_layout.addWidget(title_lbl)

        data = init_and_load_tutorials()
        menu_data = data.get("menu_config", [])

        btn_style = f"""
            QPushButton {{ background-color: {TV['surface']}; color: {TV['text']}; padding: 20px; border-radius: 10px; font-size: 16px; text-align: left; font-weight: bold; border: 1px solid {TV['border_light']}; }}
            QPushButton:hover {{ background-color: {TV['border_light']}; border-left: 8px solid {TV['blue']}; color: {TV['text_bright']}; }}
        """
        
        for item in menu_data:
            btn = QPushButton(item['title'])
            btn.setStyleSheet(btn_style)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, i=item['id']: self.show_content(i))
            toc_layout.addWidget(btn)
        
        toc_layout.addStretch()
        self.stacked_widget.addWidget(self.toc_page)
        
        # --- 📄 第 1 頁：內容閱讀頁 ---
        self.content_page = QWidget()
        content_layout = QVBoxLayout(self.content_page)
        content_layout.setContentsMargins(15, 15, 15, 15)
        
        btn_back = QPushButton("🔙 返回目錄")
        btn_back.setStyleSheet(f"QPushButton {{ background-color: {TV['red']}; color: white; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 14px; }} QPushButton:hover {{ background-color: #ff6b6b; }}")
        btn_back.setCursor(Qt.PointingHandCursor)
        btn_back.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        
        header_layout = QHBoxLayout()
        header_layout.addWidget(btn_back)
        header_layout.addStretch()
        content_layout.addLayout(header_layout)
        
        self.browser = QTextBrowser()
        self.browser.setStyleSheet(f"QTextBrowser {{ background-color: {TV['bg']}; border: none; color: {TV['text']}; }}")
        content_layout.addWidget(self.browser)
        self.stacked_widget.addWidget(self.content_page)
        
        # 💡 關鍵邏輯：如果有指定初始頁面，就直接跳轉過去
        if start_page:
            self.show_content(start_page)
        else:
            self.stacked_widget.setCurrentIndex(0)

    def show_content(self, doc_id):
        content = load_tutorial(doc_id)
        safe_html = f"<html><body style='background-color:{TV['bg']}; color:{TV['text']}; margin:0; padding:10px;'>{content}</body></html>"
        self.browser.setHtml(safe_html)
        self.stacked_widget.setCurrentIndex(1)

# ==================== SQLite 資料庫管理器 ====================
class DBManager:
    def __init__(self, db_name="quant_data.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.db_lock = threading.Lock()
        self._create_tables()
        self._init_docs()

    def _create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, action TEXT, symbol TEXT,
                    shares INTEGER, price REAL, profit REAL, note TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS system_docs (
                    doc_id TEXT PRIMARY KEY,
                    content TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    key_name TEXT PRIMARY KEY,
                    json_data TEXT
                )
            """)

    def _init_docs(self):
        doc_phone = """📱 <b>手機端量化遙控器 - 使用指南</b>

<b>【一、盤中監控篇】</b>
1. <b>登入與啟動</b>：初次使用請輸入 <code>登入 身分證 密碼 憑證密碼</code> 綁定帳戶。接著點擊「▶ 啟動盤中監控」設定參數後執行。
2. <b>盤中保護</b>：監控一旦啟動，為保護運算資源，系統將自動鎖定「參數設定」、「回測」、「更新K線」等重度功能。若重複點擊啟動，將自動引導您查看持倉。
3. <b>即時推播</b>：系統會在「領漲替換」、「無縫升級漲停」、「進場買進」、「停損/超時平倉」時主動發送推播給您。

<b>【二、盤後分析篇 (限13:30後使用)】</b>
1. <b>K線與大數據</b>：請先執行「更新 K 線數據」，若要進行進階分析，請至分析選單執行「採集大數據」。
2. <b>智能 DTW 掃描</b>：基於大數據運算族群連動的最適門檻，算完後會以表格回傳手機。
3. <b>自選與極大化</b>：可針對特定族群進行歷史回測，或使用雲端算力全參數掃描最佳「等待/持有」時間。

<b>【三、實用工具】</b>
• <b>智能看圖</b>：無需任何指令，直接在聊天框輸入股票代號 (如: <code>2330</code>) 或族群名稱 (如: <code>散熱</code>)，系統將立即顯示當日走勢圖！
• <b>緊急平倉</b>：若遇突發狀況，點擊「🛑 緊急/手動平倉」並確認，系統將以市價平倉所有持股。"""

        doc_pc = """💻 <b>電腦端主控台 - 協同運作指南</b>

<b>【一、核心運作觀念】</b>
1. <b>本機常駐</b>：您的電腦是整套量化系統的「大腦」，手機端的一切指令皆交由電腦運算。請確保程式保持開啟，且電腦「不可進入睡眠模式」。
2. <b>雙向同步</b>：手機與電腦端的持倉、設定與交易紀錄完全同步。無論從何處觸發平倉，兩邊皆會生效。

<b>【二、終端機監控 (黑畫面)】</b>
1. <b>安全攔截機制</b>：系統具備軍規級防護。若非您綁定的手機發送指令，系統會直接丟棄並在終端機印出 <code>[防護] 攔截到未授權訊息</code>。
2. <b>進度記錄</b>：當手機觸發「DTW 掃描」或「全參數掃描」時，手機會顯示百分比，而電腦終端機會即時印出每一檔股票的運算細節與詳細日誌。

<b>【三、GUI 介面操作】</b>
• 介面左側面板功能與手機端完全一致。
• <b>族群管理</b>：支援直接雙擊股票名稱查看圖表。
• 若遇任何網路異常，可直接關閉視窗，系統將安全切斷 API 連線。"""

        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO system_docs (doc_id, content) VALUES (?, ?)", ('tut_phone', doc_phone))
            self.conn.execute("INSERT OR REPLACE INTO system_docs (doc_id, content) VALUES (?, ?)", ('tut_pc', doc_pc))

    def log_trade(self, action, symbol, shares, price, profit=0.0, note=""):
        with self.db_lock:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO trade_logs (timestamp, action, symbol, shares, price, profit, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action, symbol, shares, price, profit, note)
            )

    def save_kline(self, table_name, data_dict):
        """逐 symbol 分批寫入，避免一次 concat 全部資料造成記憶體溢出"""
        if not data_dict: return
        with self.db_lock:
            with self.conn:
                self.conn.execute(f"DROP TABLE IF EXISTS [{table_name}]")
                for sym, records in data_dict.items():
                    if not records: continue
                    df = pd.DataFrame(records).assign(symbol=sym).astype(str)
                    df.to_sql(table_name, self.conn, if_exists='append', index=False)

    def load_kline(self, table_name, dates=None):
        try:
            with self.db_lock:
                if dates:
                    dates_str = "','".join(dates)
                    df = pd.read_sql(f"SELECT * FROM {table_name} WHERE date IN ('{dates_str}')", self.conn)
                else:
                    df = pd.read_sql(f"SELECT * FROM {table_name}", self.conn)
            # 向下相容：舊 DB 中的 '2min_pct_increase' 欄位自動映射
            if '2min_pct_increase' in df.columns and 'pct_increase' not in df.columns:
                df.rename(columns={'2min_pct_increase': 'pct_increase'}, inplace=True)
            cols_to_numeric = ['close', 'high', 'low', 'open', 'volume', 'rise', 'pct_increase', '漲停價', '昨日收盤價', 'highest']
            for c in cols_to_numeric:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            return {sym: group.to_dict('records') for sym, group in df.groupby('symbol')}
        except Exception: return {}

    def get_recent_kline_dates(self, n=30):
        """回傳 intraday_kline_history 中最近 n 個不重複的日期字串"""
        try:
            with self.db_lock:
                cur = self.conn.execute(
                    "SELECT DISTINCT date FROM intraday_kline_history ORDER BY date DESC LIMIT ?", (n,))
                return [row[0] for row in cur.fetchall()]
        except Exception:
            return []

    # 通用的 JSON-to-SQLite 讀寫介面
    def save_state(self, key_name, data):
        with self.db_lock:
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO system_state (key_name, json_data) VALUES (?, ?)",
                    (key_name, json.dumps(data, ensure_ascii=False))
                )

    def load_state(self, key_name, default_value=None):
        if default_value is None: default_value = {} if key_name.endswith('dict') or key_name == 'settings' else []
        try:
            with self.db_lock:
                cursor = self.conn.execute("SELECT json_data FROM system_state WHERE key_name = ?", (key_name,))
                row = cursor.fetchone()
            return json.loads(row[0]) if row else default_value
        except Exception:
            return default_value

sys_db = DBManager()
sys_state = TradingState()

# ==================== 極速直連通訊管理器 (v1.9.8.6) ====================
class CloudBrainManager:
    def __init__(self): 
        self.is_running = False
        self.session = requests.Session()
        self.offset = None
        self.task_lock = threading.Lock()
        self.ui_states = {} 

    def _get_token(self):
        t = getattr(sys_config, 'tg_bot_token', '').strip()
        return t[3:] if t.lower().startswith('bot') else t

    def _get_chat_id(self):
        # 兼容舊版設定：如果舊版的 tg_auth_key 是一長串數字(Telegram ID)，視為有效 Chat ID
        chat_id = getattr(sys_config, 'tg_chat_id', '').strip()
        fallback = getattr(sys_config, 'tg_auth_key', '').strip()
        if not chat_id and len(fallback) >= 8: 
            return fallback
        return chat_id

    def _ensure_pairing_pin(self):
        # 確保存取固定金鑰：同一台電腦生成後永久固定，除非刪除設定檔重置
        pin = getattr(sys_config, 'tg_pairing_pin', '').strip()
        if not pin or len(pin) != 6:
            pin = str(random.randint(100000, 999999))
            sys_config.tg_pairing_pin = pin
            try: save_settings()
            except Exception: pass
        return pin

    def send_message(self, text, force=False, reply_markup=None):
        token, chat_id = self._get_token(), self._get_chat_id()
        if not token or not chat_id: return None
        if not force and not getattr(sys_config, 'tg_notify_enabled', True): return None
        try: 
            payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
            if reply_markup: payload["reply_markup"] = reply_markup
            res = self.session.post(f"https://api.telegram.org/bot{token}/sendMessage", json=payload, timeout=10)
            if res.status_code == 200: return res.json().get("result", {}).get("message_id")
        except Exception: pass
        return None

    def edit_message_text(self, message_id, text, reply_markup=None):
        token, chat_id = self._get_token(), self._get_chat_id()
        if not token or not chat_id or not message_id: return
        try:
            payload = {"chat_id": chat_id, "message_id": message_id, "text": text, "parse_mode": "HTML"}
            if reply_markup: payload["reply_markup"] = reply_markup
            self.session.post(f"https://api.telegram.org/bot{token}/editMessageText", json=payload, timeout=5)
        except Exception: pass

    def send_photo(self, photo_bytes, caption=""):
        token, chat_id = self._get_token(), self._get_chat_id()
        if not token or not chat_id: return
        def _send():
            try:
                data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
                files = {"photo": ("chart.png", photo_bytes, "image/png")}
                self.session.post(f"https://api.telegram.org/bot{token}/sendPhoto", data=data, files=files, timeout=20)
            except Exception as e: logger.error(f"Telegram 傳送圖片異常: {e}")
        threading.Thread(target=_send, daemon=True).start()

    def send_document(self, file_path, caption=""):
        token, chat_id = self._get_token(), self._get_chat_id()
        if not token or not chat_id: return
        def _send():
            try:
                with open(file_path, 'rb') as f:
                    files = {"document": f}
                    data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
                    self.session.post(f"https://api.telegram.org/bot{token}/sendDocument", data=data, files=files, timeout=20)
            except Exception as e: logger.error(f"Telegram 傳送檔案異常: {e}")
        threading.Thread(target=_send, daemon=True).start()

    def send_chat_action(self, action="typing"):
        token, chat_id = self._get_token(), self._get_chat_id()
        if token and chat_id:
            try: self.session.post(f"https://api.telegram.org/bot{token}/sendChatAction", json={"chat_id": chat_id, "action": action}, timeout=5)
            except Exception: pass

    def _set_bot_menu(self):
        token = self._get_token()
        cmds = [{"command": "menu", "description": "📱 開啟全端遙控中心"}]
        try: self.session.post(f"https://api.telegram.org/bot{token}/setMyCommands", json={"commands": cmds}, timeout=5)
        except Exception: pass

    def get_bottom_keyboard(self):
        return {"keyboard": [
            [{"text": "▶ 啟動監控"}, {"text": "◉ 即時持倉"}],
            [{"text": "◈ 盤後分析"}, {"text": "◎ 自選進場"}],
            [{"text": "◆ 量能分析"}, {"text": "⊞ 族群管理"}],
            [{"text": "⊟ 更新K線"}, {"text": "📜 交易紀錄"}],
            [{"text": "📈 走勢圖表"}, {"text": "⚙ 參數設定"}],
            [{"text": "🛑 緊急平倉"}]
        ], "resize_keyboard": True, "selective": False}

    def get_analysis_menu(self):
        return {"inline_keyboard": [
            [{"text": "📥  採集相似度大數據庫", "callback_data": "cmd_fetch_data"},
             {"text": "🔬  DTW 門檻智能掃描",  "callback_data": "cmd_opt_sim"}],
            [{"text": "📊  計算平均過高間隔",   "callback_data": "cmd_avg_high"},
             {"text": "🔥  量能突破大數據",      "callback_data": "cmd_volume_breakout"}],
            [{"text": "📤  匯出 CSV 對帳單",    "callback_data": "cmd_export_csv"}],
            [{"text": "✕  關閉",                "callback_data": "cmd_close_menu"}]
        ]}

    def get_groups_keyboard(self):
        mat = load_matrix_dict_analysis()
        kb, row = [], []
        for g in mat.keys():
            row.append({"text": f"📁 {g}", "callback_data": f"grp_show_{g}"})
            if len(row) == 2:
                kb.append(row)
                row = []
        if row: kb.append(row)
        kb.append([{"text": "❌ 關閉面板", "callback_data": "cmd_close_menu"}])
        return {"inline_keyboard": kb}

    def get_slider_keyboard(self, state, prefix):
        groups = ["所有族群"] + list(load_matrix_dict_analysis().keys())
        g_idx = state.get('g_idx', 0) % len(groups)
        w, h = state.get('w', 5), state.get('h', 270)
        h_str = "尾盤(F)" if h == 270 else f"{h}分"
        
        kb = []
        if prefix == "sim": kb.append([{"text": "◀", "callback_data": f"{prefix}_g_prev"}, {"text": f"📁 族群: {groups[g_idx]}", "callback_data": "dummy"}, {"text": "▶", "callback_data": f"{prefix}_g_next"}])
        kb.append([{"text": "➖", "callback_data": f"{prefix}_w_dec"}, {"text": f"等待時間: {w}分", "callback_data": "dummy"}, {"text": "➕", "callback_data": f"{prefix}_w_inc"}])
        kb.append([{"text": "➖", "callback_data": f"{prefix}_h_dec"}, {"text": f"持有時間: {h_str}", "callback_data": "dummy"}, {"text": "➕", "callback_data": f"{prefix}_h_inc"}])
        btn_txt = "▶️ 執行回測" if prefix == "sim" else "🚀 啟動盤中監控"
        kb.append([{"text": btn_txt, "callback_data": f"{prefix}_execute"}, {"text": "❌ 取消", "callback_data": "cmd_close_menu"}])
        return {"inline_keyboard": kb}

    def get_max_builder_keyboard(self, state):
        groups = ["所有族群"] + list(load_matrix_dict_analysis().keys())
        g_idx = state.get('g_idx', 0) % len(groups)
        ws, we, hs, he = state.get('ws', 3), state.get('we', 5), state.get('hs', 10), state.get('he', 20)
        hss = "尾盤(F)" if hs == 270 else f"{hs}分"
        hes = "尾盤(F)" if he == 270 else f"{he}分"
        return {"inline_keyboard": [
            [{"text": "◀", "callback_data": "max_g_prev"}, {"text": f"📁 族群: {groups[g_idx]}", "callback_data": "dummy"}, {"text": "▶", "callback_data": "max_g_next"}],
            [{"text": "➖", "callback_data": "max_ws_dec"}, {"text": f"等(起): {ws}分", "callback_data": "dummy"}, {"text": "➕", "callback_data": "max_ws_inc"}],
            [{"text": "➖", "callback_data": "max_we_dec"}, {"text": f"等(終): {we}分", "callback_data": "dummy"}, {"text": "➕", "callback_data": "max_we_inc"}],
            [{"text": "➖", "callback_data": "max_hs_dec"}, {"text": f"持(起): {hss}", "callback_data": "dummy"}, {"text": "➕", "callback_data": "max_hs_inc"}],
            [{"text": "➖", "callback_data": "max_he_dec"}, {"text": f"持(終): {hes}", "callback_data": "dummy"}, {"text": "➕", "callback_data": "max_he_inc"}],
            [{"text": "▶️ 確認執行掃描", "callback_data": "max_execute"}, {"text": "❌ 取消", "callback_data": "cmd_close_menu"}]
        ]}

    def start(self):
        token = self._get_token()
        if not token: 
            return print("\n⚠️ [遙控中心] 尚未設定 Telegram Bot Token！")
            
        chat_id = self._get_chat_id()
        if not chat_id:
            pin = self._ensure_pairing_pin()
            print(f"\n[遙控中心] 尚未綁定 Telegram 帳號")
            print(f"[遙控中心] 請打開 Telegram，向機器人發送以下指令完成綁定：")
            print(f"[遙控中心]    綁定 {pin}\n")
            
        if self.is_running: return
        self.is_running = True
        
        def _init_and_poll():
            print("[遙控中心] 正在檢查雲端連線狀態...")
            try: 
                res = self.session.get(f"https://api.telegram.org/bot{token}/deleteWebhook?drop_pending_updates=True", timeout=10)
                if res.status_code == 200 and chat_id:
                    print("[遙控中心] ✅ 雲端連線正常，遙控核心已啟動。")
                    self._set_bot_menu()
            except Exception: pass
            self._poll()
            
        threading.Thread(target=_init_and_poll, daemon=True).start()

    def _poll(self):
        token = self._get_token()
        if self._get_chat_id():
            print("[遙控中心] Telegram 控制就緒，等待指令...")
            
        while self.is_running:
            try:
                url = f"https://api.telegram.org/bot{token}/getUpdates?timeout=10"
                if self.offset: url += f"&offset={self.offset}"
                res = self.session.get(url, timeout=15)
                
                if res.status_code == 200:
                    target_id = self._get_chat_id()
                    
                    for item in res.json().get("result", []):
                        self.offset = item["update_id"] + 1
                        
                        if "message" in item and "text" in item["message"]:
                            msg = item["message"]
                            sender_id = str(msg.get("chat", {}).get("id", ""))
                            text = msg["text"].strip()

                            if text.startswith("綁定"):
                                input_pin = text.replace("綁定", "").strip()
                                expected_pin = self._ensure_pairing_pin()
                                
                                if input_pin == expected_pin:
                                    sys_config.tg_chat_id = sender_id
                                    try: save_settings()
                                    except Exception: pass
                                    self.send_message("✅ <b>帳號綁定成功！</b>\n系統已安全記錄您的專屬 ID。\n現在請輸入 /menu 或點擊左下角按鈕開啟控制面板。", force=True)
                                    print(f"\n[遙控中心] 與 Telegram 裝置 {sender_id} 完成連線綁定！")
                                    self._set_bot_menu()
                                else:
                                    self.session.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": sender_id, "text": "❌ <b>金鑰錯誤</b>\n請重新確認電腦端終端機顯示的 6 位數密碼。", "parse_mode": "HTML"})
                                continue
                            
                            # 未綁定帳號的請求攔截
                            if sender_id != target_id:
                                if text in ["/start", "開始"]:
                                    pin = self._ensure_pairing_pin()
                                    welcome_msg = f"<b>歡迎使用量化交易終端！</b>\n\n系統偵測到您尚未綁定裝置。\n請輸入電腦端軟體顯示的綁定指令，例如：\n<code>綁定 {pin}</code>"
                                    self.session.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": sender_id, "text": welcome_msg, "parse_mode": "HTML"})
                                else:
                                    print(f"\n⚠️ [防護] 攔截到未授權訊息: 來源 ID '{sender_id}'")
                                continue
                            self._exec(text, message_id=msg.get("message_id"))
                            
                        elif "callback_query" in item:
                            cb = item["callback_query"]
                            sender_id = str(cb.get("message", {}).get("chat", {}).get("id", ""))
                            data_str, msg_id = cb.get("data"), cb.get("message", {}).get("message_id")
                            try: self.session.post(f"https://api.telegram.org/bot{token}/answerCallbackQuery", json={"callback_query_id": cb["id"]}, timeout=5)
                            except Exception: pass
                            
                            if sender_id == target_id and data_str != "dummy":
                                self._handle_callback(data_str, msg_id, sender_id)
            except Exception as e: 
                time_module.sleep(1)
            time_module.sleep(0.1)

    def _update_slider_state(self, st, action):
        if action.endswith("_w_inc"): st['w'] = min(60, st.get('w',5) + 1)
        elif action.endswith("_w_dec"): st['w'] = max(1, st.get('w',5) - 1)
        elif action.endswith("_h_inc"):
            h = st.get('h', 270); st['h'] = 5 if h == 270 else (270 if h >= 120 else h + 5)
        elif action.endswith("_h_dec"):
            h = st.get('h', 270); st['h'] = 120 if h == 270 else (270 if h <= 5 else h - 5)
        elif action.endswith("_g_next"): st['g_idx'] = st.get('g_idx',0) + 1
        elif action.endswith("_g_prev"): st['g_idx'] = st.get('g_idx',0) - 1

    def _handle_callback(self, data_str, msg_id, chat_id):
        if data_str in ["cmd_update_kline", "cmd_opt_sim", "cmd_fetch_data", "cmd_avg_high", "max_execute", "sim_execute"]:
            if getattr(sys_state, 'trading', False):
                return self.send_message("⚠️ <b>拒絕存取：盤中監控正在執行中！</b>\n為確保交易零延遲與資料庫安全，盤中禁止執行耗時運算任務。", force=True)

        if data_str == "cmd_close_menu": return self.edit_message_text(msg_id, "✅ <b>面板已關閉。</b>")
        
        elif data_str == "cmd_export_csv":
            self.edit_message_text(msg_id, "⏳ <b>正在產生完整對帳單...</b>")
            try:
                df = pd.read_sql("SELECT * FROM trade_logs ORDER BY id DESC", sys_db.conn)
                file_name = f"Trade_Logs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                df.to_csv(file_name, index=False, encoding='utf-8-sig') 
                self.send_document(file_name, f"📥 <b>您的歷史交易紀錄已匯出</b>\n結算時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
                self.edit_message_text(msg_id, "✅ <b>對帳單已成功匯出！</b>\n請查看下方檔案。")
                time_module.sleep(5)
                if os.path.exists(file_name): os.remove(file_name) 
            except Exception as e: self.edit_message_text(msg_id, f"❌ 匯出失敗：{e}")
        
        elif data_str == "cmd_volume_breakout":
            self.edit_message_text(msg_id, "⏳ <b>開始執行：策略參數掃描</b>\n系統正在計算所有族群的爆量數據，請稍候約 1~2 分鐘...")
            
            def run_vol_analysis():
                try:
                    vol_thread = RunAnalysisTaskThread()
                    vol_thread.run()
                    # 尋找最新產出的 html 檔案
                    list_of_files = glob.glob('./*.html')
                    latest_file = max(list_of_files, key=os.path.getctime) if list_of_files else None
                    
                    if latest_file:
                        with open(latest_file, 'rb') as f:
                            self.session.post(
                                f"https://api.telegram.org/bot{self._get_token()}/sendDocument",
                                data={"chat_id": chat_id, "caption": "✅ <b>分析完成！</b>\n請下載此 HTML 檔案並使用瀏覽器開啟查看。"}, 
                                files={"document": f}
                            )
                except Exception as e:
                    self.send_message(f"❌ 分析失敗: {e}", force=True)
            threading.Thread(target=run_vol_analysis, daemon=True).start()

        elif data_str.startswith("fetch_date_"):
            selected_date = data_str.replace("fetch_date_", "")
            self.edit_message_text(msg_id, f"⏳ <b>開始採集 {selected_date} 的 K 線數據...</b>\n系統於背景執行，請耐心等待。")
            
            def fetch_task():
                try:
                    # 👇 朋友，加上這一行，現場呼叫雲端大腦模組
                    brain = CloudBrainManager() 
                    brain.fetch_history_data(selected_date) 
                    
                    self.send_message(f"✅ <b>{selected_date} 數據採集完成！</b>", force=True)
                except Exception as e:
                    self.send_message(f"❌ 採集失敗: {e}", force=True)
            threading.Thread(target=fetch_task, daemon=True).start()

        elif data_str == "cmd_stop_trading":
            sys_state.stop_trading_flag = True
            self.edit_message_text(msg_id, "⏸️ <b>已發送終止指令！</b>\n系統將在本次掃描迴圈結束後，安全退出盤中監控模式。\n(注意：這不會平倉您目前的庫存)")

        elif data_str in ["tut_phone", "tut_pc"]:
            try:
                res = pd.read_sql(f"SELECT content FROM system_docs WHERE doc_id='{data_str}'", sys_db.conn)
                if not res.empty:
                    return self.edit_message_text(msg_id, res.iloc[0]['content'], {"inline_keyboard": [[{"text": "🔙 關閉指南", "callback_data": "cmd_close_menu"}]]})
            except Exception as e: print(e)
            return self.edit_message_text(msg_id, "⚠️ 讀取教學文件失敗。")

        elif data_str == "cmd_list_groups":
            self.edit_message_text(msg_id, "📁 <b>請選擇要查看的股票族群：</b>", self.get_groups_keyboard())
        elif data_str.startswith("grp_show_"):
            grp_name = data_str.replace("grp_show_", "")
            mat = load_matrix_dict_analysis()
            if grp_name in mat:
                syms = mat[grp_name]
                msg = f"📁 <b>【{grp_name}】</b> (共 {len(syms)} 檔)\n" + "━"*15 + "\n"
                load_twse_name_map() 
                for s in syms:
                    msg += f"• <code>{sn(str(s)).strip()}</code>\n"
                msg += f"\n💡 直接輸入代號即可查看走勢圖"
                kb = {"inline_keyboard": [[{"text": "🔙 返回清單", "callback_data": "cmd_list_groups"}]]}
                self.edit_message_text(msg_id, msg, kb)

        elif data_str.startswith("sim_") or data_str.startswith("trade_"):
            if chat_id not in self.ui_states: return
            st, prefix = self.ui_states[chat_id], data_str.split("_")[0]
            if data_str.endswith("_execute"):
                if prefix == "trade":
                    if getattr(sys_state, 'is_monitoring', False):
                        return self.edit_message_text(msg_id, "✅ <b>盤中監控早已啟動並持續運作中！</b>\n請放心，系統正在為您監控。\n\n💡 欲查看目前持倉，請點擊主選單的「📊 即時持倉監控」。")
                    
                    w, h = st['w'], st['h']
                    h_val = None if h == 270 else h
                    self.edit_message_text(msg_id, f"🚀 <b>啟動盤中監控</b>\n參數：等待 {w}分 / 持有 {'尾盤(F)' if h==270 else str(h)+'分'}\n系統已在背景執行。")
                    threading.Thread(target=start_trading, args=('full', w, h_val), daemon=True).start()
                else:
                    groups = ["所有族群"] + list(load_matrix_dict_analysis().keys())
                    grp = groups[st.get('g_idx', 0) % len(groups)]
                    w, h = st['w'], st['h']
                    self.edit_message_text(msg_id, f"⏳ <b>啟動回測 ({grp})</b>\n等待 {w}分 / 持有 {'尾盤(F)' if h==270 else str(h)+'分'}...")
                    cmd_txt = f"內部回測 {grp} {w} {'F' if h==270 else h}"
                    threading.Thread(target=self._run_quick_backtest, args=(cmd_txt, msg_id), daemon=True).start()
                return
            
            self._update_slider_state(st, data_str)
            title = "🎯 <b>設定自選進場 (回測)</b>" if prefix == "sim" else "▶️ <b>設定盤中監控參數</b>"
            self.edit_message_text(msg_id, f"{title}\n請點擊加減號調整數值：", self.get_slider_keyboard(st, prefix))

        elif data_str.startswith("max_"):
            if chat_id not in self.ui_states: return
            st = self.ui_states[chat_id]
            if data_str == "max_ws_inc": st['ws'] += 1
            elif data_str == "max_ws_dec": st['ws'] = max(1, st['ws'] - 1)
            elif data_str == "max_we_inc": st['we'] += 1
            elif data_str == "max_we_dec": st['we'] = max(st['ws'], st['we'] - 1)
            elif data_str == "max_hs_inc": h=st['hs']; st['hs'] = 5 if h==270 else (270 if h>=120 else h+5)
            elif data_str == "max_hs_dec": h=st['hs']; st['hs'] = 120 if h==270 else (270 if h<=5 else h-5)
            elif data_str == "max_he_inc": h=st['he']; st['he'] = 5 if h==270 else (270 if h>=120 else h+5)
            elif data_str == "max_he_dec": h=st['he']; st['he'] = 120 if h==270 else (270 if h<=5 else h-5)
            elif data_str == "max_g_next": st['g_idx'] += 1
            elif data_str == "max_g_prev": st['g_idx'] -= 1
            elif data_str == "max_execute":
                self.edit_message_text(msg_id, "✅ 參數已鎖定，準備調度運算資源...")
                groups = ["所有族群"] + list(load_matrix_dict_analysis().keys())
                grp = groups[st['g_idx'] % len(groups)]
                cmd_txt = f"內部極大化 {grp} {st['ws']} {st['we']} {st['hs']} {st['he']}"
                threading.Thread(target=self._run_maximize, args=(cmd_txt, msg_id), daemon=True).start()
                return
            self.edit_message_text(msg_id, "🎛️ <b>設定掃描參數</b>\n請點擊加減號調整數值：", self.get_max_builder_keyboard(st))

        elif data_str == "cmd_opt_sim":
            if hasattr(self, 'opt_thread') and self.opt_thread.isRunning():
                return self.send_message("⚠️ <b>掃描正在進行中</b>，請等待完成。", force=True)
            self.edit_message_text(msg_id, "⏳ <b>啟動智能 DTW 門檻掃描...</b>\n此運算較耗時，完成後將回傳報表給您。")
            self.opt_thread = OptimizeSimilarityThread(5, 270)
            dtw_logs = [] 
            def _opt_log(msg):
                try:
                    clean = re.sub(r'<[^>]+>', '', str(msg)).replace('&nbsp;', ' ').replace('&gt;', '>')
                    if clean.strip():
                        print(f"[DTW掃描] {clean}")
                        dtw_logs.append(clean)
                except Exception: pass
            def _on_dtw_finish(*args):
                print("[DTW掃描] ✅ 掃描完成，正在發送結果至手機...")
                res_text = "\n".join(dtw_logs)
                tg_msg = f"✅ <b>智能 DTW 門檻掃描結果</b>\n<pre>{html.escape(res_text)}</pre>"
                self.send_message(tg_msg, force=True)
            self.opt_thread.log_signal.connect(_opt_log, Qt.DirectConnection)
            self.opt_thread.finished_signal.connect(_on_dtw_finish, Qt.DirectConnection)
            self.opt_thread.start()

        elif data_str == "cmd_fetch_data":
            if hasattr(self, 'fetch_thread') and self.fetch_thread.isRunning():
                return self.send_message("⚠️ <b>採集正在進行中</b>，請等待完成。", force=True)
            self.edit_message_text(msg_id, "📥 <b>啟動大數據採集 (預設 5 天)...</b>\n系統正於背景處理，完成後將通知您。")
            self.fetch_thread = FetchSimilarityDataThread(5)
            def _fetch_log(msg):
                try:
                    clean = re.sub(r'<[^>]+>', '', str(msg)).replace('&nbsp;', ' ').replace('&gt;', '>')
                    if clean.strip(): print(f"[數據採集] {clean}")
                except Exception: pass
            def _on_fetch_finish(*args):
                print("\n[數據採集] ✅ 大數據採集完成！已為 DTW 掃描準備好足夠的樣本。")
                self.send_message("✅ <b>大數據採集結束</b>\n已準備好足夠的分析樣本。", force=True)
            self.fetch_thread.log_signal.connect(_fetch_log, Qt.DirectConnection)
            self.fetch_thread.finished_signal.connect(_on_fetch_finish, Qt.DirectConnection)
            self.fetch_thread.start()

        elif data_str == "cmd_avg_high":
            self.edit_message_text(msg_id, "⏳ <b>正在計算平均過高間隔...</b>")
            def _run():
                print("\n[遙控中心] 啟動計算全部族群平均過高...")
                avgs = [avg for g in load_matrix_dict_analysis().keys() if (avg := calculate_average_over_high(g))]
                if avgs:
                    ans = sum(avgs)/len(avgs)
                    self.edit_message_text(msg_id, f"✅ <b>平均過高間隔：</b>\n<code>{ans:.2f}</code> 分鐘")
                    print(f"[遙控中心] ✅ 計算完成！全市場平均過高間隔為: {ans:.2f} 分鐘")
                else:
                    self.edit_message_text(msg_id, "⚠️ 無法計算，請確認數據是否充足。")
            threading.Thread(target=_run, daemon=True).start()

        elif data_str == "cmd_update_kline":
            self.edit_message_text(msg_id, "⏳ <b>啟動 K 線採集引擎...</b>\n請稍候。")
            def _run_kline():
                if not self.task_lock.acquire(blocking=False): return
                try:
                    last_t = 0
                    def _cb(pct, msg_txt):
                        nonlocal last_t
                        now = time_module.time()
                        if now - last_t > 2.0 or pct >= 100:
                            bar = "▓" * (pct // 10) + "░" * (10 - (pct // 10))
                            self.edit_message_text(msg_id, f"🔄 <b>資料更新中...</b>\n\n進度: <code>[{bar}] {pct}%</code>\n狀態: {msg_txt}")
                            last_t = now
                    update_kline_data(tg_progress_cb=_cb)
                    self.edit_message_text(msg_id, "✅ <b>資料更新完成</b>\n進度: <code>[▓▓▓▓▓▓▓▓▓▓] 100%</code>")
                finally: self.task_lock.release()
            threading.Thread(target=_run_kline, daemon=True).start()

        elif data_str == "confirm_close_all":
            self.edit_message_text(msg_id, "✅ <b>已授權，系統正在執行市價平倉...</b>")
            threading.Thread(target=exit_trade_live, daemon=True).start()

    def _exec(self, cmd, message_id=None):
        cmd = cmd.strip()
        mat = load_matrix_dict_analysis()
        
        if re.match(r'^[a-zA-Z0-9]{4,6}$', cmd) or cmd in mat:
            self.send_chat_action("upload_photo")
            def _send_chart():
                try:
                    load_twse_name_map()
                    buf = get_group_chart_bytes(cmd) if cmd in mat else get_stock_chart_bytes(cmd)
                    if buf: 
                        display_name = cmd if cmd in mat else sn(str(cmd)).strip()
                        self.send_photo(buf, caption=f"📊 <b>{display_name} 分析圖</b>")
                    else: 
                        self.send_message(f"⚠️ 無法生成圖表，請確認是否有當日數據。", force=True)
                except Exception as e: pass
            threading.Thread(target=_send_chart, daemon=True).start()
            return

        if cmd in ["/start", "/menu", "開始", "選單"]:
            my_id = self._get_chat_id()
            _now = datetime.now(); _t = _now.time()
            _mkt = "🟢 市場開盤中" if (_now.weekday() < 5 and _now.strftime("%Y%m%d") not in _twse_holidays and time(9,0) <= _t <= time(13,30)) else "🔴 市場休市中"
            msg = (f"▲ <b>REMORA  已上線</b>\n"
                   f"━━━━━━━━━━━━━━━━━━━━\n"
                   f"{_mkt}\n"
                   f"🔑 您的專屬綁定 ID：<code>{my_id}</code>\n\n"
                   f"請點擊下方按鈕開始操作：")
            self.send_message(msg, force=True, reply_markup=self.get_bottom_keyboard())
            tut_kb = {"inline_keyboard": [[
                {"text": "📱 手機端操作指南", "callback_data": "tut_phone"},
                {"text": "💻 電腦端協作指南", "callback_data": "tut_pc"}
            ]]}
            self.send_message("<b>快速上手指南</b>（初次使用建議閱讀）", force=True, reply_markup=tut_kb)
            return

        # ── 按鈕指令對應 (同時兼容新舊按鈕文字) ──
        if cmd in ["▶ 啟動監控", "▶ 啟動盤中監控"]:
            if getattr(sys_state, 'is_monitoring', False):
                return self.send_message("<b>盤中監控正在運行中</b>\n點擊「即時持倉」查看目前庫存。", force=True)
            if not getattr(sys_config, 'esun_id', '') or not getattr(sys_config, 'esun_pwd', ''):
                msg = ("<b>尚未設定登入憑證</b>\n"
                       "請至電腦端主程式「帳戶 API 設定」填寫後，再點擊「啟動監控」。")
                return self.send_message(msg, force=True)
            chat_id = self._get_chat_id()
            self.ui_states[chat_id] = {'w': 5, 'h': 270}
            self.send_message("<b>設定盤中監控參數</b>", force=True,
                              reply_markup=self.get_slider_keyboard(self.ui_states[chat_id], "trade"))

        elif cmd.startswith("登入") or cmd in ["🔑 登入/修改帳戶"]:
            self.send_message(
                "<b>帳戶安全提醒</b>\n"
                "API 密碼請在<b>電腦端主程式</b>的「帳戶 API 設定」中修改，\n"
                "請勿透過通訊軟體傳送敏感憑證。", force=True)

        elif cmd in ["◉ 即時持倉", "📊 即時持倉監控"]:
            with sys_state.lock:
                active_pos = {k: v for k, v in sys_state.open_positions.items() if v.get('filled_shares', 0) > 0}
                if not active_pos:
                    return self.send_message("目前無已成交持倉。", force=True)
                _now = datetime.now(); _t = _now.time()
                _mkt = "🟢 開盤中" if (_now.weekday() < 5 and _now.strftime("%Y%m%d") not in _twse_holidays and time(9,0) <= _t <= time(13,30)) else "🔴 休市中"
                msg = f"<b>即時持倉快照</b>  {_mkt}\n━━━━━━━━━━━━━━\n"
                for sym, p in active_pos.items():
                    name_dict = globals().get('twse_name_map', {})
                    n = name_dict.get(str(sym), "")
                    display = f"{sym} {n}".strip()
                    filled = p.get('filled_shares', 0); target = p.get('target_shares', 0)
                    pnl_str = f"{p.get('entry_price', 0):.2f}"
                    msg += (f"• <code>{html.escape(display)}</code>\n"
                            f"  {filled}/{target} 張  均價 {pnl_str}  停損 {p.get('stop_loss', 0):.2f}\n"
                            f"━━━━━━━━━━━━━━\n")
                self.send_message(msg, force=True)

        elif cmd in ["◈ 盤後分析", "📊 盤後數據與分析"]:
            self.send_message("<b>盤後數據與分析</b>\n請選擇分析項目：",
                              force=True, reply_markup=self.get_analysis_menu())

        elif cmd in ["◎ 自選進場", "🎯 自選進場模式"]:
            chat_id = self._get_chat_id()
            self.ui_states[chat_id] = {'g_idx': 0, 'w': 5, 'h': 270}
            self.send_message("<b>自選進場模式</b>\n請設定參數：",
                              force=True, reply_markup=self.get_slider_keyboard(self.ui_states[chat_id], "sim"))

        elif cmd in ["◆ 量能分析", "🔬 策略參數掃描"]:
            chat_id = self._get_chat_id()
            self.ui_states[chat_id] = {'g_idx': 0, 'ws': 3, 'we': 5, 'hs': 10, 'he': 20}
            self.send_message("<b>策略參數掃描</b>\n請設定掃描範圍：",
                              force=True, reply_markup=self.get_max_builder_keyboard(self.ui_states[chat_id]))

        elif cmd in ["⊞ 族群管理", "📁 管理股票族群"]:
            self.send_message("<b>股票族群管理</b>\n請選擇族群：",
                              force=True, reply_markup=self.get_groups_keyboard())

        elif cmd in ["⊟ 更新K線", "🔄 更新 K 線數據"]:
            if getattr(sys_state, 'trading', False):
                return self.send_message("盤中監控執行中，無法更新資料庫。", force=True)
            keyboard = []
            days_added = 0; i = 0
            while days_added < 7:
                target_date = datetime.now() - timedelta(days=i); i += 1
                if target_date.weekday() < 5:
                    d_str = target_date.strftime("%Y-%m-%d")
                    if keyboard and len(keyboard[-1]) < 2:
                        keyboard[-1].append({"text": f"◷ {d_str}", "callback_data": f"fetch_date_{d_str}"})
                    else:
                        keyboard.append([{"text": f"◷ {d_str}", "callback_data": f"fetch_date_{d_str}"}])
                    days_added += 1
            kb = {"inline_keyboard": keyboard + [[{"text": "✕ 取消", "callback_data": "cmd_close_menu"}]]}
            self.send_message("<b>選擇要下載的交易日</b>（近 7 個交易日）", force=True, reply_markup=kb)

        elif cmd in ["📜 交易紀錄", "📜 歷史交易紀錄"]:
            try:
                df = pd.read_sql("SELECT timestamp, action, symbol, profit FROM trade_logs ORDER BY id DESC LIMIT 15", sys_db.conn)
                if df.empty: return self.send_message("尚無歷史交易紀錄。", force=True)
                msg = "📜  <b>最近 15 筆交易紀錄</b>\n━━━━━━━━━━━━━━\n"
                name_dict = globals().get('twse_name_map', {})
                for _, r in df.iterrows():
                    icon = "▲" if r['profit'] > 0 else "▼" if r['profit'] < 0 else "─"
                    s = r['symbol']; display = f"{s} {name_dict.get(str(s), '')}".strip()
                    msg += f"[{r['timestamp'][11:16]}]  {r['action']}  <code>{display}</code>  {icon}  {r['profit']:.0f}\n"
                kb = {"inline_keyboard": [[{"text": "📥 下載完整 CSV 對帳單", "callback_data": "cmd_export_csv"}]]}
                self.send_message(msg, force=True, reply_markup=kb)
            except Exception: self.send_message("無法讀取資料庫，請稍後重試。", force=True)

        elif cmd in ["📈 走勢圖表", "📈 畫圖查看走勢"]:
            self.send_message("<b>走勢圖查詢</b>\n輸入股票代號或族群名稱以查看圖表。", force=True)

        elif cmd in ["⚙ 參數設定", "⚙️ 參數設定"]:
            msg = (f"⚙  <b>系統參數快照</b>\n━━━━━━━━━━━━━━\n"
                   f"• 單筆資金：<b>{sys_config.capital_per_stock}</b> 萬\n"
                   f"• 手續費 / 折扣 / 稅：{sys_config.transaction_fee}% / {sys_config.transaction_discount}折 / {sys_config.trading_tax}%\n"
                   f"• 發動觀察時間：<b>{sys_config.momentum_minutes}</b> 分\n\n"
                   f"💡 欲修改請輸入：<code>設定 資金 500</code>")
            self.send_message(msg, force=True)

        elif cmd in ["🛑 緊急平倉", "🛑 緊急/手動平倉"]:
            kb = {"inline_keyboard": [
                [{"text": "💥  確認全數市價平倉", "callback_data": "confirm_close_all"}],
                [{"text": "⏸  退出監控模式 (不平倉)", "callback_data": "cmd_stop_trading"}],
                [{"text": "✕  取消", "callback_data": "cmd_close_menu"}]
            ]}
            self.send_message("<b>緊急操作</b>\n請選擇執行方式：", force=True, reply_markup=kb)

        elif cmd.startswith("設定"):
            if getattr(sys_state, 'trading', False):
                return self.send_message("<b>盤中監控執行中</b>\n為保護策略執行，此時無法修改核心參數。", force=True)
            parts = cmd.split()
            if len(parts) >= 3:
                k, v = parts[1], parts[2]
                try:
                    if k == "資金": sys_config.capital_per_stock = int(v)
                    elif k == "發動時間": sys_config.momentum_minutes = int(v)
                    save_settings(); self.send_message(f"✅ {k} 已更新為: {v}", force=True)
                except Exception: pass

    def _run_maximize(self, cmd_text, msg_id):
        parts = cmd_text.split()
        grp, ws, we, hs, he = parts[1], int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
        hold_opts = list(range(hs, he + 1))
        if not self.task_lock.acquire(blocking=False): return self.edit_message_text(msg_id, "⚠️ 系統忙碌中。")
        try:
            mat, dispo = load_matrix_dict_analysis(), load_disposition_stocks()
            _recent = sys_db.get_recent_kline_dates(n=30)
            d_kline, i_kline = load_kline_data(dates=_recent) if _recent else load_kline_data()
            results_df = pd.DataFrame(columns=['等待時間', '持有時間', '總利潤'])
            total_steps, step, last_t = (we - ws + 1) * len(hold_opts), 0, 0
            for w in range(ws, we + 1):
                for h_val in hold_opts:
                    tp_sum = 0
                    if grp != "所有族群":
                        data = initialize_stock_data([s for s in mat.get(grp, []) if s not in dispo], d_kline, i_kline)
                        tp, _, _, _ = process_group_data(data, w, h_val, mat, None, verbose=False)
                        if tp is not None: tp_sum = tp
                    else:
                        for g, s in mat.items():
                            data = initialize_stock_data([x for x in s if x not in dispo], d_kline, i_kline)
                            tp, _, _, _ = process_group_data(data, w, h_val, mat, None, verbose=False)
                            if tp is not None: tp_sum += tp
                    results_df = pd.concat([results_df, pd.DataFrame([{'等待時間': w, '持有時間': h_val, '總利潤': float(tp_sum)}])], ignore_index=True)
                    step += 1
                    now = time_module.time()
                    pct = int((step / total_steps) * 100)
                    if now - last_t > 2.5 or pct == 100:
                        bar = "▓" * (pct // 10) + "░" * (10 - (pct // 10))
                        self.edit_message_text(msg_id, f"💻 <b>運算最佳化中 ({grp})</b>\n\n進度: <code>[{bar}] {pct}%</code>\n測試: 等待 {w}分 / 持有 {h_val}分")
                        last_t = now
            if not results_df.empty:
                best = results_df.loc[results_df['總利潤'].idxmax()]
                res_msg = f"🏆 <b>最佳化完成 ({grp})</b>\n━━━━━━━━━━━━━━\n🥇 <b>最佳組合：</b>等 {best['等待時間']}分 / 持 {best['持有時間']}分\n💰 利潤：<b>{int(best['總利潤'])}</b> 元\n\n📊 <b>排行榜：</b>\n"
                for r, (_, row) in enumerate(results_df.sort_values(by='總利潤', ascending=False).head(3).iterrows(), 1):
                    res_msg += f"{r}. 等{row['等待時間']:>2} / 持{str(row['持有時間']):>2} ➔ {int(row['總利潤'])}元\n"
                self.edit_message_text(msg_id, res_msg)
            else: self.edit_message_text(msg_id, "⚠️ 無任何交易產生。")
        finally: self.task_lock.release()

    def _run_quick_backtest(self, cmd_text, msg_id=None):
        w, h, target_group = 3, None, None
        parts = cmd_text.replace("🧪", "").strip().split()
        if len(parts) >= 4 and parts[0] == "內部回測":
            target_group = parts[1]
            try: w = int(parts[2])
            except Exception: pass
            try: h = None if parts[3].upper() == 'F' else int(parts[3])
            except Exception: pass
        def _notify(txt, kb=None):
            if msg_id: self.edit_message_text(msg_id, txt, kb)
            else: self.send_message(txt, force=True, reply_markup=kb)
        if not msg_id: _notify(f"⏳ <b>啟動回測...</b>\n⚙️ 參數 ➔ 等待 {w} 分 / 持有 {'尾盤' if h is None else str(h)+'分'}")
        if not self.task_lock.acquire(blocking=False): return _notify("⚠️ 系統正在執行其他任務，請稍後再試。")
        try:
            mat = load_matrix_dict_analysis()
            _recent = sys_db.get_recent_kline_dates(n=30)
            d_kline, i_kline = load_kline_data(dates=_recent) if _recent else load_kline_data()
            dispo = load_disposition_stocks()
            if not mat or not i_kline: return _notify("⚠️ <b>回測失敗：</b>請先在軟體內「更新 K 線數據」。")
            tp_sum, tg_capital, all_trades = 0, 0, []
            mat_to_run = {target_group: mat[target_group]} if target_group and target_group != "所有族群" and target_group in mat else mat
            total_groups = len(mat_to_run)
            for i, (g, s) in enumerate(mat_to_run.items()):
                data = initialize_stock_data([x for x in s if x not in dispo], d_kline, i_kline)
                tp, tc, t_hist, _ = process_group_data(data, w, h, mat, None, verbose=False)
                if tp is not None: tp_sum += tp; tg_capital += tc; all_trades.extend(t_hist)
                pct = int(((i + 1) / total_groups) * 100)
                if msg_id and total_groups > 1 and (pct % 20 == 0 or pct == 100):
                    bar = "▓" * (pct // 10) + "░" * (10 - (pct // 10))
                    _notify(f"💻 <b>雲端回測運算中...</b>\n\n進度: <code>[{bar}] {pct}%</code>\n正在計算: 【{g}】")
            if tg_capital > 0:
                total_rate = tp_sum / tg_capital * 100
                td = "".join([f"• <code>{html.escape(sn(t['symbol']))}</code> | {'🔴' if t['profit']>0 else '🟢'} {int(t['profit'])}元\n" for t in all_trades])
                if len(td) > 2500: td = td[:2500] + "\n... (資料過多已省略)"
                msg = f"📊 <b>全市場回測報告</b>\n━━━━━━━━━━━━━━\n⚙️ 參數：等待 {w} 分 / 持有 {'尾盤' if h is None else str(h)+'分'}\n💰 總利潤：<b>{int(tp_sum)}</b> 元\n📈 報酬率：<b>{total_rate:.2f}%</b>\n🤝 交易筆數：<b>{len(all_trades)}</b> 筆\n━━━━━━━━━━━━━━\n<b>【進場標的清單】</b>\n{td}\n"
                _notify(msg)
            else: _notify(f"📊 <b>今日回測報告</b>\n今日無任何符合進場條件的標的。")
        finally: self.task_lock.release()

tg_bot = CloudBrainManager()

# ==================== 核心邏輯區 ====================
def calculate_dtw_pearson(df_lead, df_follow, window_start, window_end):
    if isinstance(window_start, str):
        try: window_start = pd.to_datetime(window_start, format="%H:%M:%S").time()
        except Exception: window_start = pd.to_datetime(window_start).time()
    if isinstance(window_end, str):
        try: window_end = pd.to_datetime(window_end, format="%H:%M:%S").time()
        except Exception: window_end = pd.to_datetime(window_end).time()
    sub_l = df_lead[(df_lead['time'] >= window_start) & (df_lead['time'] <= window_end)].copy()
    sub_f = df_follow[(df_follow['time'] >= window_start) & (df_follow['time'] <= window_end)].copy()
    if len(sub_l) < 3 or len(sub_f) < 3: return 0.0
    merged = pd.merge(sub_l[['time', 'high', 'low', 'close']], sub_f[['time', 'high', 'low', 'close']], on='time', suffixes=('_l', '_f'))
    if len(merged) < 3: return 0.0
    tp_l = (merged['high_l'] + merged['low_l'] + merged['close_l']) / 3
    tp_f = (merged['high_f'] + merged['low_f'] + merged['close_f']) / 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        correlation = tp_l.corr(tp_f)
    return 0.0 if pd.isna(correlation) else correlation

def load_target_symbols():
    try:
        groups = load_matrix_dict_analysis()
        symbols = {str(s) for stocks in groups.values() for s in stocks}
        return list(symbols), groups
    except Exception: return [], {}

def get_stop_loss_config(price):
    if price < 10: return sys_config.below_50, 0.01
    elif price < 50: return sys_config.below_50, 0.05
    elif price < 100: return sys_config.price_gap_50_to_100, 0.1
    elif price < 500: return sys_config.price_gap_100_to_500, 0.5
    elif price < 1000: return sys_config.price_gap_500_to_1000, 1
    else: return sys_config.price_gap_above_1000, 5

def get_max_risk_for_price(price):
    """根據股票價位回傳最大允許停損金額（元/張），對應設定中的 price_gap_* 欄位"""
    return get_stop_loss_config(price)[0]

def get_tick_price(price: float, tick_offset: int) -> float:
    """
    計算台股跳動單位穿價後的價格。
    tick_offset: 負數代表往下穿(空單更好成交)，正數代表往上穿
    """
    ticks = [
        (0, 10, 0.01), (10, 50, 0.05), (50, 100, 0.1),
        (100, 500, 0.5), (500, 1000, 1.0), (1000, float('inf'), 5.0)
    ]
    curr_p = float(price)
    step = 1 if tick_offset > 0 else -1
    
    for _ in range(abs(tick_offset)):
        current_tick = 0.01
        for lower, upper, t_size in ticks:
            # 處理恰好在整數邊界的狀況 (例如 50.0 往下穿應跳 0.05)
            if step == 1 and lower <= curr_p < upper:
                current_tick = t_size; break
            elif step == -1 and lower < curr_p <= upper:
                current_tick = t_size; break
                
        curr_p += step * current_tick
        curr_p = round(curr_p, 2)
        
    return max(curr_p, 0.01) # 價格最低不能為負數

def show_exit_menu(): print("💡 提示：在介面中，請直接點選左側面板的【🛑 緊急/手動平倉】按鈕")

def load_nb_matrix_dict():
    if os.path.exists('nb_matrix_dict.json'):
        d = json.load(open('nb_matrix_dict.json', 'r', encoding='utf-8'))
        sys_db.save_state('nb_matrix_dict', d)
        os.remove('nb_matrix_dict.json')
    return sys_db.load_state('nb_matrix_dict', default_value={})

def save_nb_matrix_dict(d):
    sys_db.save_state('nb_matrix_dict', d)

# 讀取族群字典的工具函數（直接從 matrix_dict_analysis 包裝）
def load_group_symbols():
    raw = load_matrix_dict_analysis()
    return {"consolidated_symbols": {grp: [str(s) for s in syms] for grp, syms in raw.items()}} if raw else {"consolidated_symbols": {}}

# =============================================================
# v1.9.8.6 戰略三核心：全域高級終端入口 (取代舊的 HTML 生成函數)
# =============================================================
term_win_instance = None  # 保留相容性
_main_window_ref = None   # 全域主視窗引用（分頁內嵌用）

def _show_terminal_window(temp_file, mode, actual_date):
    """主執行緒專用：在主視窗分頁中載入分析終端（不再開新視窗）"""
    global term_win_instance, _main_window_ref
    mw = _main_window_ref
    if mw is not None and hasattr(mw, '_ensure_tab'):
        key = 'terminal'
        if key in mw._tab_pages:
            page = mw._tab_pages[key]
            idx = mw.tabs.indexOf(page)
            if idx >= 0:
                # 更新現有終端
                for child in page.findChildren(QWebEngineView):
                    child.load(QUrl.fromLocalFile(temp_file))
                    break
                mw.tabs.setCurrentIndex(idx)
                mw.tabs.setTabText(idx, f'分析終端 - {actual_date}')
                return
            else:
                del mw._tab_pages[key]
        # 建立新的分析終端分頁
        tw = AdvancedTerminalWindow(temp_file=temp_file, mode=mode, date_str=actual_date)
        # 取出 central widget 嵌入分頁
        page = QWidget()
        lo = QVBoxLayout(page); lo.setContentsMargins(0, 0, 0, 0)
        browser = tw.browser
        browser.setParent(page)
        lo.addWidget(browser)
        # 保留 bridge/channel 引用
        page._tw = tw
        mw._tab_pages[key] = page
        idx = mw.tabs.addTab(page, f'分析終端 - {actual_date}')
        mw.tabs.setCurrentIndex(idx)
        term_win_instance = tw  # 保持相容性
    else:
        # 退回舊模式（獨立視窗）
        if term_win_instance is not None and term_win_instance.isVisible():
            term_win_instance.load_from_file(temp_file, mode, actual_date)
            term_win_instance.raise_(); term_win_instance.activateWindow()
        else:
            term_win_instance = AdvancedTerminalWindow(temp_file=temp_file, mode=mode, date_str=actual_date)
            term_win_instance.show()


def open_advanced_terminal_v196(initial_trades, initial_events, date_str="", mode="backtest", target_symbol=None):
    """v1.9.8.6 修正版：支援多日大數據傳輸，並完美聯動前端切換
    (供觀察清單等輕量同步場景使用；多日回測請透過 PrepareTerminalDataThread 非同步呼叫)"""
    global term_win_instance

    # 🚀 支援多日歷史 K 線抓取：直接讀取所有選定的天數
    target_dates = sys_db.load_state('last_fetched_dates')
    if not target_dates:
        actual_date = date_str if date_str else sys_db.load_state('last_fetched_date')
        target_dates = [actual_date] if actual_date else []

    actual_date = target_dates[0] if target_dates else ""

    # 🚀 數據抓取邏輯 (一次抓出所有天數的 K 線！)
    df_all = pd.DataFrame()
    if target_dates:
        try:
            dates_str = "','".join(target_dates)
            sql = f"SELECT * FROM intraday_kline_history WHERE date IN ('{dates_str}') ORDER BY date ASC, time ASC"
            df_all = pd.read_sql(sql, sys_db.conn)
        except Exception as e: logger.error(f"讀取 K 線歷史失敗: {e}")

    grouped_data = {}
    if not df_all.empty:
        df_all['symbol'] = df_all['symbol'].astype(str)
        for sym, group in df_all.groupby('symbol'):
            o = pd.to_numeric(group['open'], errors='coerce').tolist()
            c = pd.to_numeric(group['close'], errors='coerce').tolist()
            first_p = next((x for x in o if x > 0), 1.0)
            grouped_data[sym] = {
                'date': group['date'].tolist(),
                'time': [str(t)[:5] for t in group['time'].tolist()],
                'open': o, 'high': pd.to_numeric(group['high']).tolist(),
                'low': pd.to_numeric(group['low']).tolist(), 'close': c,
                'volume': pd.to_numeric(group['volume']).tolist(),
                'pct_change': [((cv - first_p) / first_p * 100) for cv in c]
            }

    # 準備 Payload
    pnl_summary = {}
    active_symbols = set()
    if initial_trades:
        for t in initial_trades:
            s = str(t['symbol']); active_symbols.add(s)
            pnl_summary[s] = pnl_summary.get(s, 0) + t.get('profit', 0)

    mat = load_matrix_dict_analysis()
    skeleton = {}
    for grp, syms in mat.items():
        skeleton[grp] = {}
        for s in syms:
            s_str = str(s)
            skeleton[grp][s_str] = grouped_data.get(s_str, {'date': [], 'time': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': [], 'pct_change': []})
            skeleton[grp][s_str]['name'] = sn(s_str).replace(s_str, '').strip()

    full_payload = {
        "chartData": skeleton, "targetCode": str(target_symbol) if target_symbol else "",
        "allTrades": initial_trades, "allEvents": initial_events,
        "pnlSummary": pnl_summary, "activeGroups": list(mat.keys())
    }
    # 同步建構 HTML 並寫磁碟（觀察清單為單股輕量場景，不影響 UI）
    html_content = AdvancedTerminalWindow.build_html_template(full_payload, mode, "3", "F", actual_date)
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, "dashboard_196.html")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    _show_terminal_window(temp_file, mode, actual_date)

# ==================== PyQt5 訊號與重導向 ====================
class EmittingStream(QObject):
    textWritten = pyqtSignal(str)
    def write(self, text): 
        try:
            if text is None: return
            text_str = str(text)
            # 過濾特定 Shioaji 噪音
            if "Response Code: 0" in text_str or "Session up" in text_str: return
            self.textWritten.emit(text_str)
        except Exception: pass
    def flush(self): pass

class SignalDispatcher(QObject):
    portfolio_updated = pyqtSignal(list)
    position_updated = pyqtSignal(dict)  # 🔧 即時持倉變動信號
    progress_updated = pyqtSignal(int, str)
    progress_visible = pyqtSignal(bool)
    plot_equity_curve = pyqtSignal(object)
    show_analysis_window = pyqtSignal(str)
    console_log = pyqtSignal(str)   # 轉發終端輸出給額外訂閱者（如 LiveTradingPanel）

ui_dispatcher = SignalDispatcher()
cached_portfolio_data = []

class PrepareTerminalDataThread(QThread):
    """背景執行緒：將 K 線 DB 查詢與 payload 建構移出主執行緒，避免多日回測白屏卡死"""
    ready = pyqtSignal(object)

    def __init__(self, all_trades, all_events, date_str, mode, parent=None):
        super().__init__(parent)
        self.all_trades = all_trades
        self.all_events = all_events
        self.date_str = date_str
        self.mode = mode

    def run(self):
        target_dates = sys_db.load_state('last_fetched_dates')
        if not target_dates:
            actual_date = self.date_str if self.date_str else sys_db.load_state('last_fetched_date')
            target_dates = [actual_date] if actual_date else []
        actual_date = target_dates[0] if target_dates else ""

        df_all = pd.DataFrame()
        if target_dates:
            try:
                dates_str = "','".join(target_dates)
                sql = f"SELECT * FROM intraday_kline_history WHERE date IN ('{dates_str}') ORDER BY date ASC, time ASC"
                with sys_db.db_lock:
                    df_all = pd.read_sql(sql, sys_db.conn)
            except Exception as e:
                logger.error(f"讀取 K 線歷史失敗: {e}")

        grouped_data = {}
        if not df_all.empty:
            df_all['symbol'] = df_all['symbol'].astype(str)
            for sym, group in df_all.groupby('symbol'):
                o = pd.to_numeric(group['open'], errors='coerce').tolist()
                c = pd.to_numeric(group['close'], errors='coerce').tolist()
                first_p = next((x for x in o if x > 0), 1.0)
                grouped_data[sym] = {
                    'date': group['date'].tolist(),
                    'time': [str(t)[:5] for t in group['time'].tolist()],
                    'open': o, 'high': pd.to_numeric(group['high']).tolist(),
                    'low': pd.to_numeric(group['low']).tolist(), 'close': c,
                    'volume': pd.to_numeric(group['volume']).tolist(),
                    'pct_change': [((cv - first_p) / first_p * 100) for cv in c]
                }

        pnl_summary = {}
        if self.all_trades:
            for t in self.all_trades:
                s = str(t['symbol'])
                pnl_summary[s] = pnl_summary.get(s, 0) + t.get('profit', 0)

        mat = load_matrix_dict_analysis()
        skeleton = {}
        for grp, syms in mat.items():
            skeleton[grp] = {}
            for s in syms:
                s_str = str(s)
                skeleton[grp][s_str] = grouped_data.get(s_str, {'date': [], 'time': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': [], 'pct_change': []})
                skeleton[grp][s_str]['name'] = sn(s_str).replace(s_str, '').strip()

        # 自動選取第一筆交易的股票作為預設顯示
        auto_code = ""
        if self.all_trades:
            auto_code = str(self.all_trades[0].get('symbol', ''))

        full_payload = {
            "chartData": skeleton, "targetCode": auto_code,
            "allTrades": self.all_trades, "allEvents": self.all_events,
            "pnlSummary": pnl_summary, "activeGroups": list(mat.keys())
        }

        # 在背景執行緒完成 JSON 序列化 + HTML 建構 + 寫磁碟，主執行緒不再凍結
        html_content = AdvancedTerminalWindow.build_html_template(full_payload, self.mode, "3", "F", actual_date)
        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, "dashboard_196.html")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # 同時在背景執行緒建構資金流動圖 HTML
        cashflow_html = ""
        if self.all_trades:
            try:
                cashflow_html = build_cashflow_html(self.all_trades)
            except Exception:
                pass

        self.ready.emit({'temp_file': temp_file, 'actual_date': actual_date, 'mode': self.mode, 'cashflow_html': cashflow_html})

STOCK_NAME_MAP = {}
def load_twse_name_map(json_path="twse_stocks_by_market.json"):
    global STOCK_NAME_MAP
    if STOCK_NAME_MAP: return
    try:
        if os.path.exists(json_path):
            STOCK_NAME_MAP = json.load(open(json_path, "r", encoding="utf-8")); return
        def fetch_isin(mode):
            r = requests.get(f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            # 將 big5 升級為 cp950，能解析更多繁體罕見字
            r.encoding = "cp950" 
            return {tds[0].text.strip()[:4]: tds[0].text.strip().split("\u3000", 1)[1] if "\u3000" in tds[0].text.strip() else tds[0].text.strip()[4:] for tr in BeautifulSoup(r.text, "lxml").select("table tr")[1:] if (tds := tr.find_all("td")) and tds[0].text.strip()[:4].isdigit()}
        STOCK_NAME_MAP = {"TSE": fetch_isin("2"), "OTC": fetch_isin("4")}
        json.dump(STOCK_NAME_MAP, open(json_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception: STOCK_NAME_MAP = {}

def get_stock_name(code):
    return next((STOCK_NAME_MAP[m][code] for m in ["TSE", "OTC"] if code in STOCK_NAME_MAP.get(m, {})), "")

def sn(sym): 
    # 過濾 Telegram HTML 敏感字元與亂碼，防止推播失敗
    clean_name = get_stock_name(sym).replace('\uFFFD', '').replace('<', '').replace('>', '').replace('&', '及')
    return f"{sym} {clean_name}"

def init_esun_client():
    global ESUN_LOGIN_PWD, ESUN_CERT_PWD
    if not os.path.exists('config.ini'): sys.exit(f"{RED}❌ 找不到玉山 API 設定檔{RESET}")
    
    if not ESUN_LOGIN_PWD: ESUN_LOGIN_PWD = "dummy_login"
    if not ESUN_CERT_PWD: ESUN_CERT_PWD = "dummy_cert"
    
    try:
        config = ConfigParser(); config.read('config.ini', encoding='utf-8-sig') 
        import unittest.mock as mock
        with mock.patch('getpass.getpass', side_effect=[ESUN_LOGIN_PWD, ESUN_CERT_PWD]):
            sdk = EsunMarketdata(config)
            sdk.login() 
            
        # 加入嚴格防呆，等待 1.5 秒讓憑證連線，並檢查能否獲取帳戶物件
        time_module.sleep(1.5)
        if not sdk.rest_client:
            raise Exception("登入被拒絕。請檢查您的身分證、密碼是否正確，或憑證可能已過期。")
            
        return sdk.rest_client
    except Exception as e: 
        sys.exit(f"{RED}玉山 API 連線失敗：{e}{RESET}")

# 域 API 速率限制器 (確保跨線程/跨次數點擊不會超過 Shioaji 限制)
class APIRateLimiter:
    def __init__(self, max_calls=550, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time_module.time()
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    logger.warning(f"API 呼叫已達 {self.max_calls} 次，自動暫停 {sleep_time:.1f} 秒")
                    time_module.sleep(sleep_time)
                self.calls = []
            self.calls.append(time_module.time())

global_rate_limiter = APIRateLimiter(max_calls=550, period=60)
historical_rate_limiter = APIRateLimiter(max_calls=550, period=60)

def safe_esun_api_call(api_func, max_retries=3, **kwargs):
    for attempt in range(max_retries + 1):
        try: 
            res = api_func(**kwargs)
            if isinstance(res, dict) and (res.get('statusCode') == 429 or 'Rate limit' in str(res)): raise Exception("429 Rate limit")
            return res
        except Exception as e:
            if any(x in str(e) for x in ["429", "Too Many Requests", "Rate limit", "502", "503", "504"]):
                if attempt < max_retries: time_module.sleep(2 ** attempt) 
                else: return None
            else: return None
    return None

def _reconnect_shioaji_if_needed():
    logger.warning("偵測到 Shioaji 異常，重連中...")
    print(f"{YELLOW}⚠️ 偵測到 Shioaji 異常，重連中...{RESET}")
    try:
        _is_live = getattr(sys_config, 'live_trading_mode', False)
        _ak = shioaji_logic.LIVE_API_KEY if _is_live else shioaji_logic.TEST_API_KEY
        _as = shioaji_logic.LIVE_API_SECRET if _is_live else shioaji_logic.TEST_API_SECRET
        sys_state.api.login(api_key=_ak, secret_key=_as)
        sys_state.api.activate_ca(ca_path=shioaji_logic.CA_CERT_PATH, ca_passwd=shioaji_logic.CA_PASSWORD)
        sys_state.api.set_order_callback(order_callback)  # 🔧 重連後補註冊 callback
        time_module.sleep(2); sys_state.to = tp.TouchOrderExecutor(sys_state.api)
        print(f"{GREEN}✅ Shioaji 重連成功！{RESET}")
    except Exception as e:
        logger.error(f"Shioaji 重連失敗: {e}")
        print(f"{RED}❌ 重連失敗: {e}{RESET}")

def safe_place_order(api_instance, contract, order, max_retries=1):
    for attempt in range(max_retries + 1):
        try: 
            # 嘗試送出委託
            trade = api_instance.place_order(contract, order)
            return trade
        except Exception as e:
            # 攔截錯誤，印出警告但不引爆程式
            error_msg = str(e)
            logger.error(f"[下單異常攔截] 第 {attempt+1} 次嘗試失敗: {error_msg}")
            print(f"⚠️ [下單異常攔截] 第 {attempt+1} 次嘗試失敗: {error_msg}")
            
            if attempt < max_retries:
                # 簡單等待一下再重試，避免連續撞擊 API 限制
                time_module.sleep(1)
                _reconnect_shioaji_if_needed()
            else:
                # 關鍵修復：這裡絕對不能用 raise e，改為 return None
                print(f"🚫 [下單放棄] 已達最大重試次數，略過此筆委託以保護程式運行。")
                return None

def safe_add_touch_condition(to_instance, tcond, max_retries=1):
    for attempt in range(max_retries + 1):
        try: return to_instance.add_condition(tcond)
        except Exception as e:
            if attempt < max_retries: _reconnect_shioaji_if_needed()
            else: logger.error(f"觸價單註冊失敗: {e}")
    return None

def safe_delete_touch_condition(to_instance, cond, max_retries=1):
    for attempt in range(max_retries + 1):
        try: return to_instance.delete_condition(cond)
        except Exception:
            if attempt < max_retries: _reconnect_shioaji_if_needed()

def calculate_pct_increase_and_highest(current_candle, historical_candles):
    """
    計算給定 K 線的分鐘動能與當日最高價
    """
    # 1. 處理最高價 (highest)
    if historical_candles:
        current_candle['highest'] = max(float(current_candle['high']), max([float(c.get('highest', c['high'])) for c in historical_candles]))
    else:
        current_candle['highest'] = current_candle['high']
        
    # 2. 處理漲幅 (rise)
    if current_candle.get('昨日收盤價', 0) > 0:
        current_candle['rise'] = (current_candle['close'] - current_candle['昨日收盤價']) / current_candle['昨日收盤價'] * 100
        # 強制四捨五入至小數點後兩位，確保 Live 與 History 完美對齊
        current_candle['rise'] = round(float(current_candle['rise']), 2) 
    else:
        current_candle['rise'] = 0.0

    # 3. 處理發動動能 (pct_increase)
    if len(historical_candles) >= 1:
        # 動態套用老闆設定的「發動動能觀察時間(momentum_minutes)」，預設往回看 1 根
        window = getattr(sys_config, 'momentum_minutes', 1)
        compare_index = max(0, len(historical_candles) - window)
        prev_candle = historical_candles[compare_index]
        
        # 動能 = 當下漲幅 - 觀察起點的漲幅
        current_candle['pct_increase'] = current_candle['rise'] - prev_candle.get('rise', 0.0)
        # 強制四捨五入至小數點後兩位
        current_candle['pct_increase'] = round(float(current_candle['pct_increase']), 2) 
    else:
        # 9:00 第一根 K 棒，上一根(昨日收盤)漲幅為0，因此動能直接等於當下漲幅(rise)
        current_candle['pct_increase'] = current_candle['rise']
        
    return current_candle

def calculate_limit_up_price(close_price):
    lu = close_price * 1.10
    return (lu // (0.01 if lu < 10 else 0.05 if lu < 50 else 0.1 if lu < 100 else 0.5 if lu < 500 else 1 if lu < 1000 else 5)) * (0.01 if lu < 10 else 0.05 if lu < 50 else 0.1 if lu < 100 else 0.5 if lu < 500 else 1 if lu < 1000 else 5)

def truncate_to_two_decimals(value):
    """
    (統一改為與回測資料庫一致的無條件捨去邏輯，消滅 0.01 的檔位誤差)
    """
    try:
        if value is None or pd.isna(value): return value
        return math.floor(float(value) * 100) / 100.0
    except (ValueError, TypeError):
        return value

# 專門用來抓取「歷史」資料的引擎
def fetch_intraday_data(client, symbol, trading_day, yesterday_close_price, start_time=None, end_time=None):
    historical_rate_limiter.wait_if_needed()
    
    # 嚴格執行 1.05 秒等待，確保符合玉山 API 歷史行情 60/min 的限制
    time_module.sleep(1.05)
    
    try:
        _from = datetime.strptime(f"{trading_day} {start_time or '09:00'}", "%Y-%m-%d %H:%M")
        to_dt = datetime.strptime(f"{trading_day} {end_time or '13:30'}", "%Y-%m-%d %H:%M")

        # 將 intraday 改為 historical，並透過 from 與 to 指定歷史日期
        api_params = {
            "symbol": symbol, 
            "timeframe": "1", 
            "from": trading_day, 
            "to": trading_day
        }
        candles_rsp = safe_esun_api_call(client.stock.historical.candles, **api_params)

        if not candles_rsp or 'data' not in candles_rsp: return pd.DataFrame()
        df = pd.DataFrame(candles_rsp['data'])
        if df.empty or 'volume' not in df.columns: return pd.DataFrame()

        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

        # 套用與實戰相同的時區對齊機制，徹底解決 0 量與空資料過濾問題
        df['datetime'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert('Asia/Taipei').dt.tz_localize(None).dt.floor('min')
        df.set_index('datetime', inplace=True)
        df = df[~df.index.duplicated(keep='last')]

        orig = df.reset_index()[['datetime', 'volume']].rename(columns={'volume': 'orig_volume'})
        df = df.reindex(pd.date_range(start=_from, end=to_dt, freq='1min')).reset_index().rename(columns={'index': 'datetime'})
        df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
        df['time'] = df['datetime'].dt.strftime('%H:%M:%S')
        df = pd.merge(df, orig, how='left', on='datetime')

        for col in ['open', 'high', 'low', 'close']:
            vals, last_v = df[col].to_numpy(), yesterday_close_price
            for i in range(len(vals)):
                v, c = df.at[i, 'volume'], df.at[i, 'close']
                if v > 0 and not pd.isna(c): last_v = c
                if pd.isna(vals[i]) or v == 0: vals[i] = last_v
            df[col] = vals

        df['volume'] = df['orig_volume'].fillna(0)
        df['symbol'] = symbol
        df['昨日收盤價'] = yesterday_close_price
        df['漲停價'] = truncate_to_two_decimals(calculate_limit_up_price(yesterday_close_price))
        df[['symbol', '昨日收盤價', '漲停價']] = df[['symbol', '昨日收盤價', '漲停價']].ffill().bfill()
        df['rise'] = (df['close'] - df['昨日收盤價']) / df['昨日收盤價'] * 100
        df['highest'] = df['high'].cummax().fillna(yesterday_close_price)

        return df[['symbol', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', '昨日收盤價', '漲停價', 'rise', 'highest']]

    except Exception as e:
        logger.error(f"抓取 {symbol} 歷史分K失敗: {e}")
        return pd.DataFrame()

def fetch_realtime_intraday_data(client, symbol, trading_day, yesterday_close_price, start_time=None, end_time=None):
    global_rate_limiter.wait_if_needed() # 確保使用 550 極速
    try:
        _from = datetime.strptime(f"{trading_day} {start_time or '09:00'}", "%Y-%m-%d %H:%M")
        to_dt = datetime.strptime(f"{trading_day} {end_time or '13:30'}", "%Y-%m-%d %H:%M")
        
        # 嚴格依照 1.9.3.2 寫法：不傳入 date 參數，讓它純粹抓當日盤中
        candles_rsp = safe_esun_api_call(client.stock.intraday.candles, symbol=symbol, timeframe='1')
        
        if not candles_rsp or 'data' not in candles_rsp: return pd.DataFrame()
        df = pd.DataFrame(candles_rsp['data'])
        if df.empty or 'volume' not in df.columns: return pd.DataFrame()
        
        # 解決 0 量問題：退回最原始的時間對齊法，杜絕 UTC 偏移
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None).dt.floor('min')
        df.set_index('datetime', inplace=True)
        
        orig = df.reset_index()[['datetime', 'volume']].rename(columns={'volume': 'orig_volume'})
        df = df.reindex(pd.date_range(start=_from, end=to_dt, freq='1min')).reset_index().rename(columns={'index': 'datetime'})
        df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
        df['time'] = df['datetime'].dt.strftime('%H:%M:%S')
        df = pd.merge(df, orig, how='left', on='datetime')
        
        for col in ['open', 'high', 'low', 'close']:
            vals, last_v = df[col].to_numpy(), yesterday_close_price
            for i in range(len(vals)):
                v, c = df.at[i, 'volume'], df.at[i, 'close']
                if v > 0 and not pd.isna(c): last_v = c
                if pd.isna(vals[i]) or v == 0: vals[i] = last_v
            df[col] = vals
            
        df['volume'] = df['orig_volume'].fillna(0)
        df['symbol'] = symbol
        df['昨日收盤價'] = yesterday_close_price
        df['漲停價'] = truncate_to_two_decimals(calculate_limit_up_price(yesterday_close_price))
        df[['symbol', '昨日收盤價', '漲停價']] = df[['symbol', '昨日收盤價', '漲停價']].ffill().bfill()
        df['rise'] = (df['close'] - df['昨日收盤價']) / df['昨日收盤價'] * 100
        df['highest'] = df['high'].cummax().fillna(yesterday_close_price)
        
        return df[['symbol', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', '昨日收盤價', '漲停價', 'rise', 'highest']]
        
    except Exception as e:
        logger.error(f"抓取 {symbol} 即時分K失敗: {e}")
        return pd.DataFrame()

def fetch_daily_kline_data(client, symbol, days=2, end_date=None):
    historical_rate_limiter.wait_if_needed()
    
    # 嚴格執行 1.05 秒等待，確保符合 60/min 限制
    time_module.sleep(1.05)
    
    if end_date is None: end_date = get_recent_trading_day()
    try:
        data = safe_esun_api_call(client.stock.historical.candles, **{"symbol": symbol, "from": (end_date - timedelta(days=days)).strftime('%Y-%m-%d'), "to": end_date.strftime('%Y-%m-%d')})
        if data and 'data' in data and data['data']: return pd.DataFrame(data['data'])
    except Exception as e: 
        logger.error(f"抓取 {symbol} 歷史日K失敗: {e}")
        pass
    return pd.DataFrame()

def get_valid_trading_day(client, symbols_to_analyze):
    base_date = get_recent_trading_day()
    
    while True:
        target_date_str = base_date.strftime('%Y-%m-%d')
        ui_dispatcher.progress_updated.emit(5, f"🔍 抽樣檢驗 {target_date_str} 是否為假日...")
        
        sample_size = min(10, len(symbols_to_analyze))
        if sample_size == 0: return base_date
            
        sampled_stocks = random.sample(symbols_to_analyze, sample_size)
        volumes = []
        
        for sym in sampled_stocks:
            try:
                df = fetch_daily_kline_data(client, sym, days=5, end_date=base_date)
                if not df.empty and 'date' in df.columns:
                    target_row = df[df['date'] == target_date_str]
                    if not target_row.empty:
                        vol = pd.to_numeric(target_row.iloc[0]['volume'], errors='coerce')
                        volumes.append(vol if not pd.isna(vol) else 0)
                    else:
                        volumes.append(0) # 沒這天的資料代表沒開盤
                else:
                    volumes.append(0)
            except Exception:
                volumes.append(0)

        # 使用 T 分配信賴區間法判斷是否開盤
        is_holiday = True
        if len(volumes) >= 2:
            mean_vol = np.mean(volumes)
            std_vol = np.std(volumes, ddof=1)
            
            if std_vol == 0:
                upper_bound = mean_vol
            else:
                n = len(volumes)
                t_crit = st.t.ppf(0.975, n-1) # 95% 信賴水準
                upper_bound = mean_vol + t_crit * (std_vol / np.sqrt(n))
                
            # 如果 95% 信賴區間上限大於 10 張，判定為有效開盤日
            if upper_bound > 10: 
                is_holiday = False
        else:
            if sum(volumes) > 0: is_holiday = False
            
        if not is_holiday:
            print(f"✅ {target_date_str} 統計檢驗通過，為有效開盤日！")
            return base_date
        
        logger.info(f"T分配判定 {target_date_str} 為國定假日(預期成交量趨近0)，往前搜尋")
        print(f"⚠️ T分配判定 {target_date_str} 為國定假日(預期成交量趨近0)，往前搜尋...")
        ui_dispatcher.progress_updated.emit(5, f"⚠️ {target_date_str} 為假日，往前推一天...")
        
        base_date -= timedelta(days=1)
        while base_date.weekday() in [5, 6]:
            base_date -= timedelta(days=1)

def get_recent_trading_day():
    today, now_time = datetime.now().date(), datetime.now().time()
    def last_fri(d):
        while d.weekday() != 4: d -= timedelta(days=1)
        return d
    w = today.weekday()
    if w in [5, 6]: return last_fri(today)
    if w == 0 and now_time < time(13, 30): return last_fri(today)
    if w > 0 and now_time < time(13, 30): return today - timedelta(days=1)
    return today

# ==================== 資料存取介面 (已全面升級為 SQLite) ====================

def print_trading_mode():
    if getattr(sys_config, 'live_trading_mode', False):
        print("⚠️  ============================================")
        print("⚠️  【正式下單模式】已啟動！")
        print("⚠️  所有委託將使用正式帳戶執行，將產生真實交易！")
        print("⚠️  ============================================")
    else:
        print("✅  【模擬下單模式】目前使用模擬帳戶，不會產生真實交易。")

def save_settings():
    # 將設定檔寫入 SQLite
    sys_db.save_state('settings', {k: getattr(sys_config, k) for k in vars(sys_config)})
    print_trading_mode()

def load_settings():
    # 🚀 確保新變數有預設值，避免被 hasattr 擋下無法讀取
    if not hasattr(sys_config, 'wait_min_avg_vol'): sys_config.wait_min_avg_vol = 30
    if not hasattr(sys_config, 'wait_max_single_vol'): sys_config.wait_max_single_vol = 80
    if not hasattr(sys_config, 'slippage_ticks'): sys_config.slippage_ticks = 3
    if not hasattr(sys_config, 'live_trading_mode'): sys_config.live_trading_mode = False
    if not hasattr(sys_config, 'min_lag_pct'): sys_config.min_lag_pct = 1.5
    if not hasattr(sys_config, 'min_height_pct'): sys_config.min_height_pct = 3.0
    if not hasattr(sys_config, 'volatility_min_range'): sys_config.volatility_min_range = 1.0
    if not hasattr(sys_config, 'min_eligible_avg_vol'): sys_config.min_eligible_avg_vol = 0
    if not hasattr(sys_config, 'min_close_price'): sys_config.min_close_price = 0
    if not hasattr(sys_config, 'allow_leader_entry'): sys_config.allow_leader_entry = False
    if not hasattr(sys_config, 'total_capital'): sys_config.total_capital = 249
    if not hasattr(sys_config, 'max_daily_entries'): sys_config.max_daily_entries = 12
    if not hasattr(sys_config, 'max_daily_stops'): sys_config.max_daily_stops = 3
    if not hasattr(sys_config, 'risk_control_enabled'): sys_config.risk_control_enabled = True
    if not hasattr(sys_config, 'cutoff_time_mins'): sys_config.cutoff_time_mins = 270
    if not hasattr(sys_config, 'ui_theme'): sys_config.ui_theme = 'classic'

    # 從 SQLite 讀取設定檔，若有舊的 settings.json 順便遷移
    if os.path.exists('settings.json'):
        s = json.load(open('settings.json', 'r', encoding='utf-8'))
        sys_db.save_state('settings', s)
        try: os.remove('settings.json')
        except Exception: pass
    else:
        s = sys_db.load_state('settings', default_value={})
    
    for k, v in s.items():
        if hasattr(sys_config, k): setattr(sys_config, k, v)
    # 向下相容：舊設定中的 'tradingview' 主題名稱映射為 'classic'
    if getattr(sys_config, 'ui_theme', '') == 'tradingview':
        sys_config.ui_theme = 'classic'

def load_matrix_dict_analysis():
    if os.path.exists('matrix_dict_analysis.json'):
        d = json.load(open('matrix_dict_analysis.json', 'r', encoding='utf-8'))
        sys_db.save_state('matrix_dict_analysis', d)
        try: os.remove('matrix_dict_analysis.json')
        except Exception: pass
        
    # 從資料庫載入後，強制將所有股票代號轉換為字串 (防呆)
    raw_data = sys_db.load_state('matrix_dict_analysis', default_value={})
    if isinstance(raw_data, dict):
        return {grp: [str(sym) for sym in syms] for grp, syms in raw_data.items()}
    return {}

def save_matrix_dict(d): 
    sys_db.save_state('matrix_dict_analysis', d)
    
def save_auto_intraday_data(data):
    # 實戰過程的分K，強制存入實戰專用表 (live)
    sys_db.save_kline('intraday_kline_live', data)
    # 同步保留一份文字快取 (用於緊急恢復)
    threading.Thread(target=lambda: json.dump(sys_state.in_memory_intraday, open('auto_intraday.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False) if os.path.exists('auto_intraday.json') else None, daemon=True).start()

def load_disposition_stocks(is_live=False):
    """
    讀取處置股清單：盤中讀 Live，回測讀歷史
    """
    if is_live:
        # 實戰：讀取今天剛抓好的名單
        return sys_db.load_state('disposition_stock_live', default_value=[])
    else:
        # 回測：讀取剛才更新 K 線時存下的當日精準歷史名單
        return sys_db.load_state('disposition_stocks', default_value=[])

def fetch_active_disposition_on_date(target_date_str):
    """
    【系統核心】獲取指定日期當天「所有正在處置中」的股票名單
    邏輯：查詢指定日前 15 天的所有公告並取聯集 (雙軌解析 + 防跳轉裝甲)
    """
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    try:
        end_date_obj = datetime.strptime(target_date_str, '%Y-%m-%d')
        start_date_obj = end_date_obj - timedelta(days=15)
        twse_start = start_date_obj.strftime('%Y%m%d')
        twse_end = end_date_obj.strftime('%Y%m%d')
    except ValueError:
        print("❌ 日期格式錯誤，預期為 YYYY-MM-DD")
        return []

    disposition_stocks = set()
    print(f"🔍 正在還原 {target_date_str} 的處置現場 (回溯區間: {twse_start} ~ {twse_end})...")

    # 1. 上市 ( TWSE )
    twse_headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        twse_url = f"https://www.twse.com.tw/announcement/punish?response=json&startDate={twse_start}&endDate={twse_end}"
        res_twse = requests.get(twse_url, headers=twse_headers, timeout=10, verify=False)
        if res_twse.status_code == 200:
            data = res_twse.json()
            if 'data' in data:
                for row in data['data']:
                    for cell in row:
                        cell_str = str(cell).strip()
                        if re.fullmatch(r'\d{4,6}', cell_str):
                            disposition_stocks.add(cell_str)
                            break
    except Exception as e:
        logger.error(f"上市處置股抓取失敗: {e}")

    # 2. 上櫃 ( TPEx ) - 雙軌解析與防彈裝甲
    tpex_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'X-Requested-With': 'XMLHttpRequest'
    }
    for i in range(16):
        current_check_date = end_date_obj - timedelta(days=i)
        if current_check_date.weekday() >= 5: continue
        roc_year = current_check_date.year - 1911
        tpex_date = f"{roc_year}/{current_check_date.strftime('%m/%d')}"
        try:
            tpex_url = f"https://www.tpex.org.tw/web/bulletin/disposal_information/disposal_information_result.php?l=zh-tw&d={tpex_date}&o=json"
            res_tpex = requests.get(tpex_url, headers=tpex_headers, timeout=5, verify=False)
            text = res_tpex.text.strip()
            
            if not text or text.startswith('<'): continue
            data_tpex = res_tpex.json()
            
            # 🟢 新版 TPEx 解析 (tables)
            if 'tables' in data_tpex:
                for table in data_tpex['tables']:
                    if 'data' in table:
                        for row in table['data']:
                            if len(row) > 2:
                                val_str = str(row[2]).strip()
                                if re.fullmatch(r'\d{4,6}', val_str):
                                    disposition_stocks.add(val_str)
            # 🟡 舊版 TPEx 解析 (aaData)
            elif 'aaData' in data_tpex:
                for row in data_tpex['aaData']:
                    for val in row:
                        val_str = str(val).strip()
                        if re.fullmatch(r'\d{4,6}', val_str):
                            disposition_stocks.add(val_str)
                            break
        except Exception: continue

    return sorted(list(disposition_stocks))

def fetch_disposition_stocks(client=None, matrix_dict=None):
    """
    盤中實戰呼叫：取得今天的「處置股」，存入 Live 資料庫 (禁空股由盤中備胎機制處理)
    """
    print(f"\n🔍 [盤前準備] 啟動市場處置股隔離機制...")
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    # 僅取得處置股名單 (流動性最差，必須盤前隔離)
    dispo_list = fetch_active_disposition_on_date(today_str)
    
    # 存入盤中專用資料庫
    sys_db.save_state('disposition_stock_live', dispo_list)
    print(f"✅ 盤前隔離完成：今日共 {len(dispo_list)} 檔處置股不予監控。")

def save_disposition_stocks(d):
    sys_db.save_state('disposition_stocks', d)

def load_kline_data(dates=None):
    daily = sys_db.load_kline('daily_kline_history', dates=dates)
    intra = sys_db.load_kline('intraday_kline_history', dates=dates)
    return daily, intra

def ensure_continuous_time_series(df):
    df['date'], df['time'] = pd.to_datetime(df['date']), pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    df.set_index(['date', 'time'], inplace=True)
    df = df.reindex(pd.MultiIndex.from_product([df.index.get_level_values('date').unique(), pd.date_range('09:00', '13:30', freq='1min').time], names=['date', 'time']))
    
    # 防呆機制，動態檢查 DataFrame 內擁有哪些欄位才進行補齊，防止 KeyError
    fill_cols = [c for c in ['symbol', '昨日收盤價', '漲停價'] if c in df.columns]
    if fill_cols:
        df[fill_cols] = df[fill_cols].ffill().bfill()
        
    if 'high' not in df.columns: df['high'] = df['close']
    
    # 確保 '昨日收盤價' 存在，否則預設為 0
    yc = df['昨日收盤價'] if '昨日收盤價' in df.columns else 0
    df['close'] = df['close'].ffill().fillna(yc)
    
    for c in ['open', 'high', 'low']: df[c] = df[c].ffill().fillna(df['close'])
    df['volume'], df['pct_increase'] = df['volume'].fillna(0), df['pct_increase'].fillna(0.0) if 'pct_increase' in df.columns else 0.0
    return df.reset_index()

def initialize_stock_data(symbols, daily, intra):
    # 這裡的 daily 和 intra 參數會由 load_kline_data() 傳入
    # 已由 load_kline_data 確保傳進來的是 'daily_kline_history' 與 'intraday_kline_history'
    return {s: ensure_continuous_time_series(pd.DataFrame(intra[s])).drop(columns=['average'], errors='ignore')
            for s in symbols if s in intra and not pd.DataFrame(intra[s]).empty}

def build_backtest_cache(db_path=None, table_name=None, dates=None):
    """一次從 DB 載入所有日期的分鐘 K 線資料，回傳預載快取。

    Args:
        db_path: 外部 DB 路徑（回歸用），None 則使用 sys_db（回測用）
        table_name: DB 資料表名稱（回歸用）
        dates: 可選，只載入指定日期

    Returns:
        {date_str: {symbol: DataFrame}}  — 與 initialize_stock_data() 輸出格式一致
    """
    if db_path and table_name:
        conn = sqlite3.connect(db_path)
        if dates:
            dates_str = "','".join(dates)
            df = pd.read_sql(f"SELECT * FROM {table_name} WHERE date IN ('{dates_str}')", conn)
        else:
            df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
        conn.close()
    else:
        with sys_db.db_lock:
            if dates:
                dates_str = "','".join(dates)
                df = pd.read_sql(f"SELECT * FROM intraday_kline_history WHERE date IN ('{dates_str}')", sys_db.conn)
            else:
                df = pd.read_sql('SELECT * FROM intraday_kline_history', sys_db.conn)

    for c in ['high', '漲停價', 'pct_increase', 'volume', 'close', 'open', 'low', '昨日收盤價', 'rise']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if dates:
        df = df[df['date'].isin(dates)]

    cache = {}
    for d, d_group in df.groupby('date'):
        sym_dict = {}
        for sym, s_group in d_group.groupby('symbol'):
            s_df = s_group.sort_values('time').copy()
            s_df = ensure_continuous_time_series(s_df).drop(columns=['average'], errors='ignore')
            sym_dict[sym] = s_df
        cache[d] = sym_dict
    return cache

def purge_disposition_from_nb(nb_dict, disposition_list):
    # 純記憶體清洗。不再覆蓋資料庫的 nb_matrix_dict，完美保護回測！
    if 'consolidated_symbols' not in nb_dict or not isinstance(nb_dict['consolidated_symbols'], dict): 
        return nb_dict
    
    cleaned_count = 0
    for grp, syms in nb_dict['consolidated_symbols'].items():
        filtered = [str(s) for s in dict.fromkeys(syms) if str(s) not in disposition_list]
        cleaned_count += (len(syms) - len(filtered))
        nb_dict['consolidated_symbols'][grp] = filtered
            
    if cleaned_count > 0:
        print(f"✅ 盤中動態防護：已從族群名單中，隔離 {cleaned_count} 檔處置股。")
        
    return nb_dict

def load_symbols_to_analyze(is_live=False):
    # 傳遞 is_live 參數，決定要用哪一份處置股名單來過濾
    dispo_list = load_disposition_stocks(is_live=is_live)
    return [str(s) for g in load_matrix_dict_analysis().values() for s in g if str(s) not in dispo_list]

def exit_trade(selected_stock_df, shares, entry_price, sell_cost, entry_fee, tax, message_log, current_time, hold_time, entry_time, use_f_exit=False, sym=""):
    current_time_str = current_time if isinstance(current_time, str) else current_time.strftime('%H:%M:%S')
    selected_stock_df['time'] = pd.to_datetime(selected_stock_df['time'], format='%H:%M:%S').dt.time

    if use_f_exit:
        end_price_series = selected_stock_df[selected_stock_df['time'] == datetime.strptime('13:30', '%H:%M').time()]['close']
        if not end_price_series.empty: end_price = end_price_series.values[0]
        else: return None, None
    else:
        entry_index_series = selected_stock_df[selected_stock_df['time'] == (datetime.strptime(entry_time, '%H:%M:%S').time() if isinstance(entry_time, str) else entry_time)].index
        if not entry_index_series.empty and entry_index_series[0] + hold_time < len(selected_stock_df): end_price = selected_stock_df.iloc[entry_index_series[0] + hold_time]['close']
        else: return None, None

    buy_cost = shares * end_price * 1000
    exit_fee = int(buy_cost * (sys_config.transaction_fee * 0.01) * (sys_config.transaction_discount * 0.01))
    profit = sell_cost - buy_cost - entry_fee - exit_fee - tax
    return_rate = profit / sell_cost * 100 if sell_cost != 0 else 0.0
    sym_str = f"{sn(sym)} " if sym else ""
    if int(profit) > 0:
        message_log.append((current_time_str, f"{RED}出場！{sym_str}進場:{entry_time} 利潤：{int(profit)} 元，報酬率：{return_rate:.2f}%{RESET}\n"))
    elif int(profit) < 0:
        message_log.append((current_time_str, f"{GREEN}出場！{sym_str}進場:{entry_time} 利潤：{int(profit)} 元，報酬率：{return_rate:.2f}%{RESET}\n"))
    else:
        message_log.append((current_time_str, f"出場！{sym_str}進場:{entry_time} 利潤：{int(profit)} 元，報酬率：{return_rate:.2f}%\n"))
    return profit, return_rate

# ------------------ Shioaji API & 平倉邏輯 ------------------
logging.getLogger('shioaji').setLevel(logging.WARNING)
_is_live = getattr(sys_config, 'live_trading_mode', False)
sys_state.api = sj.Shioaji(simulation=not _is_live)
deal_tracker = {} # 用來追蹤真實成交均價的全域變數

# ==========================================
# 🚀 全域廣播中心：計算最新損益並推播給 UI
# ==========================================
def broadcast_portfolio_update():
    if not hasattr(sys_state, 'open_positions'): return
    with sys_state.lock:
        ui_data = []
        for sym, pos_info in sys_state.open_positions.items():
            filled = pos_info.get('filled_shares', 0)
            target = pos_info.get('target_shares', 0)
            if filled > 0 or target > 0:
                # 🔧 委託中（尚未成交）
                if filled == 0:
                    ui_data.append({
                        "symbol": sym,
                        "entry_price": pos_info.get('entry_price', 0),
                        "current_price": 0,
                        "shares": 0,
                        "target_shares": target,
                        "profit": None,  # UI 顯示 '--'
                        "stop_loss": pos_info.get('stop_loss', '未設定'),
                        "entry_time": pos_info.get('entry_time', ''),
                        "return_pct": None,
                        "status": "委託中",
                    })
                    continue

                # 取得現價 (由 snapshot 或 備援機制 更新過的數值)
                try:
                    current_price = float(pos_info.get('current_price', 0))
                    entry_price = float(pos_info.get('entry_price', 0))
                except Exception:
                    current_price = 0.0
                    entry_price = float(pos_info.get('entry_price', 0))

                # --- 判斷損益計算基礎 ---
                if current_price > 0:
                    calc_price = current_price
                    has_price = True
                else:
                    calc_price = pos_info.get('entry_price', 0)
                    has_price = False

                buy_back_cost = filled * calc_price * 1000
                sell_entry_val = pos_info.get('real_entry_cost', filled * pos_info['entry_price'] * 1000)

                # 損益計算 (包含出場規費預估)
                est_exit_fee = buy_back_cost * (sys_config.transaction_fee * 0.01) * (sys_config.transaction_discount * 0.01)
                est_tax = buy_back_cost * (sys_config.trading_tax * 0.01)

                pnl = (sell_entry_val - buy_back_cost) - pos_info.get('entry_fee', 0) - est_exit_fee - est_tax
                real_pnl = pnl if has_price else 0

                # 報酬率（做空：進場價 - 現價 / 進場價 * 100）
                ret_pct = ((entry_price - calc_price) / entry_price * 100) if entry_price > 0 and has_price else 0.0

                ui_data.append({
                    "symbol": sym,
                    "entry_price": pos_info.get('entry_price', 0),
                    "current_price": current_price,
                    "shares": filled,
                    "profit": real_pnl,
                    "stop_loss": pos_info.get('stop_loss', '未設定'),
                    "entry_time": pos_info.get('entry_time', ''),
                    "return_pct": ret_pct,
                })
        try: ui_dispatcher.portfolio_updated.emit(ui_data)
        except Exception: pass

# 🔧 成交回報去重 (同一筆 exchange_seq 會被 callback 觸發 2 次)
_seen_seqs = set()

# 攔截券商 API 回報，淨化輸出並計算真實 VWAP
def order_callback(stat, content):
    try:
        # 🔧 成交去重：相同 exchange_seq 只處理一次
        if str(stat) == "OrderState.StockDeal":
            seq = content.get('exchange_seq') if isinstance(content, dict) else getattr(content, 'exchange_seq', None)
            if seq and seq in _seen_seqs: return
            if seq: _seen_seqs.add(seq)

        # 1. 攔截真實成交回報 (StockDeal)
        if str(stat) == "OrderState.StockDeal":
            sym = content.get('code', '未知')
            action_raw = content.get('action', '')
            action = "買進(平倉)" if action_raw == 'Buy' else "賣出(進場)"
            price = float(content.get('price', 0))
            qty = int(content.get('quantity', 0))
            
            # 印出即時碎單成交回報
            print(f"[成交回報] {sym} {sn(sym).replace(sym, '').strip()} {action} | 成交 {qty} 張 @ {price:.2f} 元")
            
            with sys_state.lock:
                # ====== 🚀 新增：核心訂單狀態機 ======
                if sym in sys_state.open_positions:
                    pos = sys_state.open_positions[sym]
                    
                    # 情況 A：進場空單成交
                    if action == "賣出(進場)":
                        old_qty = pos.get('filled_shares', 0)
                        new_qty = old_qty + qty
                        new_cost = pos.get('real_entry_cost', 0.0) + (price * qty)
                        
                        pos['filled_shares'] = new_qty
                        pos['real_entry_cost'] = new_cost
                        pos['entry_price'] = round(new_cost / new_qty, 2)
                        pos['current_price'] = price  # 🔧 無條件更新成交價

                        # 🚀 計算預計出場時間字串 (首筆成交時設定)
                        if old_qty == 0:
                            now_dt = datetime.now()
                            pos['entry_date'] = now_dt.strftime("%Y-%m-%d")
                            pos['entry_time'] = now_dt.strftime("%H:%M:%S")

                            # 讀取系統設定的持有時間
                            hold_cfg = str(getattr(sys_config, 'hold_time', 'F')).upper()
                            if hold_cfg == 'F':
                                pos['planned_exit'] = "13:26:00"
                            else:
                                try:
                                    hold_mins = int(hold_cfg)
                                    exit_dt = now_dt + timedelta(minutes=hold_mins)
                                    exit_str = exit_dt.strftime("%H:%M:%S")
                                    pos['planned_exit'] = exit_str if exit_str < "13:26:00" else "13:26:00"
                                except Exception:
                                    pos['planned_exit'] = "13:26:00"
                            print(f"⏱️ [時間標記] {sym} 記錄進場: {pos['entry_time']}, 預計回補: {pos['planned_exit']}")
                        
                        # 🎯 動態建倉防護：成交幾張，就掛幾張停損觸價單
                        stop_thr = pos.get('stop_loss')
                        if stop_thr and hasattr(sys_state, 'to') and sys_state.to:
                            tcond = tp.TouchOrderCond(
                                tp.TouchCmd(code=f"{sym}", close=tp.Price(price=stop_thr, trend="Equal")), 
                                tp.OrderCmd(code=f"{sym}", order=sj.Order(price=0, quantity=qty, action="Buy", order_type="IOC", price_type="MKT"))
                            )
                            result = safe_add_touch_condition(sys_state.to, tcond)
                            if result is None:
                                print(f"{Fore.RED}⚠️ [嚴重] {sym} 觸價停損單註冊失敗！系統將依賴 K 線偵測作為後備{Style.RESET_ALL}")
                            else:
                                print(f"[防護更新] {sym} 已自動為本次成交的 {qty} 張設置觸價停損單 ({stop_thr:.2f})")

                        target_shares = pos.get('target_shares', qty)
                        if new_qty >= target_shares and old_qty < target_shares:
                            tg_bot.send_message(f"🟢 <b>進場全數成交</b>\n標的: <code>{sn(sym)}</code>\n數量: {new_qty} 張\n均價: {pos['entry_price']:.2f}\n停損: {stop_thr:.2f}", force=True)
                            _play_alert('entry')
                        elif old_qty == 0:
                            tg_bot.send_message(f"🟢 <b>進場開始成交</b>\n標的: <code>{sn(sym)}</code>\n已成交: {new_qty}/{target_shares} 張\n目前均價: {pos['entry_price']:.2f}", force=True)
                            _play_alert('entry')  # 🔧 首筆成交也播放音效
                        
                        sys_db.save_state('current_position', sys_state.open_positions)
                        broadcast_portfolio_update()

                    # 情況 B：停損或平倉買單成交
                    elif action == "買進(平倉)":
                        pos['covered_shares'] = pos.get('covered_shares', 0) + qty
                        pos['real_exit_cost'] = pos.get('real_exit_cost', 0.0) + (price * qty)
                        
                        if not pos.get('sl_db_logged', False):
                            sys_db.log_trade("買回(平倉)", sym, pos.get('filled_shares', qty), pos.get('stop_loss', price), 0.0, "系統平倉/停損")
                            pos['sl_db_logged'] = True
                            tg_bot.send_message(f"🔴 <b>出場觸發</b>\n標的: <code>{sn(sym)}</code>\n系統已執行平倉！", force=True)
                            _play_alert('stop_loss')

                        # 完全平倉後清理資料庫與部位
                        if pos['covered_shares'] >= pos.get('filled_shares', 1):
                            real_vwap = round(pos['real_exit_cost'] / pos['covered_shares'], 2)
                            entry_p = pos.get('entry_price', 0)
                            closed_qty = pos.get('covered_shares', 1)
                            est_profit = round((entry_p - real_vwap) * closed_qty * 1000, 0) if entry_p > 0 else 0.0
                            sys_state.open_positions.pop(sym, None)
                            try:
                                with sys_db.db_lock:
                                    with sys_db.conn:
                                        sys_db.conn.execute('''
                                            UPDATE trade_logs SET price = ?, profit = ?
                                            WHERE symbol = ? AND (action LIKE '%平倉%' OR action LIKE '%買回%' OR action LIKE '%停損%')
                                            ORDER BY id DESC LIMIT 1
                                        ''', (real_vwap, est_profit, sym))
                            except Exception: pass

                            # 🛡️ 風控模組：停損計數 + 熔斷
                            if hasattr(sys_state, 'risk_state') and sys_config.risk_control_enabled:
                                sys_state.risk_state['daily_stops'] = sys_state.risk_state.get('daily_stops', 0) + 1
                                if sys_state.risk_state['daily_stops'] >= sys_config.max_daily_stops:
                                    sys_state.risk_state['halted'] = True
                                    halt_msg = f"🔴 風控熔斷！累計停損 {sys_state.risk_state['daily_stops']} 筆，當日停止進場"
                                    print(f"{Fore.RED}{halt_msg}{Style.RESET_ALL}")
                                    try: tg_bot.send_message(f"🔴 <b>風控熔斷</b>\n{halt_msg}", force=True)
                                    except Exception: pass

                            # 開放重新進場
                            if getattr(sys_config, 'allow_reentry', False):
                                if not hasattr(sys_state, 'previous_stop_loss'): sys_state.previous_stop_loss = set()
                                sys_state.previous_stop_loss.add(sym)

            # ====== 舊版追蹤器保留 (相容性) ======
            if sym in deal_tracker:
                deal_tracker[sym]['total_qty'] += qty
                deal_tracker[sym]['total_cost'] += (price * qty)
                
                target_qty = deal_tracker[sym]['target_qty']
                if deal_tracker[sym]['total_qty'] >= target_qty:
                    real_vwap = round(deal_tracker[sym]['total_cost'] / target_qty, 2)
                    action_type = deal_tracker[sym]['action_type']
                    entry_p = deal_tracker[sym].get('entry_price', 0)
                    est_profit = round((entry_p - real_vwap) * target_qty * 1000, 0) if entry_p > 0 else 0.0

                    print(f"📊 {sym} {sn(sym).replace(sym, '').strip()} {action_type}全數成交，真實均價為: {real_vwap:.2f} 元，損益: {est_profit:+.0f}")

                    if action_type != '進場':
                        try:
                            with sys_db.db_lock:
                                with sys_db.conn:
                                    sys_db.conn.execute('''
                                        UPDATE trade_logs
                                        SET price = ?, profit = ?
                                        WHERE symbol = ? AND (action LIKE '%平倉%' OR action LIKE '%買回%')
                                        ORDER BY id DESC LIMIT 1
                                    ''', (real_vwap, est_profit, sym))
                        except Exception: pass
                        print(f"📝 (已將最終真實均價 {real_vwap:.2f} 與損益 {est_profit:+.0f} 寫入歷史交易紀錄)")
                    else:
                        print(f"✨ (此時 GUI「即時持倉監控」與 Telegram 已瞬間新增 {sym}) ✨")
                    
                    del deal_tracker[sym]

        # 2. 攔截委託單回報 (隱藏原始的落落長字典輸出)
        elif str(stat) == "OrderState.StockOrder":
            pass 
            
    except Exception as e:
        logger.error(f"委託單回報解析錯誤: {e}", exc_info=True)
      
# 🔧 stdout 過濾器：抑制 Shioaji 內部的 OrderState 原始字典輸出
class _SjOutputFilter:
    def __init__(self, orig):
        self._orig = orig
        self._buf = ""
    def write(self, text):
        # 攔截包含 OrderState.Stock 的原始輸出
        self._buf += text
        if '\n' in self._buf:
            lines = self._buf.split('\n')
            self._buf = lines[-1]  # 保留最後不完整的行
            for line in lines[:-1]:
                if 'OrderState.Stock' not in line:
                    self._orig.write(line + '\n')
        elif len(self._buf) > 4096:
            if 'OrderState.Stock' not in self._buf:
                self._orig.write(self._buf)
            self._buf = ""
    def flush(self):
        if self._buf and 'OrderState.Stock' not in self._buf:
            self._orig.write(self._buf)
        self._buf = ""
        self._orig.flush()
    def __getattr__(self, name):
        return getattr(self._orig, name)

sys.stdout = _SjOutputFilter(sys.stdout)

try:
    print(f"{YELLOW}⏳ 正在初始化 Shioaji API 並自動登入預設帳戶...{RESET}")
    _is_live = getattr(sys_config, 'live_trading_mode', False)
    _ak = shioaji_logic.LIVE_API_KEY if _is_live else shioaji_logic.TEST_API_KEY
    _as = shioaji_logic.LIVE_API_SECRET if _is_live else shioaji_logic.TEST_API_SECRET
    sys_state.api.login(api_key=_ak, secret_key=_as)
    sys_state.api.activate_ca(ca_path=shioaji_logic.CA_CERT_PATH, ca_passwd=shioaji_logic.CA_PASSWORD)
    sys_state.api.set_order_callback(order_callback) # 🟢 註冊 Callback 攔截核心
    print(f"{GREEN}✅ Shioaji 登入成功！合約資料已就緒。{RESET}")
except Exception as e:
    _emsg = str(e)
    if "6999" in _emsg or "token_login" in _emsg:
        logger.error(f"Shioaji 登入失敗（憑證錯誤或非交易時段）：{_emsg}")
        print(f"{RED}⚠️ Shioaji 登入失敗（憑證錯誤或非交易時段）：API Key 或 Secret 有誤/過期，或目前伺服器暫不接受連線。請在交易時段重試，或至「帳戶 API 設定」更新憑證。{RESET}")
    else:
        logger.error(f"Shioaji 初始登入失敗: {_emsg}")
        print(f"{RED}⚠️ Shioaji 初始登入失敗: {_emsg}{RESET}")

try: sys_state.to = tp.TouchOrderExecutor(sys_state.api)
except Exception as _to_e:
    logger.error(f"觸價單模組初始化失敗: {_to_e}")
    print(f"{RED}⚠️ 觸價單模組初始化失敗，請稍後在介面中重新登入。{RESET}"); sys_state.to = None

def get_actual_fill_price(api_instance, code, action=sj.constant.Action.Buy, fallback_price=0.0):
    """從 Shioaji API 獲取最新的實際成交均價"""
    try:
        api_instance.update_status()
        trades = api_instance.list_trades()
        for t in reversed(trades):
            if t.contract.code == code and getattr(t.order, 'action', '') == action:
                deals = getattr(t.status, 'deals', [])
                if deals:
                    total_qty = sum(d.quantity for d in deals)
                    if total_qty > 0:
                        return sum(d.price * d.quantity for d in deals) / total_qty
        return fallback_price
    except Exception as e:
        logger.error(f"獲取實際成交價失敗 ({code}): {e}")
        return fallback_price

# 尾盤強制平倉
def exit_trade_live():
    with sys_state.lock: conditions_dict = dict(sys_state.to.conditions)
    
    # 不再從 conditions 抓數量，改從真實的 open_positions 抓取，徹底解決幽靈部位問題
    with sys_state.lock:
        exit_data = {
            code: pos.get('filled_shares', 0) - pos.get('covered_shares', 0) 
            for code, pos in sys_state.open_positions.items() 
            if pos.get('filled_shares', 0) - pos.get('covered_shares', 0) > 0
        }
    
    if not exit_data: return # 如果已經空手，就不做任何事
    
    load_twse_name_map() 
    now_time = datetime.now().time()
    is_late_market = now_time >= time(13, 25) 
    
    symbols_to_remove = []
    
    for stock_code, shares in exit_data.items():
        try:
            if str(stock_code) in STOCK_NAME_MAP.get("TSE", {}):
                contract = getattr(sys_state.api.Contracts.Stocks.TSE, f"TSE{stock_code}")
            else:
                contract = getattr(sys_state.api.Contracts.Stocks.OTC, f"OTC{stock_code}")

            # 理論結算價：精準使用當下的 1 分 K 收盤價 (而非漲停價)
            current_close = 0.0
            with sys_state.lock:
                real_limit_up = contract.limit_up
                if sys_state.in_memory_intraday and stock_code in sys_state.in_memory_intraday:
                    if len(sys_state.in_memory_intraday[stock_code]) > 0:
                        real_limit_up = sys_state.in_memory_intraday[stock_code][-1].get('漲停價', real_limit_up)
                        current_close = sys_state.in_memory_intraday[stock_code][-1].get('close', real_limit_up)

            order_type = sj.constant.OrderType.ROD if is_late_market else sj.constant.OrderType.IOC
            
            # 印出尾盤平倉訊息
            msg = f"⏱️ [時間出場] {stock_code} 尾盤強制平倉，委託價: {real_limit_up:.2f}，系統結算價(當前K線收盤): {current_close:.2f}"
            print(msg)
            _play_alert('exit')
            
            # 啟動真實均價追蹤器（同時記錄進場價以便後算損益）
            with sys_state.lock:
                _ep = sys_state.open_positions.get(stock_code, {}).get('entry_price', 0)
            deal_tracker[stock_code] = {'target_qty': shares, 'total_qty': 0, 'total_cost': 0.0, 'action_type': '平倉(尾盤)', 'time': datetime.now().strftime("%H:%M:%S"), 'entry_price': _ep}
            
            safe_place_order(sys_state.api, contract, sys_state.api.Order(action=sj.constant.Action.Buy, price=real_limit_up, quantity=shares, price_type=sj.constant.StockPriceType.LMT, order_type=order_type, order_lot=sj.constant.StockOrderLot.Common, account=sys_state.api.stock_account))
            
            if is_late_market:
                print(f"⏳ {stock_code} 進入尾盤結算模式，等待 13:30 撮合...")
            
            # 先用 1分K 結算價記錄，稍後由 Callback 覆蓋真實 VWAP
            sys_db.log_trade("平倉(尾盤)", stock_code, shares, current_close, 0.0, "13:26 強制平倉")
            symbols_to_remove.append(stock_code)

        except Exception as e: logger.error(f"平倉 {stock_code} 錯誤: {e}", exc_info=True)
        
    # 物理刪除持倉狀態並廣播 GUI (解決幽靈部位與二度買回)
    with sys_state.lock:
        for code in symbols_to_remove:
            if code in sys_state.previous_stop_loss: sys_state.previous_stop_loss.remove(code)
            sys_state.open_positions.pop(code, None)
            
            # 移除相關觸價單
            conds = sys_state.to.conditions.get(code, [])
            for c in conds: safe_delete_touch_condition(sys_state.to, c)
            sys_state.to.conditions.pop(code, None)
            
    if symbols_to_remove:
        sys_db.save_state('current_position', sys_state.open_positions)
        if hasattr(ui_dispatcher, 'position_updated'):
            ui_dispatcher.position_updated.emit(sys_state.open_positions)
        print(f"[部位清理] 已從即時持倉中移除 {', '.join(symbols_to_remove)}，GUI 與 Telegram 已同步更新。")

# 單一/超時平倉
def close_one_stock(code: str):
    with sys_state.lock:
        if code not in sys_state.open_positions:
            logger.warning(f"{code} 無持倉可平倉")
            return
        pos = sys_state.open_positions[code]
        # 🛡️ 修正為：只平倉實際「已成交」減去「已平倉」的餘額
        qty = pos.get('filled_shares', 0) - pos.get('covered_shares', 0)
    
    if qty == 0: return
    
    load_twse_name_map() 
    now_time = datetime.now().time()
    is_late_market = now_time >= time(13, 25) 
    
    try:
        if str(code) in STOCK_NAME_MAP.get("TSE", {}):
            contract = getattr(sys_state.api.Contracts.Stocks.TSE, f"TSE{code}")
        else:
            contract = getattr(sys_state.api.Contracts.Stocks.OTC, f"OTC{code}")
            
        current_close = 0.0
        with sys_state.lock:
            real_limit_up = contract.limit_up
            if sys_state.in_memory_intraday and code in sys_state.in_memory_intraday:
                if len(sys_state.in_memory_intraday[code]) > 0:
                    real_limit_up = sys_state.in_memory_intraday[code][-1].get('漲停價', real_limit_up)
                    current_close = sys_state.in_memory_intraday[code][-1].get('close', real_limit_up)

        order_type = sj.constant.OrderType.ROD if is_late_market else sj.constant.OrderType.IOC
        
        msg = f"⏱️ [超時出場] {code} 強制平倉，委託價: {real_limit_up:.2f}，系統結算價: {current_close:.2f}"
        print(msg)
        
        deal_tracker[code] = {'target_qty': qty, 'total_qty': 0, 'total_cost': 0.0, 'action_type': '平倉(超時)', 'time': datetime.now().strftime("%H:%M:%S"), 'entry_price': pos.get('entry_price', 0)}
        
        safe_place_order(sys_state.api, contract, sys_state.api.Order(action=sj.constant.Action.Buy, price=real_limit_up, quantity=qty, price_type=sj.constant.StockPriceType.LMT, order_type=order_type, order_lot=sj.constant.StockOrderLot.Common, account=sys_state.api.stock_account))
        
        if is_late_market: print(f"⏳ {code} 進入尾盤結算模式...")
        sys_db.log_trade("平倉(超時)", code, qty, current_close, 0.0, "單一/超時平倉")

    except Exception as e: logger.error(f"平倉 {code} 錯誤: {e}", exc_info=True)
    
    # 物理刪除持倉狀態並廣播 GUI
    with sys_state.lock:
        if code in sys_state.previous_stop_loss: sys_state.previous_stop_loss.remove(code)
        sys_state.open_positions.pop(code, None)
        
        conds = sys_state.to.conditions.get(code, [])
        for c in conds: safe_delete_touch_condition(sys_state.to, c)
        sys_state.to.conditions.pop(code, None)
        
    sys_db.save_state('current_position', sys_state.open_positions)
    if hasattr(ui_dispatcher, 'position_updated'):
        ui_dispatcher.position_updated.emit(sys_state.open_positions)
    print(f"[部位清理] {code} 已從即時持倉中移除。")


# 自動停損監控邏輯
def monitor_stop_loss_orders(group_positions):
    with sys_state.lock:
        if hasattr(sys_state, 'previous_stop_loss') and sys_state.previous_stop_loss:
            nb = load_matrix_dict_analysis()
            for code in list(sys_state.previous_stop_loss):
                for group, symbols in nb.items():
                    if code in symbols and group in group_positions and group_positions[group] == "已進場": 
                        group_positions[group] = False
                        logger.info(f"{group} 族群停損/平倉出場：股票 {code}，開放重新進場")
                        print(f"⚠️ {group} 族群停損/平倉出場：股票 {code}，開放重新進場。")
                sys_state.previous_stop_loss.remove(code)

def update_variable(file_path, var_name, new_value, is_raw=False):
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 確保精準匹配變數名稱，才進行替換
            if line.lstrip().startswith(var_name + "=") or line.lstrip().startswith(var_name + " ="):
                lines.append(f'{var_name} = r"{new_value}"\n' if is_raw else f'{var_name} = "{new_value}"\n')
            else:
                lines.append(line)
    with open(file_path, "w", encoding="utf-8") as f: f.writelines(lines)
    importlib.reload(shioaji_logic)

def initialize_triggered_limit_up(auto_intraday_data: dict):
    for sym, kbars in auto_intraday_data.items():
        for i in range(1, len(kbars)):
            if round(kbars[i]["high"], 2) >= round(kbars[i]["漲停價"], 2) and round(kbars[i-1]["high"], 2) < round(kbars[i]["漲停價"], 2):
                sys_state.triggered_limit_up.add(sym); break

def calculate_average_over_high(group_name=None, progress_callback=None):
    daily_kline_data, intraday_kline_data = load_kline_data()
    matrix_dict_analysis = load_matrix_dict_analysis()
    if group_name is None: group_name = input("請輸入要分析的族群名稱：")
    if group_name not in matrix_dict_analysis: return None
    symbols_to_analyze = [s for s in matrix_dict_analysis[group_name] if s not in load_disposition_stocks()]
    if not symbols_to_analyze: return None

    print(f"開始分析族群 {group_name} 中的股票...")
    group_over_high_averages, total_symbols = [], len(symbols_to_analyze) 
    
    for i, symbol in enumerate(symbols_to_analyze):
        if progress_callback: progress_callback(int((i / total_symbols) * 100), f"正在分析: {symbol}")
        if symbol not in daily_kline_data or symbol not in intraday_kline_data: continue
        
        df = pd.DataFrame(intraday_kline_data[symbol])
        if df.empty or 'time' not in df.columns: continue
        
        # 強制轉換時間格式
        df['time_dt'] = pd.to_datetime(df['time'].astype(str), format='mixed')
        df['time_only'] = df['time_dt'].dt.time
        
        # 取得欄位索引位置，避免 itertuples 命名混淆
        cols = df.columns.tolist()
        pct_idx = cols.index('pct_increase') + 1 if 'pct_increase' in cols else -1
        high_idx = cols.index('high') + 1
        highest_idx = cols.index('highest') + 1
        time_idx = cols.index('time_only') + 1

        c1, c2, peak_h, c2_time, intervals = False, False, None, None, []
        
        for row in df.itertuples():
            curr_time = row[time_idx]
            curr_high = row[high_idx]
            curr_highest = row[highest_idx]
            pct_inc = row[pct_idx] if pct_idx != -1 else 0
            
            if peak_h is None: peak_h = curr_high; continue
            
            # C1: 啟動條件 (放寬至 1.5%)
            if not c1 and pct_inc >= 1.5: 
                c1, c2 = True, False
                peak_h = curr_high
            
            # C2: 回檔判定 (從波段高點滑落)
            if c1 and not c2:
                if curr_high > peak_h:
                    peak_h = curr_high
                elif curr_high < peak_h:
                    c2_time, c2 = curr_time, True
            
            # 過高判定: 盤中最高價突破剛才的波段高點
            elif c2 and curr_highest > peak_h:
                if c2_time: 
                    t1 = datetime.combine(date.today(), c2_time)
                    t2 = datetime.combine(date.today(), curr_time)
                    diff = (t2 - t1).total_seconds() / 60
                    if 1 < diff < 60: intervals.append(diff)
                c1 = c2 = False; c2_time = None; peak_h = curr_highest

        if intervals:
            # 過濾離群值
            q1, q3 = np.percentile(intervals, 25), np.percentile(intervals, 75)
            iqr = q3 - q1
            filtered = [inv for inv in intervals if q1 - 1.5 * iqr <= inv <= q3 + 1.5 * iqr]
            if not filtered: filtered = intervals
            group_over_high_averages.append(sum(filtered) / len(filtered))

    if group_over_high_averages:
        avg = sum(group_over_high_averages) / len(group_over_high_averages)
        print(f"{group_name} 平均過高間隔：{avg:.2f} 分鐘")
        return avg
    else:
        print(f"{group_name} 沒有足夠的過高間隔數據。")
        return None

# Telegram 單股走勢圖生成引擎 (含 KeyError 防護與資料校驗)
def get_stock_chart_bytes(code):
    try:
        # 1. 取得資料，優先取實戰表，次之取歷史表
        raw_data = sys_db.load_kline('intraday_kline_live')
        if not raw_data: 
            raw_data = sys_db.load_kline('intraday_kline_history')
        
        # 使用 .get() 避免 KeyError。如果代號不存在，回傳空清單
        stock_records = raw_data.get(str(code), [])
        
        # 2. 檢查資料是否有效
        if not stock_records:
            logger.warning(f"[Telegram] 資料庫中找不到代號 {code} 的數據")
            return None
            
        df = pd.DataFrame(stock_records)
        if df.empty or 'time' not in df.columns or 'close' not in df.columns:
            return None
            
        # 3. 資料預處理
        # 確保日期與時間欄位正確合併，並強制轉型為數值以防畫圖出錯
        df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
        df = df.dropna(subset=['datetime']).sort_values(by='datetime')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. 繪圖設定 (Headless 模式，適用於伺服器端生成)
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        import matplotlib.dates as mdates # 👈 引入時間格式化神器
        
        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        
        # --- 上圖：價格走勢 ---
        ax_price = fig.add_subplot(gs[0])
        ax_price.plot(df['datetime'], df['close'], color='#2980B9', linewidth=2, label='收盤價')
        
        # 🎯 這裡加上強制 X 軸格式的魔法
        # 強制只在每小時的 00 分和 30 分顯示刻度
        locator = mdates.MinuteLocator(byminute=[0, 30])
        formatter = mdates.DateFormatter('%H:%M') # 只要 HH:MM，不要年月日
        ax_price.xaxis.set_major_locator(locator)
        ax_price.xaxis.set_major_formatter(formatter)
        
        # 標註最高與最低點
        if len(df) > 0:
            max_idx = df['close'].idxmax()
            min_idx = df['close'].idxmin()
            max_p, min_p = df.loc[max_idx, 'close'], df.loc[min_idx, 'close']
            max_t, min_t = df.loc[max_idx, 'datetime'], df.loc[min_idx, 'datetime']
            
            ax_price.plot(max_t, max_p, 'r^', markersize=8) 
            ax_price.annotate(f'最高 {max_p}', (max_t, max_p), textcoords="offset points", 
                             xytext=(0,10), ha='center', color='red', fontweight='bold')
            ax_price.plot(min_t, min_p, 'gv', markersize=8) 
            ax_price.annotate(f'最低 {min_p}', (min_t, min_p), textcoords="offset points", 
                             xytext=(0,-15), ha='center', color='green', fontweight='bold')

        # 取得股票名稱
        load_twse_name_map()
        title_text = sn(str(code)).strip()
        
        ax_price.set_title(f"{title_text} - 當日價量走勢圖", fontsize=16, fontweight='bold')
        ax_price.grid(True, linestyle='--', alpha=0.6)
        
        # --- 下圖：紅綠成交量 ---
        ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
        # 確保 open 與 close 都有資料才計算顏色
        if 'open' in df.columns:
            colors = ['#E74C3C' if c >= o else '#2ECC40' for c, o in zip(df['close'], df['open'])]
        else:
            colors = '#2980B9'
            
        ax_vol.bar(df['datetime'], df['volume'], color=colors, width=0.0005) 
        ax_vol.grid(True, linestyle='--', alpha=0.6)
        ax_vol.set_ylabel("成交量")
        
        # 套用設定後，旋轉一下 X 軸文字，以免字擠在一起
        plt.setp(ax_vol.get_xticklabels(), rotation=0, ha='center')
        
        fig.tight_layout()
        
        # 5. 轉為 Bytes 輸出
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return buf

    except Exception as e:
        logger.error(f"[Telegram] 生成圖表發生異常錯誤: {e}", exc_info=True)
        return None

def get_group_chart_bytes(group_name):
    groups = load_matrix_dict_analysis()
    if group_name not in groups: return None
    symbols = groups[group_name]
    raw_data = sys_db.load_kline('intraday_kline_live')
    if not raw_data: raw_data = sys_db.load_kline('intraday_kline_history')
    
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = Figure(figsize=(10, 5))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    all_z = []
    for sym in symbols:
        if sym not in raw_data: continue
        df = pd.DataFrame(raw_data[sym])
        if df.empty or 'close' not in df.columns: continue
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.sort_values(by='datetime')
        close = df['close']
        z = (close - close.mean()) / close.std() if close.std() != 0 else close - close.mean()
        all_z.append(pd.Series(z.values, index=df['datetime'], name=sym))
        ax.plot(df['datetime'], z, alpha=0.3, linewidth=1)
        
    if all_z:
        avg_z = pd.concat(all_z, axis=1).mean(axis=1)
        ax.plot(avg_z.index, avg_z.values, color='#E74C3C', linewidth=3, label="族群平均勢")
        
    ax.set_title(f"【{group_name}】族群連動分析 (Z-Score)", fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')
    
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf

# 動態進度條 K 線引擎
def update_kline_data(tg_progress_cb=None, target_date_str=None):
    def report_progress(p, msg):
        ui_dispatcher.progress_updated.emit(p, msg)
        if tg_progress_cb: tg_progress_cb(p, msg)

    client = init_esun_client()
    matrix_dict_analysis = load_matrix_dict_analysis()
    if not matrix_dict_analysis: return print("沒有任何族群資料，請先管理族群。")
    load_twse_name_map()

    ui_dispatcher.progress_visible.emit(True)
    
    # 判斷要採集的日期
    if target_date_str:
        valid_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
    else:
        # 如果沒傳入，用原本邏輯找最近交易日
        all_syms = [s for g in matrix_dict_analysis.values() for s in g]
        valid_date = get_valid_trading_day(client, all_syms)
        
    trading_day = valid_date.strftime('%Y-%m-%d')
    print(f"📅 【歷史採集模式】開始採集 {trading_day} 的資料 (已啟動單線程安全速率)...")

    # 將當前採集的日期寫入狀態，供回測引擎「鎖定」使用
    sys_db.save_state('last_fetched_date', trading_day)
    
    # 歷史回測「只」還原處置股現場，不再強求無法回溯的禁空名單
    report_progress(5, f"正在還原 {trading_day} 的歷史處置現場...")
    hist_dispo = fetch_active_disposition_on_date(trading_day)
    
    if hist_dispo:
        sys_db.save_state('disposition_stocks', hist_dispo)
        print(f"✅ 歷史採集：已鎖定 {trading_day} 當天 {len(hist_dispo)} 檔處置股。")
    else:
        sys_db.save_state('disposition_stocks', [])
        print(f"ℹ️ {trading_day} 當天無處置股資料。")
    
    # 這裡保留所有要分析的標的
    symbols_to_analyze = [sym for group in matrix_dict_analysis.values() for sym in group]
    total_syms = len(symbols_to_analyze)

    # 1. 抓取日K (為了昨收價)
    existing_daily_kline_data = sys_db.load_kline('daily_kline_history')
    # 單線程，避免併發封鎖
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_sym = {executor.submit(fetch_daily_kline_data, client, sym, 5, valid_date): sym for sym in symbols_to_analyze}
        for i, future in enumerate(as_completed(future_to_sym)):
            sym = future_to_sym[future]
            df = future.result()
            if not df.empty: existing_daily_kline_data[sym] = df.to_dict(orient='records')
    sys_db.save_kline('daily_kline_history', existing_daily_kline_data)

    # 2. 抓取一分K
    intraday_kline_data = {}
    # 單線程，避免併發封鎖
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_sym = {}
        for sym in symbols_to_analyze:
            daily_data = existing_daily_kline_data.get(sym, [])
            if not daily_data: continue
            sorted_daily_data = sorted(daily_data, key=lambda x: x['date'], reverse=True)
            # 取得昨收價邏輯
            yesterday_close_price = sorted_daily_data[1].get('close', 0) if len(sorted_daily_data) > 1 else sorted_daily_data[0].get('close', 0)
            future_to_sym[executor.submit(fetch_intraday_data, client, sym, trading_day, yesterday_close_price, "09:00", "13:30")] = sym

        for i, future in enumerate(as_completed(future_to_sym)):
            sym = future_to_sym[future]
            # 加上進度顯示，讓你知道現在抓到第幾檔
            report_progress(30 + int((i/total_syms)*70), f"歷史一分K: {sn(sym)} ({i+1}/{total_syms})")
            intraday_df = future.result()
            if intraday_df.empty: continue
            
            # 計算動能並寫入
            records = intraday_df.to_dict(orient='records')
            updated_records = []
            for j in range(len(records)):
                updated_records.append(calculate_pct_increase_and_highest(records[j], records[:j]))
            intraday_kline_data[sym] = updated_records

    # 清理舊數據：只刪除本次採集日期的資料，保護其他日期的歷史數據
    try:
        with sys_db.db_lock:
            with sys_db.conn:
                sys_db.conn.execute("DELETE FROM intraday_kline_history WHERE date = ?", (trading_day,))
    except Exception as e: logger.error(f"清理舊數據失敗: {e}")

    # 儲存到資料庫
    sys_db.save_kline('intraday_kline_history', intraday_kline_data)
    
    # 強制清空記憶體快取，強迫畫圖功能讀取最新的 SQLite 資料
    if hasattr(sys_state, 'in_memory_intraday'):
        sys_state.in_memory_intraday.clear()

    report_progress(100, f"{trading_day} 資料採集完畢")
    ui_dispatcher.progress_visible.emit(False)
    print("✅ 資料採集與記憶體同步完成！(單線程安全模式)")

# 回測核心引擎
def process_group_data(stock_data_collection, wait_minutes, hold_minutes, matrix_dict_analysis, actual_date, verbose=True, progress_callback=None, is_data_mining=False, risk_state=None, headless=False, _precomp=None):
    load_twse_name_map()
    in_position, has_exited, current_position, stop_loss_triggered, hold_time, reentry_count = False, False, None, False, 0, 0
    mining_active_positions = [] # 🌌 平行宇宙探勘模式專用持倉追蹤器
    trades_history = [] 
    events_log = [] 

    message_log = [] if not headless else _NullLog()
    detail_log  = [] if not headless else _NullLog()  # headless 模式：忽略所有 append，省 f-string 格式化開銷
    tracking_stocks: set[str] = set()
    leader, leader_peak_rise, leader_rise_before_decline = None, None, None
    in_waiting_period, waiting_time, pull_up_entry, limit_up_entry, first_c1_time = False, 0, False, False, None

    merged_df = None
    req_cols = ['time', 'rise', 'high', '漲停價', 'close', 'pct_increase', 'volume']
    
    # ⚡ 效能優化（precomp）：headless 回歸模式直接取用預計算資料，跳過全部 setup
    if _precomp is not None:
        _time_row_cache = _precomp['_time_row_cache']
        precalc_vols    = _precomp['precalc_vols']
        _f3avg_final    = _precomp['_f3avg_final']
        _f3avg_at       = _precomp['_f3avg_at']
        merged_records  = _precomp['merged_records']
        total_rows      = _precomp['total_rows']
        stock_data_collection = _precomp['stock_collection']
    else:
        # ⚡ 效能優化 1：預先計算開盤前三分鐘的成交量，避免在分鐘迴圈內重複操作 DataFrame
        precalc_vols = {}
        # ⚡ 效能優化 NEW-A：預建時間→列 dict，讓分鐘迴圈內的時間查詢從 O(n) 降為 O(1)
        _time_row_cache = {}  # {sym: {time_obj: row_dict}}
        for sym, df in stock_data_collection.items():
            if not all(c in df.columns for c in req_cols): continue
            for col in ['rise', 'high', 'close', 'volume', '漲停價', 'pct_increase']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # ⚡ 效能優化：ensure_continuous_time_series 保證 09:00/09:01/09:02 在 iloc[0/1/2]，直接存取取代 O(n) filter
            vols = {}
            for idx, t_str in enumerate(['09:00:00', '09:01:00', '09:02:00']):
                try: vols[t_str] = float(df.iloc[idx]['volume']) if len(df) > idx else None
                except Exception: vols[t_str] = None
            precalc_vols[sym] = vols

            # 預建 O(1) 時間查詢快取（全欄位，供分鐘迴圈內所有時間查詢使用）
            _time_row_cache[sym] = {r['time']: r for r in df.to_dict('records')}

            tmp = df[req_cols].rename(columns={c: f"{c}_{sym}" if c != 'time' else c for c in req_cols})
            merged_df = tmp if merged_df is None else pd.merge(merged_df, tmp, on='time', how='outer')

        if merged_df is None or merged_df.empty: return None, None, [], []
        merged_df.sort_values('time', inplace=True, ignore_index=True)

        # ⚡ 效能優化 NEW-B：FIRST3_AVG_VOL 在 09:02 之後就固定不變，預先算好避免每分鐘重算
        _f3avg_final = {}
        _f3avg_at = {'09:00:00': {}, '09:01:00': {}}
        for sym in precalc_vols.keys():
            all_v = [v for v in precalc_vols[sym].values() if v is not None]
            _f3avg_final[sym] = sum(all_v) / len(all_v) if all_v else 0
            for cutoff in ['09:00:00', '09:01:00']:
                sub = [v for t, v in precalc_vols[sym].items() if t <= cutoff and v is not None]
                _f3avg_at[cutoff][sym] = sum(sub) / len(sub) if sub else 0

        merged_records = merged_df.to_dict('records')
        total_rows = len(merged_records)

    total_profit = total_trades = 0; total_capital = 0

    for i, row in enumerate(merged_records):
        current_time = row['time']
        current_time_str = current_time.strftime('%H:%M:%S') if not isinstance(current_time, str) else current_time

        # ⚡ 效能優化 NEW-B：FIRST3_AVG_VOL 用預算快取，O(1) 查表取代每分鐘重算
        if current_time_str >= '09:02:00':
            FIRST3_AVG_VOL = _f3avg_final
        elif current_time_str in _f3avg_at:
            FIRST3_AVG_VOL = _f3avg_at[current_time_str]
        else:
            FIRST3_AVG_VOL = _f3avg_at.get('09:00:00', _f3avg_final)

        # 🌌 平行宇宙：多部位並行結算 (大數據探勘專用)
        if is_data_mining:
            surviving_mining_positions = []
            for pos in mining_active_positions:
                pos['hold_time'] += 1
                is_end_of_day = (current_time_str == '13:30:00')
                if is_end_of_day or (pos.get('actual_hold_minutes') and pos['hold_time'] >= pos['actual_hold_minutes']):
                    sel_df = stock_data_collection[pos['symbol']]
                    price_row = sel_df[sel_df['time'] == current_time]
                    actual_market_price = price_row.iloc[0]['close'] if not price_row.empty else pos['entry_price']
                    profit, rate = exit_trade(sel_df, pos['shares'], pos['entry_price'], pos['sell_cost'], pos['entry_fee'], pos['tax'], message_log, current_time, pos['hold_time'], pos['entry_time'], use_f_exit=is_end_of_day, sym=pos['symbol'])
                    if profit is not None:
                        total_trades += 1; total_profit += profit; total_capital += pos['sell_cost']
                        trades_history.append({'symbol': pos['symbol'], 'date': actual_date, 'entry_time': pos['entry_time'], 'entry_price': pos['entry_price'], 'exit_time': current_time_str, 'exit_price': actual_market_price, 'profit': profit, 'stop_loss': pos['stop_loss_threshold'], 'reason': '時間平倉', 'trigger_v': pos.get('trigger_v', 0), 'trigger_m': pos.get('trigger_m', 0), 'trigger_type': pos.get('trigger_type', '拉高')})
                    continue

                sel_df = stock_data_collection[pos['symbol']]
                now_row = sel_df[sel_df['time'] == current_time]
                if not now_row.empty:
                    h_now, thresh = truncate_to_two_decimals(now_row.iloc[0]['high']), truncate_to_two_decimals(pos['stop_loss_threshold'])
                    if h_now >= thresh:
                        exit_cost = pos['shares'] * thresh * 1000
                        exit_fee = int(exit_cost * (sys_config.transaction_fee*0.01) * (sys_config.transaction_discount*0.01))
                        profit = pos['sell_cost'] - exit_cost - pos['entry_fee'] - exit_fee - pos['tax']
                        rate = profit / pos['sell_cost'] * 100 if pos['sell_cost'] != 0 else 0.0
                        total_trades += 1; total_profit += profit; total_capital += pos['sell_cost']
                        trades_history.append({'symbol': pos['symbol'], 'date': actual_date, 'entry_time': pos['entry_time'], 'entry_price': pos['entry_price'], 'exit_time': current_time_str, 'exit_price': thresh, 'profit': profit, 'stop_loss': pos['stop_loss_threshold'], 'reason': '停損觸發', 'trigger_v': pos.get('trigger_v', 0), 'trigger_m': pos.get('trigger_m', 0), 'trigger_type': pos.get('trigger_type', '拉高')})
                        continue
                surviving_mining_positions.append(pos)
            mining_active_positions = surviving_mining_positions

        else:
            # 🔒 原版單一部位實戰結算邏輯
            if in_position and not has_exited:
                hold_time += 1
                is_end_of_day = (current_time_str == '13:30:00')
                if is_end_of_day or (current_position.get('actual_hold_minutes') and hold_time >= current_position['actual_hold_minutes']):
                    sel_df = stock_data_collection[current_position['symbol']]
                    price_row = sel_df[sel_df['time'] == current_time]
                    actual_market_price = price_row.iloc[0]['close'] if not price_row.empty else current_position['entry_price']
                    profit, rate = exit_trade(sel_df, current_position['shares'], current_position['entry_price'], current_position['sell_cost'], current_position['entry_fee'], current_position['tax'], message_log, current_time, hold_time, current_position['entry_time'], use_f_exit=is_end_of_day, sym=current_position['symbol'])
                    if profit is not None:
                        total_trades += 1; total_profit += profit; total_capital += current_position['sell_cost']
                        trades_history.append({'symbol': current_position['symbol'], 'date': actual_date, 'entry_time': current_position['entry_time'], 'entry_price': current_position['entry_price'], 'exit_time': current_time_str, 'exit_price': actual_market_price, 'profit': profit, 'stop_loss': current_position['stop_loss_threshold'], 'reason': '時間平倉', 'trigger_v': current_position.get('trigger_v', 0), 'trigger_m': current_position.get('trigger_m', 0), 'trigger_type': current_position.get('trigger_type', '拉高')})
                        detail_log.append(f"[{current_time_str}] 📤 出場(時間平倉) {sn(current_position['symbol'])} 進場:{current_position['entry_time']} 利潤:{int(profit)}元 ({rate:.2f}%)")
                    in_position, has_exited, current_position = False, True, None
                    continue

                sel_df = stock_data_collection[current_position['symbol']]
                now_row = sel_df[sel_df['time'] == current_time]
                if not now_row.empty:
                    h_now, thresh = truncate_to_two_decimals(now_row.iloc[0]['high']), truncate_to_two_decimals(current_position['stop_loss_threshold'])
                    if h_now >= thresh:
                        exit_cost = current_position['shares'] * thresh * 1000
                        exit_fee = int(exit_cost * (sys_config.transaction_fee*0.01) * (sys_config.transaction_discount*0.01))
                        profit = current_position['sell_cost'] - exit_cost - current_position['entry_fee'] - exit_fee - current_position['tax']
                        rate = profit / current_position['sell_cost'] * 100 if current_position['sell_cost'] != 0 else 0.0
                        message_log.append((current_time_str, f"{Fore.RED}停損！{sn(current_position['symbol'])} 進場:{current_position['entry_time']} 利潤 {int(profit)} 元 ({rate:.2f}%){Style.RESET_ALL}"))
                        detail_log.append(f"[{current_time_str}] 🛑 停損 {sn(current_position['symbol'])} 進場:{current_position['entry_time']} 利潤:{int(profit)}元 ({rate:.2f}%)")
                        total_trades += 1; total_profit += profit; total_capital += current_position['sell_cost']
                        trades_history.append({'symbol': current_position['symbol'], 'date': actual_date, 'entry_time': current_position['entry_time'], 'entry_price': current_position['entry_price'], 'exit_time': current_time_str, 'exit_price': thresh, 'profit': profit, 'stop_loss': current_position['stop_loss_threshold'], 'reason': '停損觸發', 'trigger_v': current_position.get('trigger_v', 0), 'trigger_m': current_position.get('trigger_m', 0), 'trigger_type': current_position.get('trigger_type', '拉高')})
                        
                        in_position, current_position = False, None

                        # 🛡️ 風控模組：停損計數 + 熔斷檢查
                        if risk_state and sys_config.risk_control_enabled:
                            risk_state['daily_stops'] = risk_state.get('daily_stops', 0) + 1
                            if risk_state['daily_stops'] >= sys_config.max_daily_stops:
                                risk_state['halted'] = True
                                detail_log.append(f"[{current_time_str}] 🔴 風控熔斷！累計停損 {risk_state['daily_stops']} 筆，當日停止進場")

                        reentry_success = False
                        if sys_config.allow_reentry and reentry_count < sys_config.max_reentry_times:
                            lookback_start = (datetime.combine(date.today(), current_time) - timedelta(minutes=sys_config.reentry_lookback_candles)).time()
                            for r_sym in stock_data_collection.keys():
                                df_sym = stock_data_collection[r_sym]
                                sub_df = df_sym[(df_sym['time'] >= lookback_start) & (df_sym['time'] <= current_time)]
                                for _, r_row in sub_df.iterrows():
                                    h, lup, pct, vol = r_row.get('high'), r_row.get('漲停價'), r_row.get('pct_increase'), r_row.get('volume')
                                    avgv = FIRST3_AVG_VOL.get(r_sym, 0)
                                    is_limit_up = (round(h, 2) >= round(lup, 2)) if (h is not None and pd.notna(h) and lup is not None and pd.notna(lup)) else False
                                    vol_c1 = (vol is not None and pd.notna(vol) and vol >= sys_config.min_volume_threshold)
                                    vol_c2 = (vol is not None and pd.notna(vol) and avgv > 0 and vol >= sys_config.volume_multiplier * avgv)
                                    is_pull_up = (pct is not None and pd.notna(pct) and pct >= sys_config.pull_up_pct_threshold and (vol_c1 or vol_c2))
                                    
                                    if is_limit_up or is_pull_up:
                                        reentry_success = True
                                        reentry_count += 1
                                        has_exited = False 
                                        tracking_stocks.clear(); tracking_stocks.add(r_sym)
                                        leader, in_waiting_period, waiting_time = r_sym, True, 0
                                        first_c1_time = datetime.combine(date.today(), current_time)
                                        cond_str = "漲停" if is_limit_up else "拉高"
                                        limit_up_entry, pull_up_entry = is_limit_up, not is_limit_up
                                        leader_rise_before_decline = float(r_row.get('highest', 0) or 0)
                                        detail_log.append(f"[{current_time_str}] 🔄 停損後再進場觸發！回溯發現 {sn(r_sym)} {cond_str}，啟動第 {reentry_count} 次監控")
                                        events_log.append({'time': current_time_str, 'symbol': r_sym, 'event': f'停損再進場({cond_str})', 'price': r_row['close'], 'volume': r_row['volume'], 'date': actual_date})
                                        break
                                if reentry_success: break
                                
                        if not reentry_success:
                            stop_loss_triggered = True
                            if sys_config.allow_reentry and reentry_count < sys_config.max_reentry_times:
                                has_exited = False 
                                reentry_count += 1
                                tracking_stocks.clear()
                                pull_up_entry = limit_up_entry = in_waiting_period = False
                                detail_log.append(f"[{current_time_str}] 🔄 停損結算。回溯無觸發，系統進入獵人模式 (剩餘次數: {sys_config.max_reentry_times - reentry_count})")
                                continue
                            else:
                                has_exited = True
                                break 
                        continue 
                continue 

        # 🚀 尾盤截止觸發：超過 cutoff_time_mins 後不再掃描新信號（但仍處理已進場的持倉）
        _cutoff = getattr(sys_config, 'cutoff_time_mins', 270)
        _past_cutoff = (_cutoff < 270 and i >= _cutoff)

        # 訊號掃描與條件觸發
        trigger_list = []
        for sym in precalc_vols.keys():
            if _past_cutoff: break  # 超過截止時間，不掃描新信號
            pct, vol, high, lup = row.get(f'pct_increase_{sym}'), row.get(f'volume_{sym}'), row.get(f'high_{sym}'), row.get(f'漲停價_{sym}')
            avgv = FIRST3_AVG_VOL.get(sym, 0)
            hit_limit = False
            if high is not None and pd.notna(high) and lup is not None and pd.notna(lup) and round(high, 2) >= round(lup, 2):
                if current_time_str == '09:00:00': hit_limit = True
                else:
                    prev_time = (datetime.combine(date.today(), current_time) - timedelta(minutes=1)).time()
                    # ⚡ 效能優化 NEW-A：O(1) 時間查詢取代 DataFrame filter
                    _prev_row = _time_row_cache.get(sym, {}).get(prev_time)
                    if _prev_row is None or round(_prev_row.get('high', lup), 2) < round(lup, 2): hit_limit = True
            if hit_limit: trigger_list.append({'symbol': sym, 'condition': 'limit_up'}); continue
            
            vol_c1 = (vol is not None and pd.notna(vol) and vol >= sys_config.min_volume_threshold)
            vol_c2 = (vol is not None and pd.notna(vol) and avgv > 0 and vol >= sys_config.volume_multiplier * avgv)
            vol_check = vol_c1 or vol_c2
            if pct is not None and pd.notna(pct) and pct >= sys_config.pull_up_pct_threshold and vol_check:
                trigger_list.append({'symbol': sym, 'condition': 'pull_up'})

        for item in trigger_list:
            sym, cond = item['symbol'], item['condition']
            if cond == 'limit_up':
                tracking_stocks.add(sym)
                leader, in_waiting_period, waiting_time = sym, True, 0
                if not pull_up_entry:  # 僅全新漲停觸發才重設 first_c1_time；從拉高升級則保留原時間
                    first_c1_time = datetime.combine(date.today(), current_time)
                pull_up_entry, limit_up_entry = False, True
                detail_log.append(f"[{current_time_str}] 🚀 {sn(sym)} 漲停觸發，升級為漲停進場模式")
                events_log.append({'time': current_time_str, 'symbol': sym, 'event': '漲停觸發', 'price': row.get(f'close_{sym}'), 'volume': row.get(f'volume_{sym}'), 'date': actual_date})
            else:
                if not pull_up_entry and not limit_up_entry: 
                    pull_up_entry, limit_up_entry = True, False
                    tracking_stocks.clear(); first_c1_time = datetime.combine(date.today(), current_time)
                tracking_stocks.add(sym)
                detail_log.append(f"[{current_time_str}] {sn(sym)} 拉高觸發，加入追蹤")
                events_log.append({'time': current_time_str, 'symbol': sym, 'event': '拉高觸發', 'price': row.get(f'close_{sym}'), 'volume': row.get(f'volume_{sym}'), 'date': actual_date})

        if pull_up_entry or limit_up_entry:
            for sym in precalc_vols.keys():
                if sym not in tracking_stocks and (pct := row.get(f'pct_increase_{sym}')) is not None and pd.notna(pct) and pct >= sys_config.follow_up_pct_threshold: 
                    tracking_stocks.add(sym)
                    events_log.append({'time': current_time_str, 'symbol': sym, 'event': '跟漲追蹤', 'price': row.get(f'close_{sym}'), 'volume': row.get(f'volume_{sym}'), 'date': actual_date})

        if tracking_stocks:
            # 🔧 追蹤期 DTW：從 09:00 到當前時間，需至少 10 根 K 棒才開始檢查（避免開盤前幾分鐘不穩定）
            if leader and not in_waiting_period and len(tracking_stocks) > 1 and sys_config.similarity_threshold > 0:
                dtw_window_start = time(9, 0)
                # 計算目前有幾根 K 棒
                minutes_since_open = (datetime.combine(date.today(), current_time) - datetime.combine(date.today(), time(9, 0))).seconds // 60
                if minutes_since_open >= 10:  # 至少 10 分鐘的數據才檢查
                    to_remove_dtw = [sym for sym in tracking_stocks if sym != leader and calculate_dtw_pearson(stock_data_collection[leader], stock_data_collection[sym], dtw_window_start, current_time) < sys_config.similarity_threshold]
                    for sym in to_remove_dtw:
                        tracking_stocks.remove(sym)
                        detail_log.append(f"[{current_time_str}] [DTW剔除] {sn(sym)} 相似度不足 < {sys_config.similarity_threshold}")

            max_sym, max_rise = None, None
            for sym in tracking_stocks:
                if (r := row.get(f'rise_{sym}')) is not None and pd.notna(r) and (max_rise is None or r > max_rise): max_rise, max_sym = r, sym

            if leader is None:
                leader = max_sym
                leader_peak_rise = max_rise
            elif max_sym:
                if leader_peak_rise is None: leader_peak_rise = max_rise
                if max_sym == leader:
                    if max_rise > leader_peak_rise: leader_peak_rise = max_rise
                else:
                    if max_rise > leader_peak_rise:
                        detail_log.append(f"[{current_time_str}] ✨ 領漲替換：{sn(leader)} ➔ {sn(max_sym)}")
                        events_log.append({'time': current_time_str, 'symbol': max_sym, 'event': '成為新領漲', 'price': row.get(f'close_{max_sym}'), 'volume': row.get(f'volume_{max_sym}'), 'date': actual_date})
                        leader, leader_peak_rise, leader_rise_before_decline, in_waiting_period, waiting_time = max_sym, max_rise, None, False, 0
                        first_c1_time = datetime.combine(date.today(), current_time)
            
            if leader:
                h_now = row.get(f'high_{leader}')
                prev_time = (datetime.combine(date.today(), current_time) - timedelta(minutes=1)).time()
                # ⚡ 效能優化 NEW-A：O(1) 查表取代 DataFrame filter
                _prev_ldr = _time_row_cache.get(leader, {}).get(prev_time)
                if _prev_ldr is not None and pd.notna(h_now) and h_now <= _prev_ldr.get('high', h_now + 1) and not in_waiting_period:
                    leader_highest = float(_time_row_cache.get(leader, {}).get(current_time, {}).get('highest', 0))
                    in_waiting_period, waiting_time, leader_rise_before_decline = True, 0, leader_highest
                    detail_log.append(f"[{current_time_str}] 領漲 {sn(leader)} 反轉，確立天花板 {leader_highest:.2f}%，開始等待 {wait_minutes} 分鐘")
                    events_log.append({'time': current_time_str, 'symbol': leader, 'event': '確立天花板', 'price': row.get(f'close_{leader}'), 'volume': row.get(f'volume_{leader}'), 'date': actual_date})

        if in_waiting_period:
            # 🔧 DTW 窗口永遠從 09:00 開始，確保有足夠數據點
            window_start_t = time(9, 0)
            # 等待期：每分鐘計算 DTW，但只在最後一分鐘才剔除（避免反轉期間暫時性不同步）
            if waiting_time >= wait_minutes - 1 and sys_config.similarity_threshold > 0:
                to_remove = [sym for sym in tracking_stocks if sym != leader and calculate_dtw_pearson(stock_data_collection[leader], stock_data_collection[sym], window_start_t, current_time) < sys_config.similarity_threshold]
                for sym in to_remove:
                    tracking_stocks.remove(sym)
                    detail_log.append(f"[{current_time_str}] [DTW剔除] {sn(sym)} 相似度不足 < {sys_config.similarity_threshold}")

            if leader and leader_rise_before_decline is not None and (h_now := row.get(f"high_{leader}")) is not None and pd.notna(h_now) and h_now > leader_rise_before_decline:
                detail_log.append(f"[{current_time_str}] 🔥 領漲 {sn(leader)} 突破前高 {leader_rise_before_decline}，漲勢延續，中斷等待")
                events_log.append({'time': current_time_str, 'symbol': leader, 'event': '突破前高', 'price': h_now, 'volume': row.get(f'volume_{leader}'), 'date': actual_date})
                leader_highest = float(_time_row_cache.get(leader, {}).get(current_time, {}).get('highest', 0) or 0)
                leader_rise_before_decline, in_waiting_period, waiting_time = leader_highest, False, 0
                continue  

            if waiting_time >= wait_minutes:
                in_waiting_period, waiting_time = False, 0
                eligible = []
                leader_rise_now = row.get(f'rise_{leader}')  # 取得領漲股目前漲幅（指標一：相對位置用）
                for sym in set(tracking_stocks):
                    if sym == leader: continue
                    df = stock_data_collection[sym]
                    later = df[(df['time'] >= first_c1_time.time()) & (df['time'] <= current_time)]
                    if later.empty: continue
                    
                    sub_avg_vol = later['volume'].mean() if not later.empty else 0
                    sub_max_vol = later['volume'].max() if not later.empty else 0

                    if sub_avg_vol < sys_config.wait_min_avg_vol and sub_max_vol < sys_config.wait_max_single_vol:
                        detail_log.append(f"[{current_time_str}] [篩除] {sn(sym)} 等待期量能不足 均:{sub_avg_vol:.0f}張(需≥{sys_config.wait_min_avg_vol}) 最大:{sub_max_vol:.0f}張(需≥{sys_config.wait_max_single_vol})")
                        continue

                    avgv = FIRST3_AVG_VOL.get(sym, 0)
                    vol_cond = (later['volume'] >= sys_config.volume_multiplier * avgv) if avgv > 0 else (later['volume'] >= sys_config.min_volume_threshold)
                    if not (vol_cond | (later['volume'] >= sys_config.min_volume_threshold)).any():
                        detail_log.append(f"[{current_time_str}] [篩除] {sn(sym)} 無放量K棒 (均量倍數:{sys_config.volume_multiplier}x 絕對量:{sys_config.min_volume_threshold}張)")
                        continue

                    if len(later) >= 2 and later.iloc[-1]['rise'] > later.iloc[:-1]['rise'].max() + sys_config.pullback_tolerance:
                        detail_log.append(f"[{current_time_str}] [篩除] {sn(sym)} 回調過度 末漲:{later.iloc[-1]['rise']:.2f}% > 等待期峰值:{later.iloc[:-1]['rise'].max():.2f}%+{sys_config.pullback_tolerance}")
                        continue

                    rise_now, price_now = row.get(f'rise_{sym}'), row.get(f'close_{sym}')
                    # 🔧 修正：price上限 = capital(萬) * 10000 / 1000 * 1.5 = capital * 15（一張最高價格）
                    max_entry_price = sys_config.capital_per_stock * 15
                    if rise_now is None or pd.isna(rise_now) or not (sys_config.rise_lower_bound <= rise_now <= sys_config.rise_upper_bound) or price_now is None or pd.isna(price_now) or price_now > max_entry_price:
                        detail_log.append(f"[{current_time_str}] [篩除] {sn(sym)} 漲幅/價格不符 rise:{rise_now} 範圍:[{sys_config.rise_lower_bound},{sys_config.rise_upper_bound}] 價格:{price_now} 上限:{max_entry_price}")
                        continue
                    # 🔑 指標一：相對位置 — 跟漲股漲幅必須落後領漲股至少 min_lag_pct%
                    if leader_rise_now is not None and not pd.isna(leader_rise_now) and (leader_rise_now - rise_now) < sys_config.min_lag_pct:
                        detail_log.append(f"[{current_time_str}] [篩除] {sn(sym)} 落後不足 領漲:{leader_rise_now:.2f}%-跟漲:{rise_now:.2f}%={leader_rise_now-rise_now:.2f}% < {sys_config.min_lag_pct}%")
                        continue
                    # ⚡ 效能優化 NEW-A：O(1) 查表取代 DataFrame filter
                    sym_row = _time_row_cache.get(sym, {}).get(current_time)
                    if sym_row is None: continue
                    # 🔑 指標三：高度門檻 — 標的當日最高漲幅需 ≥ min_height_pct%
                    prev_close = float(sym_row.get('昨日收盤價', 0) or 0)
                    _highest = float(sym_row.get('highest', 0) or 0)
                    if prev_close > 0 and (_highest - prev_close) / prev_close * 100 < sys_config.min_height_pct:
                        ht = (_highest - prev_close) / prev_close * 100
                        detail_log.append(f"[{current_time_str}] [篩除] {sn(sym)} 高度不足 當日最高漲幅:{ht:.2f}% < {sys_config.min_height_pct}%")
                        continue
                    # 🔑 波動度門檻：09:10 後才啟用，過濾靜止不動的股票
                    if current_time >= time(9, 10) and sys_config.volatility_min_range > 0:
                        full_day_vol = stock_data_collection[sym][stock_data_collection[sym]['time'] <= current_time]
                        if not full_day_vol.empty:
                            d_high, d_low = full_day_vol['high'].max(), full_day_vol['low'].min()
                            pc = float(sym_row.get('昨日收盤價', 0) or 0)
                            if pc > 0:
                                vol_range = (d_high - d_low) / pc * 100
                                if vol_range < sys_config.volatility_min_range:
                                    detail_log.append(f"[{current_time_str}] [篩除] {sn(sym)} 波動度不足 {vol_range:.2f}% < {sys_config.volatility_min_range}%")
                                    continue
                    # 🔑 稀薄量能門檻：全日均量需 ≥ min_eligible_avg_vol 張/分（過濾委買賣每格只有1~2張的幽靈股）
                    full_day_data = stock_data_collection[sym][stock_data_collection[sym]['time'] <= current_time]
                    if sys_config.min_eligible_avg_vol > 0 and full_day_data['volume'].mean() < sys_config.min_eligible_avg_vol:
                        detail_log.append(f"[{current_time_str}] [篩除] {sn(sym)} 全日均量不足 {full_day_data['volume'].mean():.1f}張 < {sys_config.min_eligible_avg_vol}張")
                        continue
                    # 🔑 股價下限：過濾低價股，預設0（不啟用）
                    if sys_config.min_close_price > 0 and price_now < sys_config.min_close_price:
                        detail_log.append(f"[{current_time_str}] [篩除] {sn(sym)} 股價不足 {price_now:.1f}元 < {sys_config.min_close_price}元")
                        continue
                    eligible.append({'symbol': sym, 'rise': rise_now, 'row': sym_row})

                if not eligible and getattr(sys_config, 'allow_leader_entry', False) and leader and leader in stock_data_collection:
                    # 🔑 Leader 進場：85克策略 — 無合格跟漲股時，直接做空反轉中的領漲股
                    ldr_df = stock_data_collection[leader]
                    ldr_later = ldr_df[(ldr_df['time'] >= first_c1_time.time()) & (ldr_df['time'] <= current_time)]
                    ldr_rise = row.get(f'rise_{leader}')
                    ldr_price = row.get(f'close_{leader}')
                    ldr_max_price = sys_config.capital_per_stock * 15  # capital(萬) × 10 × 1.5倍緩衝
                    ldr_row = ldr_df.loc[ldr_df['time'] == current_time]
                    if (not ldr_later.empty and ldr_rise is not None and not pd.isna(ldr_rise)
                        and sys_config.rise_lower_bound <= ldr_rise <= sys_config.rise_upper_bound
                        and ldr_price is not None and not pd.isna(ldr_price) and ldr_price <= ldr_max_price
                        and not ldr_row.empty):
                        ldr_sym_row = ldr_row.iloc[0]
                        eligible.append({'symbol': leader, 'rise': ldr_rise, 'row': ldr_sym_row})
                        detail_log.append(f"[{current_time_str}] 🎯 Leader進場：{sn(leader)} 無合格follower，嘗試直接做空領漲股 rise={ldr_rise:.2f}%")

                if not eligible:
                    detail_log.append(f"[{current_time_str}] ⚠️ 等待結束後無合格跟漲標的，放棄此波機會")
                    pull_up_entry = limit_up_entry = False; tracking_stocks.clear()
                else:
                    eligible.sort(key=lambda x: x['rise'])  # 由低到高：漲幅最落後的排在前面，優先進場

                    if is_data_mining:
                        # 🌌 探勘模式：所有合格股票全部進場，且不鎖定族群
                        for chosen in eligible:
                            rowch, entry_p = chosen['row'], chosen['row']['close']
                            shares = round((sys_config.capital_per_stock*10000)/(entry_p*1000))
                            if shares == 0: continue
                            
                            sell_cost = shares * entry_p * 1000
                            entry_fee, tax = int(sell_cost * (sys_config.transaction_fee*0.01) * (sys_config.transaction_discount*0.01)), int(sell_cost * (sys_config.trading_tax*0.01))
                            gap, tick = get_stop_loss_config(entry_p)
                            highest_on_entry = float(rowch.get('highest', 0) or 0) or entry_p
                            stop_thr = entry_p + gap/1000 if (highest_on_entry-entry_p)*1000 < gap else highest_on_entry + tick
                            # 🚀 停損緩衝：與回歸 regression_evaluator 對齊
                            _sl_cushion_m = getattr(sys_config, 'sl_cushion_pct', 0)
                            if _sl_cushion_m > 0:
                                stop_thr = round(stop_thr + entry_p * (_sl_cushion_m / 100.0), 2)

                            limit_up = row.get(f'漲停價_{chosen["symbol"]}')
                            if limit_up and pd.notna(limit_up):
                                tick_for_limit = 0.01 if limit_up < 10 else 0.05 if limit_up < 50 else 0.1 if limit_up < 100 else 0.5 if limit_up < 500 else 1 if limit_up < 1000 else 5
                                if stop_thr > limit_up - 2 * tick_for_limit: stop_thr = limit_up - 2 * tick_for_limit

                            # ======== 🚀 探勘模式：風控濾網 ========
                            max_risk_amount = (stop_thr - entry_p) * 1000
                            if max_risk_amount > get_max_risk_for_price(entry_p):
                                continue # 探勘模式也必須過濾超額風險，保持數據真實性
                            # =======================================

                            actual_hold_minutes = hold_minutes
                            if actual_hold_minutes is not None and (datetime.combine(date.today(), current_time) + timedelta(minutes=actual_hold_minutes)).time() >= time(13, 26): actual_hold_minutes = None
                            
                            trigger_v, trigger_m = 0, 0
                            try:
                                t_row = stock_data_collection[chosen['symbol']]
                                t_row = t_row[t_row['time'] == first_c1_time.time()]
                                if not t_row.empty:
                                    trigger_v = t_row.iloc[0]['volume']
                                    avgv = FIRST3_AVG_VOL.get(chosen['symbol'], 0)
                                    trigger_m = trigger_v / avgv if avgv > 0 else 0
                            except Exception: pass

                            mining_active_positions.append({
                                'symbol': chosen['symbol'], 'shares': shares, 'entry_price': entry_p, 
                                'sell_cost': sell_cost, 'entry_fee': entry_fee, 'tax': tax, 
                                'entry_time': current_time_str, 'stop_loss_threshold': stop_thr, 
                                'actual_hold_minutes': actual_hold_minutes, 'hold_time': 0,
                                'trigger_v': trigger_v, 'trigger_m': trigger_m, 
                                'trigger_type': '漲停' if limit_up_entry else '拉高'
                            })
                        # 進入後重置追蹤狀態，下午隨時可以再次觸發波段
                        pull_up_entry = limit_up_entry = False; tracking_stocks.clear()

                    else:
                        # 🔒 實戰回測模式：嚴格濾網 + 備胎換股機制
                        trade_success = False

                        # 🛡️ 風控模組：檢查每日進場上限和熔斷
                        if risk_state and sys_config.risk_control_enabled:
                            if risk_state.get('halted', False):
                                detail_log.append(f"[{current_time_str}] 🚫 風控熔斷中（累計停損{risk_state.get('daily_stops',0)}筆），跳過進場")
                                pull_up_entry = limit_up_entry = False; tracking_stocks.clear()
                                continue
                            if risk_state.get('daily_entries', 0) >= sys_config.max_daily_entries:
                                detail_log.append(f"[{current_time_str}] 🚫 已達每日進場上限 {sys_config.max_daily_entries} 檔，跳過進場")
                                pull_up_entry = limit_up_entry = False; tracking_stocks.clear()
                                continue

                        while eligible:
                            chosen = eligible[0]  # 取漲幅最落後的標的（已由低到高排序）
                            rowch, entry_p = chosen['row'], chosen['row']['close']

                            shares = round((sys_config.capital_per_stock*10000)/(entry_p*1000))
                            if shares == 0:
                                eligible.pop(0)
                                continue
                                
                            sell_cost = shares * entry_p * 1000
                            entry_fee, tax = int(sell_cost * (sys_config.transaction_fee*0.01) * (sys_config.transaction_discount*0.01)), int(sell_cost * (sys_config.trading_tax*0.01))
                            gap, tick = get_stop_loss_config(entry_p)
                            highest_on_entry = float(rowch.get('highest', 0) or 0) or entry_p
                            stop_thr = entry_p + gap/1000 if (highest_on_entry-entry_p)*1000 < gap else highest_on_entry + tick
                            # 🚀 停損緩衝：與回歸 regression_evaluator 對齊
                            _sl_cushion = getattr(sys_config, 'sl_cushion_pct', 0)
                            if _sl_cushion > 0:
                                stop_thr = round(stop_thr + entry_p * (_sl_cushion / 100.0), 2)

                            limit_up = row.get(f'漲停價_{chosen["symbol"]}')
                            if limit_up and pd.notna(limit_up):
                                tick_for_limit = 0.01 if limit_up < 10 else 0.05 if limit_up < 50 else 0.1 if limit_up < 100 else 0.5 if limit_up < 500 else 1 if limit_up < 1000 else 5
                                if stop_thr > limit_up - 2 * tick_for_limit: stop_thr = limit_up - 2 * tick_for_limit

                            # ======== 🚀 回測防禦核心：風險斷頭台 (方案七) ========
                            max_risk_amount = (stop_thr - entry_p) * 1000
                            _risk_limit = get_max_risk_for_price(entry_p)
                            if max_risk_amount > _risk_limit:
                                msg = f"⚠️ 回測拒絕進場！{sn(chosen['symbol'])} 停損風險 {int(max_risk_amount)} > {int(_risk_limit)}。啟動備胎換股！"
                                detail_log.append(f"[{current_time_str}] {msg}")
                                # 將高風險股票剔除，回到 while 迴圈開頭找下一個落後標的
                                eligible.pop(0)
                                continue
                            # ======================================================

                            actual_hold_minutes = hold_minutes
                            if actual_hold_minutes is not None and (datetime.combine(date.today(), current_time) + timedelta(minutes=actual_hold_minutes)).time() >= time(13, 26): actual_hold_minutes = None
                            
                            trigger_v, trigger_m = 0, 0
                            try:
                                t_row = stock_data_collection[chosen['symbol']]
                                t_row = t_row[t_row['time'] == first_c1_time.time()]
                                if not t_row.empty:
                                    trigger_v = t_row.iloc[0]['volume']
                                    avgv = FIRST3_AVG_VOL.get(chosen['symbol'], 0)
                                    trigger_m = trigger_v / avgv if avgv > 0 else 0
                            except Exception: pass

                            current_position = {
                                'symbol': chosen['symbol'], 'shares': shares, 'entry_price': entry_p, 
                                'sell_cost': sell_cost, 'entry_fee': entry_fee, 'tax': tax, 
                                'entry_time': current_time_str, 'stop_loss_threshold': stop_thr, 
                                'actual_hold_minutes': actual_hold_minutes,
                                'trigger_v': trigger_v, 'trigger_m': trigger_m, 
                                'trigger_type': '漲停' if limit_up_entry else '拉高'
                            }
                            in_position, has_exited, hold_time = True, False, 0
                            pull_up_entry = limit_up_entry = False; tracking_stocks.clear()
                            
                            message_log.append((current_time_str, f"{Fore.GREEN}進場！{sn(chosen['symbol'])} {shares}張 價 {entry_p:.2f} 停損 {stop_thr:.2f}{Style.RESET_ALL}"))
                            detail_log.append(f"[{current_time_str}] ✅ 進場 {sn(chosen['symbol'])} {shares}張 價:{entry_p:.2f} 停損:{stop_thr:.2f} 觸發類型:{current_position.get('trigger_type','')}")

                            # 🛡️ 風控模組：更新進場計數
                            if risk_state:
                                risk_state['daily_entries'] = risk_state.get('daily_entries', 0) + 1

                            trade_success = True
                            break # 進場成功，跳出 while eligible 備胎迴圈

                        if not trade_success:
                            # 所有的備胎都被風控濾網擋下來了 (名單 pop 到空了)
                            detail_log.append(f"[{current_time_str}] ⚠️ 該族群所有合格標的皆因風險過高被剔除，取消此次進場")
                            pull_up_entry = limit_up_entry = False; tracking_stocks.clear()

            else:
                waiting_time += 1
                if verbose: message_log.append((current_time_str, f"⏳ 領漲 {sn(leader)} 反轉，等待第 {waiting_time} 分鐘"))

    if not headless:
        message_log.sort(key=lambda x: x[0])
        for t, msg in message_log: print(f"[{t}] {msg}")

    # 詳細 log 直接寫入磁碟（不走 Qt 訊號，不影響 UI 效能）
    if detail_log and not headless:
        try:
            _log_dir = os.path.join(os.getcwd(), "temp")
            os.makedirs(_log_dir, exist_ok=True)
            _log_path = os.path.join(_log_dir, f"backtest_detail_{actual_date}.log")
            with open(_log_path, "a", encoding="utf-8") as _lf:
                _lf.write("\n".join(detail_log) + "\n")
        except Exception:
            pass

    if total_trades:
        group_rate = total_profit / total_capital * 100 if total_capital else 0.0
        c = RED if total_profit > 0 else ("" if total_profit <= 0 else "")
        if verbose and not headless: print(f"{c}🏁 族群測試完成，利潤：{int(total_profit)} 元 (報酬率：{group_rate:.2f}%){RESET}\n")
        return total_profit, total_capital, trades_history, events_log
    else:
        if verbose and not headless: print(f"🏁 族群測試完成，無任何交易觸發。\n")
        return None, None, [], events_log

# =================================================================================
# 盤中邏輯核心
def process_live_trading_logic(symbols_to_analyze, current_time_str, wait_minutes, hold_minutes, message_log, in_position, has_exited, current_position, hold_time, already_entered_stocks, stop_loss_triggered, final_check_active, final_check_count, in_waiting_period, waiting_time, leader, tracking_stocks, previous_rise_values, leader_peak_rise, leader_rise_before_decline, first_condition_one_time, can_trade, group_positions, nb_matrix_path="nb_matrix_dict.json"):
    monitor_stop_loss_orders(group_positions)
    if sys_state.quit_flag: threading.Thread(target=show_exit_menu, daemon=True).start(); sys_state.quit_flag = False
    try: trading_time = datetime.strptime(current_time_str, "%H:%M").time()
    except ValueError: return
    trading_txt = trading_time.strftime("%H:%M:%S")
    real_now_txt = datetime.now().strftime("%H:%M:%S")
    log_time_tag = f"{real_now_txt} | K線{trading_time.strftime('%H:%M')}"
    # 建立一個包含日期的 datetime，方便後續做時間相減 (光年計時法)
    current_dt_full = datetime.combine(date.today(), trading_time)

    nb_dict = load_group_symbols()
    consolidated_symbols = nb_dict.get("consolidated_symbols", {})
    if not isinstance(consolidated_symbols, dict) or not consolidated_symbols: return

    if sys_state.in_memory_intraday:
        with sys_state.lock: auto_intraday_data = sys_state.in_memory_intraday.copy()
    else:
        auto_intraday_data = sys_db.load_kline('intraday_kline_live')
        if not auto_intraday_data: return

    stock_df = {}
    for sym in symbols_to_analyze:
        df = pd.DataFrame(auto_intraday_data.get(sym, [])).copy()
        if not df.empty:
            for col in ['rise', 'high', 'low', 'open', 'close', 'volume', '漲停價', 'highest', 'pct_increase']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df["time"] = pd.to_datetime(df["time"].astype(str), format='mixed').dt.time
            df.sort_values("time", inplace=True); df.reset_index(drop=True, inplace=True)
        stock_df[sym] = df

    FIRST3_AVG_VOL: dict[str, float] = {}
    for sym, df in stock_df.items():
        if df.empty or "time" not in df.columns: FIRST3_AVG_VOL[sym] = 0; continue
        valid_times = [datetime.strptime(t, "%H:%M:%S").time() for t in ["09:00:00", "09:01:00", "09:02:00"] if datetime.strptime(t, "%H:%M:%S").time() <= trading_time]
        if not valid_times: FIRST3_AVG_VOL[sym] = 0
        else:
            first3 = df[df["time"].isin(valid_times)]
            FIRST3_AVG_VOL[sym] = first3["volume"].mean() if not first3.empty else 0

    with sys_state.lock:
        for sym, pos_info in list(sys_state.open_positions.items()):
            df_sym = stock_df.get(sym, pd.DataFrame())
            if not df_sym.empty:
                row_now = df_sym[df_sym["time"] == trading_time]
                if not row_now.empty:
                    cur_high = row_now.iloc[0]["high"]
                    sl_price = pos_info.get("stop_loss", 9999)
                    if cur_high >= sl_price:
                        # 🔧 grace period 第二輪：確認無成交後移除
                        if pos_info.get('_sl_grace_done', False) and pos_info.get('filled_shares', 0) == 0:
                            msg_remove = f"🚫 {sn(sym)} 確認無成交，從持倉監控中移除。"
                            message_log.append((log_time_tag, msg_remove))
                            sys_state.open_positions.pop(sym, None)
                            sys_db.save_state('current_position', sys_state.open_positions)
                            grp_f = next((g for g, gs in load_matrix_dict_analysis().items() if sym in gs), None)
                            if grp_f and grp_f in group_positions: group_positions[grp_f] = False
                            continue

                        # 首次偵測到停損
                        if not pos_info.get('sl_printed', False):
                            msg_sl = f"💥 [系統偵測] {sn(sym)} 當前最高價 {cur_high} 觸及停損 {sl_price:.2f}！"
                            message_log.append((log_time_tag, f"{RED}{msg_sl}{RESET}"))
                            pos_info['sl_printed'] = True

                            # 🚨 自動防護：撤銷尚未成交的進場空單委託
                            try:
                                sys_state.api.update_status()
                                for t in sys_state.api.list_trades():
                                    if t.contract.code == sym and getattr(t.order, 'action', '') == sj.constant.Action.Sell:
                                        if getattr(t.status, 'status', '').name not in ["Filled", "Cancelled", "Failed"]:
                                            sys_state.api.cancel_order(t)
                                            print(f"[防護機制] 已安全撤銷 {sn(sym)} 尚未成交的進場空單！")
                            except Exception as e: logger.warning(f"撤單檢查失敗: {e}")

                            # 🆕 安全網：如果有成交但尚未平倉，直接送 Buy 回補
                            remaining = pos_info.get('filled_shares', 0) - pos_info.get('covered_shares', 0)
                            if remaining > 0:
                                msg_cover = f"🚨 [安全網] {sn(sym)} 停損觸發，自動送出買回 {remaining} 張！"
                                message_log.append((log_time_tag, f"{RED}{msg_cover}{RESET}"))
                                threading.Thread(target=close_one_stock, args=(sym,), daemon=True).start()
                                _play_alert('stop_loss')

                            # 🔧 grace period：未成交先等一輪再移除
                            if pos_info.get('filled_shares', 0) == 0:
                                pos_info['_sl_grace_done'] = True
                                msg_grace = f"⏳ {sn(sym)} 觸及停損但尚無成交回報，等待下一根 K 線確認..."
                                message_log.append((log_time_tag, f"{YELLOW}{msg_grace}{RESET}"))

    # 邏輯防護牆檢測
    for sym, df_debug in stock_df.items():
        if df_debug.empty: continue
        row_debug = df_debug[df_debug["time"] == trading_time]
        if not row_debug.empty:
            r_d = row_debug.iloc[0]
            p2 = r_d.get('pct_increase', 0)
            if pd.notna(p2) and float(p2) >= getattr(sys_config, 'pull_up_pct_threshold', 1.4):
                vol = r_d.get('volume', 0)
                avg_v = FIRST3_AVG_VOL.get(sym, 0)
                grp_found = next((g_name for g_name, g_syms in consolidated_symbols.items() if sym in g_syms), None)
                if not grp_found: print(f"   ➤ ❌ 阻擋原因: 該股票未歸屬於任何族群。")
                else:
                    gstat = group_positions.get(grp_found)
                    stat_str = gstat if isinstance(gstat, str) else gstat.get('status', '未進場/未觀察') if isinstance(gstat, dict) else str(gstat)
                    print(f"   ➤ 所屬族群: {grp_found} | 當前狀態: {stat_str}")
                    if stat_str == "已進場": print(f"   ➤ ❌ 阻擋原因: 該族群已經進場，不會再重複觸發新股票。")
                    elif stat_str == "觀察中": print(f"   ➤ ❌ 阻擋原因: 該族群已經在觀察中，正在等待洗盤時間。")

    if not hasattr(sys_state, 'reentry_counts'): sys_state.reentry_counts = {}
    
    for grp, gstat in list(group_positions.items()):
        if gstat == "已進場" or (isinstance(gstat, dict) and gstat.get("status") == "已進場"):
            still_open = any(sym in sys_state.open_positions for sym in consolidated_symbols.get(grp, []))
            if not still_open:
                group_positions[grp] = "停損結算中" 
                reentry_count = sys_state.reentry_counts.get(grp, 0)
                
                reentry_success = False
                if sys_config.allow_reentry and reentry_count < sys_config.max_reentry_times:
                    lookback_start = (current_dt_full - timedelta(minutes=sys_config.reentry_lookback_candles)).time()
                    
                    for r_sym in consolidated_symbols.get(grp, []):
                        if r_sym not in stock_df or stock_df[r_sym].empty: continue
                        df_sym = stock_df[r_sym]
                        sub_df = df_sym[(df_sym['time'] >= lookback_start) & (df_sym['time'] <= trading_time)]
                        
                        for _, r_row in sub_df.iterrows():
                            h, lup, pct, vol = r_row.get('high'), r_row.get('漲停價'), r_row.get('pct_increase'), r_row.get('volume')
                            avgv = FIRST3_AVG_VOL.get(r_sym, 0)
                            
                            is_limit_up = (round(h, 2) >= round(lup, 2)) if (h is not None and pd.notna(h) and lup is not None and pd.notna(lup)) else False
                            vol_check = (vol >= sys_config.min_volume_threshold) or (avgv > 0 and vol >= sys_config.volume_multiplier * avgv)
                            is_pull_up = (pct is not None and pd.notna(pct) and pct >= sys_config.pull_up_pct_threshold and vol is not None and pd.notna(vol) and vol_check)
                            
                            if is_limit_up or is_pull_up:
                                reentry_success = True
                                sys_state.reentry_counts[grp] = reentry_count + 1
                                cond_str = "漲停" if is_limit_up else "拉高"
                                
                                group_positions[grp] = {
                                    "status": "觀察中", 
                                    "trigger": f"{cond_str}進場", 
                                    "start_time": current_dt_full, 
                                    "tracking": {r_sym: {"join_time": current_dt_full, "base_vol": vol, "base_rise": r_row['rise']}}, 
                                    "leader": r_sym,
                                    "leader_peak": r_row['rise'],
                                    "leader_reversal_rise": float(r_row.get('highest', h) or 0),
                                    "wait_counter": 0,
                                    "wait_start": current_dt_full
                                }
                                msg = f"🔄 停損後再進場觸發！回溯發現 {sn(r_sym)} {cond_str}，啟動第 {reentry_count+1} 次監控"
                                message_log.append((log_time_tag, f"{Fore.MAGENTA}{msg}{Style.RESET_ALL}"))
                                tg_bot.send_message(f"🔄 <b>停損再進場啟動</b>\n族群: {grp}\n標的: <code>{sn(r_sym)}</code>\n條件: {cond_str}\n系統已重新鎖定目標！", force=True)
                                break
                        if reentry_success: break
                
                if not reentry_success:
                    if sys_config.allow_reentry and reentry_count < sys_config.max_reentry_times:
                        sys_state.reentry_counts[grp] = reentry_count + 1
                        group_positions.pop(grp, None) 
                        msg = f"🔄 停損結算。回溯無觸發，系統進入獵人模式等待下一次轉折 (剩餘次數: {sys_config.max_reentry_times - (reentry_count+1)})"
                        message_log.append((log_time_tag, f"{Fore.MAGENTA}{msg}{Style.RESET_ALL}"))
                        tg_bot.send_message(f"🔄 <b>系統進入獵人模式</b>\n族群: {grp}\n系統保持空手監控，等待下一次進場時機！", force=True)
                    else:
                        group_positions[grp] = False 

    trigger_list = []
    _cutoff = getattr(sys_config, 'cutoff_time_mins', 270)
    _cutoff_time = time(9 + _cutoff // 60, _cutoff % 60) if _cutoff < 270 else time(13, 30)
    if trading_time >= _cutoff_time: pass
    else:
        for grp, syms in consolidated_symbols.items():
            gstat = group_positions.get(grp)
            is_group_entered = (gstat == "已進場" or (isinstance(gstat, dict) and gstat.get("status") == "已進場"))

            for sym in syms:
                sym_str = str(sym) 
                if sym_str not in symbols_to_analyze: continue
                
                df = stock_df.get(sym_str, pd.DataFrame())
                if df.empty: continue
                
                row_now = df[df["time"] == trading_time]
                if row_now.empty: continue
                row_now = row_now.iloc[0]

                hit_limit, pull_up = False, False
                
                current_high = row_now.get("high", 0)
                limit_up_price = row_now.get("漲停價", 9999)
                current_pct = row_now.get("pct_increase", 0)
                current_vol = row_now.get("volume", 0)

                avgv = FIRST3_AVG_VOL.get(sym_str, 0)
                vol_c1 = (current_vol >= sys_config.min_volume_threshold)
                vol_c2 = (avgv > 0 and current_vol >= sys_config.volume_multiplier * avgv)
                vol_check = vol_c1 or vol_c2

                is_touching_limit = pd.notna(current_high) and pd.notna(limit_up_price) and round(current_high, 2) >= round(limit_up_price, 2)
                sys_threshold = getattr(sys_config, 'pull_up_pct_threshold', 1.4)
                is_high_pct = pd.notna(current_pct) and current_pct >= sys_threshold

                if is_group_entered: continue

                if is_touching_limit:
                    # 判斷是否為「初次觸摸」漲停 (這一分K才摸到，上一分K沒摸到)
                    prev_t = (current_dt_full - timedelta(minutes=1)).time()
                    prev = df[df["time"] == prev_t]
                    is_first_touch = prev.empty or (not prev.empty and round(prev.iloc[0]["high"], 2) < round(limit_up_price, 2))

                    if trading_time == time(9, 0) or is_first_touch:
                        hit_limit = True
                        # 這裡保留您原本的「拉高無縫升級漲停」邏輯
                        for g2, gstat_inner in group_positions.items():
                            if isinstance(gstat_inner, dict) and gstat_inner.get("trigger") == "拉高進場" and sym_str in consolidated_symbols.get(g2, []):
                                gstat_inner["trigger"], gstat_inner["wait_start"], gstat_inner["wait_counter"], gstat_inner["leader"] = "漲停進場", current_dt_full, 0, sym_str
                                msg = f"🚀 {sn(sym_str)} 衝上漲停，{g2} 族群從拉高無縫升級為漲停進場！"
                                message_log.append((log_time_tag, f"{Fore.CYAN}{msg}{Style.RESET_ALL}"))
                                tg_bot.send_message(f"🚀 <b>漲停進場</b>\n標的: <code>{sn(sym_str)}</code>...", force = True)
                                hit_limit = False

                if is_high_pct and vol_check: 
                    pull_up = True
                    
                if hit_limit or pull_up: 
                    print(f"\n[成功觸發進場] {log_time_tag} | {sn(sym_str)}")
                    trigger_list.append({"symbol": sym_str, "group": grp, "condition": "limit_up" if hit_limit else "pull_up"})

    trigger_list.sort(key=lambda x: 0 if x["condition"] == "limit_up" else 1)
    for item in trigger_list:
        grp, cond_txt = item["group"], "漲停進場" if item["condition"] == "limit_up" else "拉高進場"
        sym = item["symbol"]
        
        if grp not in group_positions or not group_positions[grp]:
            group_positions[grp] = {
                "status": "觀察中", 
                "trigger": cond_txt, 
                "start_time": current_dt_full, 
                "tracking": {sym: {"join_time": current_dt_full, "base_vol": stock_df[sym].loc[stock_df[sym]["time"] == trading_time, "volume"].iloc[0], "base_rise": stock_df[sym].loc[stock_df[sym]["time"] == trading_time, "rise"].iloc[0]}}, 
                "leader": sym if cond_txt == "漲停進場" else None
            }
            if cond_txt == "漲停進場": 
                group_positions[grp]["wait_start"], group_positions[grp]["wait_counter"] = current_dt_full, 0
                tg_bot.send_message(f"🔥 <b>條件觸發 (漲停進場)</b>\n領漲股: <code>{sn(sym)}</code>\n已鎖定族群【{grp}】準備進場！", force=True)
            else:
                msg = f"🔥 {sn(sym)} 觸發拉高條件，已鎖定族群【{grp}】觀察中"
                message_log.append((log_time_tag, f"{YELLOW}{msg}{RESET}"))
                tg_bot.send_message(f"🔥 <b>條件觸發 (拉高進場)</b>\n領漲股: <code>{sn(sym)}</code>\n已鎖定族群【{grp}】觀察中！", force=True)
                
        elif isinstance(group_positions[grp], dict) and group_positions[grp]["status"] == "觀察中" and group_positions[grp]["trigger"] == "拉高進場" and cond_txt == "漲停進場":
            gstat = group_positions[grp]
            gstat["trigger"] = "漲停進場"
            gstat["leader"] = sym
            gstat["start_time"] = current_dt_full
            gstat["wait_start"] = current_dt_full
            gstat["wait_counter"] = 0
            
            if "tracking" not in gstat: gstat["tracking"] = {}
            gstat["tracking"][sym] = {"join_time": current_dt_full, "base_vol": stock_df[sym].loc[stock_df[sym]["time"] == trading_time, "volume"].iloc[0], "base_rise": stock_df[sym].loc[stock_df[sym]["time"] == trading_time, "rise"].iloc[0]}
            
            msg = f"🚀 {sn(sym)} 漲停觸發，強制中斷拉高，無縫升級為漲停進場模式！"
            message_log.append((log_time_tag, f"{Fore.CYAN}{msg}{Style.RESET_ALL}"))
            tg_bot.send_message(f"🚀 <b>強制升級 (漲停進場)</b>\n領漲股: <code>{sn(sym)}</code>\n族群【{grp}】已從拉高切換為漲停模式！", force=True)
        
        elif isinstance(group_positions[grp], dict) and group_positions[grp]["status"] == "觀察中" and group_positions[grp]["trigger"] == "漲停進場" and cond_txt == "漲停進場" and sym != group_positions[grp].get("leader"):
            gstat = group_positions[grp]
            old_leader = gstat.get("leader", "未知")
            gstat["leader"] = sym
            gstat["start_time"] = current_dt_full
            gstat["wait_start"] = current_dt_full
            gstat["wait_counter"] = 0
            
            if "tracking" not in gstat: gstat["tracking"] = {}
            gstat["tracking"][sym] = {"join_time": current_dt_full, "base_vol": stock_df[sym].loc[stock_df[sym]["time"] == trading_time, "volume"].iloc[0], "base_rise": stock_df[sym].loc[stock_df[sym]["time"] == trading_time, "rise"].iloc[0]}
            
            msg = f"✨ 同族群新星！{sn(sym)} 衝上漲停，取代 {sn(old_leader)} 成為新領漲並重置洗盤時間！"
            # 🎨 戰局改變 ➔ Cyan 藍綠色
            message_log.append((log_time_tag, f"{Fore.CYAN}{msg}{Style.RESET_ALL}"))
            tg_bot.send_message(f"✨ <b>漲停領漲替換</b>\n新領漲: <code>{sn(sym)}</code>\n族群【{grp}】已重置洗盤等待時間！", force=True)

    for grp, gstat in group_positions.items():
        if not (isinstance(gstat, dict) and gstat["status"] == "觀察中"): continue
        track = gstat.setdefault("tracking", {})
        
        for sym in consolidated_symbols.get(grp, []):
            sym_str = str(sym)
            df = stock_df.get(sym_str, pd.DataFrame()) 
            if df.empty: continue
            
            row_now = df[df["time"] == trading_time]
            if not row_now.empty and pd.notna(row_now.iloc[0]["pct_increase"]) and \
               row_now.iloc[0]["pct_increase"] >= sys_config.follow_up_pct_threshold and \
               sym_str not in track:
                track[sym_str] = {
                    "join_time": current_dt_full,
                    "base_vol": row_now.iloc[0]["volume"],
                    "base_rise": row_now.iloc[0]["rise"]
                }

    for grp, gstat in group_positions.items():
        if not (isinstance(gstat, dict) and gstat["status"] == "觀察中"): continue
        track = gstat.get("tracking", {})
        if not track: continue

        max_sym, max_rise = None, None
        for sym in track:
            row_now = stock_df[sym][stock_df[sym]["time"] == trading_time]
            if not row_now.empty and pd.notna(rise_now := row_now.iloc[0]["rise"]) and (max_rise is None or rise_now > max_rise): max_rise, max_sym = rise_now, sym

        if gstat.get("leader") is None: 
            gstat["leader"] = max_sym
            gstat["leader_peak"] = max_rise
        elif max_sym:
            if "leader_peak" not in gstat:
                gstat["leader_peak"] = max_rise
                
            if max_sym == gstat["leader"]:
                if max_rise > gstat["leader_peak"]:
                    gstat["leader_peak"] = max_rise
            else:
                if max_rise > gstat["leader_peak"]:
                    print(f"{Fore.CYAN}✨ 領漲替換：{sn(gstat['leader'])} ➔ {sn(max_sym)}{Style.RESET_ALL}")
                    tg_bot.send_message(f"✨ <b>領漲股替換</b>\n由 <code>{sn(gstat['leader'])}</code> 切換為 <code>{sn(max_sym)}</code>\n系統已重新計算 DTW 相似度。", force=True)
                    gstat["leader"], gstat["leader_peak"], gstat["leader_reversal_rise"], gstat["status"], gstat["wait_counter"], gstat["start_time"] = max_sym, max_rise, max_rise, "觀察中", 0, current_dt_full
                    gstat.pop("wait_start", None)
                
        lead_sym = gstat["leader"]
        if not lead_sym: continue
        df_lead = stock_df[lead_sym]
        idx_now = df_lead[df_lead["time"] == trading_time].index
        if idx_now.empty: continue
        
        if "wait_start" not in gstat and idx_now[0] - 1 >= 0 and df_lead.loc[idx_now[0], "high"] <= df_lead.loc[idx_now[0] - 1, "high"]:
            gstat["wait_start"], gstat["wait_counter"], gstat["leader_reversal_rise"] = current_dt_full, 0, float(df_lead.loc[idx_now[0], "highest"])
            msg = f"{gstat.get('trigger')} {grp} 領漲 {sn(lead_sym)} 反轉，確立天花板 {df_lead.loc[idx_now[0], 'highest']}，開始等待"
            message_log.append((log_time_tag, msg))
            tg_bot.send_message(f"⏳ <b>領漲股反轉 (開始等待)</b>\n族群: {grp}\n領漲: <code>{sn(lead_sym)}</code>\n天花板: {df_lead.loc[idx_now[0], 'highest']}\n準備計算洗盤時間...", force=True)

    # 實裝「光年計時法」 (Time-based Waiting)
    # 取代原本看迴圈的方法，我們直接用當前時間減去 wait_start 的時間，計算這期間經過了幾根 K 棒
    groups_ready = []
    for grp, gstat in group_positions.items():
        if isinstance(gstat, dict) and gstat["status"] == "觀察中" and "wait_start" in gstat:
            
            leader_sym = gstat.get("leader")
            if leader_sym and leader_sym in stock_df:
                # 設定相似度比對起點：固定從 09:00 開始（與回測/迴歸一致）
                window_start_t = time(9, 0)
                track_dict = gstat.get("tracking", {})
                to_remove = []
                
                for sym in list(track_dict.keys()):
                    if sym == leader_sym: continue
                    if sym not in stock_df: continue
                    
                    # 呼叫 DTW 演算法計算相似度
                    sim = calculate_dtw_pearson(stock_df[leader_sym], stock_df[sym], window_start_t, trading_time)
                    
                    if sim < sys_config.similarity_threshold:
                        to_remove.append(sym)
                        
                for sym in to_remove:
                    del track_dict[sym]  # 從候選清單中永久踢除
            # 計算經過的分鐘數 (光年計時法核心)
            minutes_passed = (current_dt_full - gstat["wait_start"]).total_seconds() / 60
            
            # 同步更新舊版 wait_counter 用來在 Log 或其他地方顯示
            gstat["wait_counter"] = int(minutes_passed)
            
            if minutes_passed >= wait_minutes:
                groups_ready.append(grp)

    for grp in groups_ready:
        gstat = group_positions[grp]
        filtered_track = gstat.get("tracking", {}).copy()
        leader_sym = gstat.get("leader")
        
        if not filtered_track: group_positions[grp] = False; continue

        eligible = []
        leader_rise_now = None  # 取得領漲股目前漲幅（指標一：相對位置用）
        if leader_sym and leader_sym in stock_df:
            _ldf = stock_df[leader_sym]
            _lrow = _ldf[_ldf["time"] == trading_time]
            if not _lrow.empty:
                leader_rise_now = _lrow.iloc[0]["rise"]
        for sym, info in filtered_track.items():
            if sym == leader_sym: continue
            df = stock_df[sym]
            sub = df[(df["time"] >= gstat["start_time"].time()) & (df["time"] <= trading_time)]
            if sub.empty: continue
            
            # 🚀 雙軌量能濾網：嚴格審查「觸發點到等待結束」期間的動能
            sub_avg_vol = sub['volume'].mean() if not sub.empty else 0
            sub_max_vol = sub['volume'].max() if not sub.empty else 0
            
            if sub_avg_vol < sys_config.wait_min_avg_vol and sub_max_vol < sys_config.wait_max_single_vol:
                msg = f"⚠️ [流動性濾網] {sn(sym)} 量能未達標 (均:{sub_avg_vol:.0f}張 需≥{sys_config.wait_min_avg_vol}, 最大:{sub_max_vol}張 需≥{sys_config.wait_max_single_vol})，跳過。"
                message_log.append((log_time_tag, f"{YELLOW}{msg}{RESET}"))
                continue # 流動性不足，直接跳過不加入候選名單！
            
            avgv = FIRST3_AVG_VOL.get(sym, 0)
            vol_cond_1 = (sub['volume'] >= sys_config.min_volume_threshold)
            if avgv > 0:
                vol_cond_2 = (sub['volume'] >= sys_config.volume_multiplier * avgv)
                vol_cond = vol_cond_1 | vol_cond_2
            else:
                vol_cond = vol_cond_1
                
            if not vol_cond.any(): continue
            
            if len(sub) >= 2 and sub.iloc[-1]['rise'] > sub.iloc[:-1]['rise'].max() + sys_config.pullback_tolerance: continue
            
            row_now = df[df["time"] == trading_time]
            max_entry_price = sys_config.capital_per_stock * 15
            if row_now.empty or not (sys_config.rise_lower_bound <= row_now.iloc[0]["rise"] <= sys_config.rise_upper_bound) or row_now.iloc[0]["close"] > max_entry_price: continue
            row_item = row_now.iloc[0]
            price_now = row_item["close"]
            # 🔑 指標一：相對位置 — 跟漲股漲幅必須落後領漲股至少 min_lag_pct%
            if leader_rise_now is not None and not pd.isna(leader_rise_now) and (leader_rise_now - row_item["rise"]) < sys_config.min_lag_pct: continue
            # 🔑 指標三：高度門檻 — 標的當日最高漲幅需 ≥ min_height_pct%
            prev_close = row_item.get('昨日收盤價', 0)
            if prev_close and prev_close > 0 and (row_item['highest'] - prev_close) / prev_close * 100 < sys_config.min_height_pct: continue
            # 🔑 波動範圍濾網：開盤 10 分鐘後，當日振幅需 ≥ volatility_min_range%（對齊回測）
            _vmin = getattr(sys_config, 'volatility_min_range', 0)
            if trading_time >= time(9, 10) and _vmin > 0:
                _full = stock_df[sym][stock_df[sym]["time"] <= trading_time]
                if not _full.empty:
                    _pc = float(prev_close or 0)
                    if _pc > 0 and (_full['high'].max() - _full['low'].min()) / _pc * 100 < _vmin: continue
            # 🔑 稀薄量能門檻：全日均量需 ≥ min_eligible_avg_vol 張/分（過濾委買賣每格只有1~2張的幽靈股）
            sym_today_data = stock_df[sym][stock_df[sym]["time"] <= trading_time]
            if sys_config.min_eligible_avg_vol > 0 and sym_today_data['volume'].mean() < sys_config.min_eligible_avg_vol: continue
            # 🔑 低價股過濾：股價需 ≥ min_close_price（對齊回測）
            if sys_config.min_close_price > 0 and price_now < sys_config.min_close_price: continue
            eligible.append({"symbol": sym, "rise": row_item["rise"], "row": row_item})

        # 🎯 Leader 進場備胎：無合格跟漲股時，若啟用 allow_leader_entry，嘗試直接做空領漲股（對齊回測）
        if not eligible and getattr(sys_config, 'allow_leader_entry', False) and leader_sym and leader_sym in stock_df:
            _ldr_df = stock_df[leader_sym]
            _ldr_row = _ldr_df[_ldr_df["time"] == trading_time]
            if not _ldr_row.empty:
                _ldr_rise = _ldr_row.iloc[0]["rise"]
                _ldr_price = _ldr_row.iloc[0]["close"]
                _ldr_max_price = sys_config.capital_per_stock * 15
                if (sys_config.rise_lower_bound <= _ldr_rise <= sys_config.rise_upper_bound
                        and _ldr_price <= _ldr_max_price):
                    eligible.append({"symbol": leader_sym, "rise": _ldr_rise, "row": _ldr_row.iloc[0]})
                    msg = f"🎯 Leader進場：{sn(leader_sym)} 無合格follower，嘗試直接做空領漲股 rise={_ldr_rise:.2f}%"
                    message_log.append((log_time_tag, f"{YELLOW}{msg}{RESET}"))

        if not eligible: group_positions[grp] = False; continue

        eligible.sort(key=lambda x: x["rise"])  # 由低到高：漲幅最落後的排在前面，優先進場

        # 備胎進場機制 (Fallback Selection)
        # 用 while 迴圈從漲幅最落後的開始挑，直到有一檔能通過 API 限制成功送出委託單
        trade_success = False
        
        while eligible:
            chosen = eligible[0]  # 取漲幅最落後的標的（已由低到高排序）
            stock_code_str = chosen["symbol"]

            try:
                # 檢查當沖限制 (包含先賣後買的限制)
                contract = sys_state.api.Contracts.Stocks.TSE.get(stock_code_str) or sys_state.api.Contracts.Stocks.OTC.get(stock_code_str)
                if not contract or not (getattr(contract, 'day_trade', None) in ["Yes", sj.constant.DayTrade.Yes] or getattr(getattr(contract, 'day_trade', None), 'value', None) == "Yes"):
                    logger.warning(f"{stock_code_str} 被券商禁止當沖，已從候選清單剔除")
                    print(f"⚠️ {sn(stock_code_str)} 被券商禁止當沖，已從候選清單剔除，嘗試次落後標的...")
                    eligible.pop(0)  # 剔除這檔不能當沖的股票
                    continue # 重新回到 while 迴圈開頭，找次落後標的
            except Exception as e:
                logger.warning(f"檢查 {stock_code_str} 當沖權限失敗: {e}，剔除並重試")
                print(f"⚠️ 檢查 {sn(stock_code_str)} 當沖權限失敗: {e}，剔除並重試...")
                eligible.pop(0)
                continue

            # 🛡️ 風控模組：盤中進場前檢查
            if sys_config.risk_control_enabled and hasattr(sys_state, 'risk_state'):
                _rs = sys_state.risk_state
                if _rs.get('halted', False):
                    msg_core = f"🚫 風控熔斷中（累計停損{_rs.get('daily_stops',0)}筆），跳過 {sn(stock_code_str)} 進場"
                    message_log.append((log_time_tag, f"{Fore.RED}{msg_core}{Style.RESET_ALL}"))
                    tg_bot.send_message(f"🚫 <b>風控熔斷</b>\n{msg_core}", force=True)
                    break
                if _rs.get('daily_entries', 0) >= sys_config.max_daily_entries:
                    msg_core = f"🚫 已達每日進場上限 {sys_config.max_daily_entries} 檔，跳過 {sn(stock_code_str)}"
                    message_log.append((log_time_tag, f"{Fore.YELLOW}{msg_core}{Style.RESET_ALL}"))
                    break

            # 通過檢查，開始下單流程
            row, entry_px = chosen["row"], chosen["row"]["close"]
            shares = round((sys_config.capital_per_stock * 10000) / (entry_px * 1000))
            sell_amt = shares * entry_px * 1000
            fee, tax = int(sell_amt * (sys_config.transaction_fee * 0.01) * (sys_config.transaction_discount * 0.01)), int(sell_amt * (sys_config.trading_tax * 0.01))
            gap, tick = get_stop_loss_config(entry_px)
            highest_on_entry = row["highest"] or entry_px
            stop_thr = entry_px + gap / 1000 if (highest_on_entry - entry_px) * 1000 < gap else highest_on_entry + tick
            _sl_cushion = getattr(sys_config, 'sl_cushion_pct', 0)
            if _sl_cushion > 0:
                stop_thr = round(stop_thr + entry_px * (_sl_cushion / 100.0), 2)

            limit_up = row.get(f'漲停價')
            if limit_up and pd.notna(limit_up):
                tick_for_limit = 0.01 if limit_up < 10 else 0.05 if limit_up < 50 else 0.1 if limit_up < 100 else 0.5 if limit_up < 500 else 1 if limit_up < 1000 else 5
                if stop_thr > limit_up - 2 * tick_for_limit: stop_thr = limit_up - 2 * tick_for_limit

            p_exit_dt = current_dt_full + timedelta(minutes=hold_minutes) if hold_minutes is not None and (current_dt_full + timedelta(minutes=hold_minutes)).time() < time(13, 26) else None
            p_exit_str = p_exit_dt.strftime("%H:%M:%S") if p_exit_dt else "13:26:00"

            # 試算這筆交易如果碰到停損，單張最大虧損金額是多少
            max_risk_amount = (stop_thr - entry_px) * 1000
            _risk_limit = get_max_risk_for_price(entry_px)
            if max_risk_amount > _risk_limit:
                msg = f"⚠️ 拒絕進場！{sn(stock_code_str)} 距前高停損太遠 (潛在風險: {int(max_risk_amount)} > {int(_risk_limit)} 元)。啟動備胎機制，尋找下一檔！"
                message_log.append((log_time_tag, f"{Fore.YELLOW}{msg}{Style.RESET_ALL}"))
                
                # 將這檔高風險股票從清單剔除，嘗試次落後標的
                eligible.pop(0)

                # continue 會讓程式回到 while eligible 的開頭，重新挑選下一檔！
                continue

            with sys_state.lock: 
                sys_state.open_positions[stock_code_str] = {
                    'entry_price': entry_px, 
                    'target_shares': shares,
                    'filled_shares': 0,
                    'covered_shares': 0,
                    'real_entry_cost': 0.0,
                    'real_exit_cost': 0.0,
                    'sell_cost': sell_amt, 
                    'entry_fee': fee, 
                    'stop_loss': stop_thr, 
                    'planned_exit': p_exit_str,
                    'entry_date': date.today().strftime("%Y-%m-%d"),
                    'entry_time': datetime.now().strftime("%H:%M:%S"),
                    'current_price': entry_px 
                }
            
            contract = sys_state.api.Contracts.Stocks.get(stock_code_str)
            limit_down_price = contract.limit_down
            
            # 🚀 計算穿價：拿當下1分K的收盤價，往下穿N檔 (確保不低於跌停價)
            target_price = get_tick_price(entry_px, -sys_config.slippage_ticks)
            order_price = max(target_price, limit_down_price)
            
            deal_tracker[stock_code_str] = {
                'target_qty': shares, 'total_qty': 0, 'total_cost': 0.0,
                'action_type': '進場', 'time': current_time_str
            }
            
            sys_db.save_state('current_position', sys_state.open_positions)
            if hasattr(ui_dispatcher, 'position_updated'):
                ui_dispatcher.position_updated.emit(sys_state.open_positions)
            
            # 🚀 委託單重構：使用 IOC (立即成交否則取消)，價格使用 order_price
            order = sys_state.api.Order(price=order_price, quantity=shares, action=sj.constant.Action.Sell, price_type=sj.constant.StockPriceType.LMT, order_type=sj.constant.OrderType.IOC, order_lot=sj.constant.StockOrderLot.Common, daytrade_short=True, account=sys_state.api.stock_account)
            trade_result = safe_place_order(sys_state.api, contract, order)
            
            if trade_result is None:
                logger.warning(f"下單失敗 ({stock_code_str} 遭 API 退單或無權限)，剔除並嘗試次落後標的")
                print(f"⚠️ 下單失敗 (遭 API 退單或無權限)，剔除並嘗試次落後標的...")
                eligible.pop(0)
                with sys_state.lock:
                    if stock_code_str in sys_state.open_positions:
                        del sys_state.open_positions[stock_code_str]
                continue
            
            with sys_state.lock:
                if stock_code_str not in sys_state.to.contracts: sys_state.to.contracts[stock_code_str] = contract
                group_positions[grp] = "已進場"

            sys_db.log_trade("委託(進場)", stock_code_str, shares, order_price, 0.0, "穿價IOC進場")
            msg_core = f"穿價IOC進場！{sn(stock_code_str)} {shares}張 委託價 {order_price:.2f} 停損價 {stop_thr:.2f}"
            tg_bot.send_message(f"🟡 <b>穿價IOC進場空單</b>\n標的: <code>{sn(stock_code_str)}</code>\n目標數量: {shares} 張\n穿價委託: {order_price:.2f}\n停損防護: {stop_thr:.2f}\n(IOC瞬間搓合中...)", force=True)
            message_log.append((log_time_tag, f"{Fore.GREEN}{msg_core}{Style.RESET_ALL}"))
            
            # 🛡️ 風控模組：盤中進場成功，更新計數
            if hasattr(sys_state, 'risk_state'):
                sys_state.risk_state['daily_entries'] = sys_state.risk_state.get('daily_entries', 0) + 1

            trade_success = True
            break # 委託成功，跳出備胎迴圈

        if not trade_success:
            # 如果所有的 eligible 跟漲都被當沖限制擋下 (名單被 pop 到空了)
            logger.warning(f"{grp} 族群沒有符合當沖資格的標的，該次進場機會取消")
            print(f"⚠️ {grp} 族群沒有符合當沖資格的標的，該次進場機會取消。")
            group_positions[grp] = False

    # 處理中斷等待邏輯 (原本的位置稍作調整，放在最後處理避免順序干擾)
    for grp, gstat in group_positions.items():
        if not (isinstance(gstat, dict) and gstat["status"] == "觀察中" and "wait_start" in gstat): continue

        lead = gstat.get("leader")
        if lead and gstat.get("leader_reversal_rise") is not None:
            row_now = stock_df.get(lead, pd.DataFrame())[stock_df.get(lead, pd.DataFrame())["time"] == trading_time]
            if not row_now.empty and pd.notna(row_now.iloc[0]["high"]) and float(row_now.iloc[0]["high"]) > gstat["leader_reversal_rise"]:
                msg = f"🔥 領漲 {sn(lead)} 突破前高 {gstat['leader_reversal_rise']}，漲勢延續，中斷等待！"
                message_log.append((log_time_tag, f"{Fore.CYAN}{msg}{Style.RESET_ALL}"))
                tg_bot.send_message(f"🚀 <b>突破前高 (中斷等待)</b>\n領漲股: <code>{sn(lead)}</code>\n突破前高: {gstat['leader_reversal_rise']}\n漲勢延續，已重置倒數計時！", force=True)
                gstat["leader_reversal_rise"], gstat["status"], gstat["wait_counter"] = float(row_now.iloc[0]["highest"]), "觀察中", 0
                gstat.pop("wait_start", None)
                continue

    message_log.sort(key=lambda x: x[0])
    for t, m in message_log: print(f"[{t}] {m}")
    message_log.clear()

# =================================================================================
# 盤中主控制迴圈
def generate_daily_report():
    """盤後自動報告：從 trade_logs 查詢今日所有交易紀錄並產生摘要"""
    today_str = datetime.now().strftime('%Y-%m-%d')
    try:
        df = pd.read_sql(
            "SELECT timestamp, action, symbol, shares, price, profit, note FROM trade_logs WHERE timestamp LIKE ? ORDER BY id",
            sys_db.conn, params=(f"{today_str}%",)
        )
    except Exception as e:
        print(f"[盤後報告] 讀取交易紀錄失敗：{e}")
        return

    if df.empty:
        sep = '=' * 50
        print(f"\n{sep}\n  盤後報告 - {today_str}\n{sep}\n  今日無任何交易紀錄。\n{sep}")
        try:
            tg_msg_empty = f"📋 <b>盤後報告 - {today_str}</b>\n━━━━━━━━━━━━━━\n今日無任何交易紀錄。"
            tg_bot.send_message(tg_msg_empty, force=True)
        except Exception:
            pass
        return

    # 計算各項統計
    total_trades = len(df)
    total_profit = df['profit'].sum()

    # 只統計有損益的交易（平倉類）
    closed_df = df[df['profit'] != 0]
    win_trades = len(closed_df[closed_df['profit'] > 0])
    loss_trades = len(closed_df[closed_df['profit'] < 0])
    even_trades = len(closed_df[closed_df['profit'] == 0])

    # 逐筆交易明細
    trade_lines_console = []
    trade_lines_tg = []
    for _, row in df.iterrows():
        sym_display = sn(str(row['symbol']))
        t_str = str(row['timestamp'])[11:16]
        act = row['action']
        p = row['profit']
        price_val = row['price']
        shares_val = row['shares']
        if p > 0:
            icon, tg_icon = '+', '🔴'
        elif p < 0:
            icon, tg_icon = '', '🟢'
        else:
            icon, tg_icon = '', '─'
        trade_lines_console.append(
            f"  [{t_str}] {act} {sym_display} {shares_val}張 @{price_val:.2f} 損益:{icon}{int(p)}元"
        )
        if p != 0:
            trade_lines_tg.append(
                f"[{t_str}] {act} <code>{sym_display}</code> {tg_icon} {icon}{int(p)}元"
            )

    win_rate_str = f"{win_trades/(win_trades+loss_trades)*100:.1f}%" if (win_trades + loss_trades) > 0 else "N/A"

    # Console 輸出
    sep = '=' * 50
    sep2 = '─' * 50
    detail_str = '\n'.join(trade_lines_console)
    lines = [
        "", sep,
        f"  盤後報告 - {today_str}",
        sep,
        f"  交易筆數：{total_trades} 筆",
        f"  平倉損益：{int(total_profit):+d} 元",
        f"  獲利：{win_trades} 筆 / 虧損：{loss_trades} 筆 / 持平：{even_trades} 筆",
        f"  勝率：{win_rate_str}",
        sep2,
        "  【交易明細】",
        detail_str,
        sep
    ]
    print('\n'.join(lines))

    # Telegram 推播
    tg_detail = '\n'.join(trade_lines_tg) if trade_lines_tg else '(無平倉紀錄)'
    if len(tg_detail) > 2500:
        tg_detail = tg_detail[:2500] + '\n... (資料過多已省略)'
    tg_lines = [
        f"📋 <b>盤後報告 - {today_str}</b>",
        "━━━━━━━━━━━━━━",
        f"📊 交易筆數：<b>{total_trades}</b> 筆",
        f"💰 平倉損益：<b>{int(total_profit):+d}</b> 元",
        f"✅ 獲利：{win_trades} 筆 / ❌ 虧損：{loss_trades} 筆",
        f"🎯 勝率：<b>{win_rate_str}</b>",
        "━━━━━━━━━━━━━━",
        f"<b>【交易明細】</b>",
        tg_detail
    ]
    tg_msg = '\n'.join(tg_lines)
    try:
        tg_bot.send_message(tg_msg, force=True)
    except Exception:
        pass


def start_trading(mode='full', wait_minutes=None, hold_minutes=None):
    sys_state.is_monitoring = True
    sys_state.stop_trading_flag = False

    now = datetime.now()
    if now.weekday() in [5, 6]:
        days_ahead = 7 - now.weekday()
        next_monday = (now + timedelta(days=days_ahead)).replace(hour=8, minute=30, second=0, microsecond=0)
        sleep_seconds = (next_monday - now).total_seconds()
        msg = f"🏖️ 目前為週末休市時間！系統將自動進入深度休眠。\n⏰ 預計喚醒時間：{next_monday.strftime('%m/%d %H:%M')}"
        print(f"{YELLOW}{msg}{RESET}")
        try: tg_bot.send_message(f"🏖️ <b>週末休市中</b>\n系統已進入深度休眠，將於下週一 08:30 自動喚醒並執行盤前準備！", force=True)
        except Exception: pass
        time_module.sleep(sleep_seconds)
        # 週末長時間休眠後，重新登入 Shioaji（session 可能已過期）
        try:
            _is_live = getattr(sys_config, 'live_trading_mode', False)
            _ak = shioaji_logic.LIVE_API_KEY if _is_live else shioaji_logic.TEST_API_KEY
            _as = shioaji_logic.LIVE_API_SECRET if _is_live else shioaji_logic.TEST_API_SECRET
            sys_state.api = sj.Shioaji(simulation=not _is_live)
            sys_state.api.login(api_key=_ak, secret_key=_as, contracts_timeout=10000)
            if shioaji_logic.CA_CERT_PATH and shioaji_logic.CA_PASSWORD:
                sys_state.api.activate_ca(ca_path=shioaji_logic.CA_CERT_PATH, ca_passwd=shioaji_logic.CA_PASSWORD)
            sys_state.api.set_order_callback(order_callback)  # 🔧 重連後補註冊 callback
            print("✅ [喚醒] 永豐 API 重新登入成功，準備開始盤前準備")
        except Exception as _wake_e:
            logger.error(f"[喚醒] 永豐 API 重新登入失敗：{_wake_e}")
            print(f"⚠️ [喚醒] 永豐 API 重新登入失敗：{_wake_e}")
        start_trading(mode, wait_minutes, hold_minutes)
        return

    load_twse_name_map()
    client = init_esun_client()
    matrix_dict_analysis = load_matrix_dict_analysis()
    
    # 1. 抓取當日實戰處置股 -> 存入 disposition_stock_live
    fetch_disposition_stocks(client, matrix_dict_analysis)   
    
    # 2. 取得今日實戰專用處置股清單
    live_disposition_list = load_disposition_stocks(is_live=True)
    
    # 3. 取得要監控的股票大名單 (已精準過濾實戰處置股)
    symbols_to_analyze = load_symbols_to_analyze(is_live=True)
    
    # 4. 取得原始族群名單，並在「記憶體中」進行清洗 (不寫入資料庫)
    group_symbols = load_group_symbols()
    group_symbols = purge_disposition_from_nb(group_symbols, live_disposition_list)
    
    if not group_symbols or not group_symbols.get('consolidated_symbols', {}): return print("沒有加載到族群資料。")
    group_positions = {group: False for group in group_symbols['consolidated_symbols'].keys()}

    now = datetime.now()
    pre_market_start = now.replace(hour=8, minute=30, second=0, microsecond=0)
    market_start     = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_exit      = now.replace(hour=13, minute=26, second=0, microsecond=0)
    market_end       = now.replace(hour=13, minute=30, second=0, microsecond=0)

    if now < pre_market_start:
        sys_state.trading = False
        wait_sec = (pre_market_start - now).total_seconds()
        print(f"目前為 {now.strftime('%H:%M:%S')}，將休眠至 08:30...")
        time_module.sleep(wait_sec)
        # 盤前休眠後，重新登入 Shioaji（確保 session 仍有效）
        try:
            _is_live = getattr(sys_config, 'live_trading_mode', False)
            _ak = shioaji_logic.LIVE_API_KEY if _is_live else shioaji_logic.TEST_API_KEY
            _as = shioaji_logic.LIVE_API_SECRET if _is_live else shioaji_logic.TEST_API_SECRET
            sys_state.api = sj.Shioaji(simulation=not _is_live)
            sys_state.api.login(api_key=_ak, secret_key=_as, contracts_timeout=10000)
            if shioaji_logic.CA_CERT_PATH and shioaji_logic.CA_PASSWORD:
                sys_state.api.activate_ca(ca_path=shioaji_logic.CA_CERT_PATH, ca_passwd=shioaji_logic.CA_PASSWORD)
            sys_state.api.set_order_callback(order_callback)  # 🔧 重連後補註冊 callback
            print("✅ [盤前] 永豐 API 重新登入成功")
        except Exception as _pre_e:
            logger.error(f"[盤前] 永豐 API 重新登入失敗：{_pre_e}")
            print(f"⚠️ [盤前] 永豐 API 重新登入失敗：{_pre_e}")
        start_trading(mode, wait_minutes, hold_minutes)
        return

    elif now >= market_end:
        sys_state.trading = False
        tomorrow = now + timedelta(days=1)
        if tomorrow.weekday() == 5: tomorrow = tomorrow + timedelta(days=2)
        tomorrow_pre_market = tomorrow.replace(hour=8, minute=30, second=0, microsecond=0)
        print(f"今日已收盤。系統將休眠至下次開盤日 {tomorrow_pre_market.strftime('%m/%d %H:%M')}...")
        time_module.sleep((tomorrow_pre_market - now).total_seconds())
        start_trading(mode, wait_minutes, hold_minutes) 
        return

    elif pre_market_start <= now < market_start:
        sys_state.trading = False
        print(f"進入盤前時間，開始準備【當日實戰】日K資料...")
        existing_auto_daily_data = sys_db.load_kline('daily_kline_live')
        auto_daily_data, data_is_same, initial_api_count = {}, True, 0
        for symbol in symbols_to_analyze[:20]:
            if initial_api_count >= 55: time_module.sleep(60); initial_api_count = 0
            daily_kline_df = fetch_daily_kline_data(client, symbol, days=2)
            initial_api_count += 1
            if daily_kline_df.empty: continue
            daily_kline_data = daily_kline_df.to_dict(orient='records')
            auto_daily_data[symbol] = daily_kline_data
            if existing_auto_daily_data.get(symbol) != daily_kline_data: data_is_same = False; existing_auto_daily_data[symbol] = daily_kline_data

        if not data_is_same:
            for symbol in symbols_to_analyze[20:]:
                if initial_api_count >= 55: time_module.sleep(60); initial_api_count = 0
                daily_kline_df = fetch_daily_kline_data(client, symbol, days=2)
                initial_api_count += 1
                if daily_kline_df.empty: continue
                daily_kline_data = daily_kline_df.to_dict(orient='records')
                if existing_auto_daily_data.get(symbol) != daily_kline_data: existing_auto_daily_data[symbol] = daily_kline_data

        sys_db.save_kline('daily_kline_live', existing_auto_daily_data)
        print(f"{YELLOW}實戰盤前參考數據更新完成。{RESET}")
        now = datetime.now()
        if (market_start - now).total_seconds() > 0: time_module.sleep((market_start - now).total_seconds())
        print("開盤！自動切換到盤中監控模式…")
        start_trading(mode='post', wait_minutes=wait_minutes, hold_minutes=hold_minutes)
        return

    elif market_start <= now < market_end:
        sys_state.trading = True
        print(f"盤中監控時間，直接載入【當日實戰】參考資料。")
        existing_auto_daily_data = sys_db.load_kline('daily_kline_live')
        trading_day = (datetime.now() - timedelta(minutes=1)).strftime('%Y-%m-%d')
        yesterday_close_prices = {}
        for symbol in symbols_to_analyze:
            daily_data = existing_auto_daily_data.get(symbol, [])
            if not daily_data: yesterday_close_prices[symbol] = 0
            else:
                sorted_daily_data = sorted(daily_data, key=lambda x: x['date'], reverse=True)
                if len(sorted_daily_data) > 1:
                    now2 = datetime.now()
                    yesterday_close_prices[symbol] = sorted_daily_data[0].get('close', 0) if (0 <= now2.weekday() <= 4 and 8 <= now2.hour < 15) else sorted_daily_data[1].get('close', 0)
                else: yesterday_close_prices[symbol] = sorted_daily_data[0].get('close', 0)

        print("🔁 [實戰] 開始補齊今日 09:00 到目前為止的一分K資料...")
        full_intraday_end = (now - timedelta(minutes=1)).strftime('%H:%M') if now < now.replace(hour=13, minute=30, second=0, microsecond=0) else "13:30"
        auto_intraday_data = {}
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_symbol = {}
            for symbol in symbols_to_analyze:
                if (yc := yesterday_close_prices.get(symbol, 0)) == 0: continue
                future_to_symbol[executor.submit(fetch_realtime_intraday_data, client, symbol, trading_day, yc, "09:00", full_intraday_end)] = symbol
                
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                df = future.result()
                if df.empty: continue
                updated_records = []
                records = df.to_dict(orient='records')
                for i, candle in enumerate(records): 
                    updated_records.append(calculate_pct_increase_and_highest(candle, records[:i]))
                auto_intraday_data[symbol] = pd.DataFrame(updated_records).to_dict(orient='records')

        print(f"{GREEN}✅ [實戰] 補齊完成{RESET}")
        save_auto_intraday_data(auto_intraday_data)
        initialize_triggered_limit_up(auto_intraday_data)

        has_exited, current_position, hold_time, message_log, already_entered_stocks = False, None, 0, [], []
        stop_loss_triggered, final_check_active, final_check_count, in_waiting_period, waiting_time = False, False, 0, False, 0
        leader, tracking_stocks, previous_rise_values, leader_peak_rise, leader_rise_before_decline, first_condition_one_time = None, set(), {}, None, None, None
        can_trade, exit_live_done = True, False

        # 🛡️ 風控模組：每日開盤重置（掛在 sys_state 上，盤中所有函數統一透過 sys_state.risk_state 存取）
        sys_state.risk_state = {'daily_entries': 0, 'daily_stops': 0, 'halted': False}

        while True:
            if sys_state.stop_trading_flag:
                print(f"\n{YELLOW}🛑 接收到手動終止指令，已退出盤中監控模式！{RESET}")
                sys_state.trading = False; sys_state.is_monitoring = False; sys_state.stop_trading_flag = False
                try:
                    generate_daily_report()
                except Exception as _report_err:
                    print(f"[盤後報告] 產生報告時發生錯誤：{_report_err}")
                return
            now_loop = datetime.now()
            if now_loop >= market_exit and not exit_live_done:
                print(f"🔍 13:26 觸發：檢查所有庫存平倉。"); exit_trade_live(); exit_live_done = True
            if now_loop >= market_end:
                print(f"\n⏰ 13:30 今日盤中監控結束。"); break
                
            with sys_state.lock:
                now_str = now_loop.strftime("%H:%M:%S")
                for sym, pos_info in list(sys_state.open_positions.items()):
                    p_exit = pos_info.get('planned_exit')
                    # 必須有庫存且當前時間 >= 預計出場時間
                    if p_exit and now_str >= p_exit and pos_info.get('filled_shares', 0) > pos_info.get('covered_shares', 0):
                        print(f"{Fore.RED}⏰ [持有時間到] {sn(sym)} 已達預計出場時間 {p_exit}，執行自動回補！{Style.RESET_ALL}")
                        # 清除鬧鐘並發射平倉單
                        pos_info['planned_exit'] = None 
                        threading.Thread(target=close_one_stock, args=(sym,), daemon=True).start()

            now = datetime.now()
            next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
            sleep_seconds = (next_minute - now).total_seconds()
            
            if sleep_seconds > 0:
                time_module.sleep(sleep_seconds)
            
            time_module.sleep(6)

            actual_fetch_time = next_minute - timedelta(minutes=1)
            fetch_time_str = actual_fetch_time.strftime('%H:%M') if actual_fetch_time.time() <= market_end.time() else "13:30"
            print(f"⏱ [即時] 開始取得 {fetch_time_str} 的一分K資料...")
            t_start_fetch = time_module.perf_counter()

            updated_intraday_data = {}
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_symbol = {}
                for symbol in symbols_to_analyze:
                    if (yc := yesterday_close_prices.get(symbol, 0)) == 0: continue
                    future_to_symbol[executor.submit(fetch_realtime_intraday_data, client, symbol, trading_day, yc, fetch_time_str, fetch_time_str)] = symbol
                    
                for fut in as_completed(future_to_symbol):
                    sym = future_to_symbol[fut]
                    df = fut.result()
                    
                    if df.empty:
                        prev_data = auto_intraday_data.get(sym, [])
                        if prev_data:
                            last_c = prev_data[-1].copy(); last_c['time'] = fetch_time_str + ":00"; last_c['volume'] = 0.0; raw_candle = last_c
                        else:
                            yc = yesterday_close_prices.get(sym, 0)
                            raw_candle = {'symbol': sym, 'date': trading_day, 'time': fetch_time_str, 'open': yc, 'high': yc, 'low': yc, 'close': yc, 'volume': 0.0, '昨日收盤價': yc, '漲停價': truncate_to_two_decimals(calculate_limit_up_price(yc)), 'rise': 0.0}
                    else:
                        raw_candle = df.to_dict(orient='records')[0]
                        if float(raw_candle.get('volume', 0)) == 0.0:
                            prev_data = auto_intraday_data.get(sym, [])
                            if prev_data:
                                last_c = prev_data[-1]; raw_candle['open'] = raw_candle['high'] = raw_candle['low'] = raw_candle['close'] = last_c['close']; raw_candle['rise'] = last_c.get('rise', 0.0)
                            else:
                                yc = yesterday_close_prices.get(sym, 0); raw_candle['open'] = raw_candle['high'] = raw_candle['low'] = raw_candle['close'] = yc; raw_candle['rise'] = 0.0
                    
                    candle = calculate_pct_increase_and_highest(raw_candle, auto_intraday_data.get(sym, []))
                    if '漲停價' in candle: candle['漲停價'] = truncate_to_two_decimals(candle['漲停價'])
                    
                    # 確認資料端是否有算出 1.4% (在加進 updated_intraday_data 前攔截)
                    p2_val = candle.get('pct_increase', 0)
                    if p2_val is not None and pd.notna(p2_val) and float(p2_val) >= getattr(sys_config, 'pull_up_pct_threshold', 1.4):
                        print(f"📡 [資料端觸發] {sn(sym)} | 2分漲幅: {p2_val:.2f}% | 時間: {fetch_time_str} | 成交量: {candle.get('volume', 0)}")

                    updated_intraday_data.setdefault(sym, []).append(candle)
            
            fetch_duration = time_module.perf_counter() - t_start_fetch
            print(f"✅ [即時] {fetch_time_str} 一分K取得完成，耗時 {fetch_duration:.2f} 秒")
            for sym, lst in updated_intraday_data.items():
                auto_intraday_data.setdefault(sym, []).extend(lst)
                auto_intraday_data[sym] = auto_intraday_data[sym][-1000:]
            
            with sys_state.lock:
                sys_state.in_memory_intraday = auto_intraday_data.copy()
            
            process_live_trading_logic(symbols_to_analyze, fetch_time_str, wait_minutes, hold_minutes, message_log, False, has_exited, current_position, hold_time, already_entered_stocks, stop_loss_triggered, final_check_active, final_check_count, in_waiting_period, waiting_time, leader, tracking_stocks, previous_rise_values, leader_peak_rise, leader_rise_before_decline, first_condition_one_time, can_trade, group_positions)
            
            save_auto_intraday_data(auto_intraday_data)       
            broadcast_portfolio_update()

        sys_state.trading = False

        # 盤後自動報告
        try:
            generate_daily_report()
        except Exception as _report_err:
            print(f"[盤後報告] 產生報告時發生錯誤：{_report_err}")

        tomorrow = datetime.now() + timedelta(days=1)
        if tomorrow.weekday() == 5: tomorrow = tomorrow + timedelta(days=2)
        tomorrow_pre_market = tomorrow.replace(hour=8, minute=30, second=0, microsecond=0)
        print(f"今日交易已完成。系統將自動休眠至開盤日 {tomorrow_pre_market.strftime('%m/%d %H:%M')}..."); time_module.sleep((tomorrow_pre_market - datetime.now()).total_seconds())
        start_trading(mode, wait_minutes, hold_minutes); return

# ==================== PyQt5 專業圖形介面 (GUI) ====================

def _fetch_twse_holidays():
    """從 TWSE 取得國定假日集合（yyyyMMdd 格式），用於開盤燈號判斷"""
    holidays = set()
    try:
        current_year = datetime.now().year
        headers = {'User-Agent': 'Mozilla/5.0'}
        for y in [current_year, current_year - 1]:
            roc_year = y - 1911
            url = f"https://www.twse.com.tw/holidaySchedule/holidaySchedule?response=json&queryYear={roc_year}"
            res = requests.get(url, headers=headers, timeout=5, verify=False)
            if res.status_code == 200:
                data = res.json()
                if 'data' in data:
                    for row in data['data']:
                        clean_date = str(row[0]).strip().replace("-", "")
                        holidays.add(clean_date)
    except Exception as e:
        logger.warning(f"獲取線上國定假日失敗: {e}")
    return holidays

_twse_holidays = _fetch_twse_holidays()

def _open_tutorial_tab(page_name: str):
    """各功能教學按鈕的統一跳轉函數，跳轉至主介面 TutorialWidget 並定位頁面"""
    if _main_window_ref:
        tw = _main_window_ref._ensure_tab('tutorial', '新手教學', TutorialWidget)
        if hasattr(tw, 'goto'):
            tw.goto(page_name)

def _make_tutorial_btn(page_name):
    btn = QPushButton("  ?  教學")
    btn.setFixedHeight(32)
    btn.setStyleSheet(f"""QPushButton {{ background-color: {TV['surface']}; color: {TV['text_dim']}; border: 1px solid {TV['border_light']}; border-radius: 5px; font-size: 12px; font-weight: 600; padding: 0 10px; }} QPushButton:hover {{ background-color: {TV['yellow']}; color: {TV['bg']}; border-color: {TV['yellow']}; }}""")
    btn.setCursor(Qt.PointingHandCursor)
    btn.clicked.connect(lambda: _open_tutorial_tab(page_name))
    return btn

class BaseDialog(QDialog):
    def __init__(self, title, size=(400, 300)):
        super().__init__()
        self.setWindowTitle(title); self.resize(*size)
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        self.setStyleSheet(TV_DIALOG_STYLE)

class EsunLoginDialog(QDialog):
    """專業風格登入彈窗"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("安全驗證")
        self.setFixedSize(420, 320)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {TV['panel']};
                border: 1px solid {TV['border_light']};
                border-radius: 12px;
            }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 32, 36, 32)
        root.setSpacing(0)

        # ── 頂部 Logo 區 ──
        logo_row = QHBoxLayout()
        logo_icon = QLabel("▲")
        logo_icon.setStyleSheet(f"color: {TV['blue']}; font-size: 26px; font-weight: 900;")
        logo_text = QLabel("REMORA")
        logo_text.setStyleSheet(f"color: {TV['text_bright']}; font-size: 17px; font-weight: 700; letter-spacing: 1px; margin-left: 6px;")
        logo_row.addWidget(logo_icon)
        logo_row.addWidget(logo_text)
        logo_row.addStretch()
        root.addLayout(logo_row)

        subtitle = QLabel("玉山 API  安全登入驗證")
        subtitle.setStyleSheet(f"color: {TV['text_dim']}; font-size: 12px; margin-top: 4px; margin-bottom: 22px;")
        root.addWidget(subtitle)

        # ── 表單 ──
        def field_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {TV['text_dim']}; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 3px;")
            return lbl

        def styled_input(placeholder, echo=False):
            inp = QLineEdit()
            if echo: inp.setEchoMode(QLineEdit.Password)
            inp.setPlaceholderText(placeholder)
            inp.setFixedHeight(40)
            inp.setStyleSheet(f"""
                QLineEdit {{
                    background-color: {TV['surface']};
                    color: {TV['text']};
                    border: 1px solid {TV['border_light']};
                    border-radius: 6px;
                    padding: 0 12px;
                    font-size: 13px;
                }}
                QLineEdit:focus {{ border-color: {TV['blue']}; }}
            """)
            return inp

        root.addWidget(field_label("網路登入密碼"))
        self.e_pwd = styled_input("請輸入網路登入密碼", echo=True)
        root.addWidget(self.e_pwd)

        spacer1 = QWidget(); spacer1.setFixedHeight(12); root.addWidget(spacer1)

        root.addWidget(field_label("數位憑證密碼"))
        self.e_cert = styled_input("請輸入 .p12 憑證密碼", echo=True)
        root.addWidget(self.e_cert)

        spacer2 = QWidget(); spacer2.setFixedHeight(22); root.addWidget(spacer2)

        # ── 登入按鈕 ──
        btn = QPushButton("  ▶  登入並啟動盤中監控")
        btn.setFixedHeight(44)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {TV['blue']}, stop:1 #1565c0);
                color: white;
                border: none;
                border-radius: 7px;
                font-size: 14px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #3d7aff, stop:1 {TV['blue']}); }}
            QPushButton:pressed {{ background-color: {TV['blue_dim']}; }}
        """)
        btn.clicked.connect(self.accept)
        root.addWidget(btn)

        # Enter 鍵觸發登入
        self.e_pwd.returnPressed.connect(self.e_cert.setFocus)
        self.e_cert.returnPressed.connect(self.accept)

    def get_passwords(self): return self.e_pwd.text(), self.e_cert.text()

def ensure_esun_passwords():
    global ESUN_LOGIN_PWD, ESUN_CERT_PWD
    # 略過彈出視窗輸入，直接塞入假密碼，建立「純看盤通道」
    ESUN_LOGIN_PWD = "dummy_login"
    ESUN_CERT_PWD = "dummy_cert"
    return True

class CorrelationAnalysisThread(QThread):
    finished_signal = pyqtSignal(list)
    def __init__(self, mode, wait_mins): super().__init__(); self.mode = mode; self.wait_mins = wait_mins
    def run(self):
        result_data = []
        try:
            _, history_data = load_kline_data()
            groups = load_matrix_dict_analysis()
            dispo = load_disposition_stocks() 
            for grp_name, stocks in groups.items():
                stock_dfs = {}
                for s in [x for x in stocks if x not in dispo]:
                    if s in history_data and history_data[s]:
                        df = pd.DataFrame(history_data[s])
                        if not df.empty and 'time' in df.columns: df['time'] = pd.to_datetime(df['time'], format="%H:%M:%S").dt.time; stock_dfs[s] = df
                if len(stock_dfs) < 2: continue
                if self.mode == "macro":
                    leader = None; max_rise = -999
                    for s, df in stock_dfs.items():
                        if (s_max := df['rise'].max()) > max_rise: max_rise = s_max; leader = s
                    if not leader: continue
                    w_start, w_end = time(9,0), time(13,30)
                    for s in stock_dfs.keys():
                        if s == leader: continue
                        sim = calculate_dtw_pearson(stock_dfs[leader], stock_dfs[s], w_start, w_end)
                        result_data.append({'group': grp_name, 'leader': sn(leader), 'follower': sn(s), 'window': '09:00~13:30 (全天)', 'similarity': sim})
                elif self.mode == "micro":
                    leader, start_time, wait_counter, in_waiting, leader_peak_rise = None, None, 0, False, -999
                    intercept_w_start, intercept_w_end = None, None
                    tracking_stocks = set(stock_dfs.keys())
                    time_range = [ (datetime.combine(date.today(), time(9,0)) + timedelta(minutes=i)).time() for i in range(271) ]
                    for current_t in time_range:
                        cur_max_sym, cur_max_rise = None, -999
                        for s in tracking_stocks:
                            row = stock_dfs[s][stock_dfs[s]['time'] == current_t]
                            if not row.empty and (r := row.iloc[0]['rise']) > cur_max_rise: cur_max_rise, cur_max_sym = r, s
                        if not cur_max_sym: continue
                        if leader != cur_max_sym: leader, start_time, leader_peak_rise, in_waiting, wait_counter = cur_max_sym, current_t, cur_max_rise, False, 0
                        else:
                            if cur_max_rise < leader_peak_rise and not in_waiting: in_waiting, wait_counter = True, 0
                            elif cur_max_rise > leader_peak_rise: leader_peak_rise, in_waiting = cur_max_rise, False 
                        if in_waiting:
                            wait_counter += 1
                            if wait_counter >= self.wait_mins:
                                intercept_w_end, intercept_w_start = current_t, max(time(9,0), (datetime.combine(date.today(), start_time) - timedelta(minutes=2)).time())
                                break
                    if leader and intercept_w_start and intercept_w_end:
                        window_str = f"{intercept_w_start.strftime('%H:%M')}~{intercept_w_end.strftime('%H:%M')}"
                        for s in tracking_stocks:
                            if s == leader: continue
                            sim = calculate_dtw_pearson(stock_dfs[leader], stock_dfs[s], intercept_w_start, intercept_w_end)
                            result_data.append({'group': grp_name, 'leader': sn(leader), 'follower': sn(s), 'window': window_str, 'similarity': sim})
        except Exception as e: logger.error(f"DTW 分析失敗: {e}", exc_info=True)
        self.finished_signal.emit(result_data)

class SimilarityOptimizationDialog(BaseDialog):
    def __init__(self):
        super().__init__("大數據分析中心  │  DTW 最適門檻最佳化", (1000, 880))
        layout = QVBoxLayout(self); layout.setSpacing(12); layout.setContentsMargins(16, 16, 16, 16)

        # 頂部標題列
        title_row = QHBoxLayout()
        title_lbl = QLabel("大數據分析中心 / DTW 族群連動門檻優化")
        title_lbl.setStyleSheet(f"color: {TV['text_bright']}; font-size: 16px; font-weight: 700; padding: 4px 0 8px 0;")
        btn_tut = _make_tutorial_btn('opt_sim')
        title_row.addWidget(title_lbl); title_row.addStretch(); title_row.addWidget(btn_tut)
        hdr_widget = QWidget(); hdr_widget.setLayout(title_row)
        hdr_widget.setStyleSheet(f"border-bottom: 1px solid {TV['border']};")
        layout.addWidget(hdr_widget)

        ctrl_layout = QHBoxLayout(); ctrl_layout.setSpacing(12)

        g1 = QGroupBox("STEP 1  ─  大數據庫採集")
        g1.setStyleSheet(f"QGroupBox{{color:{TV['purple']}; font-weight:700; border:1px solid {TV['border_light']}; border-radius:8px; margin-top:14px; padding-top:10px;}}")
        l1 = QFormLayout(g1); l1.setSpacing(10)
        self.days_in = QLineEdit("5")
        l1.addRow("採集天數 (建議 5~30 天):", self.days_in)
        self.btn_f = QPushButton("  ▷  執行數據採集")
        self.btn_f.setFixedHeight(38)
        self.btn_f.setStyleSheet(f"""
            QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {TV['purple']}, stop:1 #5e35b1);
                           color:white; border:none; border-radius:6px; font-weight:700; font-size:13px; }}
            QPushButton:hover {{ background: {TV['purple']}; }}
            QPushButton:disabled {{ background: {TV['surface']}; color: {TV['text_dim']}; }}
        """)
        self.btn_f.clicked.connect(self.start_f)
        l1.addRow(self.btn_f); ctrl_layout.addWidget(g1)

        g2 = QGroupBox("STEP 2  ─  智能門檻掃描")
        g2.setStyleSheet(f"QGroupBox{{color:{TV['yellow']}; font-weight:700; border:1px solid {TV['border_light']}; border-radius:8px; margin-top:14px; padding-top:10px;}}")
        l2 = QFormLayout(g2); l2.setSpacing(10)
        self.wait_in, self.hold_in = QLineEdit("5"), QLineEdit("F")
        l2.addRow("等待時間 (分):", self.wait_in)
        l2.addRow("持有時間 (分 / F=尾盤):", self.hold_in)
        self.btn_o = QPushButton("  ▷  啟動智能掃描")
        self.btn_o.setFixedHeight(38)
        self.btn_o.setStyleSheet(f"""
            QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {TV['orange']}, stop:1 #e65100);
                           color:white; border:none; border-radius:6px; font-weight:700; font-size:13px; }}
            QPushButton:hover {{ background: {TV['orange']}; }}
            QPushButton:disabled {{ background: {TV['surface']}; color: {TV['text_dim']}; }}
        """)
        self.btn_o.clicked.connect(self.start_o)
        l2.addRow(self.btn_o); ctrl_layout.addWidget(g2)
        layout.addLayout(ctrl_layout)

        # 進度條
        self.p_bar = QProgressBar(); self.p_bar.setFixedHeight(4); self.p_bar.setTextVisible(False)
        self.p_bar.setStyleSheet(f"""
            QProgressBar {{ border:none; background:{TV['surface']}; }}
            QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {TV['purple']}, stop:1 {TV['orange']}); }}
        """)
        layout.addWidget(self.p_bar); self.p_bar.hide()

        # 控制台輸出
        con_lbl = QLabel("▸  AI 引擎實況輸出")
        con_lbl.setStyleSheet(f"color: {TV['text_dim']}; font-size: 11px; font-weight: 700; letter-spacing: 1px; margin-top: 4px;")
        layout.addWidget(con_lbl)
        self.console = QTextEdit(); self.console.setReadOnly(True)
        self.console.setStyleSheet(f"""
            QTextEdit {{
                background-color: {TV['console_bg']};
                color: {TV['green']};
                font-family: 'Cascadia Code', 'Consolas', monospace;
                font-size: 13px;
                border: 1px solid {TV['border']};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        layout.addWidget(self.console, stretch=1)

    def log(self, m): self.console.append(m); self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())
    def start_f(self):
        try: d = int(self.days_in.text())
        except Exception: return QMessageBox.critical(self, "錯誤", "請輸入天數數字")
        
        # 🟢 已移除舊版玉山 API 的 7 天限制！現在可以自由輸入 15 或 30 天
        if d > 60:
            QMessageBox.warning(self, "API 提示", "為確保記憶體穩定，單次採集建議不超過 60 天。已為您調整為 60 天。")
            d = 60
            self.days_in.setText("60")

        self.console.clear(); self.btn_f.setEnabled(False)
        self.p_bar.setValue(0); self.p_bar.show()
        
        # 呼叫升級版的採集執行緒
        self.thread = FetchSimilarityDataThread(d)
        self.thread.log_signal.connect(self.log); self.thread.progress_signal.connect(self.p_bar.setValue)
        self.thread.finished_signal.connect(lambda s, m: (self.btn_f.setEnabled(True), self.log(m), self.p_bar.hide()))
        self.thread.start()
    def start_o(self):
        try: w = int(self.wait_in.text()); h = 270 if self.hold_in.text().upper() == 'F' else int(self.hold_in.text())
        except Exception: return QMessageBox.critical(self, "錯誤", "時間參數錯誤")
        self.console.clear(); self.btn_o.setEnabled(False)
        self.p_bar.setRange(0, 0); self.p_bar.show()
        self.thread = OptimizeSimilarityThread(w, h)
        self.thread.log_signal.connect(self.log); self.thread.finished_signal.connect(lambda: (self.btn_o.setEnabled(True), self.p_bar.hide(), self.p_bar.setRange(0, 100)))
        self.thread.start()

class FetchSimilarityDataThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, days_to_fetch): 
        super().__init__()
        self.days_to_fetch = days_to_fetch

    def run(self):
        db_folder = "回測大數據庫"
        os.makedirs(db_folder, exist_ok=True)
        db_path = os.path.join(db_folder, "similarity_data.db")
        
        # 清除舊有的 DB 檔案與可能殘留的 JSON 檔
        if os.path.exists(db_path):
            try: os.remove(db_path)
            except Exception: pass
        for f in glob.glob(os.path.join(db_folder, "*.json")):
            try: os.remove(f)
            except Exception: pass
            
        self.log_signal.emit("<span style='color:#9C27B0;'>⏳ 正在登入 Shioaji API (大數據批量採集模式)...</span>")
        
        # 借用全域已登入的 api 物件
        api = sys_state.api
        if not getattr(api, 'positions', None): 
            api.login(api_key=shioaji_logic.TEST_API_KEY, secret_key=shioaji_logic.TEST_API_SECRET, contracts_timeout=10000)
        
        all_symbols, _ = load_target_symbols()
        if not all_symbols: 
            return self.finished_signal.emit(False, "找不到符合的股票清單。")
            
        end_dt = datetime.today()
        # 往前推算足夠的日曆天 (乘以2加15天緩衝，以涵蓋春節等長假)
        start_dt = end_dt - timedelta(days=self.days_to_fetch * 2 + 15) 
        fetch_start = start_dt.strftime("%Y-%m-%d")
        fetch_end = end_dt.strftime("%Y-%m-%d")

        self.log_signal.emit("正在下載歷史資料並重構歷史處置股名單 (需時約10秒)...")
        dispo_set = self.fetch_multi_day_dispo(self.days_to_fetch)
        
        symbols = all_symbols 
        self.log_signal.emit(f"📋 準備採集 {len(symbols)} 檔股票之全指標大數據...")
        
        daily_database = {}
        total_syms = len(symbols)
        
        for idx, sym in enumerate(symbols):
            shioaji_limiter.wait_and_consume()
            if (idx+1) % 5 == 0:
                self.log_signal.emit(f"⏳ 正在採集 [{idx+1}/{total_syms}] {sn(sym)} ...")
                
            contract = api.Contracts.Stocks.get(sym)
            if not contract: continue
            
            try:
                # 1. 批量抓取區間資料
                kbars = api.kbars(contract, start=fetch_start, end=fetch_end)
                df = pd.DataFrame({**kbars})
                if df.empty: continue
                
                df.columns = [c.lower() for c in df.columns]
                
                # 🔴 關鍵修復：在進入補齊引擎前，先建立 ts, day 與 symbol
                df['ts'] = pd.to_datetime(df['ts'])
                df['day'] = df['ts'].dt.strftime('%Y-%m-%d')
                df['symbol'] = sym
                
                # 2. 直接套用主程序的時空平移與 0 軸補齊引擎
                df = fill_zero_volume_kbars(df)
                if df.empty: continue
                
                # 3. 計算基準價
                daily_last = df.groupby('day')['close'].last().shift(1)
                df['yesterday_close'] = df['day'].map(daily_last)
                df = df.dropna(subset=['yesterday_close'])
                if df.empty: continue
                
                # 4. 指標計算與浮點數斬斷
                df['limit_up'] = df['yesterday_close'].apply(calculate_limit_up_price)
                df['limit_up'] = df['limit_up'].apply(lambda x: math.floor(x * 100) / 100.0 if pd.notnull(x) else x)
                df['rise'] = round((df['close'] - df['yesterday_close']) / df['yesterday_close'] * 100, 2)
                df['highest'] = df.groupby('day')['high'].cummax()
                df['pct_increase'] = df.groupby('day')['rise'].diff().fillna(df['rise']).round(2)
                
                df.rename(columns={'yesterday_close': '昨日收盤價', 'limit_up': '漲停價'}, inplace=True)
                cols = ['symbol', 'day', 'time', 'open', 'high', 'low', 'close', 'volume', '昨日收盤價', '漲停價', 'rise', 'highest', 'pct_increase']
                df = df[cols]
                
                # 5. 依日期分裝並過濾處置股
                for d_str, grp in df.groupby('day'):
                    if (d_str, sym) in dispo_set: continue
                    
                    d_key = d_str.replace("-", "")
                    if d_key not in daily_database: daily_database[d_key] = {}
                    
                    grp = grp.rename(columns={'day': 'date'})
                    daily_database[d_key][sym] = grp.to_dict('records')
                    
            except Exception as e:
                self.log_signal.emit(f"  ❌ {sym} 錯誤: {e}")
                
            self.progress_signal.emit(int((idx+1)/total_syms * 100))

        # 6. 排序並只保留最近的 days_to_fetch 天實際有開盤的日期
        available_days = sorted(daily_database.keys())[-self.days_to_fetch:]
        
        # 🟢 升級：將所有數據打包，統一寫入 SQLite 資料庫
        all_records = []
        for d_key in available_days:
            for sym, recs in daily_database[d_key].items():
                all_records.extend(recs)
                
        if all_records:
            final_df = pd.DataFrame(all_records)
            conn = sqlite3.connect(db_path)
            final_df.to_sql('kline_data', conn, if_exists='replace', index=False)
            conn.close()
            self.finished_signal.emit(True, f"✅ 成功採集 {len(available_days)} 個純淨交易日，並統一封裝為 .db 資料庫！")
        else:
            self.finished_signal.emit(False, "❌ 無有效數據可儲存，請確認是否為假日或網路異常。")

    def fetch_multi_day_dispo(self, days):
        """多日處置股極速爬蟲 (已升級 TPEx 雙軌解析與防護)"""
        import requests, re
        from datetime import datetime, timedelta
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        end_obj = datetime.today()
        dispo_set = set()
        tpex_cache = {}
        
        twse_headers = {'User-Agent': 'Mozilla/5.0'}
        tpex_headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'X-Requested-With': 'XMLHttpRequest'
        }
        
        # 先緩存上櫃資料
        for i in range(days + 20):
            curr_date = end_obj - timedelta(days=i)
            if curr_date.weekday() >= 5: continue
            roc_year = curr_date.year - 1911
            tpex_date = f"{roc_year}/{curr_date.strftime('%m/%d')}"
            syms_today = []
            try:
                url = f"https://www.tpex.org.tw/web/bulletin/disposal_information/disposal_information_result.php?l=zh-tw&d={tpex_date}&o=json"
                res = requests.get(url, headers=tpex_headers, timeout=5, verify=False)
                text = res.text.strip()
                if not text or text.startswith('<'): continue
                
                data = res.json()
                if 'tables' in data:
                    for table in data['tables']:
                        if 'data' in table:
                            for row in table['data']:
                                if len(row) > 2 and re.fullmatch(r'\d{4,6}', str(row[2]).strip()):
                                    syms_today.append(str(row[2]).strip())
                elif 'aaData' in data:
                    for row in data['aaData']:
                        for val in row:
                            if re.fullmatch(r'\d{4,6}', str(val).strip()):
                                syms_today.append(str(val).strip())
                                break
            except Exception: pass
            tpex_cache[curr_date] = syms_today

        # 組合每日名單
        for i in range(days + 5):
            target_date = end_obj - timedelta(days=i)
            if target_date.weekday() >= 5: continue
            target_str = target_date.strftime('%Y-%m-%d')
            
            start_date = target_date - timedelta(days=15)
            twse_start = start_date.strftime('%Y%m%d')
            twse_end = target_date.strftime('%Y%m%d')
            
            try:
                twse_url = f"https://www.twse.com.tw/announcement/punish?response=json&startDate={twse_start}&endDate={twse_end}"
                res_twse = requests.get(twse_url, headers=twse_headers, timeout=5, verify=False).json()
                if 'data' in res_twse:
                    for row in res_twse['data']:
                        for cell in row:
                            cell_str = str(cell).strip()
                            if re.fullmatch(r'\d{4,6}', cell_str):
                                dispo_set.add((target_str, cell_str))
                                break
            except Exception: pass
            
            for j in range(16):
                check_date = target_date - timedelta(days=j)
                if check_date in tpex_cache:
                    for sym in tpex_cache[check_date]:
                        dispo_set.add((target_str, sym))
                        
        return dispo_set

class OptimizeSimilarityThread(QThread):
    log_signal = pyqtSignal(str); finished_signal = pyqtSignal()
    def __init__(self, wait_mins, hold_mins): super().__init__(); self.wait_mins = wait_mins; self.hold_mins = hold_mins
    
    def run(self):
        try:
            cap_val = sys_config.capital_per_stock
            f_rate, d_rate, t_rate = sys_config.transaction_fee*0.01, sys_config.transaction_discount*0.01, sys_config.trading_tax*0.01
            try: dispo = set(load_disposition_stocks())
            except Exception: dispo = set()
            _, groups = load_target_symbols() 

            # 🟢 升級：直接讀取統一的 DB 檔案
            db_path = "回測大數據庫/similarity_data.db"
            if not os.path.exists(db_path): 
                return self.log_signal.emit("❌ 找不到大數據庫！請先執行「資料採集」。")

            conn = sqlite3.connect(db_path)
            df_all = pd.read_sql("SELECT * FROM kline_data", conn)
            conn.close()
            
            if df_all.empty:
                return self.log_signal.emit("❌ 大數據庫內容為空！")

            unique_dates = sorted(df_all['date'].unique().tolist())

            self.log_signal.emit(f"<br><span style='color:#FFDC00;'>{'=' * 62}</span>")
            self.log_signal.emit(f"🚀 啟動邏輯同步模擬 (分析 {len(unique_dates)} 天數據)")
            self.log_signal.emit(f"<span style='color:#FFDC00;'>{'-' * 62}</span>")
            self.log_signal.emit(f"<b>{'      門檻 |     交易數 |       勝率 |     平均淨利 |   穩健分數'.replace(' ', '&nbsp;')}</b>")
            self.log_signal.emit(f"<span style='color:#FFDC00;'>{'-' * 62}</span>")

            thresholds = np.arange(0.05, 0.95, 0.05)
            stats = {th: {'signals': 0, 'pnl_sum': 0.0, 'wins': 0} for th in thresholds}
            best_th, max_score = 0.75, -999.0

            for t_date in unique_dates:
                df_day = df_all[df_all['date'] == t_date]
                # 將這天的數據還原回字典格式以相容引擎
                history_data = {str(sym): grp.to_dict('records') for sym, grp in df_day.groupby('symbol')}
                
                for th in thresholds:
                    for grp_name, stocks in groups.items():
                        valid_s = [s for s in stocks if s not in dispo and s in history_data]
                        if len(valid_s) < 2: continue
                        stock_dfs = {}
                        for sym in valid_s:
                            df = pd.DataFrame(history_data[sym])
                            if df['volume'].sum() == 0: continue 
                            df['time'] = pd.to_datetime(df['time'], format="%H:%M:%S").dt.time
                            stock_dfs[sym] = df
                        if not stock_dfs: continue

                        leader, tracking = None, set()
                        in_wait, wait_cnt, start_t, leader_peak_rise = False, 0, None, None
                        pull_up, limit_up = False, False
                        first3_vol = {s: df.iloc[0:3]['volume'].mean() for s, df in stock_dfs.items()}
                        is_busy, exit_at = False, -1
                        day_times = stock_dfs[list(stock_dfs.keys())[0]]['time'].tolist()

                        stock_records = {s: df.to_dict('records') for s, df in stock_dfs.items()}

                        for m in range(271):
                            if is_busy:
                                if m >= exit_at: is_busy = False
                                continue
                            cur_t = day_times[m]
                            row_data = {s: stock_records[s][m] for s in stock_dfs.keys()}
                            
                            trigger_list = []
                            for sym in stock_dfs.keys():
                                row, avgv = row_data[sym], first3_vol[sym]
                                if round(row['high'], 2) >= round(row['漲停價'], 2):
                                    if m == 0 or round(stock_records[sym][m-1]['high'], 2) < round(row['漲停價'], 2): trigger_list.append((sym, 'limit'))
                                elif row['pct_increase'] >= 2.0 and avgv > 0 and row['volume'] > 1.3 * avgv: trigger_list.append((sym, 'pull'))

                            for sym, cond in trigger_list:
                                if cond == 'limit':
                                    tracking.add(sym); leader, in_wait, wait_cnt = sym, True, 0
                                    if not (pull_up or limit_up): start_t = cur_t
                                    pull_up, limit_up = False, True
                                else:
                                    if not pull_up and not limit_up: pull_up, limit_up, start_t = True, False, cur_t; tracking.clear()
                                    tracking.add(sym)

                            if pull_up or limit_up:
                                for sym in stock_dfs.keys():
                                    if sym not in tracking and row_data[sym]['pct_increase'] >= 1.5: tracking.add(sym)

                            if tracking:
                                max_sym, max_r = None, -999
                                for sym in tracking:
                                    if row_data[sym]['rise'] > max_r: max_r, max_sym = row_data[sym]['rise'], sym
                                if leader != max_sym: leader, start_t, in_wait, wait_cnt, leader_peak_rise = max_sym, cur_t, False, 0, max_r 
                                if leader and row_data[leader]['high'] <= stock_records[leader][max(0, m-1)]['high'] and not in_wait: in_wait, wait_cnt, leader_peak_rise = True, 0, max_r 

                            if in_wait:
                                w_start = max(time(9,0), (datetime.combine(date.today(), start_t) - timedelta(minutes=2)).time())
                                to_remove = [sym for sym in list(tracking) if sym != leader and calculate_dtw_pearson(stock_dfs[leader], stock_dfs[sym], w_start, cur_t) < th]
                                for sym in to_remove: tracking.remove(sym)

                                if wait_cnt >= self.wait_mins:
                                    eligible = []
                                    for sym in tracking:
                                        if sym == leader: continue
                                        df_wait = stock_dfs[sym][(stock_dfs[sym]['time'] >= start_t) & (stock_dfs[sym]['time'] <= cur_t)]
                                        if not (df_wait['volume'] >= 1.5 * first3_vol[sym]).any() or (len(df_wait) >= 2 and df_wait.iloc[-1]['rise'] > df_wait.iloc[:-1]['rise'].max() + 0.5): continue
                                        if not (-1 <= row_data[sym]['rise'] <= 6) or row_data[sym]['close'] > cap_val * 1.5: continue
                                        eligible.append({'sym': sym, 'rise': row_data[sym]['rise'], 'row': row_data[sym]})

                                    if eligible:
                                        eligible.sort(key=lambda x: x['rise'], reverse=True)
                                        target = eligible[len(eligible)//2] 
                                        p_ent = target['row']['close']
                                        shrs = round((cap_val * 10000) / (p_ent * 1000))
                                        sell_total = shrs * p_ent * 1000
                                        ent_fee, tax = int(sell_total * f_rate * d_rate), int(sell_total * t_rate)
                                        gap, tick = get_stop_loss_config(p_ent)
                                        hi_on_e = target['row']['highest'] or p_ent
                                        stop_p = hi_on_e + tick if (hi_on_e - p_ent)*1000 >= gap else p_ent + gap/1000
                                        
                                        m_end = min(m + self.hold_mins, 270)
                                        for m_exit in range(m + 1, m_end + 1):
                                            r_ex = stock_records[target['sym']][m_exit]
                                            if r_ex['high'] >= stop_p or m_exit == m_end:
                                                p_exit = stop_p if r_ex['high'] >= stop_p else r_ex['close']
                                                buy_total = shrs * p_exit * 1000
                                                profit = sell_total - buy_total - ent_fee - int(buy_total * f_rate * d_rate) - tax
                                                stats[th]['signals'] += 1; stats[th]['pnl_sum'] += (profit * 100) / (buy_total - int(buy_total * f_rate * d_rate))
                                                if profit > 0: stats[th]['wins'] += 1
                                                is_busy, exit_at = True, m_exit; break
                                    
                                    pull_up = limit_up = False; leader, tracking, in_wait, wait_cnt = None, set(), False, 0
                                else:
                                    if leader and leader_peak_rise is not None and row_data[leader]['rise'] > leader_peak_rise: leader_peak_rise, in_wait, wait_cnt = row_data[leader]['rise'], False, 0
                                    else: wait_cnt += 1

            for th in thresholds:
                n = stats[th]['signals']
                avg_p, wr = (stats[th]['pnl_sum'] / n) if n > 0 else 0, (stats[th]['wins'] / n * 100) if n > 0 else 0
                score = avg_p * np.log10(n + 1) if avg_p > 0 else avg_p
                self.log_signal.emit(f"<span style='color:{'#2ECC40' if avg_p > 0 else '#FF4136'}; font-family:Consolas, \"MingLiU\", monospace;'>{f'{th:>10.2f} | {n:>10} | {wr:>8.1f} % | {avg_p:>8.1f} % | {score:>10.2f}'.replace(' ', '&nbsp;')}</span>")
                if score > max_score and n > 0: max_score, best_th = score, th
            
            self.log_signal.emit(f"<span style='color:#FFDC00;'>{'-' * 62}</span>")
            self.log_signal.emit(f"<span style='color:#2ECC40; font-weight:bold;'>同步後建議門檻：DTW &gt;= {best_th:.2f}</span>")
            self.log_signal.emit(f"<span style='color:#2ECC40;'>💡 說明：穩健分數越高，代表在該門檻下獲利越穩定。</span>")
            self.log_signal.emit(f"<span style='color:#2ECC40;'>💡 建議：可依據個人風險承受度，微調 ±0.05 的門檻值。</span>")
            self.log_signal.emit(f"<span style='color:#FFDC00;'>{'=' * 62}</span><br>")
        except Exception as e: self.log_signal.emit(f"<span style='color:#FF4136;'>❌ 系統錯誤：<br>{traceback.format_exc().replace(chr(10), '<br>')}</span>")
        finally: self.finished_signal.emit()

class LoginDialog(BaseDialog):
    def __init__(self):
        super().__init__("帳戶 API 設定  │  REMORA", (600, 540))
        root = QVBoxLayout(self); root.setSpacing(16); root.setContentsMargins(20, 20, 20, 20)

        _grp_style = lambda c: f"QGroupBox{{color:{c};font-weight:bold;border:1px solid {TV['border_light']};border-radius:8px;margin-top:14px;padding-top:10px;}}"
        _sub_style = lambda c: f"QGroupBox{{color:{c};font-weight:700;border:1px solid {TV['border_light']};border-radius:6px;margin-top:12px;padding-top:6px;}}"
        _browse_style = f"QPushButton{{background:{TV['surface']};color:{TV['text']};border:1px solid {TV['border_light']};border-radius:4px;padding:4px 8px;font-size:12px;}} QPushButton:hover{{border-color:{TV['blue']};}}"

        # ── 兩欄並排 ──
        cols = QHBoxLayout(); cols.setSpacing(16)

        # ── 左欄：玉山證券 ──
        g_esun = QGroupBox("玉山證券  /  看盤數據源"); g_esun.setStyleSheet(_grp_style(TV['yellow']))
        l_esun = QFormLayout(g_esun); l_esun.setSpacing(10)
        self.e_esun_key, self.e_esun_sec, self.e_esun_acc, self.e_esun_cert = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        if os.path.exists('config.ini'):
            c = ConfigParser(); c.read('config.ini', encoding='utf-8-sig')
            self.e_esun_key.setText(c.get('Api', 'Key', fallback='')); self.e_esun_sec.setText(c.get('Api', 'Secret', fallback=''))
            self.e_esun_acc.setText(c.get('User', 'Account', fallback='')); self.e_esun_cert.setText(c.get('Cert', 'Path', fallback=''))
        l_esun.addRow("esun_api_key:", self.e_esun_key); l_esun.addRow("esun_api_Secret:", self.e_esun_sec); l_esun.addRow("esun_user_Account:", self.e_esun_acc)
        cl = QHBoxLayout(); cl.addWidget(self.e_esun_cert)
        b1 = QPushButton("瀏覽..."); b1.setStyleSheet(_browse_style); b1.clicked.connect(lambda: self.e_esun_cert.setText(os.path.basename(QFileDialog.getOpenFileName(self, "", "", "*.p12 *.pfx")[0]))); cl.addWidget(b1)
        l_esun.addRow("esun_Cert_path:", cl)
        cols.addWidget(g_esun)

        # ── 右欄：永豐金 ──
        g_shio = QGroupBox("永豐金證券 Shioaji  /  實際下單"); g_shio.setStyleSheet(_grp_style(TV['blue']))
        l_shio = QVBoxLayout(g_shio); l_shio.setSpacing(8)

        g_sim = QGroupBox("模擬帳戶"); g_sim.setStyleSheet(_sub_style(TV['blue']))
        l_sim = QFormLayout(g_sim); l_sim.setSpacing(8)
        self.e_api = QLineEdit(shioaji_logic.TEST_API_KEY); self.e_sec = QLineEdit(shioaji_logic.TEST_API_SECRET)
        l_sim.addRow("TEST_API_KEY:", self.e_api); l_sim.addRow("TEST_API_SECRET:", self.e_sec)
        l_shio.addWidget(g_sim)

        g_live = QGroupBox("正式帳戶  ⚠ 使用真實資金"); g_live.setStyleSheet(_sub_style(TV['red']))
        l_live = QFormLayout(g_live); l_live.setSpacing(8)
        self.e_live_api = QLineEdit(getattr(shioaji_logic, 'LIVE_API_KEY', '')); self.e_live_sec = QLineEdit(getattr(shioaji_logic, 'LIVE_API_SECRET', ''))
        l_live.addRow("LIVE_API_KEY:", self.e_live_api); l_live.addRow("LIVE_API_SECRET:", self.e_live_sec)
        self.chk_live = QCheckBox("啟用正式下單模式（勾選後將使用真實資金！）")
        self.chk_live.setStyleSheet(f"color: {TV['red']}; font-weight: 700;")
        self.chk_live.setChecked(getattr(sys_config, 'live_trading_mode', False))
        self.chk_live.stateChanged.connect(self._on_live_mode_changed)
        l_live.addRow(self.chk_live)
        l_shio.addWidget(g_live)

        l_ca = QFormLayout(); l_ca.setSpacing(8)
        self.e_ca = QLineEdit(shioaji_logic.CA_CERT_PATH); self.e_pw = QLineEdit(shioaji_logic.CA_PASSWORD)
        cl2 = QHBoxLayout(); cl2.addWidget(self.e_ca)
        b2 = QPushButton("瀏覽..."); b2.setStyleSheet(_browse_style); b2.clicked.connect(lambda: self.e_ca.setText(QFileDialog.getOpenFileName(self, "", "", "*.p12 *.pfx")[0])); cl2.addWidget(b2)
        l_ca.addRow("CA_CERT_PATH:", cl2); l_ca.addRow("CA_PASSWORD:", self.e_pw)
        l_shio.addLayout(l_ca)
        cols.addWidget(g_shio)

        root.addLayout(cols)

        btn = QPushButton("✓ 儲存並套用設定"); btn.setFixedHeight(46)
        btn.setStyleSheet(f"""
            QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {TV['green']}, stop:1 #1a7a6e);
                           color: white; font-size: 14px; font-weight: 700; border: none; border-radius: 8px; }}
            QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2bc4b4, stop:1 {TV['green']}); }}
        """)
        btn.clicked.connect(self.save); root.addWidget(btn)

    def _on_live_mode_changed(self, state):
        if state:
            reply = QMessageBox.warning(self, "警告",
                "您即將啟用【正式下單模式】！\n\n所有委託將使用真實資金執行！\n\n確定要啟用嗎？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                self.chk_live.setChecked(False)

    def save(self):
        _prev_live = getattr(sys_config, 'live_trading_mode', False)
        update_variable("shioaji_logic.py", "TEST_API_KEY", self.e_api.text()); update_variable("shioaji_logic.py", "TEST_API_SECRET", self.e_sec.text())
        update_variable("shioaji_logic.py", "CA_CERT_PATH", self.e_ca.text(), is_raw=True); update_variable("shioaji_logic.py", "CA_PASSWORD", self.e_pw.text())
        update_variable('shioaji_logic.py', 'LIVE_API_KEY', self.e_live_api.text())
        update_variable('shioaji_logic.py', 'LIVE_API_SECRET', self.e_live_sec.text())
        sys_config.live_trading_mode = self.chk_live.isChecked()
        save_settings()
        print_trading_mode()
        _new_live = sys_config.live_trading_mode
        if _prev_live != _new_live:
            if getattr(sys_config, 'is_monitoring', False):
                QMessageBox.warning(self, "無法即時切換",
                    "盤中監控執行中，下單模式無法即時切換。\n"
                    "請先停止監控，再重新開啟「登入/修改帳戶」切換。")
            else:
                try:
                    try: sys_state.api.logout()
                    except Exception: pass
                    _ak = shioaji_logic.LIVE_API_KEY if _new_live else shioaji_logic.TEST_API_KEY
                    _as = shioaji_logic.LIVE_API_SECRET if _new_live else shioaji_logic.TEST_API_SECRET
                    sys_state.api = sj.Shioaji(simulation=not _new_live)
                    sys_state.api.login(api_key=_ak, secret_key=_as, contracts_timeout=10000)
                    sys_state.api.set_order_callback(order_callback)  # 🔧 重連後補註冊 callback
                except Exception as _e:
                    QMessageBox.warning(self, "切換失敗",
                        f"API 重新連線失敗（{_e}），\n設定已儲存，請重新啟動程式使其生效。")
        c = ConfigParser(); c.read('config.ini', encoding='utf-8-sig') if os.path.exists('config.ini') else None
        for sec in ['Core', 'Api', 'Cert', 'User']:
            if not c.has_section(sec): c.add_section(sec)
        c.set('Core', 'Entry', 'https://esuntradingapi-simulation.esunsec.com.tw/api/v1'); c.set('Core', 'Environment', 'SIMULATION')
        c.set('Api', 'Key', self.e_esun_key.text().strip()); c.set('Api', 'Secret', self.e_esun_sec.text().strip()); c.set('User', 'Account', self.e_esun_acc.text().strip()); c.set('Cert', 'Path', self.e_esun_cert.text().strip())
        with open('config.ini', 'w', encoding='utf-8') as f: c.write(f)
        # 無論 live mode 是否切換，都重新嘗試登入 + activate_ca（使新憑證立即生效）
        if not getattr(sys_config, 'is_monitoring', False):
            try:
                _ak = shioaji_logic.LIVE_API_KEY if _new_live else shioaji_logic.TEST_API_KEY
                _as = shioaji_logic.LIVE_API_SECRET if _new_live else shioaji_logic.TEST_API_SECRET
                try: sys_state.api.logout()
                except Exception: pass
                sys_state.api = sj.Shioaji(simulation=not _new_live)
                sys_state.api.login(api_key=_ak, secret_key=_as, contracts_timeout=10000)
                if shioaji_logic.CA_CERT_PATH and shioaji_logic.CA_PASSWORD:
                    sys_state.api.activate_ca(
                        ca_path=shioaji_logic.CA_CERT_PATH,
                        ca_passwd=shioaji_logic.CA_PASSWORD
                    )
                sys_state.api.set_order_callback(order_callback)  # 🔧 重連後補註冊 callback
                QMessageBox.information(self, "成功", "雙券商設定已儲存！永豐 API 重新連線成功。")
            except Exception as _e:
                QMessageBox.warning(self, "永豐連線失敗",
                    f"設定已儲存，但永豐 API 重新連線失敗：\n{_e}\n\n"
                    "請確認 CA 憑證路徑與密碼正確，或重新啟動程式使其生效。")
        else:
            QMessageBox.information(self, "成功", "雙券商設定已儲存！")
        self.accept()

class TradeDialog(BaseDialog):
    def __init__(self):
        super().__init__("啟動盤中監控 - 參數設定", (400, 280))
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        wrapper = QWidget(); wrapper.setMaximumWidth(520)
        layout = QFormLayout(wrapper); layout.setSpacing(12); layout.setContentsMargins(24, 24, 24, 24)
        self.w_wait = QLineEdit("3"); self.w_hold = QLineEdit("F")
        layout.addRow("等待時間 (分鐘):", self.w_wait)
        layout.addRow("持有時間 (分鐘, F=尾盤):", self.w_hold)

        btn = QPushButton("▶ 啟動盤中監控")
        btn.setFixedHeight(42)
        btn.setStyleSheet(f"""
            QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {TV['green']}, stop:1 #1a7a6e);
                           color:white; border:none; border-radius:7px; font-size:14px; font-weight:700; }}
            QPushButton:hover {{ background: {TV['green']}; }}
        """)
        btn.clicked.connect(self.run_trade); layout.addRow(btn)

        btn_login = QPushButton("設定帳戶 API")
        btn_login.setFixedHeight(38)
        btn_login.setStyleSheet(f"""
            QPushButton {{ background-color: {TV['surface']}; color: {TV['text']}; border: 1px solid {TV['border_light']}; border-radius: 7px; font-size: 13px; font-weight: 600; }}
            QPushButton:hover {{ border-color: {TV['blue']}; color: white; }}
        """)
        btn_login.clicked.connect(self.open_login_dialog); layout.addRow(btn_login)
        outer.addStretch(); outer.addWidget(wrapper, 0, Qt.AlignHCenter); outer.addStretch()

    def open_login_dialog(self):
        if _main_window_ref:
            _main_window_ref._ensure_tab('login', '帳戶設定', LoginDialog)
        else:
            if not hasattr(self, 'dlg_login') or not self.dlg_login.isVisible():
                self.dlg_login = LoginDialog(); self.dlg_login.show()
            else:
                self.dlg_login.raise_(); self.dlg_login.activateWindow()

    def run_trade(self):
        if getattr(sys_state, 'is_monitoring', False):
            return QMessageBox.warning(self, "提示", "盤中監控已在運行中，請勿重複啟動。")
        
        try: w = int(self.w_wait.text())
        except Exception: return QMessageBox.critical(self, "錯誤", "等待時間需為整數")
        try: h = None if self.w_hold.text().strip().upper() == 'F' else int(self.w_hold.text().strip().upper());
        except Exception: return QMessageBox.critical(self, "錯誤", "持有時間格式錯誤")
        if h is not None and h < 1: return QMessageBox.critical(self, "錯誤", "持有時間最少需為 1 分鐘")
        if ensure_esun_passwords():
            self.accept()
            if _main_window_ref:
                _main_window_ref.tabs.setCurrentIndex(0)
                _main_window_ref._show_toast("正在啟動盤中監控...")
            threading.Thread(target=start_trading, args=('full', w, h), daemon=True).start()

class CorrelationResultDialog(BaseDialog):
    def __init__(self, result_data, parent=None):
        super().__init__("🧬 族群連動分析結果", (850, 600))
        self.result_data = result_data 
        layout = QVBoxLayout(self)
        self.table = QTableWidget(); self.table.setColumnCount(6); self.table.setHorizontalHeaderLabels(["族群", "領漲股", "跟漲股", "時間窗", "DTW相似度", "結果"])
        self.table.setStyleSheet(f"QTableWidget {{ background-color: {TV['bg']}; color: {TV['text']}; gridline-color: {TV['border']}; }} QTableWidget::item {{ color: {TV['text']}; }} QHeaderView::section {{ background-color: {TV['surface']}; color: {TV['text_bright']}; font-weight: bold; border: 1px solid {TV['border']}; }}")
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.table.setRowCount(len(result_data))
        
        # 提取系統當前的 DTW 相似度門檻
        threshold = sys_config.similarity_threshold
        
        for i, r in enumerate(result_data):
            self.table.setItem(i, 0, QTableWidgetItem(str(r['group']))); self.table.setItem(i, 1, QTableWidgetItem(str(r['leader']))); self.table.setItem(i, 2, QTableWidgetItem(str(r['follower']))); self.table.setItem(i, 3, QTableWidgetItem(str(r['window'])))
            
            sim_item = QTableWidgetItem(f"{r['similarity']:.3f}")
            # 顏色判定改用 threshold 變數
            sim_item.setForeground(QColor("#2ECC40" if r['similarity'] >= threshold else "#FF4136"))
            self.table.setItem(i, 4, sim_item)
            
            # 文字判定改用 threshold 變數
            self.table.setItem(i, 5, QTableWidgetItem("✅ 合格" if r['similarity'] >= threshold else "❌ 剔除"))
            
        layout.addWidget(self.table)
        btn_export = QPushButton("📥 匯出 CSV"); btn_export.setStyleSheet(f"background-color: {TV['green']}; color: white; border-radius: 5px;"); btn_export.clicked.connect(self.export_to_csv); layout.addWidget(btn_export)

    def export_to_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "儲存", "族群連動分析結果.csv", "CSV 檔案 (*.csv)")
        if path:
            try:
                # 匯出 CSV 時，同樣讀取當前門檻
                threshold = sys_config.similarity_threshold
                
                with open(path, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.writer(f); writer.writerow(["族群", "領漲股", "跟漲股", "時間窗", "相似度", "結果"])
                    for r in self.result_data: 
                        # CSV 結果寫入也改用 threshold 判定
                        writer.writerow([r['group'], r['leader'], r['follower'], r['window'], f"{r['similarity']:.3f}", "合格" if r['similarity'] >= threshold else "剔除"])
                QMessageBox.information(self, "成功", f"儲存至：\n{path}")
            except Exception as e: QMessageBox.critical(self, "失敗", f"發生錯誤：\n{e}")

class CorrelationConfigDialog(BaseDialog):
    def __init__(self, main_window):
        super().__init__("設定連動分析參數", (400, 200))
        self.main_window = main_window
        layout = QVBoxLayout(self)
        btn_tut = _make_tutorial_btn('correlation')
        tut_row = QHBoxLayout(); tut_row.addStretch(); tut_row.addWidget(btn_tut)
        layout.addLayout(tut_row)

        self.mode_combo = QComboBox()
        self.mode_combo.setView(QListView()) 
        self.mode_combo.addItems(["[A] 宏觀連動 (09:00~13:30)", "[B] 微觀模擬"])
        
        self.wait_spin = QLineEdit("3")
        
        form_layout = QFormLayout()
        form_layout.addRow("分析模式：", self.mode_combo)
        form_layout.addRow("微觀等待 (分鐘)：", self.wait_spin)
        layout.addLayout(form_layout)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.start_analysis)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def start_analysis(self):
        self.accept()
        mode = "macro" if self.mode_combo.currentIndex() == 0 else "micro"
        w_mins = int(self.wait_spin.text()) if self.wait_spin.text().isdigit() else 5
        if _main_window_ref:
            _main_window_ref.tabs.setCurrentIndex(0)
            _main_window_ref._show_toast("▶ 正在執行族群連動分析...")
        # 呼叫主視窗去跑背景分析
        self.main_window.start_correlation_thread(mode, w_mins)

class AnalysisMenuDialog(BaseDialog):
    def __init__(self, main_window):
        super().__init__("盤後數據與分析中心", (700, 500))
        self.main_window = main_window
        outer = QVBoxLayout(self)
        outer.setContentsMargins(60, 30, 60, 30)

        title = QLabel("盤後數據與分析中心")
        title.setStyleSheet(f"color: {TV['text_bright']}; font-size: 20px; font-weight: 800; padding-bottom: 6px;")
        title.setAlignment(Qt.AlignCenter)
        outer.addWidget(title)

        subtitle = QLabel("選擇要執行的分析模組")
        subtitle.setStyleSheet(f"color: {TV['text_dim']}; font-size: 13px;")
        subtitle.setAlignment(Qt.AlignCenter)
        outer.addWidget(subtitle)
        outer.addSpacing(20)

        grid = QGridLayout(); grid.setSpacing(16)

        def _card(icon, name, desc, color, hover, cb):
            btn = QPushButton(f"  {icon}  {name}")
            btn.setFixedHeight(70)
            btn.setStyleSheet(f"""
                QPushButton {{ background-color: {TV['surface']}; color: {TV['text_bright']};
                    font-weight: 700; font-size: 15px; border: 1px solid {TV['border_light']};
                    border-radius: 10px; text-align: left; padding-left: 20px; }}
                QPushButton:hover {{ background-color: {color}; border-color: {color}; color: white; }}
            """)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(desc)
            btn.clicked.connect(cb)
            return btn

        grid.addWidget(_card("🏆", "最適相似度門檻", "DTW 門檻最佳化分析", TV['orange'], "#ff9800", self.click_opt_sim), 0, 0)
        grid.addWidget(_card("📈", "計算平均過高", "計算族群平均過高間隔", TV['blue'], TV['blue_hover'], self.click_avg_high), 0, 1)
        grid.addWidget(_card("🧬", "族群連動分析", "跨族群相關性掃描", TV['purple'], "#9c6dff", self.click_correlation), 1, 0)
        grid.addWidget(_card("💰", "利潤矩陣優化", "矩陣併發版回測模式", TV['red'], "#ff6b6b", self.click_maximize), 1, 1)

        outer.addLayout(grid)
        outer.addStretch()

    def click_opt_sim(self):
        self.main_window._ensure_tab('opt_sim', 'DTW最佳化', SimilarityOptimizationDialog)
    def click_avg_high(self):
        self.main_window._ensure_tab('avg_high', '平均過高', AverageHighDialog)
    def click_correlation(self):
        self.main_window._ensure_tab('corr_cfg', '連動分析', lambda: CorrelationConfigDialog(self.main_window))
    def click_maximize(self):
        self.main_window._ensure_tab('maximize', '利潤矩陣優化', MaximizeDialog)

class SimulateDialog(BaseDialog):
    def __init__(self):
        super().__init__("自選進場模式 (回測)", (400, 250))
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        wrapper = QWidget(); wrapper.setMaximumWidth(640)
        lo = QVBoxLayout(wrapper); lo.setContentsMargins(32, 32, 32, 32); lo.setSpacing(20)

        # 標題列
        title_row = QHBoxLayout()
        title = QLabel("自選回測 — 參數設定")
        title.setStyleSheet(f"color: {TV['text_bright']}; font-size: 17px; font-weight: 700;")
        title.setAlignment(Qt.AlignCenter)
        btn_tut = _make_tutorial_btn('simulate')
        title_row.addStretch(); title_row.addWidget(title); title_row.addStretch(); title_row.addWidget(btn_tut)
        lo.addLayout(title_row)

        _card_style = f"QFrame {{ background: {TV['panel']}; border: 1px solid {TV['border_light']}; border-radius: 10px; }}"
        _lbl_style = f"color: {TV['text_dim']}; font-size: 12px; font-weight: 600; letter-spacing: 1px;"
        _sub_style = f"color: {TV['text_dim']}; font-size: 12px;"
        _input_style = f"QLineEdit {{ background: {TV['bg']}; color: {TV['text_bright']}; border: 1px solid {TV['border_light']}; border-radius: 6px; font-size: 22px; font-weight: 700; padding: 6px 10px; }} QLineEdit:focus {{ border-color: {TV['blue']}; }}"
        _combo_style = f"QComboBox {{ background: {TV['bg']}; color: {TV['text_bright']}; border: 1px solid {TV['border_light']}; border-radius: 6px; font-size: 14px; font-weight: 600; padding: 6px 10px; }}"

        # 族群選擇卡片（全寬）
        grp_card = QFrame(); grp_card.setStyleSheet(_card_style); grp_card.setFixedHeight(100)
        gcl = QVBoxLayout(grp_card); gcl.setContentsMargins(20, 14, 20, 14); gcl.setSpacing(6)
        gcl.addWidget(QLabel("🏷  分析族群", styleSheet=_lbl_style))
        self.w_grp = QComboBox(); self.w_grp.setView(QListView()); self.w_grp.setStyleSheet(_combo_style)
        self.w_grp.setMaxVisibleItems(10)
        self.w_grp.addItem("所有族群"); self.w_grp.addItems(list(load_matrix_dict_analysis().keys()))
        gcl.addWidget(self.w_grp)
        lo.addWidget(grp_card)

        # 兩個並排卡片
        cards_row = QHBoxLayout(); cards_row.setSpacing(16)

        def _card(icon, label, default_val, sub_text, attr):
            card = QFrame(); card.setStyleSheet(_card_style); card.setFixedHeight(130)
            cl = QVBoxLayout(card); cl.setContentsMargins(20, 16, 20, 16); cl.setSpacing(6)
            cl.addWidget(QLabel(f"{icon}  {label}", styleSheet=_lbl_style))
            inp = QLineEdit(default_val); inp.setStyleSheet(_input_style); inp.setAlignment(Qt.AlignCenter)
            cl.addWidget(inp)
            cl.addWidget(QLabel(sub_text, styleSheet=_sub_style, alignment=Qt.AlignCenter))
            setattr(self, attr, inp)
            return card

        cards_row.addWidget(_card("⏱", "等待時間", "3", "分鐘", "w_wait"))
        cards_row.addWidget(_card("📌", "持有時間", "F", "F = 持到尾盤（13:25）", "w_hold"))
        lo.addLayout(cards_row)

        # 分析按鈕
        btn = QPushButton("▶ 開始分析")
        btn.setFixedHeight(50); btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"QPushButton {{ background: {TV['blue']}; color: white; border: none; border-radius: 10px; font-size: 15px; font-weight: 700; }} QPushButton:hover {{ background: {TV['blue_hover']}; }}")
        btn.clicked.connect(self.run_sim); lo.addWidget(btn)

        outer.addStretch(); outer.addWidget(wrapper, 0, Qt.AlignHCenter); outer.addStretch()

    def run_sim(self):
        grp = self.w_grp.currentText()
        try: w = int(self.w_wait.text())
        except Exception: return QMessageBox.critical(self, "錯誤", "等待時間需為整數")
        try: h = None if self.w_hold.text().upper() == 'F' else int(self.w_hold.text().upper())
        except Exception: return QMessageBox.critical(self, "錯誤", "持有時間格式錯誤")
        self.accept()
        # 切回系統日誌分頁並顯示提示
        if _main_window_ref:
            _main_window_ref.tabs.setCurrentIndex(0)
            _main_window_ref._show_toast(f"▶ 正在執行自選回測分析（{grp}）...")

        def _logic():
            ui_dispatcher.progress_visible.emit(True)
            mat = load_matrix_dict_analysis()
            dispo_dict = sys_db.load_state('disposition_stocks_dict', default_value={}) # 🚀 讀取處置股字典
            all_trades_multi = []
            all_events_multi = []

            target_dates = sys_db.load_state('last_fetched_dates')
            if not target_dates: # 相容舊版資料庫
                target_date = sys_db.load_state('last_fetched_date')
                target_dates = [target_date] if target_date else []
                
            if not target_dates:
                print("\n⚠️ 找不到最後採集日期標籤，請先執行更新 K 線數據！")
                ui_dispatcher.progress_visible.emit(False)
                return

            print(f"\n系統鎖定回測區間，共 {len(target_dates)} 個交易日：{', '.join(target_dates)}")

            # 回測開始前清除上次的 detail log（避免新舊資料混在一起）
            _log_dir = os.path.join(os.getcwd(), "temp")
            for _d in target_dates:
                _p = os.path.join(_log_dir, f"backtest_detail_{_d}.log")
                try:
                    if os.path.exists(_p): os.remove(_p)
                except Exception:
                    pass

            grand_total_pnl = 0
            grand_capital = 0
            grand_wins = 0
            grand_losses = 0

            # 🚀 v1.9.8.6：一次預載所有日期的 K 線資料，避免每天重複 initialize_stock_data
            _bt_cache = build_backtest_cache(dates=target_dates)

            # 💡 多日迴圈結算
            for d_idx, t_date in enumerate(target_dates):
                if len(target_dates) > 1:
                    print(f"\n" + "="*40)
                    print(f"🗓️ 正在回測第 {d_idx+1}/{len(target_dates)} 天：{t_date}")
                    print("="*40)

                day_data = _bt_cache.get(t_date, {})
                if not day_data:
                    print(f"⚠️ {t_date} 無 K 線資料，跳過。")
                    continue

                day_pnl = 0
                day_capital = 0
                day_trades = []
                day_events = []

                daily_dispo = dispo_dict.get(t_date, []) # 取得當日專屬的處置股名單

                # 多日模式關閉逐筆 verbose，避免大量 print → signal → 主執行緒訊號佇列爆炸導致 UI 凍結
                is_multi_day = len(target_dates) > 1

                if grp != "所有族群":
                    if grp not in mat: return print(f"❌ 找不到族群: {grp}")
                    stock_collection = {s: day_data[s] for s in mat[grp] if s not in daily_dispo and s in day_data}
                    tp, tc, t_hist, e_log = process_group_data(stock_collection, w, h, mat, t_date, verbose=(not is_multi_day), progress_callback=lambda p, msg: ui_dispatcher.progress_updated.emit(p, msg))
                    if t_hist: day_trades.extend(t_hist)
                    if e_log: day_events.extend(e_log)
                    if tp is not None:
                        day_pnl += tp
                        day_capital = tc
                else:
                    if len(target_dates) == 1: print("\n🌐 啟動全市場族群掃描...")
                    tp_sum, total = 0, len(mat)
                    for i, (g, s) in enumerate(mat.items()):
                        if len(target_dates) == 1: print(f"\n【開始分析族群：{g}】")
                        stock_collection = {sym: day_data[sym] for sym in s if sym not in daily_dispo and sym in day_data}
                        tp, tc, t_hist, e_log = process_group_data(stock_collection, w, h, mat, t_date, verbose=False, progress_callback=lambda p, msg: ui_dispatcher.progress_updated.emit(int((i/total)*100 + (p/total)), f"[{g}] {msg}"))
                        if tp is not None: tp_sum += tp; day_capital += tc; day_trades.extend(t_hist)
                        if e_log: day_events.extend(e_log)

                    if day_capital > 0:
                        day_pnl = tp_sum
                        day_rate = day_pnl / day_capital * 100
                        c = RED if tp_sum > 0 else (GREEN if tp_sum < 0 else "")
                        print(f"\n{c}>>> {t_date} 單日結算：總利潤 {int(tp_sum)} 元 (報酬率：{day_rate:.2f}%){RESET}")
                    else:
                        print(f"\n⚠️ {t_date} 全市場掃描完畢，單日無任何交易產生。")

                all_trades_multi.extend(day_trades)
                all_events_multi.extend(day_events)
                grand_total_pnl += day_pnl
                grand_capital += day_capital

                if day_pnl > 0: grand_wins += 1
                elif day_pnl < 0: grand_losses += 1

            # 💡 終極總結算表
            if len(target_dates) > 1:
                grand_rate = grand_total_pnl / grand_capital * 100 if grand_capital else 0.0
                c = RED if grand_total_pnl > 0 else (GREEN if grand_total_pnl < 0 else "")
                print(f"\n" + "="*45)
                print(f"多日回測總結戰報 (共 {len(target_dates)} 天)")
                print(f"📅 測試區間: {target_dates[0]} ~ {target_dates[-1]}")
                print(f"總淨利潤：{c}{int(grand_total_pnl)}{RESET} 元")
                print(f"總報酬率：{c}{grand_rate:.2f}%{RESET}")
                print(f"勝率日數：{grand_wins} 勝 {grand_losses} 敗")
                print("="*45)

            if all_trades_multi or all_events_multi:
                ui_dispatcher.plot_equity_curve.emit((all_trades_multi, all_events_multi))
            ui_dispatcher.progress_visible.emit(False)
        threading.Thread(target=_logic, daemon=True).start()

class MaximizeDialog(BaseDialog):
    def __init__(self):
        super().__init__("利潤矩陣優化模式 (矩陣併發版)", (400, 350))
        layout = QFormLayout(self)
        self.e_grp = QComboBox(); self.e_grp.setView(QListView())
        self.e_grp.addItem("所有族群"); self.e_grp.addItems(list(load_matrix_dict_analysis().keys()))
        self.e_ws, self.e_we, self.e_hs, self.e_he = QLineEdit("3"), QLineEdit("5"), QLineEdit("10"), QLineEdit("20")
        layout.addRow("族群:", self.e_grp); layout.addRow("等(起):", self.e_ws); layout.addRow("等(終):", self.e_we); layout.addRow("持(起/F):", self.e_hs); layout.addRow("持(終/F):", self.e_he)
        btn = QPushButton("🚀 啟動矩陣併發回測"); btn.setStyleSheet(f"background-color: {TV['blue']}; color: white; font-weight: bold; padding: 8px; border-radius: 5px;"); btn.clicked.connect(self.run_max); layout.addRow(btn)
        btn_tut = _make_tutorial_btn('maximize')
        layout.addRow(btn_tut)

    def run_max(self):
        grp = self.e_grp.currentText()
        try: ws, we = int(self.e_ws.text()), int(self.e_we.text())
        except Exception: return QMessageBox.critical(self, "錯誤", "等待時間必須是整數")
        hs_str, he_str = self.e_hs.text().strip().upper(), self.e_he.text().strip().upper()
        hold_options = []
        if hs_str == 'F' and he_str == 'F': hold_options = [None]
        elif hs_str != 'F' and he_str != 'F':
            try: hold_options = list(range(int(hs_str), int(he_str) + 1))
            except Exception: return QMessageBox.critical(self, "錯誤", "至少為 1 分鐘")
        else:
            try: hold_options = [int(hs_str), None] if hs_str != 'F' else [None, int(he_str)]
            except Exception: return QMessageBox.critical(self, "錯誤", "必須是整數或 F")

        self.accept()

        def _logic():
            ui_dispatcher.progress_visible.emit(True)
            ui_dispatcher.progress_updated.emit(5, "📦 正在預熱與清洗族群歷史資料...")
            
            mat = load_matrix_dict_analysis()
            dispo_dict = sys_db.load_state('disposition_stocks_dict', default_value={}) # 🚀 讀取處置股字典

            target_dates = sys_db.load_state('last_fetched_dates')
            if not target_dates: target_date = sys_db.load_state('last_fetched_date'); target_dates = [target_date] if target_date else []
            if not target_dates:
                print("\n⚠️ 找不到最後採集日期標籤，請先執行更新 K 線數據！")
                ui_dispatcher.progress_visible.emit(False)
                return
            
            # 🚀 v1.9.8.6：使用 build_backtest_cache 統一預載
            _bt_cache = build_backtest_cache(dates=target_dates)

            group_data_cache = {}
            target_groups = [grp] if grp != "所有族群" else list(mat.keys())
            for g_name in target_groups:
                date_cache = {}
                for t_date in target_dates:
                    daily_dispo = dispo_dict.get(t_date, [])
                    day_data = _bt_cache.get(t_date, {})
                    stock_collection = {s: day_data[s] for s in mat.get(g_name, []) if s not in daily_dispo and s in day_data}
                    if stock_collection:
                        date_cache[t_date] = stock_collection
                if date_cache:
                    group_data_cache[g_name] = date_cache
            
            param_matrix = [(w, h) for w in range(ws, we + 1) for h in hold_options]
            total_combinations = len(param_matrix)
            
            print(f"\n🔥 資料快取完成！啟動併發引擎，處理 {total_combinations} 組參數矩陣 (共 {len(target_dates)} 天)...\n")
            ui_dispatcher.progress_updated.emit(10, f"啟動 {total_combinations} 組矩陣運算...")
            
            results = []
            completed_tasks = 0
            progress_lock = threading.Lock()

            def run_combination(w, h_val):
                tp_sum, mat_capital = 0, 0
                for g_name, date_cache in group_data_cache.items():
                    for t_date, daily_data in date_cache.items():
                        tp, tc, _, _ = process_group_data(daily_data, w, h_val, mat, t_date, verbose=False, progress_callback=None)
                        if tp is not None:
                            tp_sum += tp
                            mat_capital += tc

                total_rate = tp_sum / mat_capital * 100 if mat_capital else 0

                nonlocal completed_tasks
                with progress_lock:
                    completed_tasks += 1
                    pct = 10 + int((completed_tasks / total_combinations) * 90)
                    ui_dispatcher.progress_updated.emit(pct, f"矩陣平行運算中... ({completed_tasks} / {total_combinations})")

                return {'等待時間': w, '持有時間': 'F' if h_val is None else h_val, '總利潤': float(tp_sum), '報酬率': float(total_rate)}

            # 3. 啟動多執行緒平行運算
            # max_workers 會自動根據你的參數數量和 CPU 核心數來調配火力
            with ThreadPoolExecutor(max_workers=min(32, total_combinations)) as executor:
                # 一口氣將 33 組任務全部發派給工人
                futures = [executor.submit(run_combination, w, h) for w, h in param_matrix]
                for future in as_completed(futures):
                    results.append(future.result())

            # 4. 排序並產出最終排行榜
            results_df = pd.DataFrame(results)
            if not results_df.empty:
                best = results_df.loc[results_df['總利潤'].idxmax()]
                c_best = RED if best['總利潤'] > 0 else (GREEN if best['總利潤'] < 0 else "")
                print(f"\n{c_best}最佳組合：等待 {best['等待時間']} 分 / 持有 {best['持有時間']} 分 / 利潤：{int(best['總利潤'])} 元{RESET}")
                
                print("\n📊 參數矩陣排行榜 (前 10 名)：")
                for rank, (idx, row) in enumerate(results_df.sort_values(by='總利潤', ascending=False).head(10).iterrows(), 1):
                    rc = RED if row['總利潤'] > 0 else (GREEN if row['總利潤'] < 0 else "")
                    print(f"   第 {rank:>2} 名: 等待 {row['等待時間']:>2} 分, 持有 {str(row['持有時間']):>2} 分 ➔ {rc}利潤: {int(row['總利潤']):>6} 元 ({row['報酬率']:.2f}%){RESET}")
                print("\n")

            ui_dispatcher.progress_visible.emit(False) 
            print("✅ 利潤矩陣優化運算完畢！")

        threading.Thread(target=_logic, daemon=True).start()

class AverageHighDialog(BaseDialog):
    def __init__(self):
        super().__init__("計算平均過高", (500, 350))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(50, 30, 50, 30)

        title = QLabel("計算平均過高間隔")
        title.setStyleSheet(f"color: {TV['text_bright']}; font-size: 18px; font-weight: 800;")
        title.setAlignment(Qt.AlignCenter)
        outer.addWidget(title)

        desc = QLabel("分析各族群中個股突破前高的平均時間間距，\n用於回測參數最佳化。")
        desc.setStyleSheet(f"color: {TV['text_dim']}; font-size: 13px;")
        desc.setAlignment(Qt.AlignCenter)
        outer.addWidget(desc)
        outer.addSpacing(20)

        # 族群選擇
        form = QFormLayout()
        self.grp_combo = QComboBox(); self.grp_combo.setView(QListView())
        self.grp_combo.addItems(list(load_matrix_dict_analysis().keys()))
        self.grp_combo.setStyleSheet(f"padding: 6px; font-size: 14px;")
        form.addRow("選擇族群：", self.grp_combo)
        outer.addLayout(form)
        outer.addSpacing(16)

        # 按鈕列
        btn_lo = QHBoxLayout(); btn_lo.setSpacing(12)
        b1 = QPushButton("分析選定族群"); b1.setFixedHeight(42)
        b1.setStyleSheet(f"QPushButton {{ background-color: {TV['blue']}; color: white; border-radius: 8px; font-weight: 700; font-size: 14px; }} QPushButton:hover {{ background-color: {TV['blue_hover']}; }}")
        b1.clicked.connect(self.run_single)
        b2 = QPushButton("分析全部族群"); b2.setFixedHeight(42)
        b2.setStyleSheet(f"QPushButton {{ background-color: {TV['green']}; color: white; border-radius: 8px; font-weight: 700; font-size: 14px; }} QPushButton:hover {{ background-color: #2bc4b4; }}")
        b2.clicked.connect(self.run_all)
        btn_tut = _make_tutorial_btn('avg_high')
        btn_lo.addWidget(b1); btn_lo.addWidget(b2); btn_lo.addWidget(btn_tut)
        outer.addLayout(btn_lo)
        outer.addStretch()

    def run_single(self):
        grp = self.grp_combo.currentText()
        if grp:
            if _main_window_ref:
                _main_window_ref.tabs.setCurrentIndex(0)
                _main_window_ref._show_toast(f"▶ 正在計算「{grp}」平均過高...")
            def _task():
                ui_dispatcher.progress_visible.emit(True)
                calculate_average_over_high(grp, progress_callback=lambda p, msg: ui_dispatcher.progress_updated.emit(p, msg))
                ui_dispatcher.progress_visible.emit(False)
            threading.Thread(target=_task, daemon=True).start()

    def run_all(self):
        if _main_window_ref:
            _main_window_ref.tabs.setCurrentIndex(0)
            _main_window_ref._show_toast("▶ 正在計算所有族群平均過高...")
        def _logic():
            ui_dispatcher.progress_visible.emit(True)
            groups, avgs, total = load_matrix_dict_analysis(), [], len(load_matrix_dict_analysis())
            for i, g in enumerate(groups.keys()):
                if avg := calculate_average_over_high(g, progress_callback=lambda p, msg: ui_dispatcher.progress_updated.emit(int((i/total)*100 + (p/total)), f"[{g}] {msg}")): avgs.append(avg)
            if avgs: print(f"\n全部族群的平均過高間隔：{sum(avgs)/len(avgs):.2f} 分鐘")
            ui_dispatcher.progress_visible.emit(False)
        threading.Thread(target=_logic, daemon=True).start()

class SettingsDialog(BaseDialog):
    def __init__(self):
        super().__init__("系統參數設定", (780, 680))
        # 捕捉本次開啟前的參數，供「取消」還原用
        _SNAP_ATTRS = [
            'capital_per_stock','transaction_fee','transaction_discount','trading_tax',
            'below_50','price_gap_50_to_100','price_gap_100_to_500','price_gap_500_to_1000','price_gap_above_1000',
            'allow_reentry','max_reentry_times','reentry_lookback_candles',
            'momentum_minutes','similarity_threshold','pull_up_pct_threshold','follow_up_pct_threshold',
            'rise_lower_bound','rise_upper_bound','volume_multiplier','min_volume_threshold',
            'wait_min_avg_vol','wait_max_single_vol','slippage_ticks','sl_cushion_pct',
            'pullback_tolerance','min_lag_pct','min_height_pct','volatility_min_range',
            'min_eligible_avg_vol','min_close_price',
            'cutoff_time_mins',
            'total_capital','max_daily_entries','max_daily_stops','risk_control_enabled',
            'tg_chat_id','sound_enabled',
        ]
        self._config_snapshot = {a: getattr(sys_config, a, None) for a in _SNAP_ATTRS}
        self.setStyleSheet(TV_DIALOG_STYLE + f"""
            QListWidget {{
                background-color: {TV['panel']}; border: none;
                border-right: 1px solid {TV['border_light']};
                font-size: 13px; color: {TV['text_dim']};
                outline: none;
            }}
            QListWidget::item {{
                padding: 12px 16px; border-radius: 0px;
            }}
            QListWidget::item:selected {{
                background-color: {TV['surface']}; color: {TV['text']};
                border-left: 3px solid {TV['blue']};
            }}
            QListWidget::item:hover:!selected {{
                background-color: {TV['border_light']};
            }}
            QScrollArea {{ background-color: transparent; border: none; }}
            QGroupBox {{ font-size: 13px; font-weight: bold; border: 1px solid {TV['border_light']}; border-radius: 6px; margin-top: 8px; padding-top: 14px; background-color: {TV['panel']}; }}
            QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding: 0 8px; color: {TV['blue']}; }}
            QCheckBox {{ color: {TV['text']}; font-size: 13px; }}
            QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {TV['border_light']}; border-radius: 3px; background-color: {TV['surface']}; }}
            QCheckBox::indicator:checked {{ background-color: {TV['blue']}; border-color: {TV['blue']}; }}
        """)

        def _make_scroll_page(*widgets):
            page = QWidget(); page.setStyleSheet(f"background-color: {TV['bg']};")
            lo = QVBoxLayout(page); lo.setContentsMargins(16, 12, 16, 12); lo.setSpacing(10)
            for w in widgets: lo.addWidget(w)
            lo.addStretch()
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(page)
            return scroll

        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        # ── Content row: left nav + right stack ──
        content_row = QWidget(); h_lo = QHBoxLayout(content_row); h_lo.setContentsMargins(0, 0, 0, 0); h_lo.setSpacing(0)

        # Left nav list
        self.nav = QListWidget()
        self.nav.setFixedWidth(192)
        nav_items = ["基本交易設定", "停損再進場", "進階策略配置", "風控模組", "通知與外觀"]
        for item in nav_items: self.nav.addItem(item)
        self.nav.setCurrentRow(0)

        # Right stacked widget
        self.stack = QStackedWidget()

        # ── PAGE 0: 基本交易設定 ──
        grp_basic = QGroupBox("基本交易與手續費")
        form_b = QFormLayout(); form_b.setLabelAlignment(Qt.AlignRight)
        self.e_cap = QLineEdit(str(sys_config.capital_per_stock)); form_b.addRow("單位投入資本額 (萬元):", self.e_cap)
        self.e_fee = QLineEdit(str(sys_config.transaction_fee)); form_b.addRow("手續費 (%):", self.e_fee)
        self.e_dis = QLineEdit(str(sys_config.transaction_discount)); form_b.addRow("手續費折數 (%):", self.e_dis)
        self.e_tax = QLineEdit(str(sys_config.trading_tax)); form_b.addRow("證交稅 (%):", self.e_tax)
        grp_basic.setLayout(form_b)

        grp_stop = QGroupBox("停損價差階距")
        form_s = QFormLayout(); form_s.setLabelAlignment(Qt.AlignRight)
        self.e_50 = QLineEdit(str(sys_config.below_50)); form_s.addRow("50 元以下:", self.e_50)
        self.e_100 = QLineEdit(str(sys_config.price_gap_50_to_100)); form_s.addRow("50 ~ 100 元:", self.e_100)
        self.e_500 = QLineEdit(str(sys_config.price_gap_100_to_500)); form_s.addRow("100 ~ 500 元:", self.e_500)
        self.e_1000 = QLineEdit(str(sys_config.price_gap_500_to_1000)); form_s.addRow("500 ~ 1000 元:", self.e_1000)
        self.e_above = QLineEdit(str(sys_config.price_gap_above_1000)); form_s.addRow("1000 元以上:", self.e_above)
        grp_stop.setLayout(form_s)
        self.stack.addWidget(_make_scroll_page(grp_basic, grp_stop))

        # ── PAGE 1: 停損再進場 ──
        grp_reentry = QGroupBox("停損再進場設定")
        form_r = QFormLayout(); form_r.setLabelAlignment(Qt.AlignRight)
        self.reentry_cb = QCheckBox("開啟停損再進場功能"); self.reentry_cb.setChecked(sys_config.allow_reentry)
        form_r.addRow(self.reentry_cb)
        self.max_reentry_spin = QSpinBox(); self.max_reentry_spin.setRange(1, 5); self.max_reentry_spin.setValue(sys_config.max_reentry_times)
        form_r.addRow("最多允許再進場次數:", self.max_reentry_spin)
        self.lookback_spin = QSpinBox(); self.lookback_spin.setRange(1, 20); self.lookback_spin.setValue(sys_config.reentry_lookback_candles)
        form_r.addRow("停損後往前檢查 K 棒數:", self.lookback_spin)
        grp_reentry.setLayout(form_r)
        self.stack.addWidget(_make_scroll_page(grp_reentry))

        # ── PAGE 2: 進階策略配置 ──
        grp_mom = QGroupBox("整體動能與關聯")
        form_m = QFormLayout(); form_m.setLabelAlignment(Qt.AlignRight)
        self.momentum_combo = QComboBox(); self.momentum_combo.setView(QListView()); self.momentum_combo.addItems(["1 分鐘", "2 分鐘", "3 分鐘", "4 分鐘", "5 分鐘"]); self.momentum_combo.setCurrentIndex(sys_config.momentum_minutes - 1)
        form_m.addRow("領漲股需連續發動幾分鐘:", self.momentum_combo)
        self.sim_thresh_spin = QDoubleSpinBox(); self.sim_thresh_spin.setRange(-1.0, 1.0); self.sim_thresh_spin.setSingleStep(0.05); self.sim_thresh_spin.setValue(sys_config.similarity_threshold)
        form_m.addRow("族群K線相似度門檻（越低越嚴格）:", self.sim_thresh_spin)
        grp_mom.setLayout(form_m)

        grp_trig = QGroupBox("觸發條件門檻")
        form_t2 = QFormLayout(); form_t2.setLabelAlignment(Qt.AlignRight)
        self.pull_up_spin = QDoubleSpinBox(); self.pull_up_spin.setRange(0.5, 10.0); self.pull_up_spin.setSingleStep(0.5); self.pull_up_spin.setValue(sys_config.pull_up_pct_threshold)
        form_t2.addRow("觸發：領漲股區間漲幅 (%):", self.pull_up_spin)
        self.follow_up_spin = QDoubleSpinBox(); self.follow_up_spin.setRange(0.01, 10.0); self.follow_up_spin.setSingleStep(0.01); self.follow_up_spin.setDecimals(2); self.follow_up_spin.setValue(sys_config.follow_up_pct_threshold)
        form_t2.addRow("觸發：跟漲股追蹤漲幅 (%):", self.follow_up_spin)
        grp_trig.setLayout(form_t2)

        grp_filt = QGroupBox("進場防護濾網 & 等待期")
        form_f = QFormLayout(); form_f.setLabelAlignment(Qt.AlignRight)
        self.rise_lower_spin = QDoubleSpinBox(); self.rise_lower_spin.setRange(-10.0, 9.0); self.rise_lower_spin.setSingleStep(0.5); self.rise_lower_spin.setValue(sys_config.rise_lower_bound)
        self.rise_upper_spin = QDoubleSpinBox(); self.rise_upper_spin.setRange(-9.0, 10.0); self.rise_upper_spin.setSingleStep(0.5); self.rise_upper_spin.setValue(sys_config.rise_upper_bound)
        form_f.addRow("進場時標的當日漲幅下限 (%):", self.rise_lower_spin)
        form_f.addRow("進場時標的當日漲幅上限 (%):", self.rise_upper_spin)
        self.vol_mult_spin = QDoubleSpinBox(); self.vol_mult_spin.setRange(0.1, 10.0); self.vol_mult_spin.setSingleStep(0.1); self.vol_mult_spin.setValue(sys_config.volume_multiplier)
        self.vol_min_spin = QSpinBox(); self.vol_min_spin.setRange(0, 10000); self.vol_min_spin.setSingleStep(10); self.vol_min_spin.setValue(sys_config.min_volume_threshold)
        form_f.addRow("觸發爆量：高於開盤均量倍數:", self.vol_mult_spin)
        form_f.addRow("觸發爆量：成交量最低門檻 (張):", self.vol_min_spin)
        self.wait_avg_vol_spin = QSpinBox(); self.wait_avg_vol_spin.setRange(0, 10000); self.wait_avg_vol_spin.setSingleStep(10); self.wait_avg_vol_spin.setValue(sys_config.wait_min_avg_vol)
        form_f.addRow("等待期：標的均量需達 (張/分):", self.wait_avg_vol_spin)
        self.wait_max_vol_spin = QSpinBox(); self.wait_max_vol_spin.setRange(0, 10000); self.wait_max_vol_spin.setSingleStep(10); self.wait_max_vol_spin.setValue(sys_config.wait_max_single_vol)
        form_f.addRow("等待期：至少一根K棒量 (張):", self.wait_max_vol_spin)
        self.slippage_spin = QSpinBox(); self.slippage_spin.setRange(0, 10); self.slippage_spin.setSingleStep(1); self.slippage_spin.setValue(sys_config.slippage_ticks)
        form_f.addRow("IOC委託向下穿價容許 (Tick):", self.slippage_spin)
        self.sl_cushion_spin = QDoubleSpinBox(); self.sl_cushion_spin.setRange(0.0, 5.0); self.sl_cushion_spin.setSingleStep(0.1); self.sl_cushion_spin.setDecimals(2); self.sl_cushion_spin.setValue(getattr(sys_config, 'sl_cushion_pct', 0.0))
        form_f.addRow("停損緩衝空間 (%):", self.sl_cushion_spin)
        self.pullback_spin = QDoubleSpinBox(); self.pullback_spin.setRange(0.0, 5.0); self.pullback_spin.setSingleStep(0.1); self.pullback_spin.setValue(sys_config.pullback_tolerance)
        form_f.addRow("突破前高容錯率 (%):", self.pullback_spin)
        # 尾盤截止觸發（cutoff_time_mins → HH:MM）
        _cutoff_mins = getattr(sys_config, 'cutoff_time_mins', 270)
        _cutoff_h, _cutoff_m = 9 + _cutoff_mins // 60, _cutoff_mins % 60
        self.cutoff_time_edit = QTimeEdit(QTime(_cutoff_h, _cutoff_m))
        self.cutoff_time_edit.setDisplayFormat("HH:mm")
        self.cutoff_time_edit.setTimeRange(QTime(9, 0), QTime(13, 30))
        form_f.addRow("尾盤截止觸發 (13:30=不截止):", self.cutoff_time_edit)
        grp_filt.setLayout(form_f)

        grp_sel = QGroupBox("選股門檻")
        form_sel = QFormLayout(); form_sel.setLabelAlignment(Qt.AlignRight)
        self.min_lag_spin = QDoubleSpinBox(); self.min_lag_spin.setRange(0, 10); self.min_lag_spin.setSingleStep(0.1); self.min_lag_spin.setValue(getattr(sys_config, 'min_lag_pct', 1.5))
        form_sel.addRow("標的漲幅需落後領漲股 (%):", self.min_lag_spin)
        self.min_height_spin = QDoubleSpinBox(); self.min_height_spin.setRange(0, 20); self.min_height_spin.setSingleStep(0.5); self.min_height_spin.setValue(getattr(sys_config, 'min_height_pct', 3.0))
        form_sel.addRow("標的今日最高漲幅需達 (%):", self.min_height_spin)
        self.volatility_range_spin = QDoubleSpinBox(); self.volatility_range_spin.setRange(0, 10); self.volatility_range_spin.setSingleStep(0.5); self.volatility_range_spin.setValue(getattr(sys_config, 'volatility_min_range', 1.0))
        form_sel.addRow("選股：當日漲跌幅活動範圍下限 (%):", self.volatility_range_spin)
        self.min_eligible_vol_spin = QSpinBox(); self.min_eligible_vol_spin.setRange(0, 500); self.min_eligible_vol_spin.setSingleStep(1); self.min_eligible_vol_spin.setValue(getattr(sys_config, 'min_eligible_avg_vol', 0))
        form_sel.addRow("選股：全日均量下限 (張/分, 0=不過濾):", self.min_eligible_vol_spin)
        self.min_close_price_spin = QSpinBox(); self.min_close_price_spin.setRange(0, 1000); self.min_close_price_spin.setSingleStep(5); self.min_close_price_spin.setValue(getattr(sys_config, 'min_close_price', 0))
        form_sel.addRow("選股：股價下限 (元, 0=不過濾):", self.min_close_price_spin)
        grp_sel.setLayout(form_sel)
        self.stack.addWidget(_make_scroll_page(grp_mom, grp_trig, grp_filt, grp_sel))

        # ── PAGE 3: 風控模組 ──
        grp_risk = QGroupBox("風控模組")
        form_rk = QFormLayout(); form_rk.setLabelAlignment(Qt.AlignRight)
        self.total_capital_spin = QDoubleSpinBox(); self.total_capital_spin.setRange(1, 99999); self.total_capital_spin.setDecimals(0); self.total_capital_spin.setValue(getattr(sys_config, 'total_capital', 249))
        form_rk.addRow("總額度 (萬元):", self.total_capital_spin)
        self.max_daily_entries_spin = QSpinBox(); self.max_daily_entries_spin.setRange(1, 50); self.max_daily_entries_spin.setValue(getattr(sys_config, 'max_daily_entries', 12))
        form_rk.addRow("每日最大進場檔數:", self.max_daily_entries_spin)
        self.max_daily_stops_spin = QSpinBox(); self.max_daily_stops_spin.setRange(1, 20); self.max_daily_stops_spin.setValue(getattr(sys_config, 'max_daily_stops', 3))
        form_rk.addRow("停損熔斷次數 (累計N筆停止進場):", self.max_daily_stops_spin)
        self.risk_control_cb = QCheckBox("啟用風控模組"); self.risk_control_cb.setChecked(getattr(sys_config, 'risk_control_enabled', True))
        form_rk.addRow(self.risk_control_cb)
        grp_risk.setLayout(form_rk)
        self.stack.addWidget(_make_scroll_page(grp_risk))

        # ── PAGE 4: Telegram授權 + 介面主題 ──
        grp_tg = QGroupBox("Telegram 終端授權")
        form_tg = QFormLayout(); form_tg.setLabelAlignment(Qt.AlignRight)
        self.e_tg_chat_id = QLineEdit(getattr(sys_config, 'tg_chat_id', ''))
        self.e_tg_chat_id.setPlaceholderText("請輸入您的專屬授權碼 (純數字)")
        form_tg.addRow("綁定授權碼:", self.e_tg_chat_id)
        grp_tg.setLayout(form_tg)

        grp_theme = QGroupBox("介面主題（重啟後生效）")
        form_th = QFormLayout(); form_th.setLabelAlignment(Qt.AlignRight)
        self.theme_combo = QComboBox(); self.theme_combo.setView(QListView())
        self.theme_combo.addItems(["🎨  經典深色（預設）", "💻  Matrix 終端（霓虹綠）"])
        _cur_theme = getattr(sys_config, 'ui_theme', 'classic')
        self.theme_combo.setCurrentIndex(1 if _cur_theme == 'matrix' else 0)
        form_th.addRow("外觀風格：", self.theme_combo)
        _theme_note = QLabel("切換主題後點「套用」，重新啟動程式即可看到新風格。")
        _theme_note.setStyleSheet(f"color: {TV['text_dim']}; font-size: 11px;")
        _theme_note.setWordWrap(True)
        form_th.addRow(_theme_note)
        grp_theme.setLayout(form_th)

        grp_sound = QGroupBox("音效提示")
        form_snd = QFormLayout(); form_snd.setLabelAlignment(Qt.AlignRight)
        self.sound_cb = QCheckBox("啟用交易音效提示（進場 / 停損 / 出場）")
        self.sound_cb.setChecked(getattr(sys_config, 'sound_enabled', True))
        form_snd.addRow(self.sound_cb)
        grp_sound.setLayout(form_snd)

        self.stack.addWidget(_make_scroll_page(grp_tg, grp_theme, grp_sound))

        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        h_lo.addWidget(self.nav); h_lo.addWidget(self.stack, 1)
        outer.addWidget(content_row, 1)

        # Save button at bottom (right-aligned)
        btn_bar = QWidget(); btn_bar.setStyleSheet(f"background-color: {TV['panel']}; border-top: 1px solid {TV['border_light']};")
        btn_bar_lo = QHBoxLayout(btn_bar); btn_bar_lo.setContentsMargins(16, 10, 16, 10); btn_bar_lo.setSpacing(10)
        btn_export = QPushButton("匯出設定"); btn_export.setFixedHeight(38); btn_export.clicked.connect(self._export_settings)
        btn_import = QPushButton("匯入設定"); btn_import.setFixedHeight(38); btn_import.clicked.connect(self._import_settings)
        btn_cancel = QPushButton("取消"); btn_cancel.setFixedHeight(38); btn_cancel.clicked.connect(self._cancel)
        btn_apply  = QPushButton("套用");  btn_apply.setFixedHeight(38);  btn_apply.clicked.connect(self.save)
        btn_bar_lo.addWidget(btn_export); btn_bar_lo.addWidget(btn_import)
        btn_bar_lo.addStretch(); btn_bar_lo.addWidget(btn_apply); btn_bar_lo.addWidget(btn_cancel)
        outer.addWidget(btn_bar)

    def _cancel(self):
        for attr, val in self._config_snapshot.items():
            if val is not None:
                setattr(sys_config, attr, val)
        self.reject()

    def _export_settings(self):
        """匯出目前所有設定值為 JSON 檔案"""
        _EXPORT_ATTRS = [
            'capital_per_stock','transaction_fee','transaction_discount','trading_tax',
            'below_50','price_gap_50_to_100','price_gap_100_to_500','price_gap_500_to_1000','price_gap_above_1000',
            'allow_reentry','max_reentry_times','reentry_lookback_candles',
            'momentum_minutes','similarity_threshold','pull_up_pct_threshold','follow_up_pct_threshold',
            'rise_lower_bound','rise_upper_bound','volume_multiplier','min_volume_threshold',
            'wait_min_avg_vol','wait_max_single_vol','slippage_ticks','sl_cushion_pct',
            'pullback_tolerance','min_lag_pct','min_height_pct','volatility_min_range',
            'min_eligible_avg_vol','min_close_price',
            'cutoff_time_mins',
            'total_capital','max_daily_entries','max_daily_stops','risk_control_enabled',
            'sound_enabled',
        ]
        data = {}
        for attr in _EXPORT_ATTRS:
            val = getattr(sys_config, attr, None)
            if val is not None:
                data[attr] = val
        default_name = f"remora_settings_{datetime.now().strftime('%Y%m%d')}.json"
        path, _ = QFileDialog.getSaveFileName(self, "匯出設定檔", default_name, "JSON 檔案 (*.json)")
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "匯出成功", f"設定已匯出至：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "匯出失敗", f"寫入檔案時發生錯誤：{e}")

    def _import_settings(self):
        """匯入 JSON 設定檔並填入 UI 欄位（不自動套用，需點「套用」確認）"""
        path, _ = QFileDialog.getOpenFileName(self, "匯入設定檔", "", "JSON 檔案 (*.json)")
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "匯入失敗", f"讀取檔案時發生錯誤：{e}")
            return

        # 將 JSON 數值填入對應的 UI 控件
        _field_map = {
            'capital_per_stock': ('line', self.e_cap),
            'transaction_fee': ('line', self.e_fee),
            'transaction_discount': ('line', self.e_dis),
            'trading_tax': ('line', self.e_tax),
            'below_50': ('line', self.e_50),
            'price_gap_50_to_100': ('line', self.e_100),
            'price_gap_100_to_500': ('line', self.e_500),
            'price_gap_500_to_1000': ('line', self.e_1000),
            'price_gap_above_1000': ('line', self.e_above),
            'allow_reentry': ('check', self.reentry_cb),
            'max_reentry_times': ('spin', self.max_reentry_spin),
            'reentry_lookback_candles': ('spin', self.lookback_spin),
            'momentum_minutes': ('combo_idx', self.momentum_combo, -1),
            'similarity_threshold': ('dspin', self.sim_thresh_spin),
            'pull_up_pct_threshold': ('dspin', self.pull_up_spin),
            'follow_up_pct_threshold': ('dspin', self.follow_up_spin),
            'rise_lower_bound': ('dspin', self.rise_lower_spin),
            'rise_upper_bound': ('dspin', self.rise_upper_spin),
            'volume_multiplier': ('dspin', self.vol_mult_spin),
            'min_volume_threshold': ('spin', self.vol_min_spin),
            'wait_min_avg_vol': ('spin', self.wait_avg_vol_spin),
            'wait_max_single_vol': ('spin', self.wait_max_vol_spin),
            'slippage_ticks': ('spin', self.slippage_spin),
            'sl_cushion_pct': ('dspin', self.sl_cushion_spin),
            'pullback_tolerance': ('dspin', self.pullback_spin),
            'min_lag_pct': ('dspin', self.min_lag_spin),
            'min_height_pct': ('dspin', self.min_height_spin),
            'volatility_min_range': ('dspin', self.volatility_range_spin),
            'min_eligible_avg_vol': ('spin', self.min_eligible_vol_spin),
            'min_close_price': ('spin', self.min_close_price_spin),
            'cutoff_time_mins': ('time', self.cutoff_time_edit),
            'total_capital': ('dspin', self.total_capital_spin),
            'max_daily_entries': ('spin', self.max_daily_entries_spin),
            'max_daily_stops': ('spin', self.max_daily_stops_spin),
            'risk_control_enabled': ('check', self.risk_control_cb),
            'sound_enabled': ('check', self.sound_cb),
        }
        loaded_count = 0
        for key, val in data.items():
            if key not in _field_map:
                continue
            entry = _field_map[key]
            try:
                kind = entry[0]
                widget = entry[1]
                if kind == 'line':
                    widget.setText(str(val))
                elif kind == 'spin':
                    widget.setValue(int(val))
                elif kind == 'dspin':
                    widget.setValue(float(val))
                elif kind == 'check':
                    widget.setChecked(bool(val))
                elif kind == 'combo_idx':
                    offset = entry[2]
                    widget.setCurrentIndex(int(val) + offset)
                elif kind == 'time':
                    mins = int(val)
                    h, m = 9 + mins // 60, mins % 60
                    widget.setTime(QTime(h, m))
                loaded_count += 1
            except Exception:
                pass

        QMessageBox.information(self, "匯入成功", f"已載入 {loaded_count} 項設定。\n請確認後點擊「套用」以生效。")

    def save(self):
        try:
            sys_config.capital_per_stock, sys_config.transaction_fee, sys_config.transaction_discount, sys_config.trading_tax = int(self.e_cap.text()), float(self.e_fee.text()), float(self.e_dis.text()), float(self.e_tax.text())
            sys_config.below_50, sys_config.price_gap_50_to_100, sys_config.price_gap_100_to_500, sys_config.price_gap_500_to_1000, sys_config.price_gap_above_1000 = float(self.e_50.text()), float(self.e_100.text()), float(self.e_500.text()), float(self.e_1000.text()), float(self.e_above.text())
            
            # 儲存停損再進場設定
            sys_config.allow_reentry = self.reentry_cb.isChecked()
            sys_config.max_reentry_times = self.max_reentry_spin.value()
            sys_config.reentry_lookback_candles = self.lookback_spin.value()
            
            sys_config.momentum_minutes = self.momentum_combo.currentIndex() + 1
            sys_config.similarity_threshold = self.sim_thresh_spin.value()
            sys_config.pull_up_pct_threshold = self.pull_up_spin.value()
            sys_config.follow_up_pct_threshold = self.follow_up_spin.value()
            sys_config.rise_lower_bound = self.rise_lower_spin.value()
            sys_config.rise_upper_bound = self.rise_upper_spin.value()
            sys_config.volume_multiplier = self.vol_mult_spin.value()
            sys_config.min_volume_threshold = self.vol_min_spin.value()
            sys_config.wait_min_avg_vol = self.wait_avg_vol_spin.value()
            sys_config.wait_max_single_vol = self.wait_max_vol_spin.value()
            sys_config.slippage_ticks = self.slippage_spin.value()
            sys_config.sl_cushion_pct = self.sl_cushion_spin.value()
            sys_config.pullback_tolerance = self.pullback_spin.value()
            _ct = self.cutoff_time_edit.time()
            sys_config.cutoff_time_mins = (_ct.hour() - 9) * 60 + _ct.minute()
            sys_config.min_lag_pct = self.min_lag_spin.value()
            sys_config.min_height_pct = self.min_height_spin.value()
            sys_config.volatility_min_range = self.volatility_range_spin.value()
            sys_config.min_eligible_avg_vol = self.min_eligible_vol_spin.value()
            sys_config.min_close_price = self.min_close_price_spin.value()

            # 🛡️ 風控模組設定
            sys_config.total_capital = self.total_capital_spin.value()
            sys_config.max_daily_entries = self.max_daily_entries_spin.value()
            sys_config.max_daily_stops = self.max_daily_stops_spin.value()
            sys_config.risk_control_enabled = self.risk_control_cb.isChecked()

            sys_config.tg_chat_id = self.e_tg_chat_id.text().strip()
            sys_config.sound_enabled = self.sound_cb.isChecked()

            # 🎨 主題設定 → 寫入 config.ini
            new_theme = 'matrix' if self.theme_combo.currentIndex() == 1 else 'classic'
            if new_theme != getattr(sys_config, 'ui_theme', 'classic'):
                sys_config.ui_theme = new_theme
                try:
                    _c = ConfigParser(); _c.read('config.ini', encoding='utf-8-sig')
                    if not _c.has_section('ui'): _c.add_section('ui')
                    _c.set('ui', 'theme', new_theme)
                    with open('config.ini', 'w', encoding='utf-8') as _f: _c.write(_f)
                    QMessageBox.information(self, "主題已切換", f"已切換至「{'Matrix 終端' if new_theme=='matrix' else '經典深色'}」主題。\n請重新啟動程式以生效。")
                except Exception as _te:
                    QMessageBox.warning(self, "警告", f"主題設定寫入失敗：{_te}")

            save_settings()
            tg_bot.start()
            print(f"✅ 授權儲存！目前綁定客戶: {'已綁定' if sys_config.tg_chat_id else '未綁定'}")
            QMessageBox.information(self, "設定已套用", "所有參數已成功儲存並套用。")
            self.accept()
        except Exception: QMessageBox.critical(self, "錯誤", "數字格式不正確")

# ==========================================
# v1.9.8.6 後端通訊橋樑：連結 JS 與 Python
# ==========================================
class TerminalBridge(QObject):
    updateChartSignal = pyqtSignal(str) 
    
    def __init__(self, parent_window):
        super().__init__()
        self.window = parent_window

    @pyqtSlot(str, str, str)
    def trigger_recalc(self, wait_t, hold_t, target_date):
        def _task():
            try:
                new_json = run_backtest_engine_196(wait_t, hold_t, target_date)
                self.updateChartSignal.emit(new_json)
            except Exception as e:
                logger.error(f"背景回測發生錯誤: {e}", exc_info=True)
        threading.Thread(target=_task, daemon=True).start()

    @pyqtSlot()
    def show_cashflow_view(self):
        self.window.show_cashflow()

class AdvancedTerminalWindow(QMainWindow):
    def __init__(self, payload=None, mode="backtest", date_str="", wait="3", hold="F", temp_file=None):
        super().__init__()
        self.setWindowTitle(f"量化分析終端 v1.9.8.6 - {date_str}")
        self.resize(1600, 900)

        self._mode = mode
        self._date_str = date_str

        self.browser = QWebEngineView()
        self.setCentralWidget(self.browser)

        self.bridge = TerminalBridge(self)
        self.bridge.updateChartSignal.connect(self.update_web_ui)
        self.channel = QWebChannel()
        self.channel.registerObject('pyobj', self.bridge)
        self.browser.page().setWebChannel(self.channel)

        # 綁定載入完成事件，根除 JS 未定義錯誤
        self.browser.loadFinished.connect(self._on_load_finished)

        # 執行載入程序
        if temp_file:
            # 快速路徑：背景執行緒已完成 HTML 建構，主執行緒只需觸發非同步載入
            self.browser.load(QUrl.fromLocalFile(temp_file))
        elif payload is not None:
            self.load_dashboard(payload, mode, wait, hold, date_str)

    def load_dashboard(self, payload, mode, wait, hold, date_str):
        self._mode = mode
        self._date_str = date_str
        html_content = AdvancedTerminalWindow.build_html_template(payload, mode, wait, hold, date_str)

        # 寫入實體暫存檔再讀取，徹底避開 QtWebEngine 字串記憶體崩潰
        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, "dashboard_196.html")

        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.browser.load(QUrl.fromLocalFile(temp_file))

    def load_from_file(self, temp_file, mode, date_str):
        """直接載入已建好的 HTML 檔（背景執行緒已完成 JSON 序列化與寫磁碟），主執行緒僅觸發 browser.load()"""
        self._mode = mode
        self._date_str = date_str
        self.browser.load(QUrl.fromLocalFile(temp_file))

    def _on_load_finished(self, ok):
        if ok and self._mode == "watchlist":
            # 保證網頁「完全載入完畢」後，才啟動背景運算，完美避開競速危害
            self.bridge.trigger_recalc("3", "F", self._date_str)

    def load_cashflow(self, html: str):
        self._cashflow_html = html

    def show_cashflow(self):
        h = getattr(self, '_cashflow_html', None)
        if h:
            self.browser.setHtml(h, QUrl("about:blank"))
        else:
            self.browser.setHtml("<html><body style='background:#1E1E1E;color:white;display:flex;align-items:center;justify-content:center;height:100vh;font-family:Microsoft JhengHei'><h2>尚無資金流動數據，請先執行回測</h2></body></html>", QUrl("about:blank"))

    @pyqtSlot(str)
    def update_web_ui(self, lightweight_json_str):
        # 安全檢查機制：確定函數存在才呼叫
        self.browser.page().runJavaScript(f"if(typeof window.updateFromPython === 'function') {{ window.updateFromPython({lightweight_json_str}); }}")

    @staticmethod
    def build_html_template(payload, mode, wait, hold, date_str):
        json_data = json.dumps(payload, ensure_ascii=False).replace("</script>", "<\\/script>")

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
            <script>if(typeof LightweightCharts==='undefined')document.write('<scr'+'ipt src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"><\\/scr'+'ipt>');</script>
            <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
            <style>
                body {{ margin: 0; display: flex; height: 100vh; background-color: #131722; color: #D1D4DC; overflow: hidden; font-family: -apple-system, 'Segoe UI', sans-serif; }}
                /* 事件標籤 overlay */
                .ev-label {{ position: absolute; z-index: 20; pointer-events: none; font-size: 11px; font-weight: 700; padding: 2px 6px; border-radius: 3px; white-space: nowrap; transform: translateX(-50%); text-shadow: none; line-height: 1.3; }}
                .trade-label {{ position: absolute; z-index: 21; pointer-events: none; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; white-space: nowrap; transform: translateX(-50%); line-height: 1.3; letter-spacing: 0.5px; border: 1px solid rgba(255,255,255,0.25); box-shadow: 0 1px 4px rgba(0,0,0,0.4); }}
                .trade-label.entry {{ background: linear-gradient(135deg, #2962FF, #1E88E5); color: #fff; }}
                .trade-label.exit {{ background: linear-gradient(135deg, #F57C00, #FFB300); color: #fff; }}
                /* 圖表主區 */
                #chart-area {{ flex: 1; position: relative; display: flex; flex-direction: column; }}
                #chart-container {{ flex: 1; position: relative; cursor: crosshair; }}
                /* 頂部工具列 */
                #top-toolbar {{ height: 38px; background: #1E222D; border-bottom: 1px solid #2A2E39; display: flex; align-items: center; padding: 0 12px; gap: 8px; flex-shrink: 0; }}
                .tb-btn {{ background: transparent; color: #B2B5BE; border: none; padding: 5px 10px; cursor: pointer; border-radius: 4px; font-weight: 600; font-size: 12px; transition: all 0.15s; }}
                .tb-btn:hover {{ background: #2A2E39; color: white; }}
                .tb-btn.active {{ color: white; background: #2962FF; }}
                .tb-sep {{ width: 1px; height: 20px; background: #363C4E; flex-shrink: 0; }}
                #chart-date-selector {{ background: #131722; color: #F7A600; border: 1px solid #434651; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; cursor: pointer; outline: none; }}
                /* 右側面板 */
                #sidebar {{ width: 330px; background: #131722; border-left: 1px solid #363C4E; display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0; }}
                #sidebar-tabs {{ display: flex; border-bottom: 1px solid #2A2E39; flex-shrink: 0; }}
                .stab {{ flex: 1; padding: 10px 0; text-align: center; cursor: pointer; font-size: 12px; font-weight: 700; color: #787B86; background: transparent; border: none; border-bottom: 2px solid transparent; transition: all 0.15s; }}
                .stab:hover {{ color: #D1D4DC; }}
                .stab.active {{ color: #2962FF; border-bottom-color: #2962FF; }}
                .stab-panel {{ display: none; flex: 1; overflow-y: auto; }}
                .stab-panel.active {{ display: flex; flex-direction: column; }}
                #backtest-panel {{ background: #1A1E29; padding: 12px; border-radius: 6px; border: 1px solid #2A2E39; margin: 10px; }}
                .param-row {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; font-size: 13px; color: #787B86; }}
                .param-input {{ width: 70px; background: #131722; color: white; border: 1px solid #363C4E; border-radius: 3px; padding: 5px; text-align: center; font-weight: bold; }}
                #btn-convert {{ width: 100%; background: #2962FF; color: white; border: none; padding: 10px; border-radius: 4px; font-weight: bold; cursor: pointer; }}
                #btn-convert:hover {{ background: #1E53E5; }}
                /* Watchlist */
                #watchlist-content {{ flex: 1; overflow-y: auto; }}
                .group-container {{ margin-bottom: 1px; }}
                .group-title {{ padding: 10px 16px; cursor: pointer; font-size: 11px; font-weight: 700; color: #787B86; text-transform: uppercase; display: flex; justify-content: space-between; letter-spacing: 0.5px; }}
                .group-title:hover {{ background: #1E222D; }}
                .group-title.active {{ color: #FFFFFF; }}
                .stock-list {{ display: none; background: #131722; }}
                .stock-item {{ display: flex; align-items: center; padding: 8px 16px; cursor: pointer; border-bottom: 1px solid #1E222D; font-size: 13px; }}
                .stock-item:hover {{ background: #2A2E39; }}
                .stock-item .symbol {{ font-weight: 600; color: #787B86; margin-right: 10px; min-width: 45px; }}
                .stock-item.traded .symbol {{ color: #FFFFFF; }}
                .pnl-tag {{ font-size: 11px; font-weight: bold; margin-left: auto; padding: 2px 6px; border-radius: 3px; }}
                .pnl-win {{ color: #FFFFFF; background-color: rgba(239,83,80,0.25); }} .pnl-loss {{ color: #FFFFFF; background-color: rgba(38,166,154,0.25); }}
                /* 右鍵選單 */
                #ctx-menu {{ display: none; position: fixed; z-index: 9999; background: #1E222D; border: 1px solid #363C4E; border-radius: 6px; padding: 4px 0; min-width: 200px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }}
                .ctx-item {{ padding: 8px 16px; cursor: pointer; font-size: 13px; color: #D1D4DC; display: flex; justify-content: space-between; }}
                .ctx-item:hover {{ background: #2A2E39; }}
                .ctx-sep {{ height: 1px; background: #2A2E39; margin: 4px 0; }}
                .ctx-dim {{ color: #787B86; font-size: 11px; }}
                /* 滾動條 */
                ::-webkit-scrollbar {{ width: 6px; }}
                ::-webkit-scrollbar-thumb {{ background: #363C4E; border-radius: 3px; }}
                #loader {{ display:none; position:absolute; top:0; left:0; width:100%; height:100%; background:rgba(19,23,34,0.9); z-index:999; justify-content:center; align-items:center; color:white; font-size:20px; font-weight:bold; }}
            </style>
        </head>
        <body>
            <div id="chart-area">
                <!-- 頂部工具列 -->
                <div id="top-toolbar">
                    <button class="tb-btn" id="btn-crosshair" title="十字線" onclick="if(window.lwChart)window.lwChart.applyOptions({{crosshair:{{mode:0}}}});">+</button>
                    <button class="tb-btn" id="btn-magnet" title="磁吸" onclick="if(window.lwChart)window.lwChart.applyOptions({{crosshair:{{mode:1}}}});">⊙</button>
                    <div class="tb-sep"></div>
                    <button id="btn-kline" class="tb-btn active">K棒</button>
                    <button id="btn-line" class="tb-btn">折線</button>
                    <div class="tb-sep"></div>
                    <select id="chart-date-selector" style="display:none;"></select>
                    <div class="tb-sep"></div>
                    <span style="color:#787B86; font-size:11px;">滾輪=縮放 | 拖拽=平移</span>
                    <div style="flex:1;"></div>
                    <button class="tb-btn" onclick="if(window.lwChart)window.lwChart.timeScale().fitContent();">自動縮放</button>
                    <button class="tb-btn" onclick="if(window.lwChart)window.lwChart.timeScale().resetTimeScale();">重置</button>
                </div>
                <div id="chart-container"></div>
                <div id="loader">🔄 正在載入歷史分析數據...</div>
            </div>

            <!-- 右鍵選單 -->
            <div id="ctx-menu">
                <div class="ctx-item" onclick="if(window.lwChart)window.lwChart.timeScale().fitContent();document.getElementById('ctx-menu').style.display='none';">自動縮放 <span class="ctx-dim">Ctrl+Home</span></div>
                <div class="ctx-sep"></div>
                <div class="ctx-item" onclick="if(window.lwChart)window.lwChart.timeScale().scrollToRealTime();document.getElementById('ctx-menu').style.display='none';">跳到最新K棒 <span class="ctx-dim">End</span></div>
                <div class="ctx-item" onclick="if(window.lwChart)window.lwChart.timeScale().resetTimeScale();document.getElementById('ctx-menu').style.display='none';">重置時間軸</div>
                <div class="ctx-sep"></div>
                <div class="ctx-item" onclick="if(window.lwChart)window.lwChart.applyOptions({{crosshair:{{mode:0}}}});document.getElementById('ctx-menu').style.display='none';">十字線模式</div>
                <div class="ctx-item" onclick="if(window.lwChart)window.lwChart.applyOptions({{crosshair:{{mode:1}}}});document.getElementById('ctx-menu').style.display='none';">磁吸模式</div>
                <div class="ctx-sep"></div>
                <div class="ctx-item ctx-dim" style="cursor:default;">Powered by Remora</div>
            </div>

            <div id="sidebar">
                <!-- 頁簽 -->
                <div id="sidebar-tabs">
                    <button class="stab active" data-panel="panel-watchlist">自選清單</button>
                    <button class="stab" data-panel="panel-backtest">回測覆盤</button>
                    <button class="stab" data-panel="panel-cashflow">資金流動</button>
                </div>
                <!-- 自選清單面板 -->
                <div id="panel-watchlist" class="stab-panel active">
                    <div id="watchlist-content"></div>
                </div>
                <!-- 回測覆盤面板 -->
                <div id="panel-backtest" class="stab-panel">
                    <div id="backtest-panel" style="display:block;">
                        <div class="param-row"><span>等待(分)</span><input type="text" id="param-wait" class="param-input" value="{wait}"></div>
                        <div class="param-row"><span>持有(分)</span><input type="text" id="param-hold" class="param-input" value="{hold}"></div>
                        <button id="btn-convert">重新分析</button>
                    </div>
                    <div id="watchlist-content-bt"></div>
                </div>
                <!-- 資金流動面板 -->
                <div id="panel-cashflow" class="stab-panel">
                    <div style="padding:20px; color:#787B86; text-align:center;">點擊上方「資金流動」查看走勢圖</div>
                </div>
            </div>
            <script>
                window.globalData = {json_data};
                window.checkedStocks = new Set();
                window.chartType = 'kline';
                window.currentMode = '{mode}';
                window.isInitialized = false;
                window.availableDates = [];   
                window.activeViewDate = '{date_str}'; 

                new QWebChannel(qt.webChannelTransport, function(channel) {{ window.pyBackend = channel.objects.pyobj; }});

                function robustFormat(t) {{ 
                    if (!t) return "";
                    let s = String(t).trim();
                    if (s.includes(':')) {{
                        let parts = s.split(':');
                        let h = parts[0].padStart(2, '0');
                        let m = parts[1].padStart(2, '0');
                        return h + ":" + m;
                    }}
                    return s.length >= 5 ? s.substring(0, 5) : s;
                }}

                // 🚀 初始化可用日期選單
                window.initDateSelector = function() {{
                    let dateSet = new Set();
                    const {{ allTrades, allEvents, chartData }} = window.globalData;
                    
                    if (allTrades) allTrades.forEach(t => {{ if(t.date) dateSet.add(t.date); }});
                    if (allEvents) allEvents.forEach(e => {{ if(e.date) dateSet.add(e.date); }});
                    if (chartData) {{
                        for (let grp in chartData) {{
                            for (let sym in chartData[grp]) {{
                                if (chartData[grp][sym] && chartData[grp][sym].date) {{
                                    chartData[grp][sym].date.forEach(d => dateSet.add(d));
                                }}
                            }}
                        }}
                    }}
                    
                    window.availableDates = Array.from(dateSet).sort();
                    
                    const selector = document.getElementById('chart-date-selector');

                    if (window.availableDates.length > 0 && selector) {{
                        selector.style.display = 'inline-block';
                        selector.innerHTML = '';

                        window.availableDates.forEach(d => {{
                            let opt = document.createElement('option');
                            opt.value = d;
                            opt.textContent = d;
                            selector.appendChild(opt);
                        }});

                        if (!window.availableDates.includes(window.activeViewDate)) {{
                            window.activeViewDate = window.availableDates[window.availableDates.length - 1];
                        }}
                        selector.value = window.activeViewDate;

                        selector.onchange = (e) => {{
                            window.activeViewDate = e.target.value;
                            window.renderWatchlist();
                            window.updateChart();
                        }};
                    }} else if (selector) {{
                        selector.style.display = 'none';
                    }}
                }};

                window.updateFromPython = function(newData) {{
                    document.getElementById('loader').style.display = 'none';
                    if (newData.allTrades) window.globalData.allTrades = newData.allTrades;
                    if (newData.allEvents) window.globalData.allEvents = newData.allEvents;
                    if (newData.pnlSummary) window.globalData.pnlSummary = newData.pnlSummary;
                    
                    window.initDateSelector();
                    window.renderWatchlist();
                    window.updateChart();
                }};

                window.renderWatchlist = function() {{
                    const div = document.getElementById('watchlist-content');
                    const divBt = document.getElementById('watchlist-content-bt');
                    div.innerHTML = '';
                    if (divBt) divBt.innerHTML = '';
                    const {{ chartData, allTrades, allEvents }} = window.globalData;

                    let targetDate = window.activeViewDate || '{date_str}';

                    // 🚀 1. 動態計算「這一天」的單日股票損益
                    let dailyPnl = {{}};
                    if (allTrades) {{
                        allTrades.forEach(t => {{
                            let d = t.date || '{date_str}'; 
                            if (d === targetDate) {{
                                dailyPnl[t.symbol] = (dailyPnl[t.symbol] || 0) + (t.profit || 0);
                            }}
                        }});
                    }}

                    // 🚀 2. 收集「這一天」有發生事件的股票
                    let eventSymbols = new Set();
                    if (allEvents) {{
                        allEvents.forEach(e => {{
                            let d = e.date || '{date_str}';
                            if (d === targetDate) {{
                                eventSymbols.add(String(e.symbol));
                            }}
                        }});
                    }}

                    if(!window.isInitialized && window.globalData.targetCode) {{
                        window.checkedStocks.add(String(window.globalData.targetCode));
                        window.isInitialized = true;
                    }}

                    let groupNames = Object.keys(chartData);
                    let groupScores = {{}};
                    groupNames.forEach(grp => {{
                        let score = 0;
                        for(let c in chartData[grp]) {{
                            if (dailyPnl[c] !== undefined) {{
                                score = 2;
                                break; 
                            }} else if (score < 2 && eventSymbols.has(c)) {{
                                score = 1;
                            }}
                        }}
                        groupScores[grp] = score;
                    }});

                    if (window.currentMode === 'backtest') {{
                        groupNames.sort((a, b) => groupScores[b] - groupScores[a]);
                    }}

                    groupNames.forEach(grp => {{
                        const gDiv = document.createElement('div');
                        gDiv.className = 'group-container';
                        
                        let gScore = groupScores[grp];
                        let hasPnl = (gScore === 2);
                        
                        let gPnl = 0;
                        if (hasPnl) {{
                            for(let c in chartData[grp]) if(dailyPnl[c] !== undefined) gPnl += dailyPnl[c];
                        }}
                        
                        let pnlHtml = (window.currentMode === 'backtest' && hasPnl) ? `<span class="pnl-tag ${{gPnl>=0?'pnl-win':'pnl-loss'}}">(${{Math.round(gPnl)}})</span>` : '';
                        
                        const title = document.createElement('div');
                        let isUselessGroup = (window.currentMode === 'backtest' && gScore === 0);
                        
                        title.className = 'group-title' + (isUselessGroup ? ' disabled' : '');
                        if (window.currentMode === 'backtest' && gScore >= 1) {{ title.classList.add('active'); }}
                        title.innerHTML = `<span>${{grp}} ${{pnlHtml}}</span> <span>▼</span>`;
                        
                        const list = document.createElement('div');
                        list.style.display = 'none';
                        let hasChecked = false;
                        
                        for(let code in chartData[grp]) {{
                            const item = document.createElement('label');
                            let pTag = (window.currentMode === 'backtest' && dailyPnl[code] !== undefined) ? `<span class="pnl-tag ${{dailyPnl[code]>=0?'pnl-win':'pnl-loss'}}">(${{Math.round(dailyPnl[code])}})</span>` : '';
                            
                            let isTraded = (window.currentMode === 'backtest' && dailyPnl[code] !== undefined);
                            item.className = 'stock-item' + (isTraded ? ' traded' : '');
                            let disableAttr = (window.currentMode === 'backtest' && isUselessGroup) ? 'disabled' : '';
                            
                            item.innerHTML = `<input type="checkbox" value="${{code}}" ${{window.checkedStocks.has(String(code))?'checked':''}} ${{disableAttr}} style="margin-right:12px;"><span class="symbol">${{code}}</span><span style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${{chartData[grp][code].name}}</span>${{pTag}}`;
                            
                            item.querySelector('input').onchange = (e) => {{ if(e.target.checked) window.checkedStocks.add(String(code)); else window.checkedStocks.delete(String(code)); window.updateChart(); }};
                            list.appendChild(item);
                            if(window.checkedStocks.has(String(code))) hasChecked = true;
                        }}
                        
                        title.onclick = () => {{
                            if (isUselessGroup) return; 
                            list.style.display = (list.style.display === 'block' ? 'none' : 'block');
                        }};
                        
                        if(hasChecked) list.style.display = 'block';
                        gDiv.appendChild(title); gDiv.appendChild(list); div.appendChild(gDiv);
                    }});
                    // 同步到回測覆盤頁簽
                    if (divBt) divBt.innerHTML = div.innerHTML;
                    // 重新綁定回測頁簽的 checkbox 事件
                    if (divBt) {{
                        divBt.querySelectorAll('input[type=checkbox]').forEach(cb => {{
                            cb.onchange = (e) => {{ if(e.target.checked) window.checkedStocks.add(String(e.target.value)); else window.checkedStocks.delete(String(e.target.value)); window.updateChart(); }};
                        }});
                        divBt.querySelectorAll('.group-title').forEach(t => {{
                            t.onclick = () => {{ let list = t.nextElementSibling; if(list) list.style.display = (list.style.display === 'block' ? 'none' : 'block'); }};
                        }});
                    }}
                }};

                // ═══════════════════════════════════════════════════════
                // Lightweight Charts 渲染引擎
                // ═══════════════════════════════════════════════════════
                window.lwChart = null;
                window.lwCandleSeries = null;
                window.lwLineSeries = null;
                window.lwVolumeSeries = null;
                window.lwCompareLines = [];

                // 時區處理：資料是台灣時間(UTC+8)，lightweight-charts 用 UTC 顯示
                // 解法：將台灣時間「偽裝」成 UTC，讓 x 軸直接顯示台灣時間
                function timeToUnix(dateStr, timeStr) {{
                    let t = robustFormat(timeStr);
                    let [h, m] = t.split(':').map(Number);
                    // 直接用 UTC 建構（不做任何時區轉換），這樣 x 軸顯示的就是台灣時間
                    return Date.UTC(
                        parseInt(dateStr.substring(0,4)),
                        parseInt(dateStr.substring(5,7)) - 1,
                        parseInt(dateStr.substring(8,10)),
                        h, m, 0
                    ) / 1000;
                }}

                function getFilteredData(code, chartData, targetDate) {{
                    let s = null;
                    for (let g in chartData) {{ if (chartData[g][code]) {{ s = chartData[g][code]; break; }} }}
                    if (!s) return null;
                    let result = {{ ohlcv: [], times: [], pct: [] }};
                    if (s.date && s.date.length === s.time.length) {{
                        for (let i = 0; i < s.time.length; i++) {{
                            if (s.date[i] === targetDate) {{
                                let ts = timeToUnix(targetDate, s.time[i]);
                                result.ohlcv.push({{ time: ts, open: s.open[i], high: s.high[i], low: s.low[i], close: s.close[i], vol: s.volume[i] }});
                                result.times.push(robustFormat(s.time[i]));
                                if (s.pct_change) result.pct.push({{ time: ts, value: s.pct_change[i] }});
                            }}
                        }}
                    }} else {{
                        for (let i = 0; i < s.time.length; i++) {{
                            let ts = timeToUnix(targetDate, s.time[i]);
                            result.ohlcv.push({{ time: ts, open: s.open[i], high: s.high[i], low: s.low[i], close: s.close[i], vol: s.volume[i] }});
                            result.times.push(robustFormat(s.time[i]));
                            if (s.pct_change) result.pct.push({{ time: ts, value: s.pct_change[i] }});
                        }}
                    }}
                    result.name = s.name || code;
                    return result;
                }}

                function ensureChart() {{
                    let container = document.getElementById('chart-container');
                    if (window.lwChart) {{
                        // 清除舊 series
                        try {{
                            if (window.lwCandleSeries) {{ window.lwChart.removeSeries(window.lwCandleSeries); window.lwCandleSeries = null; }}
                            if (window.lwLineSeries) {{ window.lwChart.removeSeries(window.lwLineSeries); window.lwLineSeries = null; }}
                            if (window.lwVolumeSeries) {{ window.lwChart.removeSeries(window.lwVolumeSeries); window.lwVolumeSeries = null; }}
                            window.lwCompareLines.forEach(s => {{ try {{ window.lwChart.removeSeries(s); }} catch(e) {{}} }});
                            window.lwCompareLines = [];
                        }} catch(e) {{}}
                        return window.lwChart;
                    }}
                    window.lwChart = LightweightCharts.createChart(container, {{
                        width: container.clientWidth,
                        height: container.clientHeight,
                        layout: {{ background: {{ color: '#131722' }}, textColor: '#D1D4DC', fontFamily: '-apple-system, sans-serif' }},
                        grid: {{ vertLines: {{ color: '#2A2E39' }}, horzLines: {{ color: '#2A2E39' }} }},
                        crosshair: {{ mode: 0, vertLine: {{ labelVisible: true }}, horzLine: {{ labelVisible: true }} }},
                        rightPriceScale: {{ borderColor: '#2A2E39' }},
                        timeScale: {{ borderColor: '#2A2E39', timeVisible: true, secondsVisible: false, tickMarkFormatter: (time) => {{ let d = new Date(time * 1000); return d.getUTCHours().toString().padStart(2,'0') + ':' + d.getUTCMinutes().toString().padStart(2,'0'); }} }},
                        localization: {{ timeFormatter: (time) => {{ let d = new Date(time * 1000); return d.getUTCHours().toString().padStart(2,'0') + ':' + d.getUTCMinutes().toString().padStart(2,'0'); }} }},
                        handleScroll: {{ mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true }},
                        handleScale: {{ axisPressedMouseMove: true, mouseWheel: true, pinch: true }},
                    }});
                    new ResizeObserver(() => {{
                        window.lwChart.applyOptions({{ width: container.clientWidth, height: container.clientHeight }});
                    }}).observe(container);
                    return window.lwChart;
                }}

                window.updateChart = async function() {{
                    const codes = Array.from(window.checkedStocks);
                    const chartData = window.globalData.chartData || {{}};
                    let container = document.getElementById('chart-container');
                    if (codes.length === 0) {{
                        if (window.lwChart) {{ window.lwChart.remove(); window.lwChart = null; }}
                        container.innerHTML = '';
                        return;
                    }}

                    let targetDate = window.activeViewDate || '{date_str}';
                    let firstData = getFilteredData(codes[0], chartData, targetDate);
                    if (!firstData || firstData.ohlcv.length === 0) {{
                        if (window.lwChart) {{ window.lwChart.remove(); window.lwChart = null; }}
                        container.innerHTML = `<div style="color:gray; padding:20px; text-align:center; font-size:16px; margin-top:50px;">此標的在 ${{targetDate}} 無 K 線資料。</div>`;
                        return;
                    }}

                    let chart = ensureChart();

                    if (codes.length === 1) {{
                        let code = String(codes[0]);
                        let d = firstData;

                        // 標題
                        let totalPnl = 0;
                        if (window.currentMode === 'backtest') {{
                            let trades = (window.globalData.allTrades || []).filter(t => String(t.symbol) === code && (t.date === targetDate || !t.date));
                            totalPnl = trades.reduce((acc, t) => acc + (t.profit || 0), 0);
                        }}
                        chart.applyOptions({{ watermark: {{ visible: false }} }});
                        let titleDiv = document.getElementById('chart-title');
                        if (!titleDiv) {{
                            titleDiv = document.createElement('div');
                            titleDiv.id = 'chart-title';
                            titleDiv.style.cssText = 'position:absolute;top:8px;left:50%;transform:translateX(-50%);z-index:10;color:#D1D4DC;font-size:15px;font-weight:700;pointer-events:none;text-shadow:0 1px 3px rgba(0,0,0,0.5);';
                            document.getElementById('chart-container').appendChild(titleDiv);
                        }}
                        if (window.currentMode === 'backtest') {{
                            let pnlColor = totalPnl >= 0 ? '#EF5350' : '#26A69A';
                            titleDiv.innerHTML = `<span>${{code}} ${{d.name}}</span> <span style="color:#787B86;">|</span> <span>${{targetDate}}</span> <span style="color:#787B86;">|</span> <span style="color:${{pnlColor}}">損益: ${{Math.round(totalPnl).toLocaleString()}}</span>`;
                        }} else {{
                            titleDiv.innerHTML = `<span>${{code}} ${{d.name}}</span> <span style="color:#787B86;">|</span> <span>${{targetDate}}</span>`;
                        }}

                        // K 線或折線
                        if (window.chartType === 'kline') {{
                            window.lwCandleSeries = chart.addCandlestickSeries({{
                                upColor: '#EF5350', downColor: '#26A69A',
                                borderUpColor: '#EF5350', borderDownColor: '#26A69A',
                                wickUpColor: '#EF5350', wickDownColor: '#26A69A',
                            }});
                            window.lwCandleSeries.setData(d.ohlcv.map(r => ({{ time: r.time, open: r.open, high: r.high, low: r.low, close: r.close }})));
                        }} else {{
                            window.lwLineSeries = chart.addLineSeries({{ color: '#2962FF', lineWidth: 2 }});
                            window.lwLineSeries.setData(d.ohlcv.map(r => ({{ time: r.time, value: r.close }})));
                        }}

                        // 成交量
                        window.lwVolumeSeries = chart.addHistogramSeries({{
                            priceFormat: {{ type: 'volume' }},
                            priceScaleId: 'vol',
                        }});
                        chart.priceScale('vol').applyOptions({{ scaleMargins: {{ top: 0.85, bottom: 0 }} }});
                        window.lwVolumeSeries.setData(d.ohlcv.map(r => ({{
                            time: r.time,
                            value: r.vol,
                            color: r.close >= r.open ? 'rgba(239,83,80,0.5)' : 'rgba(38,166,154,0.5)',
                        }})));

                        // 回測模式：標記進出場 + 事件
                        if (window.currentMode === 'backtest') {{
                            let trades = (window.globalData.allTrades || []).filter(t => String(t.symbol) === code && (t.date === targetDate || !t.date));
                            let markers = [];

                            // 進出場標記 + 矩形區塊
                            trades.forEach(t => {{
                                let entryTs = 0, exitTs = 0;
                                d.ohlcv.forEach((r, idx) => {{
                                    if (d.times[idx] === robustFormat(t.entry_time)) entryTs = r.time;
                                    if (d.times[idx] === robustFormat(t.exit_time)) exitTs = r.time;
                                }});
                                if (entryTs) markers.push({{ time: entryTs, position: 'aboveBar', color: '#2962FF', shape: 'arrowDown', text: '', size: 0.5, _label: '空 ' + t.entry_price, _tradeType: 'entry' }});
                                if (exitTs) markers.push({{ time: exitTs, position: 'belowBar', color: '#F57C00', shape: 'arrowUp', text: '', size: 0.5, _label: '補 ' + t.exit_price, _tradeType: 'exit' }});

                                // 進出場價位線 + 矩形背景
                                let series = window.lwCandleSeries || window.lwLineSeries;
                                if (series) {{
                                    series.createPriceLine({{ price: t.entry_price, color: '#3498DB', lineWidth: 1, lineStyle: 0, axisLabelVisible: true, title: '進場' }});
                                    series.createPriceLine({{ price: t.exit_price, color: '#F39C12', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: '出場' }});
                                }}

                                // 損益矩形：進場價→出場價 區間填色
                                if (entryTs && exitTs) {{
                                    let isWin = t.entry_price > t.exit_price;
                                    let fillColor = isWin ? 'rgba(38,166,154,0.15)' : 'rgba(239,83,80,0.15)';
                                    let hi = Math.max(t.entry_price, t.exit_price);
                                    let lo = Math.min(t.entry_price, t.exit_price);
                                    let rectFill = chart.addBaselineSeries({{
                                        baseValue: {{ type: 'price', price: lo }},
                                        topLineColor: 'transparent', bottomLineColor: 'transparent',
                                        topFillColor1: fillColor, topFillColor2: fillColor,
                                        bottomFillColor1: 'transparent', bottomFillColor2: 'transparent',
                                        lineWidth: 0, lastValueVisible: false, priceLineVisible: false,
                                    }});
                                    let timeRange = d.ohlcv.filter(r => r.time >= entryTs && r.time <= exitTs);
                                    if (timeRange.length > 0) rectFill.setData(timeRange.map(r => ({{ time: r.time, value: hi }})));
                                }}
                            }});

                            // 事件標記
                            let currentGroup = null;
                            let groupSymbols = [];
                            for (let g in chartData) {{
                                if (chartData[g][code]) {{ currentGroup = g; groupSymbols = Object.keys(chartData[g]); break; }}
                            }}
                            let groupLevelEvents = ['確立天花板', '成為新領漲', '突破前高', '漲停觸發'];
                            let events = (window.globalData.allEvents || []).filter(e => {{
                                let sym = String(e.symbol);
                                let evt = e.event || '';
                                let eDate = e.date || '';
                                if (eDate && eDate !== targetDate) return false;
                                let isThisStock = (sym === code);
                                let isGroupTrigger = groupSymbols.includes(sym) && groupLevelEvents.some(ge => evt.includes(ge));
                                return isThisStock || isGroupTrigger;
                            }});

                            let eventSet = new Set();
                            events.forEach(e => {{
                                let eTime = robustFormat(e.time || e.Time);
                                let evName = e.event || e.Trigger || '觸發';
                                let sym = String(e.symbol);
                                let key = eTime + "_" + evName + "_" + sym;
                                if (eventSet.has(key)) return;
                                eventSet.add(key);
                                if (sym !== code) {{
                                    let srcName = '';
                                    for (let g in chartData) {{ if(chartData[g][sym]) {{ srcName = chartData[g][sym].name || sym; break; }} }}
                                    evName = evName + '(' + srcName + ')';
                                }}
                                let ts = 0;
                                d.ohlcv.forEach((r, idx) => {{ if (d.times[idx] === eTime) ts = r.time; }});
                                if (ts === 0) return;

                                let shape = 'circle', color = '#f7a600', shortName = evName;
                                if (evName.includes('漲停')) {{ shape = 'square'; color = '#ef5350'; shortName = '漲停'; }}
                                else if (evName.includes('拉高')) {{ color = '#ff9800'; shortName = '拉高'; }}
                                else if (evName.includes('天花板')) {{ color = '#2962ff'; shape = 'arrowDown'; shortName = '天花板'; }}
                                else if (evName.includes('前高')) {{ color = '#2962ff'; shortName = '破高'; }}
                                else if (evName.includes('領漲')) {{ color = '#9b59b6'; shape = 'square'; shortName = '領漲'; }}
                                // 族群來源標記加括號
                                if (evName.includes('(')) shortName = evName;
                                markers.push({{ time: ts, position: 'aboveBar', color: color, shape: shape, text: '', size: 0.5, _label: shortName }});
                            }});

                            markers.sort((a, b) => a.time - b.time);
                            let series = window.lwCandleSeries || window.lwLineSeries;
                            if (series && markers.length > 0) series.setMarkers(markers);

                            // HTML overlay 標籤（帶背景色外框）
                            let labelContainer = document.getElementById('ev-labels');
                            if (!labelContainer) {{
                                labelContainer = document.createElement('div');
                                labelContainer.id = 'ev-labels';
                                labelContainer.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:15;overflow:hidden;';
                                document.getElementById('chart-container').appendChild(labelContainer);
                            }}
                            labelContainer.innerHTML = '';

                            // 建立 time→high/low 的查找表
                            let highMap = {{}};
                            let lowMap = {{}};
                            d.ohlcv.forEach(r => {{ highMap[r.time] = r.high; lowMap[r.time] = r.low; }});

                            window._evMarkers = markers;
                            window._highMap = highMap;
                            window._lowMap = lowMap;
                            window._labelRafId = 0;

                            function updateLabelPositions() {{
                                if (!window.lwChart || !window._evMarkers) return;
                                let container = document.getElementById('ev-labels');
                                if (!container) return;
                                container.innerHTML = '';
                                let tScale = chart.timeScale();
                                let pSeries = window.lwCandleSeries || window.lwLineSeries;
                                if (!pSeries) return;

                                window._evMarkers.forEach(mk => {{
                                    if (!mk._label) return;
                                    let x = tScale.timeToCoordinate(mk.time);
                                    if (x === null) return;

                                    let lbl = document.createElement('div');
                                    let y;

                                    if (mk._tradeType) {{
                                        // 進出場標籤（styled pill）
                                        lbl.className = 'trade-label ' + mk._tradeType;
                                        if (mk._tradeType === 'entry') {{
                                            let hp = window._highMap[mk.time];
                                            if (!hp) return;
                                            y = pSeries.priceToCoordinate(hp);
                                            if (y === null) return;
                                            y = y - 26;
                                        }} else {{
                                            let lp = window._lowMap[mk.time];
                                            if (!lp) return;
                                            y = pSeries.priceToCoordinate(lp);
                                            if (y === null) return;
                                            y = y + 8;
                                        }}
                                    }} else {{
                                        // 事件標籤
                                        lbl.className = 'ev-label';
                                        lbl.style.backgroundColor = mk.color;
                                        lbl.style.color = '#fff';
                                        let hp = window._highMap[mk.time];
                                        if (!hp) return;
                                        y = pSeries.priceToCoordinate(hp);
                                        if (y === null) return;
                                        y = y - 20;
                                    }}

                                    if (y < 2) y = 2;
                                    lbl.style.left = x + 'px';
                                    lbl.style.top = y + 'px';
                                    lbl.textContent = mk._label;
                                    container.appendChild(lbl);
                                }});
                            }}
                            setTimeout(updateLabelPositions, 200);
                            let scheduleUpdate = () => {{
                                if (window._labelRafId) cancelAnimationFrame(window._labelRafId);
                                window._labelRafId = requestAnimationFrame(updateLabelPositions);
                            }};
                            chart.timeScale().subscribeVisibleLogicalRangeChange(scheduleUpdate);
                            chart.subscribeCrosshairMove(scheduleUpdate);
                        }}
                    }} else {{
                        // 多股比較模式
                        chart.applyOptions({{ watermark: {{ visible: false }} }});
                        let titleDiv2 = document.getElementById('chart-title');
                        if (!titleDiv2) {{
                            titleDiv2 = document.createElement('div');
                            titleDiv2.id = 'chart-title';
                            titleDiv2.style.cssText = 'position:absolute;top:8px;left:50%;transform:translateX(-50%);z-index:10;color:#D1D4DC;font-size:15px;font-weight:700;pointer-events:none;';
                            document.getElementById('chart-container').appendChild(titleDiv2);
                        }}
                        titleDiv2.innerHTML = `族群連動比較 (${{codes.length}} 檔) - ${{targetDate}}`;
                        let colors = ['#2962FF', '#EF5350', '#26A69A', '#F7A600', '#7C4DFF', '#FF9800', '#00BCD4', '#E91E63'];
                        codes.forEach((c, idx) => {{
                            let sd = getFilteredData(c, chartData, targetDate);
                            if (sd && sd.pct.length > 0) {{
                                let ls = chart.addLineSeries({{ color: colors[idx % colors.length], lineWidth: 2, title: c }});
                                ls.setData(sd.pct);
                                window.lwCompareLines.push(ls);
                            }}
                        }});
                    }}

                    chart.timeScale().fitContent();
                }};

                // 全域錯誤捕捉
                window.onerror = function(msg, url, line) {{
                    console.error('JS Error:', msg, 'at line', line);
                    let container = document.getElementById('chart-container');
                    if (container && !container.querySelector('.err-msg')) {{
                        container.innerHTML = '<div class="err-msg" style="color:#ef5350;padding:20px;text-align:center;margin-top:50px;"><h3>圖表載入失敗</h3><p>' + msg + '</p><p>Line: ' + line + '</p></div>';
                    }}
                }};

                document.addEventListener('DOMContentLoaded', () => {{
                    // 檢查 LightweightCharts 是否載入
                    if (typeof LightweightCharts === 'undefined') {{
                        document.getElementById('chart-container').innerHTML = '<div style="color:#ef5350;padding:40px;text-align:center;"><h2>圖表函式庫載入失敗</h2><p>無法從 CDN 載入 LightweightCharts，請檢查網路連線。</p></div>';
                    }}

                    window.initDateSelector();

                    // 日期選擇器顯示
                    let dateSel = document.getElementById('chart-date-selector');
                    if (window.availableDates.length > 0) dateSel.style.display = 'inline-block';

                    // K棒/折線切換
                    document.getElementById('btn-kline').onclick = () => {{ window.chartType='kline'; document.getElementById('btn-kline').classList.add('active'); document.getElementById('btn-line').classList.remove('active'); window.updateChart(); }};
                    document.getElementById('btn-line').onclick = () => {{ window.chartType='line'; document.getElementById('btn-line').classList.add('active'); document.getElementById('btn-kline').classList.remove('active'); window.updateChart(); }};

                    // 回測參數重算
                    document.getElementById('btn-convert').onclick = () => {{ if(window.pyBackend) {{ document.getElementById('loader').style.display='flex'; window.pyBackend.trigger_recalc(document.getElementById('param-wait').value, document.getElementById('param-hold').value, window.activeViewDate); }} }};

                    // 右側面板頁簽切換
                    document.querySelectorAll('.stab').forEach(tab => {{
                        tab.onclick = () => {{
                            document.querySelectorAll('.stab').forEach(t => t.classList.remove('active'));
                            document.querySelectorAll('.stab-panel').forEach(p => p.classList.remove('active'));
                            tab.classList.add('active');
                            let panel = document.getElementById(tab.dataset.panel);
                            if (panel) panel.classList.add('active');
                            // 切換模式
                            let panelId = tab.dataset.panel;
                            if (panelId === 'panel-watchlist') {{ window.currentMode = 'watchlist'; }}
                            else if (panelId === 'panel-backtest') {{ window.currentMode = 'backtest'; }}
                            else if (panelId === 'panel-cashflow') {{ if(window.pyBackend) window.pyBackend.show_cashflow_view(); return; }}
                            window.renderWatchlist(); window.updateChart();
                        }};
                    }});
                    // 預設啟動對應頁簽
                    if ('{mode}' === 'backtest') {{ document.querySelector('.stab[data-panel="panel-backtest"]').click(); }}

                    // 右鍵選單
                    document.getElementById('chart-container').addEventListener('contextmenu', (e) => {{
                        e.preventDefault();
                        let menu = document.getElementById('ctx-menu');
                        menu.style.display = 'block';
                        menu.style.left = e.clientX + 'px';
                        menu.style.top = e.clientY + 'px';
                    }});
                    document.addEventListener('click', () => {{ document.getElementById('ctx-menu').style.display = 'none'; }});

                    // (左側工具列已移除，功能整合至頂部工具列)

                    window.renderWatchlist(); window.updateChart();
                }});
            </script>
        </body>
        </html>
        """

class VolumeAnalysisWindow(QMainWindow):
    """策略參數掃描專屬視窗"""
    def __init__(self, html_file_path):
        super().__init__()
        self.setWindowTitle("大數據分析儀 - 參數熱力圖與訊號分佈")
        self.resize(1600, 900)
        
        self.browser = QWebEngineView()
        
        # 允許本機 HTML 存取外部 CDN 資源，徹底根除白屏
        self.browser.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        self.browser.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        self.browser.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        
        # 👇 加入這行：強制開啟 WebGL 硬體加速，讓 Plotly 平行座標圖可以順利渲染！
        self.browser.settings().setAttribute(QWebEngineSettings.WebGLEnabled, True)
        
        self.setCentralWidget(self.browser)
        
        # 讀取瘦身後的實體 HTML 檔案
        self.browser.load(QUrl.fromLocalFile(html_file_path))

# =============================================================
# v1.9.8.6 視窗與引擎管理：秒開圖表 + 背景計算 + 修正 unhashable
# =============================================================
term_win_instance = None 

def run_backtest_engine_196(wait_mins, hold_type, target_date, cached_data=None):
    """v1.9.8.6 核心引擎：背景重算完畢後，只回傳「純損益與事件」，不回傳肥大的 K 線"""

    print(f"\n⚡ 正在背景更新回測數據 (參數: {wait_mins}/{hold_type})...")

    w = int(wait_mins)
    h = None if hold_type.upper() == 'F' else int(hold_type)

    if cached_data is not None:
        mat, (d_kline, i_kline_full), dispo = cached_data
        i_kline = {sym: [r for r in rows if r.get('date') == target_date]
                   for sym, rows in i_kline_full.items()}
    else:
        mat, (d_kline, i_kline), dispo = load_matrix_dict_analysis(), load_kline_data(dates=[target_date]), load_disposition_stocks()
        
    all_trades, all_events, active_groups, pnl_summary = [], [], [], {}

    # 🛡️ 風控模組：每日重置的共享狀態（跨族群累計）
    risk_state = {'daily_entries': 0, 'daily_stops': 0, 'halted': False}

    for grp, syms in mat.items():
        data = initialize_stock_data([x for x in syms if x not in dispo], d_kline, i_kline)
        if not data: continue
        tp, _tc, t_hist, e_log = process_group_data(data, w, h, mat, target_date, verbose=False, risk_state=risk_state)
        if t_hist or e_log:
            active_groups.append(grp)
            all_trades.extend(t_hist)
            all_events.extend(e_log)
            for t in t_hist: 
                s_code = t['symbol']
                pnl_summary[s_code] = pnl_summary.get(s_code, 0) + t.get('profit', 0)
                
    print("✅ 背景回測程序已完成。")
    
    # 只回傳最輕量的分析結果 (幾 KB 而已)
    try:
        final_json = json.dumps({
            "allTrades": all_trades, 
            "allEvents": all_events, 
            "activeGroups": active_groups, 
            "pnlSummary": pnl_summary
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"背景 JSON 打包失敗: {e}")
        final_json = "{}"
        
    return final_json

class GroupManagerDialog(BaseDialog):
    def __init__(self):
        super().__init__("管理股票族群", (800, 550)) 
        layout = QVBoxLayout(self)
        
        # 頂部萬用搜尋列
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 搜尋股票 (支援代號、名稱或混合輸入，例如: 2330, 台積電, 2330台積電)")
        self.search_input.setStyleSheet(f"QLineEdit {{ padding: 8px; font-size: 14px; border-radius: 5px; border: 1px solid {TV['border_light']}; background-color: {TV['surface']}; color: {TV['text']}; }}")
        self.search_input.returnPressed.connect(self.search_stock) 
        
        btn_search = QPushButton("搜尋並定位")
        btn_search.setStyleSheet(f"QPushButton {{ background-color: {TV['blue']}; padding: 8px 15px; font-weight: bold; color: white; border-radius: 5px; }} QPushButton:hover {{ background-color: {TV['blue_hover']}; }}")
        btn_search.clicked.connect(self.search_stock)
        
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(btn_search)
        layout.addLayout(search_layout)

        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken); layout.addWidget(line)

        # 雙清單版面結構
        splitter = QSplitter(Qt.Horizontal)
        
        # --- 左側：族群列表 ---
        left_widget = QWidget(); left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("📁 族群分類 (💡 右鍵可修改名稱/刪除)"))
        self.group_list = QListWidget()
        self.group_list.setStyleSheet(f"QListWidget {{ font-size: 15px; background-color: {TV['surface']}; color: {TV['text']}; border: 1px solid {TV['border_light']}; border-radius: 5px; }} QListWidget::item:selected {{ background-color: {TV['blue_dim']}; color: {TV['text_bright']}; }}")
        
        self.group_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.group_list.customContextMenuRequested.connect(self.show_group_context_menu)
        self.group_list.itemSelectionChanged.connect(self.on_group_selected); left_layout.addWidget(self.group_list)
        
        btn_layout_l = QHBoxLayout()
        btn_add_g = QPushButton("➕ 新增族群")
        btn_add_g.setStyleSheet(f"QPushButton {{ background-color: {TV['green']}; padding: 6px; font-weight: bold; color: white; border-radius: 5px; }} QPushButton:hover {{ background-color: #1e8e82; }}")
        btn_add_g.clicked.connect(self.add_grp)
        btn_layout_l.addWidget(btn_add_g); left_layout.addLayout(btn_layout_l)
        
        # --- 右側：個股列表 ---
        right_widget = QWidget(); right_layout = QVBoxLayout(right_widget)
        self.lbl_current_group = QLabel("📌 請選擇左側族群 (💡 雙擊個股看走勢 / 右鍵刪除)")
        self.lbl_current_group.setStyleSheet(f"color: {TV['yellow']}; font-weight: bold;")
        right_layout.addWidget(self.lbl_current_group)
        
        self.stock_list = QListWidget()
        self.stock_list.setStyleSheet(f"QListWidget {{ font-size: 15px; background-color: {TV['bg']}; color: {TV['text']}; border: 1px solid {TV['border_light']}; border-radius: 5px; }} QListWidget::item:selected {{ background-color: {TV['purple']}; color: {TV['text_bright']}; }}")
        self.stock_list.setSelectionMode(QAbstractItemView.ExtendedSelection) 
        
        # 雙擊觸發終極 HTML 畫圖引擎
        self.stock_list.itemDoubleClicked.connect(self.plot_single_stock)
        self.stock_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.stock_list.customContextMenuRequested.connect(self.show_stock_context_menu)
        right_layout.addWidget(self.stock_list)
        
        # 右下角按鈕區
        btn_layout_r = QHBoxLayout()
        btn_add_s = QPushButton("➕ 新增個股")
        btn_add_s.setStyleSheet(f"QPushButton {{ background-color: {TV['blue']}; padding: 8px; font-weight: bold; color: white; border-radius: 5px; }} QPushButton:hover {{ background-color: {TV['blue_hover']}; }}")
        btn_add_s.clicked.connect(self.add_stk)
        
        btn_layout_r.addWidget(btn_add_s)
        right_layout.addLayout(btn_layout_r)
        
        splitter.addWidget(left_widget); splitter.addWidget(right_widget); splitter.setSizes([250, 550])
        layout.addWidget(splitter); self.refresh_groups()

    # --- 以下維持原本的 CRUD 邏輯 ---
    def search_stock(self):
        query = self.search_input.text().strip()
        if not query: return
        code_match = re.search(r'\d{4,}', query)
        target_code = code_match.group() if code_match else None
        target_name = query if not target_code else None

        g_dict = load_matrix_dict_analysis()
        found_groups = []
        found_code = None

        for grp_name, symbols in g_dict.items():
            for sym in symbols:
                sym_name = sn(sym) 
                if (target_code and target_code == sym) or (target_name and target_name in sym_name):
                    found_groups.append(grp_name)
                    found_code = sym 

        if not found_groups:
            return QMessageBox.information(self, "搜尋結果", f"找不到與「{query}」相關的個股。")

        if len(found_groups) == 1:
            target_group = found_groups[0]
            items = self.group_list.findItems(target_group, Qt.MatchExactly)
            if items:
                self.group_list.setCurrentItem(items[0])
                self.on_group_selected() 
                
            target_str_start = f"{found_code} "
            for i in range(self.stock_list.count()):
                item = self.stock_list.item(i)
                if item.text().startswith(target_str_start):
                    self.stock_list.setCurrentItem(item)
                    self.stock_list.scrollToItem(item, QAbstractItemView.PositionAtCenter)
                    break
        else:
            msg = f"🔍 股票【{found_code} {sn(found_code)}】目前存在於以下 {len(found_groups)} 個族群中：\n\n"
            for g in found_groups: msg += f"• {g}\n"
            QMessageBox.information(self, "多重族群提示", msg)

    def show_group_context_menu(self, pos):
        item = self.group_list.itemAt(pos)
        if not item: return
        menu = QMenu()
        menu.setStyleSheet(f"QMenu {{ background-color: {TV['panel']}; color: {TV['text']}; border: 1px solid {TV['border_light']}; }} QMenu::item:selected {{ background-color: {TV['blue']}; color: white; }}")
        rename_action = menu.addAction("✏️ 修改名稱")
        delete_action = menu.addAction("🗑️ 刪除該族群")
        action = menu.exec_(self.group_list.mapToGlobal(pos))
        if action == rename_action: self.rename_grp(item)
        elif action == delete_action: self.del_grp(item)

    def show_stock_context_menu(self, pos):
        item = self.stock_list.itemAt(pos)
        if not item: return
        menu = QMenu()
        menu.setStyleSheet(f"QMenu {{ background-color: {TV['panel']}; color: {TV['text']}; border: 1px solid {TV['border_light']}; }} QMenu::item:selected {{ background-color: {TV['blue']}; color: white; }}")
        view_action = menu.addAction("📈 開啟高級走勢分析圖")
        delete_action = menu.addAction("🗑️ 刪除該個股")
        action = menu.exec_(self.stock_list.mapToGlobal(pos))
        if action == view_action: self.plot_single_stock(item)
        elif action == delete_action: self.del_stk(item)

    def refresh_groups(self):
        self.group_list.clear(); groups = load_matrix_dict_analysis(); self.group_list.addItems(list(groups.keys()))
        if self.group_list.count() > 0: self.group_list.setCurrentRow(0)

    def on_group_selected(self):
        self.stock_list.clear()
        if not (selected := self.group_list.currentItem()): return self.lbl_current_group.setText("📌 請選擇左側族群")
        grp_name = selected.text()
        self.lbl_current_group.setText(f"📌 {grp_name} (💡 雙擊個股開啟高級走勢圖)")
        groups = load_matrix_dict_analysis(); load_twse_name_map()
        for code in groups.get(grp_name, []): self.stock_list.addItem(sn(code))

    def add_grp(self):
        grp, ok = QInputDialog.getText(self, "新增", "輸入新族群名稱:")
        if ok and grp:
            g = load_matrix_dict_analysis()
            if grp not in g: 
                g[grp] = []; save_matrix_dict(g); self.refresh_groups()
                if items := self.group_list.findItems(grp, Qt.MatchExactly): self.group_list.setCurrentItem(items[0])

    def rename_grp(self, item):
        old_name = item.text()
        new_name, ok = QInputDialog.getText(self, "修改名稱", f"將【{old_name}】修改為：", QLineEdit.Normal, old_name)
        if ok and new_name and new_name != old_name:
            g = load_matrix_dict_analysis()
            if new_name not in g:
                g[new_name] = g.pop(old_name)
                save_matrix_dict(g); self.refresh_groups()
                if items := self.group_list.findItems(new_name, Qt.MatchExactly): self.group_list.setCurrentItem(items[0])

    def del_grp(self, item=None):
        if not item: item = self.group_list.currentItem()
        if not item: return
        if QMessageBox.question(self, '確認刪除', f"確定要刪除整個【{item.text()}】嗎？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            g = load_matrix_dict_analysis()
            if item.text() in g: del g[item.text()]; save_matrix_dict(g); self.refresh_groups()

    def add_stk(self):
        if not (selected := self.group_list.currentItem()): return QMessageBox.warning(self, "提示", "請先選擇左側族群！")
        raw_input, ok = QInputDialog.getText(self, "新增個股", "輸入股票代號 (多檔用半形逗號分隔):")
        if ok and raw_input:
            g = load_matrix_dict_analysis(); added = False
            for raw_str in raw_input.split(','):
                stock_code = resolve_stock_code(raw_str)
                if stock_code and stock_code not in g[selected.text()]: g[selected.text()].append(stock_code); added = True
            if added: save_matrix_dict(g); self.on_group_selected()

    def del_stk(self, item=None):
        selected_grp = self.group_list.currentItem()
        selected_stocks = [item] if item else self.stock_list.selectedItems()
        if not selected_grp or not selected_stocks: return
        if QMessageBox.question(self, '確認刪除', "確定要刪除選取的個股嗎？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            g = load_matrix_dict_analysis()
            for i in selected_stocks:
                code = i.text().split(' ')[0]
                if code in g[selected_grp.text()]: g[selected_grp.text()].remove(code)
            save_matrix_dict(g); self.on_group_selected()

    # ==========================================
    # v1.9.8.6 修改版：開啟內建「高級走勢分析圖」 (看盤模式)
    # ==========================================
    def plot_single_stock(self, item):
        target_code = item.text().split(' ')[0]
        self.lbl_current_group.setText(f"⏳ 正在啟動內建終端，請稍候...")
        QApplication.processEvents() 
        try:
            # 呼叫入口並傳入 target_symbol，達成秒開圖表
            open_advanced_terminal_v196(
                initial_trades=[], 
                initial_events=[], 
                date_str="", 
                mode="watchlist", 
                target_symbol=target_code
            )
            self.lbl_current_group.setText(f"📌 {target_code} 走勢圖已載入 (💡 背景回測運算中...)")
        except Exception as e:
            logger.error(f"無法開啟內建終端: {e}", exc_info=True)
            QMessageBox.critical(self, "啟動錯誤", f"無法開啟內建終端: {e}")

class DispositionDialog(BaseDialog):
    def __init__(self):
        super().__init__("處置股清單", (300, 400))
        layout = QVBoxLayout(self)
        self.text = QTextEdit(); self.text.setReadOnly(True); self.text.setStyleSheet("font-family: Consolas; font-size: 14px;")
        layout.addWidget(self.text)
        try:
            data = load_disposition_stocks()
            stocks = data if isinstance(data, list) else data.get("stock_codes", [])
            if stocks:
                load_twse_name_map()
                for i, code in enumerate(stocks, 1): self.text.append(f"{i}. {sn(code)}")
            else: self.text.append("目前無處置股。")
        except Exception: self.text.append("無法讀取處置股檔案。")

class TutorialWidget(QWidget):
    """新手教學面板 - 左側目錄 + 右側內容"""
    _CSS = """
        <style>
            body {{ color: #D1D4DC; font-size: 14px; line-height: 1.9; font-family: sans-serif; }}
            h2 {{ color: #2196F3; margin-top: 0; margin-bottom: 10px; font-size: 17px; }}
            h3 {{ color: #26a69a; margin-top: 18px; margin-bottom: 6px; font-size: 14px; }}
            li {{ margin-bottom: 7px; }}
            code {{ background: #1e2230; padding: 2px 6px; border-radius: 3px; color: #f9ca24; font-size: 13px; }}
            .tip {{ background: #1a2a1a; border-left: 3px solid #26a69a; padding: 10px 14px; margin: 10px 0; border-radius: 0 4px 4px 0; }}
            .warn {{ background: #2a1a1a; border-left: 3px solid #e74c3c; padding: 10px 14px; margin: 10px 0; border-radius: 0 4px 4px 0; }}
            .step {{ background: #1a1e2d; border: 1px solid #2a2e3d; border-radius: 6px; padding: 10px 14px; margin: 8px 0; }}
        </style>
    """

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background-color: {TV['bg']};")
        root = QHBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # ── Left nav ──
        nav = QListWidget()
        nav.setFixedWidth(190)
        nav.setStyleSheet(f"""
            QListWidget {{
                background-color: {TV['panel']}; border: none;
                border-right: 1px solid {TV['border']}; font-size: 13px;
                color: {TV['text_dim']}; outline: none; padding-top: 8px;
            }}
            QListWidget::item {{ padding: 12px 16px; }}
            QListWidget::item:selected {{
                background-color: {TV['surface']}; color: {TV['text_bright']};
                border-left: 3px solid {TV['blue']};
            }}
            QListWidget::item:hover:!selected {{ background-color: {TV['border_light']}; }}
        """)

        pages_data = [
            ("🚀  快速開始",          self._page_start()),         # 0
            ("▶  盤中監控",           self._page_live()),          # 1
            ("📊  自選回測",          self._page_backtest()),      # 2
            ("📈  盤後數據分析",      self._page_analysis()),      # 3
            ("🔬  策略參數掃描",      self._page_volume_analysis()), # 4
            ("⚙️  系統參數設定",     self._page_settings()),      # 5
            ("🔑  帳戶 API 設定",    self._page_api()),           # 6
            ("📅  更新 K 線數據",    self._page_kline()),         # 7
            ("🛑  緊急平倉",         self._page_emergency()),     # 8
            ("❓  常見問題",         self._page_faq()),           # 9
            ("🏆  DTW 門檻優化",    self._page_opt_sim()),       # 10
            ("📐  平均過高分析",     self._page_avg_high()),      # 11
            ("🧬  族群連動分析",    self._page_correlation()),    # 12
            ("💰  利潤矩陣優化",    self._page_maximize()),       # 13
            ("🎯  自選進場模式",    self._page_simulate()),       # 14
            ("📱  Telegram 綁定",   self._page_telegram()),       # 15
            ("🏦  玉山 API 申請",   self._page_esun_api()),       # 16
            ("🏦  永豐 API 申請",   self._page_sinopac_api()),    # 17
            ("📖  參數名詞解釋",    self._page_glossary()),        # 18
        ]

        # 頁面名稱 → 索引對應表，供 goto() 使用
        self._PAGE_MAP = {
            'start':           0,
            'live_trading':    1,
            'backtest':        2,
            'analysis':        3,
            'volume_analysis': 4,
            'settings':        5,
            'api':             6,
            'kline':           7,
            'emergency':       8,
            'faq':             9,
            'opt_sim':         10,
            'avg_high':        11,
            'correlation':     12,
            'maximize':        13,
            'simulate':        14,
            'telegram':        15,
            'esun_api':        16,
            'sinopac_api':     17,
            'glossary':        18,
        }

        stack = QStackedWidget()
        _browser_style = f"""
            QTextBrowser {{
                background-color: {TV['bg']}; color: {TV['text']};
                border: none; font-size: 14px; padding: 28px 36px;
            }}
        """

        for label, html in pages_data:
            nav.addItem(label)
            browser = QTextBrowser(); browser.setOpenExternalLinks(False)
            browser.setStyleSheet(_browser_style)
            browser.setHtml(self._CSS + html)
            stack.addWidget(browser)

        nav.currentRowChanged.connect(stack.setCurrentIndex)
        nav.setCurrentRow(0)
        self._nav = nav   # 保留引用，供 goto() 使用

        root.addWidget(nav); root.addWidget(stack, 1)

    def goto(self, page_name: str):
        """按名稱跳轉到指定教學頁面（由各功能的教學按鈕呼叫）"""
        idx = self._PAGE_MAP.get(page_name)
        if idx is not None:
            self._nav.setCurrentRow(idx)

    def _page_start(self):
        return """
        <h2>快速開始</h2>
        <p>歡迎使用 REMORA。以下是首次使用的建議流程：</p>
        <div class="step">① <b>設定帳戶 API</b><br>點擊左側「帳戶 API 設定」，填入永豐金 API 憑證（CA 憑證路徑、帳號、密碼）</div>
        <div class="step">② <b>更新 K 線數據</b><br>點擊「更新 K 線數據」，在月曆選擇要分析的歷史交易日，下載 K 線資料</div>
        <div class="step">③ <b>執行回測分析</b><br>點擊「自選回測」，選擇族群與等待時間，執行模擬評估策略效果</div>
        <div class="step">④ <b>確認策略後啟用</b><br>回測結果穩定後，至「系統參數設定」切換為正式下單模式，再啟動「盤中監控」</div>
        <div class="tip"><b>💡 建議：</b>初次使用請先保持模擬模式，觀察系統運作至少一週後再切換正式下單。</div>
        """

    def _page_live(self):
        return """
        <h2>盤中監控</h2>
        <p>盤中監控是系統的核心功能，會即時掃描進場訊號並自動執行交易。</p>
        <h3>啟動步驟</h3>
        <ol>
            <li>確認 API 帳戶已正確設定並登入</li>
            <li>點擊左側「盤中監控」開啟面板</li>
            <li>設定「等待時間」（進場後確認訊號的分鐘數）與「持有時間」</li>
            <li>點擊「啟動盤中監控」</li>
        </ol>
        <h3>監控面板說明</h3>
        <ul>
            <li><b>上方日誌面板</b>：即時顯示系統日誌與訊號觸發紀錄</li>
            <li><b>下方持倉表</b>：顯示目前所有已成交持倉，含進場均價、現價、損益</li>
            <li><b>右鍵選單</b>：對持倉個股可執行手動操作或查看即時 K 線</li>
        </ul>
        <h3>手動下單列</h3>
        <ul>
            <li>輸入股票代號 → 選擇買進/賣出 → 選擇價格類型 → 輸入張數 → 送出委託</li>
            <li>選擇「市價」、「漲停」、「跌停」時，價格欄位會自動鎖定並顯示對應顏色</li>
        </ul>
        <div class="warn"><b>⚠ 注意：</b>啟動監控後，系統會自動鎖定回測等重度功能以保護效能。</div>
        """

    def _page_backtest(self):
        return """
        <h2>自選回測</h2>
        <p>用歷史 K 線數據模擬策略表現，評估不同族群與參數的獲利潛力。</p>
        <h3>參數說明</h3>
        <ul>
            <li><b>分析族群</b>：選擇要測試的股票族群，或選「所有族群」進行全面掃描</li>
            <li><b>等待時間</b>（分鐘）：領漲股觸發訊號後，等幾分鐘才進場；建議初始設定為 3~5 分鐘</li>
            <li><b>持有時間</b>（分鐘）：進場後持有幾分鐘出場；填 <code>F</code> 代表持倉至收盤（13:25）</li>
        </ul>
        <h3>結果解讀</h3>
        <ul>
            <li>分析完成後結果會顯示在「系統日誌」分頁中</li>
            <li>報表包含：勝率、平均獲利、最大回撤等關鍵指標</li>
        </ul>
        <div class="tip"><b>💡 建議：</b>先用「所有族群」找出表現最佳的族群，再針對該族群細調等待/持有時間。</div>
        """

    def _page_analysis(self):
        return """
        <h2>盤後數據與分析</h2>
        <p>提供多種盤後分析工具，協助找出最具潛力的族群與進場時機。</p>
        <h3>可用工具</h3>
        <ul>
            <li><b>族群相關性分析</b>：計算族群內個股 K 線相關係數，篩選高同步性族群</li>
            <li><b>計算平均過高</b>：分析族群中個股突破前高的平均時間間距，用於最佳化等待時間</li>
            <li><b>策略參數掃描</b>：分析成交量異動，找出主力介入訊號</li>
            <li><b>利潤矩陣優化</b>：用平行運算批次測試所有等待/持有時間組合</li>
        </ul>
        <div class="tip"><b>💡 建議：</b>每個交易日收盤後，先執行「相關性分析」更新族群評分，再執行「自選回測」驗證策略。</div>
        """

    def _page_settings(self):
        return """
        <h2>系統參數設定</h2>
        <p>設定交易策略核心參數，包含資本配置、費用、停損條件與風控規則。</p>
        <h3>各設定頁說明</h3>
        <ul>
            <li><b>基本交易設定</b>：每檔投入資本、手續費折扣、停損價差階距</li>
            <li><b>停損再進場</b>：停損後是否允許再次進場，以及再進場條件</li>
            <li><b>進階策略配置</b>：策略微調（動能門檻、濾網、選股條件）</li>
            <li><b>風控模組</b>：設定總額度上限、每日最大進場檔數、停損熔斷次數</li>
            <li><b>通知設定</b>：綁定 Telegram Bot 以接收即時推播</li>
        </ul>
        <div class="warn"><b>⚠ 模擬 vs 正式下單</b>：預設為模擬模式。切換為「正式下單模式」後，系統將使用 API 真實下單，請謹慎操作。</div>
        """

    def _page_api(self):
        return """
        <h2>帳戶 API 設定</h2>
        <p>設定永豐金證券 API 憑證，讓系統可以存取您的帳戶進行自動交易。</p>
        <h3>玉山銀行</h3>
        <ul>
            <li><b>API Key</b>：玉山銀行 Open API 金鑰</li>
            <li><b>API Secret</b>：對應 Secret</li>
            <li><b>帳號</b>：您的玉山帳號</li>
            <li><b>憑證路徑</b>：CA 憑證 .pfx 檔案路徑（需事先向玉山銀行申請）</li>
        </ul>
        <h3>永豐金證券</h3>
        <ul>
            <li><b>模擬帳戶</b>：用於模擬模式測試，不會真實下單</li>
            <li><b>正式帳戶</b>：正式交易帳戶資訊，需配合 CA 憑證</li>
        </ul>
        <div class="warn"><b>⚠ 安全注意：</b>請勿將 API Key 與 Secret 分享給他人，憑證檔案妥善保管。</div>
        """

    def _page_kline(self):
        return """
        <h2>更新 K 線數據</h2>
        <p>下載並儲存指定交易日的 K 線資料，作為回測與分析的基礎。</p>
        <h3>操作步驟</h3>
        <ol>
            <li>點擊月曆上的交易日（可多選，已採集日期顯示為綠色）</li>
            <li>確認右側「已選取日期」清單正確</li>
            <li>點擊「立即採集選定日期」開始下載</li>
            <li>下載進度會顯示在系統日誌中</li>
        </ol>
        <div class="tip">
            <b>💡 建議：</b>每個交易日結束後（13:30 後）採集當日資料，維持最新的回測基礎。<br>
            初次使用建議一次採集最近 20~30 個交易日的資料，讓策略有足夠的歷史樣本。
        </div>
        """

    def _page_emergency(self):
        return """
        <h2>緊急平倉</h2>
        <p>緊急操作中心，用於突發狀況下快速處置持倉。</p>
        <h3>可用操作</h3>
        <ul>
            <li><b>一鍵全部市價平倉</b>：立即以市價賣出所有持倉，不可逆操作，請謹慎使用</li>
            <li><b>指定股票平倉</b>：輸入股票代號，僅平倉該檔股票</li>
            <li><b>退出監控（不平倉）</b>：停止自動監控，但保留現有持倉</li>
            <li><b>強制關閉程式</b>：立即終止程式，持倉不受影響（仍在您的帳戶中）</li>
        </ul>
        <div class="warn">
            <b>⚠ 重要提示：</b><br>
            「一鍵全部市價平倉」在流動性不足時可能產生大幅滑價。<br>
            強制關閉程式不會自動平倉，請確認持倉狀態後再使用。
        </div>
        """

    def _page_volume_analysis(self):
        return """
        <h2>策略參數掃描</h2>
        <p>策略參數掃描透過人工智慧演算法對策略參數進行網格搜索，自動找出最佳參數組合。搜索結果可直接套用至盤中監控與回測。</p>

        <h3>基本流程</h3>
        <div class="step">① <b>採集數據</b><br>點擊「1. 採集大數據」，選擇歷史交易日範圍（建議近 60 天），系統會從 K 線資料庫中提取分鐘線資料。</div>
        <div class="step">② <b>設定參數範圍</b><br>在參數設定表格中，勾選「固定?」代表固定該值，不勾選則設定搜索起點、終點與步長，AI 會在此範圍內搜索最佳值。</div>
        <div class="step">③ <b>啟動網格搜索</b><br>點擊「2. 開始網格搜索」，系統以多核平行方式執行 Optuna 試驗，右側終端會顯示即時進度。</div>
        <div class="step">④ <b>解讀結果</b><br>搜索完成後，結果表格依「Total PnL」降序排列，最頂端的參數組合即為 AI 推薦的最佳設定。</div>

        <h3>關鍵參數說明</h3>
        <ul>
            <li><b>等待時間</b>：領漲觸發後等待幾根 K 棒才允許進場，數值越大過濾越嚴但訊號越少</li>
            <li><b>DTW 相似度</b>：設為 0 表示關閉 DTW 過濾（預設），數值越高要求跟漲股走勢與領漲股越相似</li>
            <li><b>領漲拉高幅</b>：領漲股在當根 K 棒上漲的最低幅度（%），觸發訊號的門檻</li>
            <li><b>跟漲追蹤幅</b>：跟漲股需達到的最低漲幅（%）才被列入候選</li>
            <li><b>絕對成交量</b>：單根 K 棒的最低成交量（張），過濾低流動性股票</li>
            <li><b>尾盤停止觸發</b>：幾點後停止新的進場訊號（建議 12:30~13:00 之間）</li>
            <li><b>固定持倉時間</b>：進場後最多持倉幾分鐘（F 表示持到尾盤）</li>
        </ul>

        <h3>回歸 vs 回測</h3>
        <div class="tip">
            <b>參數搜索（策略參數掃描）</b>：AI 抽樣部分日期進行高速評估，找出候選參數，最後對全部日期做一次全量驗證。速度快，用於參數探索。<br><br>
            <b>回測（run_batch_backtest.py）</b>：使用固定參數對全部歷史資料完整回放，結果最準確，用於策略確認。
        </div>

        <h3>多核加速說明</h3>
        <p>系統自動使用多核心平行運算（最多 4 個 worker），執行速度比單進程快 2~4 倍。回歸期間終端訊息每 10 秒更新一次 worker 狀態，進度條每次試驗都會推進。</p>
        """

    def _page_opt_sim(self):
        return """
        <h2>DTW 門檻優化</h2>
        <p>本功能透過大量歷史資料回測，自動找出最佳的 DTW 相似度門檻值，讓族群連動訊號更準確。</p>
        <h3>操作步驟</h3>
        <div class="step">① <b>STEP 1 — 採集大數據庫</b><br>輸入採集天數（建議 5~30 天），點擊「執行數據採集」，系統從 K 線資料庫中提取分鐘線資料。</div>
        <div class="step">② <b>STEP 2 — 智能門檻掃描</b><br>點擊「執行門檻最佳化掃描」，系統在 DTW 門檻範圍內逐步測試，評估各門檻下的進場成功率與損益。</div>
        <div class="step">③ <b>解讀結果</b><br>結果圖表顯示各 DTW 門檻對應的平均損益，選擇損益最高且穩定的門檻值作為最終設定。</div>
        <h3>DTW 是什麼？</h3>
        <p>DTW（Dynamic Time Warping）測量兩支股票走勢的相似程度，門檻越高代表要求越相似。建議範圍 0.3~0.7，設為 0 表示關閉 DTW 過濾。</p>
        <div class="tip"><b>💡 建議：</b>先用策略參數掃描找到最佳參數組合，再用本功能精調 DTW 門檻。</div>
        """

    def _page_avg_high(self):
        return """
        <h2>平均過高分析</h2>
        <p>分析各族群中個股突破「前高」的平均時間間距，用於設定等待時間參數的上限。</p>
        <h3>操作步驟</h3>
        <div class="step">① 選擇要分析的族群（或選「分析全部族群」）</div>
        <div class="step">② 點擊「分析選定族群」，系統掃描歷史資料中各族群的過高間隔分布</div>
        <div class="step">③ 解讀輸出結果：平均過高間隔（分鐘）代表領漲股拉高後，跟漲股平均在幾分鐘內出現信號</div>
        <h3>如何使用結果？</h3>
        <p>若某族群的平均過高間隔為 4 分鐘，則在策略參數掃描中，等待時間（wait_mins）建議設定為 3~5 分鐘。</p>
        <div class="tip"><b>💡 建議：</b>對主要操作的族群定期（每週）重新計算，市場節奏會隨時間變化。</div>
        """

    def _page_correlation(self):
        return """
        <h2>族群連動分析</h2>
        <p>掃描跨族群之間的相關性，識別哪些族群在盤中有高度連動行為，協助選擇最佳監控族群。</p>
        <h3>兩種分析模式</h3>
        <ul>
            <li><b>[A] 宏觀連動（09:00~13:30）</b>：統計整個交易日的日線相關係數，用於長期族群配置決策</li>
            <li><b>[B] 盤中等待期模擬</b>：模擬盤中等待期間的即時相關性，直接對齊策略的實際執行情境</li>
        </ul>
        <h3>操作步驟</h3>
        <div class="step">① 選擇分析模式（宏觀 / 微觀）</div>
        <div class="step">② 微觀模式需設定等待時間（與策略參數掃描的 wait_mins 一致）</div>
        <div class="step">③ 點擊確認，結果以熱力圖呈現族群間相關係數（越接近 1 連動越強）</div>
        <div class="tip"><b>💡 建議：</b>選擇相關係數 > 0.7 的族群對作為主要監控對象，連動強代表信號出現後跟漲機率更高。</div>
        """

    def _page_maximize(self):
        return """
        <h2>利潤矩陣優化</h2>
        <p>對多個等待時間 × 持倉時間的組合同時進行回測，找出最佳獲利的最佳參數矩陣。</p>
        <h3>操作步驟</h3>
        <div class="step">① <b>選擇族群</b>：選擇要分析的族群（或全部族群）</div>
        <div class="step">② <b>設定等待時間範圍</b>：設定等待時間的起點與終點（如 3~5 分鐘）</div>
        <div class="step">③ <b>設定持倉時間範圍</b>：設定持倉時間起終點，F 表示持到尾盤</div>
        <div class="step">④ <b>啟動矩陣併發回測</b>：系統同時對所有參數組合跑回測，完成後輸出損益矩陣熱力圖</div>
        <h3>解讀結果</h3>
        <p>熱力圖中顏色越深代表損益越高，找出損益最高且穩定（周圍格子也有良好表現）的參數組合。</p>
        <div class="warn"><b>⚠️ 注意：</b>過高的損益可能是過擬合，建議選擇在多個參數值都有穩定表現的區域。</div>
        """

    def _page_simulate(self):
        return """
        <h2>自選進場模式</h2>
        <p>針對指定族群，使用自訂的等待時間與持倉時間，模擬歷史盤中進場的結果。</p>
        <h3>操作步驟</h3>
        <div class="step">① <b>選擇族群</b>：從下拉選單選擇要回測的族群，或選「所有族群」</div>
        <div class="step">② <b>設定等待時間</b>：領漲拉高後等幾分鐘才允許進場（建議 3~5 分鐘）</div>
        <div class="step">③ <b>設定持倉時間</b>：進場後最多持倉幾分鐘，F 表示持到尾盤（13:25 停損出場）</div>
        <div class="step">④ <b>點擊「執行回測」</b>：系統對選定族群的歷史資料逐日模擬，輸出詳細進出場紀錄</div>
        <h3>與策略參數掃描的差異</h3>
        <ul>
            <li><b>自選進場模式</b>：手動指定固定參數，完整回放每一筆交易，結果精確</li>
            <li><b>策略參數掃描</b>：AI 自動探索參數空間，找出最佳組合，速度快但有抽樣誤差</li>
        </ul>
        <div class="tip"><b>💡 建議流程：</b>先用策略參數掃描找到候選參數 → 再用自選進場模式精確驗證 → 確認後套用到盤中監控。</div>
        """

    def _page_telegram(self):
        return """
        <h2>Telegram 綁定教學</h2>
        <p>綁定 Telegram 後，系統可即時推送進場、停損、盤後報告等通知到您的手機。</p>
        <h3>綁定步驟</h3>
        <div class="step">① <b>開啟 Telegram</b><br>在 Telegram 搜尋欄輸入 <code>@duduautotrade_bot</code> 並加入</div>
        <div class="step">② <b>取得授權碼</b><br>向機器人傳送 <code>/start</code>，機器人會回覆您的專屬授權碼（純數字）</div>
        <div class="step">③ <b>回到 REMORA 設定</b><br>打開「系統參數設定 → 通知與外觀」，在「綁定授權碼」欄位貼上數字</div>
        <div class="step">④ <b>點擊「套用」</b><br>儲存後 Telegram 機器人會自動連線，可在 Telegram 中使用按鈕操控系統</div>
        <div class="tip"><b>💡 提示：</b>綁定成功後，盤中監控期間的進場、停損、出場都會即時推送到您的 Telegram。</div>
        """

    def _page_esun_api(self):
        return """
        <h2>玉山證券 API 申請教學</h2>
        <p>玉山證券 API 用於取得即時行情報價資料。</p>
        <h3>申請步驟</h3>
        <div class="step">① <b>開立玉山證券帳戶</b><br>若尚未開戶，請先至玉山證券臨櫃或線上開立證券帳戶</div>
        <div class="step">② <b>申請 API 權限</b><br>登入玉山證券官網，至「API 服務」或聯繫營業員開通程式交易 API 權限</div>
        <div class="step">③ <b>下載憑證檔案</b><br>取得 <code>.p12</code> 憑證檔案，存放在本機安全位置</div>
        <div class="step">④ <b>在 REMORA 設定</b><br>開啟「帳戶 API 設定」，選擇玉山 API，填入帳號與憑證路徑</div>
        <div class="warn"><b>⚠️ 注意：</b>憑證檔案為機密資料，請勿分享或上傳至雲端。</div>
        """

    def _page_sinopac_api(self):
        return """
        <h2>永豐金 Shioaji API 申請教學</h2>
        <p>永豐金 Shioaji 是 REMORA 主要的下單與行情 API。</p>
        <h3>申請步驟</h3>
        <div class="step">① <b>開立永豐金證券帳戶</b><br>至永豐金證券臨櫃或線上開戶，需同時開立證券與期貨帳戶</div>
        <div class="step">② <b>申請 API 權限</b><br>至 <code>Shioaji 官網</code> 註冊開發者帳號，並綁定您的永豐金帳戶</div>
        <div class="step">③ <b>取得 API Key 與憑證</b><br>登入後在「API 管理」頁面產生 API Key，並下載 CA 憑證（<code>.pfx</code> 檔案）</div>
        <div class="step">④ <b>在 REMORA 設定</b><br>開啟「帳戶 API 設定」，填入 API Key、Secret Key、CA 憑證路徑、帳號與密碼</div>
        <div class="step">⑤ <b>測試連線</b><br>點擊「登入」按鈕，確認可成功連線至永豐金伺服器</div>
        <div class="warn"><b>⚠️ 注意：</b>API Key 與憑證檔案為機密資料，切勿外洩。首次使用建議先在模擬模式下測試。</div>
        """

    def _page_glossary(self):
        return """
        <h2>參數名詞解釋</h2>
        <p>以下以白話解釋「進階策略配置」中各參數的意義與建議設定。</p>

        <h3>整體動能與關聯</h3>
        <div class="step"><b>領漲股需連續發動幾分鐘</b>（momentum_minutes）<br>
        領漲股的漲幅必須連續達標多少分鐘，系統才認定為真正的拉抬信號。<br>
        <b>調高</b>：更嚴格，減少假信號但可能錯過快速行情。<b>調低</b>：更敏感。建議值：1～3</div>

        <div class="step"><b>族群K線相似度門檻</b>（similarity_threshold）<br>
        用 DTW 演算法比較領漲股與跟漲股的走勢相似程度，數值越<b>低</b>越嚴格（要求越像）。<br>
        <b>調低</b>：只進場走勢高度相似的股票。<b>調高</b>：放寬條件。建議值：0.2～0.5</div>

        <h3>觸發條件門檻</h3>
        <div class="step"><b>領漲股區間漲幅 (%)</b>（pull_up_pct_threshold）<br>
        領漲股在觀察區間內的漲幅超過此值時，判定為「拉高觸發」。<br>
        <b>調高</b>：只抓大幅拉抬，進場次數少。<b>調低</b>：容易觸發。建議值：0.8～2.0</div>

        <div class="step"><b>跟漲股追蹤漲幅 (%)</b>（follow_up_pct_threshold）<br>
        跟漲股當日漲幅超過此值才列入候選名單。<br>
        <b>調高</b>：只選已經明顯上漲的股票。<b>調低</b>：更多候選。建議值：0.3～1.0</div>

        <h3>進場防護濾網</h3>
        <div class="step"><b>進場時標的當日漲幅下限/上限 (%)</b>（rise_lower_bound / rise_upper_bound）<br>
        進場時，標的當日漲幅必須在此區間內。避免追高或買進跌勢股。<br>
        建議值：下限 -1.0～0、上限 7.0～8.0</div>

        <div class="step"><b>觸發爆量：高於開盤均量倍數</b>（volume_multiplier）<br>
        等待期間成交量必須達到開盤均量的倍數才放行。<br>
        <b>調高</b>：要求更明顯的爆量。建議值：1.0～2.0</div>

        <div class="step"><b>觸發爆量：成交量最低門檻 (張)</b>（min_volume_threshold）<br>
        等待期間任一分鐘的成交量必須達到此張數。過濾冷門股。<br>
        建議值：500～1500</div>

        <div class="step"><b>IOC委託向下穿價容許 (Tick)</b>（slippage_ticks）<br>
        下 IOC 委託時，願意接受低於市價幾個 Tick 的成交價。數值越大越容易成交但滑價越大。<br>
        建議值：2～5</div>

        <div class="step"><b>停損緩衝空間 (%)</b>（sl_cushion_pct）<br>
        在計算出的停損價上方額外加一段緩衝。避免因微小波動誤觸停損。<br>
        建議值：0～0.3</div>

        <div class="step"><b>突破前高容錯率 (%)</b>（pullback_tolerance）<br>
        等待期間，若最新漲幅超過前面所有K棒的最高漲幅加上此容錯率，則跳過此次進場（可能是二次拉抬）。<br>
        建議值：0.3～1.0</div>

        <div class="step"><b>尾盤截止觸發</b>（cutoff_time_mins）<br>
        超過此時間後不再觸發新的進場。以分鐘表示（270 = 13:30 = 不截止）。<br>
        建議值：250～270（12:50～13:30）</div>

        <h3>選股門檻</h3>
        <div class="step"><b>標的漲幅需落後領漲股 (%)</b>（min_lag_pct）<br>
        跟漲股的漲幅必須至少落後領漲股多少百分點。確保買的是「還沒漲到位」的股票。<br>
        建議值：0.5～2.0</div>

        <div class="step"><b>標的今日最高漲幅需達 (%)</b>（min_height_pct）<br>
        標的今日歷史最高漲幅需達此值，證明它有動能。<br>
        建議值：2.5～5.0</div>

        <div class="step"><b>當日漲跌幅活動範圍下限 (%)</b>（volatility_min_range）<br>
        今日最高價與最低價之間的幅度。過濾整天不動的股票。<br>
        建議值：1.0～2.5</div>

        <div class="step"><b>全日均量下限 (張/分)</b>（min_eligible_avg_vol）<br>
        標的全日平均每分鐘成交量需達此值。0 表示不過濾。過濾流動性極低的股票。<br>
        建議值：1～5</div>

        <div class="step"><b>股價下限 (元)</b>（min_close_price）<br>
        標的股價低於此值則排除。0 表示不過濾。避免進場過低價位的股票。<br>
        建議值：0 或 10～30</div>
        """

    def _page_faq(self):
        return """
        <h2>常見問題</h2>
        <h3>K 線數據沒有資料 / 回測結果為空</h3>
        <p>請先至「更新 K 線數據」下載指定日期的資料，確認採集成功後再執行回測。</p>
        <h3>Telegram 未連線</h3>
        <p>至「系統參數設定 → 通知設定」填入您的授權碼（Chat ID）並點擊儲存。</p>
        <h3>登入 / API 連線失敗</h3>
        <p>確認永豐金 CA 憑證路徑正確，憑證需事先向永豐金申請。帳號密碼輸入後需重新儲存。</p>
        <h3>系統顯示「正式下單模式」但我不想真實下單</h3>
        <p>至「系統參數設定 → 基本交易設定」，確認「下單模式」選項回到模擬模式並儲存。</p>
        <h3>分析跑完但看不到結果</h3>
        <p>分析結果輸出到「系統日誌」分頁。點擊頂部的「系統日誌」分頁即可查看。</p>
        <div class="tip"><b>💡 若問題仍未解決：</b>請至「系統日誌」查看錯誤訊息，或重新啟動程式。</div>
        """


class TradeLogViewerDialog(BaseDialog):
    def __init__(self):
        super().__init__("📜 歷史交易紀錄與日誌", (900, 600))
        self.layout = QVBoxLayout(self)
        self.load_data()

    def load_data(self):
        # 如果已經有表格存在，先清空它 (確保重新整理時不會疊加)
        for i in reversed(range(self.layout.count())): 
            widget = self.layout.itemAt(i).widget()
            if widget is not None: widget.setParent(None)

        try:
            # 把資料庫的 id 也撈出來，作為刪除依據
            df = pd.read_sql("SELECT id, timestamp, action, symbol, shares, price, profit, note FROM trade_logs ORDER BY id DESC", sys_db.conn)
            
            self.table = QTableWidget(len(df), 7)
            self.table.setHorizontalHeaderLabels(["時間", "動作", "商品", "股數", "價格", "損益", "備註"])
            self.table.setStyleSheet(f"QTableWidget {{ background-color: {TV['bg']}; color: {TV['text']}; gridline-color: {TV['border']}; selection-background-color: {TV['blue_dim']}; }} QHeaderView::section {{ background-color: {TV['surface']}; color: {TV['text_bright']}; font-weight: bold; padding: 5px; border: 1px solid {TV['border']}; }}")
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.table.setSelectionBehavior(QTableWidget.SelectRows)
            
            for r, row in df.iterrows():
                # 第 0 欄：時間 (偷偷把 id 藏在這個 Item 裡面)
                time_item = QTableWidgetItem(str(row['timestamp']))
                time_item.setData(Qt.UserRole, int(row['id']))
                self.table.setItem(r, 0, time_item)
                
                # 動作顏色判斷（使用 in 匹配，支援 "委託(進場)"、"買回(平倉)"、"平倉(尾盤)" 等格式）
                act_item = QTableWidgetItem(str(row['action']))
                act_str = str(row['action'])
                if '進場' in act_str or '買進' in act_str:
                    act_item.setForeground(QColor(TV['red']))
                elif '平倉' in act_str or '買回' in act_str:
                    act_item.setForeground(QColor(TV['green']))
                self.table.setItem(r, 1, act_item)
                
                self.table.setItem(r, 2, QTableWidgetItem(sn(str(row['symbol']))))
                self.table.setItem(r, 3, QTableWidgetItem(str(row['shares'])))
                self.table.setItem(r, 4, QTableWidgetItem(f"{row['price']:.2f}"))
                
                prof = float(row['profit'])
                p_item = QTableWidgetItem(f"{prof:.0f}" if prof != 0 else "-")
                if prof > 0: p_item.setForeground(QColor(TV['red']))
                elif prof < 0: p_item.setForeground(QColor(TV['green']))
                self.table.setItem(r, 5, p_item)
                
                self.table.setItem(r, 6, QTableWidgetItem(str(row['note'])))
                
            # 啟用右鍵選單功能
            self.table.setContextMenuPolicy(Qt.CustomContextMenu)
            self.table.customContextMenuRequested.connect(self.show_context_menu)
            
            self.layout.addWidget(self.table)
            
        except Exception as e:
            self.layout.addWidget(QLabel(f"無法讀取資料庫：{e}"))

    # 處理右鍵選單的邏輯
    def show_context_menu(self, pos):
        item = self.table.itemAt(pos)
        if item is None: return

        row = item.row()
        symbol_name = self.table.item(row, 2).text()
        db_id = self.table.item(row, 0).data(Qt.UserRole) # 把剛才藏進去的 ID 拿出來

        menu = QMenu()
        menu.setStyleSheet(f"""
            QMenu {{ background-color: {TV['panel']}; color: {TV['text']}; border: 1px solid {TV['border_light']}; font-size: 14px; }}
            QMenu::item {{ padding: 6px 20px; }}
            QMenu::item:selected {{ background-color: {TV['red']}; color: white; }}
        """)

        del_action = menu.addAction(f"🗑️ 刪除這筆紀錄 ({symbol_name})")
        action = menu.exec_(self.table.viewport().mapToGlobal(pos))
        
        if action == del_action:
            self.delete_record(row, db_id, symbol_name)

    # 執行刪除與雙重確認
    def delete_record(self, row, db_id, symbol_name):
        # 設定訊息對話框為白字
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle('確認刪除')
        msg_box.setText(f"確定要永久刪除 【{symbol_name}】 的這筆交易紀錄嗎？")
        msg_box.setStyleSheet("""
            QMessageBox QLabel { color: #FFFFFF; font-size: 14px; font-weight: bold; }
            QMessageBox QPushButton { background-color: #34495E; color: white; padding: 6px 15px; border-radius: 4px; }
            QMessageBox QPushButton:hover { background-color: #E74C3C; }
            QMessageBox { background-color: #121212; }
        """)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        
        if msg_box.exec_() == QMessageBox.Yes:
            try:
                # 1. 從資料庫中刪除
                with sys_db.db_lock:
                    with sys_db.conn:
                        sys_db.conn.execute("DELETE FROM trade_logs WHERE id = ?", (db_id,))
                
                # 2. 直接從畫面上移除該列 (不用重新讀取資料庫，畫面更滑順)
                self.table.removeRow(row)
                
            except Exception as e:
                QMessageBox.warning(self, "錯誤", f"刪除失敗: {e}")

class EmergencyDialog(BaseDialog):
    def __init__(self):
        super().__init__("緊急操作中心", (420, 300))
        layout = QVBoxLayout(self); layout.setSpacing(10); layout.setContentsMargins(20, 20, 20, 20)

        warn_lbl = QLabel("請謹慎操作，部分動作不可逆")
        warn_lbl.setStyleSheet(f"color: {TV['red']}; font-size: 12px; font-weight: 600; padding: 4px 0 10px 0; border-bottom: 1px solid {TV['red_dim']};")
        layout.addWidget(warn_lbl)

        def _ebtn(text, color, cb, h=44):
            b = QPushButton(text); b.setFixedHeight(h); b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(f"QPushButton {{ background-color: {color}; color: white; border: none; border-radius: 7px; font-size: 14px; font-weight: 700; }} QPushButton:hover {{ opacity: 0.85; }}")
            b.clicked.connect(cb); return b

        b1 = _ebtn("⚠ 一鍵全部市價平倉", TV['red'],
                   lambda: [self.accept(), threading.Thread(target=exit_trade_live, daemon=True).start()])
        b2 = _ebtn("指定股票平倉", TV['blue'], self.single_close, h=40)
        b3 = _ebtn("⏸ 退出監控（不平倉）", TV['yellow'].replace('f7a600','d68a00'), self.stop_live, h=40)
        b3.setStyleSheet(b3.styleSheet().replace("color: white", f"color: {TV['bg']}"))
        b4 = _ebtn("✕ 強制關閉程式（不平倉）", TV['surface'], lambda: os._exit(0), h=36)
        b4.setStyleSheet(b4.styleSheet() + f"border: 1px solid {TV['border_light']};")

        for b in [b1, b2, b3, b4]: layout.addWidget(b)

    def single_close(self):
        code, ok = QInputDialog.getText(self, "單一平倉", "請輸入股票代號:")
        if ok and code: self.accept(); threading.Thread(target=close_one_stock, args=(code,), daemon=True).start()

    def stop_live(self):
        self.accept()
        sys_state.stop_trading_flag = True
        QMessageBox.information(self, "提示", "已發送終止指令。\n系統將在當前掃描週期結束後自動退出監控模式。")

class PortfolioMonitorDialog(BaseDialog):
    def __init__(self):
        super().__init__("即時持倉監控", (740, 380))
        self.last_refresh_time = 0.0

        layout = QVBoxLayout(self); layout.setSpacing(10); layout.setContentsMargins(16, 16, 16, 16)

        # 頂部控制列
        header_layout = QHBoxLayout()
        header_label = QLabel("成交時自動更新，亦可手動重新整理")
        header_label.setStyleSheet(f"color: {TV['text_dim']}; font-size: 12px;")

        self.btn_refresh = QPushButton("↻ 重新整理（3 秒間隔）")
        self.btn_refresh.setFixedHeight(32); self.btn_refresh.setCursor(Qt.PointingHandCursor)
        self.btn_refresh.setStyleSheet(f"""
            QPushButton {{ background-color: {TV['blue']}; color: white; border: none; border-radius: 5px; font-size: 12px; font-weight: 600; padding: 0 12px; }}
            QPushButton:hover {{ background-color: {TV['blue_hover']}; }}
            QPushButton:disabled {{ background-color: {TV['surface']}; color: {TV['text_dim']}; }}
        """)
        self.btn_refresh.clicked.connect(self.on_refresh_clicked)

        header_layout.addWidget(header_label); header_layout.addStretch(); header_layout.addWidget(self.btn_refresh)
        layout.addLayout(header_layout)

        # 表格
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["股票代號", "進場均價", "現價", "張數", "未實現損益", "停損價"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(f"""
            QTableWidget {{ background-color: {TV['panel']}; alternate-background-color: {TV['surface']}; gridline-color: {TV['border']}; border: none; border-radius: 6px; }}
            QTableWidget::item {{ color: {TV['text']}; padding: 6px; }}
            QTableWidget::item:selected {{ background-color: {TV['blue_dim']}; color: white; }}
            QHeaderView::section {{ background-color: {TV['surface']}; color: {TV['text_dim']}; font-weight: 700; font-size: 12px; border: none; border-right: 1px solid {TV['border']}; padding: 8px; }}
        """)
        layout.addWidget(self.table)
        
        ui_dispatcher.portfolio_updated.connect(self.update_table)
        
        # 🚀 視窗打開的瞬間，立刻主動廣播一次，消滅黑畫面空窗期！
        broadcast_portfolio_update()

    def on_refresh_clicked(self):
        now = time_module.time()
        if now - self.last_refresh_time < 3.0: return # 3秒冷卻防護
        self.last_refresh_time = now
        
        self.btn_refresh.setText("⏳ 抓取快照中...")
        self.btn_refresh.setEnabled(False)
        QApplication.processEvents()
        
        # 背景執行向永豐金要快照
        threading.Thread(target=self.fetch_snapshots, daemon=True).start()

    def fetch_snapshots(self):
        try:
            with sys_state.lock:
                symbols = list(sys_state.open_positions.keys())
            if not symbols: 
                broadcast_portfolio_update()
                return

            # 🚀 修正：正確地從 Shioaji 取得 上市(TSE) 或 上櫃(OTC) 的合約
            contracts = []
            for s in symbols:
                try:
                    # 先找上市，找不到再找上櫃
                    c = sys_state.api.Contracts.Stocks.TSE.get(s) or sys_state.api.Contracts.Stocks.OTC.get(s)
                    if c:
                        contracts.append(c)
                except Exception as e:
                    logger.warning(f"找不到 {s} 的合約: {e}")

            if not contracts:
                logger.warning("無法取得任何有效的股票合約，取消快照請求")
                broadcast_portfolio_update()
                return

            snapshots = sys_state.api.snapshots(contracts)
            snap_map = {s.code: s.close for s in snapshots}

            with sys_state.lock:
                for sym in symbols:
                    pos = sys_state.open_positions[sym]
                    s_price = snap_map.get(sym, 0)

                    # 🚀 修正：如果快照沒給價格 (非交易時段)
                    if not s_price or s_price == 0:
                        try:
                            # 🎯 改為查詢該股票在資料庫中的「最後一筆紀錄」，不分日期
                            # 這會抓到最近一個交易日的 13:30 收盤價
                            res = sys_db.conn.execute(
                                "SELECT close FROM intraday_kline_history WHERE symbol=? ORDER BY date DESC, time DESC LIMIT 1",
                                (sym,)
                            ).fetchone()
                            
                            if res:
                                pos['current_price'] = float(res[0])
                                print(f"📊 [快照補償] {sym} 抓不到快照，改用資料庫最新歷史價: {res[0]}")
                            else:
                                pos['current_price'] = 0 
                        except Exception as e:
                            logger.warning(f"讀取資料庫保底價失敗: {e}")
                            pos['current_price'] = 0
                    else:
                        pos['current_price'] = s_price
            broadcast_portfolio_update()
        except Exception as e:
            logger.error(f"背景快照執行緒錯誤: {e}", exc_info=True)

    @pyqtSlot(list)
    def update_table(self, data_list):
        # 恢復按鈕狀態
        self.btn_refresh.setText("🔄 快照刷新 (3秒冷卻)")
        self.btn_refresh.setEnabled(True)
        
        # 🚀 空手優化：沒持倉時顯示 (Span 也要改為 6 欄)
        if not data_list:
            self.table.setRowCount(1)
            self.table.setSpan(0, 0, 1, 6)
            item = QTableWidgetItem("目前無已成交之持倉")
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(QColor("#AAAAAA"))
            self.table.setItem(0, 0, item)
            return
            
        self.table.clearSpans()
        self.table.setRowCount(len(data_list))
        
        for r, d in enumerate(data_list):
            # 0. 標的 (顯示名稱+代號)
            self.table.setItem(r, 0, QTableWidgetItem(f"{sn(d['symbol'])}"))
            
            # 1. 進場均價
            self.table.setItem(r, 1, QTableWidgetItem(f"{d['entry_price']:.2f}"))
            
            # 2. 最新快照價
            self.table.setItem(r, 2, QTableWidgetItem(f"{d['current_price']:.2f}"))
            
            # 3. 🚀 張數 (新欄位)
            share_item = QTableWidgetItem(str(d['shares']))
            share_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 3, share_item)
            
            # 4. 預估損益 (賺紅賠綠)
            try:
                pnl = float(d['profit'])
            except Exception:
                pnl = 0.0

            pi = QTableWidgetItem(f"{int(pnl):,}")
            # 根據損益正負給色 (賺錢紅 #FF4136, 賠錢綠 #2ECC40)
            pi.setForeground(QColor("#FF4136" if pnl >= 0 else "#2ECC40"))
            pi.setFont(QFont("Verdana", 10, QFont.Bold))
            pi.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(r, 4, pi)
            
            # 5. 停損價
            sl_val = d['stop_loss']
            sl_str = f"{sl_val:.2f}" if isinstance(sl_val, (int, float)) else str(sl_val)
            self.table.setItem(r, 5, QTableWidgetItem(sl_str))


class LiveTradingPanel(QWidget):
    """合併盤中監控 + 即時持倉的統一分頁（狀態A：參數設定；狀態B：監控中）"""

    def __init__(self):
        super().__init__()
        lo = QVBoxLayout(self); lo.setContentsMargins(0, 0, 0, 0); lo.setSpacing(0)
        self._ctrl_widget = self._build_ctrl()
        self._monitor_widget = self._build_monitor()
        self._monitor_widget.hide()
        lo.addWidget(self._ctrl_widget)
        lo.addWidget(self._monitor_widget)

    # ── 狀態 A：參數輸入（大色塊卡片式） ──
    def _build_ctrl(self):
        ctrl = QWidget()
        outer = QVBoxLayout(ctrl); outer.setContentsMargins(0, 0, 0, 0)
        wrapper = QWidget(); wrapper.setMaximumWidth(600)
        lo = QVBoxLayout(wrapper); lo.setContentsMargins(32, 32, 32, 32); lo.setSpacing(20)

        # 標題
        title = QLabel("盤中監控 — 參數設定")
        title.setStyleSheet(f"color: {TV['text_bright']}; font-size: 17px; font-weight: 700;")
        title.setAlignment(Qt.AlignCenter)
        lo.addWidget(title)

        _card_style = f"""
            QFrame {{
                background: {TV['panel']};
                border: 1px solid {TV['border_light']};
                border-radius: 10px;
            }}
        """
        _label_style = f"color: {TV['text_dim']}; font-size: 12px; font-weight: 600; letter-spacing: 1px;"
        _val_style = f"color: {TV['text_bright']}; font-size: 28px; font-weight: 700;"
        _sub_style = f"color: {TV['text_dim']}; font-size: 12px;"
        _input_style = f"""
            QLineEdit {{
                background: {TV['bg']}; color: {TV['text_bright']};
                border: 1px solid {TV['border_light']}; border-radius: 6px;
                font-size: 24px; font-weight: 700; padding: 6px 10px;
            }}
            QLineEdit:focus {{ border-color: {TV['blue']}; }}
        """

        # 兩個並排大卡片
        cards_row = QHBoxLayout(); cards_row.setSpacing(16)

        def _card(icon, label, default_val, sub_text, attr):
            card = QFrame(); card.setStyleSheet(_card_style); card.setFixedHeight(140)
            cl = QVBoxLayout(card); cl.setContentsMargins(20, 18, 20, 18); cl.setSpacing(6)
            hdr = QLabel(f"{icon}  {label}"); hdr.setStyleSheet(_label_style)
            inp = QLineEdit(default_val); inp.setStyleSheet(_input_style); inp.setAlignment(Qt.AlignCenter)
            sub = QLabel(sub_text); sub.setStyleSheet(_sub_style); sub.setAlignment(Qt.AlignCenter)
            cl.addWidget(hdr); cl.addWidget(inp); cl.addWidget(sub)
            setattr(self, attr, inp)
            return card

        cards_row.addWidget(_card("⏱", "等待時間", "3", "分鐘", "w_wait"))
        cards_row.addWidget(_card("📌", "持有時間", "F", "F = 持到尾盤（13:25）", "w_hold"))
        lo.addLayout(cards_row)

        # ── 永豐金 Shioaji 登入卡片 ──
        _field_style = f"""QLineEdit {{ background: {TV['bg']}; color: {TV['text_bright']}; border: 1px solid {TV['border_light']}; border-radius: 5px; font-size: 11px; padding: 4px 8px; }} QLineEdit:focus {{ border-color: {TV['blue']}; }}"""
        _lbl_s = f"color: {TV['text_dim']}; font-size: 10px;"

        login_card = QFrame(); login_card.setStyleSheet(_card_style)
        lc = QVBoxLayout(login_card); lc.setContentsMargins(20, 16, 20, 16); lc.setSpacing(10)

        lc_hdr = QHBoxLayout()
        lc_title = QLabel("🔑  永豐金 Shioaji 登入")
        lc_title.setStyleSheet(f"color: {TV['text_dim']}; font-size: 12px; font-weight: 600; letter-spacing: 1px;")
        self._live_mode_cb = QCheckBox("正式下單模式")
        self._live_mode_cb.setStyleSheet(f"color: {TV['text_dim']}; font-size: 11px;")
        self._live_mode_cb.setChecked(getattr(sys_config, 'live_trading_mode', False))
        lc_hdr.addWidget(lc_title); lc_hdr.addStretch(); lc_hdr.addWidget(self._live_mode_cb)
        lc.addLayout(lc_hdr)

        login_grid = QGridLayout(); login_grid.setSpacing(6); login_grid.setColumnStretch(1, 1)
        self._e_api_key    = QLineEdit(shioaji_logic.TEST_API_KEY)
        self._e_api_key.setEchoMode(QLineEdit.Password)
        self._e_api_secret = QLineEdit(shioaji_logic.TEST_API_SECRET)
        self._e_api_secret.setEchoMode(QLineEdit.Password)
        self._e_ca_path    = QLineEdit(shioaji_logic.CA_CERT_PATH)
        self._e_ca_pass    = QLineEdit(shioaji_logic.CA_PASSWORD)
        self._e_ca_pass.setEchoMode(QLineEdit.Password)
        for e in [self._e_api_key, self._e_api_secret, self._e_ca_path, self._e_ca_pass]:
            e.setStyleSheet(_field_style)
        lbl_ak = QLabel("API Key");       lbl_ak.setStyleSheet(_lbl_s)
        lbl_as = QLabel("API Secret");    lbl_as.setStyleSheet(_lbl_s)
        lbl_cp = QLabel("CA 憑證路徑");  lbl_cp.setStyleSheet(_lbl_s)
        lbl_cw = QLabel("CA 密碼");       lbl_cw.setStyleSheet(_lbl_s)
        def _mk_eye(le):
            b = QPushButton("\U0001f441"); b.setFixedSize(28, 28); b.setCursor(Qt.PointingHandCursor); b.setCheckable(True)
            b.setStyleSheet("QPushButton { background: transparent; border: none; font-size: 14px; } QPushButton:checked { color: " + TV['blue'] + "; }")
            b.toggled.connect(lambda on: le.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
            return b
        login_grid.addWidget(lbl_ak, 0, 0); login_grid.addWidget(self._e_api_key,    0, 1); login_grid.addWidget(_mk_eye(self._e_api_key), 0, 2)
        login_grid.addWidget(lbl_as, 1, 0); login_grid.addWidget(self._e_api_secret, 1, 1); login_grid.addWidget(_mk_eye(self._e_api_secret), 1, 2)
        login_grid.addWidget(lbl_cp, 2, 0); login_grid.addWidget(self._e_ca_path,    2, 1)
        login_grid.addWidget(lbl_cw, 3, 0); login_grid.addWidget(self._e_ca_pass,    3, 1); login_grid.addWidget(_mk_eye(self._e_ca_pass), 3, 2)
        lc.addLayout(login_grid)

        btn_login = QPushButton("套用並登入")
        btn_login.setFixedHeight(32); btn_login.setCursor(Qt.PointingHandCursor)
        btn_login.setStyleSheet(f"""QPushButton {{ background: {TV['blue']}; color: white; border: none; border-radius: 5px; font-size: 12px; font-weight: 600; }} QPushButton:hover {{ background: {TV['blue_hover']}; }}""")
        btn_login.clicked.connect(self._apply_login)
        lc.addWidget(btn_login)
        lo.addWidget(login_card)

        self._live_mode_cb.toggled.connect(self._on_live_mode_toggled)

        # 啟動按鈕
        btn_start = QPushButton("▶ 啟動盤中監控")
        btn_start.setFixedHeight(50)
        btn_start.setCursor(Qt.PointingHandCursor)
        btn_start.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {TV['green']}, stop:1 #1a7a6e);
                color: white; border: none; border-radius: 10px; font-size: 15px; font-weight: 700;
            }}
            QPushButton:hover {{ background: {TV['green']}; }}
        """)
        btn_start.clicked.connect(self._start_monitoring)
        lo.addWidget(btn_start)

        outer.addStretch(); outer.addWidget(wrapper, 0, Qt.AlignHCenter); outer.addStretch()
        return ctrl

    def _on_live_mode_toggled(self, checked):
        sys_config.live_trading_mode = checked
        self._e_api_key.setText(shioaji_logic.LIVE_API_KEY if checked else shioaji_logic.TEST_API_KEY)
        self._e_api_secret.setText(shioaji_logic.LIVE_API_SECRET if checked else shioaji_logic.TEST_API_SECRET)

    def _apply_login(self):
        is_live = self._live_mode_cb.isChecked()
        ak      = self._e_api_key.text().strip()
        as_     = self._e_api_secret.text().strip()
        ca_path = self._e_ca_path.text().strip()
        ca_pass = self._e_ca_pass.text().strip()

        # 寫回模組屬性
        if is_live:
            shioaji_logic.LIVE_API_KEY    = ak
            shioaji_logic.LIVE_API_SECRET = as_
        else:
            shioaji_logic.TEST_API_KEY    = ak
            shioaji_logic.TEST_API_SECRET = as_
        shioaji_logic.CA_CERT_PATH = ca_path
        shioaji_logic.CA_PASSWORD  = ca_pass
        sys_config.live_trading_mode = is_live

        # 同步寫回 shioaji_logic.py 檔案
        try:
            _base = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
            logic_path = os.path.join(_base, 'shioaji_logic.py')
            lines = [
                f'TEST_API_KEY = "{shioaji_logic.TEST_API_KEY}"\n',
                f'TEST_API_SECRET = "{shioaji_logic.TEST_API_SECRET}"\n',
                f'CA_PASSWORD = "{shioaji_logic.CA_PASSWORD}"\n',
                f'CA_CERT_PATH = r"{shioaji_logic.CA_CERT_PATH}"\n',
                f'LIVE_API_KEY = "{shioaji_logic.LIVE_API_KEY}"\n',
                f'LIVE_API_SECRET = "{shioaji_logic.LIVE_API_SECRET}"\n',
            ]
            with open(logic_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except Exception as e:
            QMessageBox.warning(self._ctrl_widget, "警告", f"無法寫入 shioaji_logic.py：{e}")
            return

        # 嘗試登入
        try:
            sys_state.api = sj.Shioaji(simulation=not is_live)
            sys_state.api.login(api_key=ak, secret_key=as_, contracts_timeout=10000)
            if ca_path and ca_pass:
                sys_state.api.activate_ca(ca_path=ca_path, ca_passwd=ca_pass)
            sys_state.api.set_order_callback(order_callback)  # 🔧 重連後補註冊 callback
            QMessageBox.information(self._ctrl_widget, "成功", "永豐金 API 登入成功！")
        except Exception as e:
            QMessageBox.warning(self._ctrl_widget, "登入失敗", str(e))

    # ── 狀態 B：監控中 ──
    def _build_monitor(self):
        mon = QWidget()
        lo = QVBoxLayout(mon); lo.setContentsMargins(8, 8, 8, 8); lo.setSpacing(6)

        # 頂部狀態列
        hdr = QHBoxLayout()
        self._status_lbl = QLabel("盤中監控執行中")
        self._status_lbl.setStyleSheet(f"color: {TV['green']}; font-size: 13px; font-weight: 700;")
        btn_stop = QPushButton("■ 停止監控")
        btn_stop.setFixedHeight(28)
        btn_stop.setStyleSheet(f"""
            QPushButton {{ background: {TV['surface']}; color: {TV['text']}; border: 1px solid {TV['border_light']}; border-radius: 4px; font-size: 12px; padding: 0 10px; }}
            QPushButton:hover {{ border-color: {TV['red']}; color: {TV['red']}; }}
        """)
        btn_stop.clicked.connect(self._stop_monitoring)
        hdr.addWidget(self._status_lbl); hdr.addStretch(); hdr.addWidget(btn_stop)
        lo.addLayout(hdr)

        # Splitter：終端（上） + 持倉（下）
        splitter = QSplitter(Qt.Vertical)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {TV['border']}; height: 2px; }}")

        # 終端輸出 + K 線圖區域（水平分割）
        self._terminal_splitter = QSplitter(Qt.Horizontal)
        self._terminal_splitter.setStyleSheet(f"QSplitter::handle {{ background: {TV['border']}; width: 2px; }}")

        self._terminal = QTextEdit()
        self._terminal.setReadOnly(True)
        self._terminal.setStyleSheet(f"QTextEdit {{ background: {TV['bg']}; color: {TV['text']}; border: 1px solid {TV['border']}; border-radius: 4px; padding: 8px; font-size: 13px; }}")
        self._terminal_splitter.addWidget(self._terminal)

        # K 線圖容器（2x2 grid，初始隱藏）
        self._chart_container = QWidget()
        self._chart_container.setStyleSheet(f"background: {TV['bg']};")
        self._chart_grid = QGridLayout(self._chart_container)
        self._chart_grid.setContentsMargins(0, 0, 0, 0); self._chart_grid.setSpacing(2)
        self._chart_container.hide()
        self._terminal_splitter.addWidget(self._chart_container)
        self._terminal_splitter.setSizes([1, 0])

        self._chart_slots = [None, None, None, None]  # 最多 4 格
        self._chart_widgets = {}  # sym → widget

        splitter.addWidget(self._terminal_splitter)

        # 持倉區
        port_w = QWidget()
        port_lo = QVBoxLayout(port_w); port_lo.setContentsMargins(0, 4, 0, 0); port_lo.setSpacing(4)
        port_hdr = QHBoxLayout()
        port_title = QLabel("即時持倉")
        port_title.setStyleSheet(f"color: {TV['text_bright']}; font-size: 13px; font-weight: 700;")
        self._btn_refresh = QPushButton("↻ 快照刷新")
        self._btn_refresh.setFixedHeight(26)
        self._btn_refresh.setStyleSheet(f"""
            QPushButton {{ background: {TV['blue']}; color: white; border: none; border-radius: 4px; font-size: 12px; padding: 0 10px; }}
            QPushButton:hover {{ background: {TV['blue_hover']}; }}
        """)
        self._btn_refresh.clicked.connect(self._on_refresh)
        self._last_refresh = 0.0
        port_hdr.addWidget(port_title); port_hdr.addStretch(); port_hdr.addWidget(self._btn_refresh)
        port_lo.addLayout(port_hdr)

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(["股票代號", "進場均價", "現價", "張數", "未實現損益", "停損價", "進場時間", "報酬率"])
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(f"""
            QTableWidget {{ background: {TV['panel']}; alternate-background-color: {TV['surface']}; gridline-color: {TV['border']}; border: 1px solid {TV['border']}; border-radius: 4px; }}
            QTableWidget::item {{ color: {TV['text']}; padding: 5px; }}
            QTableWidget::item:selected {{ background: {TV['blue_dim']}; color: white; }}
            QHeaderView::section {{ background: {TV['surface']}; color: {TV['text_dim']}; font-weight: 700; font-size: 12px; border: none; border-right: 1px solid {TV['border']}; padding: 6px; }}
        """)
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_table_context_menu)
        self._table.doubleClicked.connect(self._on_table_double_click)
        self._table.cellClicked.connect(self._on_cell_clicked)
        port_lo.addWidget(self._table)
        splitter.addWidget(port_w)
        splitter.setSizes([350, 300])
        lo.addWidget(splitter)

        # 手動下單列
        lo.addWidget(self._build_order_bar())

        # 訊號連接
        ui_dispatcher.portfolio_updated.connect(self._update_table)
        ui_dispatcher.console_log.connect(self._append_log)

        # 🔧 3 秒自動快照刷新
        self._snap_timer = QTimer(self)
        self._snap_timer.timeout.connect(self._auto_refresh_snapshot)
        self._snap_timer.start(3000)

        return mon

    def _auto_refresh_snapshot(self):
        """有持倉時自動觸發快照刷新"""
        if not getattr(sys_state, 'open_positions', None):
            return
        if not any(p.get('filled_shares', 0) > 0 for p in sys_state.open_positions.values()):
            return
        self._on_refresh()

    def _build_order_bar(self):
        bar = QFrame()
        bar.setFrameShape(QFrame.StyledPanel)
        bar.setStyleSheet(f"QFrame {{ background: {TV['surface']}; border: 1px solid {TV['border']}; border-radius: 6px; }}")
        bar.setFixedHeight(52)
        lo = QHBoxLayout(bar); lo.setContentsMargins(10, 6, 10, 6); lo.setSpacing(8)

        lbl = QLabel("手動下單：")
        lbl.setStyleSheet(f"color: {TV['text_dim']}; font-size: 12px; font-weight: 600;")

        _input_style = f"QLineEdit {{ background: {TV['bg']}; color: {TV['text']}; border: 1px solid {TV['border_light']}; border-radius: 4px; padding: 3px 6px; font-size: 13px; }}"
        _combo_style = f"QComboBox {{ background: {TV['bg']}; color: {TV['text']}; border: 1px solid {TV['border_light']}; border-radius: 4px; padding: 3px 6px; font-size: 13px; }}"

        self._order_symbol = QLineEdit(); self._order_symbol.setPlaceholderText("代號"); self._order_symbol.setFixedWidth(60); self._order_symbol.setStyleSheet(_input_style)
        self._order_side = QComboBox(); self._order_side.addItems(["買進", "賣出"]); self._order_side.setFixedWidth(70); self._order_side.setView(QListView()); self._order_side.setStyleSheet(_combo_style)
        self._order_price_type = QComboBox(); self._order_price_type.addItems(["指定價", "市價", "漲停", "跌停"]); self._order_price_type.setFixedWidth(72); self._order_price_type.setView(QListView()); self._order_price_type.setStyleSheet(_combo_style)
        self._order_price = QLineEdit(); self._order_price.setPlaceholderText("價格"); self._order_price.setFixedWidth(80); self._order_price.setStyleSheet(_input_style)

        def _on_price_type_change(i):
            labels = {1: ("市價", "#7B2FBE", "white"), 2: ("漲停", TV['red'], "white"), 3: ("跌停", TV['green'], "#131722")}
            if i == 0:
                self._order_price.setReadOnly(False); self._order_price.setText(""); self._order_price.setPlaceholderText("價格")
                self._order_price.setStyleSheet(_input_style)
            else:
                text, bg, fg = labels[i]
                self._order_price.setReadOnly(True); self._order_price.setText(text)
                self._order_price.setStyleSheet(f"QLineEdit {{ background: {bg}; color: {fg}; border: none; border-radius: 4px; padding: 3px 6px; font-size: 13px; font-weight: 700; }}")
        self._order_price_type.currentIndexChanged.connect(_on_price_type_change)
        self._order_qty = QSpinBox(); self._order_qty.setRange(1, 999); self._order_qty.setValue(1); self._order_qty.setFixedWidth(60)
        self._order_qty.setStyleSheet(f"QSpinBox {{ background: {TV['bg']}; color: {TV['text']}; border: 1px solid {TV['border_light']}; border-radius: 4px; padding: 3px 6px; font-size: 13px; }}")

        btn_submit = QPushButton("送出委託"); btn_submit.setFixedHeight(34)
        btn_submit.setStyleSheet(f"QPushButton {{ background: {TV['blue']}; color: white; border: none; border-radius: 5px; font-size: 13px; font-weight: 700; padding: 0 12px; }} QPushButton:hover {{ background: {TV['blue_hover']}; }}")
        btn_submit.clicked.connect(self._submit_order)

        for w in [lbl, self._order_symbol, self._order_side, self._order_price_type, self._order_price, self._order_qty, QLabel("張")]:
            lo.addWidget(w)
        lo.addStretch(); lo.addWidget(btn_submit)
        return bar

    def _start_monitoring(self):
        try: w = int(self.w_wait.text())
        except Exception: QMessageBox.critical(self, "錯誤", "等待時間需為整數"); return
        try: h = None if self.w_hold.text().strip().upper() == 'F' else int(self.w_hold.text().strip())
        except Exception: QMessageBox.critical(self, "錯誤", "持有時間格式錯誤"); return
        if not ensure_esun_passwords(): return
        self._ctrl_widget.hide(); self._monitor_widget.show()
        if _main_window_ref:
            _main_window_ref._show_toast("正在啟動盤中監控...")
            # 留在盤中監控分頁（state B 有自己的終端，不跳轉）
        threading.Thread(target=start_trading, args=('full', w, h), daemon=True).start()

    def _stop_monitoring(self):
        sys_state.stop_trading_flag = True
        self._status_lbl.setText("已發送停止指令，等待本週期結束...")
        self._status_lbl.setStyleSheet(f"color: {TV['yellow']}; font-size: 13px; font-weight: 700;")

    def _on_refresh(self):
        import time as _t
        now = _t.time()
        if now - self._last_refresh < 3.0: return
        self._last_refresh = now
        threading.Thread(target=broadcast_portfolio_update, daemon=True).start()

    @pyqtSlot(list)
    def _update_table(self, data_list):
        if not data_list:
            self._table.setRowCount(1); self._table.setSpan(0, 0, 1, 8)
            item = QTableWidgetItem("目前無已成交之持倉"); item.setTextAlignment(Qt.AlignCenter); item.setForeground(QColor("#AAAAAA"))
            self._table.setItem(0, 0, item); return
        self._table.clearSpans()
        total_shares, total_pnl = 0, 0.0
        self._table.setRowCount(len(data_list) + 1)  # +1 for summary row
        for r, d in enumerate(data_list):
            sym_item = QTableWidgetItem(sn(d['symbol'])); sym_item.setData(Qt.UserRole, d['symbol'])
            sym_item.setForeground(QColor(TV['blue']))
            _uf = QFont(); _uf.setUnderline(True); sym_item.setFont(_uf)
            self._table.setItem(r, 0, sym_item)
            self._table.setItem(r, 1, QTableWidgetItem(f"{d['entry_price']:.2f}"))
            self._table.setItem(r, 2, QTableWidgetItem(f"{d['current_price']:.2f}"))
            si = QTableWidgetItem(str(d['shares'])); si.setTextAlignment(Qt.AlignCenter); self._table.setItem(r, 3, si)
            try: pnl = float(d['profit'])
            except Exception: pnl = 0.0
            pi = QTableWidgetItem(f"{int(pnl):,}"); pi.setForeground(QColor("#FF4136" if pnl >= 0 else "#2ECC40"))
            pi.setFont(QFont("Verdana", 10, QFont.Bold)); pi.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter); self._table.setItem(r, 4, pi)
            sl = d['stop_loss']; self._table.setItem(r, 5, QTableWidgetItem(f"{sl:.2f}" if isinstance(sl, (int, float)) else str(sl)))
            et = d.get('entry_time', ''); self._table.setItem(r, 6, QTableWidgetItem(et[:5] if et else '--'))
            rp = d.get('return_pct', 0.0)
            rp_item = QTableWidgetItem(f"{rp:+.2f}%"); rp_item.setForeground(QColor("#FF4136" if rp >= 0 else "#2ECC40"))
            rp_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter); self._table.setItem(r, 7, rp_item)
            total_shares += d['shares']; total_pnl += pnl
        # 加總列
        sr = len(data_list)
        _sum_bg = QColor(TV['surface'])
        _sum_fg = QColor(TV['text_dim'])
        for c in range(8):
            ci = QTableWidgetItem(""); ci.setBackground(_sum_bg); ci.setFlags(ci.flags() & ~Qt.ItemIsSelectable)
            self._table.setItem(sr, c, ci)
        hi = QTableWidgetItem("合計"); hi.setBackground(_sum_bg); hi.setForeground(_sum_fg); hi.setFont(QFont("Verdana", 10, QFont.Bold))
        self._table.setItem(sr, 0, hi)
        ts = QTableWidgetItem(str(total_shares)); ts.setBackground(_sum_bg); ts.setForeground(_sum_fg); ts.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(sr, 3, ts)
        tp = QTableWidgetItem(f"{int(total_pnl):,}"); tp.setBackground(_sum_bg)
        tp.setForeground(QColor("#FF4136" if total_pnl >= 0 else "#2ECC40"))
        tp.setFont(QFont("Verdana", 10, QFont.Bold)); tp.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._table.setItem(sr, 4, tp)

    @pyqtSlot(str)
    def _append_log(self, text):
        """接收終端輸出，顯示在監控終端（過濾 ANSI 轉義碼）"""
        import re
        clean = re.sub(r'\x1b\[[0-9;]*m', '', text)
        self._terminal.moveCursor(self._terminal.textCursor().End)
        self._terminal.insertPlainText(clean)
        self._terminal.moveCursor(self._terminal.textCursor().End)

    def _get_selected_symbol(self):
        row = self._table.currentRow()
        if row < 0: return None
        item = self._table.item(row, 0)
        return (item.data(Qt.UserRole) or item.text()) if item else None

    def _on_cell_clicked(self, row, col):
        """左鍵點擊代號欄 → 自動填入代號、現價、張數到手動下單列"""
        if col != 0: return  # 只有代號欄才觸發
        item0 = self._table.item(row, 0)
        if not item0: return
        sym = item0.data(Qt.UserRole)
        if not sym: return  # 可能點到加總列
        self._order_symbol.setText(sym)
        self._order_side.setCurrentIndex(0)  # 買進（回補做空）
        # 只有「指定價」模式才填入現價；市價/漲停/跌停保持原選項不動
        if self._order_price_type.currentText() == "指定價":
            price_item = self._table.item(row, 2)
            if price_item:
                try:
                    p = float(price_item.text())
                    if p > 0:
                        self._order_price.setText(price_item.text())
                except Exception: pass
        qty_item = self._table.item(row, 3)
        if qty_item:
            try:
                qty = int(qty_item.text())
                if qty > 0:
                    self._order_qty.setValue(qty)
            except Exception: pass

    def _on_table_context_menu(self, pos):
        row = self._table.rowAt(pos.y())
        if row < 0: return
        # 忽略加總列
        if row >= self._table.rowCount() - 1:
            item0 = self._table.item(row, 0)
            if item0 and not item0.data(Qt.UserRole): return
        self._table.selectRow(row)
        sym = self._get_selected_symbol()
        if not sym: return
        menu = QMenu()
        menu.setStyleSheet(f"QMenu {{ background: {TV['panel']}; color: {TV['text']}; border: 1px solid {TV['border_light']}; font-size: 13px; }} QMenu::item {{ padding: 6px 20px; }} QMenu::item:selected {{ background: {TV['blue']}; color: white; }}")
        act_kline = menu.addAction(f"📈 開啟即時 K 線圖（{sn(sym)}）")
        menu.addSeparator()
        act_buy = menu.addAction("🟢 手動買進"); act_sell = menu.addAction("🔴 手動賣出")
        menu.addSeparator()
        act_copy = menu.addAction("📋 複製持倉資訊")
        action = menu.exec_(self._table.viewport().mapToGlobal(pos))
        if action == act_kline: self._embed_kline(sym)
        elif action == act_buy: self._order_symbol.setText(sym); self._order_side.setCurrentIndex(0)
        elif action == act_sell: self._order_symbol.setText(sym); self._order_side.setCurrentIndex(1)
        elif action == act_copy:
            r = self._table.currentRow()
            ep = self._table.item(r, 1).text() if self._table.item(r, 1) else ""
            cp = self._table.item(r, 2).text() if self._table.item(r, 2) else ""
            sh = self._table.item(r, 3).text() if self._table.item(r, 3) else ""
            pn = self._table.item(r, 4).text() if self._table.item(r, 4) else ""
            txt = f"{sn(sym)} | 進場{ep} | 現價{cp} | {sh}張 | 損益{pn}"
            QApplication.clipboard().setText(txt)

    def _on_table_double_click(self, _index):
        sym = self._get_selected_symbol()
        if sym: self._embed_kline(sym)

    def _open_live_kline(self, sym):
        if _main_window_ref:
            _main_window_ref._ensure_tab(f'live_kline_{sym}', f'K線 {sn(sym)}', lambda s=sym: LiveKlineWidget(s))

    # ── 嵌入式 K 線圖管理 ──

    def _embed_kline(self, sym):
        """在終端右側嵌入 K 線圖（最多 2x2 = 4 格）"""
        if sym in self._chart_widgets:
            return  # 已存在

        slot = next((i for i, s in enumerate(self._chart_slots) if s is None), None)
        if slot is None:
            # 4 格已滿，替換最早的一格
            oldest = self._chart_slots[0]
            self._remove_chart(oldest)
            self._chart_slots = self._chart_slots[1:] + [None]
            slot = 3

        widget = self._create_mini_kline(sym)
        self._chart_slots[slot] = sym
        self._chart_widgets[sym] = widget

        row, col = divmod(slot, 2)
        self._chart_grid.addWidget(widget, row, col)

        self._chart_container.show()
        self._terminal_splitter.setSizes([500, 500])

    def _remove_chart(self, sym):
        """關閉並移除一個 K 線面板"""
        if sym not in self._chart_widgets: return
        w = self._chart_widgets.pop(sym)
        if hasattr(w, '_timer'): w._timer.stop()
        idx = self._chart_slots.index(sym)
        self._chart_slots[idx] = None
        self._chart_grid.removeWidget(w)
        w.deleteLater()

        if all(s is None for s in self._chart_slots):
            self._chart_container.hide()
            self._terminal_splitter.setSizes([1, 0])

    def _create_mini_kline(self, sym):
        """建立帶關閉按鈕的精簡 K 線面板"""
        frame = QWidget()
        frame.setStyleSheet(f"background: {TV['bg']};")
        vlo = QVBoxLayout(frame); vlo.setContentsMargins(0, 0, 0, 0); vlo.setSpacing(0)

        hdr = QHBoxLayout(); hdr.setContentsMargins(4, 2, 4, 0)
        title = QLabel(sn(sym))
        title.setStyleSheet(f"color: {TV['text_bright']}; font-size: 11px; font-weight: 600;")
        btn_close = QPushButton("✕")
        btn_close.setFixedSize(20, 20); btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setStyleSheet(f"QPushButton {{ background: transparent; color: #888; border: none; font-size: 12px; }} QPushButton:hover {{ color: {TV['red']}; }}")
        _sym = sym  # capture
        btn_close.clicked.connect(lambda _, s=_sym: self._remove_chart(s))
        hdr.addWidget(title); hdr.addStretch(); hdr.addWidget(btn_close)
        vlo.addLayout(hdr)

        from PyQt5.QtWebEngineWidgets import QWebEngineView
        web = QWebEngineView()
        vlo.addWidget(web)
        frame._web = web
        frame._symbol = sym

        self._load_mini_chart(web, sym)

        timer = QTimer(frame)
        timer.timeout.connect(lambda s=_sym: self._load_mini_chart(frame._web, s))
        timer.start(60000)
        frame._timer = timer

        return frame

    def _load_mini_chart(self, web, sym):
        """載入 lightweight-charts K 線圖到 QWebEngineView"""
        import json as _json
        today_str = datetime.now().strftime('%Y-%m-%d')
        entry_price = 0.0

        # 1. 安全取得 bars（用 is not None，避免空 dict {} 被當作 falsy）
        with sys_state.lock:
            _intraday = sys_state.in_memory_intraday if (hasattr(sys_state, 'in_memory_intraday') and sys_state.in_memory_intraday is not None) else {}
            bars = list(_intraday.get(sym, []))
            if sym in sys_state.open_positions:
                entry_price = float(sys_state.open_positions[sym].get('entry_price', 0))

        # 2. 若記憶體無資料，嘗試從 DB 讀取
        if not bars:
            try:
                db_data = sys_db.load_kline('intraday_kline_live')
                bars = list(db_data.get(sym, []))
            except Exception: pass

        # 3. 仍無資料 → 顯示提示並 early return
        if not bars:
            web.setHtml(f"<html><body style='background:#131722;color:#D1D4DC;display:flex;align-items:center;justify-content:center;height:100vh;font-family:Microsoft JhengHei'><h2>尚無即時資料（{sn(sym)}）</h2></body></html>")
            return

        # 4. 解析 OHLC（同時嘗試 'time' 和 'ts' key，與 LiveKlineWidget 一致）
        ohlc_data, vol_data = [], []
        for b in bars:
            try:
                t_str = b.get('time', b.get('ts', ''))
                if not t_str: continue
                t_str = str(t_str)
                if len(t_str) <= 8:
                    t_str = f"{today_str} {t_str}"
                dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
                ts = int(dt.timestamp())
                ohlc_data.append({"time": ts, "open": b.get('open', 0), "high": b.get('high', 0), "low": b.get('low', 0), "close": b.get('close', 0)})
                vol_data.append({"time": ts, "value": b.get('volume', 0), "color": "#26a69a" if b.get('close', 0) >= b.get('open', 0) else "#ef5350"})
            except Exception: continue

        if not ohlc_data:
            web.setHtml(f"<html><body style='background:#131722;color:#D1D4DC;display:flex;align-items:center;justify-content:center;height:100vh;font-family:Microsoft JhengHei'><h2>資料解析失敗（{sn(sym)}）</h2></body></html>")
            return

        # 5. 進場線
        entry_js = ""
        if entry_price > 0 and ohlc_data:
            t0, t1 = ohlc_data[0]['time'], ohlc_data[-1]['time']
            entry_js = f"const el=chart.addLineSeries({{color:'#2196F3',lineWidth:2,lineStyle:1,lastValueVisible:true,priceLineVisible:false}});el.setData([{{time:{t0},value:{entry_price}}},{{time:{t1},value:{entry_price}}}]);"

        html = f"""<!DOCTYPE html><html><head>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>*{{margin:0;padding:0;box-sizing:border-box;}}body{{background:#131722;overflow:hidden;}}
#chart{{width:100vw;height:100vh;}}</style></head><body>
<div id="chart"></div><script>
const chart=LightweightCharts.createChart(document.getElementById('chart'),{{
  layout:{{background:{{color:'#131722'}},textColor:'#D1D4DC'}},
  grid:{{vertLines:{{color:'#1e2230'}},horzLines:{{color:'#1e2230'}}}},
  crosshair:{{mode:LightweightCharts.CrosshairMode.Normal}},
  timeScale:{{timeVisible:true,secondsVisible:false}},
  rightPriceScale:{{borderColor:'#2a2e39'}},
}});
const cs=chart.addCandlestickSeries({{upColor:'#26a69a',downColor:'#ef5350',borderVisible:false,wickUpColor:'#26a69a',wickDownColor:'#ef5350'}});
cs.setData({_json.dumps(ohlc_data)});
{entry_js}
chart.timeScale().fitContent();
new ResizeObserver(()=>chart.resize(window.innerWidth,window.innerHeight)).observe(document.body);
</script></body></html>"""
        web.setHtml(html)

    def _submit_order(self):
        sym = self._order_symbol.text().strip()
        if not sym: QMessageBox.warning(self, "錯誤", "請輸入股票代號"); return

        if not getattr(sys_state, 'api', None) or not hasattr(sys_state.api, 'Contracts'):
            QMessageBox.warning(self, "錯誤", "Shioaji 尚未登入，無法下單"); return

        side = self._order_side.currentText()
        price_type = self._order_price_type.currentText()
        qty = self._order_qty.value()

        try:
            contract = sys_state.api.Contracts.Stocks[sym]
            if not contract: raise KeyError(sym)
        except Exception:
            QMessageBox.warning(self, "錯誤", f"找不到股票合約 {sym}"); return

        action = sj.constant.Action.Buy if side == "買進" else sj.constant.Action.Sell

        if price_type == "指定價":
            try: price = float(self._order_price.text())
            except Exception: QMessageBox.warning(self, "錯誤", "請輸入有效價格"); return
            sj_price_type = sj.constant.StockPriceType.LMT
            sj_order_type = sj.constant.OrderType.ROD
            price_desc = f"{price}"
        elif price_type == "市價":
            price = 0
            sj_price_type = sj.constant.StockPriceType.MKT
            sj_order_type = sj.constant.OrderType.IOC
            price_desc = "市價"
        elif price_type == "漲停":
            price = contract.limit_up
            sj_price_type = sj.constant.StockPriceType.LMT
            sj_order_type = sj.constant.OrderType.ROD
            price_desc = f"漲停 {price}"
        else:  # 跌停
            price = contract.limit_down
            sj_price_type = sj.constant.StockPriceType.LMT
            sj_order_type = sj.constant.OrderType.ROD
            price_desc = f"跌停 {price}"

        reply = QMessageBox.question(self, "確認下單",
            f"確定要 {side} {sn(sym)} {qty}張 @ {price_desc}？\n此為真實委託，確認後無法自動撤銷。",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes: return

        try:
            order = sys_state.api.Order(
                price=price, quantity=qty, action=action,
                price_type=sj_price_type, order_type=sj_order_type,
                order_lot=sj.constant.StockOrderLot.Common,
                account=sys_state.api.stock_account)
            trade = safe_place_order(sys_state.api, contract, order)
            if trade:
                print(f"[手動委託成功] {side} {sn(sym)} {qty}張 @ {price_desc}")
                QMessageBox.information(self, "委託成功", f"{side} {sn(sym)} {qty}張 已送出")
            else:
                QMessageBox.warning(self, "委託失敗", "下單未成功，請檢查連線狀態與帳戶餘額")
        except Exception as e:
            QMessageBox.critical(self, "委託異常", f"下單發生錯誤：{e}")


class LiveKlineWidget(QWidget):
    """即時 1 分 K 線圖（lightweight-charts，數據來自 in_memory_intraday）"""

    def __init__(self, symbol):
        super().__init__()
        self._symbol = symbol
        lo = QVBoxLayout(self); lo.setContentsMargins(0, 0, 0, 0); lo.setSpacing(0)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(8, 6, 8, 4)
        title = QLabel(f"📈  {sn(symbol)}  即時 1 分 K")
        title.setStyleSheet(f"color: {TV['text_bright']}; font-size: 14px; font-weight: 700;")
        btn_refresh = QPushButton("↻ 更新")
        btn_refresh.setFixedHeight(28)
        btn_refresh.setStyleSheet(f"QPushButton {{ background: {TV['blue']}; color: white; border: none; border-radius: 4px; font-size: 12px; padding: 0 10px; }} QPushButton:hover {{ background: {TV['blue_hover']}; }}")
        btn_refresh.clicked.connect(self._load_chart)
        hdr.addWidget(title); hdr.addStretch(); hdr.addWidget(btn_refresh)
        lo.addLayout(hdr)

        from PyQt5.QtWebEngineWidgets import QWebEngineView
        self._web = QWebEngineView()
        lo.addWidget(self._web)

        self._load_chart()
        self._timer = QTimer(self); self._timer.timeout.connect(self._load_chart); self._timer.start(60000)

    def _load_chart(self):
        import json as _json
        sym = self._symbol
        with sys_state.lock:
            bars = list(sys_state.in_memory_intraday.get(sym, []))
        with sys_state.lock:
            pos = dict(sys_state.open_positions.get(sym, {}))
        entry_price = pos.get('entry_price', 0)

        if not bars:
            self._web.setHtml(f"<html><body style='background:#131722;color:#D1D4DC;display:flex;align-items:center;justify-content:center;height:100vh;font-family:Microsoft JhengHei'><h2>尚無即時資料（{sn(sym)}）</h2></body></html>")
            return

        today_str = datetime.now().strftime("%Y-%m-%d")
        ohlc_data, vol_data = [], []
        for b in bars:
            try:
                t_str = b.get('time', b.get('ts', ''))
                if not t_str: continue
                if len(t_str) <= 8:  # "HH:MM:SS"
                    t_str = f"{today_str} {t_str}"
                dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
                ts = int(dt.timestamp())
                ohlc_data.append({"time": ts, "open": b.get('open', 0), "high": b.get('high', 0), "low": b.get('low', 0), "close": b.get('close', 0)})
                vol_data.append({"time": ts, "value": b.get('volume', 0), "color": "#26a69a" if b.get('close', 0) >= b.get('open', 0) else "#ef5350"})
            except Exception: continue

        entry_js = ""
        if entry_price > 0 and ohlc_data:
            t0, t1 = ohlc_data[0]['time'], ohlc_data[-1]['time']
            entry_js = f"""
            const entryLine = chart.addLineSeries({{color:'#2196F3',lineWidth:2,lineStyle:1,lastValueVisible:true,priceLineVisible:false}});
            entryLine.setData([{{time:{t0},value:{entry_price}}},{{time:{t1},value:{entry_price}}}]);
            """

        html = f"""<!DOCTYPE html><html><head>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>*{{margin:0;padding:0;box-sizing:border-box;}}body{{background:#131722;overflow:hidden;}}
#chart{{width:100vw;height:calc(100vh - 0px);}}</style></head><body>
<div id="chart"></div><script>
const chart=LightweightCharts.createChart(document.getElementById('chart'),{{
  layout:{{background:{{color:'#131722'}},textColor:'#D1D4DC'}},
  grid:{{vertLines:{{color:'#1e2230'}},horzLines:{{color:'#1e2230'}}}},
  crosshair:{{mode:LightweightCharts.CrosshairMode.Normal}},
  timeScale:{{timeVisible:true,secondsVisible:false}},
  rightPriceScale:{{borderColor:'#2a2e39'}},
}});
const cs=chart.addCandlestickSeries({{upColor:'#26a69a',downColor:'#ef5350',borderVisible:false,wickUpColor:'#26a69a',wickDownColor:'#ef5350'}});
cs.setData({_json.dumps(ohlc_data)});
{entry_js}
chart.timeScale().fitContent();
new ResizeObserver(()=>chart.resize(window.innerWidth,window.innerHeight)).observe(document.body);
</script></body></html>"""
        self._web.setHtml(html)


def build_cashflow_html(all_trades: list, all_events: list = None) -> str:
    """資金流動走勢圖（Plotly HTML）。進場=綠，獲利出場=藍，停損出場=紅。"""
    import json as _json
    from collections import defaultdict as _dd
    if not all_trades:
        return "<html><body style='background:#1E1E1E;color:white;display:flex;align-items:center;justify-content:center;height:100vh;font-family:Microsoft JhengHei'><h2>尚無交易紀錄</h2></body></html>"
    by_date = _dd(list)
    for t in all_trades:
        by_date[t.get('date','未知')].append(t)
    dates = sorted(by_date.keys())
    fee_rate = getattr(sys_config,'transaction_fee',0.1425)*0.01*getattr(sys_config,'transaction_discount',18.0)*0.01
    tax_rate = getattr(sys_config,'trading_tax',0.15)*0.01
    capital  = getattr(sys_config,'capital_per_stock',200)
    all_fig  = {}
    for d in dates:
        trades = by_date[d]
        evs = []
        for t in trades:
            ep = float(t.get('entry_price',0) or 0)
            xp = float(t.get('exit_price',0)  or 0)
            sh = max(1, round((capital*10000)/(ep*1000))) if ep>0 else 0
            eg = ep*sh*1000; ef = int(eg*fee_rate); ed = eg-ef
            xg = xp*sh*1000; xf = int(xg*fee_rate); xt = int(xg*tax_rate); xd = -(xg+xf+xt)
            is_stop = '停損' in t.get('reason','')
            sym = t.get('symbol','')
            evs.append({'time':(t.get('entry_time') or '09:00')[:5],'delta':ed,'type':'entry','sym':sym,'price':ep,'shares':sh,'trigger':t.get('trigger_type','')})
            evs.append({'time':(t.get('exit_time')  or '13:30')[:5],'delta':xd,'type':'stop' if is_stop else 'exit','sym':sym,'price':xp,'shares':sh,'reason':t.get('reason',''),'profit':t.get('profit',0)})
        evs.sort(key=lambda e:e['time'])
        merged={}
        for ev in evs:
            k=ev['time']
            if k not in merged: merged[k]={'time':k,'delta':0,'details':[],'types':set()}
            merged[k]['delta']+=ev['delta']; merged[k]['details'].append(ev); merged[k]['types'].add(ev['type'])
        pts = sorted(merged.values(),key=lambda x:x['time'])
        cum=0; xs,ys,clrs,txts,hvrs=[],[],[],[],[]
        for pt in pts:
            cum+=pt['delta']; xs.append(pt['time']); ys.append(round(cum))
            dv=pt['delta']; typ=pt['types']
            clrs.append('#E74C3C' if 'stop' in typ else '#2ECC71' if 'entry' in typ else '#3498DB')
            txts.append(f"{'+' if dv>=0 else ''}{int(dv):,}")
            lines=[f"<b>{pt['time']}</b><br>累積: {int(cum):,} 元<br>增減: {'+' if dv>=0 else ''}{int(dv):,} 元<br>───"]
            for ev in pt['details']:
                if ev['type']=='entry':
                    lines.append(f"▲ 進場 {ev['sym']} ×{ev.get('shares',0)}張 @{ev.get('price',0):.2f} ({ev.get('trigger','')})")
                else:
                    p=ev.get('profit',0); lines.append(f"{'▼ 停損' if ev['type']=='stop' else '● 出場'} {ev['sym']} ×{ev.get('shares',0)}張 @{ev.get('price',0):.2f} | 損益:{'+' if p>=0 else ''}{int(p):,} ({ev.get('reason','')})")
            hvrs.append('<br>'.join(lines))
        all_fig[d]={'x':xs,'y':ys,'colors':clrs,'text':txts,'hover':hvrs}
    dj=_json.dumps(dates,ensure_ascii=False); fj=_json.dumps(all_fig,ensure_ascii=False)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{margin:0;background:#1E1E1E;color:#E0E0E0;font-family:"Microsoft JhengHei",sans-serif}}
#ctrl{{padding:10px 16px;background:#252525;border-bottom:1px solid #333;display:flex;align-items:center;gap:12px}}
#ctrl label{{font-weight:bold;font-size:13px}}#date-sel{{background:#2C2C2C;color:#E0E0E0;border:1px solid #555;padding:4px 10px;border-radius:4px;font-size:13px}}
#pw{{width:100%;height:calc(100vh - 48px)}}</style></head>
<body>
<div id="ctrl"><label>選擇日期：</label><select id="date-sel" onchange="go(this.value)"></select>
<span id="smr" style="margin-left:auto;font-size:12px;color:#aaa"></span></div>
<div id="pw"><div id="plt" style="width:100%;height:100%"></div></div>
<script>
const D={dj},F={fj};
function go(d){{const f=F[d];if(!f)return;
  Plotly.react('plt',[{{x:f.x,y:f.y,mode:'lines+markers+text',type:'scatter',
    marker:{{color:f.colors,size:10,line:{{color:'#fff',width:1}}}},
    line:{{color:'#555',width:1.5}},text:f.text,textposition:'top center',
    textfont:{{size:11,color:f.colors}},hovertext:f.hover,hoverinfo:'text',name:''}}],
  {{title:{{text:d+' 資金流動走勢',font:{{color:'#E0E0E0',size:16}},x:0.5}},
    paper_bgcolor:'#1E1E1E',plot_bgcolor:'#252525',
    xaxis:{{title:'時間',color:'#aaa',gridcolor:'#333',tickfont:{{color:'#ccc'}}}},
    yaxis:{{title:'累積資金流量 (元)',color:'#aaa',gridcolor:'#333',tickfont:{{color:'#ccc'}},
            zeroline:true,zerolinecolor:'#888',zerolinewidth:1.5}},
    margin:{{t:60,l:90,r:40,b:60}},hovermode:'closest',showlegend:false}},
  {{responsive:false,displayModeBar:false}}).catch(function(){{}});
  const tot=f.y.length>0?f.y[f.y.length-1]:0;
  document.getElementById('smr').textContent='累積資金流: '+(tot>=0?'+':'')+tot.toLocaleString()+' 元';
}}
window.addEventListener('resize',function(){{Plotly.Plots.resize('plt');}});
const sel=document.getElementById('date-sel');
D.forEach(d=>{{const o=document.createElement('option');o.value=d;o.text=d;sel.appendChild(o);}});
if(D.length>0){{sel.value=D[0];go(D[0]);}}
</script></body></html>"""

# ==================== 主視窗 (MainWindow) ====================
class QuantMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"REMORA  v{APP_VERSION}  │  台灣當沖量化平台")
        self.resize(1300, 820)
        self.setMinimumSize(1024, 660)

        # 視窗 + 工具列 Icon
        _icon_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        _icon_path = os.path.join(_icon_dir, 'remora.ico')
        if os.path.exists(_icon_path):
            self.setWindowIcon(QIcon(_icon_path))

        # ── 全域主題 ──
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {TV['bg']}; }}
            QWidget {{ background-color: {TV['bg']}; color: {TV['text']};
                       font-family: 'Segoe UI', '微軟正黑體', sans-serif; }}
            QSplitter::handle {{ background-color: {TV['border']}; }}
            QToolTip {{ background-color: {TV['panel']}; color: {TV['text']};
                        border: 1px solid {TV['border_light']}; padding: 4px 8px; }}
            QTabWidget::pane {{ background-color: {TV['bg']}; border: none; }}
            QTabBar {{ background-color: {TV['panel']}; border-bottom: 1px solid {TV['border']}; }}
            QTabBar::tab {{ background-color: transparent; color: {TV['text_dim']}; padding: 8px 20px;
                            border: none; border-bottom: 2px solid transparent; border-right: 1px solid {TV['border']};
                            font-size: 13px; font-weight: 600; min-width: 50px; margin-right: 1px; }}
            QTabBar::tab:selected {{ color: {TV['text_bright']}; border-bottom-color: {TV['blue']}; background-color: {TV['surface']}; }}
            QTabBar::tab:hover:!selected {{ color: {TV['text']}; background-color: {TV['surface']}; }}
            QTabBar::close-button {{ subcontrol-position: right; padding: 2px; }}
        """)

        ui_dispatcher.show_analysis_window.connect(self.open_volume_analysis_window)
        self.analysis_window_instance = None

        # ── 根容器：垂直堆疊 Header / Body / StatusBar ──
        root = QWidget(); self.setCentralWidget(root)
        root_vbox = QVBoxLayout(root)
        root_vbox.setSpacing(0); root_vbox.setContentsMargins(0, 0, 0, 0)

        # ═══════════════════════════════════════════════════════
        # ① HEADER BAR
        # ═══════════════════════════════════════════════════════
        header = QFrame(); header.setFixedHeight(42)
        header.setStyleSheet(f"QFrame {{ background-color: {TV['panel']}; border-bottom: 1px solid {TV['border']}; }}")
        h_lo = QHBoxLayout(header); h_lo.setContentsMargins(16, 0, 16, 0); h_lo.setSpacing(14)

        # Logo
        logo_icon = QLabel("▲"); logo_icon.setStyleSheet(f"color: {TV['blue']}; font-size: 22px; font-weight: 900;")
        logo_text = QLabel("REMORA"); logo_text.setStyleSheet(f"color: {TV['text_bright']}; font-size: 19px; font-weight: 800; letter-spacing: 2px;")
        logo_sub  = QLabel("TERMINAL"); logo_sub.setStyleSheet(f"color: {TV['text_dim']}; font-size: 12px; font-weight: 600; letter-spacing: 2px; margin-top: 3px;")
        h_lo.addWidget(logo_icon); h_lo.addWidget(logo_text); h_lo.addWidget(logo_sub)

        def _vsep():
            f = QFrame(); f.setFrameShape(QFrame.VLine); f.setFixedHeight(22)
            f.setStyleSheet(f"color: {TV['border']};"); return f

        h_lo.addStretch()

        # 時鐘
        self.hdr_clock = QLabel()
        self.hdr_clock.setStyleSheet(f"color: {TV['text']}; font-size: 14px; font-weight: 600; font-family: 'Cascadia Code','Consolas',monospace;")
        h_lo.addWidget(self.hdr_clock)
        root_vbox.addWidget(header)

        # ═══════════════════════════════════════════════════════
        # ①.5 TOOLBAR（專業風格頂部工具列）
        # ═══════════════════════════════════════════════════════
        toolbar = QFrame(); toolbar.setFixedHeight(36)
        toolbar.setStyleSheet(f"QFrame {{ background-color: {TV['bg']}; border-bottom: 1px solid {TV['border']}; }}")
        tb_lo = QHBoxLayout(toolbar); tb_lo.setContentsMargins(16, 0, 16, 0); tb_lo.setSpacing(8)

        def _tb_btn(text, cb=None, checked=False, accent=False):
            btn = QPushButton(text); btn.setFixedHeight(26); btn.setCursor(Qt.PointingHandCursor)
            bg = TV['blue'] if accent else 'transparent'
            bg_h = TV['blue_hover'] if accent else TV['surface']
            c = TV['text_bright'] if accent else TV['text']
            btn.setStyleSheet(f"""
                QPushButton {{ background-color: {bg}; color: {c}; border: none; border-radius: 4px; font-size: 13px; font-weight: 600; padding: 0 12px; }}
                QPushButton:hover {{ background-color: {bg_h}; color: {TV['text_bright']}; }}
            """)
            if cb: btn.clicked.connect(cb)
            return btn

        def _tb_sep():
            f = QFrame(); f.setFrameShape(QFrame.VLine); f.setFixedHeight(18)
            f.setStyleSheet(f"color: {TV['border_light']};"); return f

        # 市場標籤
        mkt_lbl = QLabel("TWE"); mkt_lbl.setStyleSheet(f"color: {TV['text_dim']}; font-size: 11px; font-weight: 700; letter-spacing: 1px;")
        tb_lo.addWidget(mkt_lbl)
        tb_lo.addWidget(_tb_sep())

        # 模式指示
        self.tb_mode_lbl = QLabel("模擬模式")
        self.tb_mode_lbl.setStyleSheet(f"color: {TV['green']}; font-size: 12px; font-weight: 600;")
        tb_lo.addWidget(self.tb_mode_lbl)
        tb_lo.addWidget(_tb_sep())

        # 快捷按鈕
        tb_lo.addWidget(_tb_btn("📖 新手教學", lambda: self._ensure_tab('tutorial', '新手教學', TutorialWidget)))
        tb_lo.addWidget(_tb_sep())

        # 連線狀態
        self.tb_api_dot = QLabel("●"); self.tb_api_dot.setStyleSheet(f"color: {TV['text_dim']}; font-size: 10px;")
        self.tb_api_lbl = QLabel("API"); self.tb_api_lbl.setStyleSheet(f"color: {TV['text_dim']}; font-size: 11px;")
        tb_lo.addWidget(self.tb_api_dot); tb_lo.addWidget(self.tb_api_lbl)

        tb_lo.addStretch()

        # 股票快搜
        self.tb_search = QLineEdit()
        self.tb_search.setPlaceholderText("搜尋代號/名稱...")
        self.tb_search.setFixedWidth(180); self.tb_search.setFixedHeight(26)
        self.tb_search.setStyleSheet(f"""
            QLineEdit {{ background-color: {TV['surface']}; color: {TV['text']}; border: 1px solid {TV['border_light']};
                         border-radius: 4px; padding: 0 8px; font-size: 12px; }}
            QLineEdit:focus {{ border-color: {TV['blue']}; }}
        """)
        tb_lo.addWidget(self.tb_search)
        root_vbox.addWidget(toolbar)

        # ═══════════════════════════════════════════════════════
        # ② BODY：Sidebar + Console
        # ═══════════════════════════════════════════════════════
        body = QWidget(); body_lo = QHBoxLayout(body)
        body_lo.setSpacing(0); body_lo.setContentsMargins(0, 0, 0, 0)

        # ── 側邊欄 ──
        sidebar = QFrame(); sidebar.setFixedWidth(214)
        sidebar.setStyleSheet(f"QFrame {{ background-color: {TV['panel']}; border-right: 1px solid {TV['border']}; }}")
        sb_vbox = QVBoxLayout(sidebar); sb_vbox.setSpacing(2); sb_vbox.setContentsMargins(10, 14, 10, 14)

        def _section(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {TV['text_dim']}; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; padding: 8px 6px 3px 6px;")
            return lbl

        def _divider():
            f = QFrame(); f.setFrameShape(QFrame.HLine); f.setFixedHeight(1)
            f.setStyleSheet(f"color: {TV['border']}; margin: 5px 0;"); return f

        def _nbtn(icon_key, label, cb, accent=TV['blue'], danger=False):
            btn = QPushButton(f"   {label}")
            btn.setFixedHeight(38); btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(_tv_nav_btn_style(accent, danger))
            # SVG 圖標
            icon_color = TV['red'] if danger else TV['text']
            btn.setIcon(_svg_icon(icon_key, 18, icon_color))
            from PyQt5.QtCore import QSize
            btn.setIconSize(QSize(18, 18))
            btn.clicked.connect(cb); return btn
        

        # ── 側邊欄按鈕（全部改為分頁內嵌）──
        sb_vbox.addWidget(_section("LIVE TRADING"))
        sb_vbox.addWidget(_nbtn("play", "盤中監控", self.start_monitoring_with_login, TV['green']))
        sb_vbox.addWidget(_divider())
        sb_vbox.addWidget(_section("ANALYTICS"))
        sb_vbox.addWidget(_nbtn("analytics", "盤後數據與分析", lambda: self._ensure_tab('analysis_menu', '盤後分析', lambda: AnalysisMenuDialog(self)), TV['blue']))
        sb_vbox.addWidget(_nbtn("crosshair", "自選進場模式",   lambda: self._ensure_tab('simulate', '自選回測', SimulateDialog), TV['blue']))
        sb_vbox.addWidget(_nbtn("bar_chart", "策略參數掃描", lambda: self._ensure_tab('vol_analysis', '參數掃描', VolumeAnalysisDialog), TV['orange']))
        sb_vbox.addWidget(_divider())
        sb_vbox.addWidget(_section("SYSTEM"))
        sb_vbox.addWidget(_nbtn("grid", "管理股票族群",   lambda: self._ensure_tab('group', '族群管理', GroupManagerDialog)))
        sb_vbox.addWidget(_nbtn("key", "帳戶 API 設定",  lambda: self._ensure_tab('login', '帳戶設定', LoginDialog)))
        sb_vbox.addWidget(_nbtn("download", "更新 K 線數據",  lambda: self._ensure_tab('kline', 'K線更新', lambda: self._build_kline_panel())))
        sb_vbox.addWidget(_nbtn("list", "歷史交易紀錄",   lambda: self._ensure_tab('trade_log', '交易紀錄', TradeLogViewerDialog)))
        sb_vbox.addWidget(_nbtn("settings", "系統參數設定",   lambda: self._ensure_tab('settings', '系統設定', SettingsDialog)))
        sb_vbox.addStretch()

        # ══════════════════════════════════════════
        # 分頁主區（取代舊版獨立控制台）
        # ══════════════════════════════════════════
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self._on_tab_close)
        self._tab_pages = {}  # key → QWidget

        # ── Tab 0: 系統日誌（不可關閉）──
        console_page = QWidget()
        cp_lo = QVBoxLayout(console_page); cp_lo.setSpacing(0); cp_lo.setContentsMargins(0, 0, 0, 0)

        # 進度條
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        self.progress_bar.hide(); self.progress_bar.setFixedHeight(3); self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{ border: none; background-color: {TV['surface']}; }}
            QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {TV['blue']}, stop:1 {TV['green']}); }}
        """)
        cp_lo.addWidget(self.progress_bar)

        # 主控台輸出
        self.console = QTextEdit(); self.console.setReadOnly(True)
        self.console.setStyleSheet(f"""
            QTextEdit {{
                background-color: {TV['console_bg']};
                color: {TV['console_text']};
                font-family: 'Cascadia Code', 'Consolas', '新細明體', monospace;
                font-size: 14px; border: none; padding: 10px 14px;
                selection-background-color: {TV['blue_dim']};
            }}
            QScrollBar:vertical {{ background: {TV['console_bg']}; width: 6px; }}
            QScrollBar::handle:vertical {{ background: {TV['surface']}; border-radius: 3px; min-height: 20px; }}
        """)
        cp_lo.addWidget(self.console, stretch=1)

        # 底部工具列：CLEAR 按鈕
        console_toolbar = QFrame(); console_toolbar.setFixedHeight(30)
        console_toolbar.setStyleSheet(f"QFrame {{ background-color: {TV['panel']}; border-top: 1px solid {TV['border']}; }}")
        ct_lo = QHBoxLayout(console_toolbar); ct_lo.setContentsMargins(10, 0, 10, 0)
        btn_clr = QPushButton("CLEAR"); btn_clr.setFixedSize(58, 22); btn_clr.setCursor(Qt.PointingHandCursor)
        btn_clr.setStyleSheet(f"""
            QPushButton {{ background-color: {TV['surface']}; color: {TV['text_dim']}; border: none; border-radius: 3px; font-size: 11px; font-weight: 600; }}
            QPushButton:hover {{ background-color: {TV['border_light']}; color: {TV['text']}; }}
        """)
        btn_clr.clicked.connect(lambda: self.console.clear())
        ct_lo.addStretch(); ct_lo.addWidget(btn_clr)
        cp_lo.addWidget(console_toolbar)

        self.tabs.addTab(console_page, "系統日誌")
        # 第 0 個 Tab 不可關閉
        self.tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)
        self.tabs.tabBar().setTabButton(0, QTabBar.LeftSide, None)

        body_lo.addWidget(sidebar)
        body_lo.addWidget(self.tabs, stretch=1)
        root_vbox.addWidget(body, stretch=1)

        # ═══════════════════════════════════════════════════════
        # ③ STATUS BAR
        # ═══════════════════════════════════════════════════════
        status_bar = QFrame(); status_bar.setFixedHeight(34)
        status_bar.setStyleSheet(f"QFrame {{ background-color: {TV['panel']}; border-top: 1px solid {TV['border']}; }}")
        sb_lo = QHBoxLayout(status_bar); sb_lo.setContentsMargins(14, 0, 14, 0); sb_lo.setSpacing(12)

        def _sb_sep():
            f = QFrame(); f.setFrameShape(QFrame.VLine); f.setFixedHeight(18)
            f.setStyleSheet(f"color: {TV['border']};"); return f

        # 市場狀態
        self.sb_mkt_dot = QLabel("●"); self.sb_mkt_dot.setStyleSheet(f"color: {TV['green']}; font-size: 14px;")
        self.sb_mkt_lbl = QLabel("市場開盤中"); self.sb_mkt_lbl.setStyleSheet(f"color: {TV['green']}; font-size: 13px; font-weight: 600;")
        sb_lo.addWidget(self.sb_mkt_dot); sb_lo.addWidget(self.sb_mkt_lbl)
        sb_lo.addWidget(_sb_sep())

        # 下單模式
        self.sb_mode_lbl = QLabel("模擬模式"); self.sb_mode_lbl.setStyleSheet(f"color: {TV['green']}; font-size: 13px; font-weight: 600;")
        sb_lo.addWidget(self.sb_mode_lbl)
        sb_lo.addWidget(_sb_sep())

        # Telegram 狀態
        self.sb_tg_lbl = QLabel("●  Telegram: 待連線"); self.sb_tg_lbl.setStyleSheet(f"color: {TV['text_dim']}; font-size: 13px;")
        sb_lo.addWidget(self.sb_tg_lbl)

        sb_lo.addStretch()
        sb_ver = QLabel(f"REMORA  v{APP_VERSION}"); sb_ver.setStyleSheet(f"color: {TV['border_light']}; font-size: 12px;")
        root_vbox.addWidget(status_bar)

        # ═══════════════════════════════════════════════════════
        # ④ 訊號連接 + 定時器
        # ═══════════════════════════════════════════════════════
        self.stream = EmittingStream()
        self.stream.textWritten.connect(self.normal_output)
        self.stream.textWritten.connect(ui_dispatcher.console_log)  # 廣播給額外訂閱者
        sys.stdout = self.stream; sys.stderr = self.stream

        ui_dispatcher.progress_updated.connect(lambda p, msg: (self.progress_bar.setValue(p),))
        ui_dispatcher.progress_visible.connect(self.progress_bar.setVisible)
        ui_dispatcher.plot_equity_curve.connect(self.plot_equity)

        # 時鐘 & 狀態輪詢
        self._clock_timer = QTimer(self); self._clock_timer.timeout.connect(self._tick); self._clock_timer.start(1000)
        self._status_timer = QTimer(self); self._status_timer.timeout.connect(self._refresh_status); self._status_timer.start(4000)
        self._tick(); self._refresh_status()

    # ── 時鐘/狀態更新輔助方法 ──
    def _tick(self):
        now = datetime.now()
        self.hdr_clock.setText(now.strftime("  %Y-%m-%d  %H:%M:%S  "))
        t = now.time()
        is_open = now.weekday() < 5 and now.strftime("%Y%m%d") not in _twse_holidays and time(9, 0) <= t <= time(13, 30)
        dot_c = TV['green'] if is_open else TV['text_dim']
        lbl_t = "市場開盤中" if is_open else "市場休市中"
        self.sb_mkt_dot.setStyleSheet(f"color: {dot_c}; font-size: 14px;")
        self.sb_mkt_lbl.setStyleSheet(f"color: {dot_c}; font-size: 13px; font-weight: 600;")
        self.sb_mkt_lbl.setText(lbl_t)

    def _refresh_status(self):
        # ── API 連線燈號 ──
        try:
            api_ok = sys_state.api is not None and getattr(sys_state.api, 'stock_account', None) is not None
        except Exception:
            api_ok = False
        api_c = TV['green'] if api_ok else TV['text_dim']
        self.tb_api_dot.setStyleSheet(f"color: {api_c}; font-size: 10px;")
        self.tb_api_lbl.setStyleSheet(f"color: {api_c}; font-size: 11px;")
        # ── Telegram 燈號 ──
        connected = bool(tg_bot._get_chat_id())
        tg_c  = TV['green'] if connected else TV['text_dim']
        tg_t  = "Telegram: 已連線" if connected else "Telegram: 未綁定"
        self.sb_tg_lbl.setStyleSheet(f"color: {tg_c}; font-size: 13px;")
        self.sb_tg_lbl.setText(f"●  {tg_t}")
        is_live = getattr(sys_config, 'live_trading_mode', False)
        if is_live:
            self.sb_mode_lbl.setStyleSheet(f"color: {TV['red']}; font-size: 13px; font-weight: 700;")
            self.sb_mode_lbl.setText("⚠  正式下單模式")
            self.tb_mode_lbl.setStyleSheet(f"color: {TV['red']}; font-size: 12px; font-weight: 700;")
            self.tb_mode_lbl.setText("⚠ 正式下單模式")
        else:
            self.sb_mode_lbl.setStyleSheet(f"color: {TV['green']}; font-size: 13px; font-weight: 600;")
            self.sb_mode_lbl.setText("模擬模式")
            self.tb_mode_lbl.setStyleSheet(f"color: {TV['green']}; font-size: 12px; font-weight: 600;")
            self.tb_mode_lbl.setText("模擬模式")

    def _show_toast(self, message, duration=3000):
        """右下角滑入式提示訊息"""
        toast = QLabel(f"  {message}  ", self)
        toast.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        toast.setAttribute(Qt.WA_TranslucentBackground)
        toast.setStyleSheet(f"""
            QLabel {{
                background-color: rgba(30, 40, 60, 220);
                color: {TV['text_bright']};
                border: 1px solid {TV['blue']};
                border-radius: 8px;
                font-size: 13px;
                padding: 8px 16px;
            }}
        """)
        toast.adjustSize()
        # 定位到主視窗右下角
        geo = self.geometry()
        x = geo.x() + geo.width() - toast.width() - 20
        y = geo.y() + geo.height() - toast.height() - 50
        toast.move(x, y)
        toast.show()
        toast.raise_()
        QTimer.singleShot(duration, toast.deleteLater)

    # ══════════════════════════════════════════
    # 分頁管理核心方法
    # ══════════════════════════════════════════
    def _ensure_tab(self, key, title, factory):
        """懶建立分頁。factory 可以是 DialogClass 或 callable（回傳 QWidget）"""
        if key in self._tab_pages:
            page = self._tab_pages[key]
            idx = self.tabs.indexOf(page)
            if idx >= 0:
                self.tabs.setCurrentIndex(idx)
                return page
            else:
                del self._tab_pages[key]
        # 建立新分頁
        if callable(factory):
            try:
                page = factory()
            except TypeError:
                page = factory
        else:
            page = factory
        # 如果是 QDialog 子類，將其轉為可嵌入的 Widget
        if isinstance(page, QDialog):
            page.setWindowFlags(Qt.Widget)
            # 攔截 accept/reject，留在原分頁不跳轉
            page.accept = lambda: None
            page.reject = lambda: None
        self._tab_pages[key] = page
        idx = self.tabs.addTab(page, title)
        self.tabs.setCurrentIndex(idx)
        return page

    def _on_tab_close(self, index):
        """關閉分頁（Tab 0 系統日誌不可關閉）"""
        if index == 0:
            return
        widget = self.tabs.widget(index)
        self.tabs.removeTab(index)
        for key, page in list(self._tab_pages.items()):
            if page is widget:
                del self._tab_pages[key]
                break
        widget.deleteLater()

    def open_modeless(self, dialog_class, attr_name):
        """相容舊呼叫方式 → 改為開啟分頁"""
        self._ensure_tab(attr_name, dialog_class.__name__.replace('Dialog', ''), dialog_class)

    @pyqtSlot(str)
    def open_volume_analysis_window(self, file_path):
        """接收到訊號後，在分頁中開啟大數據分析結果"""
        key = 'vol_result'
        if key in self._tab_pages:
            page = self._tab_pages[key]
            idx = self.tabs.indexOf(page)
            if idx >= 0:
                # 更新已有的 browser
                for child in page.findChildren(QWebEngineView):
                    child.load(QUrl.fromLocalFile(file_path))
                self.tabs.setCurrentIndex(idx)
                return
            else:
                del self._tab_pages[key]
        # 建立新分頁
        page = QWidget()
        lo = QVBoxLayout(page); lo.setContentsMargins(0, 0, 0, 0)
        browser = QWebEngineView()
        browser.load(QUrl.fromLocalFile(file_path))
        lo.addWidget(browser)
        self._tab_pages[key] = page
        idx = self.tabs.addTab(page, '量能熱力圖')
        self.tabs.setCurrentIndex(idx)

    def start_monitoring_with_login(self):
        self._ensure_tab('live_trading', '盤中監控', LiveTradingPanel)

    def _show_emergency_menu(self, btn):
        """從側邊欄按鈕上方彈出緊急操作 QMenu"""
        menu = QMenu()
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {TV['surface']};
                border: 1px solid {TV['border_light']};
                border-radius: 8px;
                padding: 6px 0px;
                font-size: 13px;
                color: {TV['text']};
                min-width: 210px;
            }}
            QMenu::item {{
                padding: 10px 18px 10px 14px;
                border-radius: 4px;
                margin: 0px 4px;
            }}
            QMenu::item:selected {{
                background-color: {TV['border_light']};
                color: {TV['text']};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {TV['border']};
                margin: 4px 8px;
            }}
        """)
        a1 = menu.addAction("💥  一鍵全部市價平倉")
        a1.setData('close_all')
        a2 = menu.addAction("指定股票平倉")
        a2.setData('single_close')
        menu.addSeparator()
        a3 = menu.addAction("⏸  退出監控模式（不平倉）")
        a3.setData('stop_live')
        menu.addSeparator()
        a4 = menu.addAction("✕  強制關閉程式")
        a4.setData('force_quit')

        hint = menu.sizeHint()
        pos = btn.mapToGlobal(QPoint(0, -hint.height() - 6))
        action = menu.exec_(pos)
        if action is None:
            return
        d = action.data()
        if d == 'close_all':
            threading.Thread(target=exit_trade_live, daemon=True).start()
        elif d == 'single_close':
            code, ok = QInputDialog.getText(self, "單一平倉", "請輸入股票代號:")
            if ok and code:
                threading.Thread(target=close_one_stock, args=(code,), daemon=True).start()
        elif d == 'stop_live':
            sys_state.stop_trading_flag = True
            QMessageBox.information(self, "提示", "已發送終止指令。\n系統將在當前掃描週期結束後自動退出監控模式。")
        elif d == 'force_quit':
            os._exit(0)

    def _build_kline_panel(self):
        """建立 K 線更新面板（內嵌版，取代 exec_ 彈窗）"""
        from PyQt5.QtWidgets import QCalendarWidget, QListWidget
        from PyQt5.QtCore import QDate
        from PyQt5.QtGui import QColor
        import requests

        panel = QWidget()
        panel.setStyleSheet(TV_DIALOG_STYLE)

        # 外層加 padding，不占滿整個分頁
        outer = QVBoxLayout(panel); outer.setContentsMargins(0, 0, 0, 0)
        wrapper = QWidget(); wrapper.setMaximumWidth(960)
        main_layout = QHBoxLayout(wrapper); main_layout.setContentsMargins(32, 28, 32, 28); main_layout.setSpacing(20)
        outer.addWidget(wrapper, 0, Qt.AlignHCenter)
        outer.addStretch()

        # 左半部（70%）：標題 + 日曆
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel); left_layout.setContentsMargins(0, 0, 0, 0); left_layout.setSpacing(10)
        lbl_title = QLabel("K 線數據採集  │  選擇歷史交易日")
        lbl_title.setStyleSheet(f"color: {TV['text_bright']}; font-size: 17px; font-weight: 700; padding-bottom: 4px;")
        left_layout.addWidget(lbl_title)
        sub_lbl = QLabel("請選擇要採集的交易日（點擊多選，綠色 = 已選）：")
        sub_lbl.setStyleSheet(f"color: {TV['text_dim']}; font-size: 13px;")
        left_layout.addWidget(sub_lbl)

        # 右半部（30%）：清單 + 按鈕
        right_panel = QWidget(); right_panel.setFixedWidth(240)
        right_layout = QVBoxLayout(right_panel); right_layout.setContentsMargins(0, 36, 0, 0); right_layout.setSpacing(10)
        r_lbl = QLabel("已選取日期："); r_lbl.setStyleSheet(f"color: {TV['text_bright']}; font-size: 13px; font-weight: 600;")
        right_layout.addWidget(r_lbl)
        date_list_widget = QListWidget()
        date_list_widget.setStyleSheet(f"QListWidget {{ background: {TV['panel']}; color: {TV['text']}; border: 1px solid {TV['border']}; border-radius: 6px; font-size: 13px; }} QListWidget::item {{ padding: 4px 8px; }} QListWidget::item:selected {{ background: {TV['blue_dim']}; }}")
        right_layout.addWidget(date_list_widget, 1)

        btn_clear = QPushButton("清空清單")
        btn_clear.setFixedHeight(36)
        btn_clear.setStyleSheet(f"QPushButton {{ background: {TV['red']}; color: white; border: none; border-radius: 6px; font-size: 13px; font-weight: 600; }} QPushButton:hover {{ background: #c0392b; }}")
        btn_clear.clicked.connect(date_list_widget.clear)
        right_layout.addWidget(btn_clear)

        btn_ok = QPushButton("立即採集選定日期")
        btn_ok.setFixedHeight(42)
        btn_ok.setStyleSheet(f"QPushButton {{ background: {TV['green']}; color: white; border: none; border-radius: 6px; font-size: 13px; font-weight: 700; }} QPushButton:hover {{ background: #1a7a6e; }}")
        right_layout.addWidget(btn_ok)
        right_layout.addStretch()

        main_layout.addWidget(left_panel, 7)
        main_layout.addWidget(right_panel, 3)

        twse_holidays = _twse_holidays

        class CustomCalendar(QCalendarWidget):
            def __init__(self, holidays):
                super().__init__()
                self.holidays = holidays
                self.max_d = QDate.currentDate()
                if datetime.now().time() < datetime.strptime("13:30", "%H:%M").time():
                    self.max_d = self.max_d.addDays(-1)
                self.setMaximumDate(self.max_d)
                self.setVerticalHeaderFormat(QCalendarWidget.NoVerticalHeader)
                self.setStyleSheet(f"""
                    QCalendarWidget QWidget {{ alternate-background-color: {TV['surface']}; background-color: {TV['panel']}; color: {TV['text']}; selection-background-color: transparent; }}
                    QCalendarWidget QToolButton {{ background-color: transparent; color: {TV['text']}; font-weight: bold; font-size: 14px; padding: 4px; border-radius: 4px; }}
                    QCalendarWidget QToolButton:hover {{ background-color: {TV['blue']}; color: white; }}
                    QCalendarWidget QMenu {{ background-color: {TV['panel']}; color: {TV['text']}; }}
                    QCalendarWidget QSpinBox {{ background-color: {TV['surface']}; color: {TV['text']}; }}
                    QCalendarWidget QAbstractItemView:enabled {{ color: {TV['text']}; selection-background-color: transparent; selection-color: {TV['text']}; }}
                """)

            def paintCell(self, painter, rect, date):
                is_future = date > self.max_d
                is_weekend = date.dayOfWeek() >= 6
                is_holiday = date.toString("yyyyMMdd") in self.holidays
                painter.save()
                if is_future: painter.fillRect(rect, QColor("#1A1A1A"))
                elif is_weekend or is_holiday: painter.fillRect(rect, QColor("#4A1C1C"))
                else: painter.fillRect(rect, QColor(TV['surface']))
                date_str = date.toString("yyyy-MM-dd")
                is_selected_in_list = len(date_list_widget.findItems(date_str, Qt.MatchExactly)) > 0
                if is_selected_in_list and not is_future and not (is_weekend or is_holiday):
                    painter.fillRect(rect, QColor(TV['green']))
                if is_future: text_color = QColor(TV['border_light'])
                elif is_weekend or is_holiday: text_color = QColor(TV['red'])
                elif is_selected_in_list: text_color = QColor(TV['text_bright'])
                else: text_color = QColor(TV['text'])
                painter.setPen(text_color)
                painter.drawText(rect, Qt.AlignCenter, str(date.day()))
                painter.restore()

        calendar = CustomCalendar(twse_holidays)
        calendar.setFixedSize(560, 420)
        left_layout.addWidget(calendar)
        left_layout.addStretch()

        def on_calendar_clicked(date):
            is_future = date > calendar.max_d
            is_weekend = date.dayOfWeek() >= 6
            is_holiday = date.toString("yyyyMMdd") in twse_holidays
            if is_future or is_weekend or is_holiday: return
            date_str = date.toString("yyyy-MM-dd")
            items = date_list_widget.findItems(date_str, Qt.MatchExactly)
            if items:
                for item in items: date_list_widget.takeItem(date_list_widget.row(item))
            else:
                date_list_widget.addItem(date_str)
            all_items = [date_list_widget.item(i).text() for i in range(date_list_widget.count())]
            all_items.sort()
            date_list_widget.clear()
            date_list_widget.addItems(all_items)
            calendar.updateCells()

        calendar.clicked.connect(on_calendar_clicked)
        btn_clear.clicked.connect(calendar.updateCells)

        def on_confirm():
            selected_dates = [date_list_widget.item(i).text() for i in range(date_list_widget.count())]
            if not selected_dates:
                return QMessageBox.warning(panel, "提示", "請至少選擇一天進行採集！")
            symbols = [s for g in load_matrix_dict_analysis().values() for s in g]
            self.kline_fetch_thread = ShioajiKLineFetchThread(selected_dates, symbols, twse_holidays)
            ui_dispatcher.progress_visible.emit(True)
            self.kline_fetch_thread.progress_signal.connect(lambda p, m: ui_dispatcher.progress_updated.emit(p, m))
            self.kline_fetch_thread.finished_signal.connect(self.on_kline_fetch_finished)
            self.kline_fetch_thread.start()
            if _main_window_ref:
                _main_window_ref.tabs.setCurrentIndex(0)

        btn_ok.clicked.connect(on_confirm)
        return panel

    # ==========================================
    # v1.9.8.6 修改版：回測結果輸出 (回測模式)
    # ==========================================
    @pyqtSlot(object)
    def plot_equity(self, data_tuple):
        # 解包收到的資料
        if len(data_tuple) == 3:
            all_trades, all_events, intraday_data = data_tuple
        else:
            all_trades, all_events = data_tuple
            intraday_data = {}

        # --- 防止重複啟動 ---
        if hasattr(self, '_prep_thread') and self._prep_thread.isRunning():
            return

        target_date = sys_db.load_state('last_fetched_date')
        print("\n🔄 正在載入回測覆盤終端，請稍候...")

        # --- 把 DB 查詢與 payload 建構移到背景執行緒，避免多日回測卡死主執行緒 ---
        self._prep_thread = PrepareTerminalDataThread(all_trades, all_events, target_date, "backtest", parent=self)

        def _on_ready(result):
            # 背景執行緒已完成所有耗時工作，主執行緒只觸發非同步的 browser.load()
            _show_terminal_window(result['temp_file'], result['mode'], result['actual_date'])
            # ★ cashflow HTML 也已在背景建構完畢
            if result.get('cashflow_html'):
                try:
                    # 優先從分頁取得終端實例
                    tw = None
                    if 'terminal' in self._tab_pages:
                        tw = getattr(self._tab_pages['terminal'], '_tw', None)
                    if tw is None:
                        tw = term_win_instance
                    if tw:
                        tw.load_cashflow(result['cashflow_html'])
                except Exception as _cf_e:
                    logger.warning(f"資金流動圖載入失敗: {_cf_e}")

        self._prep_thread.ready.connect(_on_ready)
        self._prep_thread.start()

    @pyqtSlot(bool, str)
    def on_kline_fetch_finished(self, success, message):
        ui_dispatcher.progress_visible.emit(False)
        
        msg_box = QMessageBox(self)
        msg_box.setStyleSheet(TV_DIALOG_STYLE)
        
        if success:
            msg_box.setWindowTitle("採集完成")
            msg_box.setText(message)
            msg_box.setIcon(QMessageBox.Information)
        else:
            msg_box.setWindowTitle("採集失敗")
            msg_box.setText(message)
            msg_box.setIcon(QMessageBox.Critical)
            
        msg_box.exec_()

    @pyqtSlot(str)
    def normal_output(self, text):
        # 確保視窗關閉中或發生 KeyboardInterrupt 時不崩潰
        if not hasattr(self, 'console') or self.console is None: return
        try:
            if not text or not isinstance(text, str): return
            # 盤中監控期間，輸出只導向盤中監控面板的終端，不寫入系統日誌
            if getattr(sys_state, 'is_monitoring', False): return

            if "Subscription Already Exists" in text or "Event Code: 16" in text:
                return
            # 處理轉義與顏色
            h = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace(' ', '&nbsp;').replace('\n', '<br>')
            for p, c in [(r'\x1b\[(?:31|91)m|\033\[(?:31|91)m', TV['red']), (r'\x1b\[(?:32|92)m|\033\[(?:32|92)m', TV['green']), (r'\x1b\[(?:33|93)m|\033\[(?:33|93)m', TV['yellow']), (r'\x1b\[(?:34|94)m|\033\[(?:34|94)m', TV['blue']), (r'\x1b\[(?:35|95)m|\033\[(?:35|95)m', TV['purple']), (r'\x1b\[(?:36|96)m|\033\[(?:36|96)m', '#26c6da')]:
                h = re.sub(p, f'<span style="color: {c}; font-weight: bold;">', h)
            h = re.sub(r'\x1b\[0m|\x1b\[39m|\033\[0m', '</span>', h)
            
            # 執行緒安全游標操作
            cursor = self.console.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.console.setTextCursor(cursor)
            self.console.insertHtml(h)
        except (KeyboardInterrupt, SystemExit): pass
        except Exception: pass

    def start_correlation_thread(self, mode, w_mins):
        # 接收來自設定視窗的參數，並安全地在背景啟動連動分析
        print(f"\x1b[35m啟動族群連動分析 ({'微觀實戰模擬' if mode == 'micro' else '全天宏觀連動'}, 等待: {w_mins}分)...\x1b[0m")
        self.corr_thread = CorrelationAnalysisThread(mode, w_mins)
        self.corr_thread.finished_signal.connect(lambda r: (print(f"\x1b[32m✅ 分析完成，共產出 {len(r)} 筆。\x1b[0m"), self._ensure_tab('corr_result', '連動結果', lambda: CorrelationResultDialog(r, self))))
        self.corr_thread.start()

# ==========================================
# 教學文件載入引擎 (自動修復 JSON 並套用深色主題)
# ==========================================
def init_and_load_tutorials():
    doc_path = os.path.join("docs", "tutorials.json")
    
    # 定義黑藍色質感的 HTML 內容
    bg_style = f"background-color:{TV['bg']}; color:{TV['text']}; padding:20px; font-family: Microsoft JhengHei;"
    hr_style = f"border: 1px solid {TV['blue']};"
    h1_style = f"color:{TV['blue']}; text-align:center;"
    h2_style = f"color:{TV['yellow']};"

    default_data = {
        "menu_config": [
            {"id": "volume_analysis", "title": "1. 策略參數優化工具"},
            {"id": "backtest_guide", "title": "2. 回測與績效評估"},
            {"id": "risk_management", "title": "3. 風險管理與停損策略"}
        ],
        "volume_analysis": f"<div style='{bg_style}'> <h1 style='{h1_style}'>策略參數優化指南</h1> <p>協助您解讀 AI 參數優化與多元迴歸分析結果。</p> <hr style='{hr_style}'> <h2 style='{h2_style}'>1. AI 參數優化氣泡圖</h2> <ul> <li><b>右上角熱區</b>：高獲利、高勝率且具備高容錯率。</li> <li><b>判定係數 (R²)</b>：代表參數對獲利波動的解釋力，愈高愈好。</li> </ul> <hr style='{hr_style}'> <h2 style='{h2_style}'>2. 多元迴歸分析</h2> <ul> <li><b>綠色長條</b>：統計極顯著特徵，是獲利的真實關鍵。</li> </ul> </div>",
        "backtest_guide": f"<div style='{bg_style}'> <h1 style='{h1_style}'>回測數據說明</h1> <p>說明歷史交易紀錄、資金曲線、最大回撤(MDD)等指標...</p> </div>",
        "risk_management": f"<div style='{bg_style}'> <h1 style='{h1_style}'>風險管理與停損策略</h1> <p>說明動態停損、移動停利的操作細節...</p> </div>"
    }

    if not os.path.exists("docs"): os.makedirs("docs")
    if not os.path.exists(doc_path):
        with open(doc_path, "w", encoding="utf-8") as f:
            json.dump(default_data, f, ensure_ascii=False, indent=4)
        return default_data

    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception: return default_data

def load_tutorial(doc_id):
    data = init_and_load_tutorials()
    return data.get(doc_id, "⚠️ 內容缺失")

# ==================== 策略參數掃描引擎 ====================
class VolumeAnalysisDialog(BaseDialog):
    def __init__(self):
        super().__init__("策略參數掃描 - 網格搜索", (1200, 700))
        from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout, QCheckBox, QDoubleSpinBox, QGridLayout, QWidget, QSplitter
        from PyQt5.QtCore import Qt
        
        # 主佈局改為水平排列
        main_layout = QHBoxLayout(self)
        
        # 加入可拖拉的左右分割器
        self.splitter = QSplitter(Qt.Horizontal)
        
        # ==========================================
        # 👈 左半部：參數控制台面板
        # ==========================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)
        
        title_layout = QHBoxLayout()
        lbl = QLabel("策略參數網格搜索 — 勾選 = 固定常數，取消 = 範圍掃描")
        lbl.setStyleSheet(f"color: {TV['green']}; font-weight: 700; font-size: 14px; margin-bottom: 8px;")
        title_layout.addWidget(lbl)
        
        title_layout.addStretch() # 把按鈕推到右邊
        
        self.btn_help = QPushButton("  ?  判讀教學")
        self.btn_help.setFixedHeight(32)
        self.btn_help.setStyleSheet(f"""
            QPushButton {{ background-color: {TV['surface']}; color: {TV['text_dim']}; border: 1px solid {TV['border_light']}; border-radius: 5px; font-size: 12px; font-weight: 600; padding: 0 10px; }}
            QPushButton:hover {{ background-color: {TV['yellow']}; color: {TV['bg']}; border-color: {TV['yellow']}; }}
        """)
        self.btn_help.setCursor(Qt.PointingHandCursor)
        self.btn_help.clicked.connect(self.show_tutorial)
        title_layout.addWidget(self.btn_help)
        
        left_layout.addLayout(title_layout)

        self.grid_widget = QWidget()
        grid = QGridLayout(self.grid_widget)
        grid.setSpacing(10)

        headers = ["固定?", "參數名稱", "固定值 / 變動起點", "變動終點", "變動級距"]
        for col, h in enumerate(headers):
            hl = QLabel(h)
            hl.setStyleSheet(f"color: {TV['yellow']}; font-weight: 700; font-size: 11px; letter-spacing: 0.5px; border-bottom: 1px solid {TV['border_light']}; padding-bottom: 5px;")
            grid.addWidget(hl, 0, col)

        # 參數設定檔 (預設值完全依照您的需求)
        self.params_config = {
            'wait_mins': {'name': '等待時間 (分)', 'fixed': 3, 'start': 1, 'end': 5, 'step': 1, 'is_int': True},
            'dtw_thresh': {'name': 'DTW 相似度', 'fixed': 0.0, 'start': 0.0, 'end': 0.9, 'step': 0.05, 'is_int': False},
            'leader_pull': {'name': '領漲拉高幅 (%)', 'fixed': 1.4, 'start': 0.5, 'end': 2.5, 'step': 0.1, 'is_int': False},
            'follow_pull': {'name': '跟漲追蹤幅 (%)', 'fixed': 0.5, 'start': 0.0, 'end': 1.5, 'step': 0.1, 'is_int': False},
            'vol_mult': {'name': '開盤均量倍數', 'fixed': 1.1, 'start': 0.5, 'end': 2.5, 'step': 0.1, 'is_int': False},
            'vol_abs': {'name': '絕對成交量 (張)', 'fixed': 800, 'start': 0, 'end': 3000, 'step': 200, 'is_int': True},
            'wait_min_avg_vol': {'name': '等待期均量下限 (張)', 'fixed': 50, 'start': 0, 'end': 500, 'step': 20, 'is_int': True},
            'wait_max_single_vol': {'name': '單根爆量下限 (張)', 'fixed': 150, 'start': 0, 'end': 1000, 'step': 20, 'is_int': True},
            'sl_cushion_pct': {'name': '停損緩衝空間 (%)', 'fixed': 0.3, 'start': 0.0, 'end': 2.0, 'step': 0.2, 'is_int': False},
            'cutoff_mins': {'name': '尾盤停止觸發 (時:分)', 'fixed': 240, 'start': 180, 'end': 255, 'step': 15, 'is_int': True, 'is_time': True},
            'hold_mins': {'name': '固定持倉時間 (分)', 'fixed': 240, 'start': 15, 'end': 240, 'step': 15, 'is_int': True}
        }

        self.ui_elements = {}
        row = 1

        class TimeSpinBox(QTimeEdit):
            def __init__(self):
                super().__init__()
                self.setDisplayFormat("HH:mm") # 限定只顯示 時:分
                # 限制只能選擇台股時間 (選做，增加防呆)
                self.setTimeRange(QTime(9, 0), QTime(13, 30))
            
            # 把拿到的時間，轉回 AI 要的分鐘數
            def value(self):
                t = self.time()
                return (t.hour() - 9) * 60 + t.minute()
                
            # 把 AI 預設的分鐘數，換算成時間顯示出來
            def setValue(self, val):
                val = int(val)
                h = 9 + val // 60
                m = val % 60
                # 防止發生奇怪的計算錯誤
                if h >= 24: h = 23
                if h < 0: h = 0
                self.setTime(QTime(h, m))

        for key, cfg in self.params_config.items():
            cb_fixed = QCheckBox()
            cb_fixed.setChecked(key in ['cutoff_mins', 'hold_mins'])
            
            lbl_name = QLabel(cfg['name'])
            is_time_param = cfg.get('is_time', False)
            
            # 創建輸入框
            def create_spinbox(val, is_int, is_time_field=False):
                if is_time_field:
                    sb = TimeSpinBox() # 套用全新升級的魔法時鐘
                else:
                    sb = QSpinBox() if is_int else QDoubleSpinBox()
                    if not is_int: sb.setSingleStep(0.05); sb.setDecimals(2)
                    sb.setRange(-1000, 10000)
                
                sb.setStyleSheet(f"background-color: {TV['surface']}; color: {TV['text']}; padding: 5px; border: 1px solid {TV['border_light']}; border-radius: 4px;")
                sb.setValue(val)
                return sb

            # 起點和終點如果是時間參數，就顯示時鐘；級距 (Step) 依然是普通的分鐘數
            sb_start = create_spinbox(cfg['fixed'] if cb_fixed.isChecked() else cfg['start'], cfg['is_int'], is_time_param)
            sb_end = create_spinbox(cfg['end'], cfg['is_int'], is_time_param)
            sb_step = create_spinbox(cfg['step'], cfg['is_int'], False)

            # 動態切換顯示狀態
            def toggle_state(state, s_start=sb_start, s_end=sb_end, s_step=sb_step, k=key):
                is_fixed = (state == Qt.Checked)
                s_end.setVisible(not is_fixed)
                s_step.setVisible(not is_fixed)
                s_start.setValue(self.params_config[k]['fixed'] if is_fixed else self.params_config[k]['start'])

            cb_fixed.stateChanged.connect(toggle_state)
            toggle_state(Qt.Checked if cb_fixed.isChecked() else Qt.Unchecked)

            grid.addWidget(cb_fixed, row, 0)
            grid.addWidget(lbl_name, row, 1)
            grid.addWidget(sb_start, row, 2)
            grid.addWidget(sb_end, row, 3)
            grid.addWidget(sb_step, row, 4)

            self.ui_elements[key] = {'cb': cb_fixed, 'start': sb_start, 'end': sb_end, 'step': sb_step, 'is_int': cfg['is_int']}
            row += 1

        left_layout.addWidget(self.grid_widget)
        
        # 💡 將左側內容往上推，避免按鈕與進度條亂飄
        left_layout.addStretch()

        # 按鈕區
        btn_layout = QHBoxLayout(); btn_layout.setSpacing(8)

        # 天數選擇
        self.days_combo = QComboBox()
        self.days_combo.addItems(["1 日", "近 7 日", "近 15 日", "近一月 (30日)"])
        self.days_combo.setCurrentIndex(1)
        self.days_combo.setFixedHeight(42)
        self.days_combo.setStyleSheet(f"""
            QComboBox {{ background-color: {TV['surface']}; color: {TV['text']}; padding: 0 12px;
                         font-size: 13px; font-weight: 600; border-radius: 7px; border: 1px solid {TV['border_light']}; }}
            QComboBox::drop-down {{ width: 28px; border-left: 1px solid {TV['border_light']}; }}
            QComboBox QAbstractItemView {{ background-color: {TV['panel']}; color: {TV['text']};
                selection-background-color: {TV['blue']}; border: 1px solid {TV['border_light']}; }}
            QComboBox QAbstractItemView::item {{ min-height: 32px; }}
        """)
        btn_layout.addWidget(self.days_combo)

        self.btn_fetch = QPushButton("  ▷  STEP 1  採集數據")
        self.btn_fetch.setFixedHeight(42)
        self.btn_fetch.setStyleSheet(f"""
            QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {TV['blue']}, stop:1 #1565c0);
                           color:white; border:none; border-radius:7px; font-size:13px; font-weight:700; }}
            QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #3d7aff, stop:1 {TV['blue']}); }}
            QPushButton:disabled {{ background: {TV['surface']}; color: {TV['text_dim']}; }}
        """)
        self.btn_fetch.setCursor(Qt.PointingHandCursor); self.btn_fetch.clicked.connect(self.fetch_data)
        btn_layout.addWidget(self.btn_fetch)

        self.btn_analyze = QPushButton("  ▶  STEP 2  啟動 AI 網格搜索")
        self.btn_analyze.setFixedHeight(42)
        self.btn_analyze.setStyleSheet(f"""
            QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {TV['orange']}, stop:1 #e65100);
                           color:white; border:none; border-radius:7px; font-size:13px; font-weight:700; }}
            QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #ffa726, stop:1 {TV['orange']}); }}
            QPushButton:disabled {{ background: {TV['surface']}; color: {TV['text_dim']}; }}
        """)
        self.btn_analyze.setCursor(Qt.PointingHandCursor); self.btn_analyze.clicked.connect(self.start_analysis)
        btn_layout.addWidget(self.btn_analyze)
        btn_layout.setStretch(0, 2); btn_layout.setStretch(1, 3); btn_layout.setStretch(2, 4)
        left_layout.addLayout(btn_layout)

        self.p_bar = QProgressBar(); self.p_bar.setFixedHeight(4); self.p_bar.setTextVisible(False)
        self.p_bar.setStyleSheet(f"""
            QProgressBar {{ border: none; background: {TV['surface']}; border-radius: 2px; }}
            QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {TV['blue']}, stop:1 {TV['orange']}); border-radius: 2px; }}
        """)
        self.p_bar.hide(); left_layout.addWidget(self.p_bar)

        # ==========================================
        # 右半部：AI 引擎進度面板
        # ==========================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        console_header_row = QHBoxLayout()
        console_title = QLabel("▸  AI ENGINE  /  LIVE OUTPUT")
        console_title.setStyleSheet(f"color: {TV['text_dim']}; font-size: 11px; font-weight: 700; letter-spacing: 1px;")
        console_header_row.addWidget(console_title); console_header_row.addStretch()
        right_layout.addLayout(console_header_row)

        self.console = QTextEdit(); self.console.setReadOnly(True)
        self.console.setStyleSheet(f"""
            QTextEdit {{
                background-color: {TV['console_bg']};
                color: {TV['green']};
                font-family: 'Cascadia Code', 'Consolas', monospace;
                font-size: 13px;
                border: 1px solid {TV['border']};
                border-radius: 6px;
                padding: 10px;
            }}
            QScrollBar:vertical {{ background: {TV['console_bg']}; width: 5px; }}
            QScrollBar::handle:vertical {{ background: {TV['surface']}; border-radius: 2px; }}
        """)
        right_layout.addWidget(self.console, stretch=1)

        # ==========================================
        # 組合左右兩塊面板
        # ==========================================
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)
        # 設定初始比例 (約 55% 寬度給左邊參數，45% 給右邊看終端)
        self.splitter.setSizes([630, 520]) 
        
        main_layout.addWidget(self.splitter)

    def log(self, msg):
        self.console.append(msg)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())
    
    def fetch_data(self):
        days_mapping = {0: 1, 1: 7, 2: 15, 3: 30}
        selected_index = self.days_combo.currentIndex()
        days_to_fetch = days_mapping.get(selected_index, 7)
        
        self.btn_fetch.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.console.clear()
        self.p_bar.show()
        self.p_bar.setValue(0)
        
        self.log(f"⏳ 準備啟動大數據採集，目標天數：近 {days_to_fetch} 交易日...")
        
        # 呼叫 1.9.7.1 內建的 FetchAnalysisDataThread，並傳入選擇的天數
        self.fetch_thread = FetchAnalysisDataThread(days_to_fetch)
        self.fetch_thread.log_signal.connect(self.log)
        self.fetch_thread.progress_signal.connect(self.p_bar.setValue)
        
        # 完成時解鎖按鈕並顯示結果
        def on_finished(success, msg):
            self.btn_fetch.setEnabled(True)
            self.btn_analyze.setEnabled(True)
            self.p_bar.hide()
            self.log(msg)
            if success:
                QMessageBox.information(self, "採集成功", "✅ 大數據採集完成！\n現在您可以點擊「2. 開始網格搜索」讓 AI 引擎進行分析。")
            else:
                QMessageBox.critical(self, "採集失敗", msg)
                
        self.fetch_thread.finished_signal.connect(on_finished)
        self.fetch_thread.start()

    def show_tutorial(self):
        # 統一跳轉至主介面新手教學分頁，自動定位到「策略參數掃描」頁面
        global _main_window_ref
        if _main_window_ref:
            tw = _main_window_ref._ensure_tab('tutorial', '新手教學', TutorialWidget)
            if hasattr(tw, 'goto'):
                tw.goto('volume_analysis')

    def start_analysis(self):
        # 收集參數
        search_space = {}
        import numpy as np
        
        for k, els in self.ui_elements.items():
            if els['cb'].isChecked():
                # 固定值
                search_space[k] = [els['start'].value()]
            else:
                # 變動範圍
                start, end, step = els['start'].value(), els['end'].value(), els['step'].value()
                if step <= 0: return QMessageBox.critical(self, "錯誤", f"{k} 的級距必須大於 0")
                
                if els['is_int']:
                    arr = list(range(int(start), int(end) + 1, int(step)))
                else:
                    arr = [round(x, 2) for x in np.arange(start, end + (step/2), step)]
                search_space[k] = arr

        total_combinations = 1
        for v in search_space.values(): total_combinations *= len(v)
        
        if total_combinations > 50000:
            # Optuna 設定的 n_trials 為 5000 次，抽樣 70% 日期
            # 根據實測，每次 trial 約需 0.015 秒純運算
            # 加上載入資料庫的基礎 overhead (約 60~65 秒)
            ai_trials = 5000
            est_sec = int(66 + (ai_trials * 0.015))
            
            if est_sec > 60:
                eta_str = f"約 {est_sec // 60} 分鐘 {est_sec % 60} 秒"
            else:
                eta_str = f"約 {est_sec} 秒"

            reply = QMessageBox.question(
                self, "系統運算提示", 
                f"您設定的參數將產生 <b>{total_combinations:,}</b> 種組合！\n\n"
                f"🤖 系統將自動啟動 <b>AI 貝氏最佳化引擎</b>，\n"
                f"主動避開低勝率區間，僅精準採樣 {ai_trials} 組。\n\n"
                f"⏳ 預估運算需時：<b>{eta_str}</b>\n\n"
                f"確定要執行嗎？", 
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No: return

        self.log(f"📦 參數收集完畢，共 {total_combinations} 種可能組合，準備交由引擎並行運算...")
        self.btn_analyze.setEnabled(False)
        self.p_bar.show()
        self.p_bar.setValue(0)
        
        self.analysis_thread = RunAnalysisTaskThread(search_space, total_combinations)
        self.analysis_thread.log_signal.connect(self.log)
        self.analysis_thread.progress_signal.connect(self.p_bar.setValue)
        self.analysis_thread.finished_signal.connect(lambda: self.btn_analyze.setEnabled(True))
        self.analysis_thread.start()

# ---------------------------------------------------------
# v1.9.8.6 方案三：多進程 worker-mode（被 subprocess 呼叫時，不啟動 Qt）
# ---------------------------------------------------------
def _run_regression_worker_mode():
    """被 RunAnalysisTaskThread 以 subprocess 啟動時的入口。
    只跑 Optuna，不初始化 Qt，結果寫入 JSON 後退出。
    """
    import argparse, json as _json, optuna as _optuna
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)

    ap = argparse.ArgumentParser()
    ap.add_argument('--args-file')
    ap.add_argument('--output-file')
    ns = ap.parse_args()

    with open(ns.args_file, encoding='utf-8') as f:
        cfg = _json.load(f)

    db_path      = cfg['db_path']
    search_space = cfg['search_space']
    n_trials     = cfg['n_trials']
    unique_dates = cfg['unique_dates']
    sample_ratio = cfg.get('sample_ratio', 0.45)

    cache  = build_backtest_cache(db_path=db_path, table_name='intraday_kline',
                                  dates=unique_dates)
    mat    = load_matrix_dict_analysis()
    dispo  = load_disposition_stocks()

    study = _optuna.create_study(
        direction="maximize",
        pruner=_optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )

    def _w_objective(trial):
        import random as _rnd
        params = {}
        for k, v in search_space.items():
            if len(v) == 1: params[k] = v[0]
            elif isinstance(v[0], str): params[k] = trial.suggest_categorical(k, v)
            else:
                is_int = all(isinstance(x, int) for x in v)
                lo, hi = min(v), max(v)
                if is_int:
                    step = v[1] - v[0] if len(v) > 1 else 1
                    params[k] = trial.suggest_int(k, lo, hi, step=step)
                else:
                    params[k] = round(trial.suggest_float(k, lo, hi), 2)

        _min_s = max(5, len(unique_dates) // 3)
        k_dates = max(_min_s, int(len(unique_dates) * sample_ratio))
        sampled = sorted(_rnd.sample(unique_dates, k_dates))
        res = regression_evaluator(params, sampled, cache, mat, dispo, trial=trial)
        for key, val in res.items():
            trial.set_user_attr(key, val)
        return res['ai_score'] * (len(unique_dates) / len(sampled))

    study.optimize(_w_objective, n_trials=n_trials, n_jobs=1)

    results = [t.user_attrs for t in study.trials
               if t.state == _optuna.trial.TrialState.COMPLETE]
    with open(ns.output_file, 'w', encoding='utf-8') as f:
        _json.dump(results, f, ensure_ascii=False, default=str)

    sys.exit(0)


# ---------------------------------------------------------
# v1.9.8.6 預計算快取：將 process_group_data 的 setup 階段
#   移至 Optuna 迴圈外，每 trial 節省 ~20 秒
# ---------------------------------------------------------
def precompute_group_day_data(stock_data_collection):
    """預計算族群-日期的 DataFrame setup，供 process_group_data(headless) 重複使用。

    回傳 dict 或 None（若資料不足）。
    """
    req_cols = ['time', 'rise', 'high', '漲停價', 'close', 'pct_increase', 'volume']
    precalc_vols   = {}
    _time_row_cache = {}
    merged_df = None

    for sym, df in stock_data_collection.items():
        if not all(c in df.columns for c in req_cols):
            continue
        for col in ['rise', 'high', 'close', 'volume', '漲停價', 'pct_increase']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        vols = {}
        for idx, t_str in enumerate(['09:00:00', '09:01:00', '09:02:00']):
            try:    vols[t_str] = float(df.iloc[idx]['volume']) if len(df) > idx else None
            except Exception: vols[t_str] = None
        precalc_vols[sym] = vols

        _time_row_cache[sym] = {r['time']: r for r in df.to_dict('records')}

        tmp = df[req_cols].rename(columns={c: f"{c}_{sym}" if c != 'time' else c for c in req_cols})
        merged_df = tmp if merged_df is None else pd.merge(merged_df, tmp, on='time', how='outer')

    if merged_df is None or merged_df.empty:
        return None

    merged_df.sort_values('time', inplace=True, ignore_index=True)

    _f3avg_final = {}
    _f3avg_at    = {'09:00:00': {}, '09:01:00': {}}
    for sym in precalc_vols:
        all_v = [v for v in precalc_vols[sym].values() if v is not None]
        _f3avg_final[sym] = sum(all_v) / len(all_v) if all_v else 0
        for cutoff in ['09:00:00', '09:01:00']:
            sub = [v for t, v in precalc_vols[sym].items() if t <= cutoff and v is not None]
            _f3avg_at[cutoff][sym] = sum(sub) / len(sub) if sub else 0

    merged_records = merged_df.to_dict('records')
    return {
        'stock_collection':  stock_data_collection,
        '_time_row_cache':   _time_row_cache,
        'precalc_vols':      precalc_vols,
        '_f3avg_final':      _f3avg_final,
        '_f3avg_at':         _f3avg_at,
        'merged_records':    merged_records,
        'total_rows':        len(merged_records),
    }


def build_regression_precomp(cache, groups, dispo_list):
    """為所有 (族群, 日期) 預計算 setup，回傳 dict[(grp, date)] = precomp。

    在 Optuna study.optimize() 之前呼叫一次，之後每個 trial 直接查表。
    """
    result = {}
    for t_date, day_data in cache.items():
        dispo_today = dispo_list.get(t_date, []) if isinstance(dispo_list, dict) else dispo_list
        for grp, syms in groups.items():
            valid = [s for s in syms if s not in dispo_today and s in day_data]
            if len(valid) < 2:
                continue
            pc = precompute_group_day_data({s: day_data[s] for s in valid})
            if pc is not None:
                result[(grp, t_date)] = pc
    return result


# ---------------------------------------------------------
# Helper: _trunc2（無條件捨去到小數第二位，避免浮點比較誤差）
# ---------------------------------------------------------
def _trunc2(value):
    try:
        return math.floor(float(value) * 100) / 100.0
    except (ValueError, TypeError):
        return value

# ---------------------------------------------------------
# fast_evaluator：高速回歸評估器（list-of-dicts，純 Python dict 存取）
#   速度比 regression_evaluator 快 10x；用於 Optuna 400 trials 主搜尋。
#   精度略低但足以找出正確參數方向；前 30 名由 regression_evaluator 全量驗證。
#   Bug fixes vs 1.9.8.4：
#     1. cache.get(t_date, {}) 取代 cache[t_date]（避免 KeyError）
#     2. dispo 支援 dict{date:[syms]} 與 flat list 兩種格式
# ---------------------------------------------------------
def fast_evaluator(params, dates, cache, groups, dispo_list, trial=None):
    p_wait          = params['wait_mins']
    p_dtw           = params['dtw_thresh']
    p_lead          = params['leader_pull']
    p_foll          = params['follow_pull']
    p_vmult         = params['vol_mult']
    p_vabs          = params['vol_abs']
    p_wait_min_avg  = params['wait_min_avg_vol']
    p_wait_max_single = params['wait_max_single_vol']
    p_sl_cushion    = params.get('sl_cushion_pct', 0.0)
    p_hold_mins     = params.get('hold_mins', 240)
    p_cutoff_raw    = params.get('cutoff_mins', 270)
    if isinstance(p_cutoff_raw, str):
        h, m_t = map(int, p_cutoff_raw.split(':'))
        p_cutoff_mins = (h - 9) * 60 + m_t
    else:
        p_cutoff_mins = int(p_cutoff_raw)

    tp_sum, wins, losses, trades_count = 0, 0, 0, 0
    winning_days, days_traded = set(), set()
    wins_a, losses_a, trades_a, tp_sum_a = 0, 0, 0, 0

    _max_daily_stops = getattr(sys_config, 'max_daily_stops', 3)
    _max_daily_entries = getattr(sys_config, 'max_daily_entries', 12)
    _risk_enabled = getattr(sys_config, 'risk_control_enabled', True)

    for step_idx, t_date in enumerate(dates):
        # ── Bug fix 1：安全取值，避免 KeyError ──
        day_data = cache.get(t_date, {})
        if not day_data:
            continue
        day_pnl = 0
        _daily_stops_b = 0
        _daily_entries_b = 0

        for grp, syms in groups.items():
            # ── Bug fix 2：dispo 支援 dict 與 flat list ──
            dispo_today = dispo_list.get(t_date, []) if isinstance(dispo_list, dict) else dispo_list
            valid_syms = [s for s in syms if s not in dispo_today and s in day_data]
            if len(valid_syms) < 2:
                continue

            stock_dfs   = {s: day_data[s] for s in valid_syms}
            first3_vol  = {s: sum(r['volume'] for r in recs[:3]) / 3 if len(recs) >= 3 else 0
                           for s, recs in stock_dfs.items()}

            _df_cache = {}
            if p_dtw > 0:
                from datetime import time as _time_type
                for sym in valid_syms:
                    if stock_dfs[sym]:
                        _df = pd.DataFrame(stock_dfs[sym])
                        _df['time'] = pd.to_datetime(_df['time'].astype(str), format='%H:%M:%S').dt.time
                        _df_cache[sym] = _df

            leader, tracking = None, set()
            in_wait, wait_cnt, start_t, leader_peak_rise = False, 0, None, None
            leader_rise_before_decline = None   # 天花板：反轉時記錄，突破時解除等待（對齊 process_group_data）
            reentry_count = 0                   # 停損再進場計數器
            pull_up, limit_up = False, False
            is_busy, exit_at  = False, -1

            num_bars = min(len(recs) for recs in stock_dfs.values())
            if num_bars == 0:
                continue

            _cum_vol  = {s: 0      for s in valid_syms}
            _run_high = {s: -999.0 for s in valid_syms}
            _run_low  = {s: 999999.0 for s in valid_syms}

            for m in range(num_bars):
                for _s in valid_syms:
                    if m < len(stock_dfs[_s]):
                        _bar = stock_dfs[_s][m]
                        _cum_vol[_s]  += _bar.get('volume', 0)
                        _run_high[_s]  = max(_run_high[_s], _bar.get('high', -999))
                        _run_low[_s]   = min(_run_low[_s], _bar.get('low', _bar.get('close', 999999)))

                if is_busy:
                    if m >= exit_at: is_busy = False
                    continue

                if m >= p_cutoff_mins:
                    is_busy, exit_at = True, 999
                    continue

                trigger_list = []
                for sym in valid_syms:
                    row, avgv = stock_dfs[sym][m], first3_vol[sym]
                    if round(row['high'], 2) >= round(row['漲停價'], 2):
                        if m == 0 or round(stock_dfs[sym][m - 1]['high'], 2) < round(row['漲停價'], 2):
                            trigger_list.append((sym, 'limit'))
                    elif (row['pct_increase'] >= p_lead and
                          (row['volume'] >= p_vabs or (avgv > 0 and row['volume'] >= p_vmult * avgv))):
                        trigger_list.append((sym, 'pull'))

                for sym, cond in trigger_list:
                    if cond == 'limit':
                        tracking.add(sym); leader, in_wait, wait_cnt = sym, True, 0
                        if not (pull_up or limit_up): start_t = row['time']
                        pull_up, limit_up = False, True
                    else:
                        if not pull_up and not limit_up:
                            pull_up, limit_up, start_t = True, False, row['time']
                            tracking.clear()
                        tracking.add(sym)

                if pull_up or limit_up:
                    for sym in valid_syms:
                        if sym not in tracking and stock_dfs[sym][m]['pct_increase'] >= p_foll:
                            tracking.add(sym)

                if tracking:
                    max_sym, max_r = None, -999
                    for sym in tracking:
                        if stock_dfs[sym][m]['rise'] > max_r:
                            max_r, max_sym = stock_dfs[sym][m]['rise'], sym
                    if leader is None:
                        leader, leader_peak_rise = max_sym, max_r
                    elif max_sym == leader:
                        if max_r is not None and (leader_peak_rise is None or max_r > leader_peak_rise):
                            leader_peak_rise = max_r
                    elif max_r is not None and leader_peak_rise is not None and max_r > leader_peak_rise:
                        leader, start_t, in_wait, wait_cnt, leader_peak_rise = \
                            max_sym, stock_dfs[max_sym][m]['time'], False, 0, max_r
                    if leader and stock_dfs[leader][m]['high'] <= stock_dfs[leader][max(0, m - 1)]['high'] and not in_wait:
                        in_wait, wait_cnt = True, 0
                        leader_rise_before_decline = stock_dfs[leader][m].get('highest', stock_dfs[leader][m]['high'])

                if in_wait:
                    wait_cnt += 1
                    if p_dtw > 0 and wait_cnt >= p_wait - 1 and leader and len(tracking) > 1 and m >= 10:
                        ldr_df_dtw = _df_cache.get(leader, pd.DataFrame(stock_dfs[leader]))
                        curr_t_dtw = stock_dfs[leader][m]['time']
                        win_start  = stock_dfs[leader][0]['time']
                        to_rm = [s for s in tracking if s != leader and
                                 calculate_dtw_pearson(ldr_df_dtw,
                                     _df_cache.get(s, pd.DataFrame(stock_dfs[s])),
                                     win_start, curr_t_dtw) < p_dtw]
                        for s in to_rm: tracking.discard(s)

                    if wait_cnt >= p_wait:
                        if _risk_enabled and _daily_stops_b >= _max_daily_stops:
                            break
                        if _risk_enabled and _daily_entries_b >= _max_daily_entries:
                            break
                        _did_reentry = False
                        eligible = []
                        for sym in tracking:
                            if sym == leader: continue
                            df_wait   = [r for r in stock_dfs[sym][:m + 1] if r['time'] >= start_t]
                            if not df_wait: continue
                            df_wait_v = [r['volume'] for r in df_wait]
                            # OR 邏輯：v>=倍數均量 OR v>=絕對量 任一達標即放行（對齊 process_group_data）
                            if not any((first3_vol[sym] > 0 and v >= p_vmult * first3_vol[sym]) or v >= p_vabs
                                       for v in df_wait_v): continue
                            if (sum(df_wait_v) / len(df_wait_v) < p_wait_min_avg and
                                    max(df_wait_v) < p_wait_max_single): continue
                            r_now, p_now = stock_dfs[sym][m]['rise'], stock_dfs[sym][m]['close']
                            if not (sys_config.rise_lower_bound <= r_now <= sys_config.rise_upper_bound): continue
                            ldr_rise_now = stock_dfs[leader][m]['rise'] if leader in stock_dfs and m < len(stock_dfs[leader]) else None
                            if ldr_rise_now is not None and (ldr_rise_now - r_now) < getattr(sys_config, 'min_lag_pct', 0): continue
                            prev_close = stock_dfs[sym][0].get('昨日收盤價', 0) if stock_dfs[sym] else 0
                            hi_today   = stock_dfs[sym][m].get('highest', p_now)
                            if prev_close and prev_close > 0 and (hi_today - prev_close) / prev_close * 100 < getattr(sys_config, 'min_height_pct', 0): continue
                            _min_elig = getattr(sys_config, 'min_eligible_avg_vol', 0)
                            if _min_elig > 0 and _cum_vol.get(sym, 0) / (m + 1) < _min_elig: continue
                            _vmin_range = getattr(sys_config, 'volatility_min_range', 0)
                            if _vmin_range > 0 and m >= 10 and prev_close and prev_close > 0:
                                if (_run_high[sym] - _run_low[sym]) / prev_close * 100 < _vmin_range: continue
                            _ptol = getattr(sys_config, 'pullback_tolerance', 999)
                            if len(df_wait) >= 2 and df_wait[-1].get('rise', 0) > max(r.get('rise', 0) for r in df_wait[:-1]) + _ptol: continue
                            eligible.append({'sym': sym, 'rise': r_now, 'p_ent': p_now, 'hi': hi_today})

                        if eligible:
                            eligible.sort(key=lambda x: x['rise'])

                            # Mode A: 全部 eligible 各別進場（單筆勝率統計）
                            _fee_r = sys_config.transaction_fee * 0.01 * sys_config.transaction_discount * 0.01
                            for item_a in eligible:
                                p_a = item_a['p_ent']
                                shrs_a   = round((sys_config.capital_per_stock * 10000) / (p_a * 1000))
                                sell_a   = shrs_a * p_a * 1000
                                fee_a    = int(sell_a * _fee_r)
                                tax_a    = int(sell_a * sys_config.trading_tax * 0.01)
                                gap_a, tick_a = get_stop_loss_config(p_a)
                                hi_a     = item_a['hi'] or p_a
                                base_a   = hi_a + tick_a if (hi_a - p_a) * 1000 >= gap_a else p_a + gap_a / 1000
                                stop_a   = round(base_a + p_a * (p_sl_cushion / 100.0), 2)
                                _lup_a = stock_dfs[item_a['sym']][m].get('漲停價')
                                if _lup_a:
                                    _tlup = 0.01 if _lup_a < 10 else 0.05 if _lup_a < 50 else 0.1 if _lup_a < 100 else 0.5 if _lup_a < 500 else 1 if _lup_a < 1000 else 5
                                    if stop_a > _lup_a - 2 * _tlup: stop_a = _lup_a - 2 * _tlup
                                max_risk_a = (stop_a - p_a) * 1000
                                if max_risk_a > gap_a: continue
                                max_h_a  = num_bars - 1 if (m + p_hold_mins) >= 266 else (m + p_hold_mins)
                                for me_a in range(m + 1, num_bars):
                                    r_a = stock_dfs[item_a['sym']][me_a]
                                    if _trunc2(r_a['high']) >= _trunc2(stop_a) or me_a >= max_h_a:
                                        pe_a  = stop_a if r_a['high'] >= stop_a else r_a['close']
                                        bt_a  = shrs_a * pe_a * 1000
                                        pnl_a = sell_a - bt_a - fee_a - int(bt_a * _fee_r) - tax_a
                                        trades_a += 1; tp_sum_a += pnl_a
                                        if pnl_a > 0: wins_a += 1
                                        else: losses_a += 1
                                        break

                            # Mode B: DTW 篩選 → 唯一進場（單日損益）
                            if p_dtw <= 0:
                                eligible_dtw = list(eligible)
                            else:
                                ldr_df = _df_cache.get(leader, pd.DataFrame(stock_dfs[leader]))
                                curr_t = stock_dfs[leader][m]['time']
                                eligible_dtw = [
                                    item for item in eligible
                                    if calculate_dtw_pearson(ldr_df,
                                        _df_cache.get(item['sym'], pd.DataFrame(stock_dfs[item['sym']])),
                                        start_t, curr_t) >= p_dtw
                                ]

                            mode_b_exit_m   = m + 1
                            _mode_b_stop_loss = False
                            if eligible_dtw:
                                target      = eligible_dtw[0]
                                p_ent       = target['p_ent']
                                shrs        = round((sys_config.capital_per_stock * 10000) / (p_ent * 1000))
                                sell_total  = shrs * p_ent * 1000
                                _fee_r_b    = sys_config.transaction_fee * 0.01 * sys_config.transaction_discount * 0.01
                                ent_fee     = int(sell_total * _fee_r_b)
                                tax         = int(sell_total * sys_config.trading_tax * 0.01)
                                gap, tick   = get_stop_loss_config(p_ent)
                                hi_on_e     = target['hi'] or p_ent
                                base_stop   = hi_on_e + tick if (hi_on_e - p_ent) * 1000 >= gap else p_ent + gap / 1000
                                stop_p      = round(base_stop + p_ent * (p_sl_cushion / 100.0), 2)
                                _lup_p = stock_dfs[target['sym']][m].get('漲停價')
                                if _lup_p:
                                    _tlup = 0.01 if _lup_p < 10 else 0.05 if _lup_p < 50 else 0.1 if _lup_p < 100 else 0.5 if _lup_p < 500 else 1 if _lup_p < 1000 else 5
                                    if stop_p > _lup_p - 2 * _tlup: stop_p = _lup_p - 2 * _tlup
                                max_risk_p = (stop_p - p_ent) * 1000
                                if max_risk_p <= gap:
                                    _daily_entries_b += 1
                                    m_end       = num_bars - 1
                                    max_hold_m  = m_end if (m + p_hold_mins) >= 266 else (m + p_hold_mins)
                                    for m_exit in range(m + 1, m_end + 1):
                                        r_ex = stock_dfs[target['sym']][m_exit]
                                        if _trunc2(r_ex['high']) >= _trunc2(stop_p) or m_exit >= max_hold_m:
                                            _mode_b_stop_loss = (_trunc2(r_ex['high']) >= _trunc2(stop_p))
                                            p_exit    = stop_p if _mode_b_stop_loss else r_ex['close']
                                            buy_total = shrs * p_exit * 1000
                                            profit    = sell_total - buy_total - ent_fee - int(buy_total * _fee_r_b) - tax
                                            tp_sum += profit; day_pnl += profit; trades_count += 1
                                            if profit > 0: wins += 1
                                            else:          losses += 1
                                            mode_b_exit_m = m_exit
                                            break

                            if _mode_b_stop_loss:
                                _daily_stops_b += 1

                            if eligible_dtw and _mode_b_stop_loss:
                                # 停損後行為：讀 sys_config 決定是否 re-entry（對齊盤中 process_group_data）
                                _allow_re = sys_config.allow_reentry
                                _max_re   = sys_config.max_reentry_times
                                _lb_bars  = sys_config.reentry_lookback_candles
                                if not (_allow_re and reentry_count < _max_re):
                                    break  # 預設（allow_reentry=False）：停損終止，與原邏輯完全相同
                                # Re-entry 模式：往前回溯找新觸發信號
                                lb_s   = max(0, mode_b_exit_m - _lb_bars)
                                _found = False
                                for r_sym in valid_syms:
                                    for lb_m in range(lb_s, min(mode_b_exit_m + 1, len(stock_dfs[r_sym]))):
                                        lb_bar  = stock_dfs[r_sym][lb_m]
                                        _is_lu  = (round(lb_bar['high'], 2) >= round(lb_bar['漲停價'], 2) and
                                                   (lb_m == 0 or round(stock_dfs[r_sym][lb_m - 1]['high'], 2) < round(lb_bar['漲停價'], 2)))
                                        _lb_vol  = lb_bar.get('volume', 0)
                                        _lb_avgv = first3_vol.get(r_sym, 0)
                                        _is_pu  = (lb_bar.get('pct_increase', 0) >= p_lead and
                                                   (_lb_vol >= p_vabs or (_lb_avgv > 0 and _lb_vol >= p_vmult * _lb_avgv)))
                                        if _is_lu or _is_pu:
                                            reentry_count  += 1
                                            leader, tracking   = r_sym, {r_sym}
                                            in_wait, wait_cnt  = True, 0
                                            pull_up, limit_up  = not _is_lu, _is_lu
                                            start_t            = lb_bar['time']
                                            leader_rise_before_decline = lb_bar.get('highest', lb_bar.get('high', 0))
                                            _found = True; break
                                    if _found: break
                                if not _found:
                                    reentry_count += 1          # 消耗次數，進入獵人模式
                                    leader, tracking  = None, set()
                                    pull_up = limit_up = in_wait = False
                                    wait_cnt = 0
                                _did_reentry = True             # 不執行後面的 busy/reset

                            if not _did_reentry:
                                is_busy, exit_at = True, mode_b_exit_m

                        if not _did_reentry:
                            pull_up = limit_up = False
                            leader, tracking, in_wait, wait_cnt = None, set(), False, 0
                    elif leader and leader_rise_before_decline is not None and \
                            stock_dfs[leader][m]['high'] > leader_rise_before_decline:
                        # 突破前高：漲勢延續，中斷等待（對齊 process_group_data line 3554-3558）
                        leader_rise_before_decline = stock_dfs[leader][m].get('highest', stock_dfs[leader][m]['high'])
                        in_wait, wait_cnt = False, 0

        if day_pnl != 0:
            days_traded.add(t_date)
            if day_pnl > 0: winning_days.add(t_date)

        if trial is not None:
            trial.report(tp_sum, step_idx)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    pf            = (wins / losses) if losses > 0 else (99.9 if wins > 0 else 0)
    win_rate_a    = (wins_a / trades_a * 100) if trades_a > 0 else 0
    daily_win_rate = (len(winning_days) / len(days_traded) * 100) if days_traded else 0
    expectancy    = (tp_sum / trades_count) if trades_count > 0 else 0

    if trades_count < 10:
        ai_score = tp_sum * 0.01
    else:
        freq     = min(1.0, trades_count / 100) ** 0.5
        ai_score = tp_sum * (0.6 + 0.4 * freq)

    return {
        **params,
        'Total_PnL':      tp_sum,
        'NoFilter_PnL':   tp_sum_a,
        'WinRate':        win_rate_a,
        'Daily_WinRate':  daily_win_rate,
        'PF': pf, 'Expectancy': expectancy, 'Count': trades_count,
        'ai_score':       ai_score,
    }


# ---------------------------------------------------------
# v1.9.8.6 統一引擎：回歸直接呼叫 process_group_data
# ---------------------------------------------------------
def regression_evaluator(params, dates, cache, groups, dispo_list, trial=None, _precomp_cache=None):
    """回歸用的評估器：直接呼叫 process_group_data，保證與回測結果 100% 一致。

    Args:
        params: Optuna 建議的參數 dict（wait_mins, dtw_thresh, leader_pull, ...）
        dates: 日期列表
        cache: build_backtest_cache() 回傳的快取 {date: {symbol: DataFrame}}
        groups: matrix_dict_analysis（族群 dict）
        dispo_list: 處置股 dict {date: [symbols]}
        trial: Optuna trial 物件（用於剪枝）

    Returns:
        dict — 包含所有參數 + 績效指標（Total_PnL, WinRate, Count, ai_score, ...）
    """
    # 1. 將 Optuna 參數寫入 sys_config
    _param_map = {
        'dtw_thresh': 'similarity_threshold',
        'leader_pull': 'pull_up_pct_threshold',
        'follow_pull': 'follow_up_pct_threshold',
        'vol_mult': 'volume_multiplier',
        'vol_abs': 'min_volume_threshold',
        'wait_min_avg_vol': 'wait_min_avg_vol',
        'wait_max_single_vol': 'wait_max_single_vol',
        'sl_cushion_pct': 'sl_cushion_pct',
    }
    for p_key, cfg_attr in _param_map.items():
        if p_key in params:
            setattr(sys_config, cfg_attr, params[p_key])

    # cutoff_mins → cutoff_time_mins（分鐘偏移量，270=13:30=不截止）
    p_cutoff_raw = params.get('cutoff_mins', 270)
    if isinstance(p_cutoff_raw, str):
        _h, _m = map(int, p_cutoff_raw.split(':'))
        sys_config.cutoff_time_mins = (_h - 9) * 60 + _m
    else:
        sys_config.cutoff_time_mins = int(p_cutoff_raw)

    p_wait = params['wait_mins']
    p_hold = params.get('hold_mins', 240)

    # 2. 迴圈跑 dates × groups
    tp_sum = 0
    trades_count = 0
    wins, losses = 0, 0
    days_traded, winning_days = set(), set()

    for step_idx, t_date in enumerate(dates):
        day_data = cache.get(t_date, {})
        if not day_data:
            continue
        day_pnl = 0

        for grp, syms in groups.items():
            # ⚡ 優先使用預計算快取，跳過 process_group_data 的 setup 階段
            _pc = _precomp_cache.get((grp, t_date)) if _precomp_cache else None
            if _pc is not None:
                stock_collection = _pc['stock_collection']
            else:
                dispo_today = dispo_list.get(t_date, []) if isinstance(dispo_list, dict) else dispo_list
                valid_syms = [s for s in syms if s not in dispo_today and s in day_data]
                if len(valid_syms) < 2:
                    continue
                stock_collection = {s: day_data[s] for s in valid_syms}

            tp, _tc, trades_hist, _ = process_group_data(
                stock_collection, p_wait, p_hold, groups, t_date,
                verbose=False, headless=True, _precomp=_pc
            )

            if tp is not None:
                day_pnl += tp
                for t in trades_hist:
                    trades_count += 1
                    if t.get('profit', 0) > 0:
                        wins += 1
                    else:
                        losses += 1

        tp_sum += day_pnl
        if day_pnl != 0:
            days_traded.add(t_date)
        if day_pnl > 0:
            winning_days.add(t_date)

        # Optuna 中途剪枝
        if trial is not None:
            trial.report(tp_sum, step_idx)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    # 3. 計算指標
    pf = (wins / losses) if losses > 0 else (99.9 if wins > 0 else 0)
    expectancy = (tp_sum / trades_count) if trades_count > 0 else 0
    daily_win_rate = (len(winning_days) / len(days_traded) * 100) if days_traded else 0
    win_rate = (wins / trades_count * 100) if trades_count > 0 else 0

    # 4. AI 複合目標分數：以總損益為主，頻率因子加權
    if trades_count < 10:
        ai_score = tp_sum * 0.01
    else:
        freq = min(1.0, trades_count / 100) ** 0.5
        ai_score = tp_sum * (0.6 + 0.4 * freq)

    return {
        **params,
        'Total_PnL': tp_sum,
        'NoFilter_PnL': tp_sum,
        'WinRate': win_rate,
        'Daily_WinRate': daily_win_rate,
        'PF': pf, 'Expectancy': expectancy,
        'Count': trades_count,
        'ai_score': ai_score
    }




class RunAnalysisTaskThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()
    
    def __init__(self, search_space, total_combinations, parent=None):
        super().__init__(parent)
        self.search_space = search_space
        self.total_combinations = total_combinations
        self.n_trials = 5000
        
    def run(self):
        import optuna

        try:
            db_dir = "量能分析數據庫"
            db_path = os.path.join(db_dir, "analysis_data.db")
            if not os.path.exists(db_path): 
                return self.log_signal.emit("❌ 找不到數據庫，請先執行「資料採集」。")
            
            self.log_signal.emit("📥 步驟 1/3: 正在將大數據庫載入記憶體 (只需執行一次)...")

            # ── 建立 list-of-dicts 快取（fast_evaluator 使用，直接 dict 存取無 pandas 開銷）
            conn = sqlite3.connect(db_path)
            _raw_df = pd.read_sql('SELECT * FROM intraday_kline', conn)
            conn.close()
            for _c in ['high', '漲停價', 'pct_increase', 'volume', 'close', 'open', 'low',
                       '昨日收盤價', 'rise', 'highest']:
                if _c in _raw_df.columns:
                    _raw_df[_c] = pd.to_numeric(_raw_df[_c], errors='coerce')
            lod_cache: dict = {}
            for (_d, _s), _grp in _raw_df.groupby(['date', 'symbol']):
                lod_cache.setdefault(_d, {})[_s] = _grp.sort_values('time').to_dict('records')

            unique_dates = sorted(lod_cache.keys())

            # ── 建立 DataFrame 快取（regression_evaluator 全量驗證使用）
            backtest_cache = build_backtest_cache(db_path=db_path, table_name='intraday_kline')

            mat = load_matrix_dict_analysis()
            dispo = load_disposition_stocks()

            # 🚀 預計算族群×日期 setup（僅供全量驗證的 regression_evaluator 使用）
            self.log_signal.emit("⚙️ 預計算全量驗證快取（只需一次）...")
            _precomp_cache = build_regression_precomp(backtest_cache, mat, dispo)
            self.log_signal.emit(f"✅ 快取完成：{len(_precomp_cache)} 個族群-日期組合")

            # 🚀 方案二：日期採樣加速
            _SAMPLE_RATIO  = 0.70  # 每個 trial 用 70% 日期
            _MIN_SAMPLE    = max(5, len(unique_dates) // 3)  # 最少取 1/3，避免資料太少失去意義
            _FULL_VERIFY_K = 10    # 最終前 10 名做全量驗證（原 30，降低以加速最後 5%）

            self.log_signal.emit(f"🤖 步驟 2/3: 啟動 Optuna AI 優化引擎，進行 {self.n_trials} 次智能迭代（每 trial 採樣 {int(_SAMPLE_RATIO*100)}% 日期，fast 模式）...")

            # 👇 建立 Optuna 目標函數（fast_evaluator：list-of-dicts 直接存取，~10x 更快）
            def objective(trial):
                import random as _rnd
                params = {}
                for k, v in self.search_space.items():
                    if len(v) == 1:
                        params[k] = v[0]
                    elif isinstance(v[0], str):
                        params[k] = trial.suggest_categorical(k, v)
                    else:
                        is_int = all(isinstance(x, int) for x in v)
                        p_min, p_max = min(v), max(v)
                        if is_int:
                            step = v[1] - v[0] if len(v) > 1 else 1
                            params[k] = trial.suggest_int(k, p_min, p_max, step=step)
                        else:
                            params[k] = round(trial.suggest_float(k, p_min, p_max), 2)

                # 🚀 日期採樣：每個 trial 只用 45% 的日期
                k_dates = max(_MIN_SAMPLE, int(len(unique_dates) * _SAMPLE_RATIO))
                sampled_dates = sorted(_rnd.sample(unique_dates, k_dates))

                # 🚀 fast_evaluator：直接 dict 存取，不走 DataFrame pipeline
                res = fast_evaluator(params, sampled_dates, lod_cache, mat, dispo, trial=trial)

                for key, val in res.items():
                    trial.set_user_attr(key, val)

                scale = len(unique_dates) / len(sampled_dates)
                return res['ai_score'] * scale

            # 建立 Optuna 研究 (設定為求極大值)
            # 使用 MedianPruner 進行中途剪枝
            study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=10,  # 原 20 → 10：更早啟動剪枝
                    n_warmup_steps=5,     # 忽略前 5 步（避免高方差策略被誤殺）
                )
            )
            
            # 綁定進度條（基於總組合數）
            completed = [0]
            best_score = [-float('inf')]
            _tc_single = self.total_combinations
            _combos_per_trial_s = max(1, _tc_single // self.n_trials) if _tc_single > 0 else 1

            def progress_callback(study, trial):
                completed[0] += 1
                _done_s = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                _pruned_s = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                combos_s = (_done_s * _combos_per_trial_s + _pruned_s * _combos_per_trial_s * 3)
                pct_s = min(95, int(combos_s / max(1, _tc_single) * 95)) if _tc_single > 0 else min(95, int(completed[0] / self.n_trials * 95))
                self.progress_signal.emit(pct_s)

                try:
                    if completed[0] == 1:
                        self.log_signal.emit("🔍 [AI 引擎] 啟動隨機初始抽樣，探勘參數地形...")
                    elif study.best_value > best_score[0] and study.best_value > 0:
                        best_score[0] = study.best_value
                        self.log_signal.emit(f"🏆 [AI 引擎] 突破高點！最佳總損益分數 {int(best_score[0]):,} 元")
                    elif completed[0] % 40 == 0:
                        _fmt_tc = f"{_tc_single / 1e8:.1f} 億" if _tc_single >= 1e8 else f"{_tc_single / 1e4:.0f} 萬" if _tc_single >= 1e4 else f"{_tc_single:,}"
                        _fmt_ex = f"{combos_s / 1e8:.1f} 億" if combos_s >= 1e8 else f"{combos_s / 1e4:.0f} 萬" if combos_s >= 1e4 else f"{combos_s:,}"
                        self.log_signal.emit(f"⏳ 已排除 {_fmt_ex} / {_fmt_tc} 種組合 | 完成 {_done_s} 筆、剪枝 {_pruned_s} 筆")
                except Exception:
                    pass

            # ─────────────────────────────────────────────────────────────
            # 🚀 方案三：多進程平行（_fast_worker.py，不載入主程式，不會崩潰）
            #   每個 worker 各自從 DB 讀資料、跑 Optuna，完全獨立。
            #   N_WORKERS = CPU 核心數（最多 4），加速約 2-4x。
            # ─────────────────────────────────────────────────────────────
            _USE_PARALLEL = True
            _N_WORKERS = min(4, max(2, (os.cpu_count() or 2) - 1))
            goto_single = False   # 初始化，防止 locals().get() 脆弱查詢
            # ─────────────────────────────────────────────────────────────

            if _USE_PARALLEL and _N_WORKERS > 1:
                import subprocess, json as _pjson, time as _time_mod

                trials_per_worker = max(60, self.n_trials // _N_WORKERS)
                self.log_signal.emit(f"🚀 [多核模式] 啟動 {_N_WORKERS} 個平行 worker，每個跑 {trials_per_worker} trials（共 {trials_per_worker * _N_WORKERS} trials）...")

                tmpdir = os.path.join(db_dir, "_parallel_tmp")
                os.makedirs(tmpdir, exist_ok=True)

                # 快照固定 sys_config 值（不隨 Optuna 參數改變的欄位）
                _cfg_snap = {
                    'rise_lower_bound':    getattr(sys_config, 'rise_lower_bound',    -10),
                    'rise_upper_bound':    getattr(sys_config, 'rise_upper_bound',    9.6),
                    'capital_per_stock':   getattr(sys_config, 'capital_per_stock',   100),
                    'transaction_fee':     getattr(sys_config, 'transaction_fee',     0.1425),
                    'transaction_discount':getattr(sys_config, 'transaction_discount', 60),
                    'trading_tax':         getattr(sys_config, 'trading_tax',         0.3),
                    'below_50':            getattr(sys_config, 'below_50',            500),
                    'price_gap_50_to_100': getattr(sys_config, 'price_gap_50_to_100', 1000),
                    'price_gap_100_to_500':getattr(sys_config, 'price_gap_100_to_500',2000),
                    'price_gap_500_to_1000':getattr(sys_config,'price_gap_500_to_1000',3000),
                    'price_gap_above_1000':getattr(sys_config, 'price_gap_above_1000',5000),
                    'min_lag_pct':              getattr(sys_config, 'min_lag_pct',              0),
                    'min_height_pct':           getattr(sys_config, 'min_height_pct',           0),
                    'min_eligible_avg_vol':     getattr(sys_config, 'min_eligible_avg_vol',     0),
                    'volatility_min_range':     getattr(sys_config, 'volatility_min_range',     0),
                    'pullback_tolerance':       getattr(sys_config, 'pullback_tolerance',      999),
                    # Re-entry 參數：盤中、回歸、回測三路共用，由 UI 控制
                    'allow_reentry':            sys_config.allow_reentry,
                    'max_reentry_times':        sys_config.max_reentry_times,
                    'reentry_lookback_candles': sys_config.reentry_lookback_candles,
                    # 風控熔斷
                    'max_daily_stops':          getattr(sys_config, 'max_daily_stops', 3),
                    'risk_control_enabled':     getattr(sys_config, 'risk_control_enabled', True),
                }

                # 族群 dict 需序列化（key 可能是 int，轉成 str）
                _groups_serial = {str(k): list(v) for k, v in mat.items()}
                _dispo_serial  = dispo if isinstance(dispo, dict) else {}

                # 為每個 worker 準備獨立參數檔（含進度回報路徑）
                _base_payload = {
                    'db_path':           db_path,
                    'search_space':      self.search_space,
                    'n_trials':          trials_per_worker,
                    'unique_dates':      unique_dates,
                    'sample_ratio':      _SAMPLE_RATIO,
                    'groups':            _groups_serial,
                    'dispo':             _dispo_serial,
                    'sys_config_snapshot': _cfg_snap,
                }

                # 找到 _fast_worker.py 的路徑
                _worker_base = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
                _worker_script = os.path.join(_worker_base, '_fast_worker.py')
                if not os.path.exists(_worker_script):
                    self.log_signal.emit("⚠️ 找不到 _fast_worker.py，退回單進程模式")
                    goto_single = True
                else:
                    goto_single = False
                    # 啟動子進程（每個 worker 獨立參數檔 + 進度檔）
                    procs, result_files, progress_files = [], [], []
                    for _wi in range(_N_WORKERS):
                        _out = os.path.join(tmpdir, f"results_{_wi}.json")
                        _prog = os.path.join(tmpdir, f"progress_{_wi}.json")
                        result_files.append(_out)
                        progress_files.append(_prog)
                        for _f in [_out, _prog]:
                            if os.path.exists(_f): os.remove(_f)
                        _payload = {**_base_payload, 'seed': _wi, 'progress_file': _prog}
                        _af = os.path.join(tmpdir, f"worker_args_{_wi}.json")
                        with open(_af, 'w', encoding='utf-8') as _f:
                            _pjson.dump(_payload, _f, ensure_ascii=False, default=str)
                        _cmd = [sys.executable, _worker_script,
                                f'--args-file={_af}',
                                f'--output-file={_out}']
                        procs.append(subprocess.Popen(_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
                        self.log_signal.emit(f"  ▶ Worker {_wi+1}/{_N_WORKERS} 已啟動 (PID {procs[-1].pid})")

                    # 等待所有 worker 完成，輪詢進度檔更新進度條
                    total_procs = len(procs)
                    total_trials = trials_per_worker * _N_WORKERS
                    _tc = self.total_combinations
                    _combos_per_trial = max(1, _tc // total_trials) if _tc > 0 else 1
                    _prune_multiplier = 3  # 剪枝一次等同排除 3 倍組合
                    _poll_cnt = 0
                    _last_log_pct = -1
                    while any(p.poll() is None for p in procs):
                        # 讀取所有 worker 進度
                        _total_done, _total_pruned = 0, 0
                        for _pf in progress_files:
                            try:
                                if os.path.exists(_pf):
                                    with open(_pf, encoding='utf-8') as _f:
                                        _pd = _pjson.load(_f)
                                    _total_done += _pd.get('done', 0)
                                    _total_pruned += _pd.get('pruned', 0)
                            except Exception: pass
                        # 已探索組合數 = 完成數 × 每trial涵蓋量 + 剪枝數 × 倍率
                        combos_explored = (_total_done * _combos_per_trial +
                                           _total_pruned * _combos_per_trial * _prune_multiplier)
                        pct = min(95, int(combos_explored / max(1, _tc) * 95)) if _tc > 0 else min(95, int((_total_done + _total_pruned) / max(1, total_trials) * 95))
                        self.progress_signal.emit(pct)

                        _poll_cnt += 1
                        # 每 10 秒發一次狀態 log
                        if _poll_cnt % 20 == 1:
                            elapsed = int(_poll_cnt * 0.5)
                            _fmt_tc = f"{_tc / 1e8:.1f} 億" if _tc >= 1e8 else f"{_tc / 1e4:.0f} 萬" if _tc >= 1e4 else f"{_tc:,}"
                            _fmt_ex = f"{combos_explored / 1e8:.1f} 億" if combos_explored >= 1e8 else f"{combos_explored / 1e4:.0f} 萬" if combos_explored >= 1e4 else f"{combos_explored:,}"
                            self.log_signal.emit(
                                f"⏳ 已排除 {_fmt_ex} / {_fmt_tc} 種組合 | "
                                f"完成 {_total_done} 筆、剪枝 {_total_pruned} 筆（耗時 {elapsed} 秒）"
                            )
                        _time_mod.sleep(0.5)

                    # 收集所有 worker 結果
                    all_worker_results = []
                    for _wi2, _rf in enumerate(result_files):
                        if os.path.exists(_rf):
                            with open(_rf, encoding='utf-8') as _f:
                                _wres = _pjson.load(_f)
                                all_worker_results.extend(_wres)
                        else:
                            # 輸出 worker stderr 幫助 debug
                            _stderr = procs[_wi2].stderr.read().decode('utf-8', errors='replace')
                            if _stderr.strip():
                                self.log_signal.emit(f"  ⚠️ Worker {_wi2+1} 錯誤：{_stderr[:200]}")

                    self.log_signal.emit(f"✅ {_N_WORKERS} 個 worker 完成，共收集 {len(all_worker_results)} 筆試驗結果")

                    # 對前 _FULL_VERIFY_K 名做全量驗證（regression_evaluator）
                    all_worker_results.sort(key=lambda r: float(r.get('ai_score', -999)), reverse=True)
                    self.log_signal.emit(f"🔍 [AI 引擎] 對前 {_FULL_VERIFY_K} 名候選進行全樣本驗證（{len(unique_dates)} 天）...")
                    verified_results = []
                    _search_keys = set(self.search_space.keys())
                    _verify_total = min(_FULL_VERIFY_K, len(all_worker_results))
                    for _vi, _vr in enumerate(all_worker_results[:_FULL_VERIFY_K]):
                        _p_verify = {k: v for k, v in _vr.items() if k in _search_keys}
                        _full = regression_evaluator(_p_verify, unique_dates, backtest_cache, mat, dispo,
                                                     trial=None, _precomp_cache=_precomp_cache)
                        _full['_verified'] = True
                        verified_results.append(_full)
                        self.progress_signal.emit(95 + int((_vi + 1) / max(1, _verify_total) * 5))
                    self.progress_signal.emit(100)

                    results = verified_results + all_worker_results[_FULL_VERIFY_K:]

            if not _USE_PARALLEL or _N_WORKERS <= 1 or goto_single:
                # 單進程退路（_USE_PARALLEL=False 或找不到 worker）
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                # n_jobs=1：regression_evaluator 修改全域 sys_config，多執行緒會有競爭問題
                study.optimize(objective, n_trials=self.n_trials, n_jobs=1, callbacks=[progress_callback])

                # 🚀 方案二：對前 N 名候選做全量日期驗證，確保最終報告數字準確
                self.log_signal.emit(f"🔍 [AI 引擎] 對前 {_FULL_VERIFY_K} 名候選進行全樣本驗證（{len(unique_dates)} 天）...")
                complete_trials = sorted(
                    [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                    key=lambda t: t.value if t.value is not None else -999, reverse=True
                )
                verified_results = []
                for vt in complete_trials[:_FULL_VERIFY_K]:
                    p_verify = dict(vt.params)
                    full_res = regression_evaluator(p_verify, unique_dates, backtest_cache, mat, dispo, trial=None, _precomp_cache=_precomp_cache)
                    full_res['_verified'] = True
                    verified_results.append(full_res)

                results = verified_results + [
                    t.user_attrs for t in complete_trials[_FULL_VERIFY_K:]
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]

            if not results:
                return self.log_signal.emit("❌ 分析完成，但所有參數皆被 AI 淘汰或未產生交易。")

            res_df = pd.DataFrame(results)
            
            # ====== 以下產圖表的邏輯「完全不用改」 ======
            import plotly.graph_objects as go
            import json, plotly
            import numpy as np
            from scipy import stats
            
            # 1. 泡泡大小基數：依交易筆數決定，最小給 5，讓小筆數也能看見
            res_df['Display_Size'] = res_df['Count'].apply(lambda c: c if c >= 5 else 5)
            
            # 【移除所有 Jitter 邏輯，回歸 100% 真實座標】
            x_vals = res_df['Total_PnL'].tolist()
            y_vals = res_df['WinRate'].tolist()
            
            size_vals = res_df['Display_Size'].tolist()
            color_vals = res_df['wait_mins'].tolist()
            
            # customdata 依然保留所有真實資訊供 Hover 顯示
            custom_data_vals = res_df[['wait_mins', 'dtw_thresh', 'leader_pull', 'follow_pull', 'vol_mult', 'vol_abs', 'wait_min_avg_vol', 'wait_max_single_vol', 'Count', 'PF', 'Total_PnL', 'WinRate']].values.tolist()
            
            # 調整 sizeref，數字越小泡泡越大
            max_cnt = max(size_vals) if size_vals else 1
            sz_ref = 2. * max_cnt / (25.**2) if max_cnt > 0 else 1
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='markers',
                marker=dict(
                    size=size_vals, sizemode='area', sizeref=sz_ref, sizemin=8,
                    color=color_vals, colorscale='Plasma', showscale=True,
                    colorbar=dict(title=dict(text="等待(分)", font=dict(color="white")), tickfont=dict(color="white")),
                    line=dict(width=0.5, color='rgba(255, 255, 255, 0.2)'), 
                    # 💡 學術解法：使用半透明 (0.45)，重疊處自然會變亮變深，代表密度
                    opacity=0.45  
                ),
                customdata=custom_data_vals,
                hovertemplate=(
                    "<b>💰 利潤: %{x:.0f}</b><br><b>🏆 勝率: %{y:.1f}%</b><br>📊 交易筆數: %{customdata[8]} 筆<br>" +
                    "📈 獲利因子: %{customdata[9]:.2f}<br><hr><b>【參數組合】</b><br>" +
                    "等待時間: %{customdata[0]} 分<br>DTW門檻: %{customdata[1]}<br>領漲幅: %{customdata[2]}%<br>" +
                    "跟漲幅: %{customdata[3]}%<br>均量倍數: %{customdata[4]}<br>絕對量: %{customdata[5]} 張<br>" +
                    "🌟等待均量下限: %{customdata[6]} 張<br>🌟等待單根爆量下限: %{customdata[7]} 張<extra></extra>"
                )
            ))
            
            fig.update_layout(
                title=f"🎯 1. AI 智能優化氣泡圖 (已採樣 {len(res_df)} 組優質樣本)",
                xaxis_title="總利潤 (Total PnL)", yaxis_title="單筆勝率 (Win Rate %)",
                font=dict(family="Microsoft JhengHei", size=13, color="white"),
                paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E',
                margin=dict(t=60, b=40, l=60, r=60),
                hoverlabel=dict(bgcolor="#2C2C2C", font_size=14, font_family="Microsoft JhengHei"),
                dragmode='pan' 
            )
            graph_json = fig.to_json()

            # --- 迴歸分析 (維持原樣) ---
            reg_json = "{}"
            df_reg = res_df[res_df['Count'] > 0].copy()
            if len(df_reg) >= 10: 
                X = pd.DataFrame()
                X['wait_mins'] = df_reg['wait_mins']
                X['wait_mins_sq'] = df_reg['wait_mins'] ** 2 
                X['dtw_thresh'] = df_reg['dtw_thresh']
                X['leader_pull'] = df_reg['leader_pull']
                X['follow_pull'] = df_reg['follow_pull']
                X['vol_mult'] = df_reg['vol_mult']
                X['vol_abs'] = df_reg['vol_abs']
                X['vol_interact'] = df_reg['vol_mult'] * df_reg['vol_abs']
                X['wait_min_avg_vol'] = df_reg['wait_min_avg_vol']
                X['wait_max_single_vol'] = df_reg['wait_max_single_vol']
                
                # 🚀 補上三個新變數參與迴歸運算
                if 'sl_cushion_pct' in df_reg.columns: X['sl_cushion_pct'] = df_reg['sl_cushion_pct']
                if 'cutoff_mins' in df_reg.columns: X['cutoff_mins'] = df_reg['cutoff_mins']
                if 'hold_mins' in df_reg.columns: X['hold_mins'] = df_reg['hold_mins']
                
                Y = df_reg['Expectancy']
                X_valid = X.loc[:, X.nunique() > 1]
                
                if not X_valid.empty:
                    try:
                        X_norm = (X_valid - X_valid.mean()) / (X_valid.std() + 1e-9)
                        Y_norm = (Y - Y.mean()) / (Y.std() + 1e-9)
                        X_mat = np.column_stack((np.ones(len(X_norm)), X_norm.values))
                        Y_vec = Y_norm.values
                        
                        XTX_inv = np.linalg.inv(X_mat.T.dot(X_mat))
                        beta = XTX_inv.dot(X_mat.T).dot(Y_vec)
                        
                        predictions = X_mat.dot(beta)
                        sse = np.sum((Y_vec - predictions) ** 2)
                        sst = np.sum((Y_vec - np.mean(Y_vec)) ** 2)
                        rsquared = 1 - (sse / sst) if sst != 0 else 0
                        
                        # 🚀 計算 Adjusted R-squared (調整後判定係數)
                        n_samples = len(Y_vec)
                        k_features = X_valid.shape[1]
                        if n_samples > k_features + 1:
                            adj_rsquared = 1 - (1 - rsquared) * (n_samples - 1) / (n_samples - k_features - 1)
                        else:
                            adj_rsquared = rsquared
                            
                        dof = n_samples - X_mat.shape[1]
                        p_values = [1.0] * len(beta)
                        if dof > 0:
                            mse = sse / dof
                            se = np.sqrt(np.diagonal(XTX_inv) * mse)
                            t_stats = beta / (se + 1e-9)
                            p_values = [2 * (1 - stats.t.cdf(np.abs(t), dof)) for t in t_stats]
                        
                        feat_names = X_valid.columns.tolist()
                        beta_feats = beta[1:].tolist()
                        p_feats = p_values[1:]
                        
                        name_map = {
                            'wait_mins': '等待時間', 'wait_mins_sq': '等待時間(甜蜜點)',
                            'dtw_thresh': 'DTW門檻', 'leader_pull': '領漲幅',
                            'follow_pull': '跟漲幅', 'vol_mult': '均量倍數',
                            'vol_abs': '絕對量', 'vol_interact': '均量與絕對量(交互作用)',
                            'wait_min_avg_vol': '等待均量下限', 'wait_max_single_vol': '單根爆量下限',
                            'sl_cushion_pct': '停損緩衝空間', 'cutoff_mins': '尾盤停止時間', 'hold_mins': '固定持倉時間'
                        }
                        plot_names = [name_map.get(n, n) for n in feat_names]
                        bar_colors = ['#2ECC71' if p < 0.05 else '#E74C3C' if p < 0.1 else '#555555' for p in p_feats]
                        
                        reg_fig = go.Figure(data=[go.Bar(
                            x=beta_feats, y=plot_names, orientation='h', marker_color=bar_colors,
                            text=[f"P值: {p:.3f} | 影響力: {b:.2f}" for p, b in zip(p_feats, beta_feats)],
                            hoverinfo="text"
                        )])
                        
                        reg_fig.update_layout(
                            title=f"📐 2. 多元迴歸：AI 探勘樣本之參數影響力 (R² = {rsquared:.2f} | 調整後 Adj R² = {adj_rsquared:.2f})",
                            paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E',
                            font=dict(family="Microsoft JhengHei", size=13, color="white"),
                            margin=dict(t=80, b=40, l=180, r=40), # 稍微把左邊距 l 放大到 180，避免字太長被切掉
                            yaxis=dict(tickmode='linear', dtick=1), # 🚀 關鍵修復：強制每 1 單位畫一個標籤，絕不隱藏！
                            dragmode='pan'
                        )
                        reg_json = reg_fig.to_json()
                    except Exception: pass

            # --- 產出 HTML ---
            top_df = res_df.sort_values('Total_PnL', ascending=False)
            table_data_json = top_df.to_json(orient='records')
            
            out_file = os.path.abspath(os.path.join(db_dir, "grid_search_result.html"))
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>策略超參數網格搜索分析報告</title>
                <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
                <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
                <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
                <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
                <style>
                    body {{ margin: 0; background-color: #121212; color: white; font-family: -apple-system, sans-serif; overflow-x: hidden; }} 
                    #plot-container {{ width: 100vw; height: 50vh; background-color: #1E1E1E; border-bottom: 2px solid #444; }}
                    #reg-container {{ width: 100vw; height: 50vh; background-color: #1E1E1E; border-bottom: 2px solid #444; }}
                    #table-container {{ padding: 20px; height: 40vh; overflow-y: auto; background-color: #1E1E1E; }}
                    table.dataTable {{ color: white !important; }}
                    table.dataTable thead th {{ border-bottom: 1px solid #555; color: #F1C40F; }}
                    table.dataTable tbody tr {{ background-color: #1E1E1E !important; border-bottom: 1px solid #333; }}
                    table.dataTable tbody tr:nth-child(even) {{ background-color: #2C2C2C !important; }}
                    table.dataTable tbody tr:hover {{ background-color: #34495E !important; }}
                    .dataTables_wrapper .dataTables_length, .dataTables_wrapper .dataTables_filter, .dataTables_wrapper .dataTables_info, .dataTables_wrapper .dataTables_paginate {{ color: white !important; }}
                </style>
            </head>
            <body>
                <div id="plot-container"><div id="plot-div" style="width: 100%; height: 100%;"></div></div>
                <div id="reg-container" style="display: none;"><div id="reg-div" style="width: 100%; height: 100%;"></div></div>
                <div id="table-container">
                    <h2 style="margin-top:0; color:#2ECC71;">🏆 3. AI 最佳化排行榜 (前 100 名)</h2>
                    <table id="resultTable" class="display" style="width:100%">
                        <thead>
                            <tr>
                                <th>等待(分)</th><th>DTW</th><th>領漲(%)</th><th>跟漲(%)</th><th>均量倍數</th><th>絕對量</th>
                                <th>等待均量(張)</th><th>單根爆量(張)</th>
                                <th>緩衝(%)</th><th>停觸(分)</th><th>持倉(分)</th>
                                <th>總利潤</th><th>全進場利潤</th><th>筆勝率</th><th>日勝率</th><th>PF</th><th>期望值</th><th>筆數</th>
                            </tr>
                        </thead>
                    </table>
                </div>
                <script>
                    window.onerror = function(msg, url, line) {{ document.getElementById('plot-container').innerHTML = "<h2 style='color:#E74C3C;'>❌ 系統錯誤</h2><p>" + msg + " (Line: " + line + ")</p>"; }};
                    $(document).ready(function() {{
                        // 🚀 修正 1：將 displayModeBar: False 改為 false
                        Plotly.newPlot('plot-div', {graph_json}.data, {graph_json}.layout, {{displayModeBar: false, responsive: true, scrollZoom: true}});
                        var regData = {reg_json};
                        if (regData && Object.keys(regData).length > 0) {{
                            document.getElementById('reg-container').style.display = 'block';
                            Plotly.newPlot('reg-div', regData.data, regData.layout, {{displayModeBar: false, responsive: true}});
                        }}
                        $('#resultTable').DataTable({{
                            data: {table_data_json},
                            // 🚀 修正 2：加入 defaultContent 容錯機制，防止 JSON 缺漏欄位時崩潰
                            columns: [
                                {{ data: 'wait_mins', defaultContent: '-' }}, 
                                {{ data: 'dtw_thresh', defaultContent: '-' }}, 
                                {{ data: 'leader_pull', defaultContent: '-' }},
                                {{ data: 'follow_pull', defaultContent: '-' }}, 
                                {{ data: 'vol_mult', defaultContent: '-' }}, 
                                {{ data: 'vol_abs', defaultContent: '-' }},
                                {{ data: 'wait_min_avg_vol', defaultContent: '-' }}, 
                                {{ data: 'wait_max_single_vol', defaultContent: '-' }},
                                {{ data: 'sl_cushion_pct', defaultContent: '-' }},
                                {{ data: 'cutoff_mins', defaultContent: '-' }},
                                {{ data: 'hold_mins', defaultContent: '-' }},
                                {{ data: 'Total_PnL', defaultContent: '0', render: function(data) {{ return '<span style="color:' + (data >= 0 ? '#2ECC71' : '#E74C3C') + '; font-weight:bold;">' + $.fn.dataTable.render.number(',', '.', 0, '$').display(data) + '</span>'; }} }},
                                {{ data: 'NoFilter_PnL', defaultContent: '0', render: function(data) {{ return '<span style="color:' + (data >= 0 ? '#2ECC71' : '#E74C3C') + ';">' + $.fn.dataTable.render.number(',', '.', 0, '$').display(data) + '</span>'; }} }},
                                {{ data: 'WinRate', defaultContent: '0', render: function(data) {{ return Number(data).toFixed(2) + '%'; }} }},
                                {{ data: 'Daily_WinRate', defaultContent: '0', render: function(data) {{ return Number(data).toFixed(2) + '%'; }} }},
                                {{ data: 'PF', defaultContent: '0', render: function(data) {{ return Number(data).toFixed(2); }} }},
                                {{ data: 'Expectancy', defaultContent: '0', render: function(data) {{ return '<span style="color:' + (data >= 0 ? '#2ECC71' : '#E74C3C') + ';">' + Number(data).toFixed(0) + '</span>'; }} }},
                                {{ data: 'Count', defaultContent: '0' }}
                            ],
                            order: [[11, 'desc']], pageLength: 100
                        }});
                    }});
                </script>
            </body>
            </html>
            """
            with open(out_file, "w", encoding="utf-8") as f: f.write(html_content)
                
            self.log_signal.emit(f"✅ 實戰級 AI 分析完成！正在開啟內建分析視窗...")
            if hasattr(self.parent(), 'ui_dispatcher'): self.parent().ui_dispatcher.show_analysis_window.emit(os.path.abspath(out_file))
            else:
                import __main__
                if hasattr(__main__, 'ui_dispatcher'): __main__.ui_dispatcher.show_analysis_window.emit(os.path.abspath(out_file))
            
        except Exception as e: 
            import traceback
            self.log_signal.emit(f"❌ 分析出錯：{e}\n{traceback.format_exc()}")
        finally: 
            self.finished_signal.emit()

class FetchAnalysisDataThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, days_to_fetch):
        super().__init__()
        self.days_to_fetch = days_to_fetch

    def fetch_kbars_smart(self, api, contract, start_str, end_str):
        start_dt = pd.to_datetime(start_str)
        end_dt = pd.to_datetime(end_str)
        
        if (end_dt - start_dt).days <= 45:
            for attempt in range(3):
                shioaji_limiter.wait_and_consume()
                try:
                    kbars = api.kbars(contract, start=start_str, end=end_str)
                    df = pd.DataFrame({**kbars})
                    if not df.empty: return df
                except Exception: time_module.sleep(1)
            return pd.DataFrame()
        
        dfs = []
        curr_start = start_dt
        while curr_start <= end_dt:
            shioaji_limiter.wait_and_consume()
            curr_end = min(curr_start + pd.Timedelta(days=30), end_dt) 
            for attempt in range(3):
                try:
                    kbars = api.kbars(contract, start=curr_start.strftime("%Y-%m-%d"), end=curr_end.strftime("%Y-%m-%d"))
                    df = pd.DataFrame({**kbars})
                    if not df.empty: dfs.append(df); break
                except Exception: time_module.sleep(1)
            curr_start = curr_end + pd.Timedelta(days=1)
            time_module.sleep(0.05)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def run(self):
        db_folder = "量能分析數據庫"
        os.makedirs(db_folder, exist_ok=True)
        db_path = os.path.join(db_folder, "analysis_data.db")
        
        for f in glob.glob(os.path.join(db_folder, "*.html")) + glob.glob(os.path.join(db_folder, "*.json")):
            try: os.remove(f)
            except Exception: pass

        self.log_signal.emit(f"<span style='color:#9C27B0;'>正在登入 Shioaji API（目標：近 {self.days_to_fetch} 日策略參數資料）...</span>")
        
        api = sys_state.api
        if not getattr(api, 'positions', None): 
            api.login(api_key=shioaji_logic.TEST_API_KEY, secret_key=shioaji_logic.TEST_API_SECRET, contracts_timeout=10000)
            
        all_symbols, _ = load_target_symbols()
        if not all_symbols: return self.finished_signal.emit(False, "找不到股票清單。")
            
        end_dt = datetime.today()
        d_start = (end_dt - timedelta(days=self.days_to_fetch * 2 + 10)).strftime("%Y-%m-%d")
        fetch_end = end_dt.strftime("%Y-%m-%d")

        self.log_signal.emit("正在下載歷史資料：歷史處置股名單...")
        dispo_set = self.fetch_multi_day_dispo(self.days_to_fetch + 10)
        
        self.log_signal.emit(f"📋 準備採集 {len(all_symbols)} 檔股票大數據...")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS intraday_kline")
        conn.commit()
        
        success_count = 0
        for idx, sym in enumerate(all_symbols):
            if (idx+1) % 5 == 0:
                self.log_signal.emit(f"⏳ 正在採集 [{idx+1}/{len(all_symbols)}] {sn(sym)} ...")
                
            contract = api.Contracts.Stocks.get(sym)
            if not contract: continue
            
            try:
                df_all = self.fetch_kbars_smart(api, contract, d_start, fetch_end)
                if df_all.empty: continue
                
                df_all.columns = [c.lower() for c in df_all.columns]
                df_all['ts'] = pd.to_datetime(df_all['ts'])
                df_all['day'] = df_all['ts'].dt.strftime('%Y-%m-%d')
                df_all['symbol'] = sym

                available_days = sorted(df_all['day'].unique())
                target_days = available_days[-self.days_to_fetch:]
                
                df = fill_zero_volume_kbars(df_all)
                del df_all
                if df.empty: continue
                
                daily_last = df.groupby('day')['close'].last().shift(1)
                df['yesterday_close'] = df['day'].map(daily_last)
                df = df[df['day'].isin(target_days)].dropna(subset=['yesterday_close'])
                if df.empty: continue
                
                df['limit_up'] = df['yesterday_close'].apply(calculate_limit_up_price)
                df['limit_up'] = df['limit_up'].apply(lambda x: math.floor(x * 100) / 100.0 if pd.notnull(x) else x)
                df['rise'] = round((df['close'] - df['yesterday_close']) / df['yesterday_close'] * 100, 2)
                df['highest'] = df.groupby('day')['high'].cummax()
                df['pct_increase'] = df.groupby('day')['rise'].diff().fillna(df['rise']).round(2)
                
                df.rename(columns={'yesterday_close': '昨日收盤價', 'limit_up': '漲停價', 'day': 'date'}, inplace=True)
                cols = ['symbol', 'date', 'time', 'open', 'high', 'low', 'close', 'volume', '昨日收盤價', '漲停價', 'rise', 'highest', 'pct_increase']
                
                dispo_keys = {f"{d}_{s}" for d, s in dispo_set}
                df = df[~df.apply(lambda r: f"{r['date']}_{r['symbol']}" in dispo_keys, axis=1)]
                
                if not df.empty:
                    df[cols].to_sql('intraday_kline', conn, if_exists='append', index=False)
                    success_count += 1
                del df
                    
            except Exception: pass
            self.progress_signal.emit(int((idx+1)/len(all_symbols) * 100))

        conn.close()
        self.finished_signal.emit(True, f"✅ 成功採集 {success_count} 檔股票數據！")

    def fetch_multi_day_dispo(self, days):
        """多日處置股極速爬蟲 (已升級 TPEx 雙軌解析與防護)"""
        import requests, re
        from datetime import datetime, timedelta
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        end_obj = datetime.today()
        dispo_set = set()
        tpex_cache = {}
        
        twse_headers = {'User-Agent': 'Mozilla/5.0'}
        tpex_headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'X-Requested-With': 'XMLHttpRequest'
        }
        
        # 先緩存上櫃資料
        for i in range(days + 20):
            curr_date = end_obj - timedelta(days=i)
            if curr_date.weekday() >= 5: continue
            roc_year = curr_date.year - 1911
            tpex_date = f"{roc_year}/{curr_date.strftime('%m/%d')}"
            syms_today = []
            try:
                url = f"https://www.tpex.org.tw/web/bulletin/disposal_information/disposal_information_result.php?l=zh-tw&d={tpex_date}&o=json"
                res = requests.get(url, headers=tpex_headers, timeout=5, verify=False)
                text = res.text.strip()
                if not text or text.startswith('<'): continue
                
                data = res.json()
                if 'tables' in data:
                    for table in data['tables']:
                        if 'data' in table:
                            for row in table['data']:
                                if len(row) > 2 and re.fullmatch(r'\d{4,6}', str(row[2]).strip()):
                                    syms_today.append(str(row[2]).strip())
                elif 'aaData' in data:
                    for row in data['aaData']:
                        for val in row:
                            if re.fullmatch(r'\d{4,6}', str(val).strip()):
                                syms_today.append(str(val).strip())
                                break
            except Exception: pass
            tpex_cache[curr_date] = syms_today

        # 組合每日名單
        for i in range(days + 5):
            target_date = end_obj - timedelta(days=i)
            if target_date.weekday() >= 5: continue
            target_str = target_date.strftime('%Y-%m-%d')
            
            start_date = target_date - timedelta(days=15)
            twse_start = start_date.strftime('%Y%m%d')
            twse_end = target_date.strftime('%Y%m%d')
            
            try:
                twse_url = f"https://www.twse.com.tw/announcement/punish?response=json&startDate={twse_start}&endDate={twse_end}"
                res_twse = requests.get(twse_url, headers=twse_headers, timeout=5, verify=False).json()
                if 'data' in res_twse:
                    for row in res_twse['data']:
                        for cell in row:
                            cell_str = str(cell).strip()
                            if re.fullmatch(r'\d{4,6}', cell_str):
                                dispo_set.add((target_str, cell_str))
                                break
            except Exception: pass
            
            for j in range(16):
                check_date = target_date - timedelta(days=j)
                if check_date in tpex_cache:
                    for sym in tpex_cache[check_date]:
                        dispo_set.add((target_str, sym))
                        
        return dispo_set

# ==================== 程式進入點 ====================
def main():

    # ── Worker-mode：被平行回歸子進程呼叫時，不啟動 Qt，只跑 Optuna ──
    if '--regression-worker' in sys.argv:
        _run_regression_worker_mode()
        return

    # 徹底壓制 Unhandled Python exception 視窗與紅字
    def silent_exception_hook(exctype, value, tb):
        # 判定為鍵盤中斷(Ctrl+C)或正常退出時，完全靜音
        if issubclass(exctype, (KeyboardInterrupt, SystemExit)):
            return
        # 其他錯誤僅在黑色背景顯示，絕不彈窗跳框
        err_msg = "".join(traceback.format_exception(exctype, value, tb))
        sys.__stderr__.write(f"\n[系統背景隔離中] {err_msg}\n")

    sys.excepthook = silent_exception_hook

    try:
        load_settings()

        # 啟動時背景檢查更新（靜默模式，有新版才提示）
        check_for_update(silent=True)

        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication
        QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)

        # Windows 工具列獨立 icon（不繼承 python.exe 的圖示）
        if os.name == 'nt':
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("remora.v" + APP_VERSION)

        app = QApplication(sys.argv)
        app.setAttribute(Qt.AA_DisableWindowContextHelpButton)
        app.setStyle("Fusion")

        # 全域 App icon（工具列 + 視窗標題列）
        _icon_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        _icon_path = os.path.join(_icon_dir, 'remora.ico')
        if os.path.exists(_icon_path):
            app.setWindowIcon(QIcon(_icon_path))

        # ── 全域 Fusion 暗色調色盤 ──
        palette = QPalette()
        palette.setColor(QPalette.Window,          QColor(TV['bg']))
        palette.setColor(QPalette.WindowText,      QColor(TV['text']))
        palette.setColor(QPalette.Base,            QColor(TV['panel']))
        palette.setColor(QPalette.AlternateBase,   QColor(TV['surface']))
        palette.setColor(QPalette.ToolTipBase,     QColor(TV['panel']))
        palette.setColor(QPalette.ToolTipText,     QColor(TV['text']))
        palette.setColor(QPalette.Text,            QColor(TV['text']))
        palette.setColor(QPalette.Button,          QColor(TV['surface']))
        palette.setColor(QPalette.ButtonText,      QColor(TV['text']))
        palette.setColor(QPalette.BrightText,      QColor(TV['text_bright']))
        palette.setColor(QPalette.Link,            QColor(TV['blue']))
        palette.setColor(QPalette.Highlight,       QColor(TV['blue']))
        palette.setColor(QPalette.HighlightedText, QColor('#ffffff'))
        palette.setColor(QPalette.Disabled, QPalette.Text,       QColor(TV['text_dim']))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(TV['text_dim']))
        palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(TV['text_dim']))
        app.setPalette(palette)

        # ── 全域字型 (優先微軟正黑體) ──
        default_font = QFont()
        default_font.setFamily("Segoe UI")
        default_font.setPointSize(10)
        app.setFont(default_font)

        window = QuantMainWindow()
        global _main_window_ref
        _main_window_ref = window

        print("\x1b[34m" + "━" * 62 + "\x1b[0m")
        print(f"\x1b[32m▲  REMORA  v{APP_VERSION}  │  系統核心載入完成\x1b[0m")
        print(f"\x1b[36m   經典深色主題  │  背景例外防護已啟動\x1b[0m")
        print("\x1b[34m" + "━" * 62 + "\x1b[0m")

        tg_bot.start()
        window.showMaximized()

        # 首次啟動引導：自動開啟新手教學
        if not sys_db.load_state('first_launch_done', False):
            def _first_launch():
                window._ensure_tab('tutorial', '新手教學', TutorialWidget)
                sys_db.save_state('first_launch_done', True)
            QTimer.singleShot(500, _first_launch)

        print_trading_mode()
        sys.exit(app.exec_())
        
    except Exception as _main_err:
        logger.critical(f"程式主流程致命錯誤: {_main_err}", exc_info=True)
        os._exit(1)

if __name__ == "__main__":
    main()