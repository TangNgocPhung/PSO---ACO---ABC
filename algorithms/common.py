"""Tiện ích chung + CSS theme cho toàn bộ app."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np


# ============================================================================
#  THEME / CSS GLOBAL
# ============================================================================
GLOBAL_CSS = """
<style>
/* ------- IMPORT FONT ------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ------- TỔNG THỂ NỀN APP ------- */
.stApp {
    background:
        radial-gradient(at 0% 0%, rgba(99,102,241,0.10) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(236,72,153,0.08) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(59,130,246,0.08) 0px, transparent 50%),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
}

/* ------- BLOCK CHÍNH ------- */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px !important;
}

/* ------- HEADER TRANG ------- */
.page-hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    color: white;
    padding: 1.8rem 2rem;
    border-radius: 20px;
    margin-bottom: 1.4rem;
    box-shadow: 0 20px 50px -12px rgba(118,75,162,.35);
    position: relative;
    overflow: hidden;
}
.page-hero::before {
    content: "";
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,.15), transparent 70%);
    border-radius: 50%;
}
.page-hero h1, .page-hero h2 {
    margin: 0 0 .25rem 0 !important;
    color: white !important;
    font-weight: 800 !important;
    letter-spacing: -.5px;
}
.page-hero p {
    margin: 0 !important;
    color: rgba(255,255,255,.92) !important;
    font-size: 1rem;
}

/* ------- BADGE (tag thuật toán) ------- */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-weight: 600;
    font-size: .75rem;
    letter-spacing: .3px;
    margin-left: 8px;
    backdrop-filter: blur(6px);
}
.badge-pso { background: rgba(59,130,246,.95); color: white; border:1px solid rgba(255,255,255,.3); }
.badge-aco { background: rgba(245,158,11,.95); color: white; border:1px solid rgba(255,255,255,.3); }
.badge-abc { background: rgba(34,197,94,.95);  color: white; border:1px solid rgba(255,255,255,.3); }

/* ------- CARD CHO TRANG CHỦ ------- */
.problem-card {
    background: rgba(255,255,255,.85);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,.6);
    border-radius: 18px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 10px 25px -10px rgba(99,102,241,.15);
    transition: all 0.35s cubic-bezier(.4,0,.2,1);
    cursor: pointer;
    position: relative;
    overflow: hidden;
}
.problem-card::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #667eea, #f093fb);
    border-radius: 18px 18px 0 0;
}
.problem-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 25px 50px -10px rgba(99,102,241,.35);
    border-color: rgba(102,126,234,.5);
}
.problem-card h4 {
    color: #1e1b4b;
    font-weight: 700;
    font-size: 1.05rem;
    margin: 0 0 .4rem 0 !important;
    display: flex;
    align-items: center;
}
.problem-card .icon-circle {
    width: 44px; height: 44px;
    border-radius: 12px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    margin-right: 10px;
    background: linear-gradient(135deg, #e0e7ff, #fce7f3);
    box-shadow: 0 4px 10px rgba(99,102,241,.18);
}
.problem-card p {
    color: #475569;
    font-size: .88rem;
    margin: .5rem 0 0 0 !important;
    line-height: 1.45;
}

/* ------- THẺ MÔ TẢ THUẬT TOÁN (3 cột PSO/ACO/ABC) ------- */
.algo-card {
    background: rgba(255,255,255,.8);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 1.4rem 1.5rem;
    border-left: 5px solid #6366f1;
    box-shadow: 0 10px 30px -12px rgba(99,102,241,.25);
    height: 100%;
    transition: transform .3s;
}
.algo-card:hover { transform: translateY(-4px); }
.algo-card h4 {
    color: #1e1b4b;
    margin: 0 0 .5rem 0 !important;
    font-weight: 700;
}
.algo-card p {
    color: #475569;
    margin: 0 !important;
    font-size: .92rem;
    line-height: 1.55;
}
.algo-card.pso { border-left-color: #3b82f6; }
.algo-card.aco { border-left-color: #f59e0b; }
.algo-card.abc { border-left-color: #10b981; }

/* ------- METRIC ------- */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255,255,255,.95), rgba(241,245,249,.85));
    border: 1px solid rgba(99,102,241,.18);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    box-shadow: 0 6px 16px -6px rgba(99,102,241,.18);
    transition: all .25s;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 24px -8px rgba(99,102,241,.28);
    border-color: rgba(99,102,241,.35);
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: .8rem !important;
    letter-spacing: .3px;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #1e1b4b !important;
    font-weight: 800 !important;
    font-size: 1.6rem !important;
    background: linear-gradient(135deg, #4f46e5, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ============================================================
   ★★★ SIDEBAR TEXT – NHẤN MẠNH: TẤT CẢ chữ trong sidebar phải SÁNG ★★★
   Đặt SỚM trong file, dùng selector rộng để mọi element kế thừa.
   Các element cần nền sáng (input/select/alert/...) sẽ tự override sau.
   ============================================================ */
body [data-testid="stSidebar"],
body [data-testid="stSidebar"] *,
body [data-testid="stSidebar"] p,
body [data-testid="stSidebar"] span,
body [data-testid="stSidebar"] div,
body [data-testid="stSidebar"] label,
body [data-testid="stSidebar"] li,
body [data-testid="stSidebar"] small {
    color: #e0e7ff !important;
}

/* ------- SIDEBAR ------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] .stMarkdown span:not(.badge):not(.pill):not(.cstt-tip-card span) {
    color: #e0e7ff !important;
}

/* ===== TIP CARD trong sidebar – class riêng – LUÔN nền trắng + chữ tím đậm =====
   Đặt specificity cao bằng cách qualified với sidebar selector.                  */
[data-testid="stSidebar"] .cstt-tip-card,
[data-testid="stSidebar"] .cstt-tip-card * {
    background-color: transparent !important;
    color: #1e1b4b !important;
}
[data-testid="stSidebar"] .cstt-tip-card {
    background: #ffffff !important;
    background-color: #ffffff !important;
}
[data-testid="stSidebar"] .cstt-tip-card b,
[data-testid="stSidebar"] .cstt-tip-card strong {
    color: #4c1d95 !important;
    font-weight: 700 !important;
}
/* Chữ in đậm / nghiêng trong sidebar - tô vàng nhạt để nổi bật trên nền tối */
[data-testid="stSidebar"] .stMarkdown strong,
[data-testid="stSidebar"] .stMarkdown b {
    color: #fde68a !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stMarkdown em,
[data-testid="stSidebar"] .stMarkdown i {
    color: #fbcfe8 !important;
}
/* Link trong sidebar (đoạn markdown) */
[data-testid="stSidebar"] .stMarkdown a {
    color: #93c5fd !important;
    text-decoration: underline;
}
/* Inline code trong sidebar */
[data-testid="stSidebar"] .stMarkdown code {
    background: rgba(255,255,255,.15) !important;
    color: #fde68a !important;
    border: 1px solid rgba(253,230,138,.3) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label *,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
}

/* "View less" / "Show more" / pagination buttons trong sidebar – ép sáng */
[data-testid="stSidebar"] button:not([kind="primary"]):not([data-testid="stBaseButton-headerNoPadding"]) span,
[data-testid="stSidebar"] [aria-label="View less"],
[data-testid="stSidebar"] [aria-label="View more"],
[data-testid="stSidebar"] [aria-label="Show more"] {
    color: #ffffff !important;
}
/* Bất kỳ button text-only trong sidebar (nút phụ) */
[data-testid="stSidebar"] [role="button"]:not([kind="primary"]) {
    color: #e0e7ff !important;
}
[data-testid="stSidebar"] [role="button"]:not([kind="primary"]) * {
    color: #e0e7ff !important;
}
/* Sidebar headings - dùng body + .stApp prefix để tăng specificity vượt mọi rule khác */
body .stApp [data-testid="stSidebar"] h1,
body .stApp [data-testid="stSidebar"] h2,
body .stApp [data-testid="stSidebar"] h3,
body .stApp [data-testid="stSidebar"] h4,
body .stApp [data-testid="stSidebar"] h1 *,
body .stApp [data-testid="stSidebar"] h2 *,
body .stApp [data-testid="stSidebar"] h3 *,
body .stApp [data-testid="stSidebar"] h4 * {
    color: #ffffff !important;
    border-left: none !important;
    padding-left: 0 !important;
}
body .stApp [data-testid="stSidebar"] h1,
body .stApp [data-testid="stSidebar"] h2,
body .stApp [data-testid="stSidebar"] h3,
body .stApp [data-testid="stSidebar"] h4 {
    border-bottom: 2px solid rgba(255,255,255,.15) !important;
    padding-bottom: .5rem !important;
    margin-top: 1rem !important;
}

/* ----- Slider trong sidebar ----- */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, #f093fb, #f5576c) !important;
}
/* Giá trị min/max ở 2 đầu thanh trượt */
[data-testid="stSidebar"] [data-testid="stTickBar"],
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"] {
    color: #c7d2fe !important;
}
/* Nút trượt */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
    background: #f093fb !important;
    box-shadow: 0 0 0 3px rgba(240,147,251,.3) !important;
    color: #ffffff !important;
}
/* Giá trị hiện tại nổi lên trên thumb (tooltip) - chữ trắng đậm */
[data-testid="stSidebar"] .stSlider [data-baseweb="tooltip"],
[data-testid="stSidebar"] .stSlider [data-baseweb="tooltip"] *,
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-baseweb="tooltip"],
[data-testid="stSidebar"] .stSlider div[role="slider"] + div,
[data-testid="stSidebar"] [data-testid="stThumbValue"] {
    color: #ffffff !important;
    background: rgba(30,27,75,.85) !important;
    font-weight: 700 !important;
    padding: 2px 8px !important;
    border-radius: 8px !important;
}

/* ----- Select box / Multi-select trong sidebar ----- */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,.95) !important;
    border: 1px solid rgba(255,255,255,.3) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div *,
[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #1e1b4b !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #6366f1 !important;
}

/* ----- Number input / Text input trong sidebar (chuyển nền sang trắng + chữ tím đậm) ----- */
[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] [data-testid="stTextArea"] textarea,
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] input[type="text"] {
    background: #ffffff !important;
    color: #1e1b4b !important;
    border: 1px solid rgba(99,102,241,.4) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    caret-color: #6366f1 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input:focus,
[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
    border-color: #f093fb !important;
    box-shadow: 0 0 0 3px rgba(240,147,251,.25) !important;
}
/* Nút +/- của number input */
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background: rgba(255,255,255,.18) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,.25) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {
    background: rgba(240,147,251,.45) !important;
    border-color: #f093fb !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg {
    fill: #ffffff !important;
}

/* ----- Radio / Checkbox trong sidebar ----- */
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    color: #e0e7ff !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] div[role="radio"] {
    border-color: rgba(255,255,255,.5) !important;
}

/* ----- Dropdown menu (khi click vào select) ----- */
[data-baseweb="popover"] [role="listbox"] {
    background: #ffffff !important;
    border: 1px solid rgba(99,102,241,.3) !important;
    border-radius: 10px !important;
    box-shadow: 0 12px 32px -8px rgba(99,102,241,.35) !important;
}
[data-baseweb="popover"] [role="option"] {
    color: #1e1b4b !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background: linear-gradient(90deg, rgba(99,102,241,.12), rgba(240,147,251,.12)) !important;
}

/* ============================================================
   SIDEBAR NAV (danh sách 11 trang Streamlit tự sinh)
   Trước: chữ đen trên nền tím đậm → không thấy.
   Sau:  chữ trắng + hover gradient hồng-tím.
   ============================================================ */
[data-testid="stSidebarNav"] {
    background: rgba(255,255,255,.05) !important;
    border-radius: 12px !important;
    padding: 8px 6px !important;
    margin-bottom: 1rem !important;
}
[data-testid="stSidebarNav"] ul {
    padding-left: 0 !important;
}
[data-testid="stSidebarNav"] li {
    list-style: none !important;
    margin: 2px 0 !important;
}
/* Link trong nav (mọi phiên bản Streamlit) */
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] a span,
[data-testid="stSidebarNav"] a p,
[data-testid="stSidebarNavLink"],
[data-testid="stSidebarNavLink"] span,
[data-testid="stSidebarNavLink"] p,
[data-testid="stSidebarNavItems"] a,
[data-testid="stSidebarNavItems"] a span,
[data-testid="stSidebarNavItems"] a p {
    color: #e0e7ff !important;
    font-weight: 500 !important;
    font-size: .92rem !important;
    text-decoration: none !important;
}
/* Link container */
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNavLink"],
[data-testid="stSidebarNavItems"] a {
    display: block !important;
    padding: 8px 14px !important;
    border-radius: 10px !important;
    transition: all .2s ease !important;
    background: transparent !important;
}
/* Hover */
[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNavLink"]:hover,
[data-testid="stSidebarNavItems"] a:hover {
    background: linear-gradient(90deg, rgba(240,147,251,.25), rgba(99,102,241,.25)) !important;
    transform: translateX(3px);
    box-shadow: inset 3px 0 0 #f093fb;
}
[data-testid="stSidebarNav"] a:hover *,
[data-testid="stSidebarNavLink"]:hover *,
[data-testid="stSidebarNavItems"] a:hover * {
    color: #ffffff !important;
    font-weight: 600 !important;
}
/* Trang đang active */
[data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebarNavLink"][aria-current="page"],
[data-testid="stSidebarNavItems"] a[aria-current="page"] {
    background: linear-gradient(90deg, #f093fb, #f5576c) !important;
    box-shadow: 0 6px 16px -6px rgba(245,87,108,.55) !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] *,
[data-testid="stSidebarNavLink"][aria-current="page"] *,
[data-testid="stSidebarNavItems"] a[aria-current="page"] * {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Mũi tên collapse sidebar (<<) */
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stBaseButton-headerNoPadding"] svg {
    fill: #c7d2fe !important;
}
[data-testid="stSidebarCollapseButton"]:hover svg,
[data-testid="stBaseButton-headerNoPadding"]:hover svg {
    fill: #ffffff !important;
}

/* ------- BUTTON ------- */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: .6rem 1.4rem;
    font-weight: 600;
    letter-spacing: .3px;
    box-shadow: 0 8px 20px -6px rgba(102,126,234,.55);
    transition: all .25s cubic-bezier(.4,0,.2,1);
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 12px 28px -6px rgba(102,126,234,.7);
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    box-shadow: 0 8px 20px -6px rgba(245,87,108,.55);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #ec80f7 0%, #f04060 100%);
}

/* ------- EXPANDER ------- */
[data-testid="stExpander"] {
    background: rgba(255,255,255,.7);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(99,102,241,.15);
    border-radius: 14px;
    margin-bottom: .8rem;
    box-shadow: 0 4px 14px -6px rgba(99,102,241,.12);
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #4338ca !important;
    padding: .5rem 0;
}

/* ------- ALERT (success / info / warn / error) – BODY chính ------- */
.stApp [data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid rgba(99,102,241,.2);
    box-shadow: 0 4px 14px -6px rgba(99,102,241,.18);
    backdrop-filter: blur(6px);
}
/* Body alerts giữ màu mặc định Streamlit (info xanh, success xanh lá, ...) */

/* ------- ALERT TRONG SIDEBAR – ép TẤT CẢ layer divs về nền TRẮNG + chữ TÍM ĐẬM -------
   Streamlit dùng nhiều layer div lồng nhau cho alert, mỗi layer có background riêng.
   Phải phủ TẤT CẢ selector có thể có để bulletproof.                                  */
[data-testid="stSidebar"] [data-testid="stAlert"],
[data-testid="stSidebar"] [data-testid="stAlert"] > div,
[data-testid="stSidebar"] [data-testid="stAlert"] > div > div,
[data-testid="stSidebar"] [data-testid="stAlertContainer"],
[data-testid="stSidebar"] [data-testid="stNotification"],
[data-testid="stSidebar"] [data-testid="stNotificationContent"],
[data-testid="stSidebar"] [data-testid="stAlertContentInfo"],
[data-testid="stSidebar"] [data-testid="stAlertContentSuccess"],
[data-testid="stSidebar"] [data-testid="stAlertContentWarning"],
[data-testid="stSidebar"] [data-testid="stAlertContentError"],
[data-testid="stSidebar"] [data-testid*="Alert"],
[data-testid="stSidebar"] [data-testid*="alert"],
[data-testid="stSidebar"] div[class*="stAlert"],
[data-testid="stSidebar"] div[class*="alert"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1px solid rgba(99,102,241,.35) !important;
    border-radius: 14px !important;
    box-shadow: 0 6px 18px -8px rgba(0,0,0,.5) !important;
    color: #1e1b4b !important;
}
/* Mọi text con bên trong alert: tím đậm */
[data-testid="stSidebar"] [data-testid="stAlert"] *,
[data-testid="stSidebar"] [data-testid*="Alert"] *,
[data-testid="stSidebar"] [data-testid*="Notification"] * {
    color: #1e1b4b !important;
    background-color: transparent !important;
}
/* Bold trong alert: tím sâu hơn */
[data-testid="stSidebar"] [data-testid="stAlert"] strong,
[data-testid="stSidebar"] [data-testid="stAlert"] b,
[data-testid="stSidebar"] [data-testid*="Alert"] strong,
[data-testid="stSidebar"] [data-testid*="Alert"] b {
    color: #4c1d95 !important;
    font-weight: 700 !important;
}
/* Icon SVG */
[data-testid="stSidebar"] [data-testid="stAlert"] svg,
[data-testid="stSidebar"] [data-testid*="Alert"] svg {
    fill: #6366f1 !important;
}

/* ------- PROGRESS BAR ------- */
.stProgress > div > div {
    background: linear-gradient(90deg, #667eea, #f093fb, #f5576c) !important;
    border-radius: 999px !important;
}
.stProgress > div > div > div {
    background: transparent !important;
}

/* ------- SUBHEADER (chỉ áp dụng cho MAIN BODY, KHÔNG đụng sidebar) ------- */
.stApp [data-testid="stMain"] h2,
.stApp [data-testid="stMain"] h3,
.stApp section.main h2,
.stApp section.main h3,
.stApp .main h2,
.stApp .main h3 {
    color: #1e1b4b !important;
    font-weight: 700 !important;
    letter-spacing: -.3px;
}
.stApp [data-testid="stMain"] h3,
.stApp section.main h3,
.stApp .main h3 {
    font-size: 1.15rem !important;
    border-left: 4px solid #667eea;
    padding-left: 12px;
    margin-top: 1.2rem !important;
}

/* ------- DATAFRAME (body chính – nền sáng chữ tối) ------- */
.stApp [data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 14px -6px rgba(99,102,241,.18);
}
.stApp [data-testid="stDataFrame"] * {
    color: #1e1b4b !important;
}
/* Body markdown - đảm bảo bold trong main area là tím đậm */
.stApp .main .stMarkdown strong,
.stApp .main .stMarkdown b {
    color: #4338ca !important;
    font-weight: 700 !important;
}

/* ------- RADIO / CHECKBOX trong body chính (không sidebar) ------- */
.stApp .main [data-testid="stRadio"] label,
.stApp .main [data-testid="stCheckbox"] label {
    color: #1e1b4b !important;
}

/* ------- CODE BLOCK (body chính) ------- */
.stApp code {
    background: linear-gradient(135deg, #fef3c7, #fde68a) !important;
    color: #92400e !important;
    padding: 2px 6px !important;
    border-radius: 5px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .85rem !important;
}
.stApp pre {
    background: #1e1b4b !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    box-shadow: 0 8px 20px -8px rgba(30,27,75,.4);
}
.stApp pre code,
.stApp pre code * {
    background: transparent !important;
    color: #c7d2fe !important;
}
/* st.code block (st.code()) - dark with light text */
.stApp [data-testid="stCodeBlock"] {
    background: #1e1b4b !important;
    border-radius: 12px !important;
}
.stApp [data-testid="stCodeBlock"] code,
.stApp [data-testid="stCodeBlock"] pre,
.stApp [data-testid="stCodeBlock"] span {
    color: #e0e7ff !important;
    background: transparent !important;
}

/* ------- DIVIDER ------- */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,.3), transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ------- FOOTER (chân trang) ------- */
.footer-box {
    background: rgba(255,255,255,.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(99,102,241,.15);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-top: 2rem;
    text-align: center;
    color: #475569;
    box-shadow: 0 4px 14px -6px rgba(99,102,241,.15);
}
.footer-box p { margin: .25rem 0 !important; font-size: .9rem; }
.footer-box .authors {
    font-weight: 600;
    color: #1e1b4b;
}

/* ------- ANIMATION KEYFRAMES ------- */
@keyframes float {
    0%,100% { transform: translateY(0); }
    50%     { transform: translateY(-6px); }
}
.floating { animation: float 4s ease-in-out infinite; }

@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* ============================================================
   ★★★ FINAL OVERRIDE – đặt CUỐI để bảo vệ các vùng nền SÁNG
   trong sidebar khỏi bị scorched-earth làm chữ sáng.
   Vùng nền sáng: input, select, alert, tip card, popover, dropdown.
   ============================================================ */
body [data-testid="stSidebar"] .cstt-tip-card,
body [data-testid="stSidebar"] .cstt-tip-card *,
body [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
body [data-testid="stSidebar"] [data-testid="stTextInput"] input,
body [data-testid="stSidebar"] input[type="number"],
body [data-testid="stSidebar"] input[type="text"],
body [data-testid="stSidebar"] [data-baseweb="select"] > div *,
body [data-testid="stSidebar"] [data-baseweb="select"] input,
body [data-testid="stSidebar"] [data-testid="stAlert"] *,
body [data-baseweb="popover"] [role="option"],
body [data-baseweb="popover"] [role="listbox"] * {
    color: #1e1b4b !important;
}
/* Bold trong tip-card và alert vẫn tím đậm */
body [data-testid="stSidebar"] .cstt-tip-card b,
body [data-testid="stSidebar"] .cstt-tip-card strong,
body [data-testid="stSidebar"] [data-testid="stAlert"] b,
body [data-testid="stSidebar"] [data-testid="stAlert"] strong {
    color: #4c1d95 !important;
    font-weight: 700 !important;
}
/* Tip card text chính – tím đậm */
body [data-testid="stSidebar"] .cstt-tip-card span:not(b):not(strong) {
    color: #1e1b4b !important;
}
</style>
"""


def inject_css():
    """Inject CSS toàn cục — gọi ở đầu mỗi trang."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str, algo: str):
    """Header gradient cho mỗi trang bài toán."""
    inject_css()
    badge_class = {"PSO": "badge-pso", "ACO": "badge-aco", "ABC": "badge-abc"}.get(algo, "badge-pso")
    st.markdown(
        f"""
        <div class="page-hero">
            <h2><span class="floating" style="display:inline-block;">{icon}</span>
                {title}
                <span class="badge {badge_class}">{algo}</span>
            </h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def back_button():
    """Nút quay về trang chủ ở sidebar."""
    if st.sidebar.button("🏠 Về trang chủ", use_container_width=True, key="back_home"):
        st.switch_page("app.py")
    st.sidebar.markdown("<hr style='margin:.5rem 0;'/>", unsafe_allow_html=True)


# ============================================================================
#  MATPLOTLIB THEME (đồng bộ với palette UI)
# ============================================================================
def _setup_mpl_theme():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#475569",
        "axes.edgecolor": "#cbd5e1",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.alpha": 0.6,
        "xtick.color": "#64748b",
        "ytick.color": "#64748b",
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#fbfaff",
    })

_setup_mpl_theme()


def plot_convergence(history, title="Đồ thị hội tụ", ylabel="Best Fitness"):
    """Vẽ đường hội tụ với gradient fill đẹp mắt."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(history))
    ax.plot(x, history, color="#6366f1", lw=2.5, marker="o",
            markersize=4, markerfacecolor="#f093fb",
            markeredgecolor="white", markeredgewidth=1.2,
            label="Best fitness", zorder=3)
    ax.fill_between(x, history, max(history) if history else 0,
                    color="#6366f1", alpha=0.10, zorder=1)
    ax.set_xlabel("Vòng lặp", fontweight="600")
    ax.set_ylabel(ylabel, fontweight="600")
    ax.set_title(title, color="#1e1b4b", pad=15)
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False,
              facecolor="white", edgecolor="#cbd5e1")
    fig.tight_layout()
    return fig


def metric_row(items):
    """items: list of (label, value, help) tuples."""
    cols = st.columns(len(items))
    for c, (label, value, help_) in zip(cols, items):
        c.metric(label, value, help=help_)


def section_title(text: str, icon: str = "📊"):
    """Tiêu đề section có icon + gradient bar."""
    st.markdown(
        f"""
        <h3 style="margin-top:1.4rem;">
            <span style="display:inline-block;margin-right:8px;">{icon}</span>{text}
        </h3>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
#  PDF REPORT GENERATOR
# ============================================================================
def generate_pdf_report(
    problem_title: str,
    problem_subtitle: str,
    algorithm: str,
    inputs: dict,
    outputs: dict,
    figures: list,
    note: str = "",
):
    """Sinh PDF báo cáo: tiêu đề + input + output + tất cả figures.

    Args:
        problem_title:   Vd "Bài toán 12 – VRPTW"
        problem_subtitle: Vd "Vehicle Routing Problem with Time Windows"
        algorithm:       "PSO" hoặc "ACO"
        inputs:          dict các tham số đầu vào
        outputs:         dict các chỉ số kết quả
        figures:         list[(caption, matplotlib.figure.Figure)]
        note:            Ghi chú thêm (tuỳ chọn)

    Returns:
        bytes của file PDF
    """
    import io
    from datetime import datetime
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ===== TRANG 1: TIÊU ĐỀ + INPUT + OUTPUT =====
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait

        # --- Tiêu đề (dùng font Sans, không Mono để render đúng tiếng Việt) ---
        fig.text(0.5, 0.95, problem_title, ha="center", va="top",
                 fontsize=18, fontweight="bold", color="#1e1b4b",
                 family="DejaVu Sans")
        fig.text(0.5, 0.92, problem_subtitle, ha="center", va="top",
                 fontsize=11, color="#475569", style="italic",
                 family="DejaVu Sans")
        fig.text(0.5, 0.89, f"Thuật toán: {algorithm}",
                 ha="center", va="top", fontsize=10, color="#6366f1",
                 fontweight="bold", family="DejaVu Sans")

        # --- Thanh ngang ---
        ax_line = fig.add_axes([0.1, 0.87, 0.8, 0.005])
        ax_line.axhline(0, color="#6366f1", lw=2)
        ax_line.axis("off")

        # --- Helper: render dict thành table ---
        def _render_table(ax, data_dict, header_text, header_color,
                          fc_header, fc_cell):
            ax.axis("off")
            # Header đặt cách bảng 1 đoạn (1.06 = 6% chiều cao axes ở trên)
            ax.text(0, 1.06, header_text, transform=ax.transAxes,
                    fontsize=12, fontweight="bold", color=header_color,
                    family="DejaVu Sans", va="bottom")
            rows = [[str(k), str(v)] for k, v in data_dict.items()]
            if not rows:
                return
            table = ax.table(
                cellText=rows,
                colWidths=[0.55, 0.45],
                cellLoc="left",
                loc="upper left",
                bbox=[0, 0, 1, 1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            n_rows = len(rows)
            for (r, c), cell in table.get_celld().items():
                cell.set_edgecolor("#cbd5e1")
                cell.set_linewidth(0.5)
                cell.set_facecolor(fc_cell if r % 2 == 0 else "#ffffff")
                cell.set_text_props(family="DejaVu Sans", color="#1e1b4b")
                cell.PAD = 0.05
                if c == 0:
                    cell.set_text_props(fontweight="500")
                else:
                    cell.set_text_props(fontweight="bold")
                # tăng chiều cao mỗi hàng
                cell.set_height(1.0 / max(n_rows, 1))

        # --- INPUT & OUTPUT tables ---
        n_in = len(inputs)
        n_out = len(outputs)
        # Tỷ lệ chiều cao theo số dòng (tối thiểu 0.12, tối đa 0.32)
        height_in = max(0.12, min(0.32, 0.022 * n_in + 0.04))
        height_out = max(0.12, min(0.32, 0.022 * n_out + 0.04))

        # Vị trí: chừa khoảng cách rõ ràng giữa đường ngang (0.87) và bảng đầu tiên
        INPUT_TOP = 0.80     # Bảng input bắt đầu từ y=0.80 (cách line 0.87 khoảng 0.07)
        HEADER_RESERVE = 0.04  # Khoảng dành cho header "ĐẦU VÀO" / "KẾT QUẢ"
        GAP_BETWEEN = 0.05   # Khoảng cách giữa bảng input và bảng output

        ax_in = fig.add_axes([0.1, INPUT_TOP - height_in, 0.8, height_in])
        _render_table(ax_in, inputs,
                      "ĐẦU VÀO  —  Input parameters",
                      "#1e3a8a", "#eef2ff", "#eef2ff")

        output_top = INPUT_TOP - height_in - GAP_BETWEEN - HEADER_RESERVE
        ax_out = fig.add_axes([0.1, output_top - height_out, 0.8, height_out])
        _render_table(ax_out, outputs,
                      "KẾT QUẢ  —  Output metrics",
                      "#065f46", "#ecfdf5", "#ecfdf5")

        # --- Note ---
        if note:
            fig.text(0.1, 0.08, note, fontsize=9, color="#475569",
                     style="italic", wrap=True, family="DejaVu Sans")

        # --- Footer ---
        ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        fig.text(0.5, 0.04,
                 f"CSTT Demo  —  GVHD PGS.TS Lê Hoàng Thái  —  Khoá 36 (2025–2027)",
                 ha="center", fontsize=8, color="#94a3b8", style="italic",
                 family="DejaVu Sans")
        fig.text(0.5, 0.02, f"Xuất báo cáo lúc: {ts}",
                 ha="center", fontsize=8, color="#94a3b8",
                 family="DejaVu Sans")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ===== TRANG SAU: MỖI FIGURE 1 TRANG (LƯU TRỰC TIẾP – KHÔNG EMBED PNG) =====
        for caption, src_fig in figures:
            # ---- Lưu trạng thái cũ để khôi phục sau ----
            old_suptitle_text = ""
            if src_fig._suptitle is not None:
                old_suptitle_text = src_fig._suptitle.get_text()
            old_top = src_fig.subplotpars.top
            # Lưu title gốc của từng axes (để khôi phục cho Streamlit)
            old_titles = []
            for ax_obj in src_fig.axes:
                old_titles.append((ax_obj, ax_obj.get_title()))

            # ---- Lấy title đầu tiên không rỗng từ axes làm SUB-CAPTION ----
            sub_caption = ""
            for _, t in old_titles:
                if t.strip():
                    sub_caption = t
                    break
            # ---- Clear axes title để không chồng với caption ----
            for ax_obj in src_fig.axes:
                ax_obj.set_title("")

            # ---- Ghép caption: "Hình X: ... — Sub-caption gốc" ----
            full_caption = caption
            if sub_caption and sub_caption.lower() not in caption.lower():
                full_caption = f"{caption}\n{sub_caption}"

            # ---- Đẩy axes xuống chừa chỗ cho caption (2 dòng nếu có) ----
            top_margin = 0.85 if "\n" in full_caption else 0.90
            try:
                src_fig.subplots_adjust(top=top_margin)
            except Exception:
                pass

            # ---- Set suptitle ----
            src_fig.suptitle(full_caption, fontsize=13, fontweight="bold",
                             color="#1e1b4b",
                             y=0.97, family="DejaVu Sans",
                             linespacing=1.3)

            pdf.savefig(src_fig, bbox_inches="tight", dpi=150,
                        facecolor="white")

            # ---- KHÔI PHỤC để Streamlit hiển thị bình thường ----
            src_fig.suptitle(old_suptitle_text)
            for ax_obj, title in old_titles:
                ax_obj.set_title(title)
            try:
                src_fig.subplots_adjust(top=old_top)
            except Exception:
                pass

        # Metadata
        d = pdf.infodict()
        d["Title"] = problem_title
        d["Author"] = "Nhóm KHMT Khoá 36 – ĐH Sư phạm TP.HCM"
        d["Subject"] = "Báo cáo CSTT – Swarm Intelligence Demo"
        d["Creator"] = "Streamlit CSTT Demo"

    buf.seek(0)
    return buf.getvalue()


def download_pdf_button(pdf_bytes: bytes, file_name: str, key: str = "pdf_dl"):
    """Hiển thị nút tải PDF với style đồng bộ."""
    st.download_button(
        label="📄  Tải báo cáo PDF",
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
        key=key,
        use_container_width=False,
        type="primary",
    )
