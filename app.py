
import streamlit as st
import pandas as pd
import pickle
import io
import csv
from datetime import datetime

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SentimenID — CNBC Indonesia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
section[data-testid="stSidebar"]{
    transform: translateX(0px) !important;
    visibility: visible !important;
}
</style>
""", unsafe_allow_html=True)



# ── Session state defaults ────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "page" not in st.session_state:
    st.session_state.page = "prediksi"

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("lr_sentiment_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except Exception:
    model = None
    model_loaded = False

# ── Load dataset ──────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_excel("hasil_data.xlsx")

try:
    df = load_data()
    data_loaded = True
except Exception:
    df = pd.DataFrame()
    data_loaded = False

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #f4efe9;
    --surface: #ffffff;
    --surface2: #faf7f4;
    --border: #e8ddd4;
    --accent: #c45a00;
    --accent2: #3d2810;
    --muted: #a89080;
    --dark: #1a1208;
    --pos: #2d7a3a; --pos-bg: #eaf4ec; --pos-border: #b6deba;
    --neg: #b83030; --neg-bg: #fdeaea; --neg-border: #f0bcbc;
    --neu: #c45a00; --neu-bg: #fff3eb; --neu-border: #f5d5b0;
}

html, body, [class*="css"] { font-family: 'Sora', sans-serif !important; }

/* Background */
.stApp { background: var(--bg) !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding-top: 1.8rem !important; padding-bottom: 3rem !important; max-width: 1100px !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e8ddd4 !important;
    width: 300px !important;
    min-width: 300px !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1rem !important;
}

section[data-testid="stSidebar"] .block-container {
    padding: 1rem !important;
}

/* Hide default Streamlit elements */
.stDeployButton { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

/* Radio buttons as nav */
[data-testid="stSidebar"] .stRadio > label {
    display: none !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 0 8px;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    display: flex !important;
    align-items: center;
    padding: 10px 12px;
    border-radius: 8px;
    font-size: 13px !important;
    color: #5c4030 !important;
    cursor: pointer;
    transition: all .15s;
    border: none !important;
    font-weight: 400 !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: #fdf0e8;
    color: var(--accent) !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
    background: #fdf0e8;
    color: var(--accent) !important;
    font-weight: 500 !important;
}

/* Hilangkan bulatan radio */
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--surface);
    border: 0.5px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px !important;
    transition: all .2s;
}
[data-testid="metric-container"]:hover { border-color: var(--accent); transform: translateY(-2px); }
[data-testid="stMetricLabel"] { font-size: 10px !important; color: var(--muted) !important; letter-spacing: .7px !important; text-transform: uppercase !important; font-family: 'Sora', sans-serif !important; }
[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 700 !important; font-family: 'JetBrains Mono', monospace !important; }

/* Text area */
.stTextArea textarea {
    border: 0.5px solid var(--border) !important;
    border-radius: 12px !important;
    background: var(--surface) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 14px !important;
    color: var(--dark) !important;
    padding: 14px 16px !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(196,90,0,.1) !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Sora', sans-serif !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    transition: all .15s !important;
}
.stButton > button[kind="primary"],
.stButton > button:not([kind]) {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.6rem !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button:not([kind]):hover {
    background: #a34a00 !important;
    transform: translateY(-1px) !important;
    box-shadow: none !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface) !important;
    color: #5c4030 !important;
    border: 0.5px solid var(--border) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: #fdf0e8 !important;
}

/* Divider */
hr { border: none; border-top: 0.5px solid var(--border) !important; margin: 1rem 0 !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 0.5px solid var(--border) !important; border-radius: 12px !important; overflow: hidden !important; }

/* Alerts */
.stAlert { border-radius: 10px !important; font-family: 'Sora', sans-serif !important; font-size: 13px !important; }

/* Selectbox */
.stSelectbox select, .stSelectbox > div > div {
    border: 0.5px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--surface) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 13px !important;
}

/* Download button */
.stDownloadButton > button {
    background: var(--surface) !important;
    color: #5c4030 !important;
    border: 0.5px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 13px !important;
}
.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: #fdf0e8 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent !important;
    border-bottom: 0.5px solid var(--border) !important;
    padding-bottom: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 6px 6px 0 0 !important;
    color: var(--muted) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────
def color_cfg(label):
    return {
        "positif": {"bg": "#eaf4ec", "border": "#b6deba", "text": "#2d7a3a", "label": "Positif"},
        "negatif": {"bg": "#fdeaea", "border": "#f0bcbc", "text": "#b83030", "label": "Negatif"},
        "netral":  {"bg": "#fff3eb", "border": "#f5d5b0", "text": "#c45a00", "label": "Netral"},
    }.get(label, {"bg": "#fff3eb", "border": "#f5d5b0", "text": "#c45a00", "label": label.capitalize()})

def pill_html(label, small=False):
    c = color_cfg(label)
    fs = "10px" if small else "11px"
    pad = "3px 8px" if small else "4px 12px"
    return (f"<span style='background:{c['bg']};color:{c['text']};border:0.5px solid {c['border']};"
            f"border-radius:99px;font-size:{fs};font-weight:600;padding:{pad};"
            f"letter-spacing:.3px;font-family:Sora,sans-serif'>{c['label']}</span>")

def section_header(eyebrow, title, desc=""):
    st.markdown(f"""
    <div style='margin-bottom:24px'>
        <div style='font-size:10px;color:#a89080;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;font-family:Sora,sans-serif'>
            {eyebrow}
        </div>
        <div style='font-size:24px;font-weight:700;color:#1a1208;letter-spacing:-.5px;margin-bottom:4px;font-family:Sora,sans-serif'>
            {title}
        </div>
        {"" if not desc else f"<div style='font-size:13px;color:#a89080;line-height:1.6;font-family:Sora,sans-serif'>{desc}</div>"}
    </div>
    """, unsafe_allow_html=True)

def card_wrap(content_html, padding="20px 22px", radius="14px"):
    st.markdown(f"""
    <div style='background:#fff;border:0.5px solid #e8ddd4;border-radius:{radius};padding:{padding};margin-bottom:12px'>
        {content_html}
    </div>
    """, unsafe_allow_html=True)

def history_to_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["No", "Teks", "Sentimen", "Kepercayaan (%)", "Waktu"])
    for i, h in enumerate(st.session_state.history, 1):
        writer.writerow([i, h["text"], h["label"], h["conf"], h["time"]])
    return output.getvalue().encode("utf-8")

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style='padding:20px 16px 0;display:flex;align-items:center;gap:10px;'>
        <div style='width:36px;height:36px;background:#c45a00;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;
                    color:#fff;font-size:15px;font-weight:700;flex-shrink:0;
                    font-family:Sora,sans-serif'>Si</div>
        <div>
            <div style='font-size:15px;font-weight:700;color:#c45a00;font-family:Sora,sans-serif'>SentimenID</div>
            <div style='font-size:10px;color:#a89080;letter-spacing:.6px;text-transform:uppercase;font-family:Sora,sans-serif'>CNBC Indonesia</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Navigation
    hist_count = len(st.session_state.history)
    nav_options = [
        "  Prediksi Sentimen",
        f"  Riwayat Analisis ({hist_count})",
        "  Dashboard Dataset",
        "  Tabel Data",
        "  Tentang Model",
    ]
    page_keys = ["prediksi", "riwayat", "dashboard", "data", "about"]

    selected_nav = st.radio("Menu", nav_options, label_visibility="collapsed")
    current_page = page_keys[nav_options.index(selected_nav)]

    st.divider()

    # Model info
    status_color = "#2d7a3a" if model_loaded else "#b83030"
    status_text  = "Aktif" if model_loaded else "Tidak Tersedia"
    data_text    = f"{len(df):,} baris" if data_loaded else "Tidak tersedia"

    st.markdown(f"""
    <div style='background:#faf7f4;border:0.5px solid #e8ddd4;border-radius:10px;padding:14px 14px;margin:0 8px;'>
        <div style='font-size:10px;color:#a89080;letter-spacing:.7px;text-transform:uppercase;margin-bottom:10px;font-family:Sora,sans-serif'>Status</div>
        <div style='font-size:12px;color:#3d2810;line-height:2;font-family:Sora,sans-serif'>
            <span style='color:{status_color};font-weight:600'>● {status_text}</span><br>
            <span style='color:#a89080'>Model</span> &nbsp; LR · TF-IDF<br>
            <span style='color:#a89080'>Akurasi</span> &nbsp; ~83%<br>
            <span style='color:#a89080'>Dataset</span> &nbsp; {data_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Export button
    if st.session_state.history:
        st.download_button(
            label="⬇ Ekspor Riwayat CSV",
            data=history_to_csv(),
            file_name=f"riwayat_sentimen_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ═════════════════════════════════════════════════════════════
# PAGE: PREDIKSI
# ═════════════════════════════════════════════════════════════
if current_page == "prediksi":

    section_header(
        "Analisis",
        "Prediksi Sentimen",
        "Masukkan judul berita keuangan untuk mendapatkan prediksi sentimen secara otomatis menggunakan model Logistic Regression berbasis TF-IDF."
    )

    # Quick examples
    st.markdown("<div style='font-size:11px;color:#a89080;margin-bottom:8px;font-family:Sora,sans-serif'>Coba contoh:</div>", unsafe_allow_html=True)

    c1, c2, c3, _ = st.columns([1, 1, 1, 4])
    examples = {
        "✅ Positif": "Saham BBCA melonjak 5 persen didorong kinerja keuangan solid",
        "❌ Negatif": "IHSG rontok 2 persen akibat tekanan inflasi dan sentimen global",
        "➖ Netral":  "Bank Indonesia umumkan jadwal rapat dewan gubernur bulan depan",
    }
    if c1.button("✅ Positif", use_container_width=True, key="ex_pos"):
        st.session_state.input_text = examples["✅ Positif"]
        st.rerun()
    if c2.button("❌ Negatif", use_container_width=True, key="ex_neg"):
        st.session_state.input_text = examples["❌ Negatif"]
        st.rerun()
    if c3.button("➖ Netral", use_container_width=True, key="ex_neu"):
        st.session_state.input_text = examples["➖ Netral"]
        st.rerun()

    # Input
    user_input = st.text_area(
        label="",
        value=st.session_state.input_text,
        placeholder="Contoh: IHSG Menguat Setelah The Fed Menahan Suku Bunga...",
        height=120,
        label_visibility="collapsed",
        key="main_input"
    )
    char_count = len(user_input)
    st.markdown(
        f"<div style='text-align:right;font-size:11px;color:#a89080;margin-top:-8px;margin-bottom:10px;font-family:JetBrains Mono,monospace'>{char_count}/500</div>",
        unsafe_allow_html=True
    )

    col_run, col_clear, _ = st.columns([2, 1, 5])
    run_btn   = col_run.button("✦ Analisis Sentimen", use_container_width=True, type="primary")
    clear_btn = col_clear.button("Bersihkan", use_container_width=True, type="secondary")

    if clear_btn:
        st.session_state.input_text = ""
        st.rerun()

    if run_btn:
        if not user_input.strip():
            st.warning("Masukkan teks berita terlebih dahulu.")
        elif not model_loaded:
            st.error("Model tidak ditemukan. Pastikan file `lr_sentiment_model.pkl` tersedia.")
        else:
            with st.spinner("Menganalisis..."):
                prediction  = model.predict([user_input])[0]
                probability = model.predict_proba([user_input])[0]
                classes     = model.classes_
                confidence  = max(probability) * 100

                # Save to session
                st.session_state.input_text = user_input
                st.session_state.history.insert(0, {
                    "text":  user_input,
                    "label": prediction,
                    "conf":  f"{confidence:.1f}",
                    "probs": dict(zip(classes, probability)),
                    "time":  datetime.now().strftime("%H:%M"),
                    "date":  datetime.now().strftime("%d %b %Y"),
                })

            c = color_cfg(prediction)

            # Result card
            st.markdown(f"""
            <div style='background:{c["bg"]};border:0.5px solid {c["border"]};border-radius:14px;
                        padding:22px 24px;margin:18px 0 16px;
                        display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <div style='font-size:10px;color:#a89080;letter-spacing:.7px;text-transform:uppercase;
                                margin-bottom:6px;font-family:Sora,sans-serif'>Sentimen terdeteksi</div>
                    <div style='font-size:24px;font-weight:700;color:{c["text"]};letter-spacing:-.4px;font-family:Sora,sans-serif'>
                        {c["label"]}
                    </div>
                </div>
                <div style='text-align:right'>
                    <div style='font-size:10px;color:#a89080;letter-spacing:.7px;text-transform:uppercase;
                                margin-bottom:6px;font-family:Sora,sans-serif'>Kepercayaan</div>
                    <div style='font-size:24px;font-weight:700;color:#1a1208;font-family:JetBrains Mono,monospace'>
                        {confidence:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("<div style='font-size:11px;color:#a89080;margin-bottom:12px;font-family:Sora,sans-serif;letter-spacing:.3px'>Distribusi probabilitas</div>", unsafe_allow_html=True)

            bar_colors = {"positif": "#2d7a3a", "negatif": "#b83030", "netral": "#c45a00"}
            sorted_pairs = sorted(zip(classes, probability), key=lambda x: x[1], reverse=True)

            for cls, prob in sorted_pairs:
                pct   = prob * 100
                color = bar_colors.get(cls, "#c45a00")
                st.markdown(f"""
                <div style='margin-bottom:14px'>
                    <div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px;font-family:Sora,sans-serif'>
                        <span style='color:#5c4030;text-transform:capitalize;font-weight:500'>{cls}</span>
                        <span style='color:{color};font-weight:600;font-family:JetBrains Mono,monospace'>{pct:.1f}%</span>
                    </div>
                    <div style='height:9px;background:#f0e8e0;border-radius:99px;overflow:hidden'>
                        <div style='height:100%;width:{pct:.1f}%;background:{color};border-radius:99px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # Mini history
    st.markdown("""
    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px'>
        <div style='font-size:14px;font-weight:600;color:#1a1208;font-family:Sora,sans-serif'>Analisis Terakhir</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div style='padding:28px;text-align:center;border:0.5px dashed #e8ddd4;border-radius:12px;
                    font-size:13px;color:#a89080;font-family:Sora,sans-serif'>
            Belum ada analisis. Coba masukkan berita di atas.
        </div>
        """, unsafe_allow_html=True)
    else:
        for h in st.session_state.history[:3]:
            c = color_cfg(h["label"])
            st.markdown(f"""
            <div style='background:#fff;border:0.5px solid #e8ddd4;border-radius:12px;padding:12px 16px;
                        margin-bottom:8px;display:flex;align-items:center;gap:12px;'>
                <span style='background:{c["bg"]};color:{c["text"]};border:0.5px solid {c["border"]};
                             border-radius:99px;font-size:10px;font-weight:600;padding:3px 10px;
                             flex-shrink:0;font-family:Sora,sans-serif;letter-spacing:.3px'>{c["label"]}</span>
                <span style='font-size:13px;color:#5c4030;flex:1;white-space:nowrap;overflow:hidden;
                             text-overflow:ellipsis;font-family:Sora,sans-serif'>{h["text"]}</span>
                <span style='font-size:11px;color:#a89080;font-family:JetBrains Mono,monospace;flex-shrink:0'>{h["conf"]}%</span>
                <span style='font-size:11px;color:#a89080;font-family:Sora,sans-serif;flex-shrink:0'>{h["time"]}</span>
            </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE: RIWAYAT
# ═════════════════════════════════════════════════════════════
elif current_page == "riwayat":

    section_header(
        "Riwayat",
        "Riwayat Analisis",
        "Semua prediksi yang pernah dilakukan dalam sesi ini."
    )

    if not st.session_state.history:
        st.markdown("""
        <div style='padding:40px;text-align:center;border:0.5px dashed #e8ddd4;border-radius:14px;
                    font-size:13px;color:#a89080;font-family:Sora,sans-serif'>
            Belum ada riwayat analisis. Mulai dengan memasukkan berita di halaman Prediksi.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Stats
        all_hist = st.session_state.history
        total_h  = len(all_hist)
        pos_h    = sum(1 for h in all_hist if h["label"] == "positif")
        neg_h    = sum(1 for h in all_hist if h["label"] == "negatif")
        neu_h    = sum(1 for h in all_hist if h["label"] == "netral")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Analisis", total_h)
        col2.metric("Positif",  pos_h)
        col3.metric("Negatif",  neg_h)
        col4.metric("Netral",   neu_h)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Filter
        filter_opt = st.selectbox(
            "Filter sentimen:",
            ["Semua", "Positif", "Negatif", "Netral"],
            label_visibility="collapsed"
        )
        filter_map = {"Semua": None, "Positif": "positif", "Negatif": "negatif", "Netral": "netral"}
        fkey = filter_map[filter_opt]
        filtered = all_hist if fkey is None else [h for h in all_hist if h["label"] == fkey]

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        for idx, h in enumerate(filtered):
            c = color_cfg(h["label"])
            probs_html = ""
            if "probs" in h:
                for cls_n, prob_v in sorted(h["probs"].items(), key=lambda x: x[1], reverse=True):
                    bc = {"positif": "#2d7a3a", "negatif": "#b83030", "netral": "#c45a00"}.get(cls_n, "#c45a00")
                    probs_html += f"""
                    <div style='margin-bottom:8px'>
                        <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;font-family:Sora,sans-serif'>
                            <span style='color:#5c4030;text-transform:capitalize'>{cls_n}</span>
                            <span style='color:{bc};font-weight:600;font-family:JetBrains Mono,monospace'>{prob_v*100:.1f}%</span>
                        </div>
                        <div style='height:6px;background:#f0e8e0;border-radius:99px;overflow:hidden'>
                            <div style='height:100%;width:{prob_v*100:.1f}%;background:{bc};border-radius:99px'></div>
                        </div>
                    </div>"""

            with st.expander(f"{h['label'].upper()} · {h['conf']}% · {h['time']} {h['date']} · {h['text'][:60]}{'...' if len(h['text'])>60 else ''}"):
                st.markdown(f"""
                <div style='background:{c["bg"]};border:0.5px solid {c["border"]};border-radius:10px;
                            padding:14px 16px;margin-bottom:14px'>
                    <div style='font-size:13px;color:{c["text"]};font-weight:500;font-family:Sora,sans-serif;line-height:1.6'>
                        {h["text"]}
                    </div>
                    <div style='margin-top:8px;font-size:11px;color:#a89080;font-family:Sora,sans-serif'>
                        Dianalisis pukul {h["time"]} · {h["date"]}
                    </div>
                </div>
                <div style='font-size:11px;color:#a89080;margin-bottom:10px;font-family:Sora,sans-serif;letter-spacing:.3px'>
                    Distribusi probabilitas
                </div>
                {probs_html}
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        col_dl, col_clr, _ = st.columns([2, 2, 4])
        col_dl.download_button(
            label="⬇ Ekspor CSV",
            data=history_to_csv(),
            file_name=f"riwayat_sentimen_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if col_clr.button("🗑 Hapus Semua", type="secondary", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ═════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════
elif current_page == "dashboard":

    section_header(
        "Dashboard",
        "Ringkasan Dataset",
        "CNBC Indonesia Stock News · Q1 2024 – Q1 2025 · 9.819 berita keuangan"
    )

    if not data_loaded or df.empty:
        st.info("Dataset tidak ditemukan. Pastikan file `hasil_data.xlsx` tersedia di direktori yang sama.")
    else:
        total   = len(df)
        positif = len(df[df["sentimen"] == "positif"])
        negatif = len(df[df["sentimen"] == "negatif"])
        netral  = len(df[df["sentimen"] == "netral"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Data", f"{total:,}", help="Jumlah seluruh berita dalam dataset")
        col2.metric("Positif",    f"{positif:,}", delta=f"{positif/total*100:.1f}%")
        col3.metric("Negatif",    f"{negatif:,}", delta=f"{negatif/total*100:.1f}%", delta_color="inverse")
        col4.metric("Netral",     f"{netral:,}",  delta=f"{netral/total*100:.1f}%",  delta_color="off")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.divider()

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.font_manager as fm

        chart_colors = {"positif": "#2d7a3a", "negatif": "#b83030", "netral": "#c45a00"}
        counts = df["sentimen"].value_counts()

        chart_col, pie_col = st.columns([3, 2])

        # Bar chart
        with chart_col:
            st.markdown("<div style='font-size:13px;font-weight:600;color:#1a1208;margin-bottom:12px;font-family:Sora,sans-serif'>Distribusi sentimen</div>", unsafe_allow_html=True)

            for sent in ["negatif", "positif", "netral"]:
                val = counts.get(sent, 0)
                pct = val / total * 100
                clr = chart_colors[sent]
                st.markdown(f"""
                <div style='margin-bottom:16px'>
                    <div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px;font-family:Sora,sans-serif'>
                        <span style='color:#5c4030;text-transform:capitalize;font-weight:500'>{sent}</span>
                        <span style='color:#1a1208;font-weight:700;font-family:JetBrains Mono,monospace'>{val:,}</span>
                    </div>
                    <div style='height:10px;background:#f0e8e0;border-radius:99px;overflow:hidden'>
                        <div style='height:100%;width:{pct:.1f}%;background:{clr};border-radius:99px'></div>
                    </div>
                    <div style='font-size:11px;color:#a89080;margin-top:4px;font-family:Sora,sans-serif'>{pct:.1f}% dari total</div>
                </div>
                """, unsafe_allow_html=True)

        # Donut chart
        with pie_col:
            st.markdown("<div style='font-size:13px;font-weight:600;color:#1a1208;margin-bottom:12px;font-family:Sora,sans-serif'>Proporsi kelas</div>", unsafe_allow_html=True)

            fig2, ax2 = plt.subplots(figsize=(4, 3.5))
            fig2.patch.set_facecolor("#ffffff")
            ax2.set_facecolor("#ffffff")

            vals   = [counts.get(k, 0) for k in ["negatif", "positif", "netral"]]
            colors = [chart_colors["negatif"], chart_colors["positif"], chart_colors["netral"]]

            wedges, texts, autotexts = ax2.pie(
                vals,
                labels=None,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                wedgeprops={"edgecolor": "white", "linewidth": 3, "width": 0.55},
                pctdistance=0.75,
                textprops={"fontsize": 11, "color": "white", "fontweight": "bold"}
            )
            ax2.legend(
                handles=[mpatches.Patch(color=chart_colors[k], label=k.capitalize()) for k in ["negatif", "positif", "netral"]],
                loc="lower center", bbox_to_anchor=(0.5, -0.05),
                ncol=3, fontsize=10, frameon=False
            )
            ax2.text(0, 0.05, f"{total:,}", ha="center", va="center", fontsize=13, fontweight="bold", color="#1a1208")
            ax2.text(0, -0.18, "berita", ha="center", va="center", fontsize=9, color="#a89080")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        st.divider()

        # Tabs for more insights
        tab1, tab2 = st.tabs(["📋 Sample Data", "📈 Statistik Teks"])

        with tab1:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            filter_sent = st.selectbox(
                "Filter:",
                ["Semua", "Positif", "Negatif", "Netral"],
                key="dash_filter",
                label_visibility="collapsed"
            )
            fmap = {"Semua": None, "Positif": "positif", "Negatif": "negatif", "Netral": "netral"}
            fk   = fmap[filter_sent]
            disp_df = df if fk is None else df[df["sentimen"] == fk]
            st.dataframe(
                disp_df.head(25),
                use_container_width=True,
                hide_index=True,
                height=340,
            )

        with tab2:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if "judul" in df.columns or "text" in df.columns:
                text_col = "judul" if "judul" in df.columns else "text"
                df["_len"] = df[text_col].astype(str).apply(len)
                avg_len = df["_len"].mean()
                min_len = df["_len"].min()
                max_len = df["_len"].max()

                c1, c2, c3 = st.columns(3)
                c1.metric("Rata-rata panjang teks", f"{avg_len:.0f} karakter")
                c2.metric("Teks terpendek",         f"{min_len} karakter")
                c3.metric("Teks terpanjang",         f"{max_len} karakter")

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:13px;font-weight:600;color:#1a1208;margin-bottom:10px;font-family:Sora,sans-serif'>Distribusi panjang teks per sentimen</div>", unsafe_allow_html=True)

                for sent in ["positif", "negatif", "netral"]:
                    sub = df[df["sentimen"] == sent]["_len"]
                    if not sub.empty:
                        clr = chart_colors[sent]
                        st.markdown(f"""
                        <div style='margin-bottom:10px;padding:12px 16px;background:#fff;border:0.5px solid #e8ddd4;border-radius:10px'>
                            <div style='font-size:12px;color:{clr};font-weight:600;text-transform:capitalize;margin-bottom:4px;font-family:Sora,sans-serif'>{sent}</div>
                            <div style='font-size:12px;color:#5c4030;font-family:JetBrains Mono,monospace'>
                                Rata-rata {sub.mean():.0f} · Min {sub.min()} · Max {sub.max()} karakter
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Kolom teks tidak ditemukan untuk analisis statistik.")

# ═════════════════════════════════════════════════════════════
# PAGE: TABEL DATA
# ═════════════════════════════════════════════════════════════
elif current_page == "data":

    section_header(
        "Data",
        "Tabel Data",
        "Preview dan filter dataset berita CNBC Indonesia."
    )

    if not data_loaded or df.empty:
        st.info("Dataset tidak ditemukan. Pastikan file `hasil_data.xlsx` tersedia.")
    else:
        # Filter row
        fcol1, fcol2, fcol3 = st.columns([2, 2, 4])

        with fcol1:
            sent_filter = st.selectbox(
                "Sentimen",
                ["Semua", "Positif", "Negatif", "Netral"],
                label_visibility="visible"
            )
        with fcol2:
            n_rows = st.selectbox(
                "Tampilkan",
                [25, 50, 100, 200, "Semua"],
                label_visibility="visible"
            )

        fmap2 = {"Semua": None, "Positif": "positif", "Negatif": "negatif", "Netral": "netral"}
        fk2   = fmap2[sent_filter]
        disp  = df if fk2 is None else df[df["sentimen"] == fk2]
        disp  = disp if n_rows == "Semua" else disp.head(int(n_rows))

        st.markdown(f"""
        <div style='font-size:12px;color:#a89080;margin:8px 0 12px;font-family:Sora,sans-serif'>
            Menampilkan {len(disp):,} dari {len(df if fk2 is None else df[df["sentimen"]==fk2]):,} data
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            disp.reset_index(drop=True),
            use_container_width=True,
            hide_index=False,
            height=480,
        )

        # Download filtered
        csv_data = disp.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇ Unduh {len(disp):,} baris sebagai CSV",
            data=csv_data,
            file_name=f"cnbc_sentimen_{sent_filter.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# ═════════════════════════════════════════════════════════════
# PAGE: TENTANG MODEL
# ═════════════════════════════════════════════════════════════
elif current_page == "about":

    section_header(
        "Informasi",
        "Tentang Model",
        "Detail teknis, spesifikasi, dan informasi dataset yang digunakan."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div style='background:#fff;border:0.5px solid #e8ddd4;border-radius:14px;padding:20px 22px;margin-bottom:14px'>
            <div style='font-size:13px;font-weight:600;color:#1a1208;margin-bottom:12px;font-family:Sora,sans-serif'>Arsitektur Model</div>
            <p style='font-size:13px;color:#a89080;line-height:1.7;font-family:Sora,sans-serif'>
                Model menggunakan kombinasi <strong style='color:#5c4030'>TF-IDF Vectorizer</strong> dengan
                <strong style='color:#5c4030'>Logistic Regression</strong> untuk klasifikasi teks berita
                keuangan berbahasa Indonesia ke dalam tiga kelas sentimen.
            </p>
        </div>
        """, unsafe_allow_html=True)

        specs = [
            ("Algoritma",    "Logistic Regression"),
            ("Fitur Ekstrasi", "TF-IDF"),
            ("Akurasi",      "~83%"),
            ("Jumlah Kelas", "3 (positif / negatif / netral)"),
            ("Total Data",   "9.819 sampel"),
            ("Bahasa",       "Bahasa Indonesia"),
        ]
        rows_html = "".join([
            f"<div style='display:flex;justify-content:space-between;padding:9px 0;"
            f"border-bottom:0.5px solid #f0e8e0;font-size:13px;font-family:Sora,sans-serif'>"
            f"<span style='color:#a89080'>{k}</span>"
            f"<span style='color:#1a1208;font-weight:500;font-family:JetBrains Mono,monospace;font-size:12px'>{v}</span></div>"
            for k, v in specs
        ])
        st.markdown(f"""
        <div style='background:#fff;border:0.5px solid #e8ddd4;border-radius:14px;padding:20px 22px'>
            <div style='font-size:13px;font-weight:600;color:#1a1208;margin-bottom:12px;font-family:Sora,sans-serif'>Spesifikasi Teknis</div>
            {rows_html}
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        dataset_info = [
            ("Sumber",   "CNBC Indonesia"),
            ("Periode",  "Q1 2024 – Q1 2025"),
            ("Topik",    "Saham & pasar modal"),
            ("Bahasa",   "Indonesia"),
            ("Jumlah",   "9.819 berita"),
        ]
        rows2_html = "".join([
            f"<div style='display:flex;justify-content:space-between;padding:9px 0;"
            f"border-bottom:0.5px solid #f0e8e0;font-size:13px;font-family:Sora,sans-serif'>"
            f"<span style='color:#a89080'>{k}</span>"
            f"<span style='color:#1a1208;font-weight:500;font-family:JetBrains Mono,monospace;font-size:12px'>{v}</span></div>"
            for k, v in dataset_info
        ])
        st.markdown(f"""
        <div style='background:#fff;border:0.5px solid #e8ddd4;border-radius:14px;padding:20px 22px;margin-bottom:14px'>
            <div style='font-size:13px;font-weight:600;color:#1a1208;margin-bottom:12px;font-family:Sora,sans-serif'>Dataset</div>
            {rows2_html}
        </div>
        """, unsafe_allow_html=True)

        # Sentiment classes
        classes_info = [
            ("positif", "#2d7a3a", "#eaf4ec", "#b6deba", "Berita yang mengindikasikan kinerja atau kondisi pasar yang menguntungkan."),
            ("negatif", "#b83030", "#fdeaea", "#f0bcbc", "Berita yang mengindikasikan penurunan, kerugian, atau tekanan pasar."),
            ("netral",  "#c45a00", "#fff3eb", "#f5d5b0", "Berita informatif atau pengumuman tanpa arah sentimen yang jelas."),
        ]
        for label, clr, bg, border, desc in classes_info:
            st.markdown(f"""
            <div style='background:{bg};border:0.5px solid {border};border-radius:10px;
                        padding:12px 14px;margin-bottom:8px'>
                <div style='font-size:11px;font-weight:600;color:{clr};text-transform:capitalize;
                            margin-bottom:4px;font-family:Sora,sans-serif;letter-spacing:.3px'>{label}</div>
                <div style='font-size:12px;color:#5c4030;line-height:1.6;font-family:Sora,sans-serif'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
