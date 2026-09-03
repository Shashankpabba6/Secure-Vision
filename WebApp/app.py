"""
Secure Vision - Face Liveness & Deepfake Detection
Premium dark security-console UI (Streamlit).
"""
import streamlit as st
import cv2
import numpy as np
import tempfile, os, time

from config import FACE_API_PROVIDER
from face_client import get_face_client
from physiofusion import PhysioFusionPipeline
from deepfake_detection import predict_deepfake


@st.cache_resource
def get_pipeline():
    return PhysioFusionPipeline(device="cpu")

@st.cache_resource
def get_or_client():
    try:
        return get_face_client()
    except Exception:
        return None


st.set_page_config(page_title="Secure Vision", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# Design layer
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
  --bg: #070B14;
  --card: rgba(255,255,255,0.03);
  --border: rgba(255,255,255,0.08);
  --cyan: #22D3EE;
  --emerald: #34D399;
  --red: #F87171;
  --amber: #FBBF24;
  --text: #E5EAF2;
  --muted: #8B95A7;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--bg) !important;
  color: var(--text);
  font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1000px 500px at 80% -10%, rgba(34,211,238,0.07), transparent 60%),
    radial-gradient(800px 400px at 10% 110%, rgba(52,211,153,0.05), transparent 60%),
    var(--bg) !important;
}
#MainMenu, [data-testid="stHeader"], footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], .stAppDeployButton { display: none !important; }

.main .block-container {
  max-width: 1080px;
  padding-top: 1.6rem;
  padding-bottom: 4rem;
}

h1, h2, h3, .sv-display {
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: -0.02em;
}
p, li, label, .stMarkdown { color: var(--text); }
small, .sv-muted { color: var(--muted); }

/* Header bar */
.sv-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 18px 22px; margin-bottom: 8px;
  background: var(--card); border: 1px solid var(--border);
  border-radius: 16px; backdrop-filter: blur(12px);
}
.sv-brand { display: flex; align-items: center; gap: 14px; }
.sv-wordmark {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: 1.35rem; letter-spacing: 0.04em; color: var(--text);
}
.sv-wordmark span {
  background: linear-gradient(90deg, var(--cyan), var(--emerald));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.sv-tagline { font-size: 0.8rem; color: var(--muted); margin-top: 2px; }
.sv-status {
  display: flex; align-items: center; gap: 8px;
  font-family: 'Space Grotesk', sans-serif; font-size: 0.72rem;
  letter-spacing: 0.12em; color: var(--emerald);
  border: 1px solid rgba(52,211,153,0.35); border-radius: 999px;
  padding: 6px 14px; background: rgba(52,211,153,0.08);
}
.sv-dot {
  width: 8px; height: 8px; border-radius: 50%; background: var(--emerald);
  animation: sv-pulse 2s ease-in-out infinite;
}
@keyframes sv-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

/* Tabs */
[data-baseweb="tab-list"] {
  gap: 8px; border-bottom: 1px solid var(--border) !important;
  background: transparent !important;
}
button[data-baseweb="tab"] {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.95rem !important; font-weight: 500 !important;
  color: var(--muted) !important; background: transparent !important;
  padding: 12px 20px !important; border-radius: 10px 10px 0 0 !important;
  transition: color .2s ease, background .2s ease;
}
button[data-baseweb="tab"]:hover { color: var(--text) !important; }
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--cyan) !important;
  background: rgba(34,211,238,0.06) !important;
}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
  background-color: var(--cyan) !important;
}

/* Glass cards (st.container(border=True)) */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 8px 10px !important;
  backdrop-filter: blur(12px);
}

/* File uploader dropzone */
[data-testid="stFileUploaderDropzone"] {
  background: rgba(34,211,238,0.03) !important;
  border: 1.5px dashed rgba(34,211,238,0.35) !important;
  border-radius: 14px !important;
  transition: border-color .2s ease, background .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--cyan) !important;
  background: rgba(34,211,238,0.07) !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; }

/* Buttons */
.stButton > button, [data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important; letter-spacing: 0.02em;
  background: linear-gradient(90deg, rgba(34,211,238,0.15), rgba(52,211,153,0.15)) !important;
  color: var(--text) !important;
  border: 1px solid rgba(34,211,238,0.4) !important;
  border-radius: 10px !important;
  transition: transform .15s ease, filter .15s ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  filter: brightness(1.25);
  border-color: var(--cyan) !important;
}

/* Inputs */
[data-testid="stTextInput"] input, .stSelectbox > div > div {
  background: var(--card) !important; border-color: var(--border) !important;
  color: var(--text) !important; border-radius: 10px !important;
}
[data-testid="stSlider"] [role="slider"] { background: var(--cyan) !important; }

/* Camera input */
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img,
[data-testid="stImage"] img, [data-testid="stVideo"] video {
  border-radius: 14px !important;
  border: 1px solid var(--border);
}

/* Verdict banners */
.sv-banner {
  display: flex; align-items: center; gap: 20px;
  padding: 20px 24px; border-radius: 16px; margin: 14px 0;
  border: 1px solid; backdrop-filter: blur(12px);
  animation: sv-rise .35s ease both;
}
@keyframes sv-rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
.sv-banner.pass { border-color: rgba(52,211,153,0.45); background: rgba(52,211,153,0.08); box-shadow: 0 0 40px rgba(52,211,153,0.08); }
.sv-banner.fail { border-color: rgba(248,113,113,0.5); background: rgba(248,113,113,0.08); box-shadow: 0 0 40px rgba(248,113,113,0.12); }
.sv-banner.warn { border-color: rgba(251,191,36,0.45); background: rgba(251,191,36,0.07); }
.sv-verdict {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: 1.3rem; letter-spacing: 0.02em;
}
.sv-banner.pass .sv-verdict { color: var(--emerald); }
.sv-banner.fail .sv-verdict { color: var(--red); }
.sv-banner.warn .sv-verdict { color: var(--amber); }
.sv-detail { color: var(--muted); font-size: 0.88rem; margin-top: 4px; }
.sv-gauge-text {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: 17px; fill: var(--text);
}
.sv-gauge-label {
  font-family: 'Inter', sans-serif; font-size: 7.5px;
  fill: var(--muted); letter-spacing: 0.14em;
}

/* Metric tiles */
.sv-tiles { display: flex; gap: 14px; margin: 12px 0; }
.sv-tile {
  flex: 1; background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; padding: 16px 18px; backdrop-filter: blur(12px);
}
.sv-tile-value {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.5rem;
  background: linear-gradient(90deg, var(--cyan), var(--emerald));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.sv-tile-label {
  font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin-top: 4px;
}

/* Section titles */
.sv-section {
  font-family: 'Space Grotesk', sans-serif; font-weight: 600;
  font-size: 1.05rem; color: var(--text); margin: 4px 0 2px;
}

/* Expander & alerts */
[data-testid="stExpander"] {
  background: var(--card); border: 1px solid var(--border) !important;
  border-radius: 14px !important;
}
[data-testid="stAlert"] {
  background: var(--card) !important; border-radius: 12px !important;
  border: 1px solid var(--border) !important;
}
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, var(--cyan), var(--emerald)) !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Reusable components
# ---------------------------------------------------------------------------
ICON_SHIELD = """<svg width="34" height="34" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2L4 5.5V11c0 4.97 3.4 9.62 8 10.5 4.6-.88 8-5.53 8-10.5V5.5L12 2z" stroke="url(#svg)" stroke-width="1.6" fill="rgba(34,211,238,0.08)"/>
<path d="M8.5 12l2.4 2.4L15.5 9.8" stroke="#34D399" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
<defs><linearGradient id="svg" x1="4" y1="2" x2="20" y2="21"><stop stop-color="#22D3EE"/><stop offset="1" stop-color="#34D399"/></linearGradient></defs>
</svg>"""

ICON_CHECK = """<svg width="30" height="30" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="#34D399" stroke-width="1.6" fill="rgba(52,211,153,0.1)"/><path d="M8 12.5l2.6 2.6L16 9.5" stroke="#34D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>"""
ICON_ALERT = """<svg width="30" height="30" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="#F87171" stroke-width="1.6" fill="rgba(248,113,113,0.1)"/><path d="M12 7.5v5.5" stroke="#F87171" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="16.4" r="1.1" fill="#F87171"/></svg>"""
ICON_WARN  = """<svg width="30" height="30" viewBox="0 0 24 24" fill="none"><path d="M12 3.5L22 20H2L12 3.5z" stroke="#FBBF24" stroke-width="1.6" fill="rgba(251,191,36,0.08)" stroke-linejoin="round"/><path d="M12 10v4.4" stroke="#FBBF24" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="17.2" r="1.05" fill="#FBBF24"/></svg>"""


def gauge_svg(confidence: float, color: str) -> str:
    dash = max(0.0, min(100.0, confidence)) / 100 * 226.19
    return f"""<svg width="84" height="84" viewBox="0 0 84 84">
<circle cx="42" cy="42" r="36" stroke="rgba(255,255,255,0.08)" stroke-width="7" fill="none"/>
<circle cx="42" cy="42" r="36" stroke="{color}" stroke-width="7" fill="none" stroke-linecap="round"
  stroke-dasharray="{dash:.1f} 226.19" transform="rotate(-90 42 42)"/>
<text x="42" y="45" text-anchor="middle" class="sv-gauge-text">{confidence:.0f}%</text>
<text x="42" y="58" text-anchor="middle" class="sv-gauge-label">CONFIDENCE</text>
</svg>"""


def verdict_banner(level: str, title: str, detail: str = "", confidence: float = None):
    icons = {"pass": ICON_CHECK, "fail": ICON_ALERT, "warn": ICON_WARN}
    colors = {"pass": "#34D399", "fail": "#F87171", "warn": "#FBBF24"}
    gauge = gauge_svg(confidence, colors[level]) if confidence is not None else ""
    detail_html = f'<div class="sv-detail">{detail}</div>' if detail else ""
    st.markdown(
        f"""<div class="sv-banner {level}">
{icons[level]}
<div style="flex:1"><div class="sv-verdict">{title}</div>{detail_html}</div>
{gauge}
</div>""",
        unsafe_allow_html=True,
    )


def metric_tiles(items):
    tiles = "".join(
        f'<div class="sv-tile"><div class="sv-tile-value">{v}</div>'
        f'<div class="sv-tile-label">{l}</div></div>'
        for l, v in items
    )
    st.markdown(f'<div class="sv-tiles">{tiles}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""<div class="sv-header">
<div class="sv-brand">{ICON_SHIELD}
<div><div class="sv-wordmark">SECURE <span>VISION</span></div>
<div class="sv-tagline">Multi-modal liveness detection &amp; deepfake analysis</div></div>
</div>
<div class="sv-status"><div class="sv-dot"></div>SYSTEMS NOMINAL</div>
</div>""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Photo Verification", "Live Video Analysis", "Deepfake Detection"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — Single image (OpenRouter primary, PhysioFusion secondary)
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sv-section">Capture or upload a face photo for liveness verification</div>',
                unsafe_allow_html=True)

    src = st.radio("Source", ["Upload image", "Capture from webcam"], horizontal=True, key="t1_src")
    img = None

    if src == "Upload image":
        f = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"], key="t1_up")
        if f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    else:
        cam = st.camera_input("Take a picture", key="t1_cam")
        if cam:
            img = cv2.imdecode(np.frombuffer(cam.read(), np.uint8), cv2.IMREAD_COLOR)

    if img is not None:
        c1, c2 = st.columns([1, 1])
        with c1:
            with st.container(border=True):
                st.markdown('<div class="sv-section">Captured frame</div>', unsafe_allow_html=True)
                st.image(img, channels="BGR", width="stretch")

        with c2:
            client = get_or_client()
            if client is not None:
                with st.spinner("Vision model analyzing liveness..."):
                    r = client.verify_face(img)
                lbl = r.get("label", "")
                conf = r.get("confidence", 0)
                reas = r.get("reasoning", "")
                if lbl == "real":
                    verdict_banner("pass", "LIVE PERSON VERIFIED", reas, conf)
                elif lbl == "error":
                    verdict_banner("warn", "ANALYSIS UNAVAILABLE", reas)
                    st.info("Set OPENROUTER_API_KEY in .env or switch to the roboflow provider")
                else:
                    verdict_banner("fail", "SPOOF DETECTED", reas, conf)
            else:
                verdict_banner("warn", "VISION API NOT CONFIGURED",
                               "Set OPENROUTER_API_KEY in .env to enable cloud liveness checks")

            with st.expander("PhysioFusion static analysis"):
                pipe = get_pipeline()
                pf_result = pipe.process_frame(img)
                if pf_result is None:
                    st.write("No face detected")
                else:
                    st.progress(pf_result.rppg_score, text=f"Texture quality: {pf_result.rppg_score:.0%}")
                    st.progress(pf_result.depth_score, text=f"Depth naturalness: {pf_result.depth_score:.0%}")
                    st.caption(pf_result.explanation)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — Live video with PhysioFusion temporal analysis
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sv-section">Real-time liveness via webcam</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sv-muted" style="margin-bottom:10px">PhysioFusion accumulates frames over time '
        'to detect pulse (rPPG), involuntary micro-motion, and 3D depth consistency.</div>',
        unsafe_allow_html=True)

    for k in ["pf_buffer", "pf_count", "pf_last_result"]:
        if k not in st.session_state:
            st.session_state[k] = [] if "buffer" in k else (0 if "count" in k else None)

    running = st.checkbox("Start live capture", key="live_running")

    if running:
        col_settings, _ = st.columns([1, 2])
        with col_settings:
            every_n = st.slider("Analyze every N captures", 5, 50, 15, key="every_n")
            n_required = st.slider("Frames for full analysis", 15, 90, 30, key="n_req")

        status_ph = st.empty()
        frame_ph = st.empty()

        cam2 = st.camera_input("Capture frame", key=f"live_cam_{st.session_state.pf_count}")
        if cam2:
            frame = cv2.imdecode(np.frombuffer(cam2.read(), np.uint8), cv2.IMREAD_COLOR)
            frame_ph.image(frame, channels="BGR", caption=f"Frame {st.session_state.pf_count + 1}",
                           width="stretch")

            pipe = get_pipeline()
            res = pipe.process_frame(frame)
            st.session_state.pf_buffer.append(frame)
            if len(st.session_state.pf_buffer) > 300:
                st.session_state.pf_buffer = st.session_state.pf_buffer[-300:]
            st.session_state.pf_count += 1
            st.session_state.pf_last_result = res

            n = st.session_state.pf_count
            status_ph.info(
                f"Captured {n} frames | buffer {len(st.session_state.pf_buffer)} | "
                f"need {n_required} for full temporal analysis"
            )

            if res and n >= every_n and n % every_n == 0:
                if res.is_live:
                    verdict_banner("pass", "LIVE SIGNALS DETECTED", res.explanation,
                                   res.confidence * 100)
                else:
                    verdict_banner("warn", "POTENTIAL SPOOF SIGNALS", res.explanation,
                                   (1 - res.confidence) * 100)
                metric_tiles([
                    ("Heart pulse (rPPG)", f"{res.rppg_score:.0%}"),
                    ("Micro-motion", f"{res.motion_score:.0%}"),
                    ("Depth geometry", f"{res.depth_score:.0%}"),
                ])

            if n >= n_required:
                if res and res.is_live:
                    verdict_banner("pass", "LIVE PERSON CONFIRMED",
                                   f"Full temporal analysis complete. {res.explanation}",
                                   res.confidence * 100)
                elif res:
                    verdict_banner("fail", "SPOOF LIKELY",
                                   f"Full temporal analysis complete. {res.explanation}",
                                   (1 - res.confidence) * 100)

            if running:
                time.sleep(0.5)
                st.rerun()
    else:
        last = st.session_state.pf_last_result
        if last:
            if last.is_live:
                verdict_banner("pass", "LAST CAPTURE: LIVE", last.explanation,
                               last.confidence * 100)
            else:
                verdict_banner("warn", "LAST CAPTURE: UNCERTAIN", last.explanation,
                               (1 - last.confidence) * 100)

    if st.button("Reset live buffer"):
        st.session_state.pf_buffer = []
        st.session_state.pf_count = 0
        st.session_state.pf_last_result = None
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# TAB 3 — Upload video for deepfake detection
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sv-section">Upload a video to check for deepfakes</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sv-muted" style="margin-bottom:10px">ResNeXt50 + LSTM trained on face-cropped '
        'deepfake data. Adaptive face detection crops the face region before analysis.</div>',
        unsafe_allow_html=True)

    vid = st.file_uploader("Choose video", type=["mp4", "avi", "mov", "mkv"])
    if vid:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(vid.read())
            path = tmp.name

        with st.container(border=True):
            st.video(vid)

        with st.spinner("Analyzing frames for deepfake artifacts..."):
            prog = st.progress(0, text="Extracting frames...")
            try:
                prog.progress(20, text="Running ResNeXt50 + LSTM inference...")
                result, confidence, img_path = predict_deepfake(path)
                prog.progress(100, text="Done")

                if result.upper() == "FAKE":
                    verdict_banner("fail", "DEEPFAKE DETECTED",
                                   "Facial manipulation artifacts identified in the video.",
                                   confidence)
                else:
                    verdict_banner("pass", "REAL VIDEO",
                                   "No manipulation artifacts detected.", confidence)

                if img_path and os.path.exists(img_path):
                    with st.container(border=True):
                        st.markdown('<div class="sv-section">Attention heatmap — where the model looked</div>',
                                    unsafe_allow_html=True)
                        st.image(img_path, width="stretch")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
            finally:
                if os.path.exists(path):
                    os.unlink(path)

    st.divider()
    st.markdown(
        '<div class="sv-muted" style="font-size:0.85rem"><b style="color:#E5EAF2">PhysioFusion</b> — '
        'multi-physiological liveness detection fusing <b style="color:#22D3EE">rPPG</b> '
        '(blood pulse from skin color changes), <b style="color:#22D3EE">micro-motion</b> '
        '(involuntary head tremor via optical flow on facial landmarks), and '
        '<b style="color:#22D3EE">depth consistency</b> (geometric 3D structure analysis).</div>',
        unsafe_allow_html=True)
