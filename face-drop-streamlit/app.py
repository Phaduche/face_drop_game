from __future__ import annotations

import hashlib
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

from face_game import build_game_html, make_demo_portrait, prepare_face_game


st.set_page_config(
    page_title="FACE DROP",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #18171c;
        --paper: #f7f5ef;
        --lime: #d9ff63;
        --pink: #ff6fae;
        --lavender: #b8a7ff;
        --line: rgba(24, 23, 28, .12);
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 12%, rgba(184, 167, 255, .28), transparent 24rem),
            radial-gradient(circle at 92% 88%, rgba(255, 111, 174, .20), transparent 25rem),
            var(--paper);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, .64);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    .block-container {
        max-width: 1260px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: "Space Grotesk", sans-serif !important;
        color: var(--ink) !important;
    }

    p, label, [data-testid="stWidgetLabel"] {
        font-family: "DM Sans", sans-serif !important;
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: .55rem;
        border: 1px solid var(--ink);
        border-radius: 999px;
        padding: .38rem .72rem;
        font-family: "Space Grotesk", sans-serif;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .12em;
        background: var(--lime);
        box-shadow: 3px 3px 0 var(--ink);
    }

    .hero-title {
        font-family: "Space Grotesk", sans-serif;
        font-weight: 700;
        font-size: clamp(2.9rem, 7vw, 6.7rem);
        letter-spacing: -.075em;
        line-height: .88;
        margin: 1rem 0 .9rem;
        max-width: 900px;
    }

    .hero-title .outline {
        color: transparent;
        -webkit-text-stroke: 1.5px var(--ink);
    }

    .hero-copy {
        max-width: 640px;
        font-family: "DM Sans", sans-serif;
        font-size: 1.03rem;
        line-height: 1.72;
        color: rgba(24, 23, 28, .72);
        margin-bottom: 1.5rem;
    }

    .micro-row {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
        margin-bottom: 1rem;
    }

    .micro-chip {
        padding: .42rem .72rem;
        border: 1px solid var(--line);
        background: rgba(255,255,255,.7);
        border-radius: 999px;
        font-family: "DM Sans", sans-serif;
        font-size: .78rem;
        font-weight: 600;
    }

    .sidebar-brand {
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -.04em;
        margin-bottom: .25rem;
    }

    .sidebar-note {
        color: rgba(24, 23, 28, .58);
        font-family: "DM Sans", sans-serif;
        font-size: .84rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }

    .tip-card {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem;
        background: rgba(255,255,255,.68);
        font-family: "DM Sans", sans-serif;
        font-size: .87rem;
        line-height: 1.55;
    }

    div[data-testid="stFileUploader"] section,
    div[data-testid="stCameraInput"] {
        border-radius: 18px;
    }

    .stRadio > div {
        gap: .45rem;
    }

    .stRadio label {
        border: 1px solid var(--line);
        background: rgba(255,255,255,.65);
        padding: .44rem .65rem;
        border-radius: 12px;
    }

    @media (max-width: 780px) {
        .block-container { padding-top: 1rem; }
        .hero-title { font-size: 3.6rem; }
        .hero-copy { font-size: .94rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _decode_image(raw: bytes) -> np.ndarray:
    """Decode and orient a user image into BGR for OpenCV."""
    pil_image = Image.open(BytesIO(raw))
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    rgb = np.asarray(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@st.cache_data(show_spinner=False, max_entries=16)
def _prepare_cached(raw: bytes, difficulty: str) -> tuple[str, list[dict], dict]:
    image = _decode_image(raw)
    return prepare_face_game(image, difficulty=difficulty)


@st.cache_data(show_spinner=False)
def _prepare_demo(difficulty: str) -> tuple[str, list[dict], dict]:
    board_uri, pieces, metadata = prepare_face_game(make_demo_portrait(), difficulty=difficulty)
    metadata["used_fallback"] = False
    metadata["is_demo"] = True
    return board_uri, pieces, metadata


with st.sidebar:
    st.markdown('<div class="sidebar-brand">FACE DROP / LAB</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-note">사진 한 장이 바로 드래그 퍼즐로 바뀝니다.</div>',
        unsafe_allow_html=True,
    )

    source = st.radio(
        "얼굴 선택",
        ("데모로 시작", "사진 업로드", "카메라"),
        label_visibility="collapsed",
    )

    uploaded_bytes: bytes | None = None
    if source == "사진 업로드":
        upload = st.file_uploader(
            "정면 사진을 올려주세요",
            type=["jpg", "jpeg", "png", "webp"],
            help="얼굴이 정면을 보고 있고 밝은 사진일수록 결과가 좋아요.",
        )
        if upload is not None:
            uploaded_bytes = upload.getvalue()
    elif source == "카메라":
        camera = st.camera_input("정면을 보고 촬영해주세요")
        if camera is not None:
            uploaded_bytes = camera.getvalue()

    difficulty_label = st.select_slider(
        "난이도",
        options=["쉬움", "보통", "어려움"],
        value="보통",
    )
    difficulty_map = {"쉬움": "easy", "보통": "normal", "어려움": "hard"}
    difficulty = difficulty_map[difficulty_label]

    st.markdown("---")
    st.markdown(
        """
        <div class="tip-card">
          <b>사진 팁</b><br>
          정면 얼굴 · 충분한 조명 · 얼굴이 화면의 절반 이상인 사진이 가장 잘 맞습니다.
          업로드한 사진은 저장하지 않고 현재 실행 중에만 처리합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


left, right = st.columns([0.82, 1.18], gap="large")

with left:
    st.markdown('<div class="hero-kicker">◉ OPENCV FACE PUZZLE</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-title">
          PUT YOUR<br>
          <span class="outline">FACE</span> BACK.
        </div>
        <div class="hero-copy">
          흩어진 눈·코·입을 드래그해서 원래 얼굴을 완성하세요.
          정확한 위치에 가까울수록 높은 점수를 받고, 조각이 맞으면 자석처럼 고정됩니다.
        </div>
        <div class="micro-row">
          <span class="micro-chip">DRAG 4 PIECES</span>
          <span class="micro-chip">TOUCH READY</span>
          <span class="micro-chip">LIVE SCORE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        **플레이 방법**

        1. 흩어진 얼굴 조각을 잡아 이동합니다.
        2. 원래 위치에 가까우면 자동으로 붙습니다.
        3. 막히면 `HINT`, 처음부터 하려면 `RESET`을 누르세요.
        """,
    )

with right:
    try:
        if uploaded_bytes:
            digest = hashlib.sha1(uploaded_bytes).hexdigest()[:10]
            with st.spinner("얼굴 조각을 만드는 중…"):
                board_uri, pieces, metadata = _prepare_cached(uploaded_bytes, difficulty)
        else:
            digest = "demo"
            board_uri, pieces, metadata = _prepare_demo(difficulty)

        game_html = build_game_html(
            board_uri=board_uri,
            pieces=pieces,
            difficulty=difficulty,
            game_id=f"{digest}-{difficulty}",
        )
        components.html(game_html, height=820, scrolling=False)

        if metadata.get("is_demo"):
            st.caption("기본 데모 얼굴입니다. 왼쪽에서 사진이나 카메라를 선택하면 내 얼굴 퍼즐로 바뀝니다.")
        elif metadata.get("used_fallback"):
            st.caption(
                "얼굴 자동 감지가 어려워 사진 중앙을 기준으로 퍼즐을 만들었습니다. "
                "더 정확하게 하려면 밝은 정면 사진을 사용해주세요."
            )
        else:
            st.caption("OpenCV가 얼굴을 감지해 퍼즐 영역을 자동으로 맞췄습니다.")
    except Exception as exc:
        st.error(
            "이 사진으로 퍼즐을 만들지 못했어요. 정면 얼굴이 더 크게 나온 JPG 또는 PNG로 다시 시도해주세요."
        )
        with st.expander("오류 정보"):
            st.code(str(exc))
