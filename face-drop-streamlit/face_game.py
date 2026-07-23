from __future__ import annotations

import base64
import json
from dataclasses import dataclass

import cv2
import numpy as np


CANVAS_SIZE = 700


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    cx: float
    cy: float
    width: float
    height: float
    start_x: float
    start_y: float
    rotation: float


FEATURES = (
    FeatureSpec("left_eye", "왼쪽 눈", 0.34, 0.385, 0.245, 0.145, 0.055, 0.145, -10),
    FeatureSpec("right_eye", "오른쪽 눈", 0.66, 0.385, 0.245, 0.145, 0.705, 0.125, 12),
    FeatureSpec("nose", "코", 0.50, 0.565, 0.235, 0.270, 0.070, 0.670, -8),
    FeatureSpec("mouth", "입", 0.50, 0.748, 0.430, 0.180, 0.535, 0.735, 8),
)


def _png_data_uri(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG 인코딩에 실패했습니다.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def make_demo_portrait(size: int = 900) -> np.ndarray:
    """Create a deterministic, photo-like portrait using only OpenCV."""
    yy, xx = np.mgrid[0:size, 0:size]

    # Warm editorial background.
    base = np.zeros((size, size, 3), dtype=np.float32)
    base[..., 0] = 205 + 22 * (yy / size)
    base[..., 1] = 214 + 18 * (xx / size)
    base[..., 2] = 237
    glow = np.exp(-(((xx - size * 0.48) / (size * 0.58)) ** 2 + ((yy - size * 0.43) / (size * 0.68)) ** 2))
    base += glow[..., None] * np.array([15, 13, 10], dtype=np.float32)
    image = np.clip(base, 0, 255).astype(np.uint8)

    # Neck and shoulders.
    cv2.ellipse(image, (size // 2, int(size * 0.96)), (int(size * 0.41), int(size * 0.25)), 0, 180, 360, (70, 65, 75), -1, cv2.LINE_AA)
    cv2.rectangle(
        image,
        (int(size * 0.43), int(size * 0.70)),
        (int(size * 0.57), int(size * 0.90)),
        (158, 188, 221),
        -1,
    )

    # Hair silhouette and face.
    cv2.ellipse(image, (size // 2, int(size * 0.45)), (int(size * 0.265), int(size * 0.365)), 0, 0, 360, (39, 38, 50), -1, cv2.LINE_AA)
    cv2.ellipse(image, (size // 2, int(size * 0.49)), (int(size * 0.225), int(size * 0.315)), 0, 0, 360, (168, 198, 232), -1, cv2.LINE_AA)

    # Ears.
    cv2.ellipse(image, (int(size * 0.285), int(size * 0.49)), (int(size * 0.035), int(size * 0.072)), 0, 0, 360, (155, 187, 224), -1, cv2.LINE_AA)
    cv2.ellipse(image, (int(size * 0.715), int(size * 0.49)), (int(size * 0.035), int(size * 0.072)), 0, 0, 360, (155, 187, 224), -1, cv2.LINE_AA)

    # Hair fringe.
    fringe = np.array(
        [
            [int(size * 0.29), int(size * 0.33)],
            [int(size * 0.35), int(size * 0.17)],
            [int(size * 0.48), int(size * 0.13)],
            [int(size * 0.62), int(size * 0.17)],
            [int(size * 0.71), int(size * 0.30)],
            [int(size * 0.61), int(size * 0.27)],
            [int(size * 0.55), int(size * 0.31)],
            [int(size * 0.47), int(size * 0.25)],
            [int(size * 0.39), int(size * 0.31)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [fringe], (42, 40, 53), cv2.LINE_AA)

    # Brows.
    cv2.ellipse(image, (int(size * 0.34), int(size * 0.335)), (int(size * 0.075), int(size * 0.018)), -5, 190, 345, (60, 55, 66), 8, cv2.LINE_AA)
    cv2.ellipse(image, (int(size * 0.66), int(size * 0.335)), (int(size * 0.075), int(size * 0.018)), 5, 195, 350, (60, 55, 66), 8, cv2.LINE_AA)

    # Eyes.
    for center in ((int(size * 0.34), int(size * 0.385)), (int(size * 0.66), int(size * 0.385))):
        cv2.ellipse(image, center, (int(size * 0.060), int(size * 0.031)), 0, 0, 360, (242, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(image, center, int(size * 0.020), (73, 87, 95), -1, cv2.LINE_AA)
        cv2.circle(image, center, int(size * 0.010), (25, 27, 32), -1, cv2.LINE_AA)
        cv2.circle(image, (center[0] - 5, center[1] - 6), int(size * 0.0045), (255, 255, 255), -1, cv2.LINE_AA)

    # Nose, with highlight and shadow.
    nose = np.array(
        [
            [int(size * 0.50), int(size * 0.43)],
            [int(size * 0.47), int(size * 0.565)],
            [int(size * 0.50), int(size * 0.59)],
            [int(size * 0.535), int(size * 0.565)],
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [nose], False, (125, 151, 184), 6, cv2.LINE_AA)
    cv2.ellipse(image, (int(size * 0.50), int(size * 0.59)), (int(size * 0.050), int(size * 0.020)), 0, 5, 175, (114, 139, 171), 5, cv2.LINE_AA)

    # Mouth.
    mouth_center = (int(size * 0.50), int(size * 0.748))
    cv2.ellipse(image, mouth_center, (int(size * 0.090), int(size * 0.036)), 0, 0, 180, (110, 72, 112), -1, cv2.LINE_AA)
    cv2.ellipse(image, (mouth_center[0], mouth_center[1] + int(size * 0.010)), (int(size * 0.090), int(size * 0.035)), 0, 180, 360, (135, 82, 127), -1, cv2.LINE_AA)
    cv2.line(
        image,
        (mouth_center[0] - int(size * 0.075), mouth_center[1]),
        (mouth_center[0] + int(size * 0.075), mouth_center[1]),
        (80, 57, 82),
        3,
        cv2.LINE_AA,
    )

    # Very light texture keeps the OpenCV result from feeling synthetic.
    rng = np.random.default_rng(37)
    noise = rng.normal(0, 1.6, image.shape).astype(np.float32)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0.35)


def _detect_and_normalize(image: np.ndarray) -> tuple[np.ndarray, bool]:
    if image is None or image.size == 0:
        raise ValueError("비어 있는 이미지입니다.")

    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side > 1800:
        scale = 1800 / max_side
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(max(72, min(h, w) // 7), max(72, min(h, w) // 7)),
    )

    used_fallback = len(faces) == 0
    if not used_fallback:
        x, y, fw, fh = max(faces, key=lambda box: box[2] * box[3])
        side = int(max(fw, fh) * 1.62)
        cx = x + fw // 2
        cy = y + int(fh * 0.52)
    else:
        side = int(min(h, w) * 0.94)
        cx = w // 2
        cy = h // 2

    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("얼굴 영역을 잘라낼 수 없습니다.")
    normalized = cv2.resize(crop, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_AREA)
    return normalized, used_fallback


def _soft_ellipse_mask(height: int, width: int, feather: int = 17) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (width // 2, height // 2),
        (max(2, width // 2 - feather), max(2, height // 2 - feather)),
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    if feather > 0:
        kernel = max(3, feather * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), feather / 2.4)
    return mask


def prepare_face_game(
    image: np.ndarray,
    difficulty: str = "normal",
) -> tuple[str, list[dict], dict]:
    """Turn a portrait into a featureless board and four draggable PNG pieces."""
    normalized, used_fallback = _detect_and_normalize(image)
    board = normalized.copy()
    pieces: list[dict] = []

    for spec in FEATURES:
        width = int(CANVAS_SIZE * spec.width)
        height = int(CANVAS_SIZE * spec.height)
        cx = int(CANVAS_SIZE * spec.cx)
        cy = int(CANVAS_SIZE * spec.cy)
        x1 = max(0, cx - width // 2)
        y1 = max(0, cy - height // 2)
        x2 = min(CANVAS_SIZE, x1 + width)
        y2 = min(CANVAS_SIZE, y1 + height)
        width = x2 - x1
        height = y2 - y1

        crop = normalized[y1:y2, x1:x2].copy()
        alpha = _soft_ellipse_mask(height, width, feather=max(8, int(min(width, height) * 0.10)))
        rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha

        pieces.append(
            {
                "key": spec.key,
                "label": spec.label,
                "src": _png_data_uri(rgba),
                "w": width,
                "h": height,
                "targetX": x1,
                "targetY": y1,
                "startX": int(CANVAS_SIZE * spec.start_x),
                "startY": int(CANVAS_SIZE * spec.start_y),
                "rotation": spec.rotation,
            }
        )

        # Remove the feature from the game board using OpenCV inpainting, then
        # soften the transition with a feathered blend.
        full_mask = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = np.where(alpha > 38, 255, 0).astype(np.uint8)
        inpainted = cv2.inpaint(board, full_mask, 11, cv2.INPAINT_TELEA)
        softened = cv2.GaussianBlur(inpainted, (31, 31), 0)
        blend_alpha = cv2.GaussianBlur(full_mask, (35, 35), 0).astype(np.float32) / 255.0
        board = (
            board.astype(np.float32) * (1.0 - blend_alpha[..., None])
            + softened.astype(np.float32) * blend_alpha[..., None]
        ).astype(np.uint8)

    # A restrained editorial grade gives different phone photos a coherent look.
    lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    board = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    return (
        _png_data_uri(board),
        pieces,
        {"used_fallback": used_fallback, "difficulty": difficulty},
    )


def build_game_html(
    board_uri: str,
    pieces: list[dict],
    difficulty: str,
    game_id: str,
) -> str:
    thresholds = {"easy": 72, "normal": 48, "hard": 30}
    threshold = thresholds.get(difficulty, 48)
    pieces_json = json.dumps(pieces, ensure_ascii=False)
    board_json = json.dumps(board_uri)
    game_id_json = json.dumps(game_id)

    return f"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; background: transparent; color: #18171c; }}
    body {{ font-family: "DM Sans", system-ui, sans-serif; overflow: hidden; }}

    .card {{
      width: 100%;
      border: 1.4px solid #18171c;
      border-radius: 28px;
      background: rgba(255,255,255,.82);
      box-shadow: 8px 8px 0 #18171c;
      padding: 14px;
    }}

    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 2px 3px 12px;
    }}

    .title {{
      font-family: "Space Grotesk", sans-serif;
      font-size: 14px;
      font-weight: 700;
      letter-spacing: .08em;
    }}

    .status {{
      display: flex;
      align-items: center;
      gap: 7px;
      font-size: 12px;
      font-weight: 700;
    }}

    .dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #ff6fae;
      box-shadow: 0 0 0 4px rgba(255,111,174,.15);
    }}

    .stage-wrap {{
      position: relative;
      width: 100%;
      max-width: 700px;
      margin: 0 auto;
    }}

    .stage {{
      width: 100%;
      aspect-ratio: 1 / 1;
      position: relative;
      overflow: hidden;
      border: 1px solid rgba(24,23,28,.18);
      border-radius: 21px;
      background: #e9e6dc;
      touch-action: none;
      user-select: none;
      isolation: isolate;
    }}

    .board {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      pointer-events: none;
      -webkit-user-drag: none;
    }}

    .grain {{
      position: absolute;
      inset: 0;
      z-index: 1;
      opacity: .12;
      pointer-events: none;
      background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 160 160' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.6'/%3E%3C/svg%3E");
      mix-blend-mode: soft-light;
    }}

    .target {{
      position: absolute;
      z-index: 2;
      border: 2px dashed rgba(217,255,99,.0);
      background: rgba(217,255,99,0);
      border-radius: 50%;
      transition: .2s ease;
      pointer-events: none;
    }}

    .stage.show-hints .target {{
      border-color: rgba(217,255,99,.95);
      background: rgba(217,255,99,.12);
      box-shadow: 0 0 0 2px rgba(24,23,28,.28);
    }}

    .piece {{
      position: absolute;
      z-index: 5;
      cursor: grab;
      touch-action: none;
      filter: drop-shadow(0 7px 8px rgba(24,23,28,.25));
      transition: filter .16s ease, opacity .2s ease, outline .2s ease;
      will-change: left, top, transform;
    }}

    .piece img {{
      display: block;
      width: 100%;
      height: 100%;
      pointer-events: none;
      -webkit-user-drag: none;
    }}

    .piece.dragging {{
      cursor: grabbing;
      z-index: 20;
      filter: drop-shadow(0 12px 14px rgba(24,23,28,.34));
    }}

    .piece.placed {{
      cursor: default;
      filter: none;
      animation: lock .34s ease-out;
    }}

    @keyframes lock {{
      0% {{ transform: scale(.94); }}
      55% {{ transform: scale(1.055); }}
      100% {{ transform: scale(1); }}
    }}

    .complete {{
      position: absolute;
      z-index: 30;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%) scale(.92);
      width: min(78%, 460px);
      border: 1.5px solid #18171c;
      border-radius: 22px;
      padding: 23px;
      background: #d9ff63;
      box-shadow: 7px 7px 0 #18171c;
      text-align: center;
      opacity: 0;
      pointer-events: none;
      transition: .28s ease;
    }}

    .complete.visible {{
      opacity: 1;
      transform: translate(-50%, -50%) scale(1);
      pointer-events: auto;
    }}

    .complete small {{
      display: block;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .11em;
      margin-bottom: 4px;
    }}

    .complete strong {{
      display: block;
      font-family: "Space Grotesk", sans-serif;
      font-size: clamp(35px, 7vw, 58px);
      letter-spacing: -.06em;
      line-height: 1;
      margin: 5px 0 8px;
    }}

    .complete p {{
      margin: 0 0 15px;
      font-size: 13px;
      font-weight: 600;
    }}

    .toolbar {{
      display: grid;
      grid-template-columns: 1.15fr .85fr .85fr;
      gap: 8px;
      padding-top: 11px;
    }}

    button {{
      appearance: none;
      border: 1.3px solid #18171c;
      border-radius: 13px;
      padding: 11px 10px;
      font: 700 12px "DM Sans", sans-serif;
      letter-spacing: .04em;
      color: #18171c;
      background: white;
      cursor: pointer;
      transition: transform .12s ease, box-shadow .12s ease;
    }}

    button:hover {{ transform: translateY(-1px); box-shadow: 3px 3px 0 #18171c; }}
    button:active {{ transform: translateY(1px); box-shadow: none; }}
    .primary {{ background: #ff6fae; }}
    .again {{ background: white; min-width: 145px; }}

    .readout {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 11px 4px 1px;
      font-size: 12px;
      font-weight: 700;
    }}

    .progress {{
      height: 8px;
      flex: 1;
      max-width: 180px;
      border: 1px solid #18171c;
      border-radius: 99px;
      overflow: hidden;
      background: white;
    }}

    .progress > i {{
      display: block;
      width: 0;
      height: 100%;
      background: #b8a7ff;
      transition: width .25s ease;
    }}

    @media (max-width: 520px) {{
      .card {{ padding: 10px; border-radius: 22px; box-shadow: 5px 5px 0 #18171c; }}
      .stage {{ border-radius: 17px; }}
      .toolbar {{ grid-template-columns: 1fr 1fr 1fr; }}
      button {{ font-size: 10px; padding: 10px 5px; }}
      .topbar {{ padding-bottom: 9px; }}
      .title {{ font-size: 12px; }}
    }}
  </style>
</head>
<body>
  <main class="card" data-game={game_id_json}>
    <header class="topbar">
      <div class="title">FACE DROP / 01</div>
      <div class="status"><span class="dot"></span><span id="statusText">READY TO DROP</span></div>
    </header>

    <div class="stage-wrap">
      <section class="stage" id="stage" aria-label="얼굴 맞추기 게임 영역">
        <img class="board" src={board_json} alt="눈, 코, 입이 비어 있는 얼굴">
        <div class="grain"></div>
        <div class="complete" id="complete">
          <small>FACE RESTORED</small>
          <strong id="finalScore">1000</strong>
          <p id="finalMeta">정확도 100% · 0.0초</p>
          <button class="again" id="againButton">PLAY AGAIN</button>
        </div>
      </section>
    </div>

    <div class="toolbar">
      <button class="primary" id="checkButton">CHECK POSITION</button>
      <button id="hintButton">HINT</button>
      <button id="resetButton">RESET</button>
    </div>

    <div class="readout">
      <span id="pieceCount">0 / 4 PLACED</span>
      <div class="progress"><i id="progressBar"></i></div>
      <span id="timer">00:00</span>
    </div>
  </main>

  <script>
    (() => {{
      const logicalSize = {CANVAS_SIZE};
      const threshold = {threshold};
      const pieceData = {pieces_json};
      const stage = document.getElementById("stage");
      const countLabel = document.getElementById("pieceCount");
      const statusText = document.getElementById("statusText");
      const progressBar = document.getElementById("progressBar");
      const timerLabel = document.getElementById("timer");
      const complete = document.getElementById("complete");
      let placed = 0;
      let startedAt = performance.now();
      let timerHandle = null;
      let hintTimeout = null;
      let dragPenalty = 0;

      const toPercent = value => `${{(value / logicalSize) * 100}}%`;

      pieceData.forEach(data => {{
        const target = document.createElement("div");
        target.className = "target";
        target.style.left = toPercent(data.targetX);
        target.style.top = toPercent(data.targetY);
        target.style.width = toPercent(data.w);
        target.style.height = toPercent(data.h);
        target.setAttribute("aria-hidden", "true");
        stage.appendChild(target);

        const piece = document.createElement("div");
        piece.className = "piece";
        piece.dataset.key = data.key;
        piece.dataset.placed = "false";
        piece.dataset.x = data.startX;
        piece.dataset.y = data.startY;
        piece.dataset.rotation = data.rotation;
        piece.style.left = toPercent(data.startX);
        piece.style.top = toPercent(data.startY);
        piece.style.width = toPercent(data.w);
        piece.style.height = toPercent(data.h);
        piece.style.transform = `rotate(${{data.rotation}}deg)`;
        piece.setAttribute("role", "img");
        piece.setAttribute("aria-label", `드래그할 ${{data.label}} 조각`);

        const image = document.createElement("img");
        image.src = data.src;
        image.alt = data.label;
        piece.appendChild(image);
        stage.appendChild(piece);

        attachDrag(piece, data);
      }});

      function logicalPoint(event) {{
        const rect = stage.getBoundingClientRect();
        return {{
          x: (event.clientX - rect.left) * logicalSize / rect.width,
          y: (event.clientY - rect.top) * logicalSize / rect.height
        }};
      }}

      function setPiecePosition(piece, x, y) {{
        const data = pieceData.find(item => item.key === piece.dataset.key);
        x = Math.max(-data.w * .22, Math.min(logicalSize - data.w * .78, x));
        y = Math.max(-data.h * .20, Math.min(logicalSize - data.h * .78, y));
        piece.dataset.x = x;
        piece.dataset.y = y;
        piece.style.left = toPercent(x);
        piece.style.top = toPercent(y);
      }}

      function attachDrag(piece, data) {{
        let offsetX = 0;
        let offsetY = 0;

        piece.addEventListener("pointerdown", event => {{
          if (piece.dataset.placed === "true") return;
          event.preventDefault();
          const point = logicalPoint(event);
          offsetX = point.x - Number(piece.dataset.x);
          offsetY = point.y - Number(piece.dataset.y);
          piece.classList.add("dragging");
          piece.setPointerCapture(event.pointerId);
          statusText.textContent = data.label.toUpperCase();
        }});

        piece.addEventListener("pointermove", event => {{
          if (!piece.hasPointerCapture(event.pointerId)) return;
          const point = logicalPoint(event);
          setPiecePosition(piece, point.x - offsetX, point.y - offsetY);
        }});

        piece.addEventListener("pointerup", event => {{
          if (!piece.hasPointerCapture(event.pointerId)) return;
          piece.releasePointerCapture(event.pointerId);
          piece.classList.remove("dragging");
          tryPlace(piece, data, true);
        }});

        piece.addEventListener("pointercancel", event => {{
          piece.classList.remove("dragging");
        }});
      }}

      function distanceToTarget(piece, data) {{
        const currentX = Number(piece.dataset.x) + data.w / 2;
        const currentY = Number(piece.dataset.y) + data.h / 2;
        const targetX = data.targetX + data.w / 2;
        const targetY = data.targetY + data.h / 2;
        return Math.hypot(currentX - targetX, currentY - targetY);
      }}

      function tryPlace(piece, data, penalize) {{
        const distance = distanceToTarget(piece, data);
        if (distance <= threshold) {{
          setPiecePosition(piece, data.targetX, data.targetY);
          piece.style.transform = "rotate(0deg)";
          piece.dataset.placed = "true";
          piece.classList.add("placed");
          placed += 1;
          updateProgress();
          statusText.textContent = "NICE DROP";
          if (placed === pieceData.length) finishGame();
          return true;
        }}
        if (penalize) dragPenalty += Math.min(40, Math.round(distance / 12));
        statusText.textContent = distance < threshold * 2.2 ? "SO CLOSE" : "KEEP MOVING";
        return false;
      }}

      function updateProgress() {{
        countLabel.textContent = `${{placed}} / ${{pieceData.length}} PLACED`;
        progressBar.style.width = `${{(placed / pieceData.length) * 100}}%`;
      }}

      function elapsedSeconds() {{
        return (performance.now() - startedAt) / 1000;
      }}

      function formatTime(seconds) {{
        const mins = Math.floor(seconds / 60).toString().padStart(2, "0");
        const secs = Math.floor(seconds % 60).toString().padStart(2, "0");
        return `${{mins}}:${{secs}}`;
      }}

      function startTimer() {{
        clearInterval(timerHandle);
        startedAt = performance.now();
        timerLabel.textContent = "00:00";
        timerHandle = setInterval(() => {{
          timerLabel.textContent = formatTime(elapsedSeconds());
        }}, 250);
      }}

      function meanError() {{
        let sum = 0;
        pieceData.forEach(data => {{
          const piece = stage.querySelector(`.piece[data-key="${{data.key}}"]`);
          sum += distanceToTarget(piece, data);
        }});
        return sum / pieceData.length;
      }}

      function finishGame() {{
        clearInterval(timerHandle);
        const seconds = elapsedSeconds();
        const score = Math.max(100, Math.round(1000 - seconds * 5 - dragPenalty));
        document.getElementById("finalScore").textContent = score;
        document.getElementById("finalMeta").textContent = `정확도 100% · ${{seconds.toFixed(1)}}초`;
        complete.classList.add("visible");
        statusText.textContent = "FACE RESTORED";
      }}

      function resetGame() {{
        placed = 0;
        dragPenalty = 0;
        complete.classList.remove("visible");
        stage.classList.remove("show-hints");
        pieceData.forEach(data => {{
          const piece = stage.querySelector(`.piece[data-key="${{data.key}}"]`);
          piece.dataset.placed = "false";
          piece.classList.remove("placed");
          piece.style.transform = `rotate(${{data.rotation}}deg)`;
          setPiecePosition(piece, data.startX, data.startY);
        }});
        statusText.textContent = "READY TO DROP";
        updateProgress();
        startTimer();
      }}

      document.getElementById("hintButton").addEventListener("click", () => {{
        stage.classList.add("show-hints");
        statusText.textContent = "TARGETS ON";
        clearTimeout(hintTimeout);
        hintTimeout = setTimeout(() => {{
          stage.classList.remove("show-hints");
          statusText.textContent = "KEEP MOVING";
        }}, 1800);
      }});

      document.getElementById("checkButton").addEventListener("click", () => {{
        let justPlaced = 0;
        pieceData.forEach(data => {{
          const piece = stage.querySelector(`.piece[data-key="${{data.key}}"]`);
          if (piece.dataset.placed !== "true" && tryPlace(piece, data, false)) justPlaced++;
        }});
        if (!justPlaced && placed < pieceData.length) {{
          const error = meanError();
          const accuracy = Math.max(0, Math.round(100 - (error / 380) * 100));
          statusText.textContent = `현재 정확도 ${{accuracy}}%`;
        }}
      }});

      document.getElementById("resetButton").addEventListener("click", resetGame);
      document.getElementById("againButton").addEventListener("click", resetGame);
      startTimer();
    }})();
  </script>
</body>
</html>
"""
