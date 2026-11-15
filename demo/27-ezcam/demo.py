"""27｜EZCam：requests.get 拿到 JPEG bytes → np.frombuffer → cv2.imdecode → 陣列；Mock 模式；被叫才拍、不是串流。"""
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / "story" / "data" / "factory" / "roi" / "frames"
SHOT_SECONDS = 0.3        # 真的 EZCam 約 3 秒一張；Demo 縮成 0.3 秒，跑起來比較快


class FakeEZCam(BaseHTTPRequestHandler):
    """本機的假 EZCam：GET /capture 時「拍」一張（等一下再回 JPEG），/broken 回傳只傳一半的 JPEG，/busy 回 503。"""
    shots = sorted(FRAMES.glob("*.jpg"))
    count = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/busy":
            self.send_response(503)
            self.end_headers()
            return
        time.sleep(SHOT_SECONDS)                         # 被叫了才拍，要等
        data = self.shots[FakeEZCam.count % len(self.shots)].read_bytes()
        FakeEZCam.count += 1
        if self.path == "/broken":
            data = data[: len(data) // 2]                 # Wi-Fi 斷一下，只收到一半
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


server = ThreadingHTTPServer(("127.0.0.1", 0), FakeEZCam)
threading.Thread(target=server.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{server.server_port}"

print("== 1. requests.get：拿到的是 JPEG 的 bytes ==")
r = requests.get(f"{URL}/capture", timeout=10)
print(f"status {r.status_code}、Content-Type {r.headers['Content-Type']}")
print(f"r.content：{type(r.content).__name__}，{len(r.content):,} bytes，開頭 {r.content[:4]}")

print("\n== 2. np.frombuffer → cv2.imdecode：bytes 變成陣列 ==")
buf = np.frombuffer(r.content, np.uint8)
print(f"np.frombuffer → shape {buf.shape}、dtype {buf.dtype}（還是一長串壓縮過的 bytes）")
img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
print(f"cv2.imdecode → shape {img.shape}、dtype {img.dtype}（BGR，和 cv2.imread 一樣）")
same = cv2.imread(str(FRAMES / "frame_000.jpg"))
print(f"和直接 cv2.imread 同一個檔案一樣嗎？ {np.array_equal(img, same)}")
print(f"解開後 {img.size:,} 個數字，是 JPEG 的 {img.size / len(r.content):.1f} 倍")

print("\n== 3. 壞掉的回應：imdecode 回傳 None、raise_for_status ==")
r = requests.get(f"{URL}/broken", timeout=10)
print(f"只收到一半：status {r.status_code}，{len(r.content):,} bytes → imdecode：{cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)}")
r = requests.get(f"{URL}/busy", timeout=10)
print(f"攝影機忙線：status {r.status_code}，r.content {r.content}")
try:
    r.raise_for_status()
except requests.exceptions.HTTPError as e:
    print(f"raise_for_status() → HTTPError: {e.response.status_code}")

print("\n== 4. Mock 模式：沒設 EZCAM_URL 就讀本機檔案 ==")


def capture(url):
    if url:
        r = requests.get(f"{url}/capture", timeout=10)
        r.raise_for_status()
        data = r.content
    else:
        data = (FRAMES / "frame_001.jpg").read_bytes()   # 同樣是 JPEG bytes，只是從硬碟來
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"收到 {len(data)} bytes，不是完整的圖")
    return img


print(f"os.getenv('EZCAM_URL') = {os.getenv('EZCAM_URL')}")
print(f"Mock 模式 capture(None)  → {capture(None).shape}")
print(f"假 EZCam capture(URL)    → {capture(URL).shape}")

print(f"\n== 5. 被叫才拍：一張要等 {SHOT_SECONDS} 秒（真的 EZCam 約 3 秒）==")
t0 = time.perf_counter()
for _ in range(3):
    capture(URL)
elapsed = time.perf_counter() - t0
print(f"連拍 3 張，至少要 {3 * SHOT_SECONDS:.1f} 秒：{elapsed >= 3 * SHOT_SECONDS}")
server.shutdown()

print("\n== 6. 約 3 秒一張 vs 每秒 10 格：跌倒的那一刻拍得到嗎？ ==")
poses = pd.read_csv(ROOT / "story" / "data" / "care" / "pose" / "poses.csv",
                    usecols=["seq_id", "frame", "posture", "is_fall_seq", "fall_frame"])
falls = poses[poses["is_fall_seq"] == 1]
lying = falls.groupby("seq_id")["posture"].last().eq("lying")
seq = falls[falls["seq_id"] == lying[lying].index[0]]               # 第一段「倒完躺在地上」的
print(f"{seq['seq_id'].iloc[0]}，開始倒在第 {seq['fall_frame'].iloc[0]} 格")
NAME = {"standing": "站", "sitting": "坐", "lying": "躺"}
print("10 格／秒：", "".join(NAME[p] for p in seq["posture"]))
print("3 秒一張：", "".join(NAME[p] for p in seq["posture"].iloc[::30]),
      "（第 0、3 秒）")
caught = total = 0
for _, s in falls.groupby("seq_id"):
    start = int(s["fall_frame"].iloc[0])
    end = int(s.loc[s["posture"] != "standing", "frame"].min())   # 開始倒 → 姿勢變了
    for offset in range(30):                                        # 第一張剛好在第幾格拍
        shots = range(offset, 40, 30)
        caught += any(start <= f < end for f in shots)
        total += 1
print(f"{falls['seq_id'].nunique()} 段跌倒 × 30 種起拍時間 = {total} 種情形；拍到「正在倒」的 {caught / total:.3f}")
