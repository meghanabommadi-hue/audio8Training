"""Load-test the streaming server with N concurrent callers.

For every request it records chunk arrival times and simulates playback that
starts with the first chunk plus a jitter buffer; a stall is any time the
player runs out of audio (what a caller would hear as a gap).

    python bench_server.py --url http://127.0.0.1:8030 --concurrency 1 2 4 6 8 --requests 3
"""

import argparse
import http.client
import json
import threading
import time
from urllib.parse import urlparse

import numpy as np

SENTENCES = [
    "Hello, thank you for calling Kapture support. How can I help you today?",
    "Your EMI of twelve thousand rupees is due on the fifth of every month. Please pay before the due date to avoid late charges.",
    "I'm sorry for the inconvenience. Let me check that for you right away.",
    "Please keep your PAN card and Aadhaar card ready for the KYC verification.",
    "I have raised a ticket for you, and our team will call you back within twenty four hours.",
    "You can pay using UPI, net banking, or a debit card through our mobile app.",
    "Your order has been dispatched and will reach Hyderabad by Thursday evening.",
    "I completely understand your concern, ma'am. Let me transfer you to the right team.",
]
BYTES_PER_SECOND = 44100 * 2


def one_request(url, text, buffer_s, seed):
    parsed = urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=300)
    body = json.dumps({"text": text, "format": "pcm", "seed": seed})
    start = time.time()
    conn.request("POST", "/v1/tts", body=body, headers={"content-type": "application/json"})
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:200]!r}")
    arrivals = []
    while True:
        data = response.read1(65536)
        if not data:
            break
        arrivals.append((time.time() - start, len(data)))
    conn.close()
    total = time.time() - start
    first = arrivals[0][0]
    # Player: starts at first byte + buffer, pauses whenever it runs dry.
    play_until = first + buffer_s
    stall = 0.0
    for t, size in arrivals:
        if t > play_until:
            stall += t - play_until
            play_until = t
        play_until += size / BYTES_PER_SECOND
    audio = sum(size for _, size in arrivals) / BYTES_PER_SECOND
    return {"first_audio": first, "total": total, "audio": audio, "rtf": total / audio, "stall": stall}


def run_level(url, concurrency, requests, buffer_s):
    results, errors = [], []
    lock = threading.Lock()

    def client(cid):
        for r in range(requests):
            text = SENTENCES[(cid * requests + r) % len(SENTENCES)]
            try:
                res = one_request(url, text, buffer_s, seed=cid * 100 + r)
                with lock:
                    results.append(res)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(str(exc))

    threads = [threading.Thread(target=client, args=(c,)) for c in range(concurrency)]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - start
    if not results:
        return {"concurrency": concurrency, "errors": errors}
    arr = lambda k: np.array([r[k] for r in results])  # noqa: E731
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": len(errors),
        "first_audio_p50": round(float(np.percentile(arr("first_audio"), 50)), 2),
        "first_audio_p95": round(float(np.percentile(arr("first_audio"), 95)), 2),
        "rtf_p50": round(float(np.percentile(arr("rtf"), 50)), 2),
        "rtf_max": round(float(arr("rtf").max()), 2),
        "streams_with_stall": f"{int((arr('stall') > 0.1).sum())}/{len(results)}",
        "stall_s_p95": round(float(np.percentile(arr("stall"), 95)), 2),
        "audio_s_per_wall_s": round(float(arr("audio").sum() / wall), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8030")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    ap.add_argument("--requests", type=int, default=3, help="sequential requests per client")
    ap.add_argument("--buffer", type=float, default=0.3, help="client jitter buffer (s)")
    args = ap.parse_args()
    for level in args.concurrency:
        print(json.dumps(run_level(args.url, level, args.requests, args.buffer)), flush=True)


if __name__ == "__main__":
    main()
