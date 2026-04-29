#!/usr/bin/env python3
"""RK3588 RTSP/RTMP 推流脚本 - 三路硬件编码循环推流本地视频"""

import subprocess
import signal
import sys
import os

# ==================== 配置 ====================
# 协议从 url 前缀自动识别（rtsp:// 或 rtmp://）
STREAMS = [
    {
        "name": "4K",
        "video": "./data/test4k.mp4",
        "url": "rtsp://127.0.0.1:8554/live_4k",
        "bitrate": "8M",
        "fps": 30,
    },
    {
        "name": "1080P",
        "video": "./data/test1080p.mp4",
        "url": "rtmp://127.0.0.1:1935/live/1080p",
        "bitrate": "4M",
        "fps": 30,
    },
    {
        "name": "640P",
        "video": "./data/test640p.mp4",
        "url": "rtsp://127.0.0.1:8554/live_640p",
        "bitrate": "2M",
        "fps": 30,
    },
]

CODEC = "h264"          # h264 或 hevc
# ===============================================

ENCODER = f"{CODEC}_rkmpp"


def build_cmd(stream):
    url = stream["url"]
    fps = stream.get("fps", 30)
    proto = "rtmp" if url.startswith("rtmp://") else "rtsp"

    cmd = [
        "ffmpeg",
        "-stream_loop", "-1",
        "-re",
        "-hwaccel", "rkmpp",
        "-hwaccel_output_format", "drm_prime",
        "-i", stream["video"],
        "-c:v", ENCODER,
        "-rc_mode", "CBR",
        "-b:v", stream["bitrate"],
        "-g", str(fps * 2),
        "-bf", "0",
        "-r", str(fps),
        "-c:a", "aac",
        "-b:a", "128k",
        "-flags", "+low_delay",
        "-fflags", "+genpts",
    ]

    if proto == "rtsp":
        cmd += ["-rtsp_transport", "tcp", "-f", "rtsp"]
    else:
        cmd += ["-f", "flv"]

    cmd.append(url)
    return cmd


processes = []


def cleanup(signum, frame):
    for name, proc in processes:
        proc.terminate()
    for name, proc in processes:
        proc.wait()
    print("\n所有推流已停止")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    for s in STREAMS:
        if not os.path.isfile(s["video"]):
            print(f"错误: 视频文件不存在 -> {s['video']}")
            sys.exit(1)

    for s in STREAMS:
        cmd = build_cmd(s)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        processes.append((s["name"], proc))
        proto = "RTMP" if s["url"].startswith("rtmp://") else "RTSP"
        print(f"[{s['name']}] {proto} 推流开始: {s['video']} -> {s['url']} ({s['bitrate']}, {s.get('fps',30)}fps)")

    print(f"\n编码器: {ENCODER} | 共 {len(STREAMS)} 路 | 按 Ctrl+C 停止\n")

    # 轮询读取各进程输出
    import selectors
    sel = selectors.DefaultSelector()
    for name, proc in processes:
        sel.register(proc.stdout, selectors.EVENT_READ, name)

    active = len(processes)
    while active > 0:
        for key, _ in sel.select(timeout=1):
            line = key.fileobj.readline()
            if line:
                sys.stdout.buffer.write(f"[{key.data}] ".encode() + line)
                sys.stdout.buffer.flush()
            else:
                sel.unregister(key.fileobj)
                active -= 1

    # 检查退出码
    for name, proc in processes:
        ret = proc.wait()
        if ret != 0:
            print(f"[{name}] ffmpeg 异常退出, 返回码: {ret}")


if __name__ == "__main__":
    main()
