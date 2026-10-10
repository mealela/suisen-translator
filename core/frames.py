"""JPEG 编解码小工具: 定位阶段快照先以 JPEG 字节留在内存, 轨迹确认后再落盘"""
import cv2
import numpy as np


def encode_jpg(frame, quality=85):
    ok, buf = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


def decode_jpg(data):
    """接受 文件路径(str) 或 JPEG 字节, 返回 BGR ndarray / None"""
    if isinstance(data, str):
        from core.v6_common import load_image
        return load_image(data)
    if not data:
        return None
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
