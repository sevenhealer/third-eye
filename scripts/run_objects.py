#!/usr/bin/env python3
"""
Live object detection + tracking + zone events (Sprint 3).

YOLOv9-C (80 COCO classes) → ByteTracker → ZoneEventDetector
(OBJECT_ADDED / OBJECT_REMOVED) → optional persistence (system_events +
object_counts) with --persist-events.

Usage:
  python scripts/run_objects.py --source rtsp://... --show --fps 15
  python scripts/run_objects.py --source rtsp://... --persist-events \
      --camera-id cam0 --zone-id bedroom
"""
import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="third-eye live object feed")
    p.add_argument("--source", default="0", help="Camera index or RTSP URL")
    p.add_argument("--camera-id", default="cam0")
    p.add_argument("--zone-id", default="bedroom")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--model", default="yolov9c.pt",
                   help="Ultralytics weights (auto-downloads on first use)")
    p.add_argument("--imgsz", type=int, default=960,
                   help="Inference size (default 960; use 640 on slow PCIe links)")
    p.add_argument("--conf", type=float, default=0.45,
                   help="Detection confidence threshold")
    p.add_argument("--show", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gpu", type=int, default=None,
                   help="CUDA device index (sets CUDA_VISIBLE_DEVICES)")
    p.add_argument("--persist-events", action="store_true",
                   help="Write OBJECT_ADDED/REMOVED to system_events and "
                        "bucketed counts to object_counts (needs postgres)")
    return p.parse_args()


def _set_minimal_env() -> None:
    if (ROOT / ".env").exists():
        return
    defaults = {
        "APP_SECRET_KEY": "devkeydevkeydevkeydevkeydevkeydev",
        "JWT_SECRET_KEY": "devjwtdevjwtdevjwtdevjwtdevjwtdevjwtdevjwtdevjwt",
        "DATABASE_URL": "postgresql+asyncpg://u:p@localhost/db",
        "TIMESCALE_URL": "postgresql+asyncpg://u:p@localhost/ts",
        "NEO4J_PASSWORD": "secret",
        "POSTGRES_PASSWORD": "secret",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


async def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    _set_minimal_env()

    from src.core.gpu_manager import preload_cuda_libraries
    preload_cuda_libraries()

    try:
        import cv2
    except ImportError:
        print("ERROR: opencv not installed")
        sys.exit(1)

    from src.ingestion.camera import CameraReader
    from src.ingestion.frame_producer import FrameProducer
    from src.object_counting.counter import ObjectCountAggregator
    from src.object_detection.detector import ModelNotLoadedError, YOLODetector
    from src.pipeline.zone_events import ZoneEventDetector
    from src.tracking.tracker import ByteTracker

    device = "cpu"
    if not args.cpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda:0"
        except ImportError:
            pass

    detector = YOLODetector(model_name=args.model, conf_threshold=args.conf,
                            imgsz=args.imgsz, device=device)
    try:
        detector.load()
    except ModelNotLoadedError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    tracker = ByteTracker(max_age=90, min_hits=3, iou_threshold=0.3,
                          high_threshold=args.conf, low_threshold=0.1)
    tracker.reset()
    zone_detector = ZoneEventDetector(removal_grace_frames=int(args.fps * 5) or 75)
    counter = ObjectCountAggregator(args.camera_id, args.zone_id, bucket_seconds=10)

    event_store = None
    if args.persist_events:
        from src.event_detection.event_store import EventStore
        event_store = EventStore()
        print("Persistence: ON (system_events + object_counts)")

    try:
        source: str | int = int(args.source)
    except ValueError:
        source = args.source

    camera = CameraReader(source=source)
    if not camera.open():
        print(f"ERROR: Cannot open camera '{source}'")
        sys.exit(1)

    print("=" * 62)
    print(f"  model           : {args.model} @ {args.imgsz}, conf {args.conf}")
    print(f"  device          : {device}")
    print(f"  camera/zone     : {args.camera_id} / {args.zone_id}")
    print(f"  persist events  : {'yes' if event_store else 'no'}")
    print(f"  display window  : {'yes (q quits)' if args.show else 'no'}")
    print("=" * 62 + "\n")

    win = "third-eye | objects (q to quit)"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_count = 0
    last_infer_ms = 0.0
    fps_ema = 0.0
    last_frame_t = 0.0
    last_flush = time.time()

    producer = FrameProducer(camera=camera, camera_id=args.camera_id,
                             zone_id=args.zone_id, max_fps=args.fps,
                             drop_stale=True)

    async def process_frame(frame, meta) -> None:
        nonlocal frame_count, last_infer_ms, fps_ema, last_frame_t, last_flush
        frame_count += 1

        now = time.monotonic()
        if last_frame_t:
            gap = now - last_frame_t
            if gap > 0.3:
                cause = ("processing (inference slow)"
                         if last_infer_ms / 1000.0 > gap * 0.7
                         else "camera/stream (no frames arrived)")
                print(f"  ! feed stall {gap * 1000:.0f} ms — {cause}")
            inst = 1.0 / max(gap, 1e-6)
            fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
        last_frame_t = now

        t0 = time.monotonic()
        detections = detector.detect(frame)
        last_infer_ms = (time.monotonic() - t0) * 1000.0

        tracked = tracker.update(detections)

        # zone events (ADDED once per appearance, REMOVED after grace)
        events = zone_detector.update(args.camera_id, args.zone_id, tracked)
        for ev in events:
            print(f"  >> {ev.event_type}  {ev.class_name}  track={ev.track_id}")
            if event_store is not None:
                await event_store.write_event(
                    event_type=ev.event_type,
                    event_time_ns=ev.timestamp_ns,
                    camera_id=ev.camera_id,
                    zone_id=ev.zone_id,
                    track_id=ev.track_id,
                    payload={"class_name": ev.class_name},
                )

        # bucketed zone counts
        class_counts: dict[str, int] = {}
        for t in tracked:
            class_counts[t.class_name] = class_counts.get(t.class_name, 0) + 1
        counter.observe(class_counts, time.time())
        if event_store is not None and time.time() - last_flush >= 5.0:
            last_flush = time.time()
            buckets = counter.pop_closed(time.time())
            if buckets:
                await event_store.write_object_counts(
                    [(b.bucket_time, b.camera_id, b.zone_id,
                      b.object_class, b.count) for b in buckets]
                )

        if frame_count % 30 == 0:
            summary = ", ".join(f"{c}:{n}" for c, n in sorted(class_counts.items())) or "—"
            print(f"  [{frame_count:>6}] fps {fps_ema:4.1f}  infer {last_infer_ms:3.0f} ms"
                  f"  tracks {len(tracked)}  ({summary})")

        if args.show:
            for t in tracked:
                b = t.bbox.astype(int)
                color = (0, 220, 0) if t.class_name == "person" else (255, 180, 0)
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
                cv2.putText(frame, f"{t.class_name} ID:{t.track_id} {t.confidence:.2f}",
                            (b[0], max(b[1] - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            hud = (f"fps {fps_ema:4.1f}  infer {last_infer_ms:3.0f}ms  "
                   f"trk {len(tracked)}  frame {frame_count}")
            cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (240, 240, 240), 1)
            h, w = frame.shape[:2]
            scale = min(1600 / w, 900 / h, 1.0)
            disp = (cv2.resize(frame, (int(w * scale), int(h * scale)))
                    if scale < 1.0 else frame)
            cv2.imshow(win, disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                producer.stop()

    try:
        await producer.run(process_frame)
    except KeyboardInterrupt:
        pass
    finally:
        if event_store is not None:
            buckets = counter.pop_closed(time.time() + 3600)   # flush all
            if buckets:
                await event_store.write_object_counts(
                    [(b.bucket_time, b.camera_id, b.zone_id,
                      b.object_class, b.count) for b in buckets]
                )
        camera.release()
        if args.show:
            cv2.destroyAllWindows()
        print(f"\nStopped — {frame_count} frames.")


if __name__ == "__main__":
    asyncio.run(main())
