#!/usr/bin/env python3
"""
Review & correct auto-labels before fine-tuning (Sprint 3, step 3c).

The human-in-the-loop step between capture_dataset.py and train_detector.py:
step through captured frames, see YOLO-World's proposed boxes, fix them fast.
A class sidebar is always visible and a single number key assigns that class
to the selected box (no cycling). Saves corrected YOLO labels in place.
Local-first; for heavy corner-dragging at scale, Label Studio / CVAT import
the same dataset.

Usage:
  python scripts/review_labels.py --dataset datasets/room
  python scripts/review_labels.py --dataset datasets/room --split train

Controls (also drawn in the sidebar):
  click box        select it          drag empty area   draw a new box
  1..9, 0          set class of selected box (+ active class for new boxes)
  click class row  same as its number
  [ / ]            page the number->class window (datasets with >10 classes)
  d / x            delete selected    c / C   cycle class +/- (fallback)
  n / SPACE        save + next        p       save + previous
  t                toggle carry-forward (propagate prev frame's labels)
  v                copy previous frame's labels into this one (one-shot)
  a                reset this frame to the model's original auto-labels
  u                revert this image  q / ESC save + quit

Carry-forward: with a near-static camera, static objects barely move between
frames, so this frame starts from your CORRECTED previous frame instead of
the model's noisy per-frame guess — you only fix what changed. A frame is
treated as "already reviewed" once its labels differ from the original
auto-labels, so going back never clobbers your work.
An empty label file (all boxes deleted) is kept — a frame with no targets is
a valid negative sample.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

PANEL_W = 340
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX


def load_class_names(data_yaml: Path) -> list[str]:
    names = yaml.safe_load(data_yaml.read_text()).get("names", {})
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def read_labels(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return rows


def write_labels(path: Path, rows: list[list[float]]) -> None:
    path.write_text(
        "\n".join(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                  for c, cx, cy, w, h in rows)
    )


def color(idx: int) -> tuple[int, int, int]:
    palette = [(0, 220, 0), (255, 170, 0), (0, 170, 255), (255, 60, 200),
               (40, 255, 255), (180, 220, 0), (170, 90, 255), (0, 120, 255),
               (90, 255, 140), (255, 110, 90)]
    return palette[idx % len(palette)]


def main() -> None:
    p = argparse.ArgumentParser(description="third-eye label reviewer")
    p.add_argument("--dataset", required=True,
                   help="Dataset dir (has images/, labels/, data.yaml)")
    p.add_argument("--split", default="train")
    p.add_argument("--max-height", type=int, default=900, help="Display height cap")
    args = p.parse_args()

    try:
        import cv2
    except ImportError:
        print("ERROR: opencv not installed")
        sys.exit(1)

    def put(img, text, org, col, scale=0.5, bg=True):
        x, y = org
        if bg:
            (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
            cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 4, y + 3), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), FONT, scale, col, 1, cv2.LINE_AA)

    root = Path(args.dataset)
    img_dir = root / "images" / args.split
    lbl_dir = root / "labels" / args.split
    auto_dir = root / "labels_auto" / args.split   # model's original proposal
    lbl_dir.mkdir(parents=True, exist_ok=True)
    classes = load_class_names(root / "data.yaml")
    if not classes:
        print("ERROR: no class names in data.yaml")
        sys.exit(1)
    images = sorted([q for q in img_dir.iterdir()
                     if q.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not images:
        print(f"ERROR: no images in {img_dir}")
        sys.exit(1)

    st = {
        "rows": [], "sel": -1, "active": 0, "win": 0,
        "drag": None, "iw": 1, "ih": 1, "scale": 1.0, "dh": 1,
        "rowmap": [],     # [(y_top, y_bot, class_idx)] clickable class rows
        "carry": None,    # previous frame's corrected labels (for propagation)
        "propagate": True,
    }

    def clone(rows):
        return [list(r) for r in rows]

    def to_norm(x, y):
        return x / st["scale"] / st["iw"], y / st["scale"] / st["ih"]

    def assign(cls_idx: int) -> None:
        if not (0 <= cls_idx < len(classes)):
            return
        st["active"] = cls_idx
        if st["sel"] >= 0:
            st["rows"][st["sel"]][0] = cls_idx

    def on_mouse(event, x, y, flags, _param):
        img_w = int(st["iw"] * st["scale"])
        if x >= img_w:   # ---- sidebar click: pick a class row ----
            if event == cv2.EVENT_LBUTTONDOWN:
                for y0, y1, ci in st["rowmap"]:
                    if y0 <= y <= y1:
                        assign(ci)
                        break
            return
        # ---- image area: select / draw ----
        if event == cv2.EVENT_LBUTTONDOWN:
            st["drag"] = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and st["drag"] is not None:
            st["drag"] = (st["drag"][0], st["drag"][1], x, y)
        elif event == cv2.EVENT_LBUTTONUP and st["drag"] is not None:
            x0, y0, x1, y1 = st["drag"]
            st["drag"] = None
            if abs(x1 - x0) < 6 and abs(y1 - y0) < 6:
                nx, ny = to_norm(x, y)
                best, area = -1, 1e9
                for i, (_, cx, cy, bw, bh) in enumerate(st["rows"]):
                    if abs(nx - cx) <= bw / 2 and abs(ny - cy) <= bh / 2 and bw * bh < area:
                        best, area = i, bw * bh
                st["sel"] = best
            else:
                ax0, ay0 = to_norm(min(x0, x1), min(y0, y1))
                ax1, ay1 = to_norm(max(x0, x1), max(y0, y1))
                st["rows"].append([st["active"], (ax0 + ax1) / 2, (ay0 + ay1) / 2,
                                   ax1 - ax0, ay1 - ay0])
                st["sel"] = len(st["rows"]) - 1

    win = "third-eye | review labels"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    idx = 0
    while 0 <= idx < len(images):
        img_path = images[idx]
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        frame = cv2.imread(str(img_path))
        if frame is None:
            idx += 1
            continue
        ih, iw = frame.shape[:2]
        scale = min(args.max_height / ih, 1.0)

        # carry-forward: if this frame is still untouched (its labels equal the
        # model's original auto-labels) and propagation is on, start from the
        # previous corrected frame instead of the noisy per-frame guess
        auto_path = auto_dir / f"{img_path.stem}.txt"
        disk_rows = read_labels(lbl_path)
        auto_rows = read_labels(auto_path) if auto_path.exists() else None
        unreviewed = auto_rows is not None and disk_rows == auto_rows
        if st["propagate"] and st["carry"] is not None and unreviewed:
            rows = clone(st["carry"])
        else:
            rows = disk_rows
        st.update(rows=rows, sel=-1, iw=iw, ih=ih, scale=scale, dh=int(ih * scale))

        while True:
            img = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame.copy()
            dh, dw = img.shape[:2]
            canvas = np.full((dh, dw + PANEL_W, 3), 32, np.uint8)
            canvas[:dh, :dw] = img

            # boxes
            for i, (c, cx, cy, bw, bh) in enumerate(st["rows"]):
                x1, y1 = int((cx - bw / 2) * iw * scale), int((cy - bh / 2) * ih * scale)
                x2, y2 = int((cx + bw / 2) * iw * scale), int((cy + bh / 2) * ih * scale)
                col = color(int(c))
                if i == st["sel"]:
                    ov = canvas.copy()
                    cv2.rectangle(ov, (x1, y1), (x2, y2), col, -1)
                    cv2.addWeighted(ov, 0.25, canvas, 0.75, 0, canvas)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 3)
                else:
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 1)
                name = classes[int(c)] if int(c) < len(classes) else str(int(c))
                put(canvas, f"{i}:{name}", (x1, max(y1 - 4, 14)), col, 0.5)
            if st["drag"] is not None:
                x0, y0, x1, y1 = st["drag"]
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), 1)

            # ---- sidebar ----
            px = dw + 14
            put(canvas, f"{idx + 1}/{len(images)}  {img_path.name[:22]}",
                (px, 26), (255, 255, 255), 0.5, bg=False)
            put(canvas, f"boxes: {len(st['rows'])}   selected: "
                f"{st['sel'] if st['sel'] >= 0 else '-'}",
                (px, 50), (200, 200, 200), 0.45, bg=False)
            put(canvas, "CLASSES  (press number / click)", (px, 84),
                (255, 255, 255), 0.5, bg=False)

            st["rowmap"] = []
            win_start = st["win"]
            y = 108
            for ci in range(len(classes)):
                col = color(ci)
                hot = ""
                if win_start <= ci < win_start + 10:
                    d = ci - win_start + 1
                    hot = f"{d % 10}"   # 1..9 then 0
                is_active = ci == st["active"]
                sel_cls = st["rows"][st["sel"]][0] if st["sel"] >= 0 else -1
                if is_active:
                    cv2.rectangle(canvas, (px - 6, y - 15), (dw + PANEL_W - 6, y + 6),
                                  (70, 70, 70), -1)
                cv2.rectangle(canvas, (px, y - 12), (px + 14, y + 2), col, -1)
                tag = f"[{hot}] " if hot else "    "
                mark = " *" if ci == sel_cls else ""
                txt_col = (255, 255, 255) if is_active else (210, 210, 210)
                put(canvas, f"{tag}{classes[ci]}{mark}", (px + 22, y), txt_col, 0.5, bg=False)
                st["rowmap"].append((y - 15, y + 6, ci))
                y += 26
                if y > dh - 122:   # reserve bottom strip for the footer
                    break

            footer = dh - 116
            if len(classes) > 10:
                put(canvas, f"[ ] page  (showing {win_start + 1}-"
                    f"{min(win_start + 10, len(classes))})",
                    (px, footer), (160, 200, 255), 0.42, bg=False)
            prop_col = (90, 255, 140) if st["propagate"] else (140, 140, 140)
            put(canvas, f"[t] carry-forward: {'ON' if st['propagate'] else 'off'}",
                (px, dh - 90), prop_col, 0.45, bg=False)
            put(canvas, "v copy prev   a reset auto", (px, dh - 66),
                (170, 170, 170), 0.42, bg=False)
            put(canvas, "d del  drag add  n/p nav", (px, dh - 42),
                (170, 170, 170), 0.42, bg=False)
            put(canvas, "u revert   q save+quit", (px, dh - 18),
                (170, 170, 170), 0.42, bg=False)

            cv2.imshow(win, canvas)
            k = cv2.waitKey(20) & 0xFF
            if k == 255:
                continue
            if k in (ord("n"), ord(" ")):
                write_labels(lbl_path, st["rows"])
                st["carry"] = clone(st["rows"])
                idx += 1
                break
            if k == ord("p"):
                write_labels(lbl_path, st["rows"])
                st["carry"] = clone(st["rows"])
                idx = max(0, idx - 1)
                break
            if k in (ord("q"), 27):
                write_labels(lbl_path, st["rows"])
                cv2.destroyAllWindows()
                print(f"Saved. Reviewed up to image {idx + 1}/{len(images)}.")
                return
            if ord("0") <= k <= ord("9"):
                d = k - ord("0")
                assign(st["win"] + (9 if d == 0 else d - 1))
            elif k in (ord("d"), ord("x")) and st["sel"] >= 0:
                st["rows"].pop(st["sel"])
                st["sel"] = -1
            elif k in (ord("c"), ord("C")):
                step = 1 if k == ord("c") else -1
                if st["sel"] >= 0:
                    st["rows"][st["sel"]][0] = (int(st["rows"][st["sel"]][0]) + step) % len(classes)
                else:
                    st["active"] = (st["active"] + step) % len(classes)
            elif k == ord("]"):
                if st["win"] + 10 < len(classes):
                    st["win"] += 10
            elif k == ord("["):
                st["win"] = max(0, st["win"] - 10)
            elif k == ord("t"):
                st["propagate"] = not st["propagate"]
            elif k == ord("v") and st["carry"] is not None:
                st.update(rows=clone(st["carry"]), sel=-1)
            elif k == ord("a"):
                ap = auto_dir / f"{img_path.stem}.txt"
                if ap.exists():
                    st.update(rows=read_labels(ap), sel=-1)
            elif k == ord("u"):
                st.update(rows=read_labels(lbl_path), sel=-1)

    cv2.destroyAllWindows()
    print(f"Reviewed all {len(images)} images in {img_dir}.")


if __name__ == "__main__":
    main()
