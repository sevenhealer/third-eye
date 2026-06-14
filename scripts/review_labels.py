#!/usr/bin/env python3
"""
Review & correct auto-labels before fine-tuning (Sprint 3, step 3c).

The human-in-the-loop step between capture_dataset.py and train_detector.py:
step through captured frames, see YOLO-World's proposed boxes, and accept /
reject / rename / redraw each one. Saves corrected YOLO-format labels in
place. Keyboard-first, fully local (no cloud) — for heavy box-dragging at
scale, Label Studio / CVAT import the same dataset and edit corners with
mouse handles.

Usage:
  python scripts/review_labels.py --dataset datasets/room
  python scripts/review_labels.py --dataset datasets/room --split train

Controls (shown on screen):
  n / SPACE  save + next image       p  save + previous
  click      select a box            drag (empty area)  draw a new box
  d / x      delete selected box     c / C  cycle selected box's class +/-
  0-9        set active class (new boxes + reassigns selected)
  u          revert this image to what's on disk
  q / ESC    save + quit
An empty label file (all boxes deleted) is kept on purpose — a frame with no
target objects is a valid negative sample.
"""
import argparse
import sys
from pathlib import Path

import yaml


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


def _color(idx: int) -> tuple[int, int, int]:
    palette = [(0, 220, 0), (255, 180, 0), (0, 180, 255), (255, 0, 200),
               (0, 255, 255), (200, 200, 0), (180, 0, 255), (0, 120, 255)]
    return palette[idx % len(palette)]


def main() -> None:
    p = argparse.ArgumentParser(description="third-eye label reviewer")
    p.add_argument("--dataset", required=True, help="Dataset dir (has images/, labels/, data.yaml)")
    p.add_argument("--split", default="train")
    p.add_argument("--max-width", type=int, default=1400, help="Display downscale cap")
    args = p.parse_args()

    try:
        import cv2
    except ImportError:
        print("ERROR: opencv not installed")
        sys.exit(1)

    root = Path(args.dataset)
    img_dir = root / "images" / args.split
    lbl_dir = root / "labels" / args.split
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

    # mutable UI state shared with the mouse callback
    state = {
        "rows": [],          # [cls, cx, cy, w, h] normalized
        "sel": -1,           # selected box index
        "active_cls": 0,     # class for new/reassigned boxes
        "drag": None,        # (x0, y0, x1, y1) in display px while dragging
        "w": 1, "h": 1, "scale": 1.0,
    }

    def to_norm(x, y):
        return x / state["scale"] / state["w"], y / state["scale"] / state["h"]

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
            x0, y0, _, _ = state["drag"]
            state["drag"] = (x0, y0, x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drag"] is not None:
            x0, y0, x1, y1 = state["drag"]
            state["drag"] = None
            if abs(x1 - x0) < 6 and abs(y1 - y0) < 6:
                # a click: select the smallest box containing the point
                nx, ny = to_norm(x, y)
                best, best_area = -1, 1e9
                for i, (_, cx, cy, bw, bh) in enumerate(state["rows"]):
                    if abs(nx - cx) <= bw / 2 and abs(ny - cy) <= bh / 2 and bw * bh < best_area:
                        best, best_area = i, bw * bh
                state["sel"] = best
            else:
                # a drag: add a new box with the active class
                ax0, ay0 = to_norm(min(x0, x1), min(y0, y1))
                ax1, ay1 = to_norm(max(x0, x1), max(y0, y1))
                state["rows"].append([state["active_cls"],
                                      (ax0 + ax1) / 2, (ay0 + ay1) / 2,
                                      ax1 - ax0, ay1 - ay0])
                state["sel"] = len(state["rows"]) - 1

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
        h, w = frame.shape[:2]
        state.update(rows=read_labels(lbl_path), sel=-1, w=w, h=h,
                     scale=min(args.max_width / w, 1.0))

        while True:
            disp = cv2.resize(frame, None, fx=state["scale"], fy=state["scale"]) \
                if state["scale"] < 1.0 else frame.copy()
            sc = state["scale"]
            for i, (c, cx, cy, bw, bh) in enumerate(state["rows"]):
                x1, y1 = int((cx - bw / 2) * w * sc), int((cy - bh / 2) * h * sc)
                x2, y2 = int((cx + bw / 2) * w * sc), int((cy + bh / 2) * h * sc)
                col = _color(int(c))
                thick = 3 if i == state["sel"] else 1
                cv2.rectangle(disp, (x1, y1), (x2, y2), col, thick)
                name = classes[int(c)] if int(c) < len(classes) else str(int(c))
                tag = f"{i}:{name}" + ("  <sel>" if i == state["sel"] else "")
                cv2.putText(disp, tag, (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, thick)
            if state["drag"] is not None:
                x0, y0, x1, y1 = state["drag"]
                cv2.rectangle(disp, (x0, y0), (x1, y1), (255, 255, 255), 1)

            active = classes[state["active_cls"]]
            hud1 = f"{idx + 1}/{len(images)}  {img_path.name}  boxes:{len(state['rows'])}"
            hud2 = f"active class [{state['active_cls']}] {active}   n/p d c 0-9 u q"
            for j, txt in enumerate((hud1, hud2)):
                yy = 22 + j * 24
                cv2.putText(disp, txt, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(disp, txt, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(win, disp)

            k = cv2.waitKey(20) & 0xFF
            if k == 255:
                continue
            if k in (ord("n"), ord(" ")):
                write_labels(lbl_path, state["rows"])
                idx += 1
                break
            if k == ord("p"):
                write_labels(lbl_path, state["rows"])
                idx = max(0, idx - 1)
                break
            if k in (ord("q"), 27):
                write_labels(lbl_path, state["rows"])
                cv2.destroyAllWindows()
                print(f"Saved. Reviewed up to image {idx + 1}/{len(images)}.")
                return
            if k in (ord("d"), ord("x")) and state["sel"] >= 0:
                state["rows"].pop(state["sel"])
                state["sel"] = -1
            elif k in (ord("c"), ord("C")):
                step = 1 if k == ord("c") else -1
                if state["sel"] >= 0:
                    state["rows"][state["sel"]][0] = \
                        (int(state["rows"][state["sel"]][0]) + step) % len(classes)
                else:
                    state["active_cls"] = (state["active_cls"] + step) % len(classes)
            elif ord("0") <= k <= ord("9"):
                ci = k - ord("0")
                if ci < len(classes):
                    state["active_cls"] = ci
                    if state["sel"] >= 0:
                        state["rows"][state["sel"]][0] = ci
            elif k == ord("u"):
                state.update(rows=read_labels(lbl_path), sel=-1)

    cv2.destroyAllWindows()
    print(f"Reviewed all {len(images)} images in {img_dir}.")


if __name__ == "__main__":
    main()
