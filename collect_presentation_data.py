#!/usr/bin/env python3
"""
collect_presentation_data.py

Data collection pipeline that generates all images, data, and matplotlib plots
called out in PRESENTATION_REPORT.md for the "Analyzing Sensor Data" course presentation.

Runs on Raspberry Pi 5 + Camera Module v3 + PCA9685 (I2C2, GPIO 12/13 SDA/SCL)
+ dual 650nm HiLetgo laser diodes on PCA channels 6 and 7.

Output layout:
    output/presentation_data/
        slide_03_problem_statement/
        slide_04_signal_model/
        slide_05_06_hsv/
        slide_07_bgr/
        slide_08_comparison/
        slide_09_10_temporal/
        slide_11_morphology/
        slide_12_contours/
        slide_13_exposure/

Usage:
    python3 collect_presentation_data.py
    python3 collect_presentation_data.py --n-frames 60 --skip-exposure-sweep
    python3 collect_presentation_data.py --phases problem hsv temporal contours
"""

import sys
import argparse
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # headless – no display needed over SSH
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent))
from src.laser_detector import LaserDetector
from src.camera_capture import CameraCapture

# ── colour palette consistent across all plots ────────────────────────────────
C = {
    "laser":      "#E63946",
    "ambient":    "#457B9D",
    "temporal":   "#2A9D8F",
    "hsv":        "#E9C46A",
    "bgr":        "#A8DADC",
    "noise":      "#6D6875",
    "pass_green": "#52B788",
    "fail_red":   "#E63946",
    "fail_orange":"#F4A261",
}

DPI = 150  # good quality for PowerPoint

# Exposure sweep values in microseconds (short → long)
DEFAULT_EXPOSURES_US = [100, 500, 1_000, 5_000, 10_000, 30_000, 100_000]


# ─────────────────────────────────────────────────────────────────────────────
class PresentationDataCollector:
    """Orchestrates all data and image collection phases for the presentation."""

    def __init__(
        self,
        n_frames: int = 40,
        exposure_values_us: List[int] = None,
        output_dir: Path = Path("output/presentation_data"),
        phases: Optional[List[str]] = None,
    ):
        self.n_frames = n_frames
        self.exposure_values_us = exposure_values_us or DEFAULT_EXPOSURES_US
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phases = set(phases) if phases else None  # None = run all

        # ── hardware ──────────────────────────────────────────────────────────
        print("[INIT] Starting camera …")
        self.cap = CameraCapture(camera_id=0, width=640, height=480, fps=30)

        print("[INIT] Initialising HSV detector …")
        self.detector_hsv = LaserDetector(
            method="hsv", conf_threshold=0.3,
            pca_channel1=6, pca_channel2=7,
        )
        self.detector_hsv.set_camera_capture(self.cap)

        print("[INIT] Initialising BGR detector …")
        self.detector_bgr = LaserDetector(
            method="bgr", conf_threshold=0.3,
            pca_channel1=6, pca_channel2=7,
        )
        self.detector_bgr.set_camera_capture(self.cap)

        print("[INIT] Initialising temporal detector …")
        self.detector_temporal = LaserDetector(
            method="temporal", conf_threshold=0.3,
            pca_channel1=6, pca_channel2=7,
        )
        self.detector_temporal.set_camera_capture(self.cap)

        self.saved_files: List[str] = []

    # ── hardware helpers ──────────────────────────────────────────────────────

    def _lasers_on(self):
        for ctrl in (self.detector_hsv.laser_controller1,
                     self.detector_hsv.laser_controller2):
            if ctrl:
                ctrl.on()

    def _lasers_off(self):
        for ctrl in (self.detector_hsv.laser_controller1,
                     self.detector_hsv.laser_controller2):
            if ctrl:
                ctrl.off()

    def _get_picam2(self):
        """Return the raw Picamera2 instance buried inside CameraCapture."""
        return self.cap.camera.camera  # CameraCapture → PiCamera2Wrapper → Picamera2

    def _set_exposure(self, exposure_us: int):
        """Disable auto-exposure and set manual shutter time in microseconds."""
        try:
            self._get_picam2().set_controls({
                "AeEnable": False,
                "ExposureTime": int(exposure_us),
            })
        except Exception as e:
            print(f"[WARN] Could not set exposure to {exposure_us}µs: {e}")

    def _reset_exposure(self):
        """Re-enable auto-exposure."""
        try:
            self._get_picam2().set_controls({"AeEnable": True})
        except Exception as e:
            print(f"[WARN] Could not reset AE: {e}")

    def _settle_camera(self, n_discard: int = 8):
        """Discard n frames so control changes take effect before capturing data."""
        for _ in range(n_discard):
            self.cap.read()

    def _capture(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()

    def _capture_good(self) -> Optional[np.ndarray]:
        """Capture and return a frame; return None on failure."""
        ret, frame = self.cap.read()
        return frame if ret and frame is not None else None

    # ── I/O helpers ───────────────────────────────────────────────────────────

    def _subdir(self, name: str) -> Path:
        d = self.output_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_img(self, img: np.ndarray, subdir: str, filename: str):
        path = self._subdir(subdir) / filename
        cv2.imwrite(str(path), img)
        self.saved_files.append(str(path))
        print(f"  → {path}")

    def _save_fig(self, fig: plt.Figure, subdir: str, filename: str):
        path = self._subdir(subdir) / filename
        fig.savefig(str(path), dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        self.saved_files.append(str(path))
        print(f"  → {path}")

    def _should_run(self, phase: str) -> bool:
        return self.phases is None or phase in self.phases

    # ── warm-up ───────────────────────────────────────────────────────────────

    def _warmup(self):
        print("[WARMUP] Turning lasers on and letting camera settle …")
        self._lasers_on()
        self._settle_camera(n_discard=20)

    # =========================================================================
    # SLIDE 3 — Problem Statement
    # =========================================================================

    def phase_problem_statement(self):
        print("\n[PHASE] Slide 3 — Problem Statement")
        sd = "slide_03_problem_statement"
        frame = self._capture_good()
        if frame is None:
            print("  [SKIP] Could not capture frame")
            return

        # Raw frame
        self._save_img(frame, sd, "raw_frame.png")

        # Detect laser so we can annotate pixel count
        detections, debug = self.detector_hsv.detect(frame)
        annotated = frame.copy()
        total_px = frame.shape[0] * frame.shape[1]

        for det in detections:
            cx, cy = int(det["x"]), int(det["y"])
            area = int(det.get("area", 0))
            pct = 100.0 * area / total_px
            cv2.circle(annotated, (cx, cy), 18, (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 3, (0, 255, 0), -1)
            label = f"laser: {area}px ({pct:.4f}% of frame)"
            cv2.putText(annotated, label, (cx + 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        label_frame = f"Total pixels: {total_px:,}"
        cv2.putText(annotated, label_frame, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self._save_img(annotated, sd, "annotated_frame.png")

    # =========================================================================
    # SLIDE 4 — Signal Model: 3D red-channel surface
    # =========================================================================

    def phase_signal_model(self):
        print("\n[PHASE] Slide 4 — Signal Model (3D red-channel surface)")
        sd = "slide_04_signal_model"
        frame = self._capture_good()
        if frame is None:
            print("  [SKIP] Could not capture frame")
            return

        detections, _ = self.detector_hsv.detect(frame)
        self._save_img(frame, sd, "raw_frame.png")

        # Crop region around laser dot (60×60 px) or full red channel if not found
        r_channel = frame[:, :, 2].astype(np.float32)
        pad = 30
        if detections:
            cx = int(detections[0]["x"])
            cy = int(detections[0]["y"])
            y0 = max(0, cy - pad)
            y1 = min(frame.shape[0], cy + pad)
            x0 = max(0, cx - pad)
            x1 = min(frame.shape[1], cx + pad)
            crop = r_channel[y0:y1, x0:x1]
            title = f"Red channel R(x,y) — ±{pad}px around laser centroid ({cx},{cy})"
        else:
            # Fall back to centre of frame
            h, w = frame.shape[:2]
            crop = r_channel[h//2 - pad: h//2 + pad, w//2 - pad: w//2 + pad]
            title = "Red channel R(x,y) — centre crop (no laser detected)"

        rows, cols = crop.shape
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, crop, cmap="inferno", linewidth=0, antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.5, label="Red channel DN (0–255)")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        ax.set_zlabel("R(x,y)")
        ax.set_title(title, fontsize=9)
        # Mark peak
        peak_y, peak_x = np.unravel_index(crop.argmax(), crop.shape)
        ax.scatter([peak_x], [peak_y], [crop.max()],
                   color=C["laser"], s=80, zorder=5, label=f"Peak: {int(crop.max())} DN")
        ax.legend()
        self._save_fig(fig, sd, "red_channel_surface_3d.png")

        # Also save a flat heatmap for easier PowerPoint import
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im = ax2.imshow(crop, cmap="inferno", vmin=0, vmax=255)
        fig2.colorbar(im, ax=ax2, label="R channel (DN)")
        ax2.set_title("Red channel heatmap — laser dot region", fontsize=9)
        ax2.set_xlabel("x (pixels)")
        ax2.set_ylabel("y (pixels)")
        if detections:
            ax2.scatter([peak_x], [peak_y], c=C["laser"], s=60,
                        marker="x", linewidths=2, label="Peak")
            ax2.legend()
        self._save_fig(fig2, sd, "red_channel_heatmap.png")

    # =========================================================================
    # SLIDES 5-6 — HSV Method
    # =========================================================================

    def phase_hsv(self):
        print("\n[PHASE] Slides 5-6 — HSV Method")
        sd = "slide_05_06_hsv"
        frame = self._capture_good()
        if frame is None:
            print("  [SKIP] Could not capture frame")
            return

        # ── 1. Original BGR frame ─────────────────────────────────────────────
        self._save_img(frame, sd, "01_original_bgr.png")

        # ── 2. H-channel colourised (hue identity render) ─────────────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        hue_colorized = np.zeros_like(frame)
        hue_colorized[:, :, 0] = h_channel        # Hue from original
        hue_colorized[:, :, 1] = 255              # Full saturation to reveal hue
        hue_colorized[:, :, 2] = 255              # Full value
        hue_bgr = cv2.cvtColor(hue_colorized, cv2.COLOR_HSV2BGR)
        self._save_img(hue_bgr, sd, "02_h_channel_colorized.png")

        # matplotlib H-channel heatmap with red band annotations
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(h_channel, cmap="hsv", vmin=0, vmax=180, aspect="auto")
        plt.colorbar(im, ax=ax, label="Hue (OpenCV 0–180°)")
        ax.set_title("Hue channel H(x,y) with red detection bands")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        fig.text(0.15, 0.01,
                 "Red bands: H∈[0,10] (left wrap) and H∈[170,180] (right wrap)",
                 fontsize=7, color=C["laser"])
        self._save_fig(fig, sd, "03_h_channel_heatmap.png")

        # ── 3. Binary mask (before morphology) ───────────────────────────────
        mask_raw = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for h_min, h_max in self.detector_hsv.hsv_h_ranges:
            lower = np.array([h_min,
                               self.detector_hsv.hsv_s_min,
                               self.detector_hsv.hsv_v_min])
            upper = np.array([h_max, 255, 255])
            mask_raw |= cv2.inRange(hsv, lower, upper)
        self._save_img(cv2.cvtColor(mask_raw, cv2.COLOR_GRAY2BGR),
                       sd, "04_raw_threshold_mask.png")

        # ── 4. Mask after morphology ─────────────────────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_opened = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel)
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
        self._save_img(cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2BGR),
                       sd, "05_mask_after_morphology.png")

        # ── 5. Three-panel composite ──────────────────────────────────────────
        h, w = frame.shape[:2]
        panel = np.hstack([
            frame,
            hue_bgr,
            cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2BGR),
        ])
        for i, lbl in enumerate(["Original BGR", "H-channel (colorized)", "HSV Mask"]):
            cv2.putText(panel, lbl, (i * w + 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        self._save_img(panel, sd, "06_three_panel.png")

        # ── 6. Mask pixel count bar chart over N frames ───────────────────────
        print(f"  Collecting mask pixel counts over {self.n_frames} frames …")
        mask_counts = []
        conf_history = []
        for i in range(self.n_frames):
            f = self._capture_good()
            if f is None:
                continue
            dets, dbg = self.detector_hsv.detect(f)
            mask_counts.append(dbg.get("mask_pixel_count", 0))
            conf_history.append(
                float(np.mean([d["confidence"] for d in dets])) if dets else 0.0
            )

        frames_idx = list(range(len(mask_counts)))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax1.bar(frames_idx, mask_counts, color=C["hsv"], width=0.8, label="Mask pixels")
        ax1.axhline(np.mean(mask_counts), color=C["laser"], linestyle="--",
                    label=f"Mean: {np.mean(mask_counts):.0f} px")
        ax1.set_ylabel("Mask pixels passing inRange")
        ax1.set_title(f"HSV mask signal stability over {len(mask_counts)} frames")
        ax1.legend(fontsize=8)

        ax2.plot(frames_idx, conf_history, color=C["laser"], linewidth=1.5,
                 label="Mean detection confidence")
        ax2.axhline(np.mean([c for c in conf_history if c > 0]),
                    color=C["noise"], linestyle="--", linewidth=0.8)
        ax2.set_ylabel("Confidence (circularity score)")
        ax2.set_xlabel("Frame index")
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=8)
        self._save_fig(fig, sd, "07_mask_pixel_count_and_confidence.png")

    # =========================================================================
    # SLIDE 7 — BGR Method
    # =========================================================================

    def phase_bgr(self):
        print("\n[PHASE] Slide 7 — BGR Method")
        sd = "slide_07_bgr"
        frame = self._capture_good()
        if frame is None:
            print("  [SKIP] Could not capture frame")
            return

        b, g, r = cv2.split(frame)

        # Save individual channels as false-colour for clarity
        zero = np.zeros_like(b)
        b_img = cv2.merge([b, zero, zero])
        g_img = cv2.merge([zero, g, zero])
        r_img = cv2.merge([zero, zero, r])

        # BGR binary mask (replicate detector logic)
        r_thresh, diff_thresh = 100, 30
        mask_bgr = ((r > r_thresh) &
                    (r > b.astype(int) + diff_thresh) &
                    (r > g.astype(int) + diff_thresh)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_bgr = cv2.morphologyEx(mask_bgr, cv2.MORPH_OPEN, kernel)
        mask_bgr = cv2.morphologyEx(mask_bgr, cv2.MORPH_CLOSE, kernel)

        # Four-panel: B | G | R | mask
        h, w = frame.shape[:2]
        panel = np.hstack([b_img, g_img, r_img,
                           cv2.cvtColor(mask_bgr, cv2.COLOR_GRAY2BGR)])
        labels = ["Blue channel", "Green channel", "Red channel", "BGR Mask"]
        for i, lbl in enumerate(labels):
            cv2.putText(panel, lbl, (i * w + 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        self._save_img(panel, sd, "01_bgr_four_panel.png")

        # matplotlib channel profile plot across horizontal midline through laser
        dets, _ = self.detector_hsv.detect(frame)
        row = int(dets[0]["y"]) if dets else frame.shape[0] // 2
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = np.arange(w)
        ax.plot(xs, frame[row, :, 0], color=C["ambient"], linewidth=1.2, label="Blue (B)")
        ax.plot(xs, frame[row, :, 1], color=C["temporal"], linewidth=1.2, label="Green (G)")
        ax.plot(xs, frame[row, :, 2], color=C["laser"], linewidth=1.5, label="Red (R)")
        if dets:
            cx = int(dets[0]["x"])
            ax.axvline(cx, color="white", linestyle="--", linewidth=0.8,
                       label=f"Laser centroid (x={cx})")
        ax.fill_between(xs,
                        np.minimum(frame[row, :, 2].astype(int) - diff_thresh,
                                   r_thresh),
                        frame[row, :, 2],
                        where=(mask_bgr[row, :] > 0),
                        alpha=0.25, color=C["laser"], label="Passes BGR mask")
        ax.set_xlim(0, w)
        ax.set_ylim(0, 270)
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("Digital Number (DN)")
        ax.set_title(f"Channel profiles along row y={row} through laser dot")
        ax.axhline(r_thresh, color=C["laser"], linestyle=":", linewidth=0.8,
                   label=f"r_thresh={r_thresh}")
        ax.legend(fontsize=8)
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#0f0e17")
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_color("white")
        ax.legend(fontsize=8, facecolor="#222", labelcolor="white")
        self._save_fig(fig, sd, "02_channel_profile.png")

    # =========================================================================
    # SLIDE 8 — Method Comparison
    # =========================================================================

    def phase_comparison(self):
        print(f"\n[PHASE] Slide 8 — Method Comparison ({self.n_frames} frames)")
        sd = "slide_08_comparison"

        records: Dict[str, List] = defaultdict(list)
        for i in range(self.n_frames):
            f = self._capture_good()
            if f is None:
                continue
            for label, det in [("HSV", self.detector_hsv), ("BGR", self.detector_bgr)]:
                dets, dbg = det.detect(f)
                records[f"{label}_count"].append(len(dets))
                records[f"{label}_conf"].append(
                    float(np.mean([d["confidence"] for d in dets])) if dets else 0.0
                )
                records[f"{label}_time_ms"].append(
                    dbg.get("processing_time_ms", 0.0)
                )

        frames_idx = list(range(len(records["HSV_count"])))
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

        # Detection count
        axes[0].plot(frames_idx, records["HSV_count"],
                     color=C["hsv"], linewidth=1.5, label="HSV")
        axes[0].plot(frames_idx, records["BGR_count"],
                     color=C["bgr"], linewidth=1.5, label="BGR")
        axes[0].set_ylabel("Detections / frame")
        axes[0].set_title("Method comparison: HSV vs BGR over live frames")
        axes[0].legend(fontsize=8)
        axes[0].set_ylim(-0.1, max(max(records["HSV_count"] or [1]),
                                    max(records["BGR_count"] or [1])) + 1)

        # Confidence
        axes[1].plot(frames_idx, records["HSV_conf"],
                     color=C["hsv"], linewidth=1.5, label="HSV confidence")
        axes[1].plot(frames_idx, records["BGR_conf"],
                     color=C["bgr"], linewidth=1.5, label="BGR confidence")
        axes[1].set_ylabel("Mean confidence")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(fontsize=8)

        # Processing time
        axes[2].plot(frames_idx, records["HSV_time_ms"],
                     color=C["hsv"], linewidth=1.2, label="HSV")
        axes[2].plot(frames_idx, records["BGR_time_ms"],
                     color=C["bgr"], linewidth=1.2, label="BGR")
        axes[2].set_ylabel("Processing time (ms)")
        axes[2].set_xlabel("Frame index")
        axes[2].legend(fontsize=8)

        fig.tight_layout()
        self._save_fig(fig, sd, "method_comparison.png")

        # Summary bar chart
        methods = ["HSV", "BGR"]
        mean_conf = [np.mean(records[f"{m}_conf"]) for m in methods]
        mean_time = [np.mean(records[f"{m}_time_ms"]) for m in methods]

        fig2, (a1, a2) = plt.subplots(1, 2, figsize=(7, 4))
        a1.bar(methods, mean_conf, color=[C["hsv"], C["bgr"]])
        a1.set_ylim(0, 1.0)
        a1.set_ylabel("Mean confidence (circularity)")
        a1.set_title("Detection confidence")
        a2.bar(methods, mean_time, color=[C["hsv"], C["bgr"]])
        a2.set_ylabel("Mean processing time (ms)")
        a2.set_title("Latency")
        fig2.suptitle("HSV vs BGR summary", fontsize=11)
        fig2.tight_layout()
        self._save_fig(fig2, sd, "method_summary_bars.png")

    # =========================================================================
    # SLIDES 9-10 — Temporal Method + theoretical SNR
    # =========================================================================

    def phase_temporal(self):
        print("\n[PHASE] Slides 9-10 — Temporal Method")
        sd = "slide_09_10_temporal"

        # Capture frame_on (lasers already on from warmup)
        frame_on = self._capture_good()
        if frame_on is None:
            print("  [SKIP] frame_on capture failed")
            return

        # Turn lasers off, settle one frame (~33ms), capture frame_off
        self._lasers_off()
        time.sleep(0.04)
        frame_off = self._capture_good()
        self._lasers_on()  # restore

        if frame_off is None:
            print("  [SKIP] frame_off capture failed")
            return

        diff = cv2.absdiff(frame_on, frame_off)
        # Amplify for visualisation (raw diff may be dim)
        diff_vis = np.clip(diff.astype(np.int32) * 4, 0, 255).astype(np.uint8)

        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask_temporal = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_temporal = cv2.morphologyEx(mask_temporal, cv2.MORPH_OPEN, kernel)

        # Three-panel composite
        h, w = frame_on.shape[:2]
        panel = np.hstack([frame_on, frame_off, diff_vis])
        for i, lbl in enumerate(["Laser ON (frame_on)", "Laser OFF (frame_off)",
                                  "|frame_on − frame_off| ×4"]):
            cv2.putText(panel, lbl, (i * w + 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        self._save_img(panel, sd, "01_temporal_three_panel.png")

        # Also save individual frames
        self._save_img(frame_on, sd, "02_frame_on.png")
        self._save_img(frame_off, sd, "03_frame_off.png")
        self._save_img(diff_vis, sd, "04_temporal_diff_amplified.png")
        self._save_img(cv2.cvtColor(mask_temporal, cv2.COLOR_GRAY2BGR),
                       sd, "05_temporal_mask.png")

        # Difference histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(gray_diff.flatten(), bins=128, range=(0, 255),
                color=C["temporal"], edgecolor="none", alpha=0.85)
        ax.axvline(50, color=C["laser"], linestyle="--", linewidth=1.5,
                   label="Threshold = 50 DN")
        laser_px = int((gray_diff > 50).sum())
        ax.set_xlabel("Absolute difference (DN)")
        ax.set_ylabel("Pixel count")
        ax.set_title(f"Temporal diff histogram — {laser_px} pixels above threshold")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        self._save_fig(fig, sd, "06_temporal_diff_histogram.png")

        # ── Theoretical SNR curves ────────────────────────────────────────────
        # Simplified model (all values normalised to full-scale DN=255)
        sigma_read = 0.02       # read noise floor (2% FS)
        S_laser    = 0.80       # laser signal amplitude (80% FS, typically saturated)
        bg_levels  = np.linspace(0, 1.0, 200)  # ambient background 0→100%

        # Shot noise of background (Poisson: σ_shot = sqrt(Φ_bg))
        sigma_shot_bg = np.sqrt(bg_levels + 1e-6) * 0.15  # scale factor for illustration

        snr_hsv = S_laser / np.sqrt(sigma_read**2 + sigma_shot_bg**2)
        snr_temporal = np.full_like(bg_levels,
                                     S_laser / (np.sqrt(2) * sigma_read))

        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        ax2.plot(bg_levels * 100, snr_hsv,
                 color=C["hsv"], linewidth=2, label="HSV thresholding")
        ax2.plot(bg_levels * 100, snr_temporal,
                 color=C["temporal"], linewidth=2, linestyle="--",
                 label="Temporal differencing")
        ax2.fill_between(bg_levels * 100, snr_temporal, snr_hsv,
                         where=(snr_temporal > snr_hsv),
                         alpha=0.15, color=C["temporal"],
                         label="Temporal advantage region")
        ax2.set_xlabel("Background brightness (% full scale)")
        ax2.set_ylabel("Signal-to-Noise Ratio (SNR)")
        ax2.set_title("Theoretical SNR: HSV thresholding vs temporal differencing")
        ax2.set_ylim(0, snr_temporal.max() * 1.25)
        ax2.legend(fontsize=9)
        ax2.text(50, snr_temporal.max() * 0.55,
                 r"$\mathrm{SNR}_{temporal} = \frac{S_{laser}}{\sqrt{2}\,\sigma_{read}}$"
                 "\n(background independent)",
                 fontsize=8, color=C["temporal"],
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#222", alpha=0.7))
        ax2.text(60, snr_hsv[100] * 0.6,
                 r"$\mathrm{SNR}_{HSV} = \frac{S_{laser}}{\sqrt{\sigma_{read}^2 + \sigma_{bg}^2}}$"
                 "\n(degrades with ambient)",
                 fontsize=8, color=C["hsv"],
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#222", alpha=0.7))
        self._save_fig(fig2, sd, "07_snr_theoretical_curves.png")

    # =========================================================================
    # SLIDE 11 — Morphological Filtering
    # =========================================================================

    def phase_morphology(self):
        print("\n[PHASE] Slide 11 — Morphological Filtering")
        sd = "slide_11_morphology"
        frame = self._capture_good()
        if frame is None:
            print("  [SKIP] Could not capture frame")
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Stage 0: raw threshold mask
        mask_thresh = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for h_min, h_max in self.detector_hsv.hsv_h_ranges:
            lower = np.array([h_min,
                               self.detector_hsv.hsv_s_min,
                               self.detector_hsv.hsv_v_min])
            upper = np.array([h_max, 255, 255])
            mask_thresh |= cv2.inRange(hsv, lower, upper)

        # Stage 1: MORPH_OPEN (erosion → dilation)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_opened = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

        # Stage 2: MORPH_CLOSE (dilation → erosion)
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

        def to_bgr(m):
            return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

        h, w = frame.shape[:2]
        panel = np.hstack([to_bgr(mask_thresh), to_bgr(mask_opened), to_bgr(mask_closed)])
        labels = ["Raw threshold", "After MORPH_OPEN\n(erosion→dilation)",
                  "After MORPH_CLOSE\n(dilation→erosion)"]
        for i, lbl in enumerate(labels):
            lbl_line = lbl.split("\n")[0]
            cv2.putText(panel, lbl_line, (i * w + 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        self._save_img(panel, sd, "01_morph_three_stages.png")

        # Count statistics for each stage
        counts = [int(mask_thresh.sum() // 255),
                  int(mask_opened.sum() // 255),
                  int(mask_closed.sum() // 255)]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        stage_names = ["Raw threshold", "After OPEN", "After CLOSE"]
        cmaps = ["binary", "binary", "binary"]
        for ax, m, name, count in zip(axes, [mask_thresh, mask_opened, mask_closed],
                                       stage_names, counts):
            ax.imshow(m, cmap="binary_r")
            ax.set_title(f"{name}\n{count:,} white pixels", fontsize=9)
            ax.axis("off")
        fig.suptitle("Morphological filtering stages", fontsize=11)
        fig.tight_layout()
        self._save_fig(fig, sd, "02_morph_stages_matplotlib.png")

        # Pixel count bar across stages
        fig2, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(stage_names, counts, color=[C["noise"], C["hsv"], C["temporal"]])
        ax.bar_label(bars, fmt="%d px")
        ax.set_ylabel("White pixels in mask")
        ax.set_title("White pixel count at each morphological stage")
        self._save_fig(fig2, sd, "03_morph_pixel_count.png")

    # =========================================================================
    # SLIDE 12 — Contour Analysis
    # =========================================================================

    def phase_contours(self):
        print(f"\n[PHASE] Slide 12 — Contour Analysis ({self.n_frames} frames)")
        sd = "slide_12_contours"

        # ── Annotated contour image ───────────────────────────────────────────
        frame = self._capture_good()
        if frame is None:
            print("  [SKIP] Could not capture frame")
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for h_min, h_max in self.detector_hsv.hsv_h_ranges:
            lower = np.array([h_min, self.detector_hsv.hsv_s_min,
                               self.detector_hsv.hsv_v_min])
            upper = np.array([h_max, 255, 255])
            mask |= cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        annotated = frame.copy()
        # Colour by pass/fail status
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perim = cv2.arcLength(cnt, True)
            circ = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0

            if area < 1 or area > 200:
                color = (0, 0, 220)   # blue  – fails area filter
                reason = f"area={area:.0f}"
            elif circ < 0.5:
                color = (0, 140, 255)  # orange – fails circularity
                reason = f"circ={circ:.2f}"
            else:
                color = (0, 220, 80)  # green – passes all filters
                reason = f"PASS circ={circ:.2f}"

            cv2.drawContours(annotated, [cnt], -1, color, 2)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(annotated, reason, (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Legend
        for i, (lbl, col) in enumerate([
                ("PASS (area OK + circular)", (0, 220, 80)),
                ("FAIL: area out of [1,200]", (0, 0, 220)),
                ("FAIL: circularity < 0.5", (0, 140, 255)),
        ]):
            cv2.putText(annotated, lbl, (8, frame.shape[0] - 15 - i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        self._save_img(annotated, sd, "01_contour_annotated.png")

        # ── Scatter: area vs circularity over N frames ────────────────────────
        print(f"  Collecting contour scatter over {self.n_frames} frames …")
        scatter_area, scatter_circ, scatter_pass = [], [], []

        for _ in range(self.n_frames):
            f = self._capture_good()
            if f is None:
                continue
            hsv_f = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            m = np.zeros(hsv_f.shape[:2], dtype=np.uint8)
            for h_min, h_max in self.detector_hsv.hsv_h_ranges:
                lo = np.array([h_min, self.detector_hsv.hsv_s_min,
                                self.detector_hsv.hsv_v_min])
                hi = np.array([h_max, 255, 255])
                m |= cv2.inRange(hsv_f, lo, hi)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                a = cv2.contourArea(cnt)
                p = cv2.arcLength(cnt, True)
                c = (4 * np.pi * a) / (p ** 2) if p > 0 else 0
                scatter_area.append(a)
                scatter_circ.append(c)
                passed = (1 <= a <= 200) and (c >= 0.5)
                scatter_pass.append(passed)

        if scatter_area:
            areas = np.array(scatter_area)
            circs = np.array(scatter_circ)
            passed = np.array(scatter_pass)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(areas[~passed], circs[~passed], c=C["fail_red"],
                       alpha=0.4, s=20, label="Rejected")
            ax.scatter(areas[passed], circs[passed], c=C["pass_green"],
                       alpha=0.7, s=40, marker="D", label="Accepted (laser)")
            ax.axvline(1,   color="white", linestyle=":", linewidth=0.8)
            ax.axvline(200, color="white", linestyle=":", linewidth=0.8,
                       label="Area bounds [1, 200]")
            ax.axhline(0.5, color=C["hsv"], linestyle="--", linewidth=1,
                       label="Circularity floor = 0.5")
            # Shade acceptance region
            ax.fill_betweenx([0.5, 1.05], 1, 200, alpha=0.07, color=C["pass_green"])
            ax.set_xlabel("Contour area (px²)")
            ax.set_ylabel("Circularity  4πA / P²")
            ax.set_title(f"Contour scatter: area vs circularity\n"
                         f"({len(areas)} contours over {self.n_frames} frames)")
            ax.set_xlim(-5, max(areas.max() * 1.1, 220))
            ax.set_ylim(0, 1.1)
            ax.legend(fontsize=8)
            ax.set_facecolor("#1a1a2e")
            fig.patch.set_facecolor("#0f0e17")
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                         + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_color("white")
            ax.legend(fontsize=8, facecolor="#222", labelcolor="white")
            self._save_fig(fig, sd, "02_area_circularity_scatter.png")

        # Moments diagram – annotate centroid on a single detection
        dets, dbg = self.detector_hsv.detect(frame)
        if dets:
            moment_vis = frame.copy()
            d = dets[0]
            cx, cy = int(d["x"]), int(d["y"])
            bx, by, bw, bh = (int(d["x"] - d["width"] / 2),
                               int(d["y"] - d["height"] / 2),
                               int(d["width"]), int(d["height"]))
            cv2.rectangle(moment_vis, (bx, by), (bx + bw, by + bh),
                          (255, 165, 0), 2)
            cv2.circle(moment_vis, (cx, cy), 5, (0, 255, 0), -1)
            cv2.circle(moment_vis, (cx, cy), 16, (0, 255, 0), 2)
            label_lines = [
                f"Centroid: ({cx}, {cy}) [m10/m00, m01/m00]",
                f"BBox: ({bx},{by}) {bw}x{bh} px",
                f"Confidence (circularity): {d['confidence']:.3f}",
                f"Area: {d.get('area', 0):.0f} px²",
            ]
            for k, line in enumerate(label_lines):
                cv2.putText(moment_vis, line, (cx + 25, cy - 20 + k * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            self._save_img(moment_vis, sd, "03_detection_moments_annotated.png")

    # =========================================================================
    # SLIDE 13 — Camera Exposure Sweep
    # =========================================================================

    def phase_exposure(self):
        print("\n[PHASE] Slide 13 — Camera Exposure Sweep")
        sd = "slide_13_exposure"
        exp_vals = self.exposure_values_us
        frames_per_exp = max(5, self.n_frames // len(exp_vals))

        conf_means: List[float] = []
        conf_stds:  List[float] = []
        rep_frames: List[np.ndarray] = []
        valid_exps: List[int] = []

        for exp_us in exp_vals:
            print(f"  Exposure {exp_us:>7,} µs … ", end="", flush=True)
            self._set_exposure(exp_us)
            self._settle_camera(n_discard=8)

            confs = []
            best_frame = None
            for _ in range(frames_per_exp):
                f = self._capture_good()
                if f is None:
                    continue
                dets, _ = self.detector_hsv.detect(f)
                c = float(np.mean([d["confidence"] for d in dets])) if dets else 0.0
                confs.append(c)
                if best_frame is None:
                    best_frame = f.copy()

            if best_frame is not None:
                mean_c = float(np.mean(confs)) if confs else 0.0
                std_c  = float(np.std(confs))  if confs else 0.0
                conf_means.append(mean_c)
                conf_stds.append(std_c)
                rep_frames.append(best_frame)
                valid_exps.append(exp_us)
                self._save_img(best_frame, sd, f"frame_{exp_us:07d}us.png")
                print(f"conf={mean_c:.3f} ± {std_c:.3f}")
            else:
                print("capture failed")

        self._reset_exposure()
        self._settle_camera(n_discard=5)

        if not valid_exps:
            print("  [SKIP] No valid frames collected during exposure sweep")
            return

        # ── Side-by-side exposure grid ────────────────────────────────────────
        n_show = min(len(rep_frames), 6)
        indices = np.linspace(0, len(rep_frames) - 1, n_show, dtype=int)
        fig_grid, axes_g = plt.subplots(1, n_show, figsize=(3 * n_show, 3.5))
        if n_show == 1:
            axes_g = [axes_g]
        for ax, idx in zip(axes_g, indices):
            rgb = cv2.cvtColor(rep_frames[idx], cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.set_title(f"{valid_exps[idx]:,} µs\nconf={conf_means[idx]:.2f}", fontsize=8)
            ax.axis("off")
        fig_grid.suptitle("Camera frames at different exposure times", fontsize=10)
        fig_grid.tight_layout()
        self._save_fig(fig_grid, sd, "exposure_frame_grid.png")

        # ── Red-channel histograms at each exposure ────────────────────────────
        fig_hist, axes_h = plt.subplots(1, n_show, figsize=(3 * n_show, 3.5),
                                         sharey=False)
        if n_show == 1:
            axes_h = [axes_h]
        hist_cmap = cm.get_cmap("cool", n_show)
        for i, (ax, idx) in enumerate(zip(axes_h, indices)):
            r_vals = rep_frames[idx][:, :, 2].flatten()
            ax.hist(r_vals, bins=64, range=(0, 255), color=hist_cmap(i),
                    edgecolor="none", alpha=0.85)
            ax.axvline(100, color=C["laser"], linestyle="--", linewidth=0.8)
            ax.set_title(f"{valid_exps[idx]:,} µs", fontsize=8)
            ax.set_xlabel("R DN")
            if i == 0:
                ax.set_ylabel("Pixel count")
        fig_hist.suptitle("Red channel histograms at each exposure time\n"
                           "(dashed line = r_thresh=100)", fontsize=9)
        fig_hist.tight_layout()
        self._save_fig(fig_hist, sd, "exposure_red_histograms.png")

        # ── Mean red channel per exposure ─────────────────────────────────────
        mean_red = [float(f[:, :, 2].mean()) for f in rep_frames]
        fig_mr, ax_mr = plt.subplots(figsize=(6, 4))
        ax_mr.semilogx(valid_exps, mean_red, "o-", color=C["ambient"],
                        linewidth=2, markersize=6)
        ax_mr.set_xlabel("Exposure time (µs, log scale)")
        ax_mr.set_ylabel("Mean red channel (DN)")
        ax_mr.set_title("Scene brightness vs exposure time")
        ax_mr.axhline(100, color=C["noise"], linestyle=":", linewidth=0.8,
                      label="r_thresh = 100")
        ax_mr.legend(fontsize=8)
        self._save_fig(fig_mr, sd, "exposure_mean_red.png")

        # ── Detection confidence vs exposure ──────────────────────────────────
        fig_conf, ax_c = plt.subplots(figsize=(7, 4.5))
        ax_c.errorbar(valid_exps, conf_means, yerr=conf_stds,
                      fmt="o-", color=C["laser"], linewidth=2,
                      capsize=4, elinewidth=1, label="HSV confidence")
        ax_c.set_xscale("log")
        ax_c.set_xlabel("Exposure time (µs, log scale)")
        ax_c.set_ylabel("Mean detection confidence")
        ax_c.set_ylim(-0.05, 1.1)
        ax_c.set_title("Detection confidence vs camera exposure time\n"
                        "(laser ON throughout sweep)")
        # Annotate sweet spot
        if conf_means:
            best_idx = int(np.argmax(conf_means))
            ax_c.annotate(
                f"Best: {valid_exps[best_idx]:,} µs",
                xy=(valid_exps[best_idx], conf_means[best_idx]),
                xytext=(valid_exps[best_idx] * 2, conf_means[best_idx] - 0.15),
                arrowprops=dict(arrowstyle="->", color="white"),
                color="white", fontsize=9,
            )
        ax_c.legend(fontsize=8)
        ax_c.set_facecolor("#1a1a2e")
        fig_conf.patch.set_facecolor("#0f0e17")
        for item in ([ax_c.title, ax_c.xaxis.label, ax_c.yaxis.label]
                     + ax_c.get_xticklabels() + ax_c.get_yticklabels()):
            item.set_color("white")
        ax_c.legend(fontsize=8, facecolor="#222", labelcolor="white")
        self._save_fig(fig_conf, sd, "exposure_confidence_vs_exposure.png")

    # =========================================================================
    # Run all phases
    # =========================================================================

    def run(self):
        start = time.time()
        try:
            self._warmup()

            phase_map = {
                "problem":    self.phase_problem_statement,
                "signal":     self.phase_signal_model,
                "hsv":        self.phase_hsv,
                "bgr":        self.phase_bgr,
                "comparison": self.phase_comparison,
                "temporal":   self.phase_temporal,
                "morphology": self.phase_morphology,
                "contours":   self.phase_contours,
                "exposure":   self.phase_exposure,
            }

            for name, fn in phase_map.items():
                if self._should_run(name):
                    try:
                        fn()
                    except Exception as e:
                        print(f"  [ERROR] Phase '{name}' failed: {e}")
                else:
                    print(f"\n[SKIP] Phase '{name}' not in --phases list")

        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")
        finally:
            print("\n[CLEANUP] Turning lasers off …")
            self._lasers_off()
            self.cap.release()
            self._print_summary(time.time() - start)

    def _print_summary(self, elapsed: float):
        print("\n" + "=" * 65)
        print("COLLECTION COMPLETE")
        print("=" * 65)
        print(f"Elapsed: {elapsed:.1f}s")
        print(f"Output:  {self.output_dir.resolve()}")
        print(f"Files saved: {len(self.saved_files)}")
        for f in self.saved_files:
            print(f"  {f}")
        print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect presentation data for laser-detection signal processing slides"
    )
    parser.add_argument("--n-frames", type=int, default=40,
                        help="Frames to collect per statistical phase (default: 40)")
    parser.add_argument("--output-dir", type=str,
                        default="output/presentation_data",
                        help="Root output directory")
    parser.add_argument("--exposure-values", type=int, nargs="+",
                        default=DEFAULT_EXPOSURES_US,
                        metavar="US",
                        help="Exposure sweep values in microseconds")
    parser.add_argument("--phases", type=str, nargs="+",
                        choices=["problem", "signal", "hsv", "bgr",
                                 "comparison", "temporal", "morphology",
                                 "contours", "exposure"],
                        default=None,
                        help="Run only these phases (default: all)")
    parser.add_argument("--skip-exposure-sweep", action="store_true",
                        help="Shortcut to exclude the exposure sweep phase")

    args = parser.parse_args()

    phases = args.phases
    if args.skip_exposure_sweep:
        all_phases = ["problem", "signal", "hsv", "bgr",
                      "comparison", "temporal", "morphology", "contours"]
        phases = [p for p in (phases or all_phases) if p != "exposure"]

    collector = PresentationDataCollector(
        n_frames=args.n_frames,
        exposure_values_us=args.exposure_values,
        output_dir=Path(args.output_dir),
        phases=phases,
    )
    collector.run()


if __name__ == "__main__":
    main()
