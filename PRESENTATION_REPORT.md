# LaserEyes: Classical Signal Processing for Real-Time Laser Detection
### Presentation Report — "Analyzing Sensor Data"
### Graduate-Level Analysis | Estimated Duration: 6–10 minutes

---

## SLIDE 1 — Title Slide

**LaserEyes: Classical Signal Processing for Real-Time Laser Detection**

*A study in 2D discrete signal thresholding, color-space transforms, morphological filtering, and temporal differencing using an embedded camera system.*

> **[IMAGE: Place a side-by-side photo here — raw camera frame on the left, annotated detection frame with bounding box on the right. Ideally show the laser dot clearly visible as a saturated red point source in both.]*

---

## SLIDE 2 — Outline

**What This Talk Covers**

1. The Problem: Detecting a coherent light source in a noisy scene
2. The Signal Model: Images as 2D discrete signals
3. Method 1 — Color Thresholding: HSV and BGR Spaces
4. Method 2 — Temporal Frame Differencing with Laser Pulsing
5. Morphological Filtering: Nonlinear Noise Suppression
6. Contour Analysis: From Binary Mask to Spatial Detection
7. Camera Exposure: SNR Engineering at the Sensor Level
8. Comparison and Tradeoffs

---

## SLIDE 3 — The Problem Statement

**Goal:** Localize a red laser dot (≈650 nm) in real time from a digital camera feed, in the presence of environmental clutter, ambient lighting, and reflective surfaces.

**Why classical signal processing?**
- Deep learning approaches require labeled data and GPU inference; this system runs on embedded hardware (e.g., Raspberry Pi) with a PCA9685 PWM controller driving the laser over I2C
- The laser's physical properties (narrow spectral band, coherent point source, high radiance) are *exploitable priors* — we can engineer a detector around known signal characteristics without learned features

**Fundamental tension:**
- The laser dot occupies only ~10–200 pixels out of potentially millions
- The signal-to-background ratio in raw pixel space can be very low depending on scene brightness
- Goal is to maximize true-positive rate while rejecting red surfaces, specular reflections, and colored lights

> **[IMAGE: Full camera frame with laser dot visible. Annotate approximate pixel count of the laser spot vs. total frame size to illustrate the sparsity of the signal.]*

---

## SLIDE 4 — The Signal Model: Images as 2D Discrete Signals

**Framing the problem in signal processing terms:**

A digital image is a 2D discrete-space signal:

$$I(x, y) = [B(x,y),\ G(x,y),\ R(x,y)]$$

where each channel is a quantized measurement of photon flux over a rectangular aperture at pixel position $(x, y)$.

- Spatial resolution = sampling rate in 2D space (pixels per unit length)
- Bit depth (8-bit per channel) = amplitude quantization
- Frame rate = temporal sampling rate

**The laser dot as a signal:**
- Spectrally narrow (~650 nm) → localized energy in the red channel
- Physically small (coherent collimated beam) → spatially compact, near-impulse in 2D
- High irradiance → large amplitude relative to diffuse ambient sources

**Detection is a hypothesis test at each pixel:**

$$H_1: I(x,y) \in \text{laser region} \quad \text{vs.} \quad H_0: I(x,y) \in \text{background}$$

All three methods below implement different discriminant functions for this test.

> **[IMAGE: 3D surface plot of the red channel R(x,y) zoomed around the laser dot — shows the near-Gaussian amplitude peak from the beam profile. MATLAB, matplotlib, or OpenCV imshow in 3D mode work well here.]*

---

## SLIDE 5 — Color Space Theory: Why Transform?

**BGR (Blue, Green, Red) — the native camera space:**

Digital sensors sample in linear RGB (or Bayer-filtered approximation). OpenCV represents this as BGR by convention. The channels are *not independent* with respect to color perception:
- A change in illumination brightness affects all three channels simultaneously
- Expressing "redness" requires a ratio, not a single threshold

**HSV (Hue, Saturation, Value) — perceptual decomposition:**

The transformation $\text{BGR} \rightarrow \text{HSV}$ decouples:
- **Hue (H):** The chromatic identity — wavelength-dominant angle on the color wheel. Range: `[0, 180]` in OpenCV (half of 360° to fit uint8)
- **Saturation (S):** The *purity* of the color — ratio of chromatic energy to total energy. `S = 0` is gray, `S = 255` is fully saturated
- **Value (V):** Luminance — brightness independent of hue

**Why this matters for laser detection:**

The transformation is:

$$V = \max(R, G, B)$$
$$S = \frac{V - \min(R,G,B)}{V} \cdot 255$$
$$H = 30 \cdot \frac{G - B}{V - \min(R,G,B)} \quad \text{(when } V = R\text{)}$$

This means the hue of the laser dot is *invariant to illumination changes* (within limits), making it a much more robust discriminant than raw red-channel amplitude alone.

> **[IMAGE: Show the HSV color cylinder diagram, with the red laser's location annotated at H≈0 and H≈170-180 due to wrapping. This visually explains why two hue ranges are needed.]*

---

## SLIDE 6 — Method 1A: HSV Thresholding

**Pipeline:**

```
Frame (BGR) → cvtColor (BGR→HSV) → inRange [two red bands] → mask OR → Morphology → Contours
```

**Step 1 — Color Space Transform:**
```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```
This is a nonlinear point-wise transform. Each pixel $(B, G, R)$ is independently mapped to $(H, S, V)$ using the equations above.

**Step 2 — Dual-Band Hue Thresholding:**
```python
for h_min, h_max in [(0, 10), (170, 180)]:
    lower = np.array([h_min, hsv_s_min, hsv_v_min])
    upper = np.array([h_max, 255, 255])
    mask |= cv2.inRange(hsv, lower, upper)
```

`cv2.inRange` applies a 3D box filter in HSV space:

$$\text{mask}(x,y) = \begin{cases} 255 & \text{if } H \in [0,10]\cup[170,180]\ \text{AND}\ S \geq S_{min}\ \text{AND}\ V \geq V_{min} \\ 0 & \text{otherwise} \end{cases}$$

**Why two hue ranges?** Red is the *only* color that wraps around the HSV cylinder at H=0/180. A single range `[0,10]` would miss deep reds near H=175. The `OR` of two masks captures the full red lobe. This is analogous to a bandpass filter with two passbands positioned symmetrically around the wraparound point.

**Saturation floor (`hsv_s_min`):** Prevents near-white or near-gray pixels (low saturation) from passing — important for rejecting white walls under red illumination.

**Value floor (`hsv_v_min`):** Ensures the detection has enough brightness, rejecting dark brown/maroon surfaces that might otherwise satisfy the hue criterion.

> **[IMAGE: Show three panels side by side: (1) original BGR frame, (2) HSV-space visualization (can use H channel as color, S as saturation), (3) binary mask output. Label the laser dot region in each.]*

> **[DATA: Bar chart showing mask pixel counts over several frames — illustrates how stable the signal is with HSV thresholding vs. changing illumination.]*

---

## SLIDE 7 — Method 1B: BGR Thresholding

**Pipeline:**

```
Frame (BGR) → split channels → ratio thresholds → mask → Morphology → Contours
```

**Step 1 — Channel Decomposition:**
```python
b, g, r = cv2.split(frame)
```

**Step 2 — Compound Inequality Mask:**
```python
mask = ((r > r_thresh) &
        (r > b + diff_thresh) &
        (r > g + diff_thresh)).astype(np.uint8) * 255
```

This imposes three simultaneous conditions:
1. `r > 100` — absolute floor on red amplitude
2. `r > b + 30` — red must dominate blue by at least 30 DN (digital numbers)
3. `r > g + 30` — red must dominate green by at least 30 DN

**What does this compute?** Condition 2+3 implement *relative chrominance* — a proxy for hue without the nonlinear transformation. In normalized terms, if $r' = R / (R+G+B)$, a high $r'$ with low $g'$ and $b'$ corresponds to a high-H and high-S region in HSV. The subtraction threshold is a linearized approximation of the full HSV computation.

> **[IMAGE: Show the three channel images (B, G, R grayscale) side by side, with the laser dot visible as bright in R and dim in B and G. Then show the resulting binary mask.]*

---

## SLIDE 8 — HSV vs. BGR: Tradeoffs

| Property | HSV Thresholding | BGR Thresholding |
|---|---|---|
| **Illumination robustness** | High — S and V decouple hue from brightness | Moderate — absolute thresholds are illumination-sensitive |
| **Computational cost** | Higher — nonlinear transform required first | Lower — pure arithmetic on raw channels |
| **Expressiveness** | Full perceptual color control (3 independent axes) | Approximate; ratio comparison is implicit |
| **Red-wrap handling** | Explicit dual-range OR | Not needed (direct R channel comparison) |
| **Sensitivity tuning** | Intuitive via `h_min/h_max`, `s_min`, `v_min` | Less intuitive — `r_thresh` and `diff_thresh` interact |
| **White light rejection** | Excellent (S floor rejects unsaturated whites) | Weaker (white has R ≈ G ≈ B so `r > g+30` may fail under bright white light) |
| **Deep red / near-IR** | Can miss if V is low (dark reds) | Can detect dark reds if R is still dominant |

**Key insight:** HSV is the better choice when scene illumination varies significantly (outdoor, mixed lighting). BGR is an acceptable low-latency fallback when the laser is the dominant red source in a controlled environment.

**Both methods share a final cross-verification step:**
```python
is_red_bgr = (r > b + 30) and (r > g + 30) and (r > 100)
```
After contour detection, each candidate centroid is re-verified in BGR space. This fusion step reduces false positives from HSV matching on spectrally ambiguous objects.

> **[IMAGE/DATA: ROC curve or precision-recall curve showing HSV vs. BGR under varying ambient light conditions — if you can generate this from the detector logs.]*

---

## SLIDE 9 — Method 2: Temporal Frame Differencing with Laser Pulsing

**Core Idea:** Instead of discriminating the laser by color, discriminate it by *when it appears*. The laser controller (PCA9685 PWM) synchronously turns the laser ON and OFF, and the camera captures a frame at each state. The only pixels that change between the two frames are those illuminated by the laser.

**Pipeline:**

```
Laser ON → Capture frame_on → Laser OFF → wait 1ms → Capture frame_off
→ absdiff(frame_on, frame_off) → grayscale → binary threshold → Morphology → Contours
```

**Step 1 — Synchronized Acquisition:**
```python
ret_on, frame_on = self.camera_capture.read()     # laser on
self.laser_controller1.off()
self.laser_controller2.off()
time.sleep(0.001)                                  # wait for LED off
ret_off, frame_off = self.camera_capture.read()    # laser off
```

The `1ms` wait corresponds to the PCA9685 rise/fall time at 1kHz PWM frequency. The system is effectively implementing a *lock-in amplifier* concept in software: modulate the signal source, demodulate by differencing.

**Step 2 — Temporal Differencing:**
```python
diff = cv2.absdiff(frame_on, frame_off)
```

`cv2.absdiff` computes $|F_{on}(x,y) - F_{off}(x,y)|$ per channel. Since ambient lighting, background objects, and static reflections are identical in both frames, they cancel:

$$\Delta(x,y) = |I_{laser}(x,y) + I_{bg}(x,y) - I_{bg}(x,y)| = I_{laser}(x,y)$$

What survives the difference is *only* the laser contribution.

**Step 3 — Grayscale Collapse and Threshold:**
```python
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)
```

The BGR difference image is collapsed to a single channel (summing across R, G, B luminance weights), then hard-thresholded at 50 DN. This threshold sets the minimum laser-induced intensity change to register as a detection — a guard band against camera read noise and micro-motion between frames.

> **[IMAGE: Three-panel visualization: frame_on | frame_off | |frame_on - frame_off| (temporal difference). The laser dot should be the only bright region in the difference image.]*

---

## SLIDE 10 — Temporal Method: Signal Processing Interpretation

**This is equivalent to a synchronous demodulation / lock-in detection scheme:**

In classical lock-in amplifiers, a signal modulated at frequency $f_0$ is recovered by multiplying with a reference at $f_0$ and low-pass filtering. Here:
- The laser is modulated at ½ the frame rate (on for one frame, off for the next)
- `absdiff` acts as the demodulation product (unsigned correlation)
- The binary threshold is the low-pass decision stage

**Noise sources that temporal differencing suppresses:**
| Noise Source | HSV Handles? | Temporal Handles? |
|---|---|---|
| Red-colored objects (static) | Partially (HSV thresholds) | ✓ Yes — static, cancels in diff |
| Red LEDs / neon signs (static) | Partially | ✓ Yes — static, cancels |
| Ambient intensity variation | Partially (V threshold) | ✓ Yes — both frames equally affected |
| Camera rolling shutter / motion blur | N/A | ✗ No — inter-frame motion creates artifacts |
| Camera read noise | N/A | ✗ No — adds in quadrature |

**Temporal differencing tradeoff:**
- **Pro:** Near-perfect rejection of colored background objects — this is the single most discriminative method if the scene contains red-wavelength confounders
- **Con:** Latency doubles (two frame captures per detection cycle); requires hardware laser control (PCA9685 over I2C); susceptible to inter-frame scene motion producing false positives

**Effective SNR improvement:**

If background intensity is $I_{bg}$ with variance $\sigma_{bg}^2$, and laser intensity is $I_L$:
- Without differencing: $\text{SNR} = I_L / \sigma_{bg}$
- With differencing (assuming independent frames): $\text{SNR} = I_L / \sqrt{2}\,\sigma_{read}$ where $\sigma_{read}$ is camera read noise (typically much smaller than $\sigma_{bg}$)

> **[DATA: Plot SNR vs. ambient background brightness for HSV method vs. temporal method — theoretical curves showing where temporal becomes necessary.]*

---

## SLIDE 11 — Morphological Filtering: Nonlinear Noise Suppression

After any of the above thresholding methods, the binary mask is refined using **mathematical morphology**:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

**Structuring element:** A 5×5 elliptical disk — chosen because the laser dot has a circular cross-section (Gaussian beam profile). Using a matched shape SE improves morphological precision.

**MORPH_OPEN = Erosion → Dilation:**

Erosion contracts all white regions by the SE radius:

$$E(x,y) = \min_{(u,v) \in K} M(x+u, y+v)$$

This *eliminates* any white blob smaller than the SE (isolated noise pixels). Dilation then restores surviving blobs to their original size:

$$D(x,y) = \max_{(u,v) \in K} M(x+u, y+v)$$

Net effect: small noise specks (< 5px) are removed; larger blobs (laser dot) survive intact.

**MORPH_CLOSE = Dilation → Erosion:**

Performed after OPEN. Dilation bridges small gaps between white regions; erosion shrinks them back. Net effect: fills small holes in the laser spot mask (caused by saturation bloom or non-uniform illumination) without growing the overall blob.

**Signal processing analogy:**
- OPEN is analogous to a nonlinear high-pass filter (removes low-frequency spatial impulses, i.e., small blobs)
- CLOSE is analogous to a nonlinear low-pass filter (fills discontinuities in large blobs)
- Together: bandpass in object-size space, tuned by the SE size

> **[IMAGE: Three-panel mask visualization: (1) Raw threshold mask with noise, (2) After OPEN (noise removed), (3) After CLOSE (holes filled). Circle the noise pixels in panel 1 and show they're gone in panel 2.]*

---

## SLIDE 12 — Contour Analysis: From Binary Mask to Spatial Detections

Once the mask is clean, **contour analysis** extracts geometric descriptors for each candidate region:

```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

**`cv2.findContours`** implements a border-following algorithm (Suzuki & Abe, 1985). It traces the boundary of each connected white region in the binary mask:
- `RETR_EXTERNAL` — returns only the outermost contour of each blob (no nested hierarchy), which is appropriate since the laser dot is a simple filled region
- `CHAIN_APPROX_SIMPLE` — compresses collinear boundary segments to their endpoints, reducing memory footprint

**Per-contour feature extraction:**

**1. Area filter:**
```python
area = cv2.contourArea(contour)
if area < 1 or area > 200:
    continue
```
This is a *spatial bandwidth filter* — laser dots occupy 1–200 pixels based on distance and optics. Objects outside this range (large red surfaces, sub-pixel noise) are rejected. The upper bound of 200 px² corresponds roughly to a ~16×16 pixel circle, which would represent an extremely close or bloomed laser dot.

**2. Centroid via Image Moments:**
```python
M = cv2.moments(contour)
cx = M['m10'] / M['m00']
cy = M['m01'] / M['m00']
```

Image moments are the 2D analog of statistical moments. For a binary blob:
$$m_{pq} = \sum_{x}\sum_{y} x^p y^q \cdot M(x,y)$$

- $m_{00}$ = total white area (zeroth moment = mass)
- $m_{10}/m_{00}$ = centroid x (first moment / area = center of mass)
- $m_{01}/m_{00}$ = centroid y

The centroid gives sub-pixel accurate localization — more precise than the bounding box center, because it weights by pixel intensity within the blob.

**3. Circularity as Confidence Score:**
```python
perimeter = cv2.arcLength(contour, True)
circularity = (4 * np.pi * area) / (perimeter ** 2)
confidence = min(1.0, circularity)
```

The **isoperimetric quotient** (circularity):

$$C = \frac{4\pi \cdot A}{P^2}$$

For a perfect circle: $C = 1.0$. For any other shape: $C < 1.0$. A laser dot, projected through a circular aperture, produces a near-circular Gaussian-profile blob on the sensor. So $C \approx 1.0$ is a strong discriminator for the laser vs. elongated reflections, edge artifacts, or irregular red regions.

The threshold `confidence >= 0.5` means the blob must be at least 50% as "round" as a perfect circle to pass.

**4. Final BGR Cross-Verification:**
```python
b, g, r = frame[cy_int, cx_int]
is_red_bgr = (r > b + 30) and (r > g + 30) and (r > 100)
```

After all spatial filters, the centroid pixel is re-sampled in BGR space as a final gate. This catches cases where a morphologically circular object (e.g., a round orange blob) passes circularity but fails the spectral test.

> **[IMAGE: Annotated mask with contours drawn in green. Label: (1) a contour that passes area+circularity filters (laser), (2) a contour that fails area filter (large red surface), (3) a contour that fails circularity (elongated reflection). Use different colors.]*

> **[DATA: Scatter plot of (area, circularity) for all detected contours across N frames. Show the laser cluster at high circularity / small area, separated from noise clusters.]*

---

## SLIDE 13 — Camera Exposure: Engineering the Sensor SNR

**Why does exposure matter?**

A camera sensor integrates photon flux over the exposure time $T$:

$$N_{photons} = \Phi \cdot T \cdot A_{pixel}$$

where $\Phi$ is irradiance (photons/s/m²) and $A_{pixel}$ is the pixel collection area.

**The laser is a coherent, collimated, near-monochromatic source.** Its irradiance at the sensor is orders of magnitude higher than diffuse ambient light per unit solid angle. However, at long exposures, ambient light accumulates until background pixels saturate — washing out the laser-vs-background contrast.

**Reducing exposure time has three effects:**

1. **Background suppression:** Diffuse ambient light (low $\Phi_{ambient}$) accumulates fewer photons → lower DN in background pixels → better contrast for the laser

2. **Laser preservation (relative):** The laser (high $\Phi_{laser}$) still dominates even at short exposure because $\Phi_{laser} \gg \Phi_{ambient}$. Even if the laser dot is no longer fully saturated, it remains the brightest region.

3. **SNR improvement:**

$$\text{SNR} = \frac{I_{laser} - I_{bg}}{\sigma_{noise}} = \frac{(\Phi_{laser} - \Phi_{ambient}) \cdot T}{\sqrt{\sigma_{read}^2 + N_{bg}}}$$

At short $T$, $I_{bg} \to 0$ (photon shot noise of background drops), read noise $\sigma_{read}$ dominates — but for a bright laser, this is favorable.

4. **Reduced motion blur:** Shorter exposure freezes both the laser dot and any background motion, preventing the laser spot from smearing across multiple pixels (which would reduce circularity and fail the shape filter).

**Practical implication for threshold tuning:**

With very short exposure, `hsv_v_min` (Value floor) should be *lowered* — the overall scene brightness drops, and even the laser might not reach maximum V=255. Conversely, `r_thresh` in BGR should be tuned relative to scene brightness. This is why the codebase logs `center_bgr` and `center_hsv` every 30 frames — to guide empirical threshold adjustment.

> **[IMAGE: Side-by-side comparison of the same scene at two exposures: (1) long exposure — background bright, laser barely distinguishable; (2) short exposure — background dark, laser clearly visible as bright point. If possible, show histograms of the red channel below each image to illustrate the shift in pixel amplitude distribution.]*

> **[DATA: Plot detection confidence score vs. camera exposure time (sweep from 100µs to 10ms). Show the sweet spot where confidence is maximized.]*

---

## SLIDE 14 — System Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LaserEyes Pipeline                              │
│                                                                         │
│  Hardware Layer                                                         │
│  ┌─────────────┐   I2C    ┌──────────────┐   PWM   ┌──────────────┐   │
│  │ Raspberry Pi ├─────────┤   PCA9685    ├─────────┤  Laser Diode │   │
│  │   (Host)    │         │  (PWM ctrl)  │         │  650nm CW/   │   │
│  └──────┬──────┘         └──────────────┘         │  pulsed      │   │
│         │                                          └──────────────┘   │
│         │ Camera (BGR frames)                                          │
│  Signal Acquisition Layer                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Frame Buffer → 2D discrete signal I(x,y) ∈ [0,255]^3           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Detection Layer (select one method)                                   │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────────────────┐   │
│  │  HSV Method   │  │  BGR Method   │  │   Temporal Method        │   │
│  │  (color band  │  │  (ratio       │  │   (absdiff + lock-in)    │   │
│  │   thresholding│  │   thresholding│  │                          │   │
│  └───────┬───────┘  └───────┬───────┘  └────────────┬─────────────┘   │
│          └──────────────────┴──────────────────────  │                 │
│                                                       ▼                 │
│  Post-Processing Layer                                                  │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────────────┐     │
│  │ Morphological│→ │ Contour Finding  │→ │ Feature Extraction    │     │
│  │ Filtering    │  │ (Suzuki & Abe)   │  │ Area, Centroid (M),   │     │
│  │ OPEN + CLOSE │  │                  │  │ Circularity, BGR gate │     │
│  └──────────────┘  └─────────────────┘  └───────────────────────┘     │
│                                                       │                 │
│  Output Layer                                         ▼                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  [{x, y, width, height, confidence, class}]  → Servo / Control  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

> **[IMAGE: Replace ASCII diagram with a polished block diagram. Add a real camera frame in the background to make it visually grounding.]*

---

## SLIDE 15 — Conclusions and Takeaways

**What we've seen:**

| Concept | Classical SP Analogue |
|---|---|
| BGR→HSV color transform | Coordinate transform to decorrelated feature space |
| Dual-band hue threshold | Bandpass filter in hue dimension |
| Temporal differencing | Synchronous demodulation / lock-in amplifier |
| Morphological OPEN/CLOSE | Nonlinear size-selective spatial filtering |
| Circularity score | Shape-matched template confidence |
| Camera exposure control | Integration time tuning for SNR optimization |

**Key insight:** No single method is universally optimal. The hybrid dispatcher (`_detect_hybrid`) uses HSV as a fast path and falls back to adaptive thresholding when confidence is low — a practical example of *cascaded detectors* balancing throughput and robustness.

**Extensions for future work:**
- Kalman filtering on the centroid trajectory for temporal smoothing
- Adaptive threshold adjustment based on rolling mean background brightness
- Frequency-domain modulation of laser (e.g., 30 Hz square wave) and matched filter demodulation for even higher SNR in noisy scenes

> **[DATA: Final slide — show a confusion matrix or detection rate table across the three methods under different ambient light conditions (dark room, office, outdoor). Makes the tradeoffs concrete.]*

---

*Report generated for graduate presentation in "Analyzing Sensor Data".*
*Source: `lasereyes/src/laser_detector.py`*
