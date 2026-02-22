# PixelPaws Hardware

This document describes all hardware components used in the PixelPaws filming enclosure, including cameras, lenses, lighting, and power supplies.

---

## 📦 Bill of Materials

| Component | Manufacturer | Model / Part Number | Key Specs | Link |
|-----------|-------------|---------------------|-----------|------|
| **Camera** | e-con Systems | `See3CAM_CU27` · SKU: `CHLCC_BX_H02R1` | Sony STARVIS IMX462 sensor · Full HD (1080p) · USB 3.1 Gen 1 (USB-C) · 100 fps MJPEG / 60 fps UYVY · 0 lux low-light · M12 lens mount · UVC compliant (no drivers) · 5V, max 2.12W | [Product Page](https://www.e-consystems.com/usb-cameras/sony-starvis-imx462-ultra-low-light-camera.asp) |
| **Lens** | Marshall Electronics | `CV-2812-3MP` | M12 (S-mount) · 2.8–12 mm varifocal · f/1.4 · 3 MP · IR corrected · 109°–31.2° horizontal FOV · 1/2.7" sensor format · Manual focus + zoom with lock screws | [B&H Product Page](https://www.bhphotovideo.com/c/product/1428706-REG/marshall_electronics_cv_2812_3mp_m12_2_8_12mm_f_1_4_3mp.html) |
| **LED Strip (IR)** | — | SMD5050 850 nm IR strip | 850 nm infrared · SMD 5050 tri-chip · 12V DC · 60 LEDs/m · 14.4 W/m · 120° beam angle · Non-waterproof (IP20) · 3-LED cuttable · 5 m/roll · 50,000 hr lifespan | [Amazon](https://www.amazon.com/dp/B0FC2GJTR4) |
| **LED Power Supply** | — | 12V 2A LED Adapter (2-pack) | Input: 100–240V AC · Output: 12V DC @ 2A · 24W max · 5.5/2.1 mm DC barrel connector · Non-dimmable | [Amazon](https://www.amazon.com/dp/B08Y6VHMHS) |

---

## 🖨️ 3D Printed Enclosure

The filming box consists of two 3D-printed parts. STL files are in [`hardware/stl/`](hardware/stl/).

| File | Description |
|------|-------------|
| `Bottom_Chamber_with_Camera_holder.stl` | Lower chamber housing the camera module and LED strips |
| `Top_of_filming_box_original.stl` | Top lid / cover of the filming enclosure |

> **Recommended print settings:** PLA or PETG, 0.2 mm layer height, 20–30% infill, supports as needed for camera holder geometry.

---

## ⚡ Wiring Notes

- The LED strips run on **12V DC**. Connect them to the included 5.5/2.1 mm barrel adapter from the power supply.
- The camera connects to the host via **USB-C (USB 3.1 Gen 1)**. USB 2.0 backward compatible.
- Do **not** connect LED strips directly to AC mains. Always use the 12V adapter.
- LED strips are cuttable every **3 LEDs (~50 mm)**. Cut only along marked cut lines.
- The 850 nm IR LEDs emit **invisible light** — do not use for visible accent lighting.

---

## 📐 Camera Specifications (Detail)

| Parameter | Value |
|-----------|-------|
| Sensor | Sony STARVIS IMX462LQR |
| Optical Format | 1/2.8″ |
| Resolution | 1937 × 1097 (Full HD) |
| Pixel Size | 2.9 µm × 2.9 µm |
| Shutter | Electronic Rolling Shutter |
| Frame Rate | MJPEG: 100 fps @ 1080p · UYVY: 60 fps @ 1080p |
| Lens Mount | M12 (S-mount) |
| Interface | USB 3.1 Gen 1, Type-C connector |
| OS Support | Windows, Linux, Android*, macOS** |
| Operating Voltage | 5V ± 5% |
| Power | Max 2.12W / Min 0.85W |
| Operating Temp | −30°C to 60°C |

---

## 🔭 Lens Specifications (Detail)

| Parameter | Value |
|-----------|-------|
| Mount | M12 (S-mount) |
| Focal Length | 2.8–12 mm (varifocal) |
| Aperture | f/1.4 (fixed iris) |
| Sensor Compatibility | 1/2.7″, 3 MP |
| Horizontal FOV | 109° (wide) – 31.2° (tele) |
| Back Focal Length | 6.9–14.4 mm |
| IR Correction | Yes |
| Focus / Zoom | Manual with lock screws |
