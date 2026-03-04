# PixelPaws Hardware

This document describes all hardware components used in the PixelPaws filming enclosure, including cameras, lenses, lighting, power supplies, acrylic panels, and 3D printing materials.

---

## 📦 Bill of Materials

| Component | Manufacturer | Model / Part Number | Key Specs | Link |
| --- | --- | --- | --- | --- |
| **Camera** | e-con Systems | `See3CAM_CU27` · SKU: `CHLCC_BX_H02R1` | Sony STARVIS IMX462 sensor · Full HD (1080p) · USB 3.1 Gen 1 (USB-C) · 100 fps MJPEG / 60 fps UYVY · 0 lux low-light · M12 lens mount · UVC compliant (no drivers) · 5V, max 2.12W | [Product Page](https://www.e-consystems.com/usb-cameras/sony-starvis-imx462-ultra-low-light-camera.asp) |
| **Lens** | Marshall Electronics | `CV-2812-3MP` | M12 (S-mount) · 2.8–12 mm varifocal · f/1.4 · 3 MP · IR corrected · 109°–31.2° horizontal FOV · 1/2.7" sensor format · Manual focus + zoom with lock screws | [B&H Product Page](https://www.bhphotovideo.com/c/product/1428706-REG/marshall_electronics_cv_2812_3mp_m12_2_8_12mm_f_1_4_3mp.html) |
| **LED Strip (IR)** | — | SMD5050 850 nm IR strip | 850 nm infrared · SMD 5050 tri-chip · 12V DC · 60 LEDs/m · 14.4 W/m · 120° beam angle · Non-waterproof (IP20) · 3-LED cuttable · 5 m/roll · 50,000 hr lifespan | [Amazon](https://www.amazon.com/dp/B0FC2GJTR4) |
| **LED Power Supply** | — | 12V 2A LED Adapter (2-pack) | Input: 100–240V AC · Output: 12V DC @ 2A · 24W max · 5.5/2.1 mm DC barrel connector · Non-dimmable | [Amazon](https://www.amazon.com/dp/B08Y6VHMHS) |
| **Acrylic Sheet (Colored)** | TAP Plastics | Chemcast Cast Acrylic — Color | 110 mm width · 4 sheets per order · Tops are 3D printed (see below) · Cut-to-size · Glossy finish · UV stable | [TAP Plastics](https://www.tapplastics.com/product/plastics/cut_to_size_plastic/acrylic_sheets_color/341) |
| **Acrylic Sheet (Clear Cast)** | TAP Plastics | Chemcast Cast Acrylic — Clear | 189 mm width · 1 sheet per order · Cut-to-size · Optical clarity · UV stable · Non-yellowing | [TAP Plastics](https://www.tapplastics.com/product/plastics/cut_to_size_plastic/acrylic_sheets_cast_clear/510) |
| **3D Printer Filament** | eSUN | PLA+ 1.75 mm — Black | 1.75 mm diameter · Black · 1 kg (2.2 lbs) spool · PLA+ formulation | [Amazon](https://www.amazon.com/eSUN-1-75mm-Printer-Filament-2-2lbs/dp/B01EKEMDA6/) |

---

## 💻 Video Acquisition (Host Computer)

Running 3–4 cameras simultaneously at 60 fps places real demands on your host system — primarily USB controller bandwidth and GPU decode throughput. Here's what to keep in mind.

### Why It Matters

Each `See3CAM_CU27` camera running at **1080p 60 fps MJPEG** streams compressed data over USB 3.1 Gen 1 (~5 Gbps theoretical). In practice:

- A single USB host controller can comfortably handle **1–2 cameras** before bandwidth contention causes frame drops.
- Plugging all cameras into a hub or the same USB root hub is the most common cause of dropped frames — even if individual ports look like USB 3.0.
- For 3–4 cameras, you need **multiple independent USB host controllers**, either natively on the motherboard or via a PCIe expansion card.

> **Key rule:** One camera per USB host controller for reliable 60 fps multi-camera capture.

---

### Recommended Host Configurations

#### ✅ Option 1: Gaming or Creator Laptop (Current / Recommended for Portability)

A modern laptop with a dedicated **Nvidia GPU** (GTX 1060 or newer, RTX series preferred) works well for 3–4 cameras if the machine has multiple USB controllers. Look for laptops with **Thunderbolt/USB4 ports** in addition to standard USB-A — these typically map to separate root hubs.

- **GPU:** Nvidia GTX 1060+ or RTX series (CUDA-accelerated MJPEG decode reduces CPU load significantly)
- **RAM:** 16 GB minimum, 32 GB recommended for 4 simultaneous streams
- **USB:** Confirm multiple root hubs via Device Manager (Windows) or `lsusb -t` (Linux) before purchasing

#### ✅ Option 2: Desktop PC with PCIe USB Expansion Card (Best for 4 Cameras)

A desktop gives you the most flexibility. Add a dedicated USB 3.2 PCIe card with multiple independent controllers to guarantee bandwidth isolation per camera.

Recommended cards:
- **Inateck KT4005 / KTU3FR-4P** — 4 ports, each on its own controller (no shared bandwidth)
- **StarTech PEXUSB3S44V** — 4-port USB 3.0 PCIe, VIA VL805 chip, widely supported on Linux

Pair with an **Nvidia RTX 3060 or better** for comfortable 4-camera MJPEG decode at 60 fps.

#### ✅ Option 3: Mini PC / NUC (Compact Dedicated Capture Station)

For a dedicated always-on capture station, a compact machine like an **Intel NUC 13 Pro** or **ASUS NUC 14** works well. These have Thunderbolt 4 and multiple USB controllers in a small footprint. Add an external Thunderbolt dock with extra USB controllers if needed.

---

### Minimum System Requirements (3–4 Cameras @ 60 fps)

| Component | Minimum | Recommended |
| --- | --- | --- |
| **CPU** | Intel Core i5 (10th gen+) / Ryzen 5 | Intel Core i7 / Ryzen 7 or better |
| **GPU** | Nvidia GTX 1060 6 GB | Nvidia RTX 3060 or better |
| **RAM** | 16 GB | 32 GB |
| **USB** | 2+ independent USB 3.0 host controllers | 4 independent USB 3.1 Gen 1 controllers |
| **OS** | Windows 10/11, Ubuntu 20.04+ | Windows 11 or Ubuntu 22.04 LTS |
| **Storage** | SSD (for writing video) | NVMe SSD, 500 GB+ |

---

### USB Bandwidth Tips

- **Never daisy-chain cameras through a passive USB hub.** A hub shares one upstream port across all devices.
- If you must use a hub, use an **externally powered USB 3.0 hub** and limit it to **one camera per hub**.
- On Windows, use **Device Manager → Universal Serial Bus controllers** to inspect root hubs and confirm cameras are spread across them.
- On Linux, run `lsusb -t` to see the bus topology and verify each camera is on a different bus.

---

## 🪟 Acrylic Panels

Acrylic sheets are ordered cut-to-size from [TAP Plastics](https://www.tapplastics.com). Cut tolerance is ±1/32". Cut-to-size orders typically ship within 1–2 business days.

### Colored Acrylic (Box Sides)

- **Source:** [TAP Plastics — Acrylic Sheets Color](https://www.tapplastics.com/product/plastics/cut_to_size_plastic/acrylic_sheets_color/341)
- **Material:** Chemcast® Cast Acrylic (opaque/translucent color options)
- **Width:** 110 mm
- **Quantity:** 4 sheets per box
- **Notes:** Box tops are 3D printed — see [3D Printed Enclosure](#️-3d-printed-enclosure) section below.

### Clear Cast Acrylic (Viewing Panel)

- **Source:** [TAP Plastics — Cast Clear Acrylic](https://www.tapplastics.com/product/plastics/cut_to_size_plastic/acrylic_sheets_cast_clear/510)
- **Material:** Chemcast® Cell Cast Clear Acrylic
- **Width:** 189 mm
- **Quantity:** 1 sheet per box
- **Notes:** Optical clarity with 92% light transmission. UV stable and non-yellowing.

---

## 🖨️ 3D Printed Enclosure

The filming box consists of 3D-printed parts. STL files are in [`hardware/stl/`](/rslivicki/PixelPaws/blob/master/hardware/stl).

| File | Description |
| --- | --- |
| `Bottom_Chamber_with_Camera_holder.stl` | Lower chamber housing the camera module and LED strips |
| `Top_of_filming_box_original.stl` | Top lid / cover of the filming enclosure (also used as box tops for colored acrylic panels) |

> **Recommended print settings:** Use eSUN PLA+ 1.75 mm Black filament · 0.2 mm layer height · 20–30% infill · Supports as needed for camera holder geometry.

> **Filament:** [eSUN PLA+ 1.75 mm Black — Amazon](https://www.amazon.com/eSUN-1-75mm-Printer-Filament-2-2lbs/dp/B01EKEMDA6/)

---

## ⚡ Wiring Notes

* The LED strips run on **12V DC**. Connect them to the included 5.5/2.1 mm barrel adapter from the power supply.
* The camera connects to the host via **USB-C (USB 3.1 Gen 1)**. USB 2.0 backward compatible.
* Do **not** connect LED strips directly to AC mains. Always use the 12V adapter.
* LED strips are cuttable every **3 LEDs (~50 mm)**. Cut only along marked cut lines.
* The 850 nm IR LEDs emit **invisible light** — do not use for visible accent lighting.

---

## 📐 Camera Specifications (Detail)

| Parameter | Value |
| --- | --- |
| Sensor | Sony STARVIS IMX462LQR |
| Optical Format | 1/2.8″ |
| Resolution | 1937 × 1097 (Full HD) |
| Pixel Size | 2.9 µm × 2.9 µm |
| Shutter | Electronic Rolling Shutter |
| Frame Rate | MJPEG: 100 fps @ 1080p · UYVY: 60 fps @ 1080p |
| Lens Mount | M12 (S-mount) |
| Interface | USB 3.1 Gen 1, Type-C connector |
| OS Support | Windows, Linux, Android\*, macOS\*\* |
| Operating Voltage | 5V ± 5% |
| Power | Max 2.12W / Min 0.85W |
| Operating Temp | −30°C to 60°C |

---

## 🔭 Lens Specifications (Detail)

| Parameter | Value |
| --- | --- |
| Mount | M12 (S-mount) |
| Focal Length | 2.8–12 mm (varifocal) |
| Aperture | f/1.4 (fixed iris) |
| Sensor Compatibility | 1/2.7″, 3 MP |
| Horizontal FOV | 109° (wide) – 31.2° (tele) |
| Back Focal Length | 6.9–14.4 mm |
| IR Correction | Yes |
| Focus / Zoom | Manual with lock screws |
