---
name: dock-magnify
description: Add a macOS-style dock where items smoothly magnify (scale up and lift) based on horizontal cursor proximity. Use when the user wants a "macOS dock", "dock magnify", "hover magnify menu", "Dock 放大效果", or a fancy icon nav bar that grows under the mouse.
---

# macOS-style Dock Magnify

Items in a horizontal bar scale up as the cursor nears them, tapering off with distance.

## How it works
- On the container's `mousemove`, for each item compute `d = |cursorX − itemCenterX|`.
- Scale `s = max(1, 1.55 − d/150)` (1.55 = peak zoom, 150 = falloff px).
- Apply `transform: scale(s) translateY(−(s−1)*16px)` so items also lift as they grow.
- `transform-origin: bottom center` keeps them anchored to the bar; reset on `mouseleave`.

Tune peak (1.55) and falloff (150). See `example.html`.
