---
name: card-flip-details
description: A 3D card that flips on hover to reveal a back face (e.g. "View Details" + a subtitle). Use when the user wants a "flip card", "hover flip", "3D card reveal", "卡片翻轉", or image tiles that turn over to show a call-to-action on hover.
---

# 3D Flip Card (hover → View Details)

An image tile that rotates 180° on the Y axis on hover, revealing a gradient back face.

## How it works
- Wrapper `.card` sets `perspective`. Inner `.flip` uses `transform-style: preserve-3d` and transitions `transform`.
- Two `.face` layers with `backface-visibility:hidden`; the `.back` is pre-rotated `rotateY(180deg)`.
- On `.card:hover .flip { transform: rotateY(180deg); }` the tile turns to show the back.

Drop-in markup + CSS in `example.html`. Add the `on` class only when you want hover-flip enabled.
