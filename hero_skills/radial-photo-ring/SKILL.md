---
name: radial-photo-ring
description: Build a hero/landing section that arranges images or cards in a rotating RADIAL ring around a centerpiece (logo/title), with an entrance animation (pile -> burst -> line up -> settle into a perfect circle). Each portrait card is rotated so its long axis points at the centre (radial "spokes"). Includes responsive scaling (RWD) and the no-overlap sizing math. Use when the user asks for a "photo circle", "radial ring", "cards orbit a logo", "放射狀照片圓環", "圍成正圓的照片動畫", or a 21st.dev-style scroll-morph hero.
---

# Radial Photo Ring (放射狀照片圓環)

Portrait image cards fly in and settle into a true circle around a centre element,
each card rotated to point at the centre (radial spokes). Fully responsive.

## Key ideas
- **True circle:** place card i at angle `a = i/N*2π − π/2`, position `(cos a·R, sin a·R)` with **equal** x/y radius `R = min(vw,vh)·Rfac` (Rfac≈0.38).
- **Radial rotation:** rotate each card by `i/N*360°` so a *portrait* card's long axis points at the centre. (A *landscape* base with the same rotation gives a *tangential* ring instead.)
- **RWD:** scale every card by `sc = clamp(min(vw,vh)/820, 0.42, 1)`. Because R and card size both scale with the viewport, the layout is self-similar → never overlaps or overflows on any screen.
- **No overlap:** radial cards crowd at their INNER corners. Safe count ≈ `floor(2π·(R − cardH/2) / (cardW + gap))`. Fewer cards ⇒ bigger cards. Reference-safe combos (portrait, ~0.8 aspect, credit line visible, phone→desktop): 12→102×128, 14→96×120, 16→83×104, 18→74×92.
- **Entrance timeline:** pile(0) → burst(80ms) → row(1500ms) → circle(3300ms).

See `example.html` for a drop-in, self-contained implementation. Swap `IMAGES`, tune `N` and `Rfac`.

## Optional: clockwise auto-rotation
Wrap the ring in a rotating container (or spin the `.stage`) after it settles:
`.stage.spin{animation:ringspin 46s linear infinite;transform-origin:center;}`
`.stage.spin:hover{animation-play-state:paused;}`  `@keyframes ringspin{to{transform:rotate(360deg);}}`
then `setTimeout(()=>stage.classList.add('spin'), settleMs);`. Positive rotation = clockwise; pause on hover keeps cards clickable.
