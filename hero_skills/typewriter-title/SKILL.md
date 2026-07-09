---
name: typewriter-title
description: Typewriter text effect that types out a heading character-by-character with a blinking caret, then optionally types a subtitle. Use when the user wants a "typewriter effect", "typing animation", "打字機標題", "逐字浮現的標題", or an animated hero headline.
---

# Typewriter Title (打字機標題)

Types a title one character at a time with a blinking cursor, then types a subtitle.

## How it works
- A recursive `setTimeout` appends one more character every ~130ms (title) / ~26ms (subtitle).
- A caret `<span class="cur">` blinks via CSS `@keyframes`; hide it when the title finishes, then start the subtitle.

Tune the per-character delays. Works with CJK and Latin. See `example.html`.
