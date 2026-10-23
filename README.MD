# OpenCV Notes Vision

Preprocessing pipeline that turns phone photos of handwritten notes into clean,
flat page images, before they are sent to an AI model to generate flashcards
for [Brainmaxx](https://github.com/studiousdoston/Personal_Study_App-).

## Step 4: Page detection + perspective flattening

![Page detection: notebook on textured desk](docs/showcase/step4_edge-notes.jpg)

**How it works:** Otsu thresholding separates the bright page from the desk,
morphology cleans the mask, a convex hull finds 4 corners, and a perspective
transform flattens the full-resolution image.

**Problems solved along the way:**

- **Textured desk** (carbon-fiber pattern): Canny edge detection picked up the
  texture, so I switched to thresholding, which ignores it.
- **Spiral binding:** the rings made the outline jagged; a convex hull smooths
  over them before finding the corners.
- **False positives on cropped images:** already-cropped scans were being
  "detected" and partly cut off; fixed by requiring the page not to touch the
  image border.

## Step 5: Shadow removal + contrast

![Shadow removal: notebook photo](docs/showcase/step5_edge-notes.jpg)
![Shadow removal: grid paper](docs/showcase/step5_note.jpg)
![Shadow removal: lined notes](docs/showcase/step5_notes3.jpg)

**How it works:** the paper background is estimated by "erasing" the ink
(dilate + median blur on a quarter-size copy), then each color channel is
divided by it. Shadows cancel out, the paper becomes evenly white, and ink keeps
its color. A gamma curve then darkens faint ink without greying the paper.

**Design decision:** I first planned CLAHE (adaptive histogram equalization),
but for documents it also boosts paper texture and grid lines. Background
division only removes lighting differences, so grid lines stay light and colors
are preserved, which the planned highlighter-detection feature needs.

## Step 6: Resize + compress for the AI model

Pages are resized to fit Claude's standard-tier vision limits
(1568px long edge **and** 1568 visual tokens, where tokens = ⌈w/28⌉ × ⌈h/28⌉),
then encoded as JPEG quality 85.

![Token and file size savings](docs/showcase/step6_savings.png)

| Metric (8 sample pages) | Before   | After    | Saved   |
| ----------------------- | -------- | -------- | ------- |
| Visual tokens\*         | 25,088   | 10,621   | **58%** |
| File size               | 5,015 KB | 2,428 KB | **52%** |

\*"Before" = tokens a high-resolution model would charge for the full-size image.

**Lesson learned:** my first version only limited the long edge, which still
produced ~2,300-token pages. Claude's standard tier caps tokens too, so the
resize now shrinks until both limits fit.
