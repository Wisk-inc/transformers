# Pinxit Animate

Ask for an animation in words. An open video model imagines how it moves. Then Pinxit draws it by hand, one
frame after the other, on a whiteboard: the lines with a pen, the colour with markers, each frame traced over
the last one on a lightbox, so the character keeps its look from frame to frame.

```bash
pip install "helix-lm[paint]"
```

```python
from helix_lm.paint.animate import animate

clip = animate("a cute cartoon fox running in the snow, side view")
clip.save("fox.mp4")                      # the animation
clip.making_of("fox_being_drawn.mp4")     # the whiteboard, the pen in hand, every frame drawn
clip.save("fox.pinxa")                    # everything, to change or carry on later
```

```bash
helix paint animate "a cute cartoon fox running in the snow" --out fox.mp4 --making-of drawn.mp4 --keep fox.pinxa
helix paint animate --video my_clip.mp4 --lines "brush pen" --out drawn.mp4     # draw a video of your own
```

On a large GPU the cell [notebooks/pinxit_animate_cell.py](../notebooks/pinxit_animate_cell.py) does all of it.

## 1. Who does what

| part | what it does |
|---|---|
| the video model | imagines the motion from the words: what the fox looks like, how it runs, frame by frame |
| Pinxit's eye | finds in each frame the lines a person would draw and the shapes to colour |
| Pinxit's hand | draws every line with a pen and colours every shape with markers, in order, at a hand's speed |
| the lightbox | carries each drawing onto the next frame, so lines that did not change are traced, not redrawn |

To be plain about the "no noise" part: the video model is a diffusion model, so the *idea* of the animation
comes from a model that works by removing noise. Nothing on the board does. Every mark on every frame is a
pen line or a marker stroke with a position, a pressure and a time, drawn in the order a person draws.

## 2. The video models

| name | model | licence | size | where it runs |
|---|---|---|---|---|
| `animatelcm` (default without a large GPU) | AnimateLCM motion module on DreamShaper 8 | AnimateLCM: no licence on its model card; DreamShaper 8: CreativeML OpenRAIL-M | about 3 GB | a CPU, slowly |
| `animatediff` | AnimateDiff v1.5-3 motion module on DreamShaper 8 | AnimateDiff: none on the card; DreamShaper 8 as above | about 3.8 GB | a CPU or GPU |
| `wan` (default on a GPU with 40 GB or more) | Wan 2.2 TI2V 5B | Apache-2.0 | about 34 GB | a large GPU |
| `wan-1.3b` | Wan 2.1 T2V 1.3B | Apache-2.0 | about 29 GB | a GPU |
| any other name | a `diffusers` text-to-video pipeline from the Hub | its own | | |

Check a model's licence before using what it makes commercially. AnimateLCM's and AnimateDiff's model cards do
not name one.

**Fine-tuning.** Each video model takes a LoRA: `load_animator("animatelcm", lora="you/your-style-lora")` or
`--lora` on the command line. A LoRA trained on a character keeps that character the same across clips; one
trained on a drawing style changes what is imagined. Training a LoRA is not part of this release: use the
training scripts of `diffusers` for the model you picked, then load the result here.

Words are given to the model with "2d cartoon animation, clean bold outlines, flat colours, simple shapes,
plain background" added, because clean shapes are what a pen and markers draw well. `drawable=False` turns
that off.

## 3. Drawing one frame

1. The frame is seen in planes (small differences inside a shape levelled out, borders kept).
2. The borders are thinned to one pixel and followed like a pen follows them: from loose ends first, straight
   on through crossings, broken where an arm would lift. Stronger borders are pressed harder.
3. The markers are chosen once for the whole animation, from all its frames: up to ten colours, white left as
   bare board.
4. Each shape is coloured with its marker back and forth, the nib turning just inside the border. Narrow parts
   get the fine nib. Parts clearly darker than their marker are gone over again across the first layer.
5. Order: ink lines first (the long ones first, then on to whichever starts nearest), then one marker at a
   time from the lightest.

The tools:

| tool | |
|---|---|
| `pen` | a fineliner: even, a little thinner where it lands and lifts |
| `brush pen` | thin to thick with pressure |
| `pencil` | grey, broken by the paper's tooth |
| `marker`, `fine marker` | see-through colour; strokes of one marker blend while wet, a later layer darkens it, a little ink pools where the nib rests |

The hand is not steady: every line wavers a little and may run a pixel past its end. A line traced onto the
next frame is drawn again with a new waver, which is the slight "boil" of drawn animation.

## 4. Frame by frame, and why the character stays the same

An animator lays the last drawing on the lightbox and a new sheet over it. Pinxit does the same:

- how each part of the picture moved is measured (optical flow, coarse to fine; plain areas such as an
  empty sky take their motion from the detail around them, and with none near they hold still);
- every line of the last drawing is moved with it. If it still lies on a line of the new frame it is
  traced: pulled onto that line and drawn again. If not, it is left off. Lines of the new frame that nothing
  covers are drawn new;
- colour the same way, shape by shape. A stroke that lands outside its shape, or that the move would tear,
  is left off, and what it leaves bare is coloured again;
- strokes that only go over what other strokes already drew are left out, so a drawing traced frame after
  frame does not gather layers;
- a frame that changed almost everywhere (a cut) is drawn from nothing.

Like most hand-drawn animation, a video of 16 frames a second or more is drawn "on twos": one drawing for
every two frames, so Wan's 24 frames a second become 12 drawings a second. `on=1` draws every frame.

So the character keeps its look three ways: the same markers colour every frame, most of its lines are traced
rather than drawn anew, and the video model itself keeps it steady within a clip. Between clips:
`clip.extend("the fox jumps over a log")` starts the next clip from the last frame (the video model is
given it as the start) and traces on from the last drawing with the same markers. `character=` gives a
picture of the character to start from; `like=` uses an earlier animation's markers.

## 5. The whiteboard video

`clip.making_of(path)` shows the drawing being made: a whiteboard in its frame, the pens and markers on the
ledge below, the one in use in hand, moving as a hand would (slower round curves, lifting between lines,
stopping to take another marker). Every frame is drawn over a faint copy of the last one, the lightbox. Lines
being traced go faster than new ones. The first frame takes about 14 seconds, every next one about 3, and the
finished animation plays twice at the end.

## 6. Memory and changes

A `.pinxa` file keeps every frame's lines, the markers and the frames the drawing was made from.

```python
from helix_lm.paint.animate import Animation

clip = Animation.load("fox.pinxa")
clip.recolor((0.2, 0.4, 0.9), at=(0.4, 0.6))   # the marker that colours that spot, blue in every frame at once
clip.retool("brush pen")                        # every line drawn with a brush pen instead
clip.extend("the fox stops and looks at the camera")
clip.save("fox.pinxa")
clip.save("fox.mp4")
```

## 7. What was checked, and what was not

Checked here, on a CPU:

- the drawing of single pictures by eye at full size, and tests for each part (pressure and taper, wet and dry
  marker layers, thinning and following lines, colouring up to a border, the order of drawing);
- the optical flow on known shifts (found to within 0.01 pixel) and on a cartoon scene (the ball's inside
  moves with it, the still sky and sun stay at 0);
- frame-by-frame drawing on a made-up clip (a ball rolling past a still sun): most lines traced, the same
  marker for the ball in every frame, the sun traced from the first frame to the last, a cut drawn anew;
- saving and opening, recolouring, retooling, the whiteboard video, `.mp4` and `.gif` out and back in;
- AnimateLCM run for real on the CPU (see the notes in the release for how long it took and how it looked).

Not checked:

- **Wan 2.2 and Wan 2.1 were not run**: they need a large GPU and about 30 GB of disk, which this machine does
  not have. The code calls them the way `diffusers` documents; the first real run will tell.
- **The video model decides how good the motion is.** On a CPU AnimateLCM is slow and its characters can
  change shape between frames; the lightbox then has less to trace and draws more anew.
- Lines come from the borders between shapes, not from an understanding of the drawing. A face drawn from a
  photo-like frame gets the lines of its shadows, not the lines an illustrator would choose.
- Training a LoRA for the video model is not built in.
