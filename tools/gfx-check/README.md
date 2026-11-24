# gfx-check

Runs `gfx_testbed` (`examples/graphics/gfx_testbed`) and gates what it wrote. One binary, so the same checks run
on any machine, Windows included.

```sh
cargo build --release --example gfx_testbed --features bevy_solari,area_light_luts
cargo build --release -p gfx-check
target/release/gfx-check run tools/gfx-check/smoke.ron
```

Every subcommand exits with 0 when its gate passes, 1 when it fails and 2 when it could not be evaluated.

| Command | What it does |
|---|---|
| `run <matrix.ron> [--out DIR] [--adapter NAME] [--testbed PATH] [--windowed]` | The testbed once per adapter × scene × resolution × flag set; `results.csv`, `results.md`, `results.json` and one folder per run. Each run must exit cleanly, pass the log gate and not be black, and keep under the limits its matrix entry sets |
| `image stats <png> [--min-luma X]` | Size and mean luma; fails on a black frame |
| `image diff <a> <b> [--max X]` | The metric between two images |
| `image resize <in> <out> --size WxH\|<image> --filter bilinear\|area` | Resizes a frame to the output's size: `bilinear` enlarges a lower-resolution run (an upscaler's bilinear baseline), `area` reduces a higher-resolution one (the supersampled baseline) |
| `flicker <run folder>` or `flicker <frame> <frame>...` `[--max X] [--pairs]` | The metric between consecutive frames of a still camera, over every pair: prints the mean and the largest pair, and `--max` gates the mean. A run folder gives its `still-*` captures (from its `report.json`, which must be on consecutive frames) or else `a.png` and `b.png` |
| `motion <orbit run> <still run> [--max X]` | The metric between an orbit run's capture and a still run at the same pose (`--pose K`) |
| `sheet <orbit run> [--out FILE]` | The orbit run's captures tiled into one image; `run` writes it for every orbit run |
| `budget <report.json> [--max-frame-ms X] [--max-pass NAME=Y]...` | Median frame time and median GPU time of a pass, from the run's report |
| `log-gate <run folder>` or `log-gate <log> [--traced]` | Fails on "panicked", "failed to process shader" and wgpu errors, and on "not loaded" when the run is traced. The `@invariant` validation warning of the stock deferred path passes |
| `calibration <results folder> [--out FILE]` | Writes `calibration.toml` from a run of `calibration.ron` |

**The metric** is one: the normalised root-mean-square error of the tonemapped 8-bit frames, in percent of full
scale ("points"). It is what `magick compare -metric RMSE` prints in parentheses, times 100. A number means
something only against another number from the same scene and resolution.

**On Linux `run` goes through `tools/gfx-headless.sh`** (no window shows, one GPU run at a time) and refuses to
start without it unless `--windowed` is given. On Windows there is no wrapper and it opens a window.

**DLSS** finds its libraries at run time beside the binary or in `$DLSS_SDK`: export it
(`eval "$(tools/fetch-sdks.sh dlss)"`) before `run` when a matrix uses `--dlss-rr` or `--dlss-sr`.

**Timings need a quiet machine.** Frame times are wall-clock: a build running beside the testbed inflates every
run that is not bound by the GPU.

**Resizing** works on the encoded 8-bit values, like the metric. `bilinear` interpolates between the four nearest
pixel centres with the edges repeated, as a GPU's linear sampler does; `area` averages the input area each output
pixel covers, partly covered pixels by their share (a 2x2 box at a ratio of 2). On D1's frames they match
`magick -filter Triangle -resize` within 0.012 points and `magick -filter Box -resize` exactly (2026-11-24).

**Flicker over one pair is noisy.** A temporal method jitters its camera over a cycle (TAA 8 frames; TSR and DLSS
SR `8 x upscale^2`, 32 at 50 %), and two frames differ more or less depending on where in the cycle they fall. Gate
on a sequence covering the longest cycle (`gfx_tsr --still-frames 33`). `run` and `calibration` still use `a.png`
and `b.png`, which is all the testbed writes.

## Files

- `smoke.ron`: every raw mode on both scenes, on both adapters of the project's machine. It must pass.
- `black.ron`: a frame with no light. It must fail.
- `calibration.ron` and `calibration.toml`: the error against the reference and the flicker of six ways of
  showing Solari's frame on `interior` at 2560x1440, with their noise floor. Later gates are relative to it.

## A matrix file

```ron
#![enable(implicit_some)]
(
    adapters: [(name: "NVIDIA", resolutions: ["1280x720"])],   // a substring of the adapter's name
    scenes: ["interior"],
    warmup: 30,          // frames after the scene is ready, before measuring
    frames: 60,          // measured frames
    seconds: 240,        // a run is stopped after this long
    runs: [
        (name: "reference", args: ["--mode", "reference"], frames: 600),
        // `reference` names an earlier run: this run's a.png against that run's is the "error" column.
        (name: "solari", args: ["--mode", "solari"], reference: "reference", max_error: 20.0),
    ],
)
```

A run may also set `warmup`, `adapters` (the adapter names it is limited to), `max_flicker` and `max_frame_ms`.
