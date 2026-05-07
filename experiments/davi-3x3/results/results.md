# DAVI 3x3 — per-run results

_One section per run: train loss, eval corrected-macro trajectory, and per-V*-layer MAE at start / middle / end. Cross-run views (overlaid charts, beam capability bars, heatmap) live in `error_trajectories.html`._

## smoke

_Run dir: `experiments/davi-3x3/runs/20260506T203408Z_smoke`. Records: 126 total, 20 eval cycles._

### Train loss

- start: 0.1290
- end:   0.0121
- min:   0.0035 (step 2000)

### Eval corrected-macro trajectory

| step | corrected_macro | pred_mean | pred_std |
|-----:|-----:|-----:|-----:|
| 500 | 3.4776 | 0.016 | 0.312 |
| 1000 | 3.0396 | 0.590 | 0.272 |
| 1500 | 2.3892 | 1.310 | 0.300 |
| 2000 | 1.7939 | 2.007 | 0.425 |
| 2500 | 1.3411 | 2.677 | 0.638 |
| 3000 | 1.0908 | 3.195 | 0.948 |
| 3500 | 0.9982 | 3.587 | 1.234 |
| 4000 | 0.9371 | 3.921 | 1.426 |
| 4500 | 0.9076 | 4.129 | 1.575 |
| 5000 | 0.8181 | 4.215 | 1.601 |
| 5500 | 0.8051 | 4.166 | 1.576 |
| 6000 | 0.8054 | 4.199 | 1.564 |
| 6500 | 0.7966 | 4.348 | 1.707 |
| 7000 | 0.7503 | 4.371 | 1.690 |
| 7500 | 0.7340 | 4.362 | 1.664 |
| 8000 | 0.7488 | 4.392 | 1.732 |
| 8500 | 0.7728 | 4.410 | 1.759 |
| 9000 | 0.7582 | 4.420 | 1.748 |
| 9500 | 0.7867 | 4.368 | 1.715 |
| 10000 | 0.7490 | 4.364 | 1.686 |

### Per-V*-layer MAE — start / middle / end (bounded oracle, d=1..6)

| V* depth | start (step 500) | middle (step 5500) | end (step 10000) |
|------:|------:|------:|------:|
| 1 | 0.880 | 0.046 | 0.022 |
| 2 | 2.025 | 0.262 | 0.217 |
| 3 | 3.009 | 0.636 | 0.505 |
| 4 | 3.992 | 0.905 | 0.891 |
| 5 | 5.005 | 1.220 | 1.225 |
| 6 | 5.953 | 1.761 | 1.633 |

### Beam capability

_(Live solve_rate / avg_solve_len are not captured during 3x3 training. See `<run>/results/beam_eval_focused.json` and the Beam capability section of `error_trajectories.html`.)_

## full_train

_Run dir: `experiments/davi-3x3/runs/20260507T043533Z_full_train`. Records: 198 total, 32 eval cycles._

### Train loss

- start: 0.1492
- end:   0.0623
- min:   0.0030 (step 1800)

### Eval corrected-macro trajectory

| step | corrected_macro | pred_mean | pred_std |
|-----:|-----:|-----:|-----:|
| 500 | 3.4995 | 0.007 | 0.287 |
| 1000 | 2.9435 | 0.615 | 0.215 |
| 1500 | 2.2476 | 1.365 | 0.217 |
| 2000 | 1.5927 | 2.158 | 0.387 |
| 2500 | 1.1400 | 2.845 | 0.707 |
| 3000 | 0.8643 | 3.486 | 1.082 |
| 3500 | 0.8916 | 3.940 | 1.500 |
| 4000 | 0.9416 | 4.294 | 1.798 |
| 4500 | 0.9137 | 4.583 | 2.012 |
| 5000 | 0.8612 | 4.750 | 2.160 |
| 5500 | 0.8112 | 4.813 | 2.213 |
| 6000 | 0.7483 | 4.885 | 2.219 |
| 6500 | 0.6811 | 4.915 | 2.216 |
| 7000 | 0.6637 | 4.960 | 2.260 |
| 7500 | 0.7182 | 5.008 | 2.338 |
| 8000 | 0.7680 | 5.049 | 2.385 |
| 8500 | 0.7750 | 5.124 | 2.440 |
| 9000 | 0.7352 | 5.097 | 2.410 |
| 9500 | 0.6845 | 5.110 | 2.414 |
| 10000 | 0.6393 | 5.146 | 2.420 |
| 10500 | 0.7219 | 5.121 | 2.462 |
| 11000 | 0.7778 | 5.143 | 2.514 |
| 11500 | 0.8715 | 5.127 | 2.566 |
| 12000 | 0.9240 | 5.106 | 2.592 |
| 12500 | 0.8918 | 5.059 | 2.556 |
| 13000 | 0.8200 | 5.059 | 2.495 |
| 13500 | 0.7692 | 5.070 | 2.423 |
| 14000 | 0.7293 | 5.072 | 2.406 |
| 14500 | 0.6868 | 5.091 | 2.388 |
| 15000 | 0.6548 | 5.134 | 2.402 |
| 15500 | 0.7164 | 5.154 | 2.445 |
| 16000 | 0.7607 | 5.184 | 2.487 |

### Per-V*-layer MAE — start / middle / end (bounded oracle, d=1..6)

| V* depth | start (step 500) | middle (step 8500) | end (step 16000) |
|------:|------:|------:|------:|
| 1 | 1.011 | 0.094 | 0.062 |
| 2 | 2.095 | 0.293 | 0.322 |
| 3 | 3.004 | 0.586 | 0.594 |
| 4 | 3.986 | 0.900 | 0.912 |
| 5 | 4.982 | 1.195 | 1.212 |
| 6 | 5.919 | 1.582 | 1.462 |

### Beam capability

_(Live solve_rate / avg_solve_len are not captured during 3x3 training. See `<run>/results/beam_eval_focused.json` and the Beam capability section of `error_trajectories.html`.)_
