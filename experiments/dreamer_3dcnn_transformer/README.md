# DREAMER 3D-CNN + Transformer campaign

This is a standalone emotion classifier. It does not use SIC. The model uses
the reusable DREAMER 3D-CNN front end with two convolutional blocks, followed
by temporal self-attention and a classification head.

The existing tracker now accepts campaign-specific experiment paths, so this
campaign has its own run store while using the same planning and reporting
workflow:

```bash
python experiments/dreamer_3dcnn/tracker.py \
  --campaign experiments/dreamer_3dcnn_transformer/campaign.json \
  suggest --target valence

python experiments/dreamer_3dcnn/tracker.py \
  --campaign experiments/dreamer_3dcnn_transformer/campaign.json \
  plan --target valence
```

The generated launcher and run bundle live under
`experiments/dreamer_3dcnn_transformer/runs/`. Submit the printed Slurm command
on Longleaf. After downloading the completed run folder, import and inspect it:

```bash
python experiments/dreamer_3dcnn/tracker.py \
  --campaign experiments/dreamer_3dcnn_transformer/campaign.json \
  sync /path/to/downloaded/run-folder

python experiments/dreamer_3dcnn/tracker.py \
  --campaign experiments/dreamer_3dcnn_transformer/campaign.json \
  status
```

Use the same commands with `--target arousal` for the independent arousal
campaign. Each target begins with a four-fold/two-epoch smoke run, advances to a
baseline limited to the first four held-out subjects, and then performs controlled one-factor experiments around
the best completed run. Both stages set `max_folds` to 4; other subjects
still participate in training and validation. Existing planned bundles retain
their original configurations.

See [ARCHITECTURE_REPORT.md](ARCHITECTURE_REPORT.md) for the model rationale,
tensor shapes, and what each component is intended to learn.

## Tracker documentation and result figures

See the [team usage and design guide](../../docs/training-tracker.md) for the
reusable tracker, subject and model comparison plots, and an offline demo for
both architectures. Existing tracker commands remain compatible.
