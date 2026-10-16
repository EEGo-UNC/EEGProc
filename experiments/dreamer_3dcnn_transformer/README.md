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
campaign. Each target begins with a two-fold/two-epoch smoke run, advances to a
full LOSO baseline, and then performs controlled one-factor experiments around
the best completed run.

See [ARCHITECTURE_REPORT.md](ARCHITECTURE_REPORT.md) for the model rationale,
tensor shapes, and what each component is intended to learn.
