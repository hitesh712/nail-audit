"""Does the draft's "real-time fine-tuning" mechanism actually work?

Section IV-A describes live updates at LR 0.01, a memory buffer of 10 samples per
class, and a 0.7 reliability cut-off. The abstract sells it. Nothing in the
results measures it.

This implements the described procedure literally and measures three things,
because adaptation alone is not evidence of anything:

  adaptation  running accuracy over the correction stream.
  forgetting  accuracy on a retention set that is never updated on. This is the
              number that decides whether the mechanism is safe to deploy.
  stability   variance across random stream orderings.

Sets are disjoint by construction: corrections stream from the validation split,
retention is measured on the held-out test split, which no update ever touches.
The validation split was used for checkpoint selection but never for weight
updates, which is noted as a caveat rather than hidden.

Ablations: memory buffer on/off, and the specified LR 0.01 against 1e-4.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail import PAPER_CLASSES  # noqa: E402
from nail.models import build_model  # noqa: E402
from nail.train import IMAGENET_MEAN, IMAGENET_STD, TrainConfig, set_seed  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.size": 8, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})

C2I = {c: i for i, c in enumerate(PAPER_CLASSES)}


def make_tf(size: int):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_tensor(path: str, tf) -> torch.Tensor:
    with Image.open(path) as im:
        return tf(im.convert("RGB"))


@torch.no_grad()
def evaluate(model, tensors: torch.Tensor, labels: np.ndarray, device: str,
             bs: int = 64) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(tensors), bs):
        xb = tensors[i:i + bs].to(device)
        pred = model(xb).argmax(1).cpu().numpy()
        correct += int((pred == labels[i:i + bs]).sum())
    return correct / len(labels)


def run_stream(
    base_state: dict,
    cfg: TrainConfig,
    stream_t: torch.Tensor,
    stream_y: np.ndarray,
    ret_t: torch.Tensor,
    ret_y: np.ndarray,
    device: str,
    lr: float,
    use_buffer: bool,
    buffer_per_class: int = 10,
    conf_threshold: float = 0.7,
    steps: int = 1,
    eval_every: int = 25,
    seed: int = 0,
) -> pd.DataFrame:
    """One pass of the described procedure. Returns the retention trace."""
    set_seed(seed)
    model = build_model(cfg.model, num_classes=len(PAPER_CLASSES),
                        unfreeze_layer4=cfg.unfreeze_layer4, dropout=cfg.dropout)
    model.load_state_dict(copy.deepcopy(base_state))
    model.to(device)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
        weight_decay=cfg.weight_decay,
    )
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    buffer: dict[int, deque] = defaultdict(lambda: deque(maxlen=buffer_per_class))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(stream_y))

    trace = [{"n_corrections": 0, "n_seen": 0,
              "retention_acc": evaluate(model, ret_t, ret_y, device),
              "stream_acc_running": np.nan}]
    n_corr, seen_ok, seen = 0, 0, 0

    for pos, idx in enumerate(order):
        x = stream_t[idx:idx + 1].to(device)
        y = int(stream_y[idx])

        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(x), 1)
        pred = int(p.argmax(1).item())
        conf = float(p.max().item())
        seen += 1
        seen_ok += int(pred == y)

        # The trigger condition described in the draft: wrong, or unconfident.
        if pred == y and conf >= conf_threshold:
            continue

        n_corr += 1
        buffer[y].append(idx)

        xs = [stream_t[idx]]
        ys = [y]
        if use_buffer:
            pool = [(j, c) for c, items in buffer.items() for j in items if j != idx]
            if pool:
                pick = rng.choice(len(pool), size=min(8, len(pool)), replace=False)
                for k in pick:
                    j, c = pool[int(k)]
                    xs.append(stream_t[j]); ys.append(c)

        xb = torch.stack(xs).to(device)
        yb = torch.tensor(ys, dtype=torch.long, device=device)

        model.train()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            crit(model(xb), yb).backward()
            opt.step()

        if n_corr % eval_every == 0:
            trace.append({
                "n_corrections": n_corr, "n_seen": seen,
                "retention_acc": evaluate(model, ret_t, ret_y, device),
                "stream_acc_running": seen_ok / seen,
            })

    trace.append({"n_corrections": n_corr, "n_seen": seen,
                  "retention_acc": evaluate(model, ret_t, ret_y, device),
                  "stream_acc_running": seen_ok / max(seen, 1)})
    return pd.DataFrame(trace)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path,
                    default=ROOT / "results" / "runs_headline" / "single_efficientnet_b0_s0")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_final.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "21_interactive.md")
    ap.add_argument("--figdir", type=Path, default=ROOT / "results" / "figures")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hist = json.loads((args.run / "history.json").read_text())
    cfg_d = dict(hist["config"]); cfg_d.pop("classes", None)
    cfg = TrainConfig(**cfg_d)
    base_state = torch.load(args.run / "best.pt", map_location="cpu")
    tf = make_tf(cfg.image_size)

    df = pd.read_csv(args.manifest)
    stream_df = df[(df["split"] == "val") & df["label"].notna()]
    ret_df = df[(df["split"] == "test") & df["label"].notna()]
    print(f"stream {len(stream_df)} (val), retention {len(ret_df)} (test)")

    print("loading tensors...")
    stream_t = torch.stack([load_tensor(p, tf) for p in stream_df["path"]])
    ret_t = torch.stack([load_tensor(p, tf) for p in ret_df["path"]])
    stream_y = np.array([C2I[c] for c in stream_df["label"]])
    ret_y = np.array([C2I[c] for c in ret_df["label"]])

    model0 = build_model(cfg.model, num_classes=len(PAPER_CLASSES),
                         unfreeze_layer4=cfg.unfreeze_layer4, dropout=cfg.dropout)
    model0.load_state_dict(copy.deepcopy(base_state)); model0.to(device)
    baseline = evaluate(model0, ret_t, ret_y, device)
    del model0
    torch.cuda.empty_cache()
    print(f"baseline retention accuracy: {baseline:.4f}")

    conditions = [
        ("as specified (LR 0.01, buffer)", 1e-2, True),
        ("LR 0.01, no buffer", 1e-2, False),
        ("LR 1e-4, buffer", 1e-4, True),
        ("LR 1e-4, no buffer", 1e-4, False),
    ]

    all_traces, summary = {}, []
    for name, lr, buf in conditions:
        traces = []
        for s in args.seeds:
            print(f"  {name}, seed {s}")
            t = run_stream(base_state, cfg, stream_t, stream_y, ret_t, ret_y,
                           device, lr=lr, use_buffer=buf, seed=s)
            t["seed"] = s
            traces.append(t)
        cat = pd.concat(traces)
        all_traces[name] = cat
        finals = [t["retention_acc"].iloc[-1] for t in traces]
        mins = [t["retention_acc"].min() for t in traces]
        summary.append({
            "condition": name,
            "final_retention": float(np.mean(finals)),
            "final_sd": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
            "worst_retention": float(np.mean(mins)),
            "forgetting_pp": float((baseline - np.mean(finals)) * 100),
            "corrections": int(np.mean([t["n_corrections"].iloc[-1] for t in traces])),
        })

    summ = pd.DataFrame(summary)
    summ.to_csv(args.figdir.parent / "interactive_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for name, cat in all_traces.items():
        g = cat.groupby("n_corrections")["retention_acc"].mean()
        ax.plot(g.index, g.values * 100, marker="o", ms=2.5, lw=1.3, label=name)
    ax.axhline(baseline * 100, ls="--", lw=1.0, color="k", alpha=0.6,
               label="before any update")
    ax.set_xlabel("corrections applied"); ax.set_ylabel("retention accuracy (%)")
    ax.legend(fontsize=6.5)
    args.figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figdir / "fig9_interactive.png")
    plt.close(fig)

    lines = [
        "# Interactive refinement, as specified in the draft",
        "",
        f"Model `{cfg.model}`. Corrections stream from the validation split "
        f"({len(stream_df)} images); retention measured on the held-out test split "
        f"({len(ret_df)}), which no update touches. {len(args.seeds)} stream orderings.",
        "",
        f"Baseline retention accuracy before any update: **{baseline*100:.2f}%**",
        "",
        "| condition | corrections | final retention | worst retention | forgetting |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, r in summ.iterrows():
        lines.append(
            f"| {r['condition']} | {r['corrections']} | "
            f"{r['final_retention']*100:.2f}% ± {r['final_sd']*100:.2f} | "
            f"{r['worst_retention']*100:.2f}% | "
            f"**{r['forgetting_pp']:+.2f} pp** |"
        )

    spec = summ.iloc[0]
    if spec["forgetting_pp"] > 2.0:
        verdict = (
            f"**The procedure as specified is unsafe.** Applying the draft's own "
            f"settings costs {spec['forgetting_pp']:.2f} percentage points of "
            f"accuracy on data the model previously handled correctly. A mechanism "
            f"that degrades the model while appearing to 'learn' from corrections "
            f"cannot be deployed, and the abstract should not advertise it."
        )
    elif spec["forgetting_pp"] > 0.5:
        verdict = (
            f"The procedure as specified degrades retention accuracy by "
            f"{spec['forgetting_pp']:.2f} pp. That is a real cost and must be "
            f"reported alongside any adaptation benefit."
        )
    else:
        verdict = (
            f"The procedure as specified does not measurably damage retention "
            f"({spec['forgetting_pp']:+.2f} pp). The claim survives, and the "
            f"experiment supporting it belongs in the paper."
        )

    lines += [
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "Compare the learning-rate rows: the specified 1e-2 against 1e-4 under an",
        "otherwise identical protocol isolates whether the rate, rather than the idea,",
        "is the problem. The buffer rows isolate whether replay prevents forgetting.",
        "",
        "## Caveat",
        "",
        "The correction stream is the validation split, which was used for checkpoint",
        "selection though never for weight updates. A genuinely prospective stream",
        "would be preferable; this is the closest available without new data.",
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    for line in lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
