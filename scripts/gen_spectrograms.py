"""Generate mel spectrogram PNGs for README showcase samples."""

import numpy as np
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples"
OUTPUT_DIR = SAMPLES_DIR

SHOWCASE = [
    "farewell-at-dawn",
    "frozen-labyrinth",
    "shadow-throne",
    "midnight-duel",
]

CMAP = "magma"
FIG_W, FIG_H = 10, 2.5
DPI = 150


def generate_spectrogram(slug: str) -> None:
    wav_path = SAMPLES_DIR / f"{slug}.wav"
    out_path = OUTPUT_DIR / f"{slug}_spectrogram.png"

    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_alpha(0)

    img = ax.imshow(
        S_db,
        aspect="auto",
        origin="lower",
        cmap=CMAP,
        extent=[0, len(y) / sr, 0, sr // 2],
    )

    ax.set_xlabel("Time (s)", fontsize=10, color="#4a4a5a")
    ax.set_ylabel("Hz", fontsize=10, color="#4a4a5a")
    ax.tick_params(colors="#6a6a7a", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#d0d0d8")
        spine.set_linewidth(0.5)

    fig.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"{slug}: {out_path.name}")


if __name__ == "__main__":
    for slug in SHOWCASE:
        generate_spectrogram(slug)
