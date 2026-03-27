"""
Generate figures for the LCEL module.

Produces:
  - docs/images/lcel_composition.png — Pipe operator chain composition diagram
  - docs/images/parallel_speedup.png — Sequential vs parallel execution bar chart
  - docs/images/streaming_flow.png — Streaming pipeline with intermediate steps
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# --- Theme ---
plt.style.use("dark_background")
BG_COLOR = "#1a1a2e"
COLORS = ["#4f7cac", "#5a9e8f", "#9b6b9e", "#c47e3a", "#b85450"]
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4e"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "docs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fig_lcel_composition():
    """Diagram showing pipe operator chain composition."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(7, 6.5, "LCEL Chain Composition with the Pipe Operator",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=TEXT_COLOR)

    # Top row: Simple chain
    ax.text(0.5, 5.5, "Simple Chain:", fontsize=10, fontweight="bold",
            color=COLORS[0], va="center")

    simple_steps = [
        ("Prompt\nTemplate", COLORS[0]),
        ("|", None),
        ("LLM\nModel", COLORS[1]),
        ("|", None),
        ("Output\nParser", COLORS[2]),
    ]

    sx = 3.5
    for name, color in simple_steps:
        if color is None:
            ax.text(sx, 5.5, name, ha="center", va="center",
                    fontsize=16, fontweight="bold", color=COLORS[3])
            sx += 0.6
        else:
            box = FancyBboxPatch((sx - 0.8, 5.0), 1.6, 1.0,
                                  boxstyle="round,pad=0.1", facecolor=color,
                                  edgecolor=TEXT_COLOR, linewidth=1.5, alpha=0.85)
            ax.add_patch(box)
            ax.text(sx, 5.5, name, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
            sx += 2.2

    # Code annotation
    ax.text(12, 5.5, "chain = prompt | model | parser", ha="center", va="center",
            fontsize=8, color="#888888", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2a2a4e", edgecolor=GRID_COLOR))

    # Middle row: Parallel composition
    ax.text(0.5, 3.5, "Parallel Chain:", fontsize=10, fontweight="bold",
            color=COLORS[1], va="center")

    # Input
    input_box = FancyBboxPatch((2.5, 3.0), 1.5, 1.0,
                                boxstyle="round,pad=0.1", facecolor="#2a2a4e",
                                edgecolor=TEXT_COLOR, linewidth=1.5)
    ax.add_patch(input_box)
    ax.text(3.25, 3.5, "Input", ha="center", va="center",
            fontsize=10, fontweight="bold", color=TEXT_COLOR)

    # Parallel branches
    branches = [
        ("Urgency\nChain", COLORS[0], 4.2),
        ("ICD-10\nChain", COLORS[1], 3.5),
        ("Summary\nChain", COLORS[2], 2.8),
    ]

    for name, color, by in branches:
        box = FancyBboxPatch((5.5, by - 0.3), 1.8, 0.6,
                              boxstyle="round,pad=0.08", facecolor=color,
                              edgecolor=TEXT_COLOR, linewidth=1.2, alpha=0.85)
        ax.add_patch(box)
        ax.text(6.4, by, name, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")
        # Arrow from input
        ax.annotate("", xy=(5.5, by), xytext=(4.0, 3.5),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.3,
                                    connectionstyle="arc3,rad=0.1"))

    # Merge
    merge_box = FancyBboxPatch((8.5, 3.0), 1.8, 1.0,
                                boxstyle="round,pad=0.1", facecolor=COLORS[3],
                                edgecolor=TEXT_COLOR, linewidth=1.5, alpha=0.85)
    ax.add_patch(merge_box)
    ax.text(9.4, 3.5, "Merge\nResults", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")

    for _, color, by in branches:
        ax.annotate("", xy=(8.5, 3.5), xytext=(7.3, by),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.3,
                                    connectionstyle="arc3,rad=-0.1"))

    # Output
    output_box = FancyBboxPatch((11, 3.0), 1.5, 1.0,
                                 boxstyle="round,pad=0.1", facecolor="#2a2a4e",
                                 edgecolor=TEXT_COLOR, linewidth=1.5)
    ax.add_patch(output_box)
    ax.text(11.75, 3.5, "Output\nDict", ha="center", va="center",
            fontsize=9, fontweight="bold", color=TEXT_COLOR)
    ax.annotate("", xy=(11, 3.5), xytext=(10.3, 3.5),
                arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=2))

    # Bottom row: Features
    ax.text(0.5, 1.5, "Built-in:", fontsize=10, fontweight="bold",
            color=COLORS[2], va="center")

    features = [
        (".invoke()", "Single input"),
        (".stream()", "Token streaming"),
        (".batch()", "Parallel batch"),
        (".ainvoke()", "Async single"),
        (".astream()", "Async stream"),
    ]

    fx = 3.0
    for method, desc in features:
        box = FancyBboxPatch((fx - 0.7, 1.0), 1.8, 0.9,
                              boxstyle="round,pad=0.08", facecolor="#2a2a4e",
                              edgecolor=GRID_COLOR, linewidth=1)
        ax.add_patch(box)
        ax.text(fx + 0.2, 1.55, method, ha="center", va="center",
                fontsize=8, fontweight="bold", color=COLORS[2], family="monospace")
        ax.text(fx + 0.2, 1.2, desc, ha="center", va="center",
                fontsize=7, color="#888888")
        fx += 2.2

    # Bottom note
    ax.text(7, 0.3, "Every LCEL chain is a Runnable — it supports all methods above and can be composed further",
            ha="center", va="center", fontsize=9, color="#666666", style="italic")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "lcel_composition.png", dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'lcel_composition.png'}")


def fig_parallel_speedup():
    """Bar chart comparing sequential vs parallel execution times."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor(BG_COLOR)

    # Left: Detailed timing breakdown
    ax1.set_facecolor(BG_COLOR)

    chains = ["Urgency\nClassification", "ICD-10\nCoding", "Clinical\nSummary"]
    seq_times = [1.58, 1.62, 1.62]
    par_times = [1.58, 1.62, 1.62]  # Same individual times, but overlapped

    x = np.arange(len(chains))
    width = 0.35

    # Sequential bars (stacked appearance - show cumulative time)
    bars_seq = ax1.bar(x - width / 2, seq_times, width, label="Sequential",
                        color=COLORS[4], alpha=0.85, edgecolor="none")

    # Parallel bars
    bars_par = ax1.bar(x + width / 2, par_times, width, label="Parallel",
                        color=COLORS[1], alpha=0.85, edgecolor="none")

    for bars in [bars_seq, bars_par]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                     f"{height:.2f}s", ha="center", va="bottom",
                     fontsize=9, color=TEXT_COLOR)

    ax1.set_ylabel("Time per Chain (seconds)", fontsize=11, color=TEXT_COLOR)
    ax1.set_title("Individual Chain Execution Times",
                   fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(chains, fontsize=10, color=TEXT_COLOR)
    ax1.set_ylim(0, 2.2)
    ax1.tick_params(colors=TEXT_COLOR)
    ax1.spines["bottom"].set_color(GRID_COLOR)
    ax1.spines["left"].set_color(GRID_COLOR)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.grid(True, color=GRID_COLOR, alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=10, facecolor="#2a2a4e", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Right: Total time comparison
    ax2.set_facecolor(BG_COLOR)

    methods = ["Sequential", "Parallel"]
    total_times = [4.82, 1.94]
    colors = [COLORS[4], COLORS[1]]

    bars = ax2.bar(methods, total_times, width=0.5, color=colors, alpha=0.85, edgecolor="none")
    for bar, time_val in zip(bars, total_times):
        ax2.text(bar.get_x() + bar.get_width() / 2., time_val + 0.1,
                 f"{time_val:.2f}s", ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR)

    # Speedup annotation
    ax2.annotate("2.5x\nspeedup", xy=(1, 1.94), xytext=(1.35, 3.5),
                 fontsize=14, fontweight="bold", color=COLORS[1],
                 arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=2),
                 ha="center")

    # Time saved
    ax2.text(0.5, 4.5, "2.88s saved", ha="center", va="center",
             fontsize=11, color=COLORS[1], style="italic")

    ax2.set_ylabel("Total Time (seconds)", fontsize=11, color=TEXT_COLOR)
    ax2.set_title("Total Pipeline Time",
                   fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=10)
    ax2.set_ylim(0, 5.5)
    ax2.tick_params(colors=TEXT_COLOR)
    ax2.spines["bottom"].set_color(GRID_COLOR)
    ax2.spines["left"].set_color(GRID_COLOR)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, color=GRID_COLOR, alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "parallel_speedup.png", dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'parallel_speedup.png'}")


def fig_streaming_flow():
    """Diagram of streaming pipeline with intermediate steps."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(7, 6.5, "LCEL Streaming Pipeline with Intermediate Steps",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=TEXT_COLOR)

    # Pipeline stages with timing
    stages = [
        ("Preprocess", "RunnableLambda\nadd_metadata()", COLORS[0], "~1ms", 1.5),
        ("Format\nPrompt", "ChatPromptTemplate\nwith variables", COLORS[1], "~1ms", 4.2),
        ("LLM Call", "ChatOpenAI\nstreaming=True", COLORS[2], "~3.4s", 7.0),
        ("Parse\nOutput", "StrOutputParser\nchunk-by-chunk", COLORS[3], "~1ms", 9.8),
        ("Client", "Real-time\ntoken display", COLORS[4], "instant", 12.5),
    ]

    for name, desc, color, timing, px in stages:
        box = FancyBboxPatch((px - 1.0, 4.0), 2.0, 1.5,
                              boxstyle="round,pad=0.15", facecolor=color,
                              edgecolor=TEXT_COLOR, linewidth=1.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(px, 5.0, name, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(px, 4.4, desc, ha="center", va="center",
                fontsize=7, color="#e0e0e0")
        ax.text(px, 3.7, timing, ha="center", va="center",
                fontsize=8, color=color, fontweight="bold")

    # Arrows between stages
    for i in range(len(stages) - 1):
        _, _, _, _, px1 = stages[i]
        _, _, _, _, px2 = stages[i + 1]
        ax.annotate("", xy=(px2 - 1.0, 4.75), xytext=(px1 + 1.0, 4.75),
                    arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=1.8))

    # Streaming visualization
    ax.text(7, 2.8, "Streaming Token Flow", ha="center", va="center",
            fontsize=12, fontweight="bold", color=TEXT_COLOR)

    # Token timeline
    tokens = ["##", " Chief", " Complaint", " &", " HPI", " Summary", "\\n",
              "45-year-old", " female", " presenting", " with", "..."]
    tx = 1.0
    for i, token in enumerate(tokens):
        alpha = min(1.0, 0.3 + i * 0.06)
        color_idx = i % len(COLORS)
        ax.text(tx, 2.2, token, fontsize=8, color=COLORS[color_idx],
                alpha=alpha, family="monospace")
        tx += len(token) * 0.14 + 0.15

    # Time to first token
    ax.annotate("", xy=(1.0, 1.7), xytext=(0.5, 1.7),
                arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=1.5))
    ax.text(0.7, 1.4, "~340ms to\nfirst token", fontsize=8, color=COLORS[1],
            ha="center")

    # Traditional wait
    ax.plot([1.0, 11.5], [0.9, 0.9], color=COLORS[4], linewidth=2, alpha=0.5)
    ax.plot([11.5, 11.5], [0.7, 1.1], color=COLORS[4], linewidth=2, alpha=0.5)
    ax.text(6.25, 0.6, "Traditional: wait 3.4s for complete response", fontsize=8,
            color=COLORS[4], ha="center", style="italic")

    # Streaming indicator
    ax.plot([1.0, 11.5], [1.5, 1.5], color=COLORS[1], linewidth=2, alpha=0.5, linestyle="--")
    ax.text(6.25, 1.7, "Streaming: tokens appear immediately as generated", fontsize=8,
            color=COLORS[1], ha="center", style="italic")

    # Bottom: FastAPI integration note
    ax.text(7, 0.2, "Integration: FastAPI StreamingResponse  |  WebSocket  |  Server-Sent Events (SSE)",
            ha="center", va="center", fontsize=9, color="#666666", style="italic")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "streaming_flow.png", dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'streaming_flow.png'}")


if __name__ == "__main__":
    print("Generating figures for 04-lcel...")
    fig_lcel_composition()
    fig_parallel_speedup()
    fig_streaming_flow()
    print("Done.")
