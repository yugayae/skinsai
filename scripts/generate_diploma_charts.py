#!/usr/bin/env python3
"""Generate publication-grade scientific visualization charts from SkinAI experimental logs (EXP-013 to EXP-018)
for the Diploma / Project Documentation.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "artifacts" / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.sans-serif": ["DejaVu Sans", "Arial", "Segoe UI"],
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.9,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def make_fig1_backbone_and_classifier_tournament() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")

    # Left panel: Backbone progression (Hold-out / OOF Macro F1 & Mel Recall)
    backbones = [
        "EfficientNet-B0\n(Baseline)",
        "CSPDarknet-53\n(YOLOv8 cls)",
        "Swin-Tiny\n(Window ViT)",
        "ConvNeXt-Tiny\n(Modern CNN)",
        "BiomedCLIP\n(PMC-15M ViT)",
        "DINOv2 Small\n(Self-Sup ViT)",
        "Dual Fusion\n(DINOv2+Biomed)",
    ]
    macro_f1 = [0.6408, 0.6592, 0.6681, 0.6845, 0.6890, 0.7067, 0.7360]
    mel_recall = [0.814, 0.828, 0.841, 0.862, 0.917, 0.862, 0.924]

    x = np.arange(len(backbones))
    w = 0.36
    ax1.set_facecolor("#FFFFFF")
    b1 = ax1.bar(x - w / 2, macro_f1, w, label="Macro F1 (6 классов)", color="#2563EB", edgecolor="#1E3A8A")
    b2 = ax1.bar(x + w / 2, mel_recall, w, label="Melanoma Recall (tau=0.05)", color="#DC2626", edgecolor="#7F1D1D")
    ax1.set_xticks(x)
    ax1.set_xticklabels(backbones, fontsize=8.5)
    ax1.set_ylim(0.50, 1.02)
    ax1.set_ylabel("Значение метрики (0..1)", fontsize=10, fontweight="bold")
    ax1.set_title("A. Турнир 6 визуальных бэкбонов и Dual-Backbone Fusion (EXP-013 / EXP-015)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y")
    ax1.legend(loc="lower right", frameon=True)
    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{bar.get_height():.3f}", ha="center", fontsize=7.5, fontweight="bold", color="#1E3A8A")
    for bar in b2:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{bar.get_height()*100:.1f}%", ha="center", fontsize=7.5, fontweight="bold", color="#991B1B")

    # Right panel: 5-Fold OOF Macro F1 vs External Stanford DDI In-Scope AUC (EXP-018A)
    models = [
        "RandomForest\n(896D)",
        "XGBoost\n(896D)",
        "LightGBM\n(896D)",
        "LinearSVM\n(896D)",
        "Sklearn MLP\n(896D)",
        "LogReg L2\n(896D Image)",
        "LogReg L2\n(903D Meta)",
        "PyTorch Pure\nImage (896D)",
        "Hybrid PyTorch\n+ LogReg",
        "PyTorch Dual-Backbone Fusion\n(903D MetaDrop)",
    ]
    oof_f1 = [0.4846, 0.6097, 0.6121, 0.6170, 0.6370, 0.6510, 0.6610, 0.6561, 0.7091, 0.7161]
    ddi_auc = [0.7662, 0.7361, 0.7352, 0.7591, 0.7492, 0.7633, 0.6566, 0.8241, 0.8266, 0.8250]

    y = np.arange(len(models))
    h = 0.38
    ax2.set_facecolor("#FFFFFF")
    ax2.barh(y - h / 2, oof_f1, h, label="5-Fold OOF Macro F1 (N=3,767)", color="#0D9488", edgecolor="#115E59")
    ax2.barh(y + h / 2, ddi_auc, h, label="Blind Stanford DDI In-Scope AUC (N=301)", color="#7C3AED", edgecolor="#4C1D95")
    ax2.set_yticks(y)
    ax2.set_yticklabels(models, fontsize=8.5)
    ax2.set_xlim(0.40, 0.92)
    ax2.set_xlabel("Значение метрики", fontsize=10, fontweight="bold")
    ax2.set_title("Б. Классический ML против Двухголовой Fusion-сети (EXP-018A)", fontsize=11, fontweight="bold")
    ax2.grid(axis="x")
    ax2.legend(loc="lower right", frameon=True)
    for i, (v1, v2) in enumerate(zip(oof_f1, ddi_auc)):
        ax2.text(v1 + 0.006, i - h / 2, f"{v1:.3f}", va="center", fontsize=7.5, color="#115E59", fontweight="bold")
        ax2.text(v2 + 0.006, i + h / 2, f"{v2:.3f}", va="center", fontsize=7.5, color="#4C1D95", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_backbone_and_classifier_tournament.png", dpi=180)
    plt.close(fig)


def make_fig2_hyperparameter_and_imbalance_proofs() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")

    # Left: 5-tier imbalance defense ablation (EXP-013)
    stages = [
        "1. Обычный\nCross-Entropy",
        "2. + Веса классов\nw_c = N_c^(-0.5)",
        "3. + Masked Focal\nLoss (gamma=2.5)",
        "4. + Сбалансированный\nSampler + Mixup",
        "5. + Калибровка\nпорогов (tau=0.05/0.15)",
    ]
    f1_prog = [0.5812, 0.6540, 0.6985, 0.7275, 0.7360]
    mel_prog = [0.5517, 0.7241, 0.8276, 0.8621, 0.9241]

    x = np.arange(len(stages))
    ax1.set_facecolor("#FFFFFF")
    ax1.plot(x, f1_prog, marker="o", linewidth=2.6, markersize=8, color="#2563EB", label="Macro F1 (6 классов)")
    ax1.plot(x, mel_prog, marker="s", linewidth=2.6, markersize=8, color="#DC2626", label="Полнота по меланоме (Mel Recall)")
    for i, (f_v, m_v) in enumerate(zip(f1_prog, mel_prog)):
        ax1.annotate(f"{f_v:.3f}", (x[i], f_v), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8.5, fontweight="bold", color="#1E3A8A")
        ax1.annotate(f"{m_v*100:.1f}%", (x[i], m_v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color="#991B1B")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=8.5)
    ax1.set_ylim(0.48, 1.00)
    ax1.set_ylabel("Метрика", fontsize=10, fontweight="bold")
    ax1.set_title("А. Эффект 5 ступеней защиты от дисбаланса классов (EXP-013)", fontsize=11, fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc="lower right", frameon=True)

    # Right: Optuna & Single-Factor Hyperparameter Proofs (EXP-014)
    configs = [
        "Оптимизатор:\nSGD (mom=0.9)",
        "Оптимизатор:\nAdam",
        "Оптимизатор:\nAdamW (выбран)",
        "Активация:\nReLU",
        "Активация:\nGELU",
        "Активация:\nSiLU (выбрана)",
        "Focal gamma=0.0\n(CE)",
        "Focal gamma=2.5\n(выбран)",
    ]
    inner_f1 = [0.6240, 0.7085, 0.7275, 0.7112, 0.7190, 0.7275, 0.6910, 0.7275]
    colors = ["#94A3B8", "#60A5FA", "#1D4ED8", "#94A3B8", "#60A5FA", "#0D9488", "#94A3B8", "#7C3AED"]

    x2 = np.arange(len(configs))
    ax2.set_facecolor("#FFFFFF")
    bars = ax2.bar(x2, inner_f1, color=colors, edgecolor="#1E293B", width=0.58)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(configs, fontsize=8)
    ax2.set_ylim(0.55, 0.77)
    ax2.set_ylabel("Macro F1 (Inner-Val / Hold-out)", fontsize=10, fontweight="bold")
    ax2.set_title("Б. Доказательство выбора гиперпараметров (EXP-014 Optuna)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y")
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004, f"{bar.get_height():.4f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_hyperparameter_and_imbalance_proofs.png", dpi=180)
    plt.close(fig)


def make_fig3_melanoma_operating_curve_and_3zone_triage() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")

    # Left: Paired Sensitivity / Specificity / PPV across thresholds (EXP-018, N_mel=142)
    taus = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35]
    sens = [0.9577, 0.9085, 0.8944, 0.8380, 0.8239, 0.7887]
    spec = [0.8873, 0.9331, 0.9530, 0.9634, 0.9730, 0.9859]
    ppv = [0.3342, 0.4448, 0.5292, 0.5749, 0.6429, 0.7671]

    ax1.set_facecolor("#FFFFFF")
    ax1.plot(taus, sens, marker="o", linewidth=2.5, color="#DC2626", label="Sensitivity / Recall (Полнота по меланоме)")
    ax1.plot(taus, spec, marker="s", linewidth=2.5, color="#2563EB", label="Specificity (Специфичность)")
    ax1.plot(taus, ppv, marker="^", linewidth=2.2, linestyle="--", color="#059669", label="PPV / Precision (Точность положительного)")
    ax1.axvline(0.15, color="#B91C1C", linestyle=":", linewidth=1.6, label="Порог Red Zone (tau=0.15)")
    ax1.axvline(0.05, color="#EA580C", linestyle=":", linewidth=1.6, label="Порог высокой бдительности (tau=0.05)")
    ax1.set_xlabel("Порог вероятности меланомы (tau_mel)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Доля (0..1)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0.25, 1.03)
    ax1.set_title("А. Парные кривые Чувствительности / Специфичности / PPV (EXP-018, 142 меланомы)", fontsize=10.5, fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)

    # Right: 3-Zone Triage Capture Rates by Pathology Group (EXP-018)
    groups = [
        "Злокачественные\nIn-Scope (MEL/BCC/SCC)\n[OOF N=1,223]",
        "Предрак ACK\n(Актинический кератоз)\n[OOF N=730]",
        "Stanford DDI\nIn-Scope Злокачеств.\n[Blind N=124]",
        "Доброкачественные\n(NEV / SEK)\n[OOF N=1,814]",
    ]
    red_yellow_capture = [99.35, 99.73, 94.35, 13.84]
    green_safe = [0.65, 0.27, 5.65, 86.16]

    x2 = np.arange(len(groups))
    w = 0.38
    ax2.set_facecolor("#FFFFFF")
    b1 = ax2.bar(x2 - w / 2, red_yellow_capture, w, label="Направлены к врачу (Red + Yellow Zone, %)", color="#EF4444", edgecolor="#7F1D1D")
    b2 = ax2.bar(x2 + w / 2, green_safe, w, label="Зелёная зона (Green Zone, %)", color="#10B981", edgecolor="#065F46")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(groups, fontsize=8.5)
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("Доля случаев (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Б. Безопасность 3-зонной клинической маршрутизации (EXP-018)", fontsize=10.5, fontweight="bold")
    ax2.grid(axis="y")
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)
    for bar in b1:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#991B1B")
    for bar in b2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#065F46")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_melanoma_operating_curve_and_3zone_triage.png", dpi=180)
    plt.close(fig)


def make_fig4_xai_grabcut_and_speed_benchmark() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")

    # Left: Hardware Latency (ms/frame) & FPS on RTX 3060 vs CPU (EXP-013)
    archs = [
        "EfficientNet-B0\n(ONNX / GPU)",
        "CSPDarknet-53\n(YOLOv8)",
        "ConvNeXt-Tiny",
        "DINOv2 Small\n(ViT-S/14)",
        "BiomedCLIP\n(ViT-B/16)",
        "Dual Fusion\n(DINOv2+Biomed)",
    ]
    gpu_ms = [4.2, 5.1, 6.8, 8.9, 9.2, 18.1]
    cpu_ms = [18.5, 29.4, 39.0, 49.0, 52.0, 101.0]

    x = np.arange(len(archs))
    w = 0.36
    ax1.set_facecolor("#FFFFFF")
    b1 = ax1.bar(x - w / 2, gpu_ms, w, label="RTX 3060 12GB (мс/кадр)", color="#2563EB", edgecolor="#1E3A8A")
    b2 = ax1.bar(x + w / 2, cpu_ms, w, label="CPU 6-Core (мс/кадр)", color="#F59E0B", edgecolor="#78350F")
    ax1.set_xticks(x)
    ax1.set_xticklabels(archs, fontsize=8.5)
    ax1.set_ylabel("Задержка инференса (мс, меньше = быстрее)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 118)
    ax1.set_title("А. Реальная скорость инференса на RTX 3060 и CPU (EXP-013)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y")
    ax1.legend(loc="upper left", frameon=True)
    for bar in b1:
        fps = 1000.0 / bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0, f"{bar.get_height():.1f}мс\n({fps:.0f} FPS)", ha="center", fontsize=7.2, fontweight="bold", color="#1E3A8A")
    for bar in b2:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0, f"{bar.get_height():.1f}мс", ha="center", fontsize=7.5, fontweight="bold", color="#78350F")

    # Right: Deterministic XAI Biomarker Effect Sizes (Cohen's d, EXP-016) & GrabCut Off-Center IoU (EXP-018)
    metrics_lbl = [
        "Эритема CIE a*\n(Cohen's d)",
        "Цветовая энтропия\nCIE LAB (Cohen's d)",
        "Неровность границ B\n(Cohen's d)",
        "Концентрация внимания\nDINOv2 на очаге (x-крат)",
        "GrabCut Off-Center\nShift IoU (Legacy=0.565)",
        "Атлас биопсий k-NN\nMalignancy Hit@3",
    ]
    vals = [0.493, 0.393, 0.300, 0.875, 0.6202, 0.9447]
    disp_text = ["d = +0.493\n(p < 1e-9)", "d = +0.393\n(p < 1e-6)", "d = +0.300\n(p < 1e-4)", "1.75x выше фона\n(91.4% кадров)", "IoU = 0.6202\n(+9.7% к Otsu)", "94.47%\nсовпадение"]
    c_bars = ["#0D9488", "#0D9488", "#0D9488", "#7C3AED", "#2563EB", "#DC2626"]

    y = np.arange(len(metrics_lbl))
    ax2.set_facecolor("#FFFFFF")
    bars2 = ax2.barh(y, vals, color=c_bars, edgecolor="#1E293B", height=0.55)
    ax2.set_yticks(y)
    ax2.set_yticklabels(metrics_lbl, fontsize=8.5)
    ax2.set_xlim(0, 1.18)
    ax2.set_xlabel("Нормированная величина эффекта / точность", fontsize=10, fontweight="bold")
    ax2.set_title("Б. Доказательная база модуля объяснимости (Deterministic XAI) (EXP-016 / EXP-018 DS-5)", fontsize=11, fontweight="bold")
    ax2.grid(axis="x")
    for bar, txt in zip(bars2, disp_text):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, txt, va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_xai_grabcut_and_speed_benchmark.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    make_fig1_backbone_and_classifier_tournament()
    make_fig2_hyperparameter_and_imbalance_proofs()
    make_fig3_melanoma_operating_curve_and_3zone_triage()
    make_fig4_xai_grabcut_and_speed_benchmark()
    print(f"[+] Successfully generated 4 scientific charts in {OUT_DIR}")

