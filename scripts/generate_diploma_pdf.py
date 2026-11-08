#!/usr/bin/env python3
"""Generate publication-grade PDF version of `docs/DIPLOMA_PROJECT_DOCUMENTATION.md`
matching the clean academic typography of `project_example.pdf`, with embedded charts,
architecture diagram, tables, and page numbers.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = PROJECT_ROOT / "docs" / "DIPLOMA_PROJECT_DOCUMENTATION.md"
PDF_DOCS_PATH = PROJECT_ROOT / "docs" / "DIPLOMA_PROJECT_DOCUMENTATION.pdf"
PDF_ROOT_PATH = PROJECT_ROOT / "DIPLOMA_PROJECT_DOCUMENTATION.pdf"
VIS_DIR = PROJECT_ROOT / "artifacts" / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)


def generate_architecture_diagram_png() -> Path:
    """Render a crisp 4-tier architecture block diagram for Section 3 of the PDF."""
    out_path = VIS_DIR / "fig0_system_architecture_pipeline.png"
    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle, bg="#EFF6FF", border="#1D4ED8", title_color="#1E3A8A"):
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=bg,
            edgecolor=border,
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=9.2, fontweight="bold", color=title_color)
        ax.text(x + w / 2, y + h / 2 - 1.2, subtitle, ha="center", va="center", fontsize=7.6, color="#1E293B")

    ax.text(
        50,
        53.5,
        "Сквозная архитектура вычислительного конвейера SkinAI (v2.0.0-fusion)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#0F172A",
    )

    # Tier 1: Clients
    draw_box(
        3,
        39,
        26,
        11,
        "1. Клиентский уровень (RU/EN/KO)",
        "• Telegram Bot (aiogram 3 FSM)\n• Telegram Mini App (Vite + TS)\n• Опросник Глазго + Возраст/Зона",
        bg="#EFF6FF",
        border="#2563EB",
        title_color="#1E3A8A",
    )

    # Tier 2: Security & OOD Gate
    draw_box(
        35,
        39,
        28,
        11,
        "2. Входной и цветовой контроль",
        "• Magic Bytes & <15MB (RAM-only)\n• Blur Filter: Laplacian Var >= 8.0\n• CIE L*a*b* Skin OOD Gate (>=25%)",
        bg="#FEF3C7",
        border="#D97706",
        title_color="#92400E",
    )

    # Tier 3: Deterministic XAI Engine
    draw_box(
        69,
        39,
        28,
        11,
        "3. Движок «Deterministic XAI» (OpenCV)",
        "• Black-Hat фильтрация волос (9x9)\n• Attention-Seeded GrabCut контур\n• 12 биомаркеров ABCDE в CIE LAB",
        bg="#ECFDF5",
        border="#059669",
        title_color="#065F46",
    )

    # Tier 4: Dual-Backbone Vision
    draw_box(
        3,
        20,
        44,
        12,
        "4. Визуальное слияние (Dual-Backbone Fusion) + Деконфаундинг (903D)",
        "• Бэкбон 1: Meta DINOv2 Small (ViT-S/14, 384D + Карта внимания [CLS])\n• Бэкбон 2: Microsoft BiomedCLIP (ViT-B/16 PMC-15M, 512D)\n• Клинический энкодер (7D -> 32D SiLU) с 50% Metadata Dropout",
        bg="#F3E8FF",
        border="#7C3AED",
        title_color="#4C1D95",
    )

    # Tier 5: Dual-Head & Biopsy Atlas
    draw_box(
        53,
        20,
        44,
        12,
        "5. Двухголовый классификатор (ONNX) + Атлас биопсий",
        "• Head 1: Бинарный онкоскрининг (Triage AUC = 0.9453 / 0.9917)\n• Head 2: 6 нозологий (BCC, SCC, ACK, SEK, MEL, NEV) — F1 = 0.7161\n• Biopsy Atlas k-NN: Top-3 близнеца из 2 027 биопсий (Hit@3 = 94.5%)",
        bg="#E0E7FF",
        border="#4338CA",
        title_color="#312E81",
    )

    # Tier 6: 3-Zone Triage & Outputs
    draw_box(
        3,
        3,
        94,
        11,
        "6. Декаплированная 3-зонная клиническая маршрутизация, PDF-памятка и анонимный аудит",
        "• КРАСНАЯ ЗОНА (Срочный осмотр до 14 дн.): P(mel) >= 0.15 или P(malig) >= 0.62 или Шкала Глазго >= 3 баллов (NICE NG12)\n"
        "• ЖЁЛТАЯ ЗОНА (Плановый осмотр до 30 дн.): Предрак ACK P(ack) >= 0.20 (перехват 99.73% ACK), пограничный риск или OOD-кадр\n"
        "• ЗЕЛЁНАЯ ЗОНА: Низкий риск  |  • Врачебная памятка PDF (без процентов ИИ)  |  • PostgreSQL Salted Hash Audit (0% хранения фото)",
        bg="#FEE2E2",
        border="#DC2626",
        title_color="#991B1B",
    )

    # Arrows
    arrow_kw = dict(arrowstyle="->,head_width=0.35,head_length=0.45", color="#334155", lw=1.8)
    ax.annotate("", xy=(35, 44.5), xytext=(29, 44.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(69, 44.5), xytext=(63, 44.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(25, 32), xytext=(45, 39), arrowprops=arrow_kw)
    ax.annotate("", xy=(53, 26), xytext=(47, 26), arrowprops=arrow_kw)
    ax.annotate("", xy=(75, 32), xytext=(80, 39), arrowprops=arrow_kw)
    ax.annotate("", xy=(50, 14), xytext=(25, 20), arrowprops=arrow_kw)
    ax.annotate("", xy=(50, 14), xytext=(75, 20), arrowprops=arrow_kw)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def register_cyrillic_fonts():
    candidates = [
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf", "ArialCustom"),
        ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf", "SegoeCustom"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuCustom"),
    ]
    for reg_p, bold_p, name in candidates:
        if os.path.exists(reg_p):
            pdfmetrics.registerFont(TTFont(name, reg_p))
            bold_name = f"{name}-Bold"
            if os.path.exists(bold_p):
                pdfmetrics.registerFont(TTFont(bold_name, bold_p))
            else:
                bold_name = name
            pdfmetrics.registerFontFamily(name, normal=name, bold=bold_name)
            return name, bold_name
    return "Helvetica", "Helvetica-Bold"


def clean_md_inline(text: str, font_bold: str) -> str:
    """Convert markdown inline formatting (**bold**, `code`, [link](url), LaTeX symbols) to ReportLab XML."""
    s = text.strip()
    # Replace markdown links [label](url) -> label
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    # Remove emoji / problematic astral glyphs that standard Arial TTF doesn't carry
    s = (
        s.replace("🔴", "[RED] ")
        .replace("🟡", "[YELLOW] ")
        .replace("🟢", "[GREEN] ")
        .replace("📷", "")
        .replace("🔬", "")
        .replace("🧬", "")
        .replace("🚦", "")
        .replace("👤", "")
        .replace("📍", "")
    )
    # Replace common LaTeX inline formulas with readable Unicode
    latex_replacements = {
        r"$\tau_{\text{mel}} = 0.15$": "tau_mel = 0.15",
        r"$\tau_{\text{mel}}=0.15$": "tau_mel = 0.15",
        r"$\tau_{\text{mel}} = 0.05$": "tau_mel = 0.05",
        r"$\tau_{\text{mel}}=0.05$": "tau_mel = 0.05",
        r"$\tau_{\text{ack}} = 0.20$": "tau_ack = 0.20",
        r"$\tau_{\text{mal}} = 0.62$": "tau_mal = 0.62",
        r"$\gamma = 2.5$": "gamma = 2.5",
        r"$\gamma = 0.0$": "gamma = 0.0",
        r"$\alpha = 0.2$": "alpha = 0.2",
        r"$T = 1.581$": "T = 1.581",
        r"$L_2$": "L2",
        r"$\ge 7$": ">= 7",
        r"$\ge 3$": ">= 3",
        r"$\ge 128\times 128$": ">= 128x128",
        r"$\ge 8.0$": ">= 8.0",
        r"$14\times 14$": "14x14",
        r"$16\times 16$": "16x16",
        r"$9\times 9$": "9x9",
        r"$P^2 / (4\pi A)$": "P^2 / (4*pi*A)",
        r"$P(\text{mel}) \ge 0.15$": "P(mel) >= 0.15",
        r"$P(\text{ack}) \ge 0.20$": "P(ack) >= 0.20",
        r"$0.22 \le P(\text{malig}) < 0.62$": "0.22 <= P(malig) < 0.62",
        r"$1\text{–}2$": "1–2",
        r"$w_c \propto N_c^{-0.5}$": "w_c ~ N_c^(-0.5)",
        r"$w_c \propto N_c^{-1}$": "w_c ~ N_c^(-1)",
        r"$p = 0.50$": "p = 0.50",
        r"$p = 1.0$": "p = 1.0",
        r"$(1 - p_c)^{2.5}$": "(1 - p_c)^2.5",
        r"$p_c = 0.90$": "p_c = 0.90",
        r"$0.1^{2.5} = 0.00316$": "0.1^2.5 = 0.00316",
        r"$f(x) = x \cdot \sigma(x)$": "f(x) = x * sigmoid(x)",
        r"$[-1, 1]$": "[-1, 1]",
        r"$[-2, 0]$": "[-2, 0]",
        r"$\sum_{c=1}^{6} P_c = 1.0$": "sum(P_c) = 1.0",
        r"$d = +0.493, p < 10^{-9}$": "d = +0.493, p < 1e-9",
        r"$d = +0.393, p < 10^{-6}$": "d = +0.393, p < 1e-6",
        r"$d = +0.300, p < 10^{-4}$": "d = +0.300, p < 1e-4",
        r"$\Delta a^*$": "Delta a*",
        r"$\text{мм}^2$": "мм²",
        r"$\Delta_{\text{emb}} = 1 - \cos(e_{t_1}, e_{t_2})$": "Delta_emb = 1 - cos(e_t1, e_t2)",
        r"$\Delta_{\text{ABCDE}} = m_{t_2} - m_{t_1}$": "Delta_ABCDE = m_t2 - m_t1",
        r"$(\Delta \text{Asymmetry}, \Delta \text{Border}, \Delta \text{Color Entropy}, \Delta_{\text{emb}})$": "(Delta Asymmetry, Delta Border, Delta Color Entropy, Delta_emb)",
        r"$(\Delta_{\text{evol}} \ge \tau_{\text{evol}})$": "(Delta_evol >= tau_evol)",
        r"$(t_1)$": "(t1)",
        r"$\leftrightarrow$": "<->",
        r"$\to$": "->",
    }
    for k, v in latex_replacements.items():
        s = s.replace(k, v)
    # Strip any remaining inline $...$ and clean LaTeX macros
    s = re.sub(r"\$([^$]+)\$", r"\1", s)
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = (
        s.replace(r"\hat{v} = v / \|v\|_2", "v_norm = v / ||v||_2")
        .replace(r"p_{drop}", "p_drop")
        .replace(r"\_", "_")
        .replace(r"\ge", ">=")
        .replace(r"\le", "<=")
        .replace(r"\in", "в")
        .replace(r"\{", "{")
        .replace(r"\}", "}")
        .replace(r"\gamma", "gamma")
        .replace(r"\Delta", "Δ")
        .replace(r"\tau_{evol}", "tau_evol")
        .replace(r"_{evol}", "_evol")
        .replace(r"_{emb}", "_emb")
        .replace("<br>", "<br/>")
    )

    # Escape XML special chars except <br/>
    s = s.replace("&", "&amp;")
    s = s.replace("<br/>", "__BR__")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    s = s.replace("__BR__", "<br/>")

    # Convert **bold** and `code` (do not strip single * so CIE L*a*b* remains intact)
    s = re.sub(r"\*\*([^*]+(?:\*(?!\*)[^*]*)*)\*\*", rf'<font name="{font_bold}">\1</font>', s)
    s = re.sub(r"`([^`]+)`", rf'<font name="{font_bold}" color="#1E3A8A">\1</font>', s)
    return s


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont(doc.font_name, 9)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawCentredString(A4[0] / 2.0, 20, str(doc.page))
    canvas.restoreState()


def build_pdf(output_paths: List[Path]) -> None:
    font_name, font_bold = register_cyrillic_fonts()
    arch_png = generate_architecture_diagram_png()

    styles = {
        "h1": ParagraphStyle("H1", fontName=font_bold, fontSize=15, leading=19, textColor=colors.HexColor("#0F172A"), spaceBefore=6, spaceAfter=8),
        "h2": ParagraphStyle("H2", fontName=font_bold, fontSize=12.5, leading=16, textColor=colors.HexColor("#0F172A"), spaceBefore=12, spaceAfter=6),
        "h3": ParagraphStyle("H3", fontName=font_bold, fontSize=10.5, leading=14, textColor=colors.HexColor("#1E3A8A"), spaceBefore=8, spaceAfter=4),
        "h4": ParagraphStyle("H4", fontName=font_bold, fontSize=9.5, leading=13, textColor=colors.HexColor("#0F172A"), spaceBefore=6, spaceAfter=3),
        "body": ParagraphStyle("Body", fontName=font_name, fontSize=9.0, leading=12.8, textColor=colors.HexColor("#1E293B"), spaceBefore=2, spaceAfter=5),
        "quote": ParagraphStyle("Quote", fontName=font_name, fontSize=8.5, leading=12, leftIndent=14, rightIndent=10, textColor=colors.HexColor("#334155"), backColor=colors.HexColor("#F1F5F9"), borderPadding=6, spaceBefore=4, spaceAfter=6),
        "bullet": ParagraphStyle("Bullet", fontName=font_name, fontSize=8.8, leading=12.4, leftIndent=14, firstLineIndent=-8, textColor=colors.HexColor("#1E293B"), spaceBefore=1.5, spaceAfter=2.5),
        "code": ParagraphStyle("Code", fontName=font_bold, fontSize=8.2, leading=11.5, leftIndent=16, rightIndent=16, textColor=colors.HexColor("#0F172A"), backColor=colors.HexColor("#F8FAFC"), borderPadding=5, spaceBefore=4, spaceAfter=6),
        "tbl_hdr": ParagraphStyle("TblHdr", fontName=font_bold, fontSize=7.8, leading=10, textColor=colors.HexColor("#0F172A")),
        "tbl_cell": ParagraphStyle("TblCell", fontName=font_name, fontSize=7.6, leading=10, textColor=colors.HexColor("#1E293B")),
    }

    lines = MD_PATH.read_text(encoding="utf-8").splitlines()
    story = []

    i = 0
    in_mermaid = False
    in_math_block = False
    math_buf: List[str] = []

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        # Handle mermaid code block -> replace with generated architecture PNG
        if stripped.startswith("```mermaid"):
            in_mermaid = True
            i += 1
            continue
        if in_mermaid:
            if stripped.startswith("```"):
                in_mermaid = False
                if arch_png.exists():
                    with PILImage.open(arch_png) as pim:
                        w_px, h_px = pim.size
                    target_w = 515.0
                    target_h = target_w * (h_px / float(w_px))
                    story.append(Spacer(1, 4))
                    story.append(RLImage(str(arch_png), width=target_w, height=target_h))
                    story.append(Spacer(1, 6))
            i += 1
            continue

        # Handle $$ math blocks
        if stripped.startswith("$$"):
            if stripped.endswith("$$") and len(stripped) > 4:
                expr = stripped[2:-2].strip()
                expr_clean = (
                    expr.replace(r"\mathcal{L}_{\text{total}}", "L_total")
                    .replace(r"\mathcal{L}_{\text{Focal}}^{(H_1)}(\hat{y}_{\text{triage}}, y_{\text{triage}})", "L_Focal_H1(y_triage)")
                    .replace(r"\mathbb{I}(y_{\text{sub}} \neq -1) \cdot \mathcal{L}_{\text{Focal}}^{(H_2)}(\hat{y}_{\text{sub}}, y_{\text{sub}})", "I(y_sub != -1) * L_Focal_H2(y_sub)")
                    .replace(r"\mathcal{L}_{\text{Focal}}(p_c) = -\alpha_c \cdot (1 - p_c)^{\gamma} \cdot \log(p_c)", "L_Focal(p_c) = -alpha_c * (1 - p_c)^gamma * log(p_c)")
                    .replace(r"\alpha_c = \frac{N_{\text{total}}}{C \cdot N_c^{p}}", "alpha_c = N_total / (C * N_c^p)")
                    .replace(r"P_c = \frac{\exp(z_c / T)}{\sum_{j=1}^{6} \exp(z_j / T)}", "P_c = exp(z_c / T) / sum_j exp(z_j / T)")
                    .replace(r"\text{LayerNorm}", "LayerNorm")
                    .replace(r"\text{Dropout}", "Dropout")
                    .replace(r"\text{Linear}", "Linear")
                    .replace(r"\text{SiLU}", "SiLU")
                    .replace(r"\to", " -> ")
                    .replace(r"\quad", "   ")
                )
                story.append(Paragraph(clean_md_inline(expr_clean, font_bold), styles["code"]))
                i += 1
                continue
            else:
                in_math_block = not in_math_block
                i += 1
                continue

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#CBD5E1")))
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Image embedding ![alt](path)
        img_m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if img_m:
            caption, rel_p = img_m.group(1), img_m.group(2)
            img_path = (MD_PATH.parent / rel_p).resolve()
            if img_path.exists():
                with PILImage.open(img_path) as pim:
                    w_px, h_px = pim.size
                target_w = 515.0
                target_h = target_w * (h_px / float(w_px))
                if target_h > 250.0:
                    scale = 250.0 / target_h
                    target_w *= scale
                    target_h = 250.0
                story.append(Spacer(1, 4))
                story.append(RLImage(str(img_path), width=target_w, height=target_h))
                if caption:
                    story.append(Paragraph(f"<i>Рис.: {clean_md_inline(caption, font_bold)}</i>", styles["tbl_cell"]))
                story.append(Spacer(1, 6))
            i += 1
            continue

        # Markdown Table
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            if len(table_lines) >= 2:
                rows_data = []
                for r_idx, tline in enumerate(table_lines):
                    # Skip separator row |---|---|
                    if re.match(r"^\|[\s:\-|]+\|$", tline):
                        continue
                    cells = [c.strip() for c in tline.strip("|").split("|")]
                    row_styled = []
                    for c in cells:
                        st = styles["tbl_hdr"] if len(rows_data) == 0 else styles["tbl_cell"]
                        row_styled.append(Paragraph(clean_md_inline(c, font_bold), st))
                    rows_data.append(row_styled)
                if rows_data:
                    n_cols = len(rows_data[0])
                    total_w = 520.0
                    if n_cols == 6:
                        col_widths = [22, 85, 85, 65, 65, 198]
                    elif n_cols == 5:
                        col_widths = [85, 85, 80, 95, 175]
                    elif n_cols == 3:
                        col_widths = [80, 125, 315]
                    else:
                        col_widths = [total_w / n_cols] * n_cols
                    t_obj = Table(rows_data, colWidths=col_widths, repeatRows=1)
                    t_obj.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                            ]
                        )
                    )
                    story.append(Spacer(1, 4))
                    story.append(t_obj)
                    story.append(Spacer(1, 6))
            continue

        # Headings
        if stripped.startswith("# "):
            story.append(Paragraph(clean_md_inline(stripped[2:], font_bold), styles["h1"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(clean_md_inline(stripped[3:], font_bold), styles["h2"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(clean_md_inline(stripped[4:], font_bold), styles["h3"]))
        elif stripped.startswith("#### "):
            story.append(Paragraph(clean_md_inline(stripped[5:], font_bold), styles["h4"]))
        elif stripped.startswith("> "):
            story.append(Paragraph(clean_md_inline(stripped[2:], font_bold), styles["quote"]))
        elif re.match(r"^(\-|\*)\s+", stripped):
            content = re.sub(r"^(\-|\*)\s+", "", stripped)
            story.append(Paragraph(f"• {clean_md_inline(content, font_bold)}", styles["bullet"]))
        elif re.match(r"^\d+\.\s+", stripped):
            m_num = re.match(r"^(\d+\.)\s+(.*)$", stripped)
            if m_num:
                story.append(
                    Paragraph(
                        f'<font name="{font_bold}">{m_num.group(1)}</font> {clean_md_inline(m_num.group(2), font_bold)}',
                        styles["bullet"],
                    )
                )
        else:
            story.append(Paragraph(clean_md_inline(stripped, font_bold), styles["body"]))

        i += 1

    for out_p in output_paths:
        doc = SimpleDocTemplate(
            str(out_p),
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=32,
            bottomMargin=36,
            title="Документация к дипломному проекту — SkinAI",
            author="SkinAI Engineering & Research Team",
        )
        doc.font_name = font_name
        # Build with a fresh copy of flowables per output file if needed
        if out_p == output_paths[0]:
            doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
        else:
            import shutil
            shutil.copy2(output_paths[0], out_p)


if __name__ == "__main__":
    build_pdf([PDF_DOCS_PATH, PDF_ROOT_PATH])
    print(f"[+] Generated PDF: {PDF_DOCS_PATH} ({PDF_DOCS_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"[+] Copied PDF to root: {PDF_ROOT_PATH}")
