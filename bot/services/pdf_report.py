"""Clinical Encounter Summary PDF Generator for SkinAI.

Generates a clean, professional, non-diagnostic PDF memo intended for patients
to bring to an in-person dermatology consultation.

Strict Clinical Safety Guardrails (Peer Review Consensus):
- ZERO model probabilities or raw algorithm percentages to prevent doctor cognitive anchoring.
- Photo documentation with timestamp and reference ID.
- NICE NG12 Glasgow 7-point anamnesis checklist results.
- Standard clinical questions for patient-doctor discussion.
- Prominent non-diagnostic research disclaimer.
- Full support for Russian (ru), English (en), and Korean (ko).
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
import logging
import os
from typing import Any, Dict, Optional, Tuple

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

logger = logging.getLogger("skinai.bot.pdf")

# Localization texts for clinical memo
REPORT_TEXTS: Dict[str, Dict[str, Any]] = {
    "ru": {
        "title": "Памятка для очного приёма врача-дерматолога",
        "subtitle": "Сводка анамнеза и фотофиксация новообразования кожи",
        "badge": "ИССЛЕДОВАТЕЛЬСКИЙ ПРОТОТИП • НЕ ЯВЛЯЕТСЯ МЕДИЦИНСКИМ ЗАКЛЮЧЕНИЕМ",
        "meta_date": "Дата и время фиксации:",
        "meta_id": "ID исследования:",
        "meta_profile": "Тип фотоснимка:",
        "profile_smartphone": "Клиническое макрофото (смартфон)",
        "profile_dermoscopy": "Дерматоскопия",
        "photo_title": "Фотофиксация элемента кожи",
        "photo_caption": "Исходный снимок образования, переданный пациентом",
        "anamnesis_title": "Опросник динамики (Шкала Глазго / NICE NG12)",
        "feature_col": "Клинический признак",
        "weight_col": "Вес",
        "status_col": "Ответ",
        "q_change_in_size": "Увеличение размера / динамика роста",
        "q_irregular_shape": "Асимметричные / неровные контуры",
        "q_irregular_color": "Неоднородный цвет (оттенки, вкрапления)",
        "q_diameter_ge_7mm": "Наибольший диаметр ≥ 7 мм",
        "q_inflammation": "Воспаление / краснота вокруг",
        "q_oozing_or_bleeding": "Мокнутие / корочки / кровоточивость",
        "q_change_in_sensation": "Зуд / жжение / болезненность",
        "ans_yes": "Да",
        "ans_no": "Нет",
        "ans_none": "Не указано",
        "total_score_label": "Суммарный балл по шкале Глазго:",
        "threshold_note": "Порог клинической настороженности NICE NG12: ≥ 3 баллов",
        "questions_title": "Рекомендуемые вопросы врачу-дерматологу на очном приёме:",
        "questions": [
            "1. Требуется ли проведение оптической или цифровой дерматоскопии данного элемента?",
            "2. Рекомендовано ли динамическое наблюдение (повторный снимок / картирование через 3–6 месяцев)?",
            "3. Имеются ли показания для эксцизионной биопсии и гистологического исследования?",
        ],
        "disclaimer_title": "Ограничение ответственности и назначение документа:",
        "disclaimer": (
            "Данный документ сформирован автоматически экспериментальной системой SkinAI "
            "для упорядочивания анамнеза перед очным визитом к врачу. Документ НЕ является "
            "диагнозом и НЕ заменяет приём сертифицированного дерматолога или онколога. "
            "Согласно решению консилиума, вероятности нейросети исключены для предотвращения "
            "когнитивного якорения врача."
        ),
    },
    "en": {
        "title": "Clinical Summary for Dermatologist Consultation",
        "subtitle": "Lesion photodocumentation and clinical history summary",
        "badge": "RESEARCH PROTOTYPE • NOT A MEDICAL DIAGNOSIS",
        "meta_date": "Timestamp (UTC):",
        "meta_id": "Reference ID:",
        "meta_profile": "Image Modality:",
        "profile_smartphone": "Smartphone Clinical Macro",
        "profile_dermoscopy": "Dermoscopy",
        "photo_title": "Lesion Photodocumentation",
        "photo_caption": "Patient-submitted close-up photograph",
        "anamnesis_title": "Clinical History (NICE NG12 / Glasgow 7-Point)",
        "feature_col": "Clinical Feature",
        "weight_col": "Weight",
        "status_col": "Status",
        "q_change_in_size": "Recent change in size / growth",
        "q_irregular_shape": "Irregular / asymmetric border",
        "q_irregular_color": "Color variegation / uneven shades",
        "q_diameter_ge_7mm": "Largest diameter ≥ 7 mm",
        "q_inflammation": "Inflammation / surrounding erythema",
        "q_oozing_or_bleeding": "Oozing / crusting / bleeding",
        "q_change_in_sensation": "Itching / tenderness / pain",
        "ans_yes": "Yes",
        "ans_no": "No",
        "ans_none": "Not specified",
        "total_score_label": "Total Glasgow Score:",
        "threshold_note": "NICE NG12 clinical suspicion threshold: ≥ 3 points",
        "questions_title": "Suggested Questions for Doctor Consultation:",
        "questions": [
            "1. Is in-person dermoscopic examination indicated for this lesion?",
            "2. Is digital monitoring / follow-up mapping recommended in 3–6 months?",
            "3. Are there grounds for excisional biopsy and histopathology?",
        ],
        "disclaimer_title": "Clinical & Legal Disclaimer:",
        "disclaimer": (
            "This summary is generated by the experimental SkinAI system solely to facilitate "
            "structured patient-clinician communication. It is NOT a medical device and DOES NOT "
            "provide a diagnosis. Algorithmic prediction scores are intentionally omitted to "
            "avoid clinical anchoring bias."
        ),
    },
    "ko": {
        "title": "피부과 대면 진료 참고용 임상 요약서",
        "subtitle": "병변 사진 기록 및 환자 병력 요약",
        "badge": "연구용 프로토타입 • 공식 진단서가 아님",
        "meta_date": "작성 일시 (UTC):",
        "meta_id": "문서 식별 번호:",
        "meta_profile": "촬영 모드:",
        "profile_smartphone": "스마트폰 임상 근접 사진",
        "profile_dermoscopy": "더모스코피 (피부확대경)",
        "photo_title": "병변 사진 기록",
        "photo_caption": "환자 제출 원본 근접 사진",
        "anamnesis_title": "병력 문진 (Glasgow 7-Point / NICE NG12)",
        "feature_col": "임상 항목",
        "weight_col": "배점",
        "status_col": "응답",
        "q_change_in_size": "최근 크기 증가 또는 새로운 병변",
        "q_irregular_shape": "비대칭적이거나 불규칙한 가장자리",
        "q_irregular_color": "균일하지 않고 얼룩덜룩한 색조",
        "q_diameter_ge_7mm": "가장 긴 직경 7 mm 이상",
        "q_inflammation": "병변 주변 홍반 또는 염증 반응",
        "q_oozing_or_bleeding": "진물, 딱지(가피) 또는 출혈",
        "q_change_in_sensation": "가려움증, 따가움 또는 통증",
        "ans_yes": "예",
        "ans_no": "아니오",
        "ans_none": "미입력",
        "total_score_label": "글래스고 총점:",
        "threshold_note": "NICE NG12 전문의 진료 의뢰 권고 기준: ≥ 3점",
        "questions_title": "전문의 진료 시 추천 상담 질문:",
        "questions": [
            "1. 해당 병변에 대해 더모스코피(피부확대경) 정밀 검사가 필요한가요?",
            "2. 3~6개월 후 정기적인 추적 관찰(디지털 사진 기록)이 필요한가요?",
            "3. 조직 검사(절제 생검)가 필요한 임상적 소견이 있나요?",
        ],
        "disclaimer_title": "중요 임상 및 법적 고지사항:",
        "disclaimer": (
            "본 문서는 전문의 대면 진료 시 환자의 상태 설명을 돕기 위해 SkinAI 연구 시스템에 의해 "
            "자동 생성되었습니다. 본 문서는 의료기기나 법적 진단서가 아니며 전문의 대면 진료를 "
            "대신할 수 없습니다. 의사의 선입견(인지적 고착화)을 방지하기 위해 인공지능 모델 확률 수치는 "
            "의도적으로 본 요약서에서 제외되었습니다."
        ),
    },
}

# Font registration cache
_FONT_FAMILY_INITIALIZED = False
_ACTIVE_FONT_NAME = "Helvetica"
_ACTIVE_FONT_BOLD = "Helvetica-Bold"


def _initialize_pdf_fonts() -> Tuple[str, str]:
    """Discover and register system TrueType fonts supporting Cyrillic and Korean."""
    global _FONT_FAMILY_INITIALIZED, _ACTIVE_FONT_NAME, _ACTIVE_FONT_BOLD

    if _FONT_FAMILY_INITIALIZED:
        return _ACTIVE_FONT_NAME, _ACTIVE_FONT_BOLD

    # Candidates prioritized by multilingual coverage
    font_candidates = [
        # Windows standard (covers Korean, Cyrillic, Latin)
        ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf", "SkinAIMalgun"),
        # Windows Arial
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf", "SkinAIArial"),
        # Linux standard (Debian/Ubuntu)
        (
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
            "SkinAINanum",
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "SkinAIDejaVu",
        ),
    ]

    for reg_path, bold_path, base_name in font_candidates:
        if os.path.exists(reg_path):
            try:
                pdfmetrics.registerFont(TTFont(base_name, reg_path))
                if os.path.exists(bold_path):
                    bold_name = f"{base_name}-Bold"
                    pdfmetrics.registerFont(TTFont(bold_name, bold_path))
                else:
                    bold_name = base_name

                pdfmetrics.registerFontFamily(base_name, normal=base_name, bold=bold_name)
                _ACTIVE_FONT_NAME = base_name
                _ACTIVE_FONT_BOLD = bold_name
                _FONT_FAMILY_INITIALIZED = True
                logger.info(f"Registered PDF font: {_ACTIVE_FONT_NAME} from {reg_path}")
                return _ACTIVE_FONT_NAME, _ACTIVE_FONT_BOLD
            except Exception as exc:
                logger.warning(f"Failed to register font {reg_path}: {exc}")

    _FONT_FAMILY_INITIALIZED = True
    return _ACTIVE_FONT_NAME, _ACTIVE_FONT_BOLD


def generate_clinical_memo_pdf(
    image_bytes: bytes,
    lang: str = "ru",
    request_id: Optional[str] = None,
    created_at: Optional[datetime] = None,
    profile: str = "smartphone_clinical",
    glasgow_answers: Optional[Dict[str, Optional[bool]]] = None,
    glasgow_score: Optional[int] = None,
) -> bytes:
    """Generate in-memory binary PDF document for patient-doctor clinical consultation."""
    font_name, font_bold = _initialize_pdf_fonts()
    texts = REPORT_TEXTS.get(lang, REPORT_TEXTS["en"])

    if not created_at:
        created_at = datetime.now(timezone.utc)
    date_str = created_at.strftime("%Y-%m-%d %H:%M UTC")

    ref_id = request_id[:13] if request_id else "SN-" + datetime.now(timezone.utc).strftime("%y%m%d%H%M")
    profile_label = texts["profile_smartphone"] if profile == "smartphone_clinical" else texts["profile_dermoscopy"]

    # PDF Document configuration
    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buf,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32,
    )

    # Styles
    title_style = ParagraphStyle(
        "DocTitle",
        fontName=font_bold,
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0F172A"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName=font_name,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
    )
    badge_style = ParagraphStyle(
        "BadgeText",
        fontName=font_bold,
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#0F766E"),
    )
    section_h2 = ParagraphStyle(
        "SectionH2",
        fontName=font_bold,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1E3A8A"),
    )
    body_style = ParagraphStyle(
        "BodyTextRegular",
        fontName=font_name,
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1E293B"),
    )
    bold_style = ParagraphStyle(
        "BodyTextBold",
        fontName=font_bold,
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0F172A"),
    )
    caption_style = ParagraphStyle(
        "PhotoCaption",
        fontName=font_name,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#64748B"),
        alignment=1,  # Center
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        fontName=font_name,
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#334155"),
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph(f"<b>SkinAI</b> &nbsp;|&nbsp; {texts['badge']}", badge_style),
        ],
        [
            Paragraph(texts["title"], title_style),
        ],
        [
            Paragraph(texts["subtitle"], subtitle_style),
        ],
    ]
    header_table = Table(header_data, colWidths=[523])
    header_table.setStyle(
        TableStyle([
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    story.append(header_table)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0F766E"), spaceAfter=8))

    # 2. Metadata Grid
    meta_table_data = [
        [
            Paragraph(f"<b>{texts['meta_date']}</b> {date_str}", body_style),
            Paragraph(f"<b>{texts['meta_profile']}</b> {profile_label}", body_style),
        ],
        [
            Paragraph(f"<b>{texts['meta_id']}</b> <code>{ref_id}</code>", body_style),
            Paragraph("<b>Опросник:</b> Glasgow 7-Point (NICE NG12)", body_style),
        ],
    ]
    meta_table = Table(meta_table_data, colWidths=[260, 263])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 3. Two-Column Main Content: [Photo on Left] and [Glasgow Anamnesis on Right]
    # Resize lesion photo to fit neatly
    photo_flowable = None
    try:
        pil_img = PILImage.open(io.BytesIO(image_bytes))
        # Keep aspect ratio within 185 x 185 pt
        max_dim = 185.0
        w, h = pil_img.size
        scale = min(max_dim / w, max_dim / h)
        disp_w = w * scale
        disp_h = h * scale

        img_temp_buf = io.BytesIO()
        pil_img.save(img_temp_buf, format="JPEG", quality=88)
        img_temp_buf.seek(0)
        photo_flowable = RLImage(img_temp_buf, width=disp_w, height=disp_h)
    except Exception as exc:
        logger.warning(f"Could not prepare image for PDF: {exc}")
        photo_flowable = Paragraph("[Фотография недоступна]", body_style)

    # Left Column: Photo Card
    photo_box_data = [
        [Paragraph(f"<b>{texts['photo_title']}</b>", section_h2)],
        [Spacer(1, 4)],
        [photo_flowable],
        [Spacer(1, 4)],
        [Paragraph(texts["photo_caption"], caption_style)],
    ]
    photo_box = Table(photo_box_data, colWidths=[210])
    photo_box.setStyle(
        TableStyle([
            ("ALIGN", (0, 2), (0, 2), "CENTER"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )

    # Right Column: Glasgow Checklist Table
    answers = glasgow_answers or {}
    questions_def = [
        ("q_change_in_size", "+2", answers.get("change_in_size")),
        ("q_irregular_shape", "+2", answers.get("irregular_shape")),
        ("q_irregular_color", "+2", answers.get("irregular_color")),
        ("q_diameter_ge_7mm", "+1", answers.get("diameter_ge_7mm")),
        ("q_inflammation", "+1", answers.get("inflammation")),
        ("q_oozing_or_bleeding", "+1", answers.get("oozing_or_bleeding")),
        ("q_change_in_sensation", "+1", answers.get("change_in_sensation")),
    ]

    glasgow_rows = [
        [
            Paragraph(f"<b>{texts['feature_col']}</b>", bold_style),
            Paragraph(f"<b>{texts['weight_col']}</b>", bold_style),
            Paragraph(f"<b>{texts['status_col']}</b>", bold_style),
        ]
    ]

    computed_score = 0
    for q_key, weight_str, ans_val in questions_def:
        label = texts.get(q_key, q_key)
        if ans_val is True:
            ans_str = f"<font color='#B91C1C'><b>{texts['ans_yes']}</b></font>"
            computed_score += 2 if weight_str == "+2" else 1
        elif ans_val is False:
            ans_str = f"<font color='#15803D'>{texts['ans_no']}</font>"
        else:
            ans_str = f"<font color='#64748B'>{texts['ans_none']}</font>"

        glasgow_rows.append([
            Paragraph(label, body_style),
            Paragraph(weight_str, body_style),
            Paragraph(ans_str, body_style),
        ])

    final_score = glasgow_score if glasgow_score is not None else computed_score
    score_color = "#B91C1C" if final_score >= 3 else ("#D97706" if final_score > 0 else "#15803D")

    glasgow_table = Table(glasgow_rows, colWidths=[180, 42, 70])
    glasgow_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ])
    )

    right_col_data = [
        [Paragraph(f"<b>{texts['anamnesis_title']}</b>", section_h2)],
        [Spacer(1, 4)],
        [glasgow_table],
        [Spacer(1, 4)],
        [
            Paragraph(
                f"<b>{texts['total_score_label']}</b> <font color='{score_color}' size='+1.5'><b>{final_score} / 10</b></font>",
                body_style,
            )
        ],
        [Paragraph(f"<i>{texts['threshold_note']}</i>", caption_style)],
    ]
    right_col_box = Table(right_col_data, colWidths=[300])
    right_col_box.setStyle(
        TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )

    # Combine into side-by-side table
    main_grid = Table([[photo_box, right_col_box]], colWidths=[215, 308])
    main_grid.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    story.append(main_grid)
    story.append(Spacer(1, 10))

    # 4. Doctor Discussion Questions Box
    q_items = [Paragraph(f"<b>{texts['questions_title']}</b>", section_h2), Spacer(1, 3)]
    for q in texts["questions"]:
        q_items.append(Paragraph(f"• {q}", body_style))
        q_items.append(Spacer(1, 2))

    questions_card = Table([[q_items]], colWidths=[523])
    questions_card.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),  # Soft light mint green
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BBF7D0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    story.append(questions_card)
    story.append(Spacer(1, 10))

    # 5. Non-Diagnostic Disclaimer Box
    disclaimer_items = [
        Paragraph(f"<b>{texts['disclaimer_title']}</b>", bold_style),
        Spacer(1, 2),
        Paragraph(texts["disclaimer"], disclaimer_style),
    ]
    disclaimer_card = Table([[disclaimer_items]], colWidths=[523])
    disclaimer_card.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),  # Soft light red/rose
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#FECACA")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    story.append(KeepTogether(disclaimer_card))

    # Build PDF
    doc.build(story)
    pdf_bytes = pdf_buf.getvalue()
    pdf_buf.close()

    logger.info(f"Generated doctor clinical memo PDF ({len(pdf_bytes)} bytes, lang={lang}, ref={ref_id})")
    return pdf_bytes
