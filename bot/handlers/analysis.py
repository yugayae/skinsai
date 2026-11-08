"""Image analysis and Glasgow checklist interactive workflow handler."""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

from aiogram import Bot, F, Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, CallbackQuery, InlineKeyboardMarkup, Message

from bot.handlers.common import (
    get_user_lang,
    get_user_profile,
    is_disclaimer_accepted,
    is_user_allowed,
)
from bot.keyboards.inline import (
    get_age_group_keyboard,
    get_analysis_result_keyboard,
    get_anatom_site_keyboard,
    get_disclaimer_keyboard,
    get_glasgow_prompt_keyboard,
    get_glasgow_question_keyboard,
)
from bot.locales.loader import t
from bot.services.api_client import SkinAIClient
from bot.services.pdf_report import generate_clinical_memo_pdf

logger = logging.getLogger("skinai.bot.analysis")
router = Router(name="analysis")

# Initialize shared API client
api_client = SkinAIClient()

# Glasgow Checklist Question definitions
GLASGOW_QUESTIONS: List[Tuple[str, str]] = [
    ("change_in_size", "q_change_in_size"),
    ("irregular_shape", "q_irregular_shape"),
    ("irregular_color", "q_irregular_color"),
    ("diameter_ge_7mm", "q_diameter_ge_7mm"),
    ("inflammation", "q_inflammation"),
    ("oozing_or_bleeding", "q_oozing_or_bleeding"),
    ("change_in_sensation", "q_change_in_sensation"),
]


class AnalysisStates(StatesGroup):
    """FSM states for multi-modal analysis flow."""

    waiting_for_glasgow_choice = State()
    answering_glasgow = State()
    result_ready = State()


def _format_prediction_response(data: Dict[str, Any], lang: str) -> str:
    """Format structured prediction response into clean Markdown message."""
    status = data.get("status")
    if lang == "ru":
        disclaimer = data.get("disclaimer_ru")
    elif lang == "ko":
        disclaimer = data.get("disclaimer_ko")
    else:
        disclaimer = data.get("disclaimer_en")

    if not disclaimer:
        disclaimer = data.get("disclaimer_ru") or data.get("disclaimer_en", "")

    if status == "rejected_quality":
        raw_reason = data.get("rejection_reason", "low_quality")
        if "excessive_blur" in raw_reason:
            localized_reason = t("rejection_excessive_blur", lang)
        elif "dimension_too_small" in raw_reason:
            localized_reason = t("rejection_dimension_too_small", lang)
        elif "payload_exceeds_max_size" in raw_reason:
            localized_reason = t("rejection_payload_exceeds_max_size", lang)
        elif "invalid_magic_bytes" in raw_reason or "decode_error" in raw_reason:
            localized_reason = t("rejection_corrupted", lang)
        else:
            localized_reason = raw_reason

        reason_label = t("rejection_reason_label", lang)
        text_lines = [
            f"{t('status_rejected_quality_title', lang)}",
            "",
            f"🔍 **{reason_label}:** {localized_reason}",
            "",
            f"{t('quality_advice', lang)}",
        ]
        if disclaimer:
            text_lines.extend(["", "---", f"_{disclaimer}_"])
        return "\n".join(text_lines)

    if status == "abstained":
        rej_reason = data.get("rejection_reason") or ""
        if "out_of_distribution_non_skin" in rej_reason:
            abstain_msg = t("abstain_ood_non_skin", lang)
        else:
            abstain_msg = t("abstain_low_confidence", lang)
        text_lines = [
            f"{t('status_abstained_title', lang)}",
            "",
            abstain_msg,
            "",
            f"{t('quality_advice', lang)}",
        ]
        if disclaimer:
            text_lines.extend(["", "---", f"_{disclaimer}_"])
        return "\n".join(text_lines)

    # Success outcome
    if lang == "ru":
        class_name = data.get("predicted_class_name_ru")
    elif lang == "ko":
        class_name = data.get("predicted_class_name_ko")
    else:
        class_name = data.get("predicted_class_name_en")

    if not class_name:
        class_name = data.get("predicted_class_name_en") or data.get("predicted_class_id", "Unknown")

    confidence = data.get("confidence", 0.0) * 100

    lbl_top = "Наиболее вероятная категория" if lang == "ru" else ("가장 유력한 범주" if lang == "ko" else "Most probable category")
    lbl_dist = "Распределение модельных вероятностей" if lang == "ru" else ("모델 확률 분포" if lang == "ko" else "Model probability distribution")

    lines = [
        f"{t('status_success_title', lang)}",
        f"🎯 **{lbl_top}:** {class_name} ({confidence:.1f}%)",
        "",
        f"📈 **{lbl_dist}:**",
    ]

    # Show top class breakdown
    probs = data.get("class_probabilities", {})
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for cls_id, p_val in sorted_probs:
        p_pct = p_val * 100
        lines.append(f"  • `{cls_id.upper()}`: {p_pct:.1f}%")

    # Add Quantitative ABCDE Morphometry (DS-5) if available
    abcde = data.get("abcde_metrics")
    if abcde:
        asym_s = abcde.get("asymmetry_shape", 0.0)
        border_irr = abcde.get("border_irregularity", 1.0)
        c_ent = abcde.get("color_entropy", 0.0)
        c_zones = abcde.get("distinct_color_clusters", 1)
        if lang == "ru":
            lines.extend([
                "",
                "🔬 **Морфометрия очага (ABCDE + GrabCut):**",
                f"  • `A` Асимметрия формы: `{asym_s:.2f}`",
                f"  • `B` Неровность границ: `{border_irr:.2f}`",
                f"  • `C` Цветовая энтропия CIE Lab: `{c_ent:.2f}` (зон: `{c_zones}/6`)",
            ])
        elif lang == "ko":
            lines.extend([
                "",
                "🔬 **정량적 ABCDE 형태 계측 (GrabCut):**",
                f"  • `A` 형태 비대칭도: `{asym_s:.2f}` | `B` 경계 불규칙도: `{border_irr:.2f}`",
                f"  • `C` CIE Lab 색 엔트로피: `{c_ent:.2f}` (색상 구역: `{c_zones}/6`)",
            ])
        else:
            lines.extend([
                "",
                "🔬 **Quantitative ABCDE Morphometry (GrabCut):**",
                f"  • `A` Shape Asymmetry: `{asym_s:.2f}`",
                f"  • `B` Border Irregularity: `{border_irr:.2f}`",
                f"  • `C` CIE Lab Color Entropy: `{c_ent:.2f}` (zones: `{c_zones}/6`)",
            ])

    # Add Deterministic XAI #2: Top-3 Biopsy Atlas k-NN Twins if available
    twins = data.get("similar_biopsy_cases")
    if twins:
        lines.append("")
        lines.append(t("biopsy_atlas_title", lang))
        sim_label = t("biopsy_similarity", lang)
        for tw in twins[:3]:
            sub_u = str(tw.get("subtype_id", "")).upper()
            sim_pct = float(tw.get("cosine_similarity", 0.0)) * 100.0
            raw_tag = str(tw.get("malignancy_label", "")).lower()
            if "mal" in raw_tag or "зло" in raw_tag or "악성" in raw_tag:
                mal_tag = t("label_malignant", lang)
            else:
                mal_tag = t("label_benign", lang)
            raw_id = str(tw.get("image_id", "")).replace("_", " ")
            lines.append(f"  • `#{tw.get('rank', 1)}` `{sub_u}` ({mal_tag}) — {sim_label} `{sim_pct:.1f}%` ({raw_id})")

    # Add clinical anamnesis note if present
    anamnesis_eval = data.get("anamnesis_evaluation")
    if anamnesis_eval:
        score = anamnesis_eval.get("score", 0)
        if lang == "ru":
            note = anamnesis_eval.get("clinical_note_ru")
            pts_label = "баллов"
            g_title = "Опросник Глазго"
        elif lang == "ko":
            note = anamnesis_eval.get("clinical_note_ko")
            pts_label = "점"
            g_title = "글래스고 문진표"
        else:
            note = anamnesis_eval.get("clinical_note_en")
            pts_label = "pts"
            g_title = "Glasgow Checklist"
        if not note:
            note = anamnesis_eval.get("clinical_note_en") or ""
        lines.extend([
            "",
            f"📋 **{g_title}:** {score}/10 {pts_label}",
            f"_{note}_",
        ])

    # Clinical Triage Engine result
    triage = data.get("triage")
    if triage:
        if lang == "ru":
            title = triage.get("title_ru")
            rec = triage.get("recommendation_ru")
            lbl_triage = "Клиническая маршрутизация"
            lbl_days = f"⏱ Срок консультации: до {triage.get('urgency_days')} дн."
            lbl_rationale = "📌 *Обоснование:*"
        elif lang == "ko":
            title = triage.get("title_ko")
            rec = triage.get("recommendation_ko")
            lbl_triage = "임상적 조치 권고"
            lbl_days = f"⏱ 권고 기한: {triage.get('urgency_days')}일 이내"
            lbl_rationale = "📌 *판정 근거:*"
        else:
            title = triage.get("title_en")
            rec = triage.get("recommendation_en")
            lbl_triage = "Clinical Triage"
            lbl_days = f"⏱ Recommended timeframe: within {triage.get('urgency_days')} days"
            lbl_rationale = "📌 *Rationale:*"

        if not title:
            title = triage.get("title_en") or triage.get("tier", "")
        if not rec:
            rec = triage.get("recommendation_en") or ""

        days = triage.get("urgency_days")
        zone = triage.get("triage_zone", "yellow")
        zone_badge = {"red": "🔴 RED ZONE", "yellow": "🟡 YELLOW ZONE", "green": "🟢 GREEN ZONE"}.get(zone, "🟡 YELLOW ZONE")
        rationales = triage.get("rationales_ru" if lang == "ru" else ("rationales_ko" if lang == "ko" else "rationales_en"), triage.get("rationale", []))

        lines.extend([
            "",
            f"🚦 **{lbl_triage} [{zone_badge}]:** {title}",
            f"💡 {rec}",
        ])
        if days:
            lines.append(lbl_days)
        if rationales:
            lines.append(lbl_rationale)
            for r in rationales:
                lines.append(f"  • {r}")

    # Referral recommendation
    if data.get("referral_recommended"):
        lines.extend([
            "",
            f"{t('referral_alert_title', lang)}",
            f"{t('referral_alert_body', lang)}",
        ])
        reason = data.get("referral_reason")
        if reason:
            lines.append(f"*(Основание: {reason})*")

    lbl_time = "Время анализа" if lang == "ru" else ("분석 소요 시간" if lang == "ko" else "Analysis latency")
    ms_label = "мс" if lang in ("ru", "ko") else "ms"
    lines.extend([
        "",
        f"⏱ {lbl_time}: {data.get('latency_ms', 0):.0f} {ms_label}",
    ])
    if disclaimer:
        lines.extend([
            "---",
            f"_{disclaimer}_",
        ])

    return "\n".join(lines)


async def _safe_edit_result(
    status_msg: Message,
    result_text: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
    lang: str = "ru",
) -> None:
    """Safely edit status message with Markdown; fall back to clean plain text if entity parsing fails."""
    try:
        await status_msg.edit_text(result_text, reply_markup=reply_markup, parse_mode="Markdown")
    except Exception as parse_err:
        logger.warning(f"Telegram Markdown parse failed: {parse_err}. Falling back to plain text presentation.")
        clean_text = (
            result_text.replace("**", "")
            .replace("*", "")
            .replace("`", "")
            .replace("___", "")
            .replace("__", "")
            .replace("_", "")
        )
        try:
            await status_msg.edit_text(clean_text, reply_markup=reply_markup, parse_mode=None)
        except Exception as edit_err:
            logger.error(f"Failed to edit status message in plain text: {edit_err}")
            await status_msg.edit_text(t("error_connection", lang))


@router.message(F.photo | F.document)
async def handle_lesion_photo(message: Message, state: FSMContext, bot: Bot) -> None:
    """Ingest skin lesion image into ephemeral memory and offer optional Glasgow anamnesis."""
    user_id = message.from_user.id if message.from_user else 0

    lang = get_user_lang(user_id)

    if not is_user_allowed(user_id):
        await message.answer(
            t("pilot_mode_restricted", lang, user_id=user_id),
            parse_mode="Markdown",
        )
        return

    # Check terms agreement
    if not is_disclaimer_accepted(user_id):
        text = f"{t('welcome_title', lang)}\n\n{t('disclaimer_text', lang)}"
        await message.answer(text, reply_markup=get_disclaimer_keyboard(lang), parse_mode="Markdown")
        return

    # Download file in memory (ZERO disk writes)
    buf = io.BytesIO()
    if message.photo:
        file_id = message.photo[-1].file_id
    elif message.document:
        if not (message.document.mime_type and message.document.mime_type.startswith("image/")):
            await message.answer(t("invalid_image_format", lang))
            return
        file_id = message.document.file_id
    else:
        return

    file = await bot.get_file(file_id)
    if not file.file_path:
        await message.answer(t("error_generic", lang))
        return

    await bot.download_file(file.file_path, buf)
    image_bytes = buf.getvalue()

    # Store bytes in FSM session
    await state.update_data(
        image_bytes=image_bytes,
        glasgow_answers={},
        current_question_idx=0,
    )
    await state.set_state(AnalysisStates.waiting_for_glasgow_choice)

    await message.answer(
        t("ask_glasgow_prompt", lang),
        reply_markup=get_glasgow_prompt_keyboard(lang),
        parse_mode="Markdown",
    )


@router.callback_query(F.data == "glasgow:skip")
async def cb_skip_glasgow(callback: CallbackQuery, state: FSMContext) -> None:
    """Skip anamnesis questionnaire and trigger direct image inference."""
    try:
        await callback.answer()
    except Exception:
        pass

    user_id = callback.from_user.id
    lang = get_user_lang(user_id)
    profile = get_user_profile(user_id)

    data = await state.get_data()
    image_bytes = data.get("image_bytes")

    if not image_bytes:
        if callback.message:
            await callback.message.edit_text(
                t("session_expired", lang),
                parse_mode="Markdown",
            )
        return

    if callback.message:
        status_msg = await callback.message.edit_text(t("processing_image", lang))
        try:
            resp = await api_client.predict_image(
                image_bytes=image_bytes,
                profile=profile,
                user_id=user_id,
                anamnesis_dict=None,
            )
            result_text = _format_prediction_response(resp, lang)
            kb = get_analysis_result_keyboard(lang) if resp.get("status") == "success" else None
            await _safe_edit_result(status_msg, result_text, reply_markup=kb, lang=lang)

            # Preserve ephemeral state for optional doctor PDF generation
            await state.set_state(AnalysisStates.result_ready)
            await state.update_data(last_prediction=resp, glasgow_answers={})
        except Exception as exc:
            logger.error(f"Inference API request failed: {exc}")
            await status_msg.edit_text(t("error_connection", lang))


@router.callback_query(F.data == "glasgow:start")
async def cb_start_glasgow(callback: CallbackQuery, state: FSMContext) -> None:
    """Start interactive Glasgow 7-point checklist questionnaire."""
    try:
        await callback.answer()
    except Exception:
        pass

    user_id = callback.from_user.id
    lang = get_user_lang(user_id)

    data = await state.get_data()
    if not data.get("image_bytes"):
        if callback.message:
            await callback.message.edit_text(
                t("session_expired", lang),
                parse_mode="Markdown",
            )
        return

    await state.set_state(AnalysisStates.answering_glasgow)
    await state.update_data(current_question_idx=0, glasgow_answers={})

    field_name, question_key = GLASGOW_QUESTIONS[0]
    prompt_text = t(
        "glasgow_question_step",
        lang,
        current=1,
        total=len(GLASGOW_QUESTIONS),
        question=t(question_key, lang),
    )

    if callback.message:
        try:
            await callback.message.edit_text(
                prompt_text,
                reply_markup=get_glasgow_question_keyboard(lang, step_idx=0),
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.warning(f"Failed to edit message in cb_start_glasgow: {e}")


@router.callback_query(F.data.startswith("g_ans:"))
async def cb_answer_glasgow_question(callback: CallbackQuery, state: FSMContext) -> None:
    """Process an answer to a checklist question and progress to next question or final inference."""
    try:
        await callback.answer()
    except Exception:
        pass

    parts = callback.data.split(":")
    if len(parts) < 3:
        return
    step_idx = int(parts[1])
    ans = parts[2]  # "yes", "no", or "skip"

    user_id = callback.from_user.id
    lang = get_user_lang(user_id)
    profile = get_user_profile(user_id)

    data = await state.get_data()
    answers = data.get("glasgow_answers", {})
    image_bytes = data.get("image_bytes")

    if not image_bytes:
        if callback.message:
            await callback.message.edit_text(
                t("session_expired", lang),
                parse_mode="Markdown",
            )
        return

    # Record answer for the current question
    if step_idx < len(GLASGOW_QUESTIONS):
        field_name = GLASGOW_QUESTIONS[step_idx][0]
        if ans == "yes":
            answers[field_name] = True
        elif ans == "no":
            answers[field_name] = False
        # if skip: remains unrecorded / None

    next_idx = step_idx + 1

    # If user clicked skip, run inference immediately
    if ans == "skip":
        if callback.message:
            status_msg = await callback.message.edit_text(t("processing_image", lang))
            try:
                resp = await api_client.predict_image(
                    image_bytes=image_bytes,
                    profile=profile,
                    user_id=user_id,
                    anamnesis_dict=answers,
                )
                result_text = _format_prediction_response(resp, lang)
                kb = get_analysis_result_keyboard(lang) if resp.get("status") == "success" else None
                await _safe_edit_result(status_msg, result_text, reply_markup=kb, lang=lang)

                await state.set_state(AnalysisStates.result_ready)
                await state.update_data(last_prediction=resp, glasgow_answers=answers)
            except Exception as exc:
                logger.error(f"Inference API request failed: {exc}")
                await status_msg.edit_text(t("error_connection", lang))
        return

    # If all 7 Glasgow questions answered, offer fast De-Confounded Age Group step (FS-6)
    if next_idx >= len(GLASGOW_QUESTIONS):
        await state.update_data(glasgow_answers=answers)
        age_prompt = t("age_step_prompt", lang)
        if callback.message:
            await callback.message.edit_text(
                age_prompt,
                reply_markup=get_age_group_keyboard(lang),
                parse_mode="Markdown",
            )
        return

    # Present next question
    await state.set_state(AnalysisStates.answering_glasgow)
    await state.update_data(current_question_idx=next_idx, glasgow_answers=answers)
    next_field, next_key = GLASGOW_QUESTIONS[next_idx]
    prompt_text = t(
        "glasgow_question_step",
        lang,
        current=next_idx + 1,
        total=len(GLASGOW_QUESTIONS),
        question=t(next_key, lang),
    )

    if callback.message:
        try:
            await callback.message.edit_text(
                prompt_text,
                reply_markup=get_glasgow_question_keyboard(lang, step_idx=next_idx),
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.warning(f"Failed to edit question message: {e}")


@router.callback_query(F.data.startswith("g_age:"))
async def cb_set_age_group(callback: CallbackQuery, state: FSMContext) -> None:
    """Record optional patient age group and ask for anatomical lesion site (FS-6)."""
    try:
        await callback.answer()
    except Exception:
        pass

    user_id = callback.from_user.id
    lang = get_user_lang(user_id)
    val = callback.data.split(":", 1)[1]

    data = await state.get_data()
    answers = data.get("glasgow_answers", {})
    if val != "skip" and val.isdigit():
        answers["patient_age"] = int(val)
    await state.update_data(glasgow_answers=answers)

    site_prompt = t("site_step_prompt", lang)
    if callback.message:
        await callback.message.edit_text(
            site_prompt,
            reply_markup=get_anatom_site_keyboard(lang),
            parse_mode="Markdown",
        )


@router.callback_query(F.data.startswith("g_site:"))
async def cb_set_anatom_site(callback: CallbackQuery, state: FSMContext) -> None:
    """Record optional anatomical lesion site and trigger final multimodal inference (FS-6)."""
    try:
        await callback.answer()
    except Exception:
        pass

    user_id = callback.from_user.id
    lang = get_user_lang(user_id)
    profile = get_user_profile(user_id)
    val = callback.data.split(":", 1)[1]

    data = await state.get_data()
    answers = data.get("glasgow_answers", {})
    image_bytes = data.get("image_bytes")

    if not image_bytes:
        if callback.message:
            await callback.message.edit_text(
                t("session_expired", lang),
                parse_mode="Markdown",
            )
        return

    if val != "skip":
        answers["anatom_site"] = val

    if callback.message:
        status_msg = await callback.message.edit_text(t("processing_image", lang))
        try:
            resp = await api_client.predict_image(
                image_bytes=image_bytes,
                profile=profile,
                user_id=user_id,
                anamnesis_dict=answers,
            )
            result_text = _format_prediction_response(resp, lang)
            kb = get_analysis_result_keyboard(lang) if resp.get("status") == "success" else None
            await _safe_edit_result(status_msg, result_text, reply_markup=kb, lang=lang)

            await state.set_state(AnalysisStates.result_ready)
            await state.update_data(last_prediction=resp, glasgow_answers=answers)
        except Exception as exc:
            logger.error(f"Inference API request failed: {exc}")
            await status_msg.edit_text(t("error_connection", lang))



@router.callback_query(F.data == "pdf:download")
async def cb_download_pdf(callback: CallbackQuery, state: FSMContext) -> None:
    """Generate and deliver doctor consultation memo PDF in memory without disk persistence."""
    user_id = callback.from_user.id
    lang = get_user_lang(user_id)
    profile = get_user_profile(user_id)

    try:
        await callback.answer(t("pdf_generating", lang))
    except Exception:
        pass

    data = await state.get_data()
    image_bytes = data.get("image_bytes")
    answers = data.get("glasgow_answers", {})
    last_pred = data.get("last_prediction", {})
    req_id = last_pred.get("request_id")
    score = last_pred.get("anamnesis_evaluation", {}).get("score", 0)

    if not image_bytes:
        if callback.message:
            await callback.message.answer(t("pdf_expired_error", lang))
        return

    try:
        pdf_bytes = generate_clinical_memo_pdf(
            image_bytes=image_bytes,
            lang=lang,
            request_id=req_id,
            profile=profile,
            glasgow_answers=answers,
            glasgow_score=score,
        )
        short_id = req_id[:8] if req_id else "consultation"
        doc_name = f"SkinAI_Memo_{short_id}_{lang}.pdf"

        if callback.message:
            await callback.message.answer_document(
                document=BufferedInputFile(pdf_bytes, filename=doc_name),
                caption=t("pdf_sent_caption", lang),
            )
    except Exception as exc:
        logger.error(f"Failed to generate clinical memo PDF: {exc}")
        if callback.message:
            await callback.message.answer(t("error_generic", lang))

