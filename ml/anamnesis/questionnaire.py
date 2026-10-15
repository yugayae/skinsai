"""Evidence-based clinical anamnesis engine based on the Weighted Glasgow 7-Point Checklist (NICE NG12).

Provides:
1. Validated scoring for dermatological triage.
2. Bayesian prior adjustment for multi-modal fusion with computer vision predictions.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

RiskLevel = Literal["low", "moderate", "high"]
TriageTier = Literal["urgent_referral", "routine_consultation", "self_monitoring", "inconclusive_quality"]


class TriageDecision(BaseModel):
    """Clinical triage recommendation separating vision model probabilities from anamnesis."""

    tier: TriageTier
    title_ru: str
    title_en: str
    title_ko: Optional[str] = None
    recommendation_ru: str
    recommendation_en: str
    recommendation_ko: Optional[str] = None
    rationale: list[str] = Field(default_factory=list)
    referral_recommended: bool
    is_benign_observation: bool = False
    urgency_days: Optional[int] = None



class GlasgowChecklistInput(BaseModel):
    """Clinical questionnaire inputs based on NICE NG12 Glasgow 7-Point Checklist."""

    # Major Features (Weighted 2 points each)
    change_in_size: Optional[bool] = Field(
        default=None,
        description="Has the lesion recently grown or changed in size? (Major: 2 pts)"
    )
    irregular_shape: Optional[bool] = Field(
        default=None,
        description="Is the border or outline irregular / asymmetrical? (Major: 2 pts)"
    )
    irregular_color: Optional[bool] = Field(
        default=None,
        description="Is the color uneven, with patches of black, brown, white, or red? (Major: 2 pts)"
    )

    # Minor Features (Weighted 1 point each)
    diameter_ge_7mm: Optional[bool] = Field(
        default=None,
        description="Is the largest diameter 7 mm or larger (larger than a pencil eraser)? (Minor: 1 pt)"
    )
    inflammation: Optional[bool] = Field(
        default=None,
        description="Is there visible redness, swelling, or an inflammatory halo around the lesion? (Minor: 1 pt)"
    )
    oozing_or_bleeding: Optional[bool] = Field(
        default=None,
        description="Has the lesion bled, crusted, or had liquid discharge? (Minor: 1 pt)"
    )
    change_in_sensation: Optional[bool] = Field(
        default=None,
        description="Has there been itchiness, tingling, tenderness, or pain? (Minor: 1 pt)"
    )

    # Demographic context
    patient_age: Optional[int] = Field(
        default=None,
        ge=0,
        le=120,
        description="Approximate patient age in years"
    )


class GlasgowChecklistResult(BaseModel):
    """Evidence-based scoring result from Glasgow 7-Point Checklist assessment."""

    score: int = Field(..., ge=0, le=10, description="Total weighted Glasgow score (0 to 10)")
    major_score: int = Field(..., ge=0, le=6, description="Points from major features (2 pts each)")
    minor_score: int = Field(..., ge=0, le=4, description="Points from minor features (1 pt each)")
    is_high_risk: bool = Field(
        ...,
        description="True if score >= 3 (NICE referral threshold for suspected skin malignancy)"
    )
    risk_level: RiskLevel = Field(..., description="Categorized risk: 'low' (<2), 'moderate' (2), 'high' (>=3)")
    positive_features: list[str] = Field(default_factory=list, description="List of confirmed clinical features")
    clinical_note_ru: str
    clinical_note_en: str
    clinical_note_ko: Optional[str] = None


class AnamnesisEngine:
    """Computes Glasgow checklist scores and applies Bayesian fusion to vision model probabilities."""

    @staticmethod
    def evaluate_glasgow(anamnesis: GlasgowChecklistInput) -> GlasgowChecklistResult:
        """Compute the weighted Glasgow 7-point checklist score according to NICE NG12 guidelines."""
        major_score = 0
        minor_score = 0
        positive_features = []

        if anamnesis.change_in_size is True:
            major_score += 2
            positive_features.append("Изменение размера / динамика роста (+2)")
        if anamnesis.irregular_shape is True:
            major_score += 2
            positive_features.append("Неровный контур / асимметрия (+2)")
        if anamnesis.irregular_color is True:
            major_score += 2
            positive_features.append("Неоднородный цвет / пестрота окраски (+2)")

        if anamnesis.diameter_ge_7mm is True:
            minor_score += 1
            positive_features.append("Диаметр >= 7 мм (+1)")
        if anamnesis.inflammation is True:
            minor_score += 1
            positive_features.append("Воспаление / краснота вокруг (+1)")
        if anamnesis.oozing_or_bleeding is True:
            minor_score += 1
            positive_features.append("Кровоточивость / мокнутие / корочки (+1)")
        if anamnesis.change_in_sensation is True:
            minor_score += 1
            positive_features.append("Зуд / покалывание / болезненность (+1)")

        total_score = major_score + minor_score

        if total_score >= 3:
            risk_level: RiskLevel = "high"
            is_high = True
            note_ru = (
                f"Суммарный балл Глазго: {total_score} (порог риска >= 3). "
                "Выявлены клинические признаки, требующие очного осмотра врачом-дерматологом или онкологом."
            )
            note_en = (
                f"Total Glasgow score: {total_score} (NICE referral threshold >= 3). "
                "Clinical signs present warranting urgent face-to-face specialist dermatological assessment."
            )
            note_ko = (
                f"글래스고 점수: {total_score}점 (NICE NG12 의뢰 기준 >= 3점). "
                "전문의의 긴급 대면 진료가 필요한 임상적 징후가 확인되었습니다."
            )
        elif total_score == 2:
            risk_level = "moderate"
            is_high = False
            note_ru = (
                f"Суммарный балл Глазго: {total_score}. Умеренный уровень настороженности. "
                "Рекомендуется наблюдение за динамикой образования и плановая консультация врача."
            )
            note_en = (
                f"Total Glasgow score: {total_score}. Moderate clinical suspicion. "
                "Monitor for further evolution and schedule routine clinical review."
            )
            note_ko = (
                f"글래스고 점수: {total_score}점. 중간 수준의 주의가 필요합니다. "
                "병변 변화를 관찰하며 정기 진료를 권장합니다."
            )
        else:
            risk_level = "low"
            is_high = False
            note_ru = (
                f"Суммарный балл Глазго: {total_score} (ниже порога риска). "
                "Явных признаков злокачественной динамики по опроснику не отмечено. Сохраняйте плановое наблюдение."
            )
            note_en = (
                f"Total Glasgow score: {total_score} (below referral threshold). "
                "No immediate red flag evolution noted in questionnaire. Maintain routine skin observation."
            )
            note_ko = (
                f"글래스고 점수: {total_score}점 (기준치 미만). "
                "악성 의심 증상이 확인되지 않았습니다. 지속적으로 정기 관찰을 유지하세요."
            )

        return GlasgowChecklistResult(
            score=total_score,
            major_score=major_score,
            minor_score=minor_score,
            is_high_risk=is_high,
            risk_level=risk_level,
            positive_features=positive_features,
            clinical_note_ru=note_ru,
            clinical_note_en=note_en,
            clinical_note_ko=note_ko,
        )

    @classmethod
    def determine_triage(
        cls,
        image_probabilities: Dict[str, float],
        anamnesis: Optional[GlasgowChecklistInput] = None,
        is_blurry: bool = False,
        entropy: Optional[float] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> TriageDecision:
        """Evaluate clinical triage tier without multiplying probabilities.

        Preserves pure model calibration while establishing safe referral rules:
        - URGENT_REFERRAL: Glasgow score >= 3 OR model mel >= mel_thresh OR scc >= scc_thresh
        - ROUTINE_CONSULTATION: model bcc >= bcc_thresh OR ack >= ack_thresh OR Glasgow 1-2 OR skipped questionnaire OR elevated entropy
        - SELF_MONITORING: Strict conjunction (Glasgow == 0 AND all questions answered AND benign probability >= 0.80 AND low entropy)
        - INCONCLUSIVE_QUALITY: Blurry image or OOD
        """
        th = {
            "mel_threshold": 0.20,
            "scc_threshold": 0.30,
            "bcc_threshold": 0.35,
            "ack_threshold": 0.30,
            "entropy_limit": 1.35,
            "benign_confidence": 0.80,
        }
        if thresholds:
            th.update(thresholds)

        rationale = []

        # 1. Quality / Blurry Gate
        if is_blurry:
            return TriageDecision(
                tier="inconclusive_quality",
                title_ru="Снимок не пригоден для точного анализа",
                title_en="Image unsuitable for definitive analysis",
                title_ko="정확한 분석에 부적합한 이미지 품질",
                recommendation_ru=(
                    "Резкость снимка недостаточна для анализа. Пожалуйста, сделайте повторный "
                    "четкий макроснимок при хорошем освещении или обратитесь на очный приём к врачу."
                ),
                recommendation_en=(
                    "Image sharpness is insufficient. Please retake a clear, well-focused macro photo "
                    "in good lighting, or consult a doctor directly if concerned."
                ),
                recommendation_ko=(
                    "선명도가 부족합니다. 밝은 자연광 아래에서 초점을 정확히 맞추고 다시 촬영하시거나 "
                    "의사와 직접 상담하세요."
                ),
                rationale=["Низкая резкость (размытие кадра)"],
                referral_recommended=False,
            )

        # 2. Anamnesis evaluation
        glasgow_res = None
        is_skipped = True
        if anamnesis is not None:
            has_answers = any(
                v is not None
                for v in [
                    anamnesis.change_in_size,
                    anamnesis.irregular_shape,
                    anamnesis.irregular_color,
                    anamnesis.diameter_ge_7mm,
                    anamnesis.inflammation,
                    anamnesis.oozing_or_bleeding,
                    anamnesis.change_in_sensation,
                ]
            )
            if has_answers:
                is_skipped = False
                glasgow_res = cls.evaluate_glasgow(anamnesis)

        mel_p = image_probabilities.get("mel", 0.0)
        scc_p = image_probabilities.get("scc", 0.0)
        bcc_p = image_probabilities.get("bcc", 0.0)
        ack_p = image_probabilities.get("ack", 0.0)
        benign_p = image_probabilities.get("nev", 0.0) + image_probabilities.get("sek", 0.0)

        # 3. URGENT REFERRAL check
        if glasgow_res and glasgow_res.is_high_risk:
            rationale.append(f"Шкала Глазго: {glasgow_res.score} баллов (критический порог NICE NG12 >= 3)")
        if mel_p >= th["mel_threshold"]:
            rationale.append(f"Модельная оценка подозрения на меланому ({mel_p * 100:.1f}%)")
        if scc_p >= th["scc_threshold"]:
            rationale.append(f"Модельная оценка подозрения на плоскоклеточную карциному ({scc_p * 100:.1f}%)")

        if rationale:
            return TriageDecision(
                tier="urgent_referral",
                title_ru="🚨 Рекомендована приоритетная очная консультация",
                title_en="🚨 Priority face-to-face consultation recommended",
                title_ko="🚨 우선 대면 진료 권고",
                recommendation_ru=(
                    "Выявлены клинические или оптические признаки, требующие приоритетного "
                    "очного осмотра врачом-дерматологом или онкологом с проведением дерматоскопии."
                ),
                recommendation_en=(
                    "Clinical or visual features detected warranting priority face-to-face "
                    "specialist assessment by a dermatologist or oncologist."
                ),
                recommendation_ko=(
                    "피부과 또는 종양내과 전문의의 우선 대면 진찰 및 더모스코피 검사가 "
                    "필요한 임상적 또는 영상적 징후가 감지되었습니다."
                ),
                rationale=rationale,
                referral_recommended=True,
                urgency_days=14,
            )

        # 4. ROUTINE CONSULTATION check
        routine_rationale = []
        if bcc_p >= th["bcc_threshold"]:
            routine_rationale.append(f"Подозрение на базальноклеточную карциному ({bcc_p * 100:.1f}%)")
        if ack_p >= th["ack_threshold"]:
            routine_rationale.append(f"Подозрение на актинический кератоз ({ack_p * 100:.1f}%)")
        if glasgow_res and glasgow_res.score in (1, 2):
            routine_rationale.append(f"Шкала Глазго: {glasgow_res.score} балл(а) (единичные симптомы)")
        if is_skipped:
            routine_rationale.append("Опросник анамнеза не заполнен")
        if entropy is not None and entropy > th["entropy_limit"]:
            routine_rationale.append("Повышенная неопределенность распределения вероятностей")

        if routine_rationale or benign_p < th["benign_confidence"]:
            if not routine_rationale:
                routine_rationale.append("Недостаточная уверенность модели в доброкачественности")
            return TriageDecision(
                tier="routine_consultation",
                title_ru="🩺 Рекомендована плановая консультация дерматолога",
                title_en="🩺 Routine dermatologist consultation recommended",
                title_ko="🩺 피부과 정기 대면 진료 권고",
                recommendation_ru=(
                    "Рекомендуется плановая очная консультация врача-дерматолога "
                    "для дифференциальной диагностики новообразования."
                ),
                recommendation_en=(
                    "A routine clinical review by a certified dermatologist is recommended "
                    "for accurate differential diagnosis."
                ),
                recommendation_ko=(
                    "정확한 감별 진단을 위해 피부과 전문의와의 정기 대면 상담을 권장합니다."
                ),
                rationale=routine_rationale,
                referral_recommended=False,
                urgency_days=30,
            )

        # 5. SELF MONITORING (Strict conjunction only)
        return TriageDecision(
            tier="self_monitoring",
            title_ru="🔍 Наблюдение за новообразованием в динамике",
            title_en="🔍 Routine observation and monitoring",
            title_ko="🔍 병변 경과 관찰",
            recommendation_ru=(
                "На данном снимке автоматический анализ не выявил признаков, требующих срочного визита.\n\n"
                "⚠️ ВНИМАНИЕ: Автоматический анализ фотографии со смартфона не исключает патологию "
                "и не заменяет очный осмотр врача-дерматолога. Рекомендуется повторный осмотр или визит "
                "к врачу при появлении малейших изменений размера, формы или цвета."
            ),
            recommendation_en=(
                "Automatic analysis of this photo did not identify immediate red flags requiring urgent care.\n\n"
                "⚠️ CAUTION: Automated smartphone photo screening does not rule out skin disease or cancer "
                "and never replaces clinical dermoscopy. Please see a dermatologist if the lesion changes in any way."
            ),
            recommendation_ko=(
                "자동 분석 결과 긴급한 조치가 필요한 위험 징후는 발견되지 않았습니다.\n\n"
                "⚠️ 주의: 스마트폰 사진을 통한 자동 분석은 질환을 완전히 배제하지 못하며 "
                "의사의 대면 진료를 대신할 수 없습니다. 크기나 형태 변화 시 피부과를 방문하세요."
            ),
            rationale=[
                "Опросник Глазго: 0 баллов (симптомы эволюции отсутствуют)",
                f"Высокая модельная оценка доброкачественности ({benign_p * 100:.1f}%)",
            ],
            referral_recommended=False,
            is_benign_observation=True,
        )

    @classmethod
    def apply_bayesian_fusion(
        cls,
        image_probabilities: Dict[str, float],
        anamnesis: Optional[GlasgowChecklistInput] = None,
    ) -> Dict[str, Any]:
        """Decoupled evaluation maintaining backward compatibility without multiplying probabilities."""
        triage = cls.determine_triage(image_probabilities, anamnesis)
        top_class = max(image_probabilities, key=image_probabilities.get)
        mel_risk = image_probabilities.get("mel", 0.0)
        checklist_res = cls.evaluate_glasgow(anamnesis) if anamnesis else None

        return {
            "fused_probabilities": image_probabilities,  # Pure calibrated probabilities without multiplier tampering!
            "top_class": top_class,
            "confidence": image_probabilities[top_class],
            "melanoma_risk_score": mel_risk,
            "referral_recommended": triage.referral_recommended,
            "checklist_evaluation": checklist_res.model_dump() if checklist_res else None,
            "triage": triage.model_dump(),
        }
