import type { CaseData } from '../types';

export const SAMPLE_CASES: Record<string, CaseData> = {
  case1: {
    id: 'SKIN-9842A',
    timestamp: '2026-10-14 18:45 UTC',
    // Realistic SVG/Data URI representation of a lesion for demo/offline preview
    imageUrl:
      'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300"><defs><radialGradient id="g1" cx="45%" cy="40%" r="55%"><stop offset="0%" stop-color="%231a0f0a"/><stop offset="35%" stop-color="%233e1c10"/><stop offset="70%" stop-color="%237c3f25"/><stop offset="85%" stop-color="%23a86343"/><stop offset="100%" stop-color="%23e2a78f"/></radialGradient><filter id="rough"><feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="4" result="noise"/><feDisplacementMap in="SourceGraphic" in2="noise" scale="14" xChannelSelector="R" yChannelSelector="G"/></filter></defs><rect width="100%" height="100%" fill="%23ecd2c1"/><path filter="url(%23rough)" d="M150,60 C210,65 240,110 230,165 C220,220 185,245 135,235 C85,225 65,180 75,130 C85,80 110,55 150,60 Z" fill="url(%23g1)"/><circle cx="130" cy="120" r="14" fill="%23110804" opacity="0.8"/><circle cx="170" cy="155" r="9" fill="%23260c04" opacity="0.7"/><circle cx="140" cy="180" r="6" fill="%23451508" opacity="0.6"/></svg>',
    triageTier: 'URGENT_REFERRAL',
    profile: 'smartphone_clinical',
    qualityPassed: true,
    glasgowScore: 5,
    glasgowAnswers: {
      major_change_size: true,
      major_irregular_shape: true,
      major_irregular_color: false,
      minor_diameter_7mm: true,
      minor_inflammation: false,
      minor_oozing: false,
      minor_sensation: false,
    },
    doctorQuestions: {
      ru: [
        'Требуется ли проведение цифровой дерматоскопии с фотофиксацией?',
        'Рекомендуется ли наблюдение в динамике или эксцизионная биопсия?',
        'Нужно ли провести профилактический осмотр всех остальных родинок на теле?',
      ],
      en: [
        'Is digital dermoscopy with photo-mapping recommended for this lesion?',
        'Do you recommend close follow-up or an excisional biopsy?',
        'Should a full-body total skin examination (TSE) be conducted?',
      ],
      ko: [
        '이 병변에 대해 디지털 더모스코피 및 사진 촬영 기록이 필요한가요?',
        '지속적인 경과 관찰과 절제 생검 중 어느 쪽을 권고하시나요?',
        '신체 전반의 전신 피부 검진(TSE)을 함께 받는 것이 좋을까요?',
      ],
    },
    notes: 'Зафиксировано изменение размера за последние 3 месяца.',
  },

  case2: {
    id: 'SKIN-5519B',
    timestamp: '2026-10-14 16:20 UTC',
    imageUrl:
      'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300"><defs><radialGradient id="g2" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="%235c3a21"/><stop offset="50%" stop-color="%238b5a32"/><stop offset="85%" stop-color="%23b88558"/><stop offset="100%" stop-color="%23f0c2a8"/></radialGradient><filter id="rough2"><feTurbulence type="fractalNoise" baseFrequency="0.03" numOctaves="3" result="noise"/><feDisplacementMap in="SourceGraphic" in2="noise" scale="8" xChannelSelector="R" yChannelSelector="G"/></filter></defs><rect width="100%" height="100%" fill="%23f5dac9"/><path filter="url(%23rough2)" d="M150,85 C195,85 225,115 220,155 C215,195 185,220 145,218 C105,216 85,185 88,150 C92,112 115,85 150,85 Z" fill="url(%23g2)"/><circle cx="150" cy="150" r="10" fill="%234a2c14" opacity="0.6"/></svg>',
    triageTier: 'ROUTINE_CONSULTATION',
    profile: 'smartphone_clinical',
    qualityPassed: true,
    glasgowScore: 1,
    glasgowAnswers: {
      major_change_size: false,
      major_irregular_shape: false,
      major_irregular_color: false,
      minor_diameter_7mm: false,
      minor_inflammation: false,
      minor_oozing: false,
      minor_sensation: true,
    },
    doctorQuestions: {
      ru: [
        'Является ли образование себорейным кератозом или дермальным невусом?',
        'Имеются ли показания к удалению по эстетическим или медицинским причинам?',
        'Требуется ли периодический повторный осмотр?',
      ],
      en: [
        'Is this lesion consistent with a seborrheic keratosis or dermal nevus?',
        'Are there any medical or cosmetic indications for removal?',
        'Is periodic follow-up recommended?',
      ],
      ko: [
        '이 병변이 지루각화증이나 진피내 모반 소견과 일치하나요?',
        '미용적 또는 의학적 이유로 제거가 필요한 상황인가요?',
        '주기적인 추적 관찰이 권장되나요?',
      ],
    },
    notes: 'Периодический легкий зуд при трении об одежду.',
  },

  case3: {
    id: 'SKIN-1204C',
    timestamp: '2026-10-14 11:10 UTC',
    imageUrl:
      'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300"><defs><radialGradient id="g3" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="%23684227"/><stop offset="65%" stop-color="%23825838"/><stop offset="90%" stop-color="%23ab7f5a"/><stop offset="100%" stop-color="%23f9ded0"/></radialGradient><filter id="smooth"><feGaussianBlur stdDeviation="1.5"/></filter></defs><rect width="100%" height="100%" fill="%23fae4d7"/><circle cx="150" cy="150" r="45" fill="url(%23g3)" filter="url(%23smooth)"/></svg>',
    triageTier: 'SELF_MONITORING',
    profile: 'smartphone_clinical',
    qualityPassed: true,
    glasgowScore: 0,
    glasgowAnswers: {
      major_change_size: false,
      major_irregular_shape: false,
      major_irregular_color: false,
      minor_diameter_7mm: false,
      minor_inflammation: false,
      minor_oozing: false,
      minor_sensation: false,
    },
    doctorQuestions: {
      ru: [
        'Какова оптимальная периодичность профилактического самоосмотра?',
        'На какие первичные изменения стоит обращать внимание в первую очередь?',
      ],
      en: [
        'What is the recommended interval for self-skin examination?',
        'What early subtle changes should prompt an earlier visit?',
      ],
      ko: [
        '자가 피부 검진은 어느 정도 주기로 하는 것이 적절한가요?',
        '어떤 초기 변화가 있을 때 바로 내원해야 하나요?',
      ],
    },
    notes: 'Стабильное образование, без жалоб.',
  },
};

export function getInitialCase(): CaseData {
  // Check for URL query param or hash data (e.g. from Telegram WebApp start_param)
  if (typeof window !== 'undefined') {
    const params = new URLSearchParams(window.location.search);
    const rawData = params.get('data');
    if (rawData) {
      try {
        const decoded = JSON.parse(decodeURIComponent(rawData));
        if (decoded.id && decoded.triageTier) {
          return decoded as CaseData;
        }
      } catch (e) {
        console.warn('Failed to parse URL case data:', e);
      }
    }
  }
  return SAMPLE_CASES.case1;
}
