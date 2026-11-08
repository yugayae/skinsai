export const ko = {
  appName: 'SkinAI Clinical',
  appSubtitle: '피부 병변 임상 분류 및 트리아지 가이드',
  profileBadge: '스마트폰 임상 프로필',
  demoCases: '임상 증례',
  case1: '증례 1: 높은 의심도 (우선 진료 권고)',
  case2: '증례 2: 중간 의심도 (정기 검진 권고)',
  case3: '증례 3: 낮은 의심도 (자가 모니터링)',

  triage: {
    URGENT_REFERRAL: {
      title: '피부과 전문의 대면 진료를 권장합니다',
      subtitle: '1~2주 이내 우선 진찰 권고',
      badge: '우선 진료 권고',
      description: '사진 분석과 글래스고 체크리스트 응답을 종합할 때, 피부과 전문의의 더모스코피 대면 진료가 필요합니다.',
    },
    ROUTINE_CONSULTATION: {
      title: '피부과 정기 상담을 권장합니다',
      subtitle: '정기적인 병변 확인 권고',
      badge: '정기 상담 권고',
      description: '비전형적인 특징이나 병력 징후가 확인되었습니다. 전문의와의 정기 예방 진료를 권장합니다.',
    },
    SELF_MONITORING: {
      title: '정기적 자가 모니터링',
      subtitle: '급성 비전형적 변화 위험 낮음',
      badge: '자가 모니터링',
      description: '뚜렷한 비전형적 징후는 발견되지 않았습니다. ABCD 규칙에 따라 매월 자가 관찰을 지속하십시오.',
    },
    INCONCLUSIVE_QUALITY: {
      title: '사진 품질 미달',
      subtitle: '양호한 환경에서 재촬영 필요',
      badge: '품질 미달',
      description: '초점이 흐리거나 너무 어둡거나 멀리서 촬영되었습니다. 자연광 아래에서 선명하게 다시 촬영해 주세요.',
    },
  },

  sections: {
    photoTitle: '피부 병변 사진',
    triageTitle: '트리아지 권고 사항',
    glasgowTitle: '글래스고 7점 체크리스트 (Glasgow 7-Point Checklist)',
    glasgowScore: '글래스고 척도 점수:',
    glasgowInterpretation: '임상적 의의: 총점 3점 이상이거나 주요 징후가 1개 이상인 경우 영국 NICE NG12 가이드라인에 따라 대면 진료가 권고됩니다.',
    abcdTitle: '임상 참고 지침: ABCD 규칙',
    doctorMemoTitle: '전문의 대면 진료용 메모',
    questionsTitle: '의사에게 질문할 추천 사항',
    downloadPdfBtn: '📄 의사 상담 메모 다운로드 (PDF)',
    preparingPdf: '메모 생성 중...',
    disclaimerTitle: '의학적 비진단 고지',
    disclaimerText: 'SkinAI는 연구용 도구이며 의학적 진단을 제공하지 않습니다. 본 메모는 피부과 전문의 또는 종양학 전문의의 대면 진료를 대체하지 않습니다.',
  },

  glasgow: {
    majorHeader: '주요 징후 (항목당 2점)',
    minorHeader: '부차적 징후 (항목당 1점)',
    major_change_size: '최근 병변의 크기가 커지거나 변화함',
    major_irregular_shape: '불규칙하고 비대칭적인 모양 또는 울퉁불퉁한 경계',
    major_irregular_color: '얼룩덜룩하고 불균일한 색상 (갈색, 검정, 분홍 등)',
    minor_diameter_7mm: '병변의 최대 직경이 7mm 이상임',
    minor_inflammation: '병변 주위의 염증 징후 또는 발적',
    minor_oozing: '진물, 딱지, 궤양 또는 출혈 발생',
    minor_sensation: '가려움증, 작열감, 통증 또는 감각 이상',
    yes: '예',
    no: '아니오',
  },

  abcd: {
    a: {
      letter: 'A',
      title: 'Asymmetry (비대칭성)',
      desc: '병변의 중심을 기준으로 좌우 또는 상하가 대칭을 이루지 않습니다.',
    },
    b: {
      letter: 'B',
      title: 'Border (경계 불규칙)',
      desc: '가장자리가 들쭉날쭉하거나 톱니 모양, 경계가 모호합니다.',
    },
    c: {
      letter: 'C',
      title: 'Color (색조 다양성)',
      desc: '단일 색상이 아니며 흑색, 갈색, 청색, 백색이 혼재되어 있습니다.',
    },
    d: {
      letter: 'D',
      title: 'Diameter (직경 6mm 이상)',
      desc: '크기가 6mm를 초과합니다 (단, 초기 흑색종은 더 작을 수 있습니다).',
    },
  },

  memoQuestions: [
    '이 병변에 대해 디지털 더모스코피 및 사진 촬영 기록이 필요한가요?',
    '지속적인 경과 관찰과 절제 생검 중 어느 쪽을 권고하시나요?',
    '신체 전반의 전신 피부 검진(TSE)을 함께 받는 것이 좋을까요?',
  ],
};
