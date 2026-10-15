export const en = {
  appName: 'SkinAI Clinical',
  appSubtitle: 'Clinical Triage & Skin Lesion Route Guide',
  profileBadge: 'Smartphone Profile',
  demoCases: 'Clinical Cases',
  case1: 'Case 1: High Suspicion (Urgent Referral)',
  case2: 'Case 2: Moderate Suspicion (Routine Check)',
  case3: 'Case 3: Low Suspicion (Self-Monitoring)',

  triage: {
    URGENT_REFERRAL: {
      title: 'In-person dermatologist evaluation recommended',
      subtitle: 'Priority consultation within 1–2 weeks',
      badge: 'Urgent Referral',
      description: 'The lesion image analysis combined with Glasgow checklist responses indicates the need for an in-person dermoscopic examination by a dermatologist.',
    },
    ROUTINE_CONSULTATION: {
      title: 'Routine dermatologist consultation recommended',
      subtitle: 'Scheduled examination of the lesion',
      badge: 'Routine Consultation',
      description: 'Moderate features or anamnesis flags were identified. An in-person routine skin examination is advised.',
    },
    SELF_MONITORING: {
      title: 'Routine Self-Monitoring',
      subtitle: 'Low probability of acute atypical features',
      badge: 'Self-Monitoring',
      description: 'No pronounced atypical features detected. Continue monthly self-examinations following the ABCD rule.',
    },
    INCONCLUSIVE_QUALITY: {
      title: 'Insufficient Image Quality',
      subtitle: 'Retake photo under better conditions',
      badge: 'Poor Quality',
      description: 'The image appears blurry, under/overexposed, or too far. Please take a sharp photo in good daylight.',
    },
  },

  sections: {
    photoTitle: 'Skin Lesion Image',
    triageTitle: 'Triage Recommendation',
    glasgowTitle: 'Glasgow 7-Point Checklist',
    glasgowScore: 'Glasgow Checklist Score:',
    glasgowInterpretation: 'Clinical context: Score ≥ 3 points or presence of any major sign qualifies for clinical referral under NICE NG12 guidelines.',
    abcdTitle: 'Clinical Reference: ABCD Rule',
    doctorMemoTitle: 'Doctor Encounter Clinical Memo',
    questionsTitle: 'Recommended Questions for Your Doctor',
    downloadPdfBtn: '📄 Download Doctor Memo (PDF)',
    preparingPdf: 'Generating memo...',
    disclaimerTitle: 'Medical Non-Claim Disclaimer',
    disclaimerText: 'SkinAI is an experimental research tool and does NOT provide medical diagnosis. This memo does not replace an in-person consultation with a certified dermatologist.',
  },

  glasgow: {
    majorHeader: 'Major Signs (2 points each)',
    minorHeader: 'Minor Signs (1 point each)',
    major_change_size: 'Recent change or increase in size',
    major_irregular_shape: 'Irregular, asymmetric shape or uneven outline',
    major_irregular_color: 'Irregular color variation (shades of brown, black, pink)',
    minor_diameter_7mm: 'Maximum diameter 7 mm or greater',
    minor_inflammation: 'Signs of inflammation or erythema around the lesion',
    minor_oozing: 'Oozing, crusting, ulceration, or bleeding',
    minor_sensation: 'Itching, burning, tenderness, or altered sensation',
    yes: 'Yes',
    no: 'No',
  },

  abcd: {
    a: {
      letter: 'A',
      title: 'Asymmetry',
      desc: 'One half of the mole does not mirror the other half in shape or outline.',
    },
    b: {
      letter: 'B',
      title: 'Border',
      desc: 'Edges are irregular, ragged, notched, or blurred.',
    },
    c: {
      letter: 'C',
      title: 'Color',
      desc: 'Color is uneven, with shades of tan, brown, black, white, red, or blue.',
    },
    d: {
      letter: 'D',
      title: 'Diameter',
      desc: 'Lesion is greater than 6 mm across (though melanomas can be smaller).',
    },
  },

  memoQuestions: [
    'Is digital dermoscopy with photo-mapping recommended for this lesion?',
    'Do you recommend close follow-up or an excisional biopsy?',
    'Should a full-body total skin examination (TSE) be conducted?',
  ],
};
