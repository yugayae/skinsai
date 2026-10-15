export type TriageTier =
  | 'URGENT_REFERRAL'
  | 'ROUTINE_CONSULTATION'
  | 'SELF_MONITORING'
  | 'INCONCLUSIVE_QUALITY';

export type Language = 'ru' | 'en' | 'ko';

export interface GlasgowChecklist {
  major_change_size: boolean;
  major_irregular_shape: boolean;
  major_irregular_color: boolean;
  minor_diameter_7mm: boolean;
  minor_inflammation: boolean;
  minor_oozing: boolean;
  minor_sensation: boolean;
}

export interface CaseData {
  id: string;
  timestamp: string;
  imageUrl: string;
  triageTier: TriageTier;
  glasgowAnswers: GlasgowChecklist;
  glasgowScore: number;
  profile: 'smartphone_clinical' | 'dermoscopy';
  qualityPassed: boolean;
  doctorQuestions: {
    ru: string[];
    en: string[];
    ko: string[];
  };
  notes?: string;
}
