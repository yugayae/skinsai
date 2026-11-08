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

export interface ABCDEMetricsData {
  asymmetry_shape: number;
  border_irregularity: number;
  color_entropy: number;
  distinct_color_clusters: number;
}

export interface CaseData {
  id: string;
  timestamp: string;
  imageUrl: string;
  triageTier: TriageTier;
  triageZone?: 'red' | 'yellow' | 'green';
  glasgowAnswers: GlasgowChecklist;
  glasgowScore: number;
  profile: 'smartphone_clinical' | 'dermoscopy';
  qualityPassed: boolean;
  abcdeMetrics?: ABCDEMetricsData;
  predictedClassName?: string;
  confidence?: number;
  doctorQuestions: {
    ru: string[];
    en: string[];
    ko: string[];
  };
  notes?: string;
}
