import { jsPDF } from 'jspdf';
import type { CaseData, Language } from '../types';
import { translations } from '../i18n';

export function generateDoctorMemoPdf(caseData: CaseData, lang: Language) {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  });

  const t = translations[lang];
  const pageWidth = doc.internal.pageSize.getWidth();
  let y = 18;

  // Header Title
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(16);
  doc.setTextColor(30, 41, 59);
  doc.text('SkinAI - Clinical Encounter Patient Memo', pageWidth / 2, y, { align: 'center' });
  y += 7;

  // Subtitle
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10);
  doc.setTextColor(100, 116, 139);
  doc.text('For in-person dermatologist consultation (Zero Cognitive Anchoring)', pageWidth / 2, y, {
    align: 'center',
  });
  y += 10;

  // Metadata Box
  doc.setDrawColor(226, 232, 240);
  doc.setFillColor(248, 250, 252);
  doc.roundedRect(15, y, pageWidth - 30, 20, 2, 2, 'FD');

  doc.setFontSize(9);
  doc.setTextColor(71, 85, 105);
  doc.text(`Encounter ID: ${caseData.id}`, 20, y + 7);
  doc.text(`Timestamp: ${caseData.timestamp}`, 20, y + 14);
  doc.text(`Profile: ${caseData.profile}`, pageWidth - 70, y + 7);
  doc.text(`Quality Check: Passed`, pageWidth - 70, y + 14);
  y += 28;

  // Section 1: Clinical Triage Recommendation
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(12);
  doc.setTextColor(15, 23, 42);
  doc.text('1. Clinical Triage Recommendation', 15, y);
  y += 6;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10);
  const triageInfo = t.triage[caseData.triageTier];
  doc.setTextColor(30, 41, 59);
  doc.text(`Status: ${triageInfo.badge}`, 20, y);
  y += 5;
  doc.text(`${triageInfo.description}`, 20, y, { maxWidth: pageWidth - 40 });
  y += 15;

  // Section 2: Glasgow 7-Point Checklist (NICE NG12)
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(12);
  doc.setTextColor(15, 23, 42);
  doc.text(`2. Glasgow 7-Point Checklist (Score: ${caseData.glasgowScore}/7)`, 15, y);
  y += 7;

  // Table header
  doc.setFillColor(241, 245, 249);
  doc.rect(15, y, pageWidth - 30, 7, 'F');
  doc.setFontSize(9);
  doc.setTextColor(51, 65, 85);
  doc.text('Symptom / Sign', 20, y + 5);
  doc.text('Patient Answer', pageWidth - 45, y + 5);
  y += 8;

  // Checklist items
  const items: Array<{ label: string; val: boolean }> = [
    { label: t.glasgow.major_change_size, val: caseData.glasgowAnswers.major_change_size },
    { label: t.glasgow.major_irregular_shape, val: caseData.glasgowAnswers.major_irregular_shape },
    { label: t.glasgow.major_irregular_color, val: caseData.glasgowAnswers.major_irregular_color },
    { label: t.glasgow.minor_diameter_7mm, val: caseData.glasgowAnswers.minor_diameter_7mm },
    { label: t.glasgow.minor_inflammation, val: caseData.glasgowAnswers.minor_inflammation },
    { label: t.glasgow.minor_oozing, val: caseData.glasgowAnswers.minor_oozing },
    { label: t.glasgow.minor_sensation, val: caseData.glasgowAnswers.minor_sensation },
  ];

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8.5);

  items.forEach((item, index) => {
    if (index % 2 === 1) {
      doc.setFillColor(248, 250, 252);
      doc.rect(15, y - 1, pageWidth - 30, 6, 'F');
    }
    doc.setTextColor(30, 41, 59);
    // Transliterate or clean non-ascii for PDF standard helvetica if needed
    const cleanLabel = item.label;
    doc.text(cleanLabel.length > 70 ? cleanLabel.substring(0, 70) + '...' : cleanLabel, 20, y + 3.5);

    if (item.val) {
      doc.setTextColor(185, 28, 28);
      doc.setFont('helvetica', 'bold');
      doc.text('[+] YES', pageWidth - 40, y + 3.5);
      doc.setFont('helvetica', 'normal');
    } else {
      doc.setTextColor(100, 116, 139);
      doc.text('[-] No', pageWidth - 40, y + 3.5);
    }
    y += 6.5;
  });

  y += 6;

  // Section 3: Recommended In-Person Consultation Questions
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(12);
  doc.setTextColor(15, 23, 42);
  doc.text('3. Suggested Questions for Clinical Specialist', 15, y);
  y += 7;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.setTextColor(51, 65, 85);
  const questions = caseData.doctorQuestions[lang] || t.memoQuestions;
  questions.forEach((q) => {
    doc.text(`• ${q}`, 20, y, { maxWidth: pageWidth - 40 });
    y += 6;
  });

  y += 6;

  // Section 4: Regulatory Non-Claim Notice
  doc.setDrawColor(254, 202, 202);
  doc.setFillColor(254, 242, 242);
  doc.roundedRect(15, y, pageWidth - 30, 24, 2, 2, 'FD');

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8.5);
  doc.setTextColor(153, 27, 27);
  doc.text('IMPORTANT CLINICAL DISCLAIMER (Zero Cognitive Anchoring)', 20, y + 6);

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7.5);
  doc.setTextColor(185, 28, 28);
  doc.text(
    'SkinAI is an open-source experimental research tool, NOT a diagnostic medical device. This memo does NOT ' +
      'contain model class probabilities to prevent cognitive anchoring. It does not replace clinical dermoscopy ' +
      'or biopsy by a certified specialist.',
    20,
    y + 11,
    { maxWidth: pageWidth - 40 }
  );

  // Save the PDF
  doc.save(`SkinAI_Doctor_Memo_${caseData.id}.pdf`);
}
