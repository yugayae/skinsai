import './style.css';
import type { CaseData, Language } from './types';
import { setLanguage, initLanguage, t } from './i18n';
import { SAMPLE_CASES, getInitialCase } from './services/sampleData';
import { initTelegram, triggerHaptic, triggerNotificationHaptic, getTelegramApp } from './services/telegram';
import { generateDoctorMemoPdf } from './services/pdfExport';

// State
let currentCase: CaseData = getInitialCase();
let currentLang: Language = initLanguage();

function renderApp() {
  const appContainer = document.getElementById('app');
  if (!appContainer) return;

  const currentT = t();
  const triageData = currentT.triage[currentCase.triageTier];
  const glasgow = currentCase.glasgowAnswers;

  // Determine dark mode icon
  const isDark = document.documentElement.classList.contains('dark');
  const themeIcon = isDark ? '☀️' : '🌙';

  appContainer.innerHTML = `
    <!-- Header -->
    <header class="app-header">
      <div class="brand-wrapper">
        <div class="brand-logo-pulse">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
          </svg>
        </div>
        <div class="brand-text">
          <h1>${currentT.appName}</h1>
          <p>${currentT.appSubtitle}</p>
        </div>
      </div>
      <div class="header-controls">
        <div class="lang-selector">
          <button class="lang-btn ${currentLang === 'ru' ? 'active' : ''}" data-lang="ru">RU</button>
          <button class="lang-btn ${currentLang === 'en' ? 'active' : ''}" data-lang="en">EN</button>
          <button class="lang-btn ${currentLang === 'ko' ? 'active' : ''}" data-lang="ko">KO</button>
        </div>
        <button class="theme-toggle-btn" id="theme-toggle" title="Toggle theme">
          ${themeIcon}
        </button>
      </div>
    </header>

    <!-- Case Switcher (Demo & Live) -->
    <div class="case-switcher-bar">
      <button class="case-tab-btn ${currentCase.id === SAMPLE_CASES.case1.id ? 'active' : ''}" data-case="case1">
        ${currentT.case1}
      </button>
      <button class="case-tab-btn ${currentCase.id === SAMPLE_CASES.case2.id ? 'active' : ''}" data-case="case2">
        ${currentT.case2}
      </button>
      <button class="case-tab-btn ${currentCase.id === SAMPLE_CASES.case3.id ? 'active' : ''}" data-case="case3">
        ${currentT.case3}
      </button>
    </div>

    <!-- Triage Banner -->
    <section class="triage-banner tier-${currentCase.triageTier}">
      <div class="triage-header-row">
        <span class="triage-badge">
          <span class="triage-badge-pulse"></span>
          ${triageData.badge}
        </span>
        <span class="meta-chip">${currentT.profileBadge}</span>
      </div>
      <h2 class="triage-title">${triageData.title}</h2>
      <p class="triage-desc">${triageData.description}</p>
    </section>

    <!-- Lesion Photo Card -->
    <div class="card">
      <div class="card-title">
        <span>${currentT.sections.photoTitle}</span>
        <span class="meta-chip">ID: ${currentCase.id}</span>
      </div>
      <div class="lesion-container">
        <div class="lesion-image-frame">
          <img src="${currentCase.imageUrl}" alt="Skin lesion photo" id="lesion-img" />
        </div>
        <div class="lesion-details-row">
          <span>📅 ${currentCase.timestamp}</span>
          <span>⚡ ${currentCase.qualityPassed ? 'Качество: OK' : 'Низкое качество'}</span>
        </div>
      </div>
    </div>

    <!-- Glasgow 7-Point Scorecard -->
    <div class="card">
      <div class="card-title">
        <span>${currentT.sections.glasgowTitle}</span>
      </div>
      
      <div class="glasgow-score-meter">
        <div class="score-badge">${currentCase.glasgowScore} / 7</div>
        <div class="score-progress-bar-bg">
          <div class="score-progress-bar-fill" style="width: ${Math.min(100, (currentCase.glasgowScore / 7) * 100)}%;"></div>
        </div>
      </div>

      <div class="glasgow-citation">
        ${currentT.sections.glasgowInterpretation}
      </div>

      <div class="glasgow-group-label">${currentT.glasgow.majorHeader}</div>
      <div class="checklist-list">
        <div class="checklist-item ${glasgow.major_change_size ? 'is-positive' : ''}">
          <span class="checklist-text">${currentT.glasgow.major_change_size}</span>
          <span class="checklist-status ${glasgow.major_change_size ? 'yes' : 'no'}">
            ${glasgow.major_change_size ? currentT.glasgow.yes : currentT.glasgow.no}
          </span>
        </div>
        <div class="checklist-item ${glasgow.major_irregular_shape ? 'is-positive' : ''}">
          <span class="checklist-text">${currentT.glasgow.major_irregular_shape}</span>
          <span class="checklist-status ${glasgow.major_irregular_shape ? 'yes' : 'no'}">
            ${glasgow.major_irregular_shape ? currentT.glasgow.yes : currentT.glasgow.no}
          </span>
        </div>
        <div class="checklist-item ${glasgow.major_irregular_color ? 'is-positive' : ''}">
          <span class="checklist-text">${currentT.glasgow.major_irregular_color}</span>
          <span class="checklist-status ${glasgow.major_irregular_color ? 'yes' : 'no'}">
            ${glasgow.major_irregular_color ? currentT.glasgow.yes : currentT.glasgow.no}
          </span>
        </div>
      </div>

      <div class="glasgow-group-label">${currentT.glasgow.minorHeader}</div>
      <div class="checklist-list">
        <div class="checklist-item ${glasgow.minor_diameter_7mm ? 'is-positive' : ''}">
          <span class="checklist-text">${currentT.glasgow.minor_diameter_7mm}</span>
          <span class="checklist-status ${glasgow.minor_diameter_7mm ? 'yes' : 'no'}">
            ${glasgow.minor_diameter_7mm ? currentT.glasgow.yes : currentT.glasgow.no}
          </span>
        </div>
        <div class="checklist-item ${glasgow.minor_inflammation ? 'is-positive' : ''}">
          <span class="checklist-text">${currentT.glasgow.minor_inflammation}</span>
          <span class="checklist-status ${glasgow.minor_inflammation ? 'yes' : 'no'}">
            ${glasgow.minor_inflammation ? currentT.glasgow.yes : currentT.glasgow.no}
          </span>
        </div>
        <div class="checklist-item ${glasgow.minor_oozing ? 'is-positive' : ''}">
          <span class="checklist-text">${currentT.glasgow.minor_oozing}</span>
          <span class="checklist-status ${glasgow.minor_oozing ? 'yes' : 'no'}">
            ${glasgow.minor_oozing ? currentT.glasgow.yes : currentT.glasgow.no}
          </span>
        </div>
        <div class="checklist-item ${glasgow.minor_sensation ? 'is-positive' : ''}">
          <span class="checklist-text">${currentT.glasgow.minor_sensation}</span>
          <span class="checklist-status ${glasgow.minor_sensation ? 'yes' : 'no'}">
            ${glasgow.minor_sensation ? currentT.glasgow.yes : currentT.glasgow.no}
          </span>
        </div>
      </div>
    </div>

    <!-- ABCD Educational Guide -->
    <div class="card">
      <div class="card-title">
        <span>${currentT.sections.abcdTitle}</span>
      </div>
      <div class="abcd-grid">
        <div class="abcd-card">
          <div class="abcd-header">
            <span class="abcd-letter">${currentT.abcd.a.letter}</span>
            <span class="abcd-name">${currentT.abcd.a.title}</span>
          </div>
          <p class="abcd-desc">${currentT.abcd.a.desc}</p>
        </div>
        <div class="abcd-card">
          <div class="abcd-header">
            <span class="abcd-letter">${currentT.abcd.b.letter}</span>
            <span class="abcd-name">${currentT.abcd.b.title}</span>
          </div>
          <p class="abcd-desc">${currentT.abcd.b.desc}</p>
        </div>
        <div class="abcd-card">
          <div class="abcd-header">
            <span class="abcd-letter">${currentT.abcd.c.letter}</span>
            <span class="abcd-name">${currentT.abcd.c.title}</span>
          </div>
          <p class="abcd-desc">${currentT.abcd.c.desc}</p>
        </div>
        <div class="abcd-card">
          <div class="abcd-header">
            <span class="abcd-letter">${currentT.abcd.d.letter}</span>
            <span class="abcd-name">${currentT.abcd.d.title}</span>
          </div>
          <p class="abcd-desc">${currentT.abcd.d.desc}</p>
        </div>
      </div>
    </div>

    <!-- Doctor Encounter Questions & Export Action -->
    <div class="card">
      <div class="card-title">
        <span>${currentT.sections.doctorMemoTitle}</span>
      </div>
      <div class="glasgow-citation">
        ${currentT.sections.questionsTitle}
      </div>
      <ul class="questions-list">
        ${(currentCase.doctorQuestions[currentLang] || currentT.memoQuestions)
          .map(
            (q) => `
          <li class="question-item">
            <span class="question-bullet">•</span>
            <span>${q}</span>
          </li>
        `
          )
          .join('')}
      </ul>

      <button class="btn-primary-action" id="download-memo-btn">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="12" y1="18" x2="12" y2="12"></line>
          <line x1="9" y1="15" x2="15" y2="15"></line>
        </svg>
        ${currentT.sections.downloadPdfBtn}
      </button>
    </div>

    <!-- Regulatory Non-Claim Disclaimer -->
    <div class="disclaimer-card">
      <div class="disclaimer-title">${currentT.sections.disclaimerTitle}</div>
      <div>${currentT.sections.disclaimerText}</div>
    </div>
  `;

  attachEventListeners();
  updateTelegramMainButton();
}

function updateTelegramMainButton() {
  const tg = getTelegramApp();
  if (tg?.MainButton) {
    const currentT = t();
    tg.MainButton.setText(currentT.sections.downloadPdfBtn);
    tg.MainButton.show();
    tg.MainButton.onClick(() => {
      triggerHaptic('medium');
      generateDoctorMemoPdf(currentCase, currentLang);
    });
  }
}

function attachEventListeners() {
  // Language buttons
  document.querySelectorAll<HTMLButtonElement>('.lang-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const newLang = btn.dataset.lang as Language;
      if (newLang && newLang !== currentLang) {
        triggerHaptic('light');
        setLanguage(newLang);
        currentLang = newLang;
        renderApp();
      }
    });
  });

  // Theme toggle
  const themeToggle = document.getElementById('theme-toggle');
  themeToggle?.addEventListener('click', () => {
    triggerHaptic('light');
    document.documentElement.classList.toggle('dark');
    renderApp();
  });

  // Case switcher
  document.querySelectorAll<HTMLButtonElement>('.case-tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const caseKey = btn.dataset.case as keyof typeof SAMPLE_CASES;
      if (caseKey && SAMPLE_CASES[caseKey]) {
        triggerHaptic('medium');
        currentCase = SAMPLE_CASES[caseKey];
        renderApp();
      }
    });
  });

  // PDF Download Button
  const downloadBtn = document.getElementById('download-memo-btn');
  downloadBtn?.addEventListener('click', () => {
    triggerNotificationHaptic('success');
    generateDoctorMemoPdf(currentCase, currentLang);
  });
}

// Initial Boot
document.addEventListener('DOMContentLoaded', () => {
  initTelegram();
  renderApp();
});

// Run once immediately in case DOM is already loaded
if (document.readyState === 'interactive' || document.readyState === 'complete') {
  initTelegram();
  renderApp();
}
