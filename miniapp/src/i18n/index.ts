import { ru } from './ru';
import { en } from './en';
import { ko } from './ko';
import type { Language } from '../types';

export const translations = { ru, en, ko };

let currentLang: Language = 'ru';

export function getLanguage(): Language {
  return currentLang;
}

export function setLanguage(lang: Language) {
  currentLang = lang;
  localStorage.setItem('skinai_lang', lang);
  document.documentElement.lang = lang;
}

export function initLanguage(): Language {
  const saved = localStorage.getItem('skinai_lang') as Language;
  if (saved && (saved === 'ru' || saved === 'en' || saved === 'ko')) {
    currentLang = saved;
  } else {
    // Detect browser or Telegram language
    const navLang = navigator.language.toLowerCase();
    if (navLang.startsWith('ko')) {
      currentLang = 'ko';
    } else if (navLang.startsWith('en')) {
      currentLang = 'en';
    } else {
      currentLang = 'ru';
    }
  }
  document.documentElement.lang = currentLang;
  return currentLang;
}

export function t() {
  return translations[currentLang];
}
