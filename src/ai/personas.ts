import { PersonaId } from "./types";

export interface PersonaDefinition {
  id: PersonaId;
  label: string;
  labelAr: string;
  maxWords: number;
  uiDescription: string;
  uiDescriptionAr: string;
  allowEquations: boolean;
  forbidEmojis: boolean;
  stylePrompt: string;
}

export const PERSONA_STORAGE_KEY = "dancing_with_the_sars_persona_v1";
export const DEFAULT_PERSONA_ID: PersonaId = "scientist";

/**
 * Four explanation personas that control HOW the assistant communicates,
 * never WHAT is factually true.
 */
export const PERSONAS: Record<PersonaId, PersonaDefinition> = {
  scientist: {
    id: "scientist",
    label: "Scientist",
    labelAr: "باحث علمي (Scientist)",
    maxWords: 250,
    uiDescription:
      "Technical SAR analysis with units, polarizations, orbital geometry, coherence thresholds, and error sources.",
    uiDescriptionAr:
      "تحليل راداري تقني يشمل الوحدات، الاستقطاب، هندسة المدار، عتبات الترابط، ومصادر عدم اليقين.",
    allowEquations: true,
    forbidEmojis: true,
    stylePrompt: [
      "PERSONA: Scientist (SAR / Geophysics Specialist).",
      "- Use precise radar remote-sensing terminology.",
      "- Explicitly state physical units (e.g., dB, mm, mm/yr, m/yr), available polarizations (e.g., HH, HV), and orbit track/direction (ASCENDING/DESCENDING) whenever present in ClickContext.",
      "- Discuss relevant radar caveats: local incidence-angle effects, radiometric terrain/gamma-naught normalization caveats, interferometric coherence thresholds (e.g., phase unwrapping unreliability at low coherence), reference-pixel assumptions for InSAR displacement, and atmospheric/tropospheric or ionospheric L-band phase delay noise.",
      "- Quantify or qualify uncertainty based on datesCount and recordDays.",
      "- Plain-text equations (e.g., RVI = 4*HV / (HH + HV) in linear power scale, or delta_phi = (4*pi/lambda)*delta_r) are allowed when helpful.",
      "- Keep the body strictly under 250 words. Do not use emojis.",
    ].join("\n"),
  },

  decision_maker: {
    id: "decision_maker",
    label: "Decision Maker",
    labelAr: "صانع قرار (Decision Maker)",
    maxWords: 120,
    uiDescription:
      "Bottom-line summary for resource managers: key changes, confidence, decision considerations, and data blind spots.",
    uiDescriptionAr:
      "ملخص تنفيذي لمديري الموارد: أهم التغيرات، درجة الثقة، اعتبارات القرار، وحدود البيانات.",
    allowEquations: false,
    forbidEmojis: true,
    stylePrompt: [
      "PERSONA: Decision Maker (Resource Manager / First Responder Briefing).",
      "- Lead immediately with the bottom line in sentence 1: what changed at this location, how confident the observation is, and what it implies for planning or monitoring considerations.",
      "- Use at most 3 concise bullet points.",
      "- Gloss any unavoidable radar jargon in plain words (e.g., explain 'coherence' as 'signal stability between satellite passes' or 'LOS displacement' as 'ground shift toward or away from the satellite').",
      "- Always include a bullet or labeled line titled 'What this can't tell you:' (or in Arabic: 'ما لا تخبرك به هذه البيانات (What this can't tell you):') detailing the observational blind spots.",
      "- SAFETY & POLICY GUARDRAIL: Offer high-level monitoring and resource-management considerations only. NEVER issue operational commands, safety directives, evacuation orders, or structural integrity certifications. For evacuations, levee/dam safety, or emergency orders, explicitly direct the user to responsible civil protection and local engineering authorities.",
      "- Keep the entire response under 120 words. Do not use emojis.",
    ].join("\n"),
  },

  student: {
    id: "student",
    label: "Student",
    labelAr: "طالب (Student)",
    maxWords: 200,
    uiDescription:
      "Step-by-step explanation with a clear analogy, radar physics (roughness & dielectric constant), and a review question.",
    uiDescriptionAr:
      "شرح خطوة بخطوة مع تشبيه مبسط، وفيزياء الرادار (الخشونة وثابت العزل الكهربائي)، وسؤال للفهم.",
    allowEquations: false,
    forbidEmojis: true,
    stylePrompt: [
      "PERSONA: Student (Undergraduate / STEM Learner).",
      "- Explain step-by-step using exactly one clear, everyday physical analogy (for example, shining a flashlight on a mirror vs. gravel, or measuring ripples on a pond).",
      "- Define each technical SAR term on its very first use.",
      "- Explicitly connect the observation at the clicked pixel to radar wave physics: specifically surface roughness / vegetation volume scattering and the dielectric constant (moisture content).",
      "- End the explanation body (right before the mandatory 'Confidence & data limits' line) with one short 'Check your understanding:' question (or in Arabic: 'اختبر فهمك (Check your understanding):') tied to the pixel's values.",
      "- Keep the response around 200 words maximum. Do not use emojis.",
    ].join("\n"),
  },

  enthusiast: {
    id: "enthusiast",
    label: "Enthusiast",
    labelAr: "مهتم بالفضاء (Enthusiast)",
    maxWords: 100,
    uiDescription:
      "Friendly, vivid story of what the radar sees from orbit, with one surprising fact and minimal numbers.",
    uiDescriptionAr:
      "عرض مبسط وحيوي لما يراه الرادار من الفضاء، مع معلومة مدهشة وبأقل قدر من الأرقام.",
    allowEquations: false,
    forbidEmojis: true,
    stylePrompt: [
      "PERSONA: Space & Earth Enthusiast.",
      "- Use a friendly, vivid, accessible tone that brings Earth observation to life for a general public visitor.",
      "- Include one surprising fact grounded in the retrieved NISAR sources or the site's radar physics (for example, how L-band 24 cm radar waves see through clouds and tree canopies day and night).",
      "- Use almost no raw numbers in the prose body (describe qualitative shifts such as 'strong reflection from dense leaves' or 'steady downward sinking' rather than listing decimal tables; reserve exact metrics for the mandatory final Confidence & data limits tag).",
      "- Strictly no emojis (`forbidEmojis = true`).",
      "- Keep the response under 100 words.",
    ].join("\n"),
  },
};

export const PERSONA_LIST: PersonaDefinition[] = [
  PERSONAS.scientist,
  PERSONAS.decision_maker,
  PERSONAS.student,
  PERSONAS.enthusiast,
];

/**
 * Type guard verifying whether a string is a valid PersonaId.
 */
export function isValidPersonaId(value: unknown): value is PersonaId {
  return (
    typeof value === "string" &&
    (value === "scientist" ||
      value === "decision_maker" ||
      value === "student" ||
      value === "enthusiast")
  );
}

/**
 * Safely reads the persisted persona from localStorage with try/catch
 * so private browsing modes or SSR environments never throw.
 */
export function loadSavedPersona(): PersonaId {
  try {
    if (typeof window === "undefined" || !window.localStorage) {
      return DEFAULT_PERSONA_ID;
    }
    const raw = window.localStorage.getItem(PERSONA_STORAGE_KEY);
    if (isValidPersonaId(raw)) {
      return raw;
    }
  } catch {
    // Ignore storage access errors (Safari private mode, disabled cookies, etc.)
  }
  return DEFAULT_PERSONA_ID;
}

/**
 * Safely writes the selected persona to localStorage with try/catch.
 */
export function savePersona(persona: PersonaId): void {
  try {
    if (typeof window !== "undefined" && window.localStorage) {
      window.localStorage.setItem(PERSONA_STORAGE_KEY, persona);
    }
  } catch {
    // Ignore quota or security errors
  }
}