import { z } from "zod";

/**
 * Case-study site identifiers supported by "Dancing with the SARs".
 */
export type SiteId = "california" | "nile_delta" | "himalaya";

/**
 * Radar orbital pass direction.
 */
export type OrbitDirection = "ASCENDING" | "DESCENDING";

/**
 * Supported explanation personas.
 * Personas alter HOW the assistant explains, never WHAT is factually true.
 */
export type PersonaId =
  | "scientist"
  | "decision_maker"
  | "student"
  | "enthusiast";

/**
 * Supported UI and response languages.
 */
export type SupportedLanguage = "en" | "ar";

/**
 * Time-series arrays optionally attached to a clicked location.
 */
export interface ClickSeries {
  dates: string[];
  hv_db?: (number | null)[];
  coh?: (number | null)[];
  disp_mm?: (number | null)[];
}

/**
 * Per-site metadata notes injected from static catalog.json.
 */
export interface SiteNotes {
  why?: string;
  limits?: string;
}

/**
 * Context injected when a user clicks a pixel/point on the MapLibre map.
 * The assistant never reads rasters directly; it only sees these pre-extracted values.
 */
export interface ClickContext {
  siteId: SiteId;
  lon: number;
  lat: number;
  date: string; // ISO date (YYYY-MM-DD) of the selected scene
  track?: number;
  direction?: OrbitDirection;
  values: Record<string, number | null>; // e.g. hh_db, hv_db, rvi, coherence, los_disp_mm, disp_rate_mm_yr, velocity_m_yr, soil_moisture
  series?: ClickSeries;
  polsAvailable: string[]; // e.g. ["HH", "HV"] or ["HH"]
  datesCount: number;
  recordDays: number;
  siteNotes?: SiteNotes; // from catalog.json
  activeLayer?: string;
}

/**
 * ISO date regex (YYYY-MM-DD or full ISO-8601 timestamp).
 */
const ISO_DATE_REGEX = /^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?)?$/;

/**
 * Safe key regex for raster layer value dictionaries.
 */
const VALUE_KEY_REGEX = /^[a-zA-Z0-9_.-]{1,64}$/;

/**
 * Server-side Zod validation schema for ClickContext.
 * Rejects malformed coordinates, non-finite numbers, and oversized string payloads
 * that could be used as prompt-injection vectors.
 */
export const ClickContextSchema: z.ZodType<ClickContext> = z
  .object({
    siteId: z.enum(["california", "nile_delta", "himalaya"]),
    lon: z.number().finite().min(-180).max(180),
    lat: z.number().finite().min(-90).max(90),
    date: z
      .string()
      .min(10)
      .max(40)
      .regex(ISO_DATE_REGEX, "date must be a valid ISO-8601 date string"),
    track: z.number().int().min(1).max(500).optional(),
    direction: z.enum(["ASCENDING", "DESCENDING"]).optional(),
    values: z
      .record(
        z.string().regex(VALUE_KEY_REGEX, "Invalid value metric key"),
        z.number().finite().nullable()
      )
      .refine((obj) => Object.keys(obj).length <= 32, {
        message: "values object cannot exceed 32 metrics",
      }),
    series: z
      .object({
        dates: z
          .array(z.string().regex(ISO_DATE_REGEX))
          .max(500),
        hv_db: z.array(z.number().finite().nullable()).max(500).optional(),
        coh: z.array(z.number().finite().nullable()).max(500).optional(),
        disp_mm: z.array(z.number().finite().nullable()).max(500).optional(),
      })
      .strict()
      .optional(),
    polsAvailable: z
      .array(z.string().min(1).max(16))
      .min(1)
      .max(8),
    datesCount: z.number().int().min(0).max(10000),
    recordDays: z.number().int().min(0).max(100000),
    siteNotes: z
      .object({
        why: z.string().max(1000).optional(),
        limits: z.string().max(1000).optional(),
      })
      .strict()
      .optional(),
    activeLayer: z.string().min(1).max(64).optional(),
  })
  .strict();

/**
 * JSON Schema (Draft-07) representation of ClickContext for external tooling,
 * documentation, and OpenAPI validation.
 */
export const ClickContextJsonSchema = {
  $schema: "http://json-schema.org/draft-07/schema#",
  title: "ClickContext",
  type: "object",
  additionalProperties: false,
  required: [
    "siteId",
    "lon",
    "lat",
    "date",
    "values",
    "polsAvailable",
    "datesCount",
    "recordDays",
  ],
  properties: {
    siteId: {
      type: "string",
      enum: ["california", "nile_delta", "himalaya"],
    },
    lon: { type: "number", minimum: -180, maximum: 180 },
    lat: { type: "number", minimum: -90, maximum: 90 },
    date: { type: "string", pattern: ISO_DATE_REGEX.source },
    track: { type: "integer", minimum: 1, maximum: 500 },
    direction: { type: "string", enum: ["ASCENDING", "DESCENDING"] },
    values: {
      type: "object",
      propertyNames: { pattern: VALUE_KEY_REGEX.source },
      additionalProperties: { type: ["number", "null"] },
    },
    series: {
      type: "object",
      additionalProperties: false,
      required: ["dates"],
      properties: {
        dates: {
          type: "array",
          items: { type: "string", pattern: ISO_DATE_REGEX.source },
        },
        hv_db: { type: "array", items: { type: ["number", "null"] } },
        coh: { type: "array", items: { type: ["number", "null"] } },
        disp_mm: { type: "array", items: { type: ["number", "null"] } },
      },
    },
    polsAvailable: {
      type: "array",
      minItems: 1,
      maxItems: 8,
      items: { type: "string" },
    },
    datesCount: { type: "integer", minimum: 0 },
    recordDays: { type: "integer", minimum: 0 },
    siteNotes: {
      type: "object",
      additionalProperties: false,
      properties: {
        why: { type: "string", maxLength: 1000 },
        limits: { type: "string", maxLength: 1000 },
      },
    },
    activeLayer: { type: "string", maxLength: 64 },
  },
} as const;

/**
 * Whitelisted map layer identifiers for Phase 2 structured actions.
 */
export const ALLOWED_LAYER_IDS = [
  "hh_db",
  "hv_db",
  "rvi",
  "coherence",
  "change",
  "los_disp_mm",
  "disp_rate_mm_yr",
  "velocity_m_yr",
  "soil_moisture",
] as const;

export type AllowedLayerId = (typeof ALLOWED_LAYER_IDS)[number];

/**
 * Phase 2 optional structured navigation actions.
 * The UI only executes actions that validate against MapActionSchema.
 */
export type MapAction =
  | { type: "flyTo"; siteId: SiteId }
  | { type: "setLayer"; layerId: AllowedLayerId }
  | { type: "setDate"; date: string }
  | { type: "setPersona"; persona: PersonaId };

export const MapActionSchema: z.ZodType<MapAction> = z.discriminatedUnion(
  "type",
  [
    z
      .object({
        type: z.literal("flyTo"),
        siteId: z.enum(["california", "nile_delta", "himalaya"]),
      })
      .strict(),
    z
      .object({
        type: z.literal("setLayer"),
        layerId: z.enum(ALLOWED_LAYER_IDS),
      })
      .strict(),
    z
      .object({
        type: z.literal("setDate"),
        date: z.string().regex(ISO_DATE_REGEX),
      })
      .strict(),
    z
      .object({
        type: z.literal("setPersona"),
        persona: z.enum(["scientist", "decision_maker", "student", "enthusiast"]),
      })
      .strict(),
  ]
);

/**
 * Filters an untrusted array of potential actions, keeping only schema-valid
 * whitelisted actions and silently dropping everything else.
 */
export function filterValidActions(rawActions: unknown): MapAction[] {
  if (!Array.isArray(rawActions)) {
    return [];
  }
  const valid: MapAction[] = [];
  for (const candidate of rawActions) {
    const parsed = MapActionSchema.safeParse(candidate);
    if (parsed.success) {
      valid.push(parsed.data);
    }
  }
  return valid;
}

/**
 * Attribution entry in public/rag/sources.json.
 */
export interface Source {
  id: string;
  title: string;
  url: string;
  attribution: string;
}

/**
 * Source citation returned to the client with a citation tag (e.g. "S1").
 */
export interface CitedSource extends Source {
  citationTag: string; // e.g., "S1", "S2"
  section?: string;
  page?: number | null;
  score?: number;
}

/**
 * Chunk representation inside public/rag/index.json.
 */
export interface RagChunkRecord {
  id: string;
  source_id: string;
  section: string;
  page: number | null;
  text: string;
  embedding: string; // base64-encoded float16 or float32 vector
  dtype?: "float16" | "float32";
}

/**
 * Ranked chunk returned by server/retrieval.ts.
 */
export interface RetrievedChunk {
  id: string;
  citationTag: string; // "S1", "S2", ...
  sourceId: string;
  sourceTitle: string;
  sourceUrl: string;
  attribution: string;
  section: string;
  page: number | null;
  text: string;
  score: number;
}

/**
 * Incoming chat request payload sent from ChatPanel to /api/chat.
 */
export interface ChatRequest {
  question: string;
  persona: PersonaId;
  clickContext?: ClickContext;
  language?: SupportedLanguage;
  enableActions?: boolean;
}

export const ChatRequestSchema: z.ZodType<ChatRequest> = z
  .object({
    question: z
      .string()
      .trim()
      .min(1, "Question cannot be empty")
      .max(1200, "Question exceeds maximum allowed length of 1200 characters"),
    persona: z.enum(["scientist", "decision_maker", "student", "enthusiast"]),
    clickContext: ClickContextSchema.optional(),
    language: z.enum(["en", "ar"]).optional(),
    enableActions: z.boolean().optional(),
  })
  .strict();

/**
 * Structured response returned by /api/chat (or final SSE event in streaming mode).
 */
export interface ChatResponse {
  answer: string;
  persona: PersonaId;
  language: SupportedLanguage;
  sources: CitedSource[];
  confidenceTag: string;
  actions?: MapAction[];
  isFallback?: boolean;
}