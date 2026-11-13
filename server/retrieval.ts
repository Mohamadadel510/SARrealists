import fs from "node:fs";
import path from "node:path";
import {
  CitedSource,
  ClickContext,
  RagChunkRecord,
  RetrievedChunk,
  Source,
} from "../src/ai/types";

interface RagIndexFile {
  version: number;
  model: string;
  provider: string;
  dimension: number;
  dtype: "float16" | "float32";
  chunk_count: number;
  chunks: RagChunkRecord[];
}

interface DecodedChunk {
  id: string;
  sourceId: string;
  section: string;
  page: number | null;
  text: string;
  vector: Float32Array;
}

interface LoadedRagStore {
  dimension: number;
  model: string;
  chunks: DecodedChunk[];
  sourcesById: Map<string, Source>;
}

/**
 * Module-scoped singleton cache so Vercel serverless functions decode the RAG index
 * only once per cold start.
 */
let cachedStore: LoadedRagStore | null = null;

/**
 * Converts a 16-bit IEEE 754 half-precision integer to a 32-bit float.
 */
export function decodeFloat16Word(h: number): number {
  const sign = (h & 0x8000) >> 15;
  const exponent = (h & 0x7c00) >> 10;
  const fraction = h & 0x03ff;

  if (exponent === 0) {
    if (fraction === 0) {
      return sign === 1 ? -0.0 : 0.0;
    }
    // Subnormal half-precision number
    const val = (fraction / 1024) * Math.pow(2, -14);
    return sign === 1 ? -val : val;
  }

  if (exponent === 0x1f) {
    return fraction === 0 ? (sign === 1 ? -Infinity : Infinity) : NaN;
  }

  const val = (1 + fraction / 1024) * Math.pow(2, exponent - 15);
  return sign === 1 ? -val : val;
}

/**
 * Decodes a base64-encoded float16 or float32 vector into a normalized Float32Array.
 */
export function decodeBase64Embedding(
  b64: string,
  dtype: "float16" | "float32" = "float16"
): Float32Array {
  const buf = Buffer.from(b64, "base64");
  if (dtype === "float32") {
    const len = Math.floor(buf.byteLength / 4);
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      out[i] = buf.readFloatLE(i * 4);
    }
    return normalizeVector(out);
  }

  const len = Math.floor(buf.byteLength / 2);
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const word = buf.readUInt16LE(i * 2);
    out[i] = decodeFloat16Word(word);
  }
  return normalizeVector(out);
}

/**
 * Returns a unit-length L2-normalized copy of a Float32Array.
 */
export function normalizeVector(vec: Float32Array): Float32Array {
  let sumSq = 0;
  for (let i = 0; i < vec.length; i++) {
    sumSq += vec[i] * vec[i];
  }
  const norm = Math.sqrt(sumSq);
  if (norm === 0 || !Number.isFinite(norm)) {
    return vec;
  }
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    out[i] = vec[i] / norm;
  }
  return out;
}

/**
 * Computes cosine similarity between two L2-normalized Float32Arrays.
 * Returns 0 if dimensions mismatch.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length === 0 || a.length !== b.length) {
    return 0;
  }
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

/**
 * Loads `public/rag/index.json` and `public/rag/sources.json` into memory once per cold start.
 * Supports injecting custom paths for unit testing.
 */
export function loadRagStore(options?: {
  indexPath?: string;
  sourcesPath?: string;
  forceReload?: boolean;
}): LoadedRagStore {
  if (cachedStore && !options?.forceReload) {
    return cachedStore;
  }

  const indexPath =
    options?.indexPath ??
    path.resolve(process.cwd(), "public", "rag", "index.json");
  const sourcesPath =
    options?.sourcesPath ??
    path.resolve(process.cwd(), "public", "rag", "sources.json");

  if (!fs.existsSync(indexPath) || !fs.existsSync(sourcesPath)) {
    cachedStore = {
      dimension: 0,
      model: "uninitialized",
      chunks: [],
      sourcesById: new Map<string, Source>(),
    };
    return cachedStore;
  }

  const rawIndex = JSON.parse(
    fs.readFileSync(indexPath, "utf-8")
  ) as RagIndexFile;
  const rawSources = JSON.parse(
    fs.readFileSync(sourcesPath, "utf-8")
  ) as Source[];

  const sourcesById = new Map<string, Source>();
  for (const src of rawSources) {
    sourcesById.set(src.id, src);
  }

  const defaultDtype = rawIndex.dtype ?? "float16";
  const decodedChunks: DecodedChunk[] = rawIndex.chunks.map((c) => ({
    id: c.id,
    sourceId: c.source_id,
    section: c.section,
    page: c.page ?? null,
    text: c.text,
    vector: decodeBase64Embedding(c.embedding, c.dtype ?? defaultDtype),
  }));

  cachedStore = {
    dimension: rawIndex.dimension,
    model: rawIndex.model,
    chunks: decodedChunks,
    sourcesById,
  };
  return cachedStore;
}

/**
 * Builds the composite retrieval query from the user question, clicked site, and active layer
 * as required by the specification: `Retrieval query = user question + site + active layer`.
 */
export function buildRetrievalQuery(
  question: string,
  clickContext?: ClickContext
): string {
  const parts: string[] = [question.trim()];
  if (clickContext?.siteId) {
    const siteDescriptive: Record<ClickContext["siteId"], string> = {
      california: "California wildfires vegetation backscatter RVI L-band SAR",
      nile_delta:
        "Nile Delta agriculture soil moisture coastal erosion land subsidence InSAR",
      himalaya:
        "Himalaya Gangotri glacier velocity ice motion cryosphere pixel offsets InSAR",
    };
    parts.push(`Site: ${clickContext.siteId} (${siteDescriptive[clickContext.siteId]})`);
  }
  if (clickContext?.activeLayer) {
    parts.push(`Active Layer: ${clickContext.activeLayer}`);
  }
  return parts.join(" | ");
}

/**
 * Computes a lexical keyword-overlap score in [0, 1] used only when no query embedding
 * vector is available (e.g., offline mode or embedding rate-limit fallback).
 */
function computeLexicalOverlapScore(query: string, text: string): number {
  const qTokens = query
    .toLowerCase()
    .split(/[^a-z0-9\u0600-\u06FF]+/)
    .filter((t) => t.length >= 3);
  if (qTokens.length === 0) {
    return 0;
  }
  const uniqueQ = Array.from(new Set(qTokens));
  const hay = text.toLowerCase();
  let hits = 0;
  for (const token of uniqueQ) {
    if (hay.includes(token)) {
      hits += 1;
    }
  }
  return hits / uniqueQ.length;
}

export interface RetrieveOptions {
  question: string;
  clickContext?: ClickContext;
  queryEmbedding?: Float32Array | number[];
  topK?: number;
  minScore?: number;
  indexPath?: string;
  sourcesPath?: string;
}

export interface RetrievalResult {
  query: string;
  chunks: RetrievedChunk[];
  citedSources: CitedSource[];
  usedLexicalFallback: boolean;
}

/**
 * Retrieves top-k chunks by cosine similarity with a configurable similarity floor
 * and deduplication by `(sourceId, section)` so a single section does not crowd out
 * diverse sources. Assigns sequential citation tags `[S1]..[Sk]`.
 */
export function retrieveTopChunks(options: RetrieveOptions): RetrievalResult {
  const topK = options.topK ?? 5;
  const minScore = options.minScore ?? 0.28;
  const store = loadRagStore({
    indexPath: options.indexPath,
    sourcesPath: options.sourcesPath,
  });

  const compositeQuery = buildRetrievalQuery(
    options.question,
    options.clickContext
  );

  if (store.chunks.length === 0) {
    return {
      query: compositeQuery,
      chunks: [],
      citedSources: [],
      usedLexicalFallback: false,
    };
  }

  let queryVec: Float32Array | null = null;
  if (options.queryEmbedding) {
    const rawArr =
      options.queryEmbedding instanceof Float32Array
        ? options.queryEmbedding
        : new Float32Array(options.queryEmbedding);
    if (rawArr.length === store.dimension) {
      queryVec = normalizeVector(rawArr);
    }
  }

  const usedLexicalFallback = queryVec === null;
  const effectiveFloor = usedLexicalFallback
    ? Math.min(minScore, 0.15)
    : minScore;

  const scored: Array<{ chunk: DecodedChunk; score: number }> = [];
  for (const chunk of store.chunks) {
    const score = queryVec
      ? cosineSimilarity(queryVec, chunk.vector)
      : computeLexicalOverlapScore(
          compositeQuery,
          `${chunk.section} ${chunk.text}`
        );
    if (score >= effectiveFloor) {
      scored.push({ chunk, score });
    }
  }

  scored.sort((a, b) => b.score - a.score);

  // Deduplicate by (sourceId + normalized section heading)
  const seenSourceSections = new Set<string>();
  const selectedChunks: RetrievedChunk[] = [];
  const citedSources: CitedSource[] = [];

  for (const item of scored) {
    if (selectedChunks.length >= topK) {
      break;
    }
    const dedupeKey = `${item.chunk.sourceId}::${item.chunk.section.trim().toLowerCase()}`;
    if (seenSourceSections.has(dedupeKey)) {
      continue;
    }
    seenSourceSections.add(dedupeKey);

    const srcMeta = store.sourcesById.get(item.chunk.sourceId) ?? {
      id: item.chunk.sourceId,
      title: item.chunk.sourceId,
      url: "",
      attribution: "NASA / JPL / ASF",
    };

    const citationIndex = selectedChunks.length + 1;
    const citationTag = `S${citationIndex}`;

    selectedChunks.push({
      id: item.chunk.id,
      citationTag,
      sourceId: srcMeta.id,
      sourceTitle: srcMeta.title,
      sourceUrl: srcMeta.url,
      attribution: srcMeta.attribution,
      section: item.chunk.section,
      page: item.chunk.page,
      text: item.chunk.text,
      score: Number(item.score.toFixed(4)),
    });

    citedSources.push({
      id: srcMeta.id,
      title: srcMeta.title,
      url: srcMeta.url,
      attribution: srcMeta.attribution,
      citationTag,
      section: item.chunk.section,
      page: item.chunk.page,
      score: Number(item.score.toFixed(4)),
    });
  }

  return {
    query: compositeQuery,
    chunks: selectedChunks,
    citedSources,
    usedLexicalFallback,
  };
}