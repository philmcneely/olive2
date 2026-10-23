// Typed fetch helper for GET /api/search (local FTS5 index over titles, transcripts, summaries and insight moments).
//
// The route returns a bare JSONResponse, so the shapes mirror `SearchResponse` in backend/src/olive/schemas.py
// (also generated into schema.d.ts as `SearchResponse`). No LLM is involved.

import type { OpenApiSchemas } from "./types";

export type SearchResponse = OpenApiSchemas["SearchResponse"];
export type SearchMeetingResult = OpenApiSchemas["SearchMeetingResult"];
export type SearchHit = OpenApiSchemas["SearchHit"];
export type SearchHitKind = SearchHit["kind"];

/** Snippet highlight delimiters emitted by the backend (control chars, so transcript text is never parsed as HTML). */
const MARK_OPEN = "\u0002";
const MARK_CLOSE = "\u0003";

export interface SnippetPart {
  text: string;
  match: boolean;
}

/** Split a backend snippet into plain / highlighted parts. */
export function parseSnippet(snippet: string): SnippetPart[] {
  const parts: SnippetPart[] = [];
  let rest = snippet;
  while (rest.length > 0) {
    const open = rest.indexOf(MARK_OPEN);
    if (open < 0) {
      parts.push({ text: rest, match: false });
      break;
    }
    if (open > 0) parts.push({ text: rest.slice(0, open), match: false });
    const close = rest.indexOf(MARK_CLOSE, open + 1);
    const end = close < 0 ? rest.length : close;
    parts.push({ text: rest.slice(open + 1, end), match: true });
    rest = close < 0 ? "" : rest.slice(close + 1);
  }
  return parts;
}

export async function searchMeetings(
  q: string,
  opts: { limit?: number; kinds?: SearchHitKind[]; signal?: AbortSignal } = {}
): Promise<SearchResponse> {
  const sp = new URLSearchParams({ q });
  if (opts.limit) sp.set("limit", String(opts.limit));
  if (opts.kinds && opts.kinds.length > 0) sp.set("kinds", opts.kinds.join(","));
  const res = await fetch(`/api/search?${sp.toString()}`, { signal: opts.signal });
  if (!res.ok) {
    const data = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(data.error || `Search failed (${res.status})`);
  }
  return (await res.json()) as SearchResponse;
}
