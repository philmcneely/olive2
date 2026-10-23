import { useEffect, useState } from "react";
import type { ReactElement } from "react";
import { Search } from "lucide-react";
import { Badge, Card, CardBody, Input } from "./ui";
import { meetingHref } from "../api/insights";
import { parseSnippet, searchMeetings, type SearchHit, type SearchHitKind, type SearchMeetingResult, type SearchResponse } from "../api/search";

interface SearchViewProps {
  /** Navigate the SPA to an app path (e.g. `/meetings/{id}?t=1234&q=term`). */
  onNavigate: (path: string) => void;
}

const DEBOUNCE_MS = 250;
const STORAGE_KEY = "olive.search.query";

const KIND_LABEL: Record<SearchHitKind, string> = {
  title: "Title",
  transcript: "Transcript",
  summary: "Summary",
  moment: "Moment",
};

const KIND_BADGE: Record<SearchHitKind, "primary" | "info" | "success" | "warning"> = {
  title: "primary",
  transcript: "info",
  summary: "success",
  moment: "warning",
};

function loadQuery(): string {
  try {
    return window.sessionStorage.getItem(STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

function saveQuery(q: string): void {
  try {
    window.sessionStorage.setItem(STORAGE_KEY, q);
  } catch {
    // storage unavailable; the query just isn't restored
  }
}

function formatWhen(ms: number): string {
  return new Date(ms).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function formatClock(ms: number): string {
  const total = Math.floor(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = String(m).padStart(h > 0 ? 2 : 1, "0");
  return `${h > 0 ? `${h}:` : ""}${mm}:${String(s).padStart(2, "0")}`;
}

/** First highlighted term in a snippet; used as the in-transcript highlight when deep-linking. */
function firstMatch(hit: SearchHit): string | null {
  return parseSnippet(hit.snippet).find((p) => p.match)?.text ?? null;
}

function hitHref(meetingId: string, hit: SearchHit): string {
  // Transcript and moment hits carry a timestamp the meeting page scrolls to; others open the meeting.
  if ((hit.kind === "transcript" || hit.kind === "moment") && hit.startMs !== null && hit.startMs !== undefined) {
    return meetingHref(meetingId, hit.startMs, firstMatch(hit));
  }
  return meetingHref(meetingId);
}

function Snippet({ text }: { text: string }): ReactElement {
  return (
    <p className="text-sm leading-relaxed text-base-content/80">
      {parseSnippet(text).map((part, i) =>
        part.match ? (
          <mark key={i} className="rounded bg-primary/30 px-0.5 text-base-content">
            {part.text}
          </mark>
        ) : (
          <span key={i}>{part.text}</span>
        )
      )}
    </p>
  );
}

function ResultCard({ result, onOpen }: { result: SearchMeetingResult; onOpen: (path: string) => void }): ReactElement {
  return (
    <Card>
      <CardBody className="gap-2">
        <div className="flex flex-wrap items-baseline justify-between gap-x-3">
          <a
            href={meetingHref(result.meetingId)}
            onClick={(e) => {
              e.preventDefault();
              onOpen(meetingHref(result.meetingId));
            }}
            className="font-semibold hover:text-primary"
          >
            {result.title}
          </a>
          <span className="text-xs text-base-content/60">{formatWhen(result.startTime)}</span>
        </div>
        <ul className="space-y-1.5">
          {result.hits.map((hit, i) => {
            const href = hitHref(result.meetingId, hit);
            return (
              <li key={`${hit.kind}-${hit.startMs ?? "x"}-${i}`}>
                <a
                  href={href}
                  onClick={(e) => {
                    e.preventDefault();
                    onOpen(href);
                  }}
                  className="flex min-h-11 items-start gap-2 rounded-lg px-2 py-1.5 hover:bg-base-300/60"
                >
                  <span className="mt-0.5 flex shrink-0 items-center gap-1.5">
                    <Badge variant={KIND_BADGE[hit.kind]} soft>
                      {KIND_LABEL[hit.kind]}
                    </Badge>
                    {hit.startMs !== null && hit.startMs !== undefined && (
                      <span className="font-mono text-xs text-base-content/50">{formatClock(hit.startMs)}</span>
                    )}
                  </span>
                  <Snippet text={hit.snippet} />
                </a>
              </li>
            );
          })}
        </ul>
      </CardBody>
    </Card>
  );
}

export function SearchView({ onNavigate }: SearchViewProps): ReactElement {
  const [query, setQuery] = useState(loadQuery);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    saveQuery(query);
    const q = query.trim();
    if (!q) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    const timer = window.setTimeout(() => {
      searchMeetings(q, { signal: controller.signal })
        .then((res) => {
          setData(res);
          setError(null);
        })
        .catch((exc: unknown) => {
          if (controller.signal.aborted) return;
          setError(exc instanceof Error ? exc.message : "Search failed");
        })
        .finally(() => {
          if (!controller.signal.aborted) setLoading(false);
        });
    }, DEBOUNCE_MS);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  return (
    <div className="space-y-5">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Search</h1>
        <p className="text-sm text-base-content/60">Find anything across every meeting: titles, transcripts, summaries and insight moments.</p>
      </header>

      <label className="relative block">
        <span className="sr-only">Search all meetings</span>
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-base-content/50" aria-hidden="true" />
        <Input
          autoFocus
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search meetings, e.g. “budget deadline”"
          className="min-h-11 pl-9"
        />
      </label>

      {error && <div className="rounded-xl border border-error bg-error/10 p-3 text-sm text-error">{error}</div>}

      {data && !error && (
        <p className="text-sm text-base-content/60" aria-live="polite">
          {loading
            ? "Searching…"
            : data.total === 0
              ? "No meetings match."
              : `${data.total} meeting${data.total === 1 ? "" : "s"} match${data.total > data.results.length ? ` (showing top ${data.results.length})` : ""}`}
        </p>
      )}

      <div className="space-y-3">
        {data?.results.map((r) => (
          <ResultCard key={r.meetingId} result={r} onOpen={onNavigate} />
        ))}
      </div>
    </div>
  );
}
