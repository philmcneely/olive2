import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactElement } from "react";
import { ChevronDown, Copy, Download, Link2 } from "lucide-react";
import { notify } from "./ui";
import { ShareLinksModal } from "./ShareLinksModal";
import { copyText, exportUrl, fetchExportText, type ExportFormat } from "../api/share";

/** The slice of GET /api/config this menu needs. */
export interface ShareConfig {
  shareLinksEnabled?: boolean;
  shareDefaultTtlHours?: number;
}

const FORMATS: { format: ExportFormat; label: string; hint: string }[] = [
  { format: "txt", label: "Plain text", hint: ".txt" },
  { format: "md", label: "Markdown", hint: ".md" },
  { format: "srt", label: "Subtitles", hint: ".srt" },
  { format: "vtt", label: "WebVTT", hint: ".vtt" },
  { format: "pdf", label: "PDF", hint: ".pdf" },
  { format: "docx", label: "Word", hint: ".docx" },
];

/** Export / copy / share dropdown for the meeting detail action row. */
export function ExportShareMenu({
  meetingId,
  title,
  hasTranscript,
  config,
}: {
  meetingId: string;
  title: string;
  hasTranscript: boolean;
  config: ShareConfig | null;
}): ReactElement {
  const [open, setOpen] = useState(false);
  const [includeSummary, setIncludeSummary] = useState(false);
  const [timestamps, setTimestamps] = useState(true);
  const [showShare, setShowShare] = useState(false);
  // Text is prefetched when the menu opens so the click handler can copy synchronously
  // (Safari drops user activation across an await, and execCommand needs it).
  const [prefetch, setPrefetch] = useState<{ key: string; text: string } | null>(null);
  const [prefetchError, setPrefetchError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  const key = `${meetingId}|${includeSummary}|${timestamps}`;
  const sharingEnabled = config?.shareLinksEnabled === true;

  useEffect(() => {
    if (!open || !hasTranscript) return;
    let cancelled = false;
    setPrefetchError(null);
    fetchExportText(meetingId, "txt", { summary: includeSummary, timestamps })
      .then((text) => {
        if (!cancelled) setPrefetch({ key, text });
      })
      .catch((err: unknown) => {
        if (!cancelled) setPrefetchError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [open, hasTranscript, meetingId, includeSummary, timestamps, key]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const ready = prefetch?.key === key;

  const handleCopy = useCallback(() => {
    if (!prefetch || prefetch.key !== key) return;
    void copyText(prefetch.text).then((ok) => {
      if (ok) {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
        notify("success", "Transcript copied to clipboard");
      } else {
        notify("error", "Copy failed. Use an export download instead.");
      }
    });
  }, [prefetch, key]);

  return (
    <div ref={rootRef} className={`dropdown dropdown-end shrink-0 ${open ? "dropdown-open" : ""}`}>
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="btn btn-primary btn-soft inline-flex min-h-[44px] items-center gap-1.5 border hover:border-primary/60 shadow-md"
        title={`Export or share "${title}"`}
      >
        <Download className="h-4 w-4" aria-hidden="true" />
        <span>Export</span>
        <ChevronDown className="h-4 w-4" aria-hidden="true" />
      </button>
      {open && (
        <div
          role="menu"
          className="dropdown-content z-30 mt-2 w-72 rounded-box border border-base-300 bg-base-200 p-2 shadow-xl"
        >
          {!hasTranscript && (
            <p className="px-2 py-1.5 text-xs text-base-content/70">Transcribe this meeting to enable export and sharing.</p>
          )}
          <div className="flex flex-col gap-1 px-2 pb-2 pt-1 text-xs">
            <label className="flex min-h-8 cursor-pointer items-center justify-between gap-2">
              <span>Include summary</span>
              <input
                type="checkbox"
                className="toggle toggle-sm toggle-primary"
                checked={includeSummary}
                onChange={(e) => setIncludeSummary(e.target.checked)}
              />
            </label>
            <label className="flex min-h-8 cursor-pointer items-center justify-between gap-2">
              <span>Include timestamps</span>
              <input
                type="checkbox"
                className="toggle toggle-sm toggle-primary"
                checked={timestamps}
                onChange={(e) => setTimestamps(e.target.checked)}
              />
            </label>
          </div>
          <ul className="menu menu-sm w-full p-0">
            <li className="menu-title">Download</li>
            {FORMATS.map((f) => (
              <li key={f.format} className={hasTranscript ? "" : "menu-disabled"}>
                {hasTranscript ? (
                  <a
                    role="menuitem"
                    href={exportUrl(meetingId, f.format, { summary: includeSummary, timestamps })}
                    download
                    onClick={() => setOpen(false)}
                    className="min-h-9"
                  >
                    <Download className="h-4 w-4" aria-hidden="true" />
                    <span className="flex-1">{f.label}</span>
                    <span className="text-xs text-base-content/60">{f.hint}</span>
                  </a>
                ) : (
                  <span className="min-h-9">
                    <Download className="h-4 w-4" aria-hidden="true" />
                    <span className="flex-1">{f.label}</span>
                    <span className="text-xs">{f.hint}</span>
                  </span>
                )}
              </li>
            ))}
            <li className="menu-title">Other</li>
            <li className={hasTranscript && ready ? "" : "menu-disabled"}>
              <button
                type="button"
                role="menuitem"
                disabled={!hasTranscript || !ready}
                onClick={handleCopy}
                className="min-h-9"
              >
                <Copy className="h-4 w-4" aria-hidden="true" />
                <span className="flex-1">
                  {copied ? "Copied" : hasTranscript && !ready && !prefetchError ? "Preparing copy…" : "Copy transcript"}
                </span>
              </button>
            </li>
            {prefetchError && <li className="px-3 py-1 text-xs text-error">{prefetchError}</li>}
            <li
              className={hasTranscript && sharingEnabled ? "" : "menu-disabled"}
              title={sharingEnabled ? undefined : "Sharing is disabled in settings"}
            >
              <button
                type="button"
                role="menuitem"
                disabled={!hasTranscript || !sharingEnabled}
                onClick={() => {
                  setOpen(false);
                  setShowShare(true);
                }}
                className="min-h-9"
              >
                <Link2 className="h-4 w-4" aria-hidden="true" />
                <span className="flex-1">Share link…</span>
                {!sharingEnabled && <span className="text-xs">disabled</span>}
              </button>
            </li>
          </ul>
        </div>
      )}
      {showShare && (
        <ShareLinksModal
          meetingId={meetingId}
          title={title}
          defaultTtlHours={config?.shareDefaultTtlHours ?? 168}
          onClose={() => setShowShare(false)}
        />
      )}
    </div>
  );
}
