// Typed fetch helpers for Export & Share (backend/src/olive/routes_export_share.py).
//
// These routes are include_in_schema=False, so they are absent from the generated
// schema.d.ts and typed by hand here. Link JSON is camelCase (see _link_json).

export type ExportFormat = "txt" | "md" | "srt" | "vtt" | "pdf" | "docx";
export type ShareStatus = "active" | "expired" | "revoked";

export interface ShareLink {
  id: string;
  meetingId: string;
  label: string | null;
  includeSummary: boolean;
  /** epoch ms */
  createdAt: number;
  /** epoch ms, null = never expires */
  expiresAt: number | null;
  revokedAt: number | null;
  lastAccessedAt: number | null;
  accessCount: number;
  status: ShareStatus;
}

export interface CreateShareResponse {
  link: ShareLink;
  /** Returned ONCE, never listed again. */
  token: string;
  /** Relative path, e.g. "/share/<token>". */
  path: string;
}

export interface ShareListResponse {
  enabled: boolean;
  links: ShareLink[];
}

export interface CreateShareBody {
  /** Omit for the server default; null = never expires. */
  expiresInHours?: number | null;
  label?: string;
  includeSummary?: boolean;
}

export class ShareApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ShareApiError";
    this.status = status;
  }
  get disabled(): boolean {
    return this.status === 403;
  }
}

async function failure(res: Response): Promise<ShareApiError> {
  const data = (await res.json().catch(() => ({}))) as { error?: string; detail?: unknown };
  const detail = typeof data.detail === "string" ? data.detail : undefined;
  return new ShareApiError(data.error || detail || `Request failed (${res.status})`, res.status);
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) throw await failure(res);
  return (await res.json()) as T;
}

const base = (meetingId: string) => `/api/meetings/${encodeURIComponent(meetingId)}`;

export function exportUrl(
  meetingId: string,
  format: ExportFormat,
  opts: { summary?: boolean; timestamps?: boolean } = {},
): string {
  const sp = new URLSearchParams({
    format,
    summary: opts.summary ? "1" : "0",
    timestamps: opts.timestamps === false ? "0" : "1",
  });
  return `${base(meetingId)}/export?${sp.toString()}`;
}

/** Fetch an export as text (for the clipboard). */
export async function fetchExportText(
  meetingId: string,
  format: ExportFormat = "txt",
  opts: { summary?: boolean; timestamps?: boolean } = {},
): Promise<string> {
  const res = await fetch(exportUrl(meetingId, format, opts));
  if (!res.ok) throw await failure(res);
  return res.text();
}

export const createShareLink = (meetingId: string, body: CreateShareBody) =>
  request<CreateShareResponse>(`${base(meetingId)}/share`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

export const listShareLinks = (meetingId: string) => request<ShareListResponse>(`${base(meetingId)}/share`);

export const revokeShareLink = (meetingId: string, linkId: string) =>
  request<{ ok: boolean }>(`${base(meetingId)}/share/${encodeURIComponent(linkId)}`, { method: "DELETE" });

/** True when this page is served from a loopback host, where a share link is unreachable from other machines. */
export function isLoopbackHost(hostname: string = window.location.hostname): boolean {
  return hostname === "localhost" || hostname === "::1" || hostname === "[::1]" || /^127\./.test(hostname);
}

/**
 * Copy text to the clipboard. navigator.clipboard is undefined on non-secure contexts
 * (http://192.168.x.x), so fall back to a hidden textarea + execCommand("copy").
 * Must be called synchronously from a user gesture with text already in hand.
 */
export async function copyText(text: string): Promise<boolean> {
  if (window.isSecureContext && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // fall through to the legacy path
    }
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.top = "0";
  ta.style.left = "-9999px";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  const prevFocus = document.activeElement as HTMLElement | null;
  ta.focus();
  ta.select();
  ta.setSelectionRange(0, text.length);
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  document.body.removeChild(ta);
  prevFocus?.focus?.();
  return ok;
}
