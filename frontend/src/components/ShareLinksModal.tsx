import { useCallback, useEffect, useState } from "react";
import type { ReactElement } from "react";
import { AlertTriangle, Copy } from "lucide-react";
import { Badge, Button, Field, Input, Modal, Toggle, confirmAction, notify } from "./ui";
import {
  ShareApiError,
  copyText,
  createShareLink,
  isLoopbackHost,
  listShareLinks,
  revokeShareLink,
  type ShareLink,
  type ShareStatus,
} from "../api/share";

const EXPIRY_CHOICES: { value: string; label: string; hours: number | null }[] = [
  { value: "1", label: "1 hour", hours: 1 },
  { value: "24", label: "24 hours", hours: 24 },
  { value: "168", label: "7 days", hours: 168 },
  { value: "720", label: "30 days", hours: 720 },
  { value: "never", label: "Never", hours: null },
];

const STATUS_VARIANT: Record<ShareStatus, "success" | "warning" | "error"> = {
  active: "success",
  expired: "warning",
  revoked: "error",
};

function pickDefaultExpiry(ttlHours: number): string {
  // 0 (or negative) means "never" per shareDefaultTtlHours.
  if (ttlHours <= 0) return "never";
  const exact = EXPIRY_CHOICES.find((c) => c.hours === ttlHours);
  return exact ? exact.value : "168";
}

function fmt(ms: number | null): string {
  return ms === null ? "—" : new Date(ms).toLocaleString();
}

/** Create, list and revoke public share links for one meeting. */
export function ShareLinksModal({
  meetingId,
  title,
  defaultTtlHours,
  onClose,
}: {
  meetingId: string;
  title: string;
  defaultTtlHours: number;
  onClose: () => void;
}): ReactElement {
  const [links, setLinks] = useState<ShareLink[] | null>(null);
  const [enabled, setEnabled] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [expiry, setExpiry] = useState(() => pickDefaultExpiry(defaultTtlHours));
  const [label, setLabel] = useState("");
  const [includeSummary, setIncludeSummary] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [created, setCreated] = useState<{ url: string } | null>(null);
  const [copied, setCopied] = useState(false);
  const loopback = isLoopbackHost();

  const reload = useCallback(async () => {
    try {
      const res = await listShareLinks(meetingId);
      setLinks(res.links);
      setEnabled(res.enabled);
      setLoadError(null);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err));
    }
  }, [meetingId]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    setCreateError(null);
    setCopied(false);
    try {
      const choice = EXPIRY_CHOICES.find((c) => c.value === expiry);
      const res = await createShareLink(meetingId, {
        expiresInHours: choice ? choice.hours : null,
        label: label.trim() || undefined,
        includeSummary,
      });
      setCreated({ url: `${window.location.origin}${res.path}` });
      setLabel("");
      void reload();
    } catch (err) {
      setCreateError(
        err instanceof ShareApiError && err.disabled
          ? "Sharing is disabled in settings."
          : err instanceof Error
          ? err.message
          : String(err),
      );
    } finally {
      setCreating(false);
    }
  };

  const handleCopy = () => {
    if (!created) return;
    void copyText(created.url).then((ok) => {
      if (ok) {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } else {
        notify("error", "Copy failed. Select the link and copy it manually.");
      }
    });
  };

  const handleRevoke = async (link: ShareLink) => {
    const name = link.label || "this link";
    if (
      !(await confirmAction({
        title: "Revoke share link",
        message: `Revoke ${name}? Anyone holding it will lose access immediately.`,
        confirmLabel: "Revoke",
        danger: true,
      }))
    ) {
      return;
    }
    try {
      await revokeShareLink(meetingId, link.id);
      notify("success", "Share link revoked");
      await reload();
    } catch (err) {
      notify("error", err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <Modal onClose={onClose} title="Share link" ariaLabel={`Share links for ${title}`} maxWidth="max-w-2xl">
      <div className="space-y-4 text-sm">
        {loopback && (
          <div className="alert alert-warning alert-soft" role="alert">
            <AlertTriangle className="h-4 w-4 shrink-0" aria-hidden="true" />
            <span>
              You are viewing Olive at <span className="font-mono">{window.location.host}</span>. A link to a loopback address only works on this machine. Open Olive by its network
              address or hostname before sharing.
            </span>
          </div>
        )}

        {!enabled && (
          <div className="alert alert-warning alert-soft">
            <span>Sharing is disabled in settings. You can still revoke existing links.</span>
          </div>
        )}

        <p className="text-xs text-base-content/70">
          Share links are <span className="font-semibold text-base-content">live</span>: edits and renames to this meeting show up in the shared view.
          Anyone with the link can read the transcript until it expires or you revoke it.
        </p>

        {enabled && (
          <form onSubmit={handleCreate} className="space-y-3 rounded-xl border border-base-300 bg-base-200/60 p-3">
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Expires" htmlFor="share-expiry">
                <select
                  id="share-expiry"
                  className="select w-full"
                  value={expiry}
                  onChange={(e) => setExpiry(e.target.value)}
                >
                  {EXPIRY_CHOICES.map((c) => (
                    <option key={c.value} value={c.value}>
                      {c.label}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Label (optional)" htmlFor="share-label">
                <Input
                  id="share-label"
                  value={label}
                  maxLength={80}
                  onChange={(e) => setLabel(e.target.value)}
                  placeholder="e.g. Sent to Dana"
                />
              </Field>
            </div>
            <label className="flex cursor-pointer items-center gap-3">
              <Toggle
                className="toggle-sm toggle-primary"
                checked={includeSummary}
                onChange={(e) => setIncludeSummary(e.target.checked)}
              />
              <span>Include the summary in the shared view</span>
            </label>
            {createError && <div className="rounded-xl border border-error bg-error/10 p-3 text-xs text-error">{createError}</div>}
            <div className="flex justify-end">
              <Button type="submit" variant="primary" disabled={creating}>
                {creating ? "Creating…" : "Create link"}
              </Button>
            </div>
          </form>
        )}

        {created && (
          <div className="space-y-2 rounded-xl border border-success/40 bg-success/10 p-3">
            <p className="text-xs font-semibold text-success">
              Link created. This is the only time it is shown. Copy it now; it cannot be retrieved later.
            </p>
            <div className="flex items-center gap-2">
              <input
                readOnly
                aria-label="Share link"
                value={created.url}
                onFocus={(e) => e.currentTarget.select()}
                className="input w-full min-w-0 font-mono text-xs"
              />
              <Button variant="primary" onClick={handleCopy} className="shrink-0">
                <Copy className="h-4 w-4" aria-hidden="true" />
                {copied ? "Copied" : "Copy"}
              </Button>
            </div>
          </div>
        )}

        <div>
          <h3 className="mb-1 text-sm font-semibold">Existing links</h3>
          <p className="mb-2 text-xs text-base-content/70">
            Older links cannot be copied again because Olive does not store the link itself. Revoke and create a new one if you lost it.
          </p>
          {loadError && <div className="rounded-xl border border-error bg-error/10 p-3 text-xs text-error">{loadError}</div>}
          {links === null && !loadError && <p className="text-xs text-base-content/70">Loading…</p>}
          {links !== null && links.length === 0 && <p className="text-xs text-base-content/70">No share links yet.</p>}
          {links !== null && links.length > 0 && (
            <ul className="space-y-2">
              {links.map((link) => (
                <li key={link.id} className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-xl border border-base-300 p-3">
                  <div className="min-w-0 flex-1 space-y-0.5">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant={STATUS_VARIANT[link.status]} soft>
                        {link.status}
                      </Badge>
                      <span className="truncate font-medium">{link.label || "Untitled link"}</span>
                      {link.includeSummary && <Badge variant="info" soft>summary</Badge>}
                    </div>
                    <p className="text-xs text-base-content/70">
                      Created {fmt(link.createdAt)} · Expires {link.expiresAt === null ? "never" : fmt(link.expiresAt)} · {link.accessCount}{" "}
                      {link.accessCount === 1 ? "view" : "views"}
                      {link.lastAccessedAt !== null ? ` · last ${fmt(link.lastAccessedAt)}` : ""}
                    </p>
                  </div>
                  {link.status === "active" && (
                    <Button variant="dangerSoft" size="sm" onClick={() => void handleRevoke(link)}>
                      Revoke
                    </Button>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </Modal>
  );
}
