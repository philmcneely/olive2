import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactElement, ReactNode } from "react";
import { AlertTriangle, ArrowLeft, Pencil, Plus, RefreshCw, Trash2, Users, X } from "lucide-react";
import { Badge, Button, Card, CardBody, Checkbox, Input, Modal, Textarea, confirmAction, notify } from "./ui";
import {
  insightGroupsApi,
  type InsightGroup,
  type InsightGroupAggregate,
  type InsightGroupMemberPoint,
  type InsightGroupSnapshot,
  type InsightGroupTrends,
} from "../api/insightGroups";
import { InsightsApiError, insightsApi, meetingHref } from "../api/insights";
import { InsightGroupCompareSection } from "./InsightGroupCompare";

type Open = (meetingId: string, startMs?: number | null) => void;

function errorMessage(exc: unknown, fallback: string): string {
  return exc instanceof Error ? exc.message : fallback;
}

function fmtNum(n: number): string {
  return (Math.round(n * 100) / 100).toFixed(2);
}

function fmtDimension(name: string): string {
  return name.replace(/[_-]+/g, " ");
}

function fmtDate(ms: number): string {
  return new Date(ms).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

/** Same label InsightsView uses: every score is an LLM read of the transcript. */
function InferredBadge(): ReactElement {
  return (
    <span title="Scored by a language model from the transcript text. Treat as a lead to check, not a measurement.">
      <Badge variant="outline">model-inferred</Badge>
    </span>
  );
}

function SectionHeader({ title, hint, right }: { title: string; hint?: ReactNode; right?: ReactNode }): ReactElement {
  return (
    <div className="flex flex-wrap items-start justify-between gap-2">
      <div className="min-w-0">
        <h2 className="text-lg font-semibold tracking-tight text-base-content">{title}</h2>
        {hint && <p className="text-xs text-base-content/70">{hint}</p>}
      </div>
      {right}
    </div>
  );
}

function axesLabel(axes?: string[] | null): string {
  return axes && axes.length > 0 ? axes.map(fmtDimension).join(", ") : "all axes";
}

/* ------------------------------------------------------------------ */
/* Axis multi-select (dimension keys come from the Insights overview)  */
/* ------------------------------------------------------------------ */

function useAvailableAxes(): string[] {
  const [axes, setAxes] = useState<string[]>([]);
  useEffect(() => {
    let cancelled = false;
    insightsApi
      .overview()
      .then((o) => {
        if (!cancelled) setAxes(Object.keys(o.series));
      })
      .catch(() => {
        // The picker is optional: an empty selection already means "all axes".
      });
    return () => {
      cancelled = true;
    };
  }, []);
  return axes;
}

function AxisPicker({ available, value, onChange }: { available: string[]; value: string[]; onChange: (axes: string[]) => void }): ReactElement {
  // Keep axes already saved on the group selectable even if no signals currently report them.
  const options = useMemo(() => Array.from(new Set([...available, ...value])), [available, value]);
  if (options.length === 0) {
    return <p className="text-xs text-base-content/70">No axes available yet. The group will analyze every axis once meetings are analyzed.</p>;
  }
  return (
    <div className="flex flex-wrap gap-1.5" role="group" aria-label="Axes to analyze">
      {options.map((a) => {
        const on = value.includes(a);
        return (
          <button
            key={a}
            type="button"
            aria-pressed={on}
            onClick={() => onChange(on ? value.filter((x) => x !== a) : [...value, a])}
            className={`btn btn-xs min-h-8 capitalize ${on ? "btn-primary" : "btn-ghost border-base-300"}`}
          >
            {fmtDimension(a)}
          </button>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Create / edit modal                                                 */
/* ------------------------------------------------------------------ */

function GroupFormModal({ group, onClose, onSaved }: { group: InsightGroup | null; onClose: () => void; onSaved: (g: InsightGroup) => void }): ReactElement {
  const available = useAvailableAxes();
  const [name, setName] = useState(group?.name ?? "");
  const [description, setDescription] = useState(group?.description ?? "");
  const [axes, setAxes] = useState<string[]>(group?.axes ?? []);
  const [saving, setSaving] = useState(false);

  const save = async () => {
    if (!name.trim()) return;
    setSaving(true);
    try {
      const body = { name: name.trim(), description: description.trim() || null, axes: axes.length > 0 ? axes : null };
      const saved = group ? await insightGroupsApi.update(group.id, body) : await insightGroupsApi.create(body);
      onSaved(saved);
    } catch (exc) {
      notify("error", errorMessage(exc, "Failed to save group"));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal onClose={onClose} title={group ? "Edit group" : "New group"} ariaLabel={group ? "Edit group" : "New group"}>
      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          void save();
        }}
      >
        <label className="block space-y-1">
          <span className="text-sm font-medium">Name</span>
          <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Q3 planning reviews" maxLength={120} autoFocus required />
        </label>
        <label className="block space-y-1">
          <span className="text-sm font-medium">Description</span>
          <Textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={2} placeholder="Optional" />
        </label>
        <div className="space-y-1">
          <span className="text-sm font-medium">Axes to analyze</span>
          <p className="text-xs text-base-content/70">Leave all unselected to analyze every axis.</p>
          <AxisPicker available={available} value={axes} onChange={setAxes} />
        </div>
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" variant="primary" disabled={saving || !name.trim()}>
            {saving ? "Saving…" : group ? "Save" : "Create group"}
          </Button>
        </div>
      </form>
    </Modal>
  );
}

/* ------------------------------------------------------------------ */
/* Meeting picker (multi-select, used for "Add meetings")              */
/* ------------------------------------------------------------------ */

interface PickerMeeting {
  id: string;
  title: string;
  startTime: number;
}

const PICKER_PAGE = 50;

function MeetingPicker({
  excludeIds,
  busy,
  confirmLabel,
  onConfirm,
}: {
  excludeIds: Set<string>;
  busy: boolean;
  confirmLabel: string;
  onConfirm: (ids: string[]) => void;
}): ReactElement {
  const [search, setSearch] = useState("");
  const [debounced, setDebounced] = useState("");
  const [meetings, setMeetings] = useState<PickerMeeting[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const requestId = useRef(0);

  useEffect(() => {
    const t = setTimeout(() => setDebounced(search), 300);
    return () => clearTimeout(t);
  }, [search]);

  const load = useCallback(
    async (offset: number) => {
      const id = ++requestId.current;
      setLoading(true);
      setError(null);
      try {
        const params = new URLSearchParams({ limit: String(PICKER_PAGE), offset: String(offset) });
        if (debounced.trim()) params.set("search", debounced.trim());
        const res = await fetch(`/api/meetings?${params.toString()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as {
          meetings?: PickerMeeting[];
          pagination?: { hasMore?: boolean };
        };
        if (id !== requestId.current) return;
        const incoming = data.meetings ?? [];
        setMeetings((prev) => (offset === 0 ? incoming : [...prev, ...incoming.filter((m) => !prev.some((p) => p.id === m.id))]));
        setHasMore(data.pagination?.hasMore ?? incoming.length === PICKER_PAGE);
      } catch (exc) {
        if (id === requestId.current) setError(errorMessage(exc, "Failed to load meetings"));
      } finally {
        if (id === requestId.current) setLoading(false);
      }
    },
    [debounced]
  );

  useEffect(() => {
    void load(0);
  }, [load]);

  const visible = meetings.filter((m) => !excludeIds.has(m.id));
  const toggle = (id: string) => setPicked((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

  return (
    <div className="space-y-3">
      <Input type="search" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search meetings…" aria-label="Search meetings" />
      {error && <div className="rounded-xl border border-error bg-error/10 p-3 text-sm text-error">{error}</div>}
      <ul className="max-h-72 divide-y divide-base-300 overflow-y-auto rounded-xl border border-base-300">
        {visible.map((m) => (
          <li key={m.id}>
            <label className="flex min-h-11 cursor-pointer items-center gap-3 px-3 py-2 hover:bg-base-200">
              <Checkbox checked={picked.includes(m.id)} onChange={() => toggle(m.id)} aria-label={`Select ${m.title}`} />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium">{m.title}</span>
                <span className="block text-xs text-base-content/70">{fmtDate(m.startTime)}</span>
              </span>
            </label>
          </li>
        ))}
        {!loading && visible.length === 0 && <li className="p-4 text-center text-sm text-base-content/70">No meetings to add.</li>}
        {loading && <li className="p-4 text-center text-sm text-base-content/70">Loading meetings…</li>}
      </ul>
      {hasMore && !loading && (
        <Button variant="ghost" size="sm" onClick={() => void load(meetings.length)}>
          Load more
        </Button>
      )}
      <div className="flex items-center justify-end gap-3">
        <span className="text-xs text-base-content/70">{picked.length} selected</span>
        <Button variant="primary" disabled={busy || picked.length === 0} onClick={() => onConfirm(picked)}>
          {busy ? "Adding…" : confirmLabel}
        </Button>
      </div>
    </div>
  );
}

function AddMeetingsModal({ group, onClose, onAdded }: { group: InsightGroup; onClose: () => void; onAdded: () => void }): ReactElement {
  const [busy, setBusy] = useState(false);
  const exclude = useMemo(() => new Set((group.members ?? []).map((m) => m.meetingId)), [group.members]);
  const add = async (ids: string[]) => {
    setBusy(true);
    try {
      await insightGroupsApi.addMembers(group.id, ids);
      notify("success", `Added ${ids.length} meeting${ids.length === 1 ? "" : "s"} to ${group.name}.`);
      onAdded();
    } catch (exc) {
      notify("error", errorMessage(exc, "Failed to add meetings"));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal onClose={onClose} title={`Add meetings to ${group.name}`} ariaLabel="Add meetings to group" maxWidth="max-w-2xl">
      <MeetingPicker excludeIds={exclude} busy={busy} confirmLabel="Add to group" onConfirm={(ids) => void add(ids)} />
    </Modal>
  );
}

/**
 * Pick an existing group or create one, then add the given meetings to it.
 * Opened from the meetings list after multi-selecting meetings.
 */
export function AddToGroupModal({ meetingIds, onClose, onDone }: { meetingIds: string[]; onClose: () => void; onDone: () => void }): ReactElement {
  const [groups, setGroups] = useState<InsightGroup[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [groupId, setGroupId] = useState("");
  const [newName, setNewName] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    insightGroupsApi
      .list()
      .then((g) => {
        if (cancelled) return;
        setGroups(g);
        setGroupId(g[0]?.id ?? "");
      })
      .catch((exc) => {
        if (cancelled) return;
        setGroups([]);
        setError(exc instanceof InsightsApiError && exc.disabled ? "Insights is turned off on this server." : errorMessage(exc, "Failed to load groups"));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const creating = newName.trim().length > 0;
  const submit = async () => {
    setBusy(true);
    try {
      const target = creating ? await insightGroupsApi.create({ name: newName.trim() }) : groups?.find((g) => g.id === groupId);
      if (!target) return;
      await insightGroupsApi.addMembers(target.id, meetingIds);
      notify("success", `Added ${meetingIds.length} meeting${meetingIds.length === 1 ? "" : "s"} to ${target.name}.`);
      onDone();
    } catch (exc) {
      notify("error", errorMessage(exc, "Failed to add to group"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal onClose={onClose} title="Add to group" ariaLabel="Add selected meetings to a group">
      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <p className="text-sm text-base-content/70">
          {meetingIds.length} meeting{meetingIds.length === 1 ? "" : "s"} selected. A meeting can belong to many groups.
        </p>
        {error && <div className="rounded-xl border border-error bg-error/10 p-3 text-sm text-error">{error}</div>}
        {groups === null ? (
          <p className="text-sm text-base-content/70">Loading groups…</p>
        ) : (
          <>
            {groups.length > 0 && (
              <label className="block space-y-1">
                <span className="text-sm font-medium">Existing group</span>
                <select className="select select-md w-full" value={groupId} disabled={creating} onChange={(e) => setGroupId(e.target.value)}>
                  {groups.map((g) => (
                    <option key={g.id} value={g.id}>
                      {g.name} ({g.memberCount})
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label className="block space-y-1">
              <span className="text-sm font-medium">{groups.length > 0 ? "Or create a new group" : "New group name"}</span>
              <Input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Group name" maxLength={120} />
            </label>
          </>
        )}
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" variant="primary" disabled={busy || groups === null || (!creating && !groupId)}>
            {busy ? "Adding…" : creating ? "Create and add" : "Add to group"}
          </Button>
        </div>
      </form>
    </Modal>
  );
}

/* ------------------------------------------------------------------ */
/* Aggregate                                                           */
/* ------------------------------------------------------------------ */

function AxisBars({ axes }: { axes: Record<string, number> }): ReactElement {
  const entries = Object.entries(axes);
  if (entries.length === 0) return <p className="text-sm text-base-content/70">No scored axes yet.</p>;
  const scale = Math.max(1, ...entries.map(([, v]) => Math.abs(v)));
  return (
    <ul className="space-y-2">
      {entries.map(([axis, value]) => (
        <li key={axis} className="grid grid-cols-[minmax(0,8rem)_1fr_3rem] items-center gap-3 text-sm">
          <span className="truncate capitalize" title={fmtDimension(axis)}>
            {fmtDimension(axis)}
          </span>
          <span
            className="relative h-2.5 rounded-full bg-base-300"
            role="img"
            aria-label={`${fmtDimension(axis)} ${fmtNum(value)}`}
          >
            <span
              className={`absolute inset-y-0 rounded-full ${value < 0 ? "bg-warning" : "bg-primary"}`}
              style={{ width: `${(Math.abs(value) / scale) * 100}%`, ...(value < 0 ? { right: 0 } : { left: 0 }) }}
            />
          </span>
          <span className="text-right font-mono text-xs">{fmtNum(value)}</span>
        </li>
      ))}
    </ul>
  );
}

function AggregatePanel({ aggregate }: { aggregate: InsightGroupAggregate }): ReactElement {
  const { coverage, moments } = aggregate;
  const unanalyzed = coverage.totalMembers - coverage.analyzedMembers;
  return (
    <Card>
      <CardBody className="space-y-3">
        <SectionHeader
          title="Group scores"
          hint={`Averaged over ${coverage.analyzedMembers} of ${coverage.totalMembers} meetings that have been analyzed.`}
          right={<InferredBadge />}
        />
        {unanalyzed > 0 && (
          <div role="alert" className="alert alert-warning alert-soft text-sm">
            <AlertTriangle className="h-4 w-4 shrink-0" aria-hidden="true" />
            <span>
              {unanalyzed} meeting{unanalyzed === 1 ? " is" : "s are"} not analyzed yet and not included. Run a backfill from Insights, then refresh.
            </span>
          </div>
        )}
        <div className="flex flex-wrap gap-2">
          <Badge variant="success" soft>{moments.good} good moments</Badge>
          <Badge variant="error" soft>{moments.bad} bad moments</Badge>
          <Badge variant="ghost">coverage {coverage.analyzedMembers}/{coverage.totalMembers}</Badge>
        </div>
        <AxisBars axes={aggregate.axes} />
      </CardBody>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Trend charts                                                        */
/* ------------------------------------------------------------------ */

interface ChartPoint {
  label: string;
  value: number;
  /** Tooltip detail, e.g. the meeting title or member count. */
  detail: string;
  meetingId: string | null;
  startMs: number | null;
}

function GroupTrendChart({ points, onOpen, label }: { points: ChartPoint[]; onOpen: Open; label: string }): ReactElement {
  const W = 640;
  const H = 150;
  const PAD_X = 28;
  const PAD_Y = 18;
  if (points.length === 0) {
    return <p className="text-sm text-base-content/70">No data for this axis yet.</p>;
  }
  const values = points.map((p) => p.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const x = (i: number) => (points.length === 1 ? W / 2 : PAD_X + (i * (W - PAD_X * 2)) / (points.length - 1));
  const y = (v: number) => H - PAD_Y - ((v - min) / span) * (H - PAD_Y * 2);
  const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ");
  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-40 w-full min-w-[420px] text-primary" role="img" aria-label={label}>
        <line x1={PAD_X} x2={W - PAD_X} y1={H - PAD_Y} y2={H - PAD_Y} className="stroke-base-300" strokeWidth="1" />
        <path d={path} fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
        {points.map((p, i) => {
          const id = p.meetingId;
          return (
            <g
              key={`${p.label}-${i}`}
              role={id ? "button" : undefined}
              tabIndex={id ? 0 : undefined}
              aria-label={`${p.label}: ${fmtNum(p.value)}, ${p.detail}.${id ? " Open meeting." : ""}`}
              className={`outline-none [&:focus-visible>circle]:stroke-base-content ${id ? "cursor-pointer" : ""}`}
              onClick={() => id && onOpen(id, p.startMs)}
              onKeyDown={(e) => {
                if ((e.key === "Enter" || e.key === " ") && id) {
                  e.preventDefault();
                  onOpen(id, p.startMs);
                }
              }}
            >
              <title>{`${p.label} · ${fmtNum(p.value)} · ${p.detail}`}</title>
              <circle cx={x(i)} cy={y(p.value)} r={5} className="fill-primary stroke-base-100" strokeWidth="2" />
              {(i === 0 || i === points.length - 1 || points.length <= 8) && (
                <text x={x(i)} y={H - 3} textAnchor="middle" className="fill-base-content/60 text-[10px]">
                  {p.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function timelinePoints(series: InsightGroupMemberPoint[], axis: string): ChartPoint[] {
  return [...series]
    .sort((a, b) => a.startMs - b.startMs)
    .filter((p) => typeof p.axisScores[axis] === "number")
    .map((p) => ({ label: fmtDate(p.startMs), value: p.axisScores[axis], detail: p.title, meetingId: p.meetingId, startMs: p.startMs }));
}

/** A snapshot is taken per add; it drills to the member most recently added at or before it. */
function snapshotPoints(snapshots: InsightGroupSnapshot[], group: InsightGroup, axis: string): ChartPoint[] {
  const members = [...(group.members ?? [])].sort((a, b) => a.addedAt - b.addedAt);
  return [...snapshots]
    .sort((a, b) => a.createdAt - b.createdAt)
    .filter((s) => typeof s.axes[axis] === "number")
    .map((s) => {
      const anchor = [...members].reverse().find((m) => m.addedAt <= s.createdAt) ?? members[0];
      return {
        label: fmtDate(s.createdAt),
        value: s.axes[axis],
        detail: `${s.memberCount} meetings${s.reason ? ` (${s.reason})` : ""}`,
        meetingId: anchor?.meetingId ?? null,
        startMs: anchor?.startMs ?? null,
      };
    });
}

function TrendsPanel({ group, trends, axes, onOpen }: { group: InsightGroup; trends: InsightGroupTrends; axes: string[]; onOpen: Open }): ReactElement {
  const [axis, setAxis] = useState(axes[0] ?? "");
  const current = axes.includes(axis) ? axis : axes[0] ?? "";
  return (
    <Card>
      <CardBody className="space-y-4">
        <SectionHeader
          title="Trends"
          hint="Each point opens its meeting. The timeline follows meeting dates; the snapshot history shows how the group score moved as meetings were added."
          right={<InferredBadge />}
        />
        {axes.length === 0 ? (
          <p className="text-sm text-base-content/70">Trends appear once the group has analyzed meetings.</p>
        ) : (
          <>
            <div className="flex flex-wrap gap-1.5" role="group" aria-label="Axis">
              {axes.map((a) => (
                <button
                  key={a}
                  type="button"
                  aria-pressed={a === current}
                  onClick={() => setAxis(a)}
                  className={`btn btn-xs min-h-8 capitalize ${a === current ? "btn-primary" : "btn-ghost border-base-300"}`}
                >
                  {fmtDimension(a)}
                </button>
              ))}
            </div>
            <div className="space-y-1">
              <h3 className="text-sm font-semibold">Per meeting, by date</h3>
              <GroupTrendChart points={timelinePoints(trends.memberSeries, current)} onOpen={onOpen} label={`${fmtDimension(current)} per member meeting`} />
            </div>
            <div className="space-y-1">
              <h3 className="text-sm font-semibold">Group score as it grew</h3>
              <GroupTrendChart points={snapshotPoints(trends.snapshots, group, current)} onOpen={onOpen} label={`${fmtDimension(current)} group score by snapshot`} />
            </div>
          </>
        )}
      </CardBody>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Group detail                                                        */
/* ------------------------------------------------------------------ */

function GroupDetail({ groupId, onBack, onOpen, onDeleted }: { groupId: string; onBack: () => void; onOpen: Open; onDeleted: () => void }): ReactElement {
  const [group, setGroup] = useState<InsightGroup | null>(null);
  const [trends, setTrends] = useState<InsightGroupTrends | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [adding, setAdding] = useState(false);
  const [snapshotting, setSnapshotting] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [g, t] = await Promise.all([insightGroupsApi.get(groupId), insightGroupsApi.trends(groupId)]);
      setGroup(g);
      setTrends(t);
    } catch (exc) {
      setError(errorMessage(exc, "Failed to load group"));
    } finally {
      setLoading(false);
    }
  }, [groupId]);

  useEffect(() => {
    setLoading(true);
    void load();
  }, [load]);

  const removeMember = async (meetingId: string, title: string) => {
    try {
      await insightGroupsApi.removeMember(groupId, meetingId);
      notify("success", `Removed ${title} from the group.`);
      await load();
    } catch (exc) {
      notify("error", errorMessage(exc, "Failed to remove meeting"));
    }
  };

  const deleteGroup = async () => {
    if (!group) return;
    if (!(await confirmAction({ title: "Delete group", message: `Delete "${group.name}"? The meetings themselves are not affected.`, confirmLabel: "Delete", danger: true }))) return;
    try {
      await insightGroupsApi.remove(group.id);
      onDeleted();
    } catch (exc) {
      notify("error", errorMessage(exc, "Failed to delete group"));
    }
  };

  const snapshot = async () => {
    setSnapshotting(true);
    try {
      await insightGroupsApi.snapshot(groupId);
      await load();
    } catch (exc) {
      notify("error", errorMessage(exc, "Failed to take snapshot"));
    } finally {
      setSnapshotting(false);
    }
  };

  const axes = useMemo(() => {
    if (!group) return [];
    const configured = group.axes && group.axes.length > 0 ? group.axes : null;
    const scored = new Set<string>(Object.keys(group.aggregate?.axes ?? {}));
    for (const p of trends?.memberSeries ?? []) for (const a of Object.keys(p.axisScores)) scored.add(a);
    return configured ? configured.filter((a) => scored.has(a)) : Array.from(scored);
  }, [group, trends]);

  if (loading) return <p className="text-sm text-base-content/70">Loading group…</p>;
  if (error || !group) {
    return (
      <div className="space-y-3">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ArrowLeft className="h-4 w-4" aria-hidden="true" /> Groups
        </Button>
        <div className="rounded-xl border border-error bg-error/10 p-3 text-sm text-error">{error ?? "Group not found"}</div>
      </div>
    );
  }

  const members = [...(group.members ?? [])].sort((a, b) => b.startMs - a.startMs);

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 space-y-1">
          <Button variant="ghost" size="sm" onClick={onBack} className="-ml-3">
            <ArrowLeft className="h-4 w-4" aria-hidden="true" /> Groups
          </Button>
          <h1 className="break-words text-2xl font-bold tracking-tight text-base-content">{group.name}</h1>
          {group.description && <p className="text-sm text-base-content/70">{group.description}</p>}
          <p className="text-xs text-base-content/70">
            {group.memberCount} meeting{group.memberCount === 1 ? "" : "s"} · analyzing {axesLabel(group.axes)}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="primary" onClick={() => setAdding(true)}>
            <Plus className="h-4 w-4" aria-hidden="true" /> Add meetings
          </Button>
          <Button onClick={() => void snapshot()} disabled={snapshotting} title="Record the current group scores as a snapshot">
            <RefreshCw className="h-4 w-4" aria-hidden="true" /> Snapshot
          </Button>
          <Button onClick={() => setEditing(true)} aria-label="Edit group">
            <Pencil className="h-4 w-4" aria-hidden="true" /> Edit
          </Button>
          <Button variant="dangerSoft" onClick={() => void deleteGroup()} aria-label="Delete group">
            <Trash2 className="h-4 w-4" aria-hidden="true" /> Delete
          </Button>
        </div>
      </div>

      {group.memberCount === 0 ? (
        <div className="rounded-2xl border border-dashed border-base-300 bg-base-200/40 py-12 text-center text-sm text-base-content/70">
          No meetings in this group yet. Use Add meetings to pick some.
        </div>
      ) : (
        <>
          {group.aggregate && <AggregatePanel aggregate={group.aggregate} />}
          {trends && <TrendsPanel group={group} trends={trends} axes={axes} onOpen={onOpen} />}
        </>
      )}

      <Card>
        <CardBody className="space-y-3">
          <SectionHeader title="Meetings" hint="Removing a meeting re-aggregates the group." />
          <ul className="divide-y divide-base-300">
            {members.map((m) => (
              <li key={m.meetingId} className="flex items-center gap-2 py-1">
                <button type="button" onClick={() => onOpen(m.meetingId)} className="min-h-11 min-w-0 flex-1 text-left hover:text-primary">
                  <span className="block truncate text-sm font-medium">{m.title}</span>
                  <span className="block text-xs text-base-content/70">{fmtDate(m.startMs)}</span>
                </button>
                <button
                  type="button"
                  onClick={() => void removeMember(m.meetingId, m.title)}
                  className="btn btn-ghost btn-circle min-h-11 min-w-11 text-base-content/50 hover:bg-error/10 hover:text-error"
                  aria-label={`Remove ${m.title} from group`}
                  title="Remove from group"
                >
                  <X className="h-4 w-4" aria-hidden="true" />
                </button>
              </li>
            ))}
          </ul>
        </CardBody>
      </Card>

      {editing && (
        <GroupFormModal
          group={group}
          onClose={() => setEditing(false)}
          onSaved={() => {
            setEditing(false);
            void load();
          }}
        />
      )}
      {adding && (
        <AddMeetingsModal
          group={group}
          onClose={() => setAdding(false)}
          onAdded={() => {
            setAdding(false);
            void load();
          }}
        />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Groups list                                                         */
/* ------------------------------------------------------------------ */

export function InsightGroupsView({ onNavigate }: { onNavigate: (path: string) => void }): ReactElement {
  const [groups, setGroups] = useState<InsightGroup[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [disabled, setDisabled] = useState(false);
  const [creating, setCreating] = useState(false);

  const open: Open = useCallback((meetingId, startMs) => onNavigate(meetingHref(meetingId, startMs)), [onNavigate]);

  const loadGroups = useCallback(async () => {
    setError(null);
    try {
      setGroups(await insightGroupsApi.list());
      setDisabled(false);
    } catch (exc) {
      if (exc instanceof InsightsApiError && exc.disabled) setDisabled(true);
      else setError(errorMessage(exc, "Failed to load groups"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadGroups();
  }, [loadGroups]);

  if (selectedId) {
    return (
      <GroupDetail
        groupId={selectedId}
        onBack={() => {
          setSelectedId(null);
          void loadGroups();
        }}
        onOpen={open}
        onDeleted={() => {
          setSelectedId(null);
          void loadGroups();
        }}
      />
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-base-content">Groups</h1>
          <p className="text-sm text-base-content/70">
            Hand-picked sets of meetings, scored together. A meeting can be in many groups, and each group shows how its scores moved as it grew.
          </p>
        </div>
        {!disabled && (
          <Button variant="primary" onClick={() => setCreating(true)}>
            <Plus className="h-4 w-4" aria-hidden="true" /> New group
          </Button>
        )}
      </div>

      {disabled ? (
        <div role="alert" className="alert alert-info alert-soft text-sm">
          Insights is turned off on this server. Enable it in the backend settings, then reload.
        </div>
      ) : (
        <>
          {error && <div className="rounded-xl border border-error bg-error/10 p-3 text-sm text-error">{error}</div>}
          {loading ? (
            <p className="text-sm text-base-content/70">Loading groups…</p>
          ) : groups.length === 0 && !error ? (
            <div className="rounded-2xl border border-dashed border-base-300 bg-base-200/40 py-16 text-center">
              <p className="text-base-content/70">No groups yet. Create one, or select meetings in the Meetings tab and choose Add to group.</p>
            </div>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2">
              {groups.map((g) => (
                <button
                  key={g.id}
                  type="button"
                  onClick={() => setSelectedId(g.id)}
                  className="group flex h-full flex-col gap-2 rounded-xl border border-base-300 bg-base-200/90 p-4 text-left shadow-sm transition hover:border-primary/60 hover:bg-base-200"
                >
                  <span className="flex items-center gap-2">
                    <Users className="h-4 w-4 shrink-0 text-base-content/50" aria-hidden="true" />
                    <span className="min-w-0 flex-1 truncate text-lg font-semibold group-hover:text-primary">{g.name}</span>
                  </span>
                  {g.description && <span className="line-clamp-2 text-sm text-base-content/70">{g.description}</span>}
                  <span className="mt-auto flex flex-wrap items-center gap-1.5 text-xs text-base-content/70">
                    <Badge variant="ghost">{g.memberCount} meeting{g.memberCount === 1 ? "" : "s"}</Badge>
                    <span className="capitalize">{axesLabel(g.axes)}</span>
                  </span>
                </button>
              ))}
            </div>
          )}
          {groups.length > 0 && <InsightGroupCompareSection groups={groups} onOpen={open} />}
        </>
      )}

      {creating && (
        <GroupFormModal
          group={null}
          onClose={() => setCreating(false)}
          onSaved={(g) => {
            setCreating(false);
            setSelectedId(g.id);
          }}
        />
      )}
    </div>
  );
}
