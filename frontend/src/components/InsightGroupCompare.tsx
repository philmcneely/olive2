import { useEffect, useState } from "react";
import type { ReactElement } from "react";
import { Badge, Button, Card, CardBody, Checkbox } from "./ui";
import { insightGroupsApi, type InsightGroup, type InsightGroupCompare } from "../api/insightGroups";

type Open = (meetingId: string, startMs?: number | null) => void;

/** One bar per compared series. The overall corpus is neutral; groups cycle through theme colors. */
const GROUP_FILLS = ["fill-primary", "fill-secondary", "fill-accent", "fill-info", "fill-success", "fill-warning"];
const GROUP_SWATCH = ["bg-primary", "bg-secondary", "bg-accent", "bg-info", "bg-success", "bg-warning"];
const OVERALL_FILL = "fill-base-content/40";
const OVERALL_ID = "__overall__";

interface Series {
  id: string;
  label: string;
  fill: string;
  swatch: string;
  scores: Record<string, number>;
  meetingIds: string[];
}

function fmtNum(n: number): string {
  return (Math.round(n * 100) / 100).toFixed(2);
}

function fmtDimension(name: string): string {
  return name.replace(/[_-]+/g, " ");
}

function buildSeries(data: InsightGroupCompare): Series[] {
  const out: Series[] = data.groups.map((g, i) => ({
    id: g.id,
    label: g.name,
    fill: GROUP_FILLS[i % GROUP_FILLS.length],
    swatch: GROUP_SWATCH[i % GROUP_SWATCH.length],
    scores: g.scores,
    meetingIds: g.meetingIds,
  }));
  if (data.default) {
    out.unshift({
      id: OVERALL_ID,
      label: "All meetings",
      fill: OVERALL_FILL,
      swatch: "bg-base-content/40",
      scores: data.default.scores,
      meetingIds: data.default.meetingIds,
    });
  }
  return out;
}

/** Grouped bars: one cluster per axis, one bar per series. Clicking a bar selects its meeting set. */
function CompareChart({
  axes,
  series,
  selected,
  onSelect,
}: {
  axes: string[];
  series: Series[];
  selected: { seriesId: string; axis: string } | null;
  onSelect: (seriesId: string, axis: string) => void;
}): ReactElement {
  const CLUSTER_W = 28 + series.length * 26;
  const H = 180;
  const PAD_TOP = 14;
  const PAD_BOTTOM = 34;
  const values = series.flatMap((s) => axes.map((a) => s.scores[a]).filter((v): v is number => typeof v === "number"));
  const max = Math.max(1, ...values);
  const min = Math.min(0, ...values);
  const span = max - min || 1;
  const plotH = H - PAD_TOP - PAD_BOTTOM;
  const y = (v: number) => PAD_TOP + ((max - v) / span) * plotH;
  const zero = y(0);
  const W = Math.max(320, axes.length * CLUSTER_W + 16);
  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} style={{ minWidth: W }} className="h-52 w-full" role="img" aria-label="Per-axis scores by group">
        <line x1={8} x2={W - 8} y1={zero} y2={zero} className="stroke-base-300" strokeWidth="1" />
        {axes.map((axis, ai) => {
          const x0 = 8 + ai * CLUSTER_W + 14;
          return (
            <g key={axis}>
              {series.map((s, si) => {
                const v = s.scores[axis];
                const bx = x0 + si * 26;
                if (typeof v !== "number") {
                  return (
                    <text key={s.id} x={bx + 10} y={zero - 3} textAnchor="middle" className="fill-base-content/40 text-[10px]">
                      <title>{`${s.label}: no data for ${fmtDimension(axis)}`}</title>
                      –
                    </text>
                  );
                }
                const top = Math.min(y(v), zero);
                const h = Math.max(1, Math.abs(y(v) - zero));
                const isSel = selected?.seriesId === s.id && selected.axis === axis;
                const clickable = s.meetingIds.length > 0;
                return (
                  <g
                    key={s.id}
                    role={clickable ? "button" : undefined}
                    tabIndex={clickable ? 0 : undefined}
                    aria-label={`${s.label}, ${fmtDimension(axis)}: ${fmtNum(v)}${clickable ? ". Show meetings." : ""}`}
                    aria-pressed={clickable ? isSel : undefined}
                    className={clickable ? "cursor-pointer outline-none" : undefined}
                    onClick={() => clickable && onSelect(s.id, axis)}
                    onKeyDown={(e) => {
                      if (clickable && (e.key === "Enter" || e.key === " ")) {
                        e.preventDefault();
                        onSelect(s.id, axis);
                      }
                    }}
                  >
                    <title>{`${s.label} · ${fmtDimension(axis)} · ${fmtNum(v)}`}</title>
                    <rect
                      x={bx}
                      y={top}
                      width={20}
                      height={h}
                      rx={2}
                      className={`${s.fill} ${isSel ? "stroke-base-content" : "stroke-transparent"}`}
                      strokeWidth="2"
                    />
                    <text x={bx + 10} y={v < 0 ? top + h + 10 : top - 3} textAnchor="middle" className="fill-base-content/70 text-[9px]">
                      {fmtNum(v)}
                    </text>
                  </g>
                );
              })}
              <text
                x={x0 + (series.length * 26 - 6) / 2}
                y={H - 14}
                textAnchor="middle"
                className="fill-base-content/70 text-[10px] capitalize"
              >
                {fmtDimension(axis)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Pick several groups (and optionally the whole corpus) and compare their per-axis scores side by side. */
export function InsightGroupCompareSection({ groups, onOpen }: { groups: InsightGroup[]; onOpen: Open }): ReactElement {
  const [picked, setPicked] = useState<string[]>([]);
  const [includeDefault, setIncludeDefault] = useState(true);
  const [data, setData] = useState<InsightGroupCompare | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<{ seriesId: string; axis: string } | null>(null);

  // Drop picks for groups that no longer exist (deleted in another tab).
  useEffect(() => {
    setPicked((prev) => {
      const next = prev.filter((id) => groups.some((g) => g.id === id));
      return next.length === prev.length ? prev : next;
    });
  }, [groups]);

  useEffect(() => {
    if (picked.length === 0 && !includeDefault) {
      setData(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    setSelected(null);
    insightGroupsApi
      .compare(picked, includeDefault)
      .then((res) => !cancelled && setData(res))
      .catch((exc: unknown) => !cancelled && setError(exc instanceof Error ? exc.message : "Failed to compare groups"))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [picked, includeDefault]);

  const toggle = (id: string) => setPicked((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  const series = data ? buildSeries(data) : [];
  const chosen = selected ? series.find((s) => s.id === selected.seriesId) : null;

  return (
    <Card>
      <CardBody className="space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div className="min-w-0">
            <h2 className="text-lg font-semibold tracking-tight text-base-content">Compare groups</h2>
            <p className="text-xs text-base-content/70">
              Average score per axis for each selected group, against all analyzed meetings. Select a bar to see the meetings behind it.
            </p>
          </div>
          <span title="Scored by a language model from the transcript text. Treat as a lead to check, not a measurement.">
            <Badge variant="outline">model-inferred</Badge>
          </span>
        </div>

        <fieldset className="space-y-2">
          <legend className="sr-only">Groups to compare</legend>
          <div className="flex flex-wrap gap-x-5 gap-y-2">
            {groups.map((g) => (
              <label key={g.id} className="flex cursor-pointer items-center gap-2 text-sm">
                <Checkbox className="checkbox-sm" checked={picked.includes(g.id)} onChange={() => toggle(g.id)} />
                <span className="truncate">{g.name}</span>
                <span className="text-xs text-base-content/60">({g.memberCount})</span>
              </label>
            ))}
          </div>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <Checkbox className="checkbox-sm" checked={includeDefault} onChange={(e) => setIncludeDefault(e.target.checked)} />
            <span>Include overall (all meetings)</span>
          </label>
        </fieldset>

        {error && <div className="rounded-xl border border-error bg-error/10 p-3 text-sm text-error">{error}</div>}

        {picked.length === 0 && !includeDefault ? (
          <p className="text-sm text-base-content/70">Select at least one group to compare.</p>
        ) : loading && !data ? (
          <p className="text-sm text-base-content/70">Comparing…</p>
        ) : data && data.axes.length === 0 ? (
          <p className="text-sm text-base-content/70">No scored axes yet for the selection.</p>
        ) : data ? (
          <>
            <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs" aria-label="Legend">
              {series.map((s) => (
                <li key={s.id} className="flex items-center gap-1.5">
                  <span className={`inline-block h-2.5 w-2.5 rounded-sm ${s.swatch}`} aria-hidden="true" />
                  <span>{s.label}</span>
                </li>
              ))}
            </ul>
            <CompareChart
              axes={data.axes}
              series={series}
              selected={selected}
              onSelect={(seriesId, axis) =>
                setSelected((prev) => (prev?.seriesId === seriesId && prev.axis === axis ? null : { seriesId, axis }))
              }
            />
            {chosen && selected && (
              <div className="space-y-1.5 rounded-xl border border-base-300 bg-base-200/60 p-3">
                <p className="text-sm">
                  <span className="font-semibold">{chosen.label}</span> · <span className="capitalize">{fmtDimension(selected.axis)}</span>{" "}
                  <span className="font-mono">{fmtNum(chosen.scores[selected.axis] ?? 0)}</span> from {chosen.meetingIds.length} analyzed meeting
                  {chosen.meetingIds.length === 1 ? "" : "s"}:
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {chosen.meetingIds.slice(0, 12).map((id) => (
                    <Button key={id} variant="ghost" size="sm" className="font-mono" onClick={() => onOpen(id)}>
                      {id.slice(0, 8)}
                    </Button>
                  ))}
                  {chosen.meetingIds.length > 12 && <span className="text-xs">+{chosen.meetingIds.length - 12} more</span>}
                </div>
              </div>
            )}
          </>
        ) : null}
      </CardBody>
    </Card>
  );
}
