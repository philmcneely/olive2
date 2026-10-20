// Typed fetch helpers for /api/insights/groups/*.
//
// Like insights.ts, the generated schema.d.ts types these routes' 2xx bodies as `{}` (bare JSONResponse),
// so the shapes below mirror the Pydantic models in backend/src/olive/schemas.py ("Insight Groups").
// Aggregation is computed server-side from existing per-meeting signals; no LLM call is made.

import { InsightsApiError, type InsightsMomentCounts } from "./insights";

export interface InsightGroupMember {
  meetingId: string;
  title: string;
  /** Meeting start, epoch ms. */
  startMs: number;
  addedAt: number;
}

/** One member meeting on the group's timeline. `startMs` is the meeting's epoch start. */
export interface InsightGroupMemberPoint {
  meetingId: string;
  title: string;
  startMs: number;
  axisScores: Record<string, number>;
  moments: InsightsMomentCounts;
}

export interface InsightGroupCoverage {
  analyzedMembers: number;
  totalMembers: number;
}

export interface InsightGroupAggregate {
  axes: Record<string, number>;
  moments: InsightsMomentCounts;
  coverage: InsightGroupCoverage;
  series?: InsightGroupMemberPoint[];
}

export interface InsightGroup {
  id: string;
  name: string;
  description?: string | null;
  /** Axis (dimension) keys to analyze; null/empty means all. */
  axes?: string[] | null;
  memberCount: number;
  createdAt: number;
  updatedAt: number;
  members?: InsightGroupMember[] | null;
  aggregate?: InsightGroupAggregate | null;
}

export interface InsightGroupSnapshot {
  createdAt: number;
  memberCount: number;
  axes: Record<string, number>;
  reason?: string | null;
}

export interface InsightGroupTrends {
  snapshots: InsightGroupSnapshot[];
  memberSeries: InsightGroupMemberPoint[];
}

/** Result of a membership change: the new aggregate and the snapshot taken (null if nothing changed). */
export interface InsightGroupChange {
  aggregate: InsightGroupAggregate;
  snapshot?: InsightGroupSnapshot | null;
}

export interface InsightGroupInput {
  name: string;
  description?: string | null;
  axes?: string[] | null;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const data = (await res.json().catch(() => ({}))) as { error?: string };
    throw new InsightsApiError(data.error || `Request failed (${res.status})`, res.status);
  }
  return (await res.json()) as T;
}

function json(method: string, body?: unknown): RequestInit {
  return { method, headers: { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) };
}

const base = "/api/insights/groups";
const gid = (id: string) => `${base}/${encodeURIComponent(id)}`;

export const insightGroupsApi = {
  list: () => request<{ groups: InsightGroup[] }>(base).then((r) => r.groups),
  create: (body: InsightGroupInput) => request<InsightGroup>(base, json("POST", body)),
  get: (id: string) => request<InsightGroup>(gid(id)),
  update: (id: string, body: Partial<InsightGroupInput>) => request<InsightGroup>(gid(id), json("PATCH", body)),
  remove: (id: string) => request<{ deleted: boolean }>(gid(id), { method: "DELETE" }),
  addMembers: (id: string, meetingIds: string[]) => request<InsightGroupChange>(`${gid(id)}/members`, json("POST", { meetingIds })),
  removeMember: (id: string, meetingId: string) =>
    request<InsightGroupChange>(`${gid(id)}/members/${encodeURIComponent(meetingId)}`, { method: "DELETE" }),
  trends: (id: string) => request<InsightGroupTrends>(`${gid(id)}/trends`),
  snapshot: (id: string) => request<InsightGroupChange>(`${gid(id)}/snapshot`, json("POST")),
};
