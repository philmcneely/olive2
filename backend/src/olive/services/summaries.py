"""Meeting summary generation and artifact management service (port of summaries/service.ts)."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..db import DbHandle, get_db
from ..layout import meeting_paths
from ..logging_db import logger
from ..schemas import LlmGenerateResult, SummaryGenerationStatus, Template
from ..settings import resolve_paths
from ..speaker_labels import is_generic_speaker_label
from ..workflows import QUEUE_SUMMARIZE, enqueue_stage_job, ensure_dbos_launched_async
from .llm import LlmService
from .search import safe_reindex
from .templates import TemplateService

SUMMARY_STATUS_TTL_S = 60.0


async def _call_llm_generate(llm_service: Any, options: dict[str, Any]) -> LlmGenerateResult:
    # Check instance attribute overrides first (e.g. in tests)
    inst_dict = getattr(llm_service, "__dict__", {})
    fn = (
        inst_dict.get("generate_text")
        or inst_dict.get("generateText")
        or getattr(llm_service, "generate_text", None)
        or getattr(llm_service, "generateText", None)
    )
    if not callable(fn):
        raise AttributeError("llm_service has no generate_text/generateText method")
    res = fn(options)
    if inspect.isawaitable(res):
        res = await res
    return res if isinstance(res, LlmGenerateResult) else LlmGenerateResult.model_validate(res)


class SummaryService:
    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        db: Engine | DbHandle | None = None,
        meetings_dir: str | None = None,
        llm_service: LlmService | Any | None = None,
        template_service: TemplateService | None = None,
    ) -> None:
        opts = options or {}
        raw_db = db or opts.get("db")
        if isinstance(raw_db, DbHandle):
            self.db: Engine = raw_db.engine
        elif isinstance(raw_db, Engine):
            self.db = raw_db
        else:
            self.db = get_db()

        self.meetings_dir = str(
            meetings_dir
            or opts.get("meetingsDir")
            or opts.get("meetings_dir")
            or resolve_paths().meetings_dir
        )
        self.llm_service = (
            llm_service or opts.get("llmService") or opts.get("llm_service") or LlmService(db=self.db)
        )
        self.template_service = (
            template_service
            or opts.get("templateService")
            or opts.get("template_service")
            or TemplateService(self.db)
        )
        self._generation_statuses: dict[str, SummaryGenerationStatus] = {}

    def get_summary_generation_status(self, meeting_id: str) -> SummaryGenerationStatus | None:
        status = self._generation_statuses.get(meeting_id)
        return status.model_copy() if status else None

    getSummaryGenerationStatus = get_summary_generation_status

    async def start_summary_generation(self, options: dict[str, Any]) -> SummaryGenerationStatus:
        meeting_id = str(options.get("meetingId") or options.get("meeting_id") or "")
        existing = self._generation_statuses.get(meeting_id)
        if existing and existing.stage == "running":
            raise RuntimeError("Summary generation is already in progress for this meeting")

        # Validate upfront before backgrounding
        await self._load_meeting_with_transcript(meeting_id)

        status = SummaryGenerationStatus(
            meetingId=meeting_id,
            stage="running",
            startedAt=int(time.time() * 1000),
            provider=options.get("provider"),
            model=options.get("model"),
        )
        self._generation_statuses[meeting_id] = status

        async def _run_bg() -> None:
            try:
                res = await self.generate_summary(options)
                self._generation_statuses[meeting_id] = SummaryGenerationStatus(
                    meetingId=meeting_id,
                    stage="done",
                    startedAt=status.startedAt,
                    provider=res["provider"],
                    model=res["model"],
                    templateName=res["templateName"],
                    artifactId=res["artifactId"],
                )
            except Exception as exc:
                msg = str(exc)
                logger.error(
                    "Background summary generation failed",
                    {"meetingId": meeting_id, "error": msg},
                )
                self._generation_statuses[meeting_id] = SummaryGenerationStatus(
                    meetingId=meeting_id,
                    stage="error",
                    startedAt=status.startedAt,
                    provider=status.provider,
                    model=status.model,
                    error=msg,
                )
            finally:
                self._prune_status_later(meeting_id)

        async def _enqueue_summary() -> None:
            try:
                await ensure_dbos_launched_async(meetings_dir=self.meetings_dir)
                await enqueue_stage_job(
                    QUEUE_SUMMARIZE,
                    _run_bg,
                    deduplication_id=f"summary:{meeting_id}:{status.startedAt}",
                )
            except Exception:
                await _run_bg()

        asyncio.create_task(_enqueue_summary())
        return status.model_copy()

    startSummaryGeneration = start_summary_generation

    def _prune_status_later(self, meeting_id: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        def _cleanup() -> None:
            cur = self._generation_statuses.get(meeting_id)
            if cur and cur.stage != "running":
                self._generation_statuses.pop(meeting_id, None)

        loop.call_later(SUMMARY_STATUS_TTL_S, _cleanup)

    async def _load_meeting_with_transcript(
        self, meeting_id: str
    ) -> tuple[Mapping[Any, Any], str, Mapping[Any, Any]]:
        with self.db.connect() as conn:
            meeting = (
                conn.execute(
                    text("SELECT * FROM meetings WHERE id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")

            artifacts = (
                conn.execute(
                    text("SELECT * FROM artifacts WHERE meeting_id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .all()
            )

        folder = meeting_paths(
            self.meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"]
        ).folder

        primary_tid = meeting.get("primary_transcript_artifact_id")
        transcript_artifact = next((a for a in artifacts if a["id"] == primary_tid), None)
        if not transcript_artifact:
            transcript_artifact = next((a for a in artifacts if a["kind"] == "transcript"), None)

        if not transcript_artifact:
            raise ValueError("No transcript found for meeting. Please transcribe the recording first.")

        full_path = Path(folder) / transcript_artifact["path"]
        if not full_path.exists():
            raise ValueError(f"Transcript artifact not found on disk at: {transcript_artifact['path']}")

        return meeting, folder, transcript_artifact

    async def generate_summary(self, options: dict[str, Any]) -> dict[str, Any]:
        meeting_id = str(options.get("meetingId") or options.get("meeting_id") or "")
        meeting, folder, transcript_artifact = await self._load_meeting_with_transcript(meeting_id)

        transcript_full_path = Path(folder) / transcript_artifact["path"]
        raw_transcript = transcript_full_path.read_text(encoding="utf-8")

        if transcript_artifact["format"] == "json":
            try:
                parsed = json.loads(raw_transcript)
                segs = parsed.get("segments") if isinstance(parsed, dict) else None
                if isinstance(segs, list):
                    formatted_dialogue = "\n\n".join(
                        f"{seg.get('speaker') or 'Unknown'}: {seg.get('text', '')}"
                        for seg in segs
                        if isinstance(seg, dict)
                    )
                else:
                    formatted_dialogue = raw_transcript
            except Exception:
                formatted_dialogue = raw_transcript
        else:
            formatted_dialogue = raw_transcript

        # 2. Locate template
        template: Template | None = None
        template_id = options.get("templateId") or options.get("template_id")
        if template_id:
            template = await self.template_service.get_template(str(template_id))
        if not template:
            template = await self.template_service.get_default_template()
        if not template:
            all_tpls = await self.template_service.list_templates()
            template = all_tpls[0] if all_tpls else None
        if not template:
            raise RuntimeError("No summary template available")

        # 3. Fetch meeting speakers
        with self.db.connect() as conn:
            speaker_rows = (
                conn.execute(
                    text(
                        """
                    SELECT speakers.name
                    FROM meeting_speakers
                    INNER JOIN speakers ON speakers.id = meeting_speakers.speaker_id
                    WHERE meeting_speakers.meeting_id = :mid
                    """
                    ),
                    {"mid": meeting_id},
                )
                .mappings()
                .all()
            )

        # Generic labels are not people; don't present them as participants.
        speaker_names = (
            ", ".join(r["name"] for r in speaker_rows if not is_generic_speaker_label(r["name"]))
            or "Participants"
        )
        date_formatted = datetime.fromtimestamp(int(meeting["start_time"]) / 1000.0).strftime(
            "%b %d, %Y, %I:%M %p"
        )

        def replace_vars(s: str) -> str:
            return (
                s.replace("{{title}}", meeting["title"])
                .replace("{{date}}", date_formatted)
                .replace("{{speakers}}", speaker_names)
                .replace("{{transcript}}", formatted_dialogue)
            )

        system_prompt = replace_vars(template.systemPrompt or "")
        user_prompt = replace_vars(template.userPrompt)

        # 5. Stage run
        stage_run_id = str(uuid.uuid4())
        now = int(time.time() * 1000)

        with self.db.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO stage_runs (
                      id, meeting_id, stage, status, attempts, started_at, created_at, updated_at
                    ) VALUES (
                      :id, :mid, 'summary_generate', 'running', 1, :now, :now, :now
                    )
                    ON CONFLICT(meeting_id, stage) DO UPDATE SET
                      id = :id,
                      status = 'running',
                      attempts = stage_runs.attempts + 1,
                      started_at = :now,
                      finished_at = NULL,
                      last_error = NULL,
                      updated_at = :now
                    """
                ),
                {"id": stage_run_id, "mid": meeting["id"], "now": now},
            )

        logger.debug(
            "Prompt rendered for summary generation",
            {
                "category": "summary",
                "meetingId": meeting["id"],
                "template": template.name,
                "promptLength": len(user_prompt),
            },
        )

        try:
            llm_result = await _call_llm_generate(
                self.llm_service,
                {
                    "provider": options.get("provider"),
                    "model": options.get("model"),
                    "thinkingLevel": options.get("thinkingLevel") or options.get("thinking_level"),
                    "systemPrompt": system_prompt or None,
                    "prompt": user_prompt,
                },
            )
        except Exception as exc:
            err_msg = str(exc)
            fin_now = int(time.time() * 1000)
            with self.db.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE stage_runs
                        SET status = 'error', last_error = :err, finished_at = :fin, updated_at = :fin
                        WHERE id = :id
                        """
                    ),
                    {"err": err_msg, "fin": fin_now, "id": stage_run_id},
                )
            raise

        # 6. Save summary markdown file to disk
        summaries_dir = Path(folder) / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        artifact_id = str(uuid.uuid4())
        filename = f"summary_{now}_{artifact_id[:8]}.md"
        relative_path = f"summaries/{filename}"
        full_path = summaries_dir / filename
        full_path.write_text(llm_result.text, encoding="utf-8")

        provider_label = f"{llm_result.provider}:{llm_result.model}:{template.name}"
        set_primary_opt = options.get("setPrimary") if "setPrimary" in options else options.get("set_primary")
        should_set_primary = set_primary_opt is not False and (
            not meeting.get("primary_summary_artifact_id") or set_primary_opt is True
        )
        fin_now = int(time.time() * 1000)

        with self.db.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO artifacts (
                      id, meeting_id, recording_id, kind, provider, format, path, created_at
                    ) VALUES (
                      :id, :mid, NULL, 'summary', :provider, 'md', :path, :now
                    )
                    """
                ),
                {
                    "id": artifact_id,
                    "mid": meeting["id"],
                    "provider": provider_label,
                    "path": relative_path,
                    "now": now,
                },
            )
            conn.execute(
                text(
                    "UPDATE stage_runs SET status = 'done', finished_at = :fin, updated_at = :fin WHERE id = :id"
                ),
                {"fin": fin_now, "id": stage_run_id},
            )
            if should_set_primary:
                conn.execute(
                    text(
                        "UPDATE meetings SET primary_summary_artifact_id = :aid, updated_at = :fin WHERE id = :mid"
                    ),
                    {"aid": artifact_id, "fin": fin_now, "mid": meeting["id"]},
                )

        safe_reindex(self.db, meeting["id"], self.meetings_dir)
        logger.info(
            "Generated summary for meeting",
            {
                "meetingId": meeting["id"],
                "artifactId": artifact_id,
                "template": template.name,
                "provider": llm_result.provider,
                "model": llm_result.model,
                "durationMs": llm_result.durationMs,
            },
        )

        try:
            from .usage import record_llm_usage

            record_llm_usage(
                self.db,
                meeting_id=meeting["id"],
                kind="summary",
                provider=llm_result.provider,
                model=llm_result.model,
                input_tokens=llm_result.usage.inputTokens if llm_result.usage else 0,
                output_tokens=llm_result.usage.outputTokens if llm_result.usage else 0,
                total_tokens=llm_result.usage.totalTokens if llm_result.usage else 0,
                created_at=fin_now,
                source=f"summary:{artifact_id}",
                detail={"artifactId": artifact_id, "template": template.name},
            )
        except Exception:
            pass

        return {
            "artifactId": artifact_id,
            "provider": llm_result.provider,
            "model": llm_result.model,
            "templateName": template.name,
            "content": llm_result.text,
            "isPrimary": should_set_primary,
            "durationMs": llm_result.durationMs,
            "usage": llm_result.usage.model_dump(exclude_none=True),
        }

    generateSummary = generate_summary

    async def delete_summary(self, meeting_id: str, artifact_id: str) -> None:
        with self.db.connect() as conn:
            meeting = (
                conn.execute(
                    text("SELECT * FROM meetings WHERE id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")

            artifact = (
                conn.execute(
                    text(
                        "SELECT * FROM artifacts WHERE id = :aid AND meeting_id = :mid AND kind = 'summary'"
                    ),
                    {"aid": artifact_id, "mid": meeting_id},
                )
                .mappings()
                .first()
            )
            if not artifact:
                raise ValueError(f"Summary artifact not found: {artifact_id}")

        folder = Path(
            meeting_paths(
                self.meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"]
            ).folder
        )
        full_path = folder / artifact["path"]
        try:
            full_path.unlink(missing_ok=True)
        except Exception:
            pass

        now = int(time.time() * 1000)
        with self.db.begin() as conn:
            if meeting.get("primary_summary_artifact_id") == artifact_id:
                rem = (
                    conn.execute(
                        text(
                            """
                        SELECT id FROM artifacts
                        WHERE meeting_id = :mid AND kind = 'summary' AND id != :aid
                        ORDER BY created_at DESC LIMIT 1
                        """
                        ),
                        {"mid": meeting_id, "aid": artifact_id},
                    )
                    .mappings()
                    .first()
                )
                conn.execute(
                    text(
                        "UPDATE meetings SET primary_summary_artifact_id = :next_id, updated_at = :now "
                        "WHERE id = :mid"
                    ),
                    {"next_id": rem["id"] if rem else None, "now": now, "mid": meeting_id},
                )
            conn.execute(text("DELETE FROM artifacts WHERE id = :aid"), {"aid": artifact_id})

        safe_reindex(self.db, meeting_id, self.meetings_dir)
        logger.info("Deleted summary artifact", {"meetingId": meeting_id, "artifactId": artifact_id})

    deleteSummary = delete_summary

    async def set_primary_summary(self, meeting_id: str, artifact_id: str) -> None:
        with self.db.connect() as conn:
            meeting = (
                conn.execute(
                    text("SELECT * FROM meetings WHERE id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")

            artifact = (
                conn.execute(
                    text(
                        "SELECT * FROM artifacts WHERE id = :aid AND meeting_id = :mid AND kind = 'summary'"
                    ),
                    {"aid": artifact_id, "mid": meeting_id},
                )
                .mappings()
                .first()
            )
            if not artifact:
                raise ValueError(f"Summary artifact not found: {artifact_id}")

        now = int(time.time() * 1000)
        with self.db.begin() as conn:
            conn.execute(
                text(
                    "UPDATE meetings SET primary_summary_artifact_id = :aid, updated_at = :now WHERE id = :mid"
                ),
                {"aid": artifact_id, "now": now, "mid": meeting_id},
            )

        safe_reindex(self.db, meeting_id, self.meetings_dir)
        logger.info("Updated primary summary artifact", {"meetingId": meeting_id, "artifactId": artifact_id})

    setPrimarySummary = set_primary_summary
