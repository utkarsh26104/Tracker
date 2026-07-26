"""A small history/archive table for published reports - separate from
LangGraph's own checkpoints, which persist per-thread execution state and
aren't meant to be queried as a report archive. Only reports that actually
get published (approved) land here."""

from datetime import datetime, timedelta, timezone

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS report_history (
    id SERIAL PRIMARY KEY,
    job_id TEXT NOT NULL,
    company TEXT,
    report_type TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# report_type's allowed values grew (added 'strategy') after this table was
# already live - CREATE TABLE IF NOT EXISTS doesn't retroactively update an
# existing table's constraint, so migrate it explicitly and idempotently
# rather than assuming a fresh table.
_DROP_OLD_CONSTRAINT_SQL = "ALTER TABLE report_history DROP CONSTRAINT IF EXISTS report_history_report_type_check;"

_ADD_CONSTRAINT_SQL = """
ALTER TABLE report_history ADD CONSTRAINT report_history_report_type_check
    CHECK (report_type IN ('company', 'comparison_matrix', 'strategy'));
"""

_INSERT_SQL = """
INSERT INTO report_history (job_id, company, report_type, content, created_at)
VALUES (%s, %s, %s, %s, %s)
"""

_LIST_SQL = """
SELECT job_id, company, report_type, content, created_at
FROM report_history
ORDER BY created_at DESC
LIMIT %s
"""

_RECENT_COMPANY_REPORT_SQL = """
SELECT content
FROM report_history
WHERE company = %s AND report_type = 'company' AND created_at > %s
ORDER BY created_at DESC
LIMIT 1
"""


async def init_history_table(pool) -> None:
    async with pool.connection() as conn:
        await conn.execute(_CREATE_TABLE_SQL)
        await conn.execute(_DROP_OLD_CONSTRAINT_SQL)
        await conn.execute(_ADD_CONSTRAINT_SQL)


async def save_report(pool, job_id: str, company: str | None, report_type: str, content: str) -> None:
    async with pool.connection() as conn:
        await conn.execute(_INSERT_SQL, (job_id, company, report_type, content, datetime.now(timezone.utc)))


async def list_history(pool, limit: int = 50) -> list[dict]:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(_LIST_SQL, (limit,))
        rows = await cur.fetchall()
    # The pool's connections default to row_factory=dict_row (see
    # app/db/checkpointer.py), so rows already come back as dicts.
    return [{**row, "created_at": row["created_at"].isoformat()} for row in rows]


async def get_recent_company_report(pool, company: str, max_age_days: int) -> str | None:
    """Most recent published report for this exact company, if one exists
    within max_age_days - lets callers skip re-researching a company whose
    data is still fresh enough rather than always hitting the web."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(_RECENT_COMPANY_REPORT_SQL, (company, cutoff))
        row = await cur.fetchone()
    return row["content"] if row else None
