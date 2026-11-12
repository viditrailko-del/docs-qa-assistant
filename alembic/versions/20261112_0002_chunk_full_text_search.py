"""chunks.search_tsv: generated full-text search column + GIN index

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-12

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SEARCH_TSV_SQL = (
    "setweight(to_tsvector('english', coalesce(heading_path, '')), 'B') || "
    "setweight(to_tsvector('english', coalesce(text, '')), 'A')"
)


def upgrade() -> None:
    # GENERATED ... STORED: Postgres fills it for existing rows too, so no backfill script needed.
    op.add_column(
        "chunks",
        sa.Column("search_tsv", postgresql.TSVECTOR(), sa.Computed(SEARCH_TSV_SQL, persisted=True), nullable=True),
    )
    op.create_index("ix_chunks_search_tsv", "chunks", ["search_tsv"], unique=False, postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("ix_chunks_search_tsv", table_name="chunks", postgresql_using="gin")
    op.drop_column("chunks", "search_tsv")
