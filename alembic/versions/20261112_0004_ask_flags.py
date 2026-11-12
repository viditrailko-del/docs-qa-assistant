"""asks.flags: verification flags from app/checks

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-12

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # NOT NULL on a table with rows: the server default fills existing rows with [].
    # Old answers can be re-checked with `python -m scripts.recheck_asks` (checks are pure).
    op.add_column(
        "asks",
        sa.Column("flags", sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
                  server_default=sa.text("'[]'"), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("asks", "flags")
