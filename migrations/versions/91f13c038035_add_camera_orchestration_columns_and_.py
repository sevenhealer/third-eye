"""add camera orchestration columns and process status table

Revision ID: 91f13c038035
Revises: 
Create Date: 2026-06-23 13:08:31.354040

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = '91f13c038035'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("cameras", sa.Column("gpu_id", sa.Integer(), nullable=True))
    op.add_column(
        "cameras",
        sa.Column(
            "desired_state", sa.String(length=20), nullable=False, server_default="stopped"
        ),
    )
    op.create_check_constraint(
        "cameras_desired_state_check",
        "cameras",
        "desired_state IN ('running', 'stopped')",
    )
    op.add_column(
        "cameras",
        sa.Column("config_version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "cameras",
        sa.Column("launch_args", JSONB(), nullable=True, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column(
        "cameras",
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    op.create_table(
        "camera_process_status",
        sa.Column("camera_id", sa.String(length=100), primary_key=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("gpu_id", sa.Integer(), nullable=True),
        sa.Column("state", sa.String(length=20), nullable=False, server_default="stopped"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_exit_code", sa.Integer(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("restart_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("config_version_applied", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.camera_id"], ondelete="CASCADE"),
    )
    op.create_check_constraint(
        "camera_process_status_state_check",
        "camera_process_status",
        "state IN ('stopped', 'starting', 'running', 'crashed', 'stopping')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("camera_process_status")
    op.drop_constraint("cameras_desired_state_check", "cameras", type_="check")
    op.drop_column("cameras", "updated_at")
    op.drop_column("cameras", "launch_args")
    op.drop_column("cameras", "config_version")
    op.drop_column("cameras", "desired_state")
    op.drop_column("cameras", "gpu_id")
