"""fix audit_log_compute_hash bytea cast on backslash content

Revision ID: 64c57cb0dbe8
Revises: 91f13c038035
Create Date: 2026-06-23 13:54:40.551442

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '64c57cb0dbe8'
down_revision: Union[str, Sequence[str], None] = '91f13c038035'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Bare ::bytea on concatenated text tries to parse the string using
    # bytea's *escape literal* syntax (backslash-escapes), not "just give me
    # the UTF-8 bytes" - so any details JSONB whose text form contains a
    # backslash that isn't a valid bytea escape (e.g. a JSON value that is
    # itself a JSON-looking string, a Windows path, a regex) raised
    # "invalid input syntax for type bytea" and 500'd the whole request.
    # convert_to(..., 'UTF8') encodes the text directly with no escape
    # parsing, which is what a content hash actually wants.
    op.execute("""
        CREATE OR REPLACE FUNCTION public.audit_log_compute_hash()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $function$
        BEGIN
            NEW.current_hash := encode(
                sha256(
                    convert_to(
                        COALESCE(NEW.log_id::text, '')
                        || COALESCE(NEW.event_time::text, '')
                        || COALESCE(NEW.actor_id::text, '')
                        || COALESCE(NEW.action_type, '')
                        || COALESCE(NEW.resource_type, '')
                        || COALESCE(NEW.resource_id, '')
                        || COALESCE(NEW.details::text, '')
                        || COALESCE(NEW.prev_hash, ''),
                        'UTF8'
                    )
                ),
                'hex'
            );
            RETURN NEW;
        END;
        $function$
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("""
        CREATE OR REPLACE FUNCTION public.audit_log_compute_hash()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $function$
        BEGIN
            NEW.current_hash := encode(
                sha256(
                    (   COALESCE(NEW.log_id::text, '')
                     || COALESCE(NEW.event_time::text, '')
                     || COALESCE(NEW.actor_id::text, '')
                     || COALESCE(NEW.action_type, '')
                     || COALESCE(NEW.resource_type, '')
                     || COALESCE(NEW.resource_id, '')
                     || COALESCE(NEW.details::text, '')
                     || COALESCE(NEW.prev_hash, '')
                    )::bytea
                ),
                'hex'
            );
            RETURN NEW;
        END;
        $function$
    """)
