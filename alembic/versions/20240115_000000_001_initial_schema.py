"""Initial schema for solar panel fault detection

Revision ID: 001
Revises:
Create Date: 2024-01-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create panels table
    op.create_table(
        'panels',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('location', sa.String(500), nullable=True),
        sa.Column('installation_date', sa.DateTime, nullable=True),
        sa.Column('capacity_kw', sa.Float, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Create risk level enum type for PostgreSQL
    risk_level = sa.Enum('Low', 'Medium', 'High', name='risklevelenum')

    # Create fault_events table
    op.create_table(
        'fault_events',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('panel_id', sa.String(36), sa.ForeignKey('panels.id', ondelete='CASCADE'), nullable=False),
        sa.Column('fault_class', sa.String(50), nullable=False),
        sa.Column('confidence', sa.Float, nullable=False),
        sa.Column('severity_score', sa.Float, nullable=False),
        sa.Column('risk_level', risk_level, nullable=False),
        sa.Column('fault_area_ratio', sa.Float, nullable=True),
        sa.Column('temperature_score', sa.Float, nullable=True),
        sa.Column('growth_rate', sa.Float, nullable=True),
        sa.Column('rgb_image_path', sa.String(500), nullable=True),
        sa.Column('thermal_image_path', sa.String(500), nullable=True),
        sa.Column('gradcam_overlay_path', sa.String(500), nullable=True),
        sa.Column('segmentation_overlay_path', sa.String(500), nullable=True),
        sa.Column('alert_triggered', sa.Integer, server_default='0'),
        sa.Column('alert_acknowledged', sa.Integer, server_default='0'),
        sa.Column('detected_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )

    # Create inference_logs table
    op.create_table(
        'inference_logs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('endpoint', sa.String(100), nullable=False),
        sa.Column('panel_id', sa.String(36), nullable=True),
        sa.Column('request_id', sa.String(36), nullable=True),
        sa.Column('client_ip', sa.String(45), nullable=True),
        sa.Column('fault_class', sa.String(50), nullable=True),
        sa.Column('confidence', sa.Float, nullable=True),
        sa.Column('severity_score', sa.Float, nullable=True),
        sa.Column('risk_level', sa.String(20), nullable=True),
        sa.Column('latency_ms', sa.Float, nullable=True),
        sa.Column('status_code', sa.Integer, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )

    # Create indexes for better query performance
    op.create_index('ix_fault_events_panel_id', 'fault_events', ['panel_id'])
    op.create_index('ix_fault_events_detected_at', 'fault_events', ['detected_at'])
    op.create_index('ix_fault_events_risk_level', 'fault_events', ['risk_level'])
    op.create_index('ix_inference_logs_created_at', 'inference_logs', ['created_at'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_inference_logs_created_at')
    op.drop_index('ix_fault_events_risk_level')
    op.drop_index('ix_fault_events_detected_at')
    op.drop_index('ix_fault_events_panel_id')

    # Drop tables
    op.drop_table('inference_logs')
    op.drop_table('fault_events')
    op.drop_table('panels')

    # Drop enum type
    op.execute('DROP TYPE IF EXISTS risklevelenum')
