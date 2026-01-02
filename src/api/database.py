"""Database configuration and models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    Enum as SQLEnum,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import enum

from src.config import get_config

# Base class for models
Base = declarative_base()


class RiskLevelEnum(str, enum.Enum):
    """Risk level enumeration for database."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class Panel(Base):
    """Solar panel model."""

    __tablename__ = "panels"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name = Column(String(255), nullable=False)
    location = Column(String(500))
    installation_date = Column(DateTime)
    capacity_kw = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    fault_events = relationship("FaultEvent", back_populates="panel")


class FaultEvent(Base):
    """Fault event model."""

    __tablename__ = "fault_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    panel_id = Column(String(36), ForeignKey("panels.id"), nullable=False)

    # Classification results
    fault_class = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)

    # Severity assessment
    severity_score = Column(Float, nullable=False)
    risk_level = Column(SQLEnum(RiskLevelEnum), nullable=False)
    fault_area_ratio = Column(Float)
    temperature_score = Column(Float)
    growth_rate = Column(Float)

    # Image paths
    rgb_image_path = Column(String(500))
    thermal_image_path = Column(String(500))
    gradcam_overlay_path = Column(String(500))
    segmentation_overlay_path = Column(String(500))

    # Alert status
    alert_triggered = Column(Integer, default=0)  # 0=False, 1=True
    alert_acknowledged = Column(Integer, default=0)

    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    panel = relationship("Panel", back_populates="fault_events")


class InferenceLog(Base):
    """Log of inference requests."""

    __tablename__ = "inference_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    endpoint = Column(String(100), nullable=False)
    panel_id = Column(String(36))

    # Request info
    request_id = Column(String(36))
    client_ip = Column(String(45))

    # Results
    fault_class = Column(String(50))
    confidence = Column(Float)
    severity_score = Column(Float)
    risk_level = Column(String(20))

    # Performance
    latency_ms = Column(Float)
    status_code = Column(Integer)
    error_message = Column(Text)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)


# Database session management
class DatabaseManager:
    """Manage database connections and sessions."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        import os

        # Check for environment variable first
        self.database_url = (
            database_url
            or os.environ.get("DATABASE_URL")
            or get_config().database.url
        )

        # Determine if using SQLite
        self.is_sqlite = self.database_url.startswith("sqlite")

        # Convert to async URL if needed
        if self.database_url.startswith("postgresql://"):
            self.async_url = self.database_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
        elif self.is_sqlite:
            # SQLite async uses aiosqlite
            self.async_url = self.database_url.replace(
                "sqlite:///", "sqlite+aiosqlite:///"
            )
        else:
            self.async_url = self.database_url

        # Create sync engine (SQLite needs check_same_thread=False for testing)
        if self.is_sqlite:
            self.sync_engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False}
            )
        else:
            self.sync_engine = create_engine(self.database_url)

        # Create session factories
        self.sync_session_factory = sessionmaker(bind=self.sync_engine)

        # Create async engine and session factory
        try:
            if self.is_sqlite:
                # SQLite async needs special handling
                self.async_engine = create_async_engine(
                    self.async_url,
                    connect_args={"check_same_thread": False}
                )
            else:
                self.async_engine = create_async_engine(self.async_url)

            self.async_session_factory = sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        except Exception:
            self.async_engine = None
            self.async_session_factory = None

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(self.sync_engine)

    async def get_async_session(self) -> AsyncSession:
        """Get an async database session."""
        async with self.async_session_factory() as session:
            yield session

    def get_sync_session(self):
        """Get a sync database session."""
        session = self.sync_session_factory()
        try:
            yield session
        finally:
            session.close()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_db(database_url: Optional[str] = None) -> DatabaseManager:
    """Initialize the database."""
    global _db_manager
    _db_manager = DatabaseManager(database_url)
    _db_manager.create_tables()
    return _db_manager
