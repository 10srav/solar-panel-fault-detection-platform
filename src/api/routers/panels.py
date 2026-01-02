"""Panel management API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.database import (
    Panel,
    FaultEvent,
    RiskLevelEnum,
    get_db_manager,
)
from src.api.schemas import (
    PanelCreate,
    PanelUpdate,
    PanelResponse,
    FaultEventResponse,
    PanelHistoryResponse,
)

router = APIRouter(prefix="/panels", tags=["Panels"])


async def get_db() -> AsyncSession:
    """Get database session."""
    db_manager = get_db_manager()
    async for session in db_manager.get_async_session():
        yield session


@router.post(
    "",
    response_model=PanelResponse,
    summary="Create Panel",
    description="Register a new solar panel for monitoring.",
)
async def create_panel(
    panel: PanelCreate,
    db: AsyncSession = Depends(get_db),
) -> PanelResponse:
    """Create a new panel."""
    new_panel = Panel(
        id=str(uuid4()),
        name=panel.name,
        location=panel.location,
        installation_date=panel.installation_date,
        capacity_kw=panel.capacity_kw,
    )

    db.add(new_panel)
    await db.commit()
    await db.refresh(new_panel)

    return PanelResponse.model_validate(new_panel)


@router.get(
    "",
    response_model=list[PanelResponse],
    summary="List Panels",
    description="Get all registered panels.",
)
async def list_panels(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
) -> list[PanelResponse]:
    """List all panels with pagination."""
    result = await db.execute(select(Panel).offset(skip).limit(limit))
    panels = result.scalars().all()

    return [PanelResponse.model_validate(p) for p in panels]


@router.get(
    "/{panel_id}",
    response_model=PanelResponse,
    summary="Get Panel",
    description="Get a specific panel by ID.",
)
async def get_panel(
    panel_id: str,
    db: AsyncSession = Depends(get_db),
) -> PanelResponse:
    """Get a panel by ID."""
    result = await db.execute(select(Panel).where(Panel.id == panel_id))
    panel = result.scalar_one_or_none()

    if not panel:
        raise HTTPException(status_code=404, detail="Panel not found")

    return PanelResponse.model_validate(panel)


@router.put(
    "/{panel_id}",
    response_model=PanelResponse,
    summary="Update Panel",
    description="Update panel information.",
)
async def update_panel(
    panel_id: str,
    update: PanelUpdate,
    db: AsyncSession = Depends(get_db),
) -> PanelResponse:
    """Update a panel."""
    result = await db.execute(select(Panel).where(Panel.id == panel_id))
    panel = result.scalar_one_or_none()

    if not panel:
        raise HTTPException(status_code=404, detail="Panel not found")

    # Update fields
    if update.name is not None:
        panel.name = update.name
    if update.location is not None:
        panel.location = update.location
    if update.installation_date is not None:
        panel.installation_date = update.installation_date
    if update.capacity_kw is not None:
        panel.capacity_kw = update.capacity_kw

    panel.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(panel)

    return PanelResponse.model_validate(panel)


@router.delete(
    "/{panel_id}",
    summary="Delete Panel",
    description="Delete a panel and its fault history.",
)
async def delete_panel(
    panel_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete a panel."""
    result = await db.execute(select(Panel).where(Panel.id == panel_id))
    panel = result.scalar_one_or_none()

    if not panel:
        raise HTTPException(status_code=404, detail="Panel not found")

    await db.delete(panel)
    await db.commit()

    return {"message": "Panel deleted successfully"}


@router.get(
    "/{panel_id}/history",
    response_model=PanelHistoryResponse,
    summary="Get Panel History",
    description="Get fault event history for a panel.",
)
async def get_panel_history(
    panel_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> PanelHistoryResponse:
    """Get fault history for a panel."""
    # Get panel
    result = await db.execute(select(Panel).where(Panel.id == panel_id))
    panel = result.scalar_one_or_none()

    if not panel:
        raise HTTPException(status_code=404, detail="Panel not found")

    # Get fault events
    events_result = await db.execute(
        select(FaultEvent)
        .where(FaultEvent.panel_id == panel_id)
        .order_by(FaultEvent.detected_at.desc())
        .offset(skip)
        .limit(limit)
    )
    fault_events = events_result.scalars().all()

    # Get counts
    total_result = await db.execute(
        select(func.count(FaultEvent.id)).where(FaultEvent.panel_id == panel_id)
    )
    total_events = total_result.scalar() or 0

    high_risk_result = await db.execute(
        select(func.count(FaultEvent.id)).where(
            FaultEvent.panel_id == panel_id,
            FaultEvent.risk_level == RiskLevelEnum.HIGH,
        )
    )
    high_risk_events = high_risk_result.scalar() or 0

    # Get latest severity
    latest_event = fault_events[0] if fault_events else None
    latest_severity = latest_event.severity_score if latest_event else None

    return PanelHistoryResponse(
        panel=PanelResponse.model_validate(panel),
        fault_events=[
            FaultEventResponse(
                id=e.id,
                panel_id=e.panel_id,
                fault_class=e.fault_class,
                confidence=e.confidence,
                severity_score=e.severity_score,
                risk_level=e.risk_level.value,
                fault_area_ratio=e.fault_area_ratio,
                temperature_score=e.temperature_score,
                growth_rate=e.growth_rate,
                alert_triggered=bool(e.alert_triggered),
                alert_acknowledged=bool(e.alert_acknowledged),
                detected_at=e.detected_at,
            )
            for e in fault_events
        ],
        total_events=total_events,
        high_risk_events=high_risk_events,
        latest_severity=latest_severity,
    )


@router.post(
    "/{panel_id}/events",
    response_model=FaultEventResponse,
    summary="Create Fault Event",
    description="Record a new fault event for a panel.",
)
async def create_fault_event(
    panel_id: str,
    fault_class: str,
    confidence: float,
    severity_score: float,
    risk_level: str,
    fault_area_ratio: Optional[float] = None,
    temperature_score: Optional[float] = None,
    growth_rate: Optional[float] = None,
    db: AsyncSession = Depends(get_db),
) -> FaultEventResponse:
    """Create a fault event for a panel."""
    # Verify panel exists
    result = await db.execute(select(Panel).where(Panel.id == panel_id))
    panel = result.scalar_one_or_none()

    if not panel:
        raise HTTPException(status_code=404, detail="Panel not found")

    # Create event
    event = FaultEvent(
        id=str(uuid4()),
        panel_id=panel_id,
        fault_class=fault_class,
        confidence=confidence,
        severity_score=severity_score,
        risk_level=RiskLevelEnum(risk_level),
        fault_area_ratio=fault_area_ratio,
        temperature_score=temperature_score,
        growth_rate=growth_rate,
        alert_triggered=1 if risk_level == "High" else 0,
    )

    db.add(event)
    await db.commit()
    await db.refresh(event)

    return FaultEventResponse(
        id=event.id,
        panel_id=event.panel_id,
        fault_class=event.fault_class,
        confidence=event.confidence,
        severity_score=event.severity_score,
        risk_level=event.risk_level.value,
        fault_area_ratio=event.fault_area_ratio,
        temperature_score=event.temperature_score,
        growth_rate=event.growth_rate,
        alert_triggered=bool(event.alert_triggered),
        alert_acknowledged=bool(event.alert_acknowledged),
        detected_at=event.detected_at,
    )
