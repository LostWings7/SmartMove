"""
SafeCity Loop V2 — Safer Route Navigation Router
Endpoints:
- GET /api/routes/compare (Compare Fastest Route vs Lower-Risk Route)
- POST /api/routes/compare
"""
from fastapi import APIRouter, Depends, Query, Body
from sqlalchemy.orm import Session
from typing import Optional
from backend.database import get_db
from backend.schemas import RouteComparisonResultOut, RouteOptionOut, RouteWaypointOut
from backend.services.route_engine import RouteEngine

router = APIRouter()
route_engine = RouteEngine()


@router.get("/routes/compare", response_model=RouteComparisonResultOut)
def compare_routes_get(
    origin_id: Optional[int] = Query(None),
    destination_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Compares Fastest Route vs Lower-Risk Route between two monitored road locations.
    Outputs distance, travel time, pothole risk, junction risk, and overall calculated risk.
    ⚠️ Neutral Presentation: Both routes are presented without forcing the user to select one.
    """
    orig = origin_id or 1
    dest = destination_id or 2
    if orig == dest:
        dest = 2 if orig != 2 else 3

    res = route_engine.compare_routes(orig, dest, db)
    return RouteComparisonResultOut(
        origin_name=res.origin_name,
        destination_name=res.destination_name,
        neutral_advisory=res.neutral_advisory,
        routes=[
            RouteOptionOut(
                route_type=r.route_type,
                route_key=r.route_key,
                distance_km=r.distance_km,
                time_minutes=r.time_minutes,
                pothole_risk=r.pothole_risk,
                junction_risk=r.junction_risk,
                overall_calculated_risk=r.overall_calculated_risk,
                risk_classification=r.risk_classification,
                description=r.description,
                risk_penalty=r.risk_penalty,
                route_cost=r.route_cost,
                high_risk_segments_count=r.high_risk_segments_count,
                hazards_count=r.hazards_count,
                conflict_hotspots_count=r.conflict_hotspots_count,
                explanation=r.explanation,
                maneuvers=r.maneuvers,
                waypoints=[
                    RouteWaypointOut(
                        latitude=w.latitude,
                        longitude=w.longitude,
                        name=w.name,
                        road_id=w.road_id,
                        risk_score=w.risk_score,
                    )
                    for w in r.waypoints
                ],
            )
            for r in res.routes
        ],
    )


@router.post("/routes/compare", response_model=RouteComparisonResultOut)
def compare_routes_post(
    origin_id: int = Body(..., embed=True),
    destination_id: int = Body(..., embed=True),
    db: Session = Depends(get_db),
):
    return compare_routes_get(origin_id, destination_id, db)
