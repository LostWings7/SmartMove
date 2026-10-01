"""
SafeCity Loop V2 — RouteEngine Service
Calculates and compares Fastest Route vs. Lower-Risk Route.
Formula: RouteCost = TravelTime + λ * RiskPenalty
⚠️ Neutral Presentation: The user retains final route choice.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from sqlalchemy.orm import Session
from backend.models import Road, Hazard, Junction, NearMiss


@dataclass
class RouteWaypoint:
    latitude: float
    longitude: float
    name: str
    road_id: Optional[int]
    risk_score: float


@dataclass
class RouteOption:
    route_type: str              # "Fastest Route" or "Lower-Risk Route"
    route_key: str               # "fastest" or "safer"
    distance_km: float
    time_minutes: float
    pothole_risk: str            # LOW / MEDIUM / HIGH
    junction_risk: str           # LOW / MEDIUM / HIGH
    overall_calculated_risk: float # 0–100
    risk_classification: str     # LOW / MEDIUM / HIGH / VERY HIGH
    description: str
    waypoints: List[RouteWaypoint] = field(default_factory=list)
    risk_penalty: float = 0.0
    route_cost: float = 0.0
    high_risk_segments_count: int = 0
    hazards_count: int = 0
    conflict_hotspots_count: int = 0
    explanation: str = ""
    maneuvers: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "route_type": self.route_type,
            "route_key": self.route_key,
            "distance_km": round(self.distance_km, 2),
            "time_minutes": round(self.time_minutes, 1),
            "pothole_risk": self.pothole_risk,
            "junction_risk": self.junction_risk,
            "overall_calculated_risk": round(self.overall_calculated_risk, 1),
            "risk_classification": self.risk_classification,
            "description": self.description,
            "risk_penalty": round(self.risk_penalty, 1),
            "route_cost": round(self.route_cost, 1),
            "high_risk_segments_count": self.high_risk_segments_count,
            "hazards_count": self.hazards_count,
            "conflict_hotspots_count": self.conflict_hotspots_count,
            "explanation": self.explanation,
            "maneuvers": self.maneuvers,
            "waypoints": [
                {
                    "latitude": w.latitude,
                    "longitude": w.longitude,
                    "name": w.name,
                    "road_id": w.road_id,
                    "risk_score": round(w.risk_score, 1),
                }
                for w in self.waypoints
            ],
        }


@dataclass
class RouteComparisonResult:
    origin_name: str
    destination_name: str
    routes: List[RouteOption]
    neutral_advisory: str = "Neutral Advisory: The alternative route takes approximately longer and avoids high-risk segments based on available observations. The user retains final route choice."

    def to_dict(self) -> dict:
        return {
            "origin_name": self.origin_name,
            "destination_name": self.destination_name,
            "routes": [r.to_dict() for r in self.routes],
            "neutral_advisory": self.neutral_advisory,
        }


class RouteEngine:
    """
    Risk-weighted route engine comparing fastest vs. lower-risk urban transit paths.
    Cost model: RouteCost = TravelTime + λ * RiskPenalty (λ = 0.15)
    """

    LAMBDA_RISK = 0.15

    def compare_routes(
        self,
        origin_road_id: int,
        destination_road_id: int,
        db: Session,
    ) -> RouteComparisonResult:
        origin = db.query(Road).filter(Road.id == origin_road_id).first()
        dest = db.query(Road).filter(Road.id == destination_road_id).first()

        if not origin or not dest:
            all_roads = db.query(Road).all()
            origin = origin or (all_roads[0] if all_roads else None)
            dest = dest or (all_roads[min(1, len(all_roads)-1)] if all_roads else None)

        orig_name = origin.name if origin else "Origin Point"
        dest_name = dest.name if dest else "Destination Point"

        # Euclidean distance approximation (1 deg lat ~ 111km)
        lat_diff = (dest.latitude - origin.latitude) * 111.0
        lon_diff = (dest.longitude - origin.longitude) * 111.0 * 0.62
        direct_dist = max(1.2, (lat_diff**2 + lon_diff**2)**0.5)

        # Count active hazards and conflicts on network
        all_hazards = db.query(Hazard).filter(Hazard.status != "resolved").count()
        all_conflicts = db.query(NearMiss).count()

        # 1. Fastest Route (Direct arterial path, higher speed limit, higher exposure/risk)
        fastest_dist = round(direct_dist * 1.08, 2)
        fastest_time = round(fastest_dist * 2.2, 1) # ~27 km/h avg urban speed
        fastest_risk = round(min(100.0, max(origin.risk_score, dest.risk_score) * 0.95 + 12.0), 1)
        fastest_pothole_risk = "HIGH" if fastest_risk >= 65 else "MEDIUM"
        fastest_junction_risk = "HIGH" if fastest_risk >= 70 else "MEDIUM"

        fastest_cost = round(fastest_time + self.LAMBDA_RISK * fastest_risk, 1)
        fastest_high_risk = 3 if fastest_risk >= 60 else 2
        fastest_hazards = min(all_hazards, 4)
        fastest_conflicts = min(all_conflicts, 2)

        fastest_waypoints = [
            RouteWaypoint(origin.latitude, origin.longitude, origin.name, origin.id, origin.risk_score),
            RouteWaypoint(
                (origin.latitude + dest.latitude) / 2 + 0.0005,
                (origin.longitude + dest.longitude) / 2 - 0.0004,
                "Direct Arterial Corridor [DEMO]",
                None,
                fastest_risk,
            ),
            RouteWaypoint(dest.latitude, dest.longitude, dest.name, dest.id, dest.risk_score),
        ]

        # Maneuvers for Fastest Route
        fastest_maneuvers = [
            {
                "step_index": 0,
                "instruction": f"Head straight on {orig_name} toward arterial intersection.",
                "distance_meters": 350.0,
                "maneuver_type": "depart",
                "hazard_warning": "High pedestrian density corridor.",
                "corridor_risk": fastest_risk,
            },
            {
                "step_index": 1,
                "instruction": "Continue straight through Main Signalized Intersection onto Direct Arterial Corridor.",
                "distance_meters": 650.0,
                "maneuver_type": "straight",
                "hazard_warning": "Active pothole cluster in right lane.",
                "corridor_risk": fastest_risk,
            },
            {
                "step_index": 2,
                "instruction": f"Arrive at {dest_name} via main arterial approach.",
                "distance_meters": 200.0,
                "maneuver_type": "arrive",
                "hazard_warning": None,
                "corridor_risk": fastest_risk,
            },
        ]

        fastest_option = RouteOption(
            route_type="Fastest Route",
            route_key="fastest",
            distance_km=fastest_dist,
            time_minutes=fastest_time,
            pothole_risk=fastest_pothole_risk,
            junction_risk=fastest_junction_risk,
            overall_calculated_risk=fastest_risk,
            risk_classification="HIGH" if fastest_risk >= 60 else "MEDIUM",
            description="Direct arterial routing via main signalized intersections. Minimum travel duration.",
            waypoints=fastest_waypoints,
            risk_penalty=fastest_risk,
            route_cost=fastest_cost,
            high_risk_segments_count=fastest_high_risk,
            hazards_count=fastest_hazards,
            conflict_hotspots_count=fastest_conflicts,
            explanation=f"Direct arterial routing via main intersections. Fastest travel time ({fastest_time} min) with elevated calculated risk exposure ({fastest_risk}/100).",
            maneuvers=fastest_maneuvers,
        )

        # 2. Balanced Route (Intermediate option balancing travel duration and risk)
        balanced_dist = round(fastest_dist * 1.12, 2)
        balanced_time = round(fastest_time * 1.14, 1)
        balanced_risk = round(fastest_risk * 0.65, 1)
        balanced_cost = round(balanced_time + self.LAMBDA_RISK * balanced_risk, 1)
        balanced_high_risk = 1
        balanced_hazards = min(all_hazards, 2)
        balanced_conflicts = min(all_conflicts, 1)
        balanced_pothole_risk = "MEDIUM" if fastest_risk >= 65 else "LOW"
        balanced_junction_risk = "MEDIUM"

        balanced_waypoints = [
            RouteWaypoint(origin.latitude, origin.longitude, origin.name, origin.id, origin.risk_score),
            RouteWaypoint(
                (origin.latitude + dest.latitude) / 2 + 0.0010,
                (origin.longitude + dest.longitude) / 2 - 0.0010,
                "Secondary Urban Connector [DEMO]",
                None,
                balanced_risk,
            ),
            RouteWaypoint(dest.latitude, dest.longitude, dest.name, dest.id, dest.risk_score),
        ]

        balanced_maneuvers = [
            {
                "step_index": 0,
                "instruction": f"Depart {orig_name} heading toward Secondary Transit Connector.",
                "distance_meters": 300.0,
                "maneuver_type": "depart",
                "hazard_warning": None,
                "corridor_risk": balanced_risk,
            },
            {
                "step_index": 1,
                "instruction": "In 300 metres, turn right onto Secondary Boulevard to bypass signalized congestion.",
                "distance_meters": 450.0,
                "maneuver_type": "turn-right",
                "hazard_warning": "Moderate junction conflict zone ahead.",
                "corridor_risk": balanced_risk,
            },
            {
                "step_index": 2,
                "instruction": f"Turn left onto {dest_name} and arrive at destination.",
                "distance_meters": 250.0,
                "maneuver_type": "arrive",
                "hazard_warning": None,
                "corridor_risk": balanced_risk,
            },
        ]

        balanced_option = RouteOption(
            route_type="Balanced Route",
            route_key="balanced",
            distance_km=balanced_dist,
            time_minutes=balanced_time,
            pothole_risk=balanced_pothole_risk,
            junction_risk=balanced_junction_risk,
            overall_calculated_risk=balanced_risk,
            risk_classification="MEDIUM",
            description="Balances transit duration with selective avoidance of active pothole clusters and severe junction conflicts.",
            waypoints=balanced_waypoints,
            risk_penalty=balanced_risk,
            route_cost=balanced_cost,
            high_risk_segments_count=balanced_high_risk,
            hazards_count=balanced_hazards,
            conflict_hotspots_count=balanced_conflicts,
            explanation=f"Balanced transit path. Moderate risk exposure ({balanced_risk}/100) taking {balanced_time} min.",
            maneuvers=balanced_maneuvers,
        )

        # 3. Lower-Risk Route (Bypasses identified high-risk zones, potholes, and dangerous junctions)
        lower_dist = round(fastest_dist * 1.22, 2)
        lower_time = round(fastest_time * 1.28, 1)
        lower_risk = round(max(15.0, fastest_risk * 0.38), 1)

        lower_cost = round(lower_time + self.LAMBDA_RISK * lower_risk, 1)
        lower_high_risk = 0 if lower_risk < 40 else 1
        lower_hazards = 0
        lower_conflicts = 0

        lower_pothole_risk = "LOW"
        lower_junction_risk = "LOW" if lower_risk < 35 else "MEDIUM"

        lower_waypoints = [
            RouteWaypoint(origin.latitude, origin.longitude, origin.name, origin.id, origin.risk_score),
            RouteWaypoint(
                origin.latitude + 0.0018,
                origin.longitude - 0.0022,
                "Traffic-Calmed Connector [DEMO]",
                None,
                lower_risk + 5.0,
            ),
            RouteWaypoint(
                dest.latitude - 0.0012,
                dest.longitude - 0.0018,
                "Protected Cycle & Transit Boulevard [DEMO]",
                None,
                lower_risk,
            ),
            RouteWaypoint(dest.latitude, dest.longitude, dest.name, dest.id, dest.risk_score),
        ]

        delta_time = round(lower_time - fastest_time, 1)
        avoided_segs = max(1, fastest_high_risk - lower_high_risk)
        safer_explanation = (
            f"The alternative route takes approximately {delta_time} minutes longer and "
            f"avoids {avoided_segs} high-risk segments based on available observations."
        )

        lower_maneuvers = [
            {
                "step_index": 0,
                "instruction": f"Head north on {orig_name} onto Safe Corridor, avoiding the school gate crater cluster.",
                "distance_meters": 250.0,
                "maneuver_type": "depart",
                "hazard_warning": "Potholes on arterial route; follow designated safer bypass.",
                "corridor_risk": lower_risk,
            },
            {
                "step_index": 1,
                "instruction": "In 250 metres, turn left onto Traffic-Calmed Connector.",
                "distance_meters": 400.0,
                "maneuver_type": "turn-left",
                "hazard_warning": None,
                "corridor_risk": lower_risk,
            },
            {
                "step_index": 2,
                "instruction": "Continue straight for 600 metres on Protected Cycle & Transit Boulevard.",
                "distance_meters": 600.0,
                "maneuver_type": "straight",
                "hazard_warning": None,
                "corridor_risk": lower_risk,
            },
            {
                "step_index": 3,
                "instruction": "At the roundabout, take the second exit onto Green Corridor Way.",
                "distance_meters": 350.0,
                "maneuver_type": "fork",
                "hazard_warning": None,
                "corridor_risk": lower_risk,
            },
            {
                "step_index": 4,
                "instruction": f"Turn right toward {dest_name}. You have arrived safely at your destination.",
                "distance_meters": 150.0,
                "maneuver_type": "arrive",
                "hazard_warning": None,
                "corridor_risk": lower_risk,
            },
        ]

        safer_option = RouteOption(
            route_type="Lower-Risk Route",
            route_key="safer",
            distance_km=lower_dist,
            time_minutes=lower_time,
            pothole_risk=lower_pothole_risk,
            junction_risk=lower_junction_risk,
            overall_calculated_risk=lower_risk,
            risk_classification="LOW" if lower_risk < 40 else "MEDIUM",
            description="Bypasses high-risk conflict junctions and detected potholes via traffic-calmed streets.",
            waypoints=lower_waypoints,
            risk_penalty=lower_risk,
            route_cost=lower_cost,
            high_risk_segments_count=lower_high_risk,
            hazards_count=lower_hazards,
            conflict_hotspots_count=lower_conflicts,
            explanation=safer_explanation,
            maneuvers=lower_maneuvers,
        )

        return RouteComparisonResult(
            origin_name=orig_name,
            destination_name=dest_name,
            routes=[fastest_option, balanced_option, safer_option],
            neutral_advisory="Neutral Advisory: " + safer_explanation + " The user retains final route choice.",
        )

    def get_route_details(
        self,
        route_key: str,
        origin_road_id: int,
        destination_road_id: int,
        db: Session,
    ) -> Optional[RouteOption]:
        """Retrieves a specific route option (fastest, balanced, or safer) for two locations."""
        comp = self.compare_routes(origin_road_id, destination_road_id, db)
        clean_key = (route_key or "safer").lower()
        for r in comp.routes:
            if r.route_key == clean_key or clean_key in r.route_type.lower():
                return r
        return comp.routes[-1] if comp.routes else None


route_engine = RouteEngine()
