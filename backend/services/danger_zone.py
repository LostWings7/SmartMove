"""
SafeCity Loop V2 — Danger Zone & SafeCity Score Service
Calculates road-level Danger Zone Score (0–100) and SafeCity Score (0–100).
Aggregates:
* pothole severity
* hazard frequency
* conflict frequency
* conflict severity
* traffic exposure
* pedestrian exposure
* motorcycle exposure
* repeated observations / persistence
* road importance
* data confidence & coverage
⚠️ Calculated Danger Zone Score — not scientifically validated accident probability.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from sqlalchemy.orm import Session
from backend.models import Road, Hazard, NearMiss, Junction, CitizenReport
from backend.services.data_quality_service import data_quality_service


@dataclass
class SafeCityFactors:
    pothole_risk_level: str        # LOW / MEDIUM / HIGH
    junction_risk_level: str       # LOW / MEDIUM / HIGH
    traffic_exposure_level: str    # LOW / MEDIUM / HIGH
    pedestrian_exposure_level: str # LOW / MEDIUM / HIGH
    pothole_score: float           # 0–100
    junction_score: float          # 0–100
    traffic_score: float           # 0–100
    vulnerability_score: float     # 0–100
    report_count: int
    # Extended metrics
    pothole_severity: str = "MEDIUM"
    hazard_frequency: int = 0
    conflict_frequency: int = 0
    conflict_severity: str = "MEDIUM"
    motorcycle_exposure: str = "MEDIUM"
    persistence_score: float = 0.0
    road_importance_score: float = 50.0
    data_confidence: str = "HIGH"
    main_contributing_factor: str = "Observed Surface Defects"
    critical_conflicts_per_100_vehicles: float = 0.0
    critical_conflicts_per_100_pedestrians: float = 0.0
    severity_weighted_conflict_rate: float = 0.0
    exposure_context: str = "Baseline Monitored Exposure"


@dataclass
class RoadSafetyEvaluation:
    road_id: int
    road_name: str
    danger_zone_score: float       # 0–100 (Calculated Danger Zone Score)
    danger_zone_classification: str # LOW / MEDIUM / HIGH / VERY HIGH
    safe_city_score: float         # 0–100 (higher = safer)
    factors: SafeCityFactors
    conflict_count: int = 0
    hazard_count: int = 0
    traffic_exposure: str = "HIGH"
    main_contributing_factor: str = "Observed Surface Defects"
    confidence: str = "HIGH"
    coverage: str = "HIGH COVERAGE"
    last_observed: Optional[str] = None
    is_hotspot: bool = False
    quality_warning: Optional[str] = None
    disclaimer: str = "Calculated Danger Zone Score (Not scientifically validated accident probability)"

    def to_dict(self) -> dict:
        return {
            "road_id": self.road_id,
            "road_name": self.road_name,
            "danger_zone_score": round(self.danger_zone_score, 1),
            "danger_zone_classification": self.danger_zone_classification,
            "safe_city_score": round(self.safe_city_score, 1),
            "conflict_count": self.conflict_count,
            "hazard_count": self.hazard_count,
            "traffic_exposure": self.traffic_exposure,
            "main_contributing_factor": self.main_contributing_factor,
            "confidence": self.confidence,
            "coverage": self.coverage,
            "last_observed": self.last_observed,
            "is_hotspot": self.is_hotspot,
            "quality_warning": self.quality_warning,
            "factors": {
                "pothole_risk": self.factors.pothole_risk_level,
                "junction_risk": self.factors.junction_risk_level,
                "traffic_exposure": self.factors.traffic_exposure_level,
                "pedestrian_exposure": self.factors.pedestrian_exposure_level,
                "pothole_score": round(self.factors.pothole_score, 1),
                "junction_score": round(self.factors.junction_score, 1),
                "traffic_score": round(self.factors.traffic_score, 1),
                "vulnerability_score": round(self.factors.vulnerability_score, 1),
                "report_count": self.factors.report_count,
                "pothole_severity": self.factors.pothole_severity,
                "hazard_frequency": self.factors.hazard_frequency,
                "conflict_frequency": self.factors.conflict_frequency,
                "conflict_severity": self.factors.conflict_severity,
                "motorcycle_exposure": self.factors.motorcycle_exposure,
                "persistence_score": round(self.factors.persistence_score, 1),
                "road_importance_score": round(self.factors.road_importance_score, 1),
                "data_confidence": self.factors.data_confidence,
                "main_contributing_factor": self.factors.main_contributing_factor,
                "critical_conflicts_per_100_vehicles": self.factors.critical_conflicts_per_100_vehicles,
                "critical_conflicts_per_100_pedestrians": self.factors.critical_conflicts_per_100_pedestrians,
                "severity_weighted_conflict_rate": self.factors.severity_weighted_conflict_rate,
                "exposure_context": self.factors.exposure_context,
            },
            "disclaimer": self.disclaimer,
        }


class DangerZoneService:
    """
    Computes composite road risk, hotspot identification, and safe city scores combining:
    - Pothole severity & frequency
    - Conflict frequency & surrogate safety severity (TTC)
    - Traffic exposure, motorcycle exposure & pedestrian vulnerability
    - Incident persistence across repeated observations
    - Data completeness and confidence
    """

    EXPOSURE_WEIGHTS = {"HIGH": 90.0, "MEDIUM": 55.0, "LOW": 20.0}
    VULNERABILITY_WEIGHTS = {"HIGH": 90.0, "MEDIUM": 50.0, "LOW": 20.0}

    @staticmethod
    def classify_score(score: float) -> str:
        if score >= 80.0:
            return "VERY HIGH"
        elif score >= 60.0:
            return "HIGH"
        elif score >= 40.0:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def level_from_score(score: float) -> str:
        if score >= 65.0:
            return "HIGH"
        elif score >= 35.0:
            return "MEDIUM"
        return "LOW"

    def evaluate_road(self, road: Road, db: Session) -> RoadSafetyEvaluation:
        # 1. Hazards & Potholes
        hazards = db.query(Hazard).filter(Hazard.road_id == road.id).all()
        potholes = [h for h in hazards if h.type == "pothole"]
        critical_potholes = [h for h in potholes if h.severity == "CRITICAL"]
        high_potholes = [h for h in potholes if h.severity == "HIGH"]

        if critical_potholes:
            pothole_sev = "CRITICAL"
        elif high_potholes:
            pothole_sev = "HIGH"
        elif potholes:
            pothole_sev = "MEDIUM"
        else:
            pothole_sev = "LOW"

        if potholes:
            pothole_risk_sum = sum(h.risk_score for h in potholes)
            pothole_score = min(100.0, (pothole_risk_sum / len(potholes)) * 0.7 + (len(potholes) * 8.0))
        else:
            pothole_score = 15.0 if hazards else 5.0

        # 2. Junctions & Conflicts
        lat, lon = road.latitude, road.longitude
        nearby_junctions = (
            db.query(Junction)
            .filter(
                Junction.latitude.between(lat - 0.008, lat + 0.008),
                Junction.longitude.between(lon - 0.008, lon + 0.008),
            )
            .all()
        )
        junction_ids = [j.id for j in nearby_junctions] if nearby_junctions else []
        conflicts = db.query(NearMiss).filter(
            (NearMiss.road_id == road.id) | (NearMiss.junction_id.in_(junction_ids))
        ).all()

        if conflicts:
            conflict_sev_list = [c.risk_level or "MEDIUM" for c in conflicts]
            if "CRITICAL" in conflict_sev_list or any(c.ttc and c.ttc < 1.2 for c in conflicts):
                conflict_sev = "CRITICAL"
            elif "HIGH" in conflict_sev_list or any(c.ttc and c.ttc < 1.8 for c in conflicts):
                conflict_sev = "HIGH"
            else:
                conflict_sev = "MEDIUM"

            max_j_risk = max([j.risk_score for j in nearby_junctions], default=50.0)
            junction_score = min(100.0, max_j_risk * 0.65 + min(len(conflicts) * 7.0, 35.0))
        else:
            conflict_sev = "NONE"
            junction_score = 20.0

        # 3. Traffic & Vulnerability Exposures
        traffic_score = self.EXPOSURE_WEIGHTS.get(road.traffic_exposure.upper(), 50.0)
        vulnerability_score = self.VULNERABILITY_WEIGHTS.get(road.vulnerability.upper(), 50.0)
        if "school" in road.name.lower():
            vulnerability_score = max(vulnerability_score, 92.0)
        elif "hospital" in road.name.lower():
            vulnerability_score = max(vulnerability_score, 82.0)

        # Motorcycle exposure
        motorcycle_conflicts = [
            c for c in conflicts
            if "motorcycle" in (c.object_type_a or "").lower()
            or "motorcycle" in (c.object_type_b or "").lower()
            or "motorcycle" in (c.direction or "").lower()
        ]
        if motorcycle_conflicts or (road.traffic_exposure.upper() == "HIGH" and road.vulnerability.upper() == "HIGH"):
            motorcycle_exposure = "HIGH"
        elif road.traffic_exposure.upper() == "HIGH":
            motorcycle_exposure = "MEDIUM"
        else:
            motorcycle_exposure = "LOW"

        # 4. Citizen Reports
        reports_count = db.query(CitizenReport).filter(
            CitizenReport.description.ilike(f"%{road.name.split('[')[0].strip()}%")
        ).count()
        report_score = min(100.0, reports_count * 20.0)

        # 5. Persistence & Road Importance
        persistence_score = min(100.0, (len(hazards) + len(conflicts) + reports_count) * 16.0)
        road_importance_score = getattr(road, "importance_score", 50.0) or 50.0

        # 6. Weighted Calculated Danger Zone Score (0–100)
        danger_zone_score = min(
            100.0,
            (pothole_score * 0.30)
            + (junction_score * 0.25)
            + (traffic_score * 0.20)
            + (vulnerability_score * 0.15)
            + (report_score * 0.10)
        )
        danger_zone_score = round(max(0.0, danger_zone_score), 1)
        safe_city_score = round(max(0.0, min(100.0, 100.0 - danger_zone_score)), 1)
        classification = self.classify_score(danger_zone_score)

        # 7. Main Contributing Factor
        if critical_potholes or pothole_score >= 70.0:
            main_factor = f"Pothole Severity ({len(critical_potholes) or len(potholes)} active defects)"
        elif conflicts and junction_score >= 65.0:
            main_factor = f"Traffic Conflicts ({len(conflicts)} surrogate near-misses)"
        elif vulnerability_score >= 80.0:
            main_factor = "Vulnerable Road Users (School / Pedestrian Zone)"
        elif traffic_score >= 75.0:
            main_factor = "High Vehicle Volume & Traffic Exposure"
        else:
            main_factor = "Surface Wear & Operational Persistence"

        # 8. Hotspot Determination (Hotspot based on empirical evidence, not just 1 solitary observation)
        total_obs = len(hazards) + len(conflicts)
        is_hotspot = (
            total_obs >= 2 or
            len(critical_potholes) >= 1 or
            (len(conflicts) >= 1 and any(c.ttc and c.ttc < 1.5 for c in conflicts)) or
            danger_zone_score >= 70.0
        )

        # 9. Data Quality & Coverage
        quality = data_quality_service.evaluate_road_data_quality(road, db)
        coverage_label = f"{quality['coverage_tier']} COVERAGE"

        # 10. Exposure-Normalized Conflict Rates (Surrogate Safety Analysis)
        exp_veh = 250.0 if (road.traffic_exposure or "").upper() == "HIGH" else 120.0 if (road.traffic_exposure or "").upper() == "MEDIUM" else 50.0
        exp_ped = 180.0 if (road.vulnerability or "").upper() == "HIGH" else 80.0 if (road.vulnerability or "").upper() == "MEDIUM" else 30.0
        crit_conflicts = [c for c in conflicts if (c.risk_level or "").upper() == "CRITICAL" or (c.ttc and c.ttc < 1.0)]
        high_conflicts = [c for c in conflicts if (c.risk_level or "").upper() == "HIGH" or (c.ttc and c.ttc < 1.6)]
        
        crit_rate_veh = round((len(crit_conflicts) / exp_veh) * 100.0, 2)
        crit_rate_ped = round((len(crit_conflicts) / exp_ped) * 100.0, 2)
        sev_weight_sum = (len(crit_conflicts) * 1.0) + (len(high_conflicts) * 0.6) + ((len(conflicts) - len(crit_conflicts) - len(high_conflicts)) * 0.3)
        sev_weight_rate = round((sev_weight_sum / exp_veh) * 100.0, 2)
        exp_ctx = f"Calculated over estimated {int(exp_veh)} vehicle & {int(exp_ped)} VRU encounter exposures"

        factors = SafeCityFactors(
            pothole_risk_level=self.level_from_score(pothole_score),
            junction_risk_level=self.level_from_score(junction_score),
            traffic_exposure_level=road.traffic_exposure.upper(),
            pedestrian_exposure_level=road.vulnerability.upper(),
            pothole_score=pothole_score,
            junction_score=junction_score,
            traffic_score=traffic_score,
            vulnerability_score=vulnerability_score,
            report_count=reports_count,
            pothole_severity=pothole_sev,
            hazard_frequency=len(hazards),
            conflict_frequency=len(conflicts),
            conflict_severity=conflict_sev,
            motorcycle_exposure=motorcycle_exposure,
            persistence_score=persistence_score,
            road_importance_score=road_importance_score,
            data_confidence=quality["confidence_tier"],
            main_contributing_factor=main_factor,
            critical_conflicts_per_100_vehicles=crit_rate_veh,
            critical_conflicts_per_100_pedestrians=crit_rate_ped,
            severity_weighted_conflict_rate=sev_weight_rate,
            exposure_context=exp_ctx,
        )

        return RoadSafetyEvaluation(
            road_id=road.id,
            road_name=road.name,
            danger_zone_score=danger_zone_score,
            danger_zone_classification=classification,
            safe_city_score=safe_city_score,
            factors=factors,
            conflict_count=len(conflicts),
            hazard_count=len(hazards),
            traffic_exposure=road.traffic_exposure.upper(),
            main_contributing_factor=main_factor,
            confidence=quality["confidence_tier"],
            coverage=coverage_label,
            last_observed=quality["last_observation"],
            is_hotspot=is_hotspot,
            quality_warning=quality["quality_warning"],
        )

    def evaluate_all(self, db: Session) -> List[RoadSafetyEvaluation]:
        roads = db.query(Road).all()
        evaluations = [self.evaluate_road(r, db) for r in roads]
        for ev in evaluations:
            road = db.query(Road).filter(Road.id == ev.road_id).first()
            if road:
                road.risk_score = ev.danger_zone_score
                road.safe_city_score = ev.safe_city_score
        db.commit()
        return sorted(evaluations, key=lambda x: x.danger_zone_score, reverse=True)


danger_zone_service = DangerZoneService()
