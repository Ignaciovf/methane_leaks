import os
import math
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


@dataclass
class FacilityCandidate:
    source: str
    lat: float
    lon: float
    name: Optional[str]
    address: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    category_id: Optional[str] = None
    category_name: Optional[str] = None
    distance_km: Optional[float] = None
    direction_score: Optional[float] = None
    sector_score: Optional[float] = None
    text_score: Optional[float] = None
    score: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class MapboxFacilityFinder:
    SEARCHBOX_BASE = "https://api.mapbox.com/search/searchbox/v1"
    TILEQUERY_BASE = "https://api.mapbox.com/v4"
    STREETS_V8 = "mapbox.mapbox-streets-v8"
    SESSION = requests.Session()

    # seed terms (en + es) for methane-relevant facilities
    SECTOR_KEYWORDS = {
        "landfill": ["landfill", "waste", "dump", "biogas", "vertedero", "relleno sanitario", "basural"],
        "oil_gas": ["pipeline", "natural gas", "gas pipeline", "oil", "compressor station", "refinery", "lng",
                    "oleoducto", "gasoducto", "planta de gas", "estación compresora"],
        "wastewater": ["wastewater", "sewage", "sewage treatment plant", "wwtp", "depuradora", "planta de tratamiento",
                       "aguas residuales", "edar", "alcantarillado"],
        "agriculture": ["dairy", "feedlot", "farm", "manure", "biogas", "granja", "lechería", "establo"],
        "coal_mine": ["coal mine", "mine", "mina de carbón", "mina"],
    }
    GENERIC_TERMS = ["industrial", "chemical", "energy", "refinery", "pipeline", "landfill", "waste", "wastewater",
                     "sewage", "biogas", "gas", "oil", "vertedero", "relleno sanitario", "depuradora", "edar"]

    def __init__(
            self,
            mapbox_token: Optional[str] = None,
            openai_client: Optional[Any] = None,
            openai_model: str = "text-embedding-3-small",
            user_agent: str = "methane-leak-notifier/1.0 (facility-finder)",
            verbose: bool = False,
    ):
        self.token = mapbox_token or os.getenv("MAPBOX_ACCESS_TOKEN")
        if not self.token:
            raise RuntimeError("MAPBOX_ACCESS_TOKEN not set.")
        self.SESSION.headers.update({"User-Agent": user_agent})
        self._using_secret = self.token.startswith("sk.")

        self.client = openai_client
        if self.client is None and os.getenv("OPENAI_API_KEY"):
            if OpenAI is None:
                raise RuntimeError("openai package not installed, but OPENAI_API_KEY is set.")
            self.client = OpenAI()
        self.embedding_model = openai_model
        self._emb_cache: Dict[str, List[float]] = {}
        self.verbose = verbose

    # -------- logging helper --------
    def _log(self, debug: Optional[List[str]], msg: str):
        if debug is not None:
            debug.append(msg)

    # -------- PUBLIC ENTRY --------
    def find_likely_culprits(
            self,
            lat: float,
            lon: float,
            plume_bearing_deg: Optional[float],
            radius_km: float = 10.0,
            leak_type_hint: Optional[str] = None,
            limit: int = 15,
            engine: str = "auto",  # "auto" | "searchbox" | "tilequery"
            debug: Optional[List[str]] = None,
    ) -> List[FacilityCandidate]:
        """
        Simplified search:
          - engine="auto": try SearchBox /forward, then fallback to Tilequery
          - engine="searchbox": only SearchBox /forward
          - engine="tilequery": only Tilequery
        """
        self._log(debug,
                  f"params lat={lat}, lon={lon}, radius_km={radius_km}, bearing={plume_bearing_deg}, hint={leak_type_hint}, engine={engine}")
        cands: List[FacilityCandidate] = []

        if engine in ("auto", "searchbox"):
            terms = self._seed_terms(leak_type_hint)
            # build a minimal OR-style set: try a few best terms only to reduce rate-limit issues
            q_list = list(dict.fromkeys(terms[:6])) or ["industrial"]
            self._log(debug, f"searchbox terms: {q_list}")

            for q in q_list:
                c = self._searchbox_forward(q, lat, lon, radius_km, limit=limit, debug=debug)
                self._log(debug, f"/forward q='{q}' -> {len(c)} features")
                cands += c

        if engine in ("auto", "tilequery"):
            if engine == "tilequery" or (engine == "auto" and not cands):
                if self._using_secret:
                    self._log(debug, "NOTE: 'sk-' token detected. Tilequery prefers a 'pk-' token with Tiles:Read.")
                c = self._tilequery_pois(lat, lon, radius_km, limit=limit, hint=leak_type_hint, debug=debug)
                self._log(debug, f"tilequery kept {len(c)} features")
                cands += c

        # Score & rank
        for c in cands:
            c.distance_km = self._haversine_km(lat, lon, c.lat, c.lon)
            c.direction_score = self._direction_alignment_score(lat, lon, c.lat, c.lon, plume_bearing_deg)
            c.sector_score = self._sector_score(c, leak_type_hint)
            c.text_score = self._text_match_score(c, leak_type_hint)
            c.score = (
                    self._distance_score(c.distance_km, radius_km) * 0.40 +
                    (c.direction_score or 0) * 0.25 +
                    (c.sector_score or 0) * 0.20 +
                    (c.text_score or 0) * 0.15
            )

        # Dedup by (name,rounded coords)
        out: Dict[Tuple[str, float, float], FacilityCandidate] = {}
        for c in sorted(cands, key=lambda x: (x.score or 0), reverse=True):
            key = ((c.name or "").lower(), round(c.lat or 0.0, 5), round(c.lon or 0.0, 5))
            if key not in out:
                out[key] = c
        final = list(out.values())[: max(1, min(int(limit or 15), 50))]
        self._log(debug, f"final candidates after scoring & dedupe: {len(final)}")
        return final

    # -------- Search Box: Forward (single endpoint) --------
    def _searchbox_forward(self, query: str, lat: float, lon: float, radius_km: float, limit: int = 15,
                           debug: Optional[List[str]] = None) -> List[FacilityCandidate]:
        bbox = self._bbox_from_radius(lat, lon, radius_km)
        url = f"{self.SEARCHBOX_BASE}/forward"
        limit = max(1, min(int(limit or 10), 10))  # <-- SearchBox requires 1..10
        params = {
            "access_token": self.token,
            "q": query,
            "proximity": f"{lon},{lat}",
            "limit": limit,
            "bbox": ",".join(map(str, bbox)),
            "types": "poi,category",
            "language": "en",
        }
        self._log(debug, f"GET {url} params={params}")
        r = self.SESSION.get(url, params=params, timeout=30)
        self._log(debug, f"status {r.status_code}")
        if r.status_code in (400, 404, 422):
            try:
                self._log(debug, f"error: {r.text[:500]}")
            except Exception:
                pass
            return []
        r.raise_for_status()
        js = r.json()
        out = []
        for f in js.get("features", []):
            out.append(self._candidate_from_searchbox_feat(f, "searchbox-forward"))
        return out

    def _candidate_from_searchbox_feat(self, f: Dict[str, Any], source: str) -> FacilityCandidate:
        geom = f.get("geometry", {}).get("coordinates", [None, None])
        props = f.get("properties", {})
        context = props.get("context", {}) or {}
        lat, lon = geom[1], geom[0]
        name = props.get("name") or props.get("full_address") or props.get("name_preferred")
        cat_id = props.get("category") or props.get("poi_category")
        cat_name = props.get("poi_category_name") or props.get("maki")
        address = context.get("address", {}).get("name") or props.get("place_formatted")
        phone = props.get("tel")
        website = props.get("website")
        return FacilityCandidate(
            source=source,
            lat=lat, lon=lon,
            name=name,
            address=address, phone=phone, website=website,
            category_id=cat_id, category_name=cat_name,
            raw=f
        )

    # -------- Tilequery fallback (clamped & robust) --------
    def _tilequery_pois(self, lat: float, lon: float, radius_km: float, limit: int = 50, hint: Optional[str] = None,
                        debug: Optional[List[str]] = None) -> List[FacilityCandidate]:
        base_url = f"{self.TILEQUERY_BASE}/{self.STREETS_V8}/tilequery/{lon},{lat}.json"
        limit = max(1, min(int(limit or 15), 50))

        def do_req(layers: Optional[str], radius_m: int) -> requests.Response:
            params = {"access_token": self.token, "radius": radius_m, "limit": limit, "dedupe": "true"}
            if layers: params["layers"] = layers
            self._log(debug, f"GET {base_url} params={params}")
            r = self.SESSION.get(base_url, params=params, timeout=30)
            self._log(debug, f"status {r.status_code}")
            if r.status_code >= 400:
                try:
                    self._log(debug, f"error body: {r.text[:500]}")
                except Exception:
                    pass
            return r

        radius_m = int(radius_km * 1000)
        r = do_req("poi_label,landuse", radius_m)
        if r.status_code in (401, 403, 422):
            self._log(debug, "Retrying WITHOUT 'layers'…")
            r = do_req(None, radius_m)
        if r.status_code >= 400:
            smaller = max(int(radius_m / 2), 5000)
            self._log(debug, f"Retrying with smaller radius={smaller}…")
            r = do_req(None, smaller)
        r.raise_for_status()
        js = r.json()
        raw = js.get("features", []) or []
        self._log(debug, f"tilequery raw features: {len(raw)}")

        kept: List[FacilityCandidate] = []
        for feat in raw:
            props = feat.get("properties", {}) or {}
            layer = props.get("tilequery", {}).get("layer") or feat.get("layer") or ""
            center = feat.get("geometry", {}).get("coordinates", [None, None])
            name = props.get("name_en") or props.get("name")
            if not self._poi_relevant(props, name, hint):
                continue
            kept.append(FacilityCandidate(
                source=f"tilequery:{layer}",
                lat=center[1], lon=center[0],
                name=name,
                category_id=(props.get("class") or props.get("type")),
                category_name=(props.get("class") or props.get("type")),
                raw=feat
            ))
        return kept

    # -------- relevance filter --------
    def _poi_relevant(self, props: Dict[str, Any], name: Optional[str], hint: Optional[str]) -> bool:
        cls = (props.get("class") or "").lower()
        typ = (props.get("type") or "").lower()
        maki = (props.get("maki") or "").lower()
        nm = (name or "").lower()
        haystack = " ".join([cls, typ, maki, nm])

        # sector-specific when available
        if hint and hint in self.SECTOR_KEYWORDS:
            terms = [t.lower() for t in self.SECTOR_KEYWORDS[hint]]
        else:
            terms = []

        # broad industrial / methane-adjacent keywords (EN + ES)
        COMMON = [
            "industrial", "industry", "factory", "plant", "works", "utility", "station", "compressor", "compresora",
            "pipeline", "gasoducto", "oleoducto", "gas", "oil", "refinery", "refinería", "lng",
            "waste", "landfill", "dump", "basural", "vertedero", "relleno sanitario",
            "wastewater", "sewage", "treatment", "tratamiento", "depuradora", "edar", "aguas residuales",
            "biogas", "mina", "mine", "coal"
        ]
        terms = list(dict.fromkeys(terms + COMMON))

        # hard includes by class/type
        if cls in {"industrial", "works", "factory", "wastewater_plant", "wastewater", "utility"}:
            return True
        if typ in {"industrial", "factory", "plant", "wastewater_plant", "utility"}:
            return True

        # keyword match anywhere (class/type/maki/name)
        if any(k in haystack for k in terms):
            return True

        return False

    # -------- Diagnostics: raw Tilequery summary --------
    def debug_tilequery(self, lat: float, lon: float, radius_km: float, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit or 15), 50))
        url = f"{self.TILEQUERY_BASE}/{self.STREETS_V8}/tilequery/{lon},{lat}.json"
        params = {
            "access_token": self.token,
            "radius": int(radius_km * 1000),
            "limit": limit,
            "layers": "poi_label,landuse",
            "dedupe": "true",
        }
        r = self.SESSION.get(url, params=params, timeout=30)
        r.raise_for_status()
        features = r.json().get("features", []) or []
        out = []
        for f in features:
            p = f.get("properties", {}) or {}
            coords = f.get("geometry", {}).get("coordinates", [None, None])
            out.append({
                "layer": p.get("tilequery", {}).get("layer") or f.get("layer"),
                "class": p.get("class"),
                "type": p.get("type"),
                "maki": p.get("maki"),
                "name": p.get("name_en") or p.get("name"),
                "lon": coords[0],
                "lat": coords[1],
            })
        return out

    # -------- Scoring & utils --------
    def _distance_score(self, d_km: float, radius_km: float) -> float:
        if d_km is None: return 0.0
        return max(0.0, 1.0 - (d_km / max(radius_km, 1e-6)))

    def _direction_alignment_score(self, src_lat: float, src_lon: float, cand_lat: float, cand_lon: float,
                                   plume_bearing_deg: Optional[float]) -> float:
        if plume_bearing_deg is None: return 0.5
        target = (plume_bearing_deg + 180.0) % 360.0
        brng = self._bearing_deg(src_lat, src_lon, cand_lat, cand_lon)
        diff = self._ang_diff_deg(target, brng)
        if diff <= 30: return 1.0
        if diff >= 120: return 0.0
        return max(0.0, 1.0 - (diff - 30) / 90.0)

    def _sector_score(self, c: FacilityCandidate, hint: Optional[str]) -> float:
        if not hint:
            return 1.0 if (c.category_name or c.category_id) else 0.4
        tag = f"{c.category_name or ''} {c.category_id or ''}".lower()
        return 1.0 if hint.lower() in tag else 0.4

    def _text_match_score(self, c: FacilityCandidate, hint: Optional[str]) -> float:
        if not self.client: return 0.5
        context = (hint or "methane leak") + " methane plume likely sources: " + json.dumps({
            "name": c.name, "category": c.category_name or c.category_id, "address": c.address
        }, ensure_ascii=False)
        return self._cosine_sim(self._embed(context), self._embed(hint or "methane leak sources"))

    def _seed_terms(self, hint: Optional[str]) -> List[str]:
        if hint and hint in self.SECTOR_KEYWORDS:
            return self.SECTOR_KEYWORDS[hint]
        if hint:
            return [hint]
        return ["landfill", "wastewater", "pipeline", "compressor station", "refinery", "LNG", "industrial plant",
                "vertedero", "depuradora"]

    def _bbox_from_radius(self, lat: float, lon: float, r_km: float) -> Tuple[float, float, float, float]:
        dlat = r_km / 110.574
        dlon = r_km / (111.320 * math.cos(math.radians(lat)) + 1e-9)
        return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2) -> float:
        R = 6371.0088
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(
            dlon / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))

    @staticmethod
    def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
        y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
        x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
            math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2 - lon1))
        brng = math.degrees(math.atan2(y, x))
        return (brng + 360.0) % 360.0

    @staticmethod
    def _ang_diff_deg(a, b) -> float:
        return abs((a - b + 180) % 360 - 180)

    def _embed(self, text: str) -> List[float]:
        if text in self._emb_cache:
            return self._emb_cache[text]
        if not self.client:
            return [0.0] * 10
        resp = self.client.embeddings.create(model=self.embedding_model, input=text)
        vec = resp.data[0].embedding
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]
        self._emb_cache[text] = vec
        return vec

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        return max(0.0, min(1.0, (dot + 1) / 2))
