from mapbox_facility_finder import MapboxFacilityFinder, FacilityCandidate


def _mk_cand(lat, lon, name, source="src"):
    return FacilityCandidate(source=source, lat=lat, lon=lon, name=name)


def test_dedupe_prefers_stronger_name():
    finder = MapboxFacilityFinder(mapbox_token="fake", openai_client=None)
    c1 = _mk_cand(0.0, 0.0, "(unnamed)", source="primary")
    c2 = _mk_cand(0.0, 0.0005, "Good Plant", source="secondary")
    merged = finder._dedupe_candidates([c1, c2], threshold_km=0.2)
    assert len(merged) == 1
    assert merged[0].name == "Good Plant"
    assert merged[0].source == "secondary"


def test_dedupe_keeps_far_candidates():
    finder = MapboxFacilityFinder(mapbox_token="fake", openai_client=None)
    # roughly 1km apart
    c1 = _mk_cand(0.0, 0.0, "Plant A")
    c2 = _mk_cand(0.0, 0.01, "Plant B")
    merged = finder._dedupe_candidates([c1, c2], threshold_km=0.2)
    assert len(merged) == 2
