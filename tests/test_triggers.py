"""Per-category trigger paths (vvv.config.AnalysisConfig.triggers, vvv.selection)."""

import awkward as ak
import numpy as np
import pytest

from vvv.config import DATA_PRODUCTION_TRIGGERS, DEFAULT, JETHT, JETHT_OR_SINGLEMUON, AnalysisConfig
from vvv.objects import CATEGORIES
from vvv.selection import EventSelection, category_triggers, preselection, trigger_columns, trigger_paths

HADRONIC = ("boosted3", "boosted2_resolved", "boosted2_unresolved")
LEPTONIC = ("boosted2_leptonic_1lep", "boosted2_leptonic_2lep")
JETHT_EVERYWHERE = AnalysisConfig(triggers={c: JETHT for c in CATEGORIES})


def _events(jetht, singlemuon, n_ak8):
    return ak.Array({
        "DST": {"PFScouting_JetHT": jetht, "PFScouting_SingleMuon": singlemuon},
        "ScoutingFatPFJetRecluster": [[{"pt": 200.0}] * n for n in n_ak8],
    })


# JetHT only, SingleMuon only, neither, both.
EVENTS = _events([True, False, False, True], [False, True, False, True], [2, 2, 2, 2])


def test_default_paths():
    assert set(DEFAULT.triggers) == set(CATEGORIES)
    for c in HADRONIC:
        assert DEFAULT.triggers[c] == JETHT
    for c in LEPTONIC:
        assert DEFAULT.triggers[c] == JETHT_OR_SINGLEMUON
    assert set(trigger_paths(DEFAULT)) <= set(DATA_PRODUCTION_TRIGGERS)


def test_category_triggers():
    got = category_triggers(EVENTS, DEFAULT)
    for c in HADRONIC:
        np.testing.assert_array_equal(got[c], [True, False, False, True])
    for c in LEPTONIC:
        np.testing.assert_array_equal(got[c], [True, True, False, True])
    jetht = category_triggers(EVENTS, JETHT_EVERYWHERE)
    for c in CATEGORIES:
        np.testing.assert_array_equal(jetht[c], [True, False, False, True])


def test_trigger_columns():
    cols = trigger_columns(EVENTS, DEFAULT)
    assert list(cols) == ["DST_PFScouting_JetHT", "DST_PFScouting_SingleMuon"]
    assert all(v.dtype == bool for v in cols.values())


def test_preselection_takes_the_or_of_all_paths():
    events = _events([True, False, False, True], [False, True, False, True], [2, 2, 2, 1])
    np.testing.assert_array_equal(ak.to_numpy(preselection(events, DEFAULT)), [True, True, False, False])
    np.testing.assert_array_equal(ak.to_numpy(preselection(events, JETHT_EVERYWHERE)), [True, False, False, False])


def test_paths_outside_the_data_production_are_refused():
    triggers = dict(DEFAULT.triggers, boosted2_leptonic_2lep=JETHT + ("DST_PFScouting_DoubleMuon",))
    with pytest.raises(ValueError, match="DoubleMuon"):
        AnalysisConfig(triggers=triggers)
    with pytest.raises(ValueError, match="no trigger path"):
        AnalysisConfig(triggers=dict(DEFAULT.triggers, boosted3=()))


def test_every_category_needs_paths():
    with pytest.raises(ValueError, match="categories"):
        EventSelection(AnalysisConfig(triggers={"boosted3": JETHT}))
