"""NanoEvents schema for the ScoutingNano productions used by the analysis.

coffea's ScoutingNanoAODSchema only knows the HLT-level ``ScoutingJet`` and
``ScoutingFatJet`` collections. The productions here store re-clustered jets
(``ScoutingPFJetRecluster2`` AK4, ``ScoutingFatPFJetRecluster`` AK8 and their
CHS variants), and leptons carry their mass as ``_m``.
"""

from coffea.nanoevents import ScoutingNanoAODSchema, transforms

AK4 = "ScoutingPFJetRecluster2"
AK8 = "ScoutingFatPFJetRecluster"
AK4_CHS = "ScoutingPFJetReclusterCHS"
AK8_CHS = "ScoutingFatPFJetReclusterCHS"


class VVVScoutingSchema(ScoutingNanoAODSchema):
    # Offline NanoAOD cross-references (Jet_electronIdx1, ...) do not exist here.
    warn_missing_crossrefs = False

    mixins = {
        **ScoutingNanoAODSchema.mixins,
        AK4: "Jet",
        AK8: "FatJet",
        AK4_CHS: "Jet",
        AK8_CHS: "FatJet",
    }

    all_cross_references = {
        **ScoutingNanoAODSchema.all_cross_references,
        f"{AK4}_genJetIdx": "GenJet",
        f"{AK4}_genPartIdx": "GenPart",
        f"{AK8}_genPartIdx": "GenPart",
        "ScoutingElectron_genPartIdx": "GenPart",
        "ScoutingMuonVtx_genPartIdx": "GenPart",
    }

    # Candidate mixins require a charge; the AK8 collections have none.
    full_like_items = {
        **ScoutingNanoAODSchema.full_like_items,
        **{
            f"{name}_charge": (
                transforms.full_like_from_offsets_form,
                (f"o{name}", 0.0),
            )
            for name in (AK8, AK8_CHS)
        },
    }

    # Renamed rather than aliased: the vector mixins refuse both "m" and "mass".
    rename_items = {
        **ScoutingNanoAODSchema.rename_items,
        "ScoutingElectron_m": "ScoutingElectron_mass",
        "ScoutingMuonVtx_m": "ScoutingMuonVtx_mass",
    }

    alias_items = {
        **ScoutingNanoAODSchema.alias_items,
        "ScoutingElectron_charge": "ScoutingElectron_bestTrack_charge",
    }
