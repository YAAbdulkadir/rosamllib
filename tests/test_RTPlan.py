import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import RTPlanStorage, generate_uid

from rosamllib.dicoms import RTPlan


def make_rtplan_with_geometry() -> RTPlan:
    ds = Dataset()
    ds.Modality = "RTPLAN"
    ds.SOPClassUID = RTPlanStorage
    ds.SOPInstanceUID = generate_uid()

    ds.RTPlanLabel = "PLAN_GEOM"
    ds.RTPlanName = "Geometry Plan"
    ds.RTPlanDescription = "Plan with beams and control points"

    # ----- BeamSequence with geometry -----
    beam1 = Dataset()
    beam1.BeamNumber = 1
    beam1.BeamName = "B1"
    beam1.TreatmentMachineName = "LINAC1"
    beam1.BeamType = "DYNAMIC"

    # BeamLimitingDeviceSequence: X jaw + MLCX
    jaw_x = Dataset()
    jaw_x.RTBeamLimitingDeviceType = "X"
    jaw_x.NumberOfLeafJawPairs = 1

    mlc_x = Dataset()
    mlc_x.RTBeamLimitingDeviceType = "MLCX"
    mlc_x.NumberOfLeafJawPairs = 2

    beam1.BeamLimitingDeviceSequence = Sequence([jaw_x, mlc_x])

    # ControlPointSequence with isocenter & angles & positions
    cp0 = Dataset()
    cp0.ControlPointIndex = 0
    cp0.IsocenterPosition = [10.0, 20.0, 30.0]
    cp0.GantryAngle = 0.0
    cp0.BeamLimitingDeviceAngle = 10.0
    cp0.PatientSupportAngle = 270.0

    # BeamLimitingDevicePositionSequence for cp0
    jaw_pos = Dataset()
    jaw_pos.LeafJawPositions = [-50.0, 50.0]  # X jaws

    mlc_pos = Dataset()
    mlc_pos.LeafJawPositions = [-40.0, -10.0, 10.0, 40.0]  # 2 leaf pairs (left/right)

    cp0.BeamLimitingDevicePositionSequence = Sequence([jaw_pos, mlc_pos])

    cp1 = Dataset()
    cp1.ControlPointIndex = 1
    cp1.IsocenterPosition = [10.0, 20.0, 30.0]
    cp1.GantryAngle = 90.0
    cp1.BeamLimitingDeviceAngle = 20.0
    cp1.PatientSupportAngle = 270.0

    cp1.BeamLimitingDevicePositionSequence = Sequence([jaw_pos, mlc_pos])

    beam1.ControlPointSequence = Sequence([cp0, cp1])

    # Second beam, simpler
    beam2 = Dataset()
    beam2.BeamNumber = 2
    beam2.BeamName = "B2"
    beam2.TreatmentMachineName = "LINAC1"
    beam2.BeamType = "STATIC"

    ds.BeamSequence = Sequence([beam1, beam2])

    # FractionGroupSequence
    fg = Dataset()
    fg.FractionGroupNumber = 1
    fg.NumberOfFractionsPlanned = 10
    ds.FractionGroupSequence = Sequence([fg])

    rtplan = RTPlan.from_dataset(ds)
    return rtplan


def test_rtplan_from_dataset_valid():
    ds = Dataset()
    ds.Modality = "RTPLAN"
    ds.SOPClassUID = RTPlanStorage
    ds.SOPInstanceUID = generate_uid()
    ds.RTPlanLabel = "TEST"
    ds.RTPlanName = "Test Plan"
    ds.RTPlanDescription = "Desc"

    rtplan = RTPlan.from_dataset(ds)

    assert isinstance(rtplan, RTPlan)
    assert rtplan.plan_label == "TEST"
    assert rtplan.plan_name == "Test Plan"
    assert rtplan.plan_description == "Desc"


def test_rtplan_from_dataset_invalid_modality_raises():
    ds = Dataset()
    ds.Modality = "CT"
    ds.SOPClassUID = RTPlanStorage
    ds.SOPInstanceUID = generate_uid()

    with pytest.raises(ValueError):
        RTPlan.from_dataset(ds)


def test_rtplan_beam_and_fraction_helpers():
    rtplan = make_rtplan_with_geometry()

    assert rtplan.num_beams == 2
    assert len(rtplan.beam_sequence) == 2

    beam1 = rtplan.get_beam_by_number(1)
    beam2 = rtplan.get_beam_by_number(2)
    assert beam1 is not None
    assert beam2 is not None
    assert beam1.BeamName == "B1"
    assert beam2.BeamName == "B2"

    # Fraction group helpers
    assert rtplan.num_fraction_groups == 1
    assert len(rtplan.fraction_group_sequence) == 1
    assert rtplan.fraction_group_sequence[0].FractionGroupNumber == 1


def test_rtplan_isocenter_and_angles():
    rtplan = make_rtplan_with_geometry()

    # Isocenter from first CP of beam 1
    iso = rtplan.get_beam_isocenter(beam_number=1, cp_index=0)
    assert iso == (10.0, 20.0, 30.0)

    # beam_isocenters dict
    iso_map = rtplan.beam_isocenters
    assert 1 in iso_map
    assert iso_map[1] == (10.0, 20.0, 30.0)

    # Angles
    gantry_angles = rtplan.get_gantry_angles()
    collimator_angles = rtplan.get_collimator_angles()
    couch_angles = rtplan.get_couch_angles()

    assert 1 in gantry_angles
    assert gantry_angles[1] == [0.0, 90.0]

    assert 1 in collimator_angles
    assert collimator_angles[1] == [10.0, 20.0]

    assert 1 in couch_angles
    assert couch_angles[1] == [270.0, 270.0]


def test_rtplan_jaw_and_mlc_helpers():
    rtplan = make_rtplan_with_geometry()

    # Jaw positions for beam 1, CP0
    jaw_x = rtplan.get_jaw_positions(beam_number=1, axis="X", cp_index=0)
    assert jaw_x == (-50.0, 50.0)

    # MLC positions for beam 1, CP0
    mlc_positions = rtplan.get_mlc_leaf_positions(beam_number=1, cp_index=0)
    # Flat list of positions
    assert mlc_positions == [-40.0, -10.0, 10.0, 40.0]


def test_rtplan_iter_control_points():
    rtplan = make_rtplan_with_geometry()

    all_cps = list(rtplan.iter_control_points())
    assert len(all_cps) == 2  # only beam 1 has CPs

    beam1_cps = list(rtplan.iter_control_points(beam_number=1))
    assert len(beam1_cps) == 2
    assert beam1_cps[0].ControlPointIndex == 0
    assert beam1_cps[1].ControlPointIndex == 1

    # Beam 2 has no CPs in this synthetic plan
    beam2_cps = list(rtplan.iter_control_points(beam_number=2))
    assert beam2_cps == []
