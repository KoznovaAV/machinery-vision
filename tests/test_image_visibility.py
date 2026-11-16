"""Visibility refusals protect incomplete observations without predicting a model."""
from PIL import Image
from src.models.image_visibility import assess_visibility, withhold_unreliable_details


def test_dark_scene_and_machine_withhold_family_and_dirt():
    dark = Image.new('RGB', (64, 64), (15, 15, 15))
    evidence = assess_visibility(dark, dark)
    prediction = dict(family='mtz_1221_family', family_candidate='mtz_1221_family')
    attributes = {'dirty': {'status': 'assessed', 'value': True}}
    withhold_unreliable_details(prediction, attributes, evidence)
    assert prediction['family'] is None and prediction['family_candidate'] is None
    assert prediction['raw_family_candidate'] == 'mtz_1221_family'
    assert attributes['dirty']['value'] is None


def test_dark_background_alone_does_not_withhold_visible_machine():
    scene = Image.new('RGB', (64, 64), (15, 15, 15))
    machine = Image.new('RGB', (64, 64), (180, 180, 180))
    evidence = assess_visibility(scene, machine)
    prediction = dict(family='mtz_80_family')
    attributes = {'dirty': {'value': True}}
    withhold_unreliable_details(prediction, attributes, evidence)
    assert prediction['family'] == 'mtz_80_family' and attributes['dirty']['value']


def test_partial_machine_is_visible_but_details_need_another_frame():
    evidence = dict(low_light=False)
    prediction = dict(family='mtz_80_family', family_candidate='mtz_80_family')
    attributes = {'dirty': {'value': True}}
    withhold_unreliable_details(prediction, attributes, evidence, partial=True)
    assert prediction['status'] == 'manual_review' and prediction['family'] is None
    assert attributes['dirty']['value'] is None


def test_uncertain_overlap_withholds_surface_state_but_keeps_scene_weather():
    from src.models.image_visibility import withhold_ambiguous_surface_state
    attributes={'dirty':{'value':True},'wet':{'value':False},'snow':{'value':True}}
    assert withhold_ambiguous_surface_state(attributes,[0,0,100,100],[[80,50,130,120]])
    assert attributes['dirty']['value'] is None and attributes['wet']['value'] is None
    assert attributes['snow']['value'] is True


def test_separate_machine_is_not_affected_by_distant_uncertain_region():
    from src.models.image_visibility import withhold_ambiguous_surface_state
    attributes={'dirty':{'value':True},'wet':{'value':False}}
    assert not withhold_ambiguous_surface_state(attributes,[0,0,100,100],[[200,0,300,100]])
    assert attributes['dirty']['value'] is True
