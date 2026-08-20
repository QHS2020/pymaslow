"""Unit tests for the pymaslow.prompts module."""

import pytest

import pymaslow
from pymaslow import prompts


def test_datasets_registered():
    assert set(prompts.DATASETS) == {"capture24", "etri", "dailylog2016"}
    assert set(prompts.PROMPT_TEMPLATES) == set(prompts.DATASETS)
    assert prompts.LLM_SYSTEM_MESSAGE == "You are a helpful assistant"
    assert prompts.DEFAULT_MODEL == "deepseek-reasoner"


def test_capture24_prompt_verbatim_structure():
    p = prompts.build_capture24_prompt(
        "bicycling, mountain, uphill, vigorous", "bicycling"
    )
    assert '"bicycling, mountain, uphill, vigorous"' in p
    assert '"bicycling"' in p
    assert "maslow hierarchy" in p
    assert "number 1 to 5" in p
    # same as the hierarchy-module builder
    assert p == pymaslow.build_hierarchy_prompt(
        "bicycling, mountain, uphill, vigorous", "bicycling"
    )


def test_etri_prompt_format():
    p = prompts.build_etri_prompt("work", "WITH_ONE", "other_indoor")
    assert '"work---WITH_ONE---other_indoor"' in p
    assert "action---condition---place" in p
    assert "number 1 to 5" in p


def test_dailylog2016_prompt_with_and_without_minor():
    p_full = prompts.build_dailylog2016_prompt("13", "Socializing", "Somethingelse")
    assert "13---Socializing---Somethingelse." in p_full
    assert "'hierarchies':[1,2]" in p_full  # required dict answer format

    p_no_minor = prompts.build_dailylog2016_prompt(15, "PersonalGrooming")
    assert "15---PersonalGrooming." in p_no_minor
    assert "---Somethingelse" not in p_no_minor


def test_get_prompt_template():
    for dataset in prompts.DATASETS:
        template = prompts.get_prompt_template(dataset)
        assert isinstance(template, str) and len(template) > 50
    with pytest.raises(ValueError):
        prompts.get_prompt_template("unknown")


def test_build_prompt_dispatch():
    assert prompts.build_prompt(
        "capture24", activity="sleeping", classification="inactivity quiet/light"
    ) == prompts.build_capture24_prompt("sleeping", "inactivity quiet/light")
    assert prompts.build_prompt(
        "etri", action="work", condition="WITH_ONE", place="other_indoor"
    ) == prompts.build_etri_prompt("work", "WITH_ONE", "other_indoor")
    assert prompts.build_prompt(
        "dailylog2016", start_time="13", major="Socializing", minor="Somethingelse"
    ) == prompts.build_dailylog2016_prompt("13", "Socializing", "Somethingelse")
    with pytest.raises(ValueError):
        prompts.build_prompt("unknown", activity="x", classification="y")


def test_prompts_exposed_at_top_level():
    assert pymaslow.ETRI_PROMPT_TEMPLATE == prompts.ETRI_PROMPT_TEMPLATE
    assert pymaslow.DAILYLOG2016_PROMPT_TEMPLATE == prompts.DAILYLOG2016_PROMPT_TEMPLATE
    assert pymaslow.CAPTURE24_PROMPT_TEMPLATE == pymaslow.LLM_PROMPT_TEMPLATE
