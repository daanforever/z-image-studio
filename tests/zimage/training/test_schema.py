from __future__ import annotations

from pathlib import Path

import pytest

from zimage.prefs import load_ui_prefs
from zimage.prefs import store as prefs_store
from zimage.prefs.store import dump_document, load_document
from zimage.training import (
    CACHE_LATENT_CHANNELS,
    CACHE_LATENT_DTYPE,
    CACHE_LATENT_SPATIAL_DIVISOR,
    CACHE_PROMPT_EMBED_DTYPE,
    CACHE_PROMPT_EMBED_HIDDEN_SIZE,
    CACHE_PROMPT_EMBED_PADDED,
    CACHE_TENSOR_SCHEMA,
    CACHE_TENSOR_SCHEMA_VERSION,
    IMMUTABLE_CACHE_FIELDS,
    IMMUTABLE_JOB_FIELDS,
    IMMUTABLE_MVP_FIELDS,
    KNOWN_MAIN_SOURCE,
    KNOWN_TURBO_SOURCE,
    REBUILD_REQUIRED_JOB_FIELDS,
    SAMPLING_PARAMETER_KEYS,
    TrainingConfigError,
    UpdateClassification,
    classify_job_update,
    is_immutable_job_field,
    is_immutable_mvp_field,
    job_create_template,
    load_job_document,
    merge_sample_parameters,
    resolve_stop_condition,
    resolve_training_paths,
    validate_job_document,
)

JOB_MINIMAL = (
    Path(__file__).resolve().parents[2] / "fixtures" / "training" / "job_minimal.yaml"
)
REQUIREMENTS = Path(__file__).resolve().parents[3] / "requirements.txt"


def test_training_dependency_baseline():
    requirements = set(REQUIREMENTS.read_text(encoding="utf-8").splitlines())
    assert "diffusers>=0.40.0,<0.41.0" in requirements
    assert "transformers>=4.51.0" in requirements
    assert "torchao>=0.15.0" in requirements
    assert "safetensors>=0.8.0" in requirements
    assert "python-slugify[unidecode]>=8.0.0" in requirements
    assert "peft>=0.20.0,<0.21.0" in requirements


def test_foundation_dependency_import_smoke():
    import diffusers
    import torchao.float8
    import transformers

    assert isinstance(diffusers.ZImagePipeline, type)
    assert isinstance(diffusers.ZImageTransformer2DModel, type)
    assert isinstance(transformers.Qwen3Model, type)
    assert isinstance(torchao.float8.Float8LinearConfig, type)
    assert callable(torchao.float8.convert_to_float8_training)


def test_load_minimal_job_fixture():
    job = load_job_document(JOB_MINIMAL)
    assert job["job_name"] == "minimal"
    assert job["model"]["main_transformer"]["path"] == KNOWN_MAIN_SOURCE
    assert job["model"]["main_transformer"]["revision"] is None
    assert job["model"]["sampling_transformer"] == {
        "path": KNOWN_TURBO_SOURCE,
        "revision": None,
    }
    assert job["datasets"] == []
    assert job["lora"]["rank"] == 4
    assert job["lora"]["alpha"] == 4
    assert job["lora"]["targets"] == ["to_k", "to_q", "to_v", "to_out.0"]
    assert job["precision"] == "fp8"
    assert job["gradient_checkpointing"] is True
    assert job["checkpoint_every"] == 100
    assert job["epochs"] == 1
    assert job["max_steps"] == 500
    assert job["optimizer"]["learning_rate"] == pytest.approx(1.0e-4)
    assert job["sampling"]["samples"] == [{"prompt": ""}]


def test_create_template_empty_datasets_and_placeholder_sample():
    doc = job_create_template()
    assert doc["datasets"] == []
    assert doc["sampling"]["samples"] == [{"prompt": ""}]
    assert doc["job_name"] == "Мой стиль"
    assert doc["model"]["main_transformer"] == {
        "path": KNOWN_MAIN_SOURCE,
        "revision": None,
    }
    assert doc["model"]["sampling_transformer"] == {
        "path": KNOWN_TURBO_SOURCE,
        "revision": None,
    }
    parsed = validate_job_document(doc)
    assert parsed["datasets"] == []
    assert parsed["sampling"]["samples"] == [{"prompt": ""}]
    assert parsed["model"]["main_transformer"]["revision"] is None
    assert parsed["model"]["sampling_transformer"]["revision"] is None


def test_max_steps_wins_over_epochs():
    job = validate_job_document(job_create_template())
    assert job["epochs"] == 1
    assert job["max_steps"] == 500
    assert resolve_stop_condition(job) == ("max_steps", 500)

    job["max_steps"] = None
    assert resolve_stop_condition(job) == ("epochs", 1)

    explicit = job_create_template()
    explicit["max_steps"] = None
    parsed = validate_job_document(explicit)
    assert parsed["max_steps"] is None
    assert resolve_stop_condition(parsed) == ("epochs", 1)


def test_immutable_field_list():
    assert IMMUTABLE_JOB_FIELDS == frozenset(
        {
            "model.main_transformer.path",
            "model.main_transformer.revision",
            "lora.rank",
            "lora.alpha",
            "lora.targets",
        }
    )
    assert IMMUTABLE_CACHE_FIELDS == frozenset(
        {
            "cache.tensor_schema",
            "cache.schema_version",
        }
    )
    assert IMMUTABLE_MVP_FIELDS == IMMUTABLE_JOB_FIELDS | IMMUTABLE_CACHE_FIELDS
    assert is_immutable_job_field("lora.rank")
    assert not is_immutable_job_field("optimizer.learning_rate")
    assert is_immutable_mvp_field("model.main_transformer.revision")
    assert is_immutable_mvp_field("cache.tensor_schema")
    assert is_immutable_mvp_field("cache.schema_version")
    assert not is_immutable_mvp_field("optimizer.learning_rate")


def test_cache_tensor_schema_constants():
    assert CACHE_TENSOR_SCHEMA_VERSION == 1
    assert CACHE_LATENT_DTYPE == "bf16"
    assert CACHE_LATENT_CHANNELS == 16
    assert CACHE_LATENT_SPATIAL_DIVISOR == 8
    assert CACHE_PROMPT_EMBED_DTYPE == "bf16"
    assert CACHE_PROMPT_EMBED_HIDDEN_SIZE == 2560
    assert CACHE_PROMPT_EMBED_PADDED is False
    assert CACHE_TENSOR_SCHEMA.version == CACHE_TENSOR_SCHEMA_VERSION
    assert CACHE_TENSOR_SCHEMA.latent.shape_description == "[16, H/8, W/8]"
    assert CACHE_TENSOR_SCHEMA.prompt_embedding.shape_description == "[valid_tokens, 2560]"
    assert CACHE_TENSOR_SCHEMA.prompt_embedding.padded is False


def test_missing_training_section_written_on_first_resolve():
    assert "training" not in load_document()
    paths = resolve_training_paths()
    doc = load_document()
    assert doc["training"]["datasets_dir"] == "./datasets"
    assert doc["training"]["jobs_dir"] == "./jobs"
    assert paths.datasets_dir == "./datasets"
    assert paths.jobs_dir == "./jobs"
    again = resolve_training_paths()
    assert again.datasets_dir == doc["training"]["datasets_dir"]
    assert again.jobs_dir == doc["training"]["jobs_dir"]


def test_training_section_is_bootstrapped_exactly_once(monkeypatch):
    calls: list[tuple[str, dict]] = []
    real_update_section = prefs_store.update_section

    def spy_update_section(section: str, value: dict) -> None:
        calls.append((section, value))
        real_update_section(section, value)

    monkeypatch.setattr(prefs_store, "update_section", spy_update_section)

    resolve_training_paths()
    resolve_training_paths()

    assert calls == [
        (
            "training",
            {"datasets_dir": "./datasets", "jobs_dir": "./jobs"},
        )
    ]


def test_bootstrap_preserves_ui_section():
    dump_document({"ui": {"prompt": "keep-me"}})
    resolve_training_paths()
    doc = load_document()
    assert doc["ui"]["prompt"] == "keep-me"
    assert doc["training"]["datasets_dir"] == "./datasets"


def test_existing_valid_training_section_not_rewritten():
    dump_document(
        {"training": {"datasets_dir": "./my-data", "jobs_dir": "./my-jobs"}}
    )
    paths = resolve_training_paths()
    assert paths.datasets_dir == "./my-data"
    assert paths.jobs_dir == "./my-jobs"
    assert load_document()["training"]["datasets_dir"] == "./my-data"


@pytest.mark.parametrize(
    "section",
    [
        {},
        {"datasets_dir": "./datasets"},
        {"jobs_dir": "./jobs"},
        {"datasets_dir": "", "jobs_dir": "./jobs"},
        {"datasets_dir": "./datasets", "jobs_dir": "   "},
        {"datasets_dir": 1, "jobs_dir": "./jobs"},
        {"datasets_dir": "./datasets", "jobs_dir": None},
        None,
        "nope",
    ],
)
def test_invalid_training_section_raises(section):
    dump_document({"training": section})
    before = load_document()
    with pytest.raises(TrainingConfigError):
        resolve_training_paths()
    assert load_document() == before


def test_load_ui_prefs_when_training_missing():
    dump_document({"ui": {"prompt": "hello"}})
    assert load_ui_prefs()["prompt"] == "hello"


def test_load_ui_prefs_when_training_invalid():
    dump_document(
        {
            "ui": {"prompt": "still-ok"},
            "training": {"datasets_dir": "", "jobs_dir": None},
        }
    )
    assert load_ui_prefs()["prompt"] == "still-ok"
    with pytest.raises(TrainingConfigError):
        resolve_training_paths()


def test_merge_sample_overrides_common_parameters():
    job = job_create_template()
    common = job["sampling"]["common_parameters"]
    sample = {"prompt": "a cat", "seed": 7, "width": 512}
    merged = merge_sample_parameters(common, sample)
    assert merged["prompt"] == "a cat"
    assert merged["seed"] == 7
    assert merged["width"] == 512
    assert merged["height"] == 1024
    assert merged["guidance_scale"] == 0.0
    assert merged["negative_prompt"] == ""


def test_sampling_uses_exact_diffusers_parameter_names():
    assert SAMPLING_PARAMETER_KEYS == frozenset(
        {
            "guidance_scale",
            "time_shift",
            "num_inference_steps",
            "width",
            "height",
            "seed",
            "prompt",
            "negative_prompt",
        }
    )
    template = job_create_template()
    assert set(template["sampling"]["common_parameters"]) == SAMPLING_PARAMETER_KEYS

    for alias in ("guidance", "steps"):
        job = job_create_template()
        job["sampling"]["samples"] = [{alias: 1}]
        with pytest.raises(TrainingConfigError, match="unknown"):
            validate_job_document(job)


def test_every_sampling_parameter_may_be_overridden_per_sample():
    common = job_create_template()["sampling"]["common_parameters"]
    sample = {
        "guidance_scale": 2.5,
        "time_shift": 4.0,
        "num_inference_steps": 20,
        "width": 768,
        "height": 512,
        "seed": -1,
        "prompt": "subject",
        "negative_prompt": "artifact",
    }
    assert merge_sample_parameters(common, sample) == sample


def test_dataset_name_accepts_folder_and_absolute_path():
    names = (
        "my_folder",
        "./relative_images",
        "D:/datasets/my_images",
        "/mnt/data/images",
    )
    for name in names:
        job = job_create_template()
        job["datasets"] = [{"name": name, "default_caption": "a photo"}]
        parsed = validate_job_document(job)
        assert parsed["datasets"][0]["name"] == name
        assert parsed["datasets"][0]["default_caption"] == "a photo"


def test_hf_and_local_model_roots_accept_optional_revisions():
    job = job_create_template()
    job["model"]["main_transformer"] = {
        "path": "organization/model",
        "revision": "refs/pr/12",
    }
    job["model"]["sampling_transformer"] = {
        "path": "../local model",
        "revision": None,
    }
    parsed = validate_job_document(job)
    assert parsed["model"]["main_transformer"] == {
        "path": "organization/model",
        "revision": "refs/pr/12",
    }
    assert parsed["model"]["sampling_transformer"] == {
        "path": "../local model",
        "revision": None,
    }


def test_unknown_job_keys_raise():
    job = job_create_template()
    job["init_adapter"] = "nope"
    with pytest.raises(TrainingConfigError, match="unknown"):
        validate_job_document(job)

    job = job_create_template()
    job["lora"]["init_adapter"] = True
    with pytest.raises(TrainingConfigError, match="unknown"):
        validate_job_document(job)

    job = job_create_template()
    job["model"]["extra"] = True
    with pytest.raises(TrainingConfigError, match="unknown"):
        validate_job_document(job)


def test_invalid_job_fields_raise():
    job = job_create_template()
    job["job_name"] = ""
    with pytest.raises(TrainingConfigError):
        validate_job_document(job)

    job = job_create_template()
    job["model"]["main_transformer"]["path"] = ""
    with pytest.raises(TrainingConfigError):
        validate_job_document(job)

    job = job_create_template()
    job["precision"] = "float32"
    with pytest.raises(TrainingConfigError):
        validate_job_document(job)


def test_turbo_rejected_as_main_transformer():
    job = job_create_template()
    job["model"]["main_transformer"]["path"] = KNOWN_TURBO_SOURCE
    with pytest.raises(TrainingConfigError, match="model.main_transformer"):
        validate_job_document(job)


def test_job_update_classification():
    current = job_create_template()
    assert classify_job_update(current, current) == (
        UpdateClassification.NO_CHANGE,
        (),
    )

    hot = job_create_template()
    hot["optimizer"]["learning_rate"] = 5.0e-5
    classification, changed = classify_job_update(current, hot)
    assert classification is UpdateClassification.APPLY_AT_STEP
    assert changed == ("optimizer.learning_rate",)

    rebuild = job_create_template()
    rebuild["max_sequence_length"] = 256
    classification, changed = classify_job_update(current, rebuild)
    assert classification is UpdateClassification.REBUILD_REQUIRED
    assert changed == ("max_sequence_length",)
    assert "max_sequence_length" in REBUILD_REQUIRED_JOB_FIELDS
    assert "scheduler" not in REBUILD_REQUIRED_JOB_FIELDS

    warmup = job_create_template()
    warmup["scheduler"]["warmup_steps"] = 10
    classification, changed = classify_job_update(current, warmup)
    assert classification is not UpdateClassification.REBUILD_REQUIRED
    assert classification is UpdateClassification.APPLY_AT_STEP
    assert changed == ("scheduler.warmup_steps",)

    immutable = job_create_template()
    immutable["lora"]["rank"] = 8
    classification, changed = classify_job_update(current, immutable)
    assert classification is UpdateClassification.REJECTED_IMMUTABLE
    assert changed == ("lora.rank",)

    immutable_path = job_create_template()
    immutable_path["model"]["main_transformer"]["path"] = "org/other-main"
    classification, changed = classify_job_update(current, immutable_path)
    assert classification is UpdateClassification.REJECTED_IMMUTABLE
    assert changed == ("model.main_transformer.path",)

    rebuild_sampling = job_create_template()
    rebuild_sampling["model"]["sampling_transformer"]["path"] = "org/other-turbo"
    classification, changed = classify_job_update(current, rebuild_sampling)
    assert classification is UpdateClassification.REBUILD_REQUIRED
    assert changed == ("model.sampling_transformer.path",)
    assert "model.sampling_transformer" in REBUILD_REQUIRED_JOB_FIELDS

    omit_sampling = job_create_template()
    del omit_sampling["model"]["sampling_transformer"]
    classification, changed = classify_job_update(current, omit_sampling)
    assert classification is UpdateClassification.REBUILD_REQUIRED
    assert changed == ("model.sampling_transformer",)


def test_sampling_transformer_may_be_omitted():
    job = job_create_template()
    del job["model"]["sampling_transformer"]
    parsed = validate_job_document(job)
    assert "sampling_transformer" not in parsed["model"]


def test_empty_model_requires_main_transformer():
    job = job_create_template()
    job["model"] = {}
    with pytest.raises(TrainingConfigError, match="model.main_transformer"):
        validate_job_document(job)


def test_null_sampling_transformer_is_invalid():
    job = job_create_template()
    job["model"]["sampling_transformer"] = None
    with pytest.raises(TrainingConfigError, match="must be a mapping"):
        validate_job_document(job)


def test_top_level_transformer_keys_must_be_nested_under_model():
    job = job_create_template()
    job["main_transformer"] = job["model"]["main_transformer"]
    job["sampling_transformer"] = job["model"]["sampling_transformer"]
    del job["model"]
    with pytest.raises(TrainingConfigError, match="nested under model"):
        validate_job_document(job)

    job = job_create_template()
    job["main_transformer"] = {"path": KNOWN_MAIN_SOURCE}
    with pytest.raises(TrainingConfigError, match="nested under model"):
        validate_job_document(job)


def test_template_calls_return_independent_documents():
    first = job_create_template()
    second = job_create_template()
    first["lora"]["targets"].append("changed")
    first["sampling"]["samples"][0]["prompt"] = "changed"
    assert second["lora"]["targets"] == ["to_k", "to_q", "to_v", "to_out.0"]
    assert second["sampling"]["samples"] == [{"prompt": ""}]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("lora", "rank"), 0),
        (("lora", "alpha"), 0),
        (("lora", "dropout"), 1.1),
        (("optimizer", "learning_rate"), 0),
        (("optimizer", "weight_decay"), -1),
        (("sampling", "common_parameters", "width"), 0),
        (("sampling", "common_parameters", "guidance_scale"), float("nan")),
    ],
)
def test_invalid_numeric_values_raise(path, value):
    job = job_create_template()
    target = job
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(TrainingConfigError):
        validate_job_document(job)
