"""One-shot CUDA tensor census over production ``run_job``.

Loads ``tests/simulation/config.yaml`` and prints a per-phase VRAM census
to stdout.

Not a pytest module: filename does not match ``test_*.py`` / ``*_test.py``.
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = Path(__file__).parent / "simulation" / "config.yaml"
_JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})
_TOP_LEFTOVER_GROUPS = 40
_MODULE_BUCKETS = (
    "vae",
    "text_encoder",
    "main_transformer",
    "sampling_transformer",
    "optimizer_state",
    "preview_embed_maps",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot CUDA tensor census over production run_job using "
            f"{CONFIG_PATH.as_posix()}."
        ),
    )
    return parser.parse_args(argv)


def _fail(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", ())
    if oom_type and isinstance(exc, oom_type):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    text = str(exc).casefold()
    return "out of memory" in text and "cuda" in text


def _is_cuda_tensor(obj: Any) -> bool:
    import torch

    try:
        return isinstance(obj, torch.Tensor) and bool(obj.is_cuda)
    except Exception:
        return False


def _tensor_nbytes(tensor: Any) -> int:
    try:
        return int(tensor.nbytes)
    except Exception:
        return int(tensor.numel() * tensor.element_size())


def _nested_tensors(value: Any) -> list[Any]:
    found: list[Any] = []

    def walk(item: Any) -> None:
        if item is None:
            return
        try:
            import torch

            if isinstance(item, torch.Tensor):
                found.append(item)
                return
        except Exception:
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                walk(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                walk(nested)

    walk(value)
    return found


def _module_tensors(module: Any) -> list[Any]:
    if module is None:
        return []
    found: list[Any] = []
    try:
        found.extend(list(module.parameters(recurse=True)))
    except Exception:
        pass
    try:
        found.extend(list(module.buffers(recurse=True)))
    except Exception:
        pass
    return found


def _optimizer_tensors(optimizer: Any) -> list[Any]:
    if optimizer is None:
        return []
    state = getattr(optimizer, "state", None)
    if not isinstance(state, Mapping):
        return []
    found: list[Any] = []
    for value in state.values():
        found.extend(_nested_tensors(value))
    return found


def _cuda_nbytes_and_ids(tensors: list[Any]) -> tuple[int, set[int]]:
    ids: set[int] = set()
    seen_storage: set[tuple[int, int]] = set()
    total = 0
    for tensor in tensors:
        if not _is_cuda_tensor(tensor):
            continue
        ids.add(id(tensor))
        try:
            nbytes = _tensor_nbytes(tensor)
        except Exception:
            continue
        try:
            key = (int(tensor.untyped_storage().data_ptr()), nbytes)
            if key in seen_storage:
                continue
            seen_storage.add(key)
        except Exception:
            pass
        total += nbytes
    return total, ids


def _leftover_groups(named_ids: set[int]) -> tuple[list[dict[str, Any]], int]:
    import torch

    groups: dict[tuple[tuple[int, ...], str], dict[str, int]] = {}
    total = 0
    seen: set[int] = set()
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor) or not obj.is_cuda:
                continue
            token = id(obj)
            if token in named_ids or token in seen:
                continue
            seen.add(token)
            nbytes = _tensor_nbytes(obj)
            key = (tuple(int(dim) for dim in obj.shape), str(obj.dtype))
        except Exception:
            continue
        bucket = groups.setdefault(key, {"count": 0, "nbytes": 0})
        bucket["count"] += 1
        bucket["nbytes"] += nbytes
        total += nbytes
    ranked = sorted(groups.items(), key=lambda item: item[1]["nbytes"], reverse=True)
    leftover = [
        {
            "shape": list(shape),
            "dtype": dtype,
            "count": stats["count"],
            "nbytes": stats["nbytes"],
        }
        for (shape, dtype), stats in ranked[:_TOP_LEFTOVER_GROUPS]
    ]
    return leftover, total


def _cuda_memory() -> tuple[int, int, int]:
    import torch

    cuda = torch.cuda
    if not cuda.is_available():
        return 0, 0, 0
    try:
        cuda.synchronize()
    except Exception:
        pass
    return (
        int(cuda.memory_allocated()),
        int(cuda.memory_reserved()),
        int(cuda.max_memory_allocated()),
    )


def _empty_phase(phase: str, error: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "phase": phase,
        "memory_allocated": 0,
        "memory_reserved": 0,
        "max_memory_allocated": 0,
        "vae": 0,
        "text_encoder": 0,
        "main_transformer": 0,
        "sampling_transformer": 0,
        "optimizer_state": 0,
        "preview_embed_maps": 0,
        "leftover_groups": [],
        "leftover_nbytes": 0,
    }
    if error:
        record["error"] = error
    return record


def _print_census(snapshot: Mapping[str, Any]) -> None:
    from zimage.training.gpu_usage import format_bytes

    print(
        f"census phase={snapshot['phase']} "
        f"allocated={format_bytes(snapshot['memory_allocated'])} "
        f"reserved={format_bytes(snapshot['memory_reserved'])} "
        f"peak={format_bytes(snapshot['max_memory_allocated'])}",
        flush=True,
    )
    print(
        "census   "
        f"vae={format_bytes(snapshot['vae'])} "
        f"text_encoder={format_bytes(snapshot['text_encoder'])} "
        f"main_transformer={format_bytes(snapshot['main_transformer'])} "
        f"sampling_transformer={format_bytes(snapshot['sampling_transformer'])} "
        f"optimizer_state={format_bytes(snapshot['optimizer_state'])} "
        f"preview_embed_maps={format_bytes(snapshot['preview_embed_maps'])} "
        f"leftover={format_bytes(snapshot['leftover_nbytes'])}",
        flush=True,
    )
    for group in snapshot.get("leftover_groups") or []:
        print(
            "census   leftover "
            f"shape={group['shape']} "
            f"dtype={group['dtype']} "
            f"count={group['count']} "
            f"nbytes={format_bytes(group['nbytes'])}",
            flush=True,
        )
    error = snapshot.get("error")
    if error:
        print(f"census   error={error}", flush=True)


class VramCensus:
    """Callable ``gpu_usage_probe`` that records a per-phase tensor census."""

    def __init__(self) -> None:
        self.phases: list[dict[str, Any]] = []
        self.components: Any = None
        self.runtime: dict[str, Any] | None = None
        self.transformer: Any = None
        self.optimizer: Any = None
        self.preview_prompt_embeddings: Any = None
        self.preview_negative_embeddings: Any = None
        self.preview_sampler: Any = None

    def __call__(self, phase: str, components: Any = None) -> None:
        if components is not None:
            self.components = components
        self.record(phase)

    def capture_runtime(self, runtime: Mapping[str, Any] | None) -> None:
        if not isinstance(runtime, Mapping):
            return
        self.runtime = dict(runtime) if not isinstance(runtime, dict) else runtime
        components = runtime.get("components")
        if components is not None:
            self.components = components
        transformer = runtime.get("transformer")
        if transformer is not None:
            self.transformer = transformer
        optimizer = runtime.get("optimizer")
        if optimizer is not None:
            self.optimizer = optimizer
        prompts = runtime.get("preview_prompt_embeddings")
        if prompts is not None:
            self.preview_prompt_embeddings = prompts
        negatives = runtime.get("preview_negative_embeddings")
        if negatives is not None:
            self.preview_negative_embeddings = negatives
        sampler = runtime.get("preview_sampler")
        if sampler is not None:
            self.preview_sampler = sampler

    def record(self, phase: str) -> None:
        try:
            snapshot = self._snapshot(phase)
        except Exception as exc:
            snapshot = _empty_phase(phase, error=f"{type(exc).__name__}: {exc}")
            self.phases.append(snapshot)
            _print_census(snapshot)
            if _is_cuda_oom(exc):
                raise
            return
        self.phases.append(snapshot)
        _print_census(snapshot)

    def _snapshot(self, phase: str) -> dict[str, Any]:
        gc.collect()
        allocated, reserved, peak = _cuda_memory()
        buckets = self._named_buckets()
        named_ids: set[int] = set()
        nbytes = {key: 0 for key in _MODULE_BUCKETS}
        for key, tensors in buckets.items():
            amount, ids = _cuda_nbytes_and_ids(tensors)
            nbytes[key] = amount
            named_ids.update(ids)
        leftover, leftover_nbytes = _leftover_groups(named_ids)
        return {
            "phase": phase,
            "memory_allocated": allocated,
            "memory_reserved": reserved,
            "max_memory_allocated": peak,
            "vae": nbytes["vae"],
            "text_encoder": nbytes["text_encoder"],
            "main_transformer": nbytes["main_transformer"],
            "sampling_transformer": nbytes["sampling_transformer"],
            "optimizer_state": nbytes["optimizer_state"],
            "preview_embed_maps": nbytes["preview_embed_maps"],
            "leftover_groups": leftover,
            "leftover_nbytes": leftover_nbytes,
        }

    def _named_buckets(self) -> dict[str, list[Any]]:
        components = self.components
        main = self.transformer
        if main is None and components is not None:
            main = getattr(components, "main_transformer", None)
        vae = getattr(components, "vae", None) if components is not None else None
        text_encoder = (
            getattr(components, "text_encoder", None)
            if components is not None
            else None
        )
        sampling = (
            getattr(components, "sampling_transformer", None)
            if components is not None
            else None
        )
        preview_tensors = _nested_tensors(self.preview_prompt_embeddings)
        preview_tensors.extend(_nested_tensors(self.preview_negative_embeddings))
        sampler = self.preview_sampler
        if sampler is not None:
            preview_tensors.extend(
                _nested_tensors(getattr(sampler, "prompt_embeddings", None))
            )
            preview_tensors.extend(
                _nested_tensors(getattr(sampler, "negative_prompt_embeddings", None))
            )
        return {
            "vae": _module_tensors(vae),
            "text_encoder": _module_tensors(text_encoder),
            "main_transformer": _module_tensors(main),
            "sampling_transformer": _module_tensors(sampling),
            "optimizer_state": _optimizer_tensors(self.optimizer),
            "preview_embed_maps": preview_tensors,
        }


def _install_hooks(census: VramCensus) -> Any:
    import zimage.training.loop as loop_mod
    import zimage.training.sampling as sampling_mod

    originals = {
        "_load_lifecycle": loop_mod._load_lifecycle,
        "_build_runtime": loop_mod._build_runtime,
        "_write_checkpoint_then_sample": loop_mod._write_checkpoint_then_sample,
        "_pause_training_runtime": loop_mod._pause_training_runtime,
        "_sample_previews": loop_mod._sample_previews,
        "_restore_training_runtime": loop_mod._restore_training_runtime,
        "_run_pipeline": sampling_mod.UnfusedPreviewSampler._run_pipeline,
    }

    def load_lifecycle(job: Mapping[str, Any], injected: Mapping[str, Any]):
        result = originals["_load_lifecycle"](job, injected)
        census.components = result[0]
        census.record("load")
        return result

    def build_runtime(*args: Any, **kwargs: Any):
        runtime = originals["_build_runtime"](*args, **kwargs)
        census.capture_runtime(runtime)
        return runtime

    def write_checkpoint_then_sample(
        job_dir: Path,
        state: Any,
        runtime: dict[str, Any],
        injected: Mapping[str, Any],
    ):
        census.capture_runtime(runtime)
        census.record("step")
        return originals["_write_checkpoint_then_sample"](
            job_dir, state, runtime, injected
        )

    def pause_training_runtime(transformer: Any, optimizer: Any, injected: Any):
        result = originals["_pause_training_runtime"](
            transformer, optimizer, injected
        )
        census.optimizer = optimizer
        census.transformer = transformer
        census.record("preview_pause")
        return result

    def sample_previews(*args: Any, **kwargs: Any):
        result = originals["_sample_previews"](*args, **kwargs)
        census.record("preview_end")
        return result

    def restore_training_runtime(
        transformer: Any,
        optimizer: Any,
        training_device: Any,
        injected: Any,
    ):
        result = originals["_restore_training_runtime"](
            transformer, optimizer, training_device, injected
        )
        census.optimizer = optimizer
        census.transformer = transformer
        census.record("restore")
        return result

    def run_pipeline(self: Any, merged: Mapping[str, Any]):
        image = originals["_run_pipeline"](self, merged)
        census.preview_sampler = self
        census.record("preview_run")
        return image

    loop_mod._load_lifecycle = load_lifecycle
    loop_mod._build_runtime = build_runtime
    loop_mod._write_checkpoint_then_sample = write_checkpoint_then_sample
    loop_mod._pause_training_runtime = pause_training_runtime
    loop_mod._sample_previews = sample_previews
    loop_mod._restore_training_runtime = restore_training_runtime
    sampling_mod.UnfusedPreviewSampler._run_pipeline = run_pipeline

    def restore() -> None:
        loop_mod._load_lifecycle = originals["_load_lifecycle"]
        loop_mod._build_runtime = originals["_build_runtime"]
        loop_mod._write_checkpoint_then_sample = originals[
            "_write_checkpoint_then_sample"
        ]
        loop_mod._pause_training_runtime = originals["_pause_training_runtime"]
        loop_mod._sample_previews = originals["_sample_previews"]
        loop_mod._restore_training_runtime = originals["_restore_training_runtime"]
        sampling_mod.UnfusedPreviewSampler._run_pipeline = originals["_run_pipeline"]

    return restore


def _preview_jpegs(job_dir: Path) -> list[Path]:
    previews = job_dir / "previews"
    if not previews.is_dir():
        return []
    return sorted(
        path
        for path in previews.rglob("*")
        if path.is_file() and path.suffix.casefold() in _JPEG_SUFFIXES
    )


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)

    from zimage.config import load_dotenv
    from zimage.training.schema import load_job_document

    load_dotenv()
    document = load_job_document(CONFIG_PATH)

    import torch

    if not torch.cuda.is_available():
        _fail("CUDA is not available")

    from zimage.training.jobs import create_or_open_job, save_job_config
    from zimage.training.loop import run_job

    census = VramCensus()
    restore_hooks = _install_hooks(census)
    try:
        with tempfile.TemporaryDirectory(prefix="vram-census-") as tmp:
            tmp_path = Path(tmp)
            job_dir = create_or_open_job(document["job_name"], tmp_path / "jobs")
            (job_dir / "config.yaml").unlink()
            save_job_config(job_dir, document)
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            except Exception:
                pass
            code = run_job(job_dir, gpu_usage_probe=census)
            if code != 0:
                _fail(f"run_job exited {code}", code if code else 1)
            if not _preview_jpegs(job_dir):
                _fail("missing preview JPEG under previews/")
            recorded = {entry.get("phase") for entry in census.phases}
            missing = [
                name
                for name in ("preview_run", "preview_end")
                if name not in recorded
            ]
            if missing:
                _fail("missing census phases: " + ", ".join(missing))
    finally:
        restore_hooks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
