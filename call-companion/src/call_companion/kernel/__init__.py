"""Kernel adapters; heavy backends are imported lazily by build_kernel."""

from .base import BaseKernel, KernelAuthError, KernelClient, KernelError, KernelUnavailable

__all__ = ["BaseKernel", "KernelAuthError", "KernelClient", "KernelError", "KernelUnavailable", "build_kernel"]


def build_kernel(settings) -> KernelClient:
    from .cli import CliKernel
    from .composite import CompositeKernel
    from .http import HttpKernel
    from .snapshot import SnapshotKernel

    cfg = settings.kernel
    backends: list[KernelClient] = []
    for name in cfg.backends:
        if name == "snapshot":
            backends.append(SnapshotKernel(cfg.snapshot_path))
        elif name == "http" and cfg.http.base_url:
            backends.append(
                HttpKernel(
                    cfg.http.base_url,
                    settings.kernel_token(),
                    timeout_s=cfg.http.timeout_s,
                    brief_timeout_s=cfg.http.brief_timeout_s,
                )
            )
        elif name == "cli":
            backends.append(CliKernel(cfg.cli.command, cfg.cli.timeout_s))
    if not backends:
        raise KernelError("не настроен ни один источник kernel: проверьте [kernel] в конфиге")
    return backends[0] if len(backends) == 1 else CompositeKernel(backends, cfg.search_budget_s)
