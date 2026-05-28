"""Build a page listing all `peaks` added xarray accessors based on the
`functions_to_register` dictionary defined in `peaks.core.accessors.dataarray_accessors`
plus the methods on the `.tr` namespace accessor in `peaks.time_resolved.accessors`

Write to `docs/source/explanations/peaks_accessors.md`
"""

from __future__ import annotations

import importlib
from pathlib import Path
from textwrap import dedent

from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata


# formatters and row builders
def _format_table(rows: list[tuple[str, str]]) -> str:
    header = "| Accessor | Source function |\n"
    line = "|---|---|\n"
    body = "".join(f"| `.{name}()` | {{py:func}}`{src}` |\n" for name, src in rows)
    return header + line + body


def _registry_rows(module_name: str, registry_attr: str) -> list[tuple[str, str]]:
    mod = importlib.import_module(module_name)
    registry = getattr(mod, registry_attr)
    rows: list[tuple[str, str]] = []
    for sub_module, func_names in registry.items():
        full_module = f"peaks.core.{sub_module}"
        for func_name in func_names:
            rows.append((func_name, f"{full_module}.{func_name}"))
    return rows


def _dataarray_rows() -> list[tuple[str, str]]:
    return _registry_rows(
        "peaks.core.accessors.dataarray_accessors",
        "functions_to_register",
    )


def _dataset_rows() -> list[tuple[str, str]]:
    return _registry_rows(
        "peaks.core.accessors.dataset_accessors",
        "functions_to_register",
    )


def _datatree_direct_rows() -> list[tuple[str, str]]:
    return _registry_rows(
        "peaks.core.accessors.datatree_accessors",
        "functions_to_register_for_direct_accessor",
    )


def _datatree_iter_rows() -> list[tuple[str, str]]:
    return _registry_rows(
        "peaks.core.accessors.datatree_accessors",
        "functions_to_register_for_iterable_accessor",
    )


# dedicated row builder for time-resolved accessors
def _tr_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    tr_mod = importlib.import_module("peaks.time_resolved.accessors")
    tr_cls = tr_mod.TRAccessors
    for attr_name, attr_value in vars(tr_cls).items():
        if attr_name.startswith("_") or not callable(attr_value):
            continue
        source_function = getattr(attr_value, "__wrapped__", attr_value)
        src = f"{source_function.__module__}.{source_function.__name__}"
        rows.append((f"tr.{attr_name}", src))
    return rows


# assemble the full page content
def _build_page() -> str:
    intro = dedent(
        """\
        (peaks-accessors)=
        # `peaks` accessors

        `peaks` registers many of its core functions as
        [xarray accessors](https://docs.xarray.dev/en/latest/internals/extending-xarray.html) which
        can be called as methods acting directly on an {py:class}`xarray.DataArray`, {py:class}`xarray.Dataset`, or
        {py:class}`xarray.DataTree`. For example, instead of `peaks.core.process.data_select.MDC(da, E=6.2, dE=0.01)`,
        `da.MDC(6.2, 0.01)` can be used, where `da` is the {py:class}`xarray.DataArray` holding the data loaded by `peaks`.

        ## DataArray accessors

        Called directly on an {py:class}`xarray.DataArray`, e.g. `da.EDC(...)`.

        """
    )

    dataset_heading = dedent(
        """\

            ## Dataset accessors

            Called directly on an {py:class}`xarray.Dataset`.

            """
    )

    datatree_direct_heading = dedent(
        """\

        ## DataTree accessors (direct)

        Called directly on an {py:class}`xarray.DataTree` and acting on the tree as a whole.

        """
    )

    datatree_iter_heading = dedent(
        """\

        ## DataTree accessors (iterable)

        Called on an {py:class}`xarray.DataTree` and acting over the tree by applying the source function to each
        {py:class}`xarray.DataArray` in the {py:class}`xarray.DataTree`.

        """
    )

    tr_heading = dedent(
        """\

        ## Time-resolved accessors (`.tr` namespace)

        Accessed via the `.tr` namespace, e.g. `da.tr.set_t0(...)`. These work directly on {py:class}`xarray.DataArray`s and
        iteratively on {py:class}`xarray.DataTree`s.

        """
    )

    tr_rows = _tr_rows()
    tr_section = _format_table(tr_rows) if tr_rows else "_None detected._\n"

    return (
        intro
        + _format_table(_dataarray_rows())
        + dataset_heading
        + _format_table(_dataset_rows())
        + datatree_direct_heading
        + _format_table(_datatree_direct_rows())
        + datatree_iter_heading
        + _format_table(_datatree_iter_rows())
        + tr_heading
        + tr_section
    )


# sphinx hooks
def _on_builder_inited(app: Sphinx) -> None:
    out_path = Path(app.srcdir) / "explanations" / "peaks_accessors.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_build_page(), encoding="utf-8")


def setup(app: Sphinx) -> ExtensionMetadata:
    app.connect("builder-inited", _on_builder_inited)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
