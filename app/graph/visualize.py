from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Sequence


PIPELINE_PATH = Path(__file__).resolve().parents[1] / "core" / "pipeline.py"


def build_mermaid_graph(pipeline_path: Path = PIPELINE_PATH) -> str:
    """Return a Mermaid diagram from the current QA graph builder source."""
    nodes, entry_point, edges = _read_graph_definition(pipeline_path)
    lines = ["flowchart TD"]
    for node in ["START", *nodes, "END"]:
        lines.append(f'    {node}["{node}"]')
    lines.append(f"    START --> {entry_point}")
    for source, target in edges:
        lines.append(f"    {source} --> {target}")
    return "\n".join(lines)


def _read_graph_definition(
    pipeline_path: Path,
) -> tuple[list[str], str, list[tuple[str, str]]]:
    """Read LangGraph node and edge declarations from `_build_graph`."""
    tree = ast.parse(pipeline_path.read_text(encoding="utf-8"))
    build_graph = _find_build_graph(tree)
    nodes: list[str] = []
    edges: list[tuple[str, str]] = []
    entry_point = ""

    for statement in build_graph.body:
        if not isinstance(statement, ast.Expr):
            continue
        call = statement.value
        if not isinstance(call, ast.Call):
            continue
        name = _call_name(call)
        if name == "add_node":
            nodes.append(_node_name(call.args[0]))
        elif name == "set_entry_point":
            entry_point = _node_name(call.args[0])
        elif name == "add_edge":
            edges.append(
                (
                    _node_name(call.args[0]),
                    _node_name(call.args[1]),
                )
            )

    return nodes, entry_point, edges


def _find_build_graph(tree: ast.Module) -> ast.FunctionDef:
    """Return the `_build_graph` function node from the pipeline module."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_build_graph":
            return node
    raise ValueError("_build_graph was not found in pipeline.py")


def _call_name(call: ast.Call) -> str:
    """Return the attribute name for a graph-builder method call."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _node_name(node: ast.AST) -> str:
    """Return a Mermaid-safe node name from a LangGraph call argument."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id == "END":
        return "END"
    return ast.unparse(node)


def main(argv: Sequence[str] | None = None) -> None:
    """Print or write the current QA LangGraph Mermaid visualization."""
    parser = argparse.ArgumentParser(
        description="Render the current QA LangGraph as Mermaid text."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the Mermaid output file.",
    )
    args = parser.parse_args(argv)

    mermaid = build_mermaid_graph()
    if args.output:
        args.output.write_text(mermaid, encoding="utf-8")
        return

    print(mermaid)


if __name__ == "__main__":
    main()
