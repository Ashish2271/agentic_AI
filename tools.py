#!/usr/bin/env python3
"""
tools.py — Tool implementations for the OpenRouter agent.
Each function matches a tool name and receives **kwargs from the model's tool_call arguments.
"""

import os
import subprocess
import glob


# ── Tool definitions (sent to the API) ───────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (or overwrite) content to a file. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return stdout + stderr. Timeout: 30 s.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories inside a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory (defaults to current directory).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for a text pattern across files matching a glob, like grep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text or substring to search for.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files, e.g. '**/*.py'. Defaults to '**/*'.",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Root directory to search in. Defaults to current directory.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]


# ── Tool implementations ───────────────────────────────────────────────────────

def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        lines = content.splitlines()
        return f"[{len(lines)} lines]\n{content}"
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def run_command(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        parts.append(f"exit code: {result.returncode}")
        return "\n".join(parts) if parts else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error running command: {e}"


def list_directory(path: str = ".") -> str:
    try:
        entries = sorted(os.listdir(path))
        if not entries:
            return f"(empty directory: {path})"
        lines = []
        for entry in entries:
            full = os.path.join(path, entry)
            tag = "/" if os.path.isdir(full) else ""
            lines.append(f"  {entry}{tag}")
        return f"{path}/\n" + "\n".join(lines)
    except FileNotFoundError:
        return f"Error: directory not found: {path}"
    except Exception as e:
        return f"Error listing directory: {e}"


def search_files(pattern: str, glob: str = "**/*", directory: str = ".") -> str:
    try:
        matches = []
        for filepath in __import__("glob").glob(
            os.path.join(directory, glob), recursive=True
        ):
            if not os.path.isfile(filepath):
                continue
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if pattern in line:
                            matches.append(f"{filepath}:{i}: {line.rstrip()}")
            except Exception:
                continue
        if not matches:
            return f"No matches for '{pattern}' in {glob}"
        return "\n".join(matches[:100])  # cap at 100 results
    except Exception as e:
        return f"Error searching files: {e}"


# ── Dispatch ──────────────────────────────────────────────────────────────────

TOOL_MAP = {
    "read_file":      read_file,
    "write_file":     write_file,
    "run_command":    run_command,
    "list_directory": list_directory,
    "search_files":   search_files,
}


def execute_tool(name: str, arguments: dict) -> str:
    """Look up and run a tool by name. Returns a string result."""
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'"
    try:
        return fn(**arguments)
    except TypeError as e:
        return f"Error: bad arguments for '{name}': {e}"