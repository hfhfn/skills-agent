"""
LangChain Tools 定义

使用 LangChain 1.0 的 @tool 装饰器和 ToolRuntime 定义工具：
- load_skill: 加载 Skill 详细指令（Level 2）
- bash: 执行命令/脚本（Level 3）
- read_file: 读取文件

ToolRuntime 提供访问运行时信息的统一接口：
- state: 可变的执行状态
- context: 不可变的配置（如 skill_loader）
"""

import subprocess
import fnmatch
import re
from pathlib import Path
from dataclasses import dataclass, field

from langchain.tools import tool, ToolRuntime

from .skill_loader import SkillLoader
from .stream import resolve_path


# 工具输出最大字符数（防止单次工具调用撑爆 LLM 上下文窗口）
# 30,000 字符 ≈ 7,500-10,000 tokens，对大多数模型是安全的
MAX_TOOL_OUTPUT_CHARS = 30000


def _truncate_output(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """截断工具输出，防止超出 LLM 上下文窗口"""
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + f"\n\n... [output truncated: {len(text):,} chars total, showing first {max_chars:,}]"
        + "\n[Tip: redirect large output to a file, then use read_file with offset to process in segments]"
    )


@dataclass
class SkillAgentContext:
    """
    Agent 运行时上下文

    通过 ToolRuntime[SkillAgentContext] 在 tool 中访问
    """
    skill_loader: SkillLoader
    working_directory: Path = field(default_factory=Path.cwd)


@tool
def load_skill(skill_name: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Load a skill's detailed instructions.

    This tool reads the SKILL.md file for the specified skill and returns
    its complete instructions. Use this when the user's request matches
    a skill's description from the available skills list.

    The skill's instructions will guide you on how to complete the task,
    which may include running scripts via the bash tool.

    Args:
        skill_name: Name of the skill to load (e.g., 'news-extractor')
    """
    loader = runtime.context.skill_loader

    # 尝试加载 skill
    skill_content = loader.load_skill(skill_name)

    if not skill_content:
        # 列出可用的 skills（从已扫描的元数据中获取）
        skills = loader.scan_skills()
        if skills:
            available = [s.name for s in skills]
            return f"Skill '{skill_name}' not found. Available skills: {', '.join(available)}"
        else:
            return f"Skill '{skill_name}' not found. No skills are currently available."

    # 获取 skill 路径信息
    skill_path = skill_content.metadata.skill_path
    scripts_dir = skill_path / "scripts"

    # 构建路径信息
    # 使用 --directory 让 uv 以 skill 目录为项目根，
    # 这样 uv run 会使用 skill 自己的 pyproject.toml 和虚拟环境
    path_info = f"""
## Skill Path Info

- **Skill Directory**: `{skill_path}`
- **Scripts Directory**: `{scripts_dir}`

**Important**: Skills have their own dependencies (pyproject.toml). You MUST use `--directory` so `uv` resolves the skill's own virtual environment:
```bash
uv run --directory {skill_path} scripts/script_name.py [args]
```

Do NOT run `uv run {scripts_dir}/script_name.py` directly — that would use the main project's dependencies, which may lack required packages.
"""

    # 返回 instructions 和路径信息
    return f"""# Skill: {skill_name}

## Instructions

{skill_content.instructions}
{path_info}
"""


@tool
def bash(command: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Execute a shell command (bash on Unix/macOS, cmd.exe on Windows).

    Use this for:
    - Running skill scripts (e.g., `uv run path/to/script.py args`)
    - Installing dependencies
    - File operations
    - Any shell command

    Important for Skills:
    - Script code does NOT enter the context, only the output does
    - This is Level 3 of the Skills loading mechanism
    - Follow the skill's instructions for exact command syntax

    Cross-platform Note:
    - On Unix/macOS: Uses /bin/sh (bash-compatible)
    - On Windows: Uses cmd.exe (different syntax, e.g., use 'dir' instead of 'ls')
    - For portable scripts, use Python scripts via `uv run script.py`

    Args:
        command: The shell command to execute
    """
    cwd = str(runtime.context.working_directory)

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 分钟超时
        )

        parts = []

        # 状态标记（与 ToolResultFormatter 配合）
        if result.returncode == 0:
            parts.append("[OK]")
        else:
            parts.append(f"[FAILED] Exit code: {result.returncode}")

        parts.append("")  # 空行分隔

        if result.stdout:
            parts.append(result.stdout.rstrip())

        if result.stderr:
            if result.stdout:
                parts.append("")
            parts.append("--- stderr ---")
            parts.append(result.stderr.rstrip())

        if not result.stdout and not result.stderr:
            parts.append("(no output)")

        return _truncate_output("\n".join(parts))

    except subprocess.TimeoutExpired:
        return "[FAILED] Command timed out after 300 seconds."
    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
def read_file(file_path: str, runtime: ToolRuntime[SkillAgentContext], offset: int = 0, limit: int = 2000) -> str:
    """
    Read the contents of a file, with support for paginated reading.

    Use this to:
    - Read skill documentation files
    - View script output files
    - Inspect any text file

    For large files (HTML, logs, data), use offset to read in segments:
      1. read_file("large.html")                → lines 1-2000
      2. read_file("large.html", offset=2000)   → lines 2001-4000
      3. Continue until you have enough content

    Strategy for summarizing large files:
      - Read a segment, extract/summarize key information
      - Read the next segment, repeat
      - Combine all segment summaries into a final result

    Args:
        file_path: Path to the file (absolute or relative to working directory)
        offset: Starting line number (0-based, default 0). Use to page through large files.
        limit: Max lines to read per call (default 2000, capped at 2000).
    """
    path = resolve_path(file_path, runtime.context.working_directory)

    if not path.exists():
        return f"[Error] File not found: {file_path}"

    if not path.is_file():
        return f"[Error] Not a file: {file_path}"

    try:
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")
        total_lines = len(lines)
        file_size = len(content)

        # 分段参数
        max_lines = max(1, min(limit, 2000))
        start = max(0, offset)
        end = min(start + max_lines, total_lines)
        selected = lines[start:end]

        # 添加行号
        numbered_lines = []
        for i, line in enumerate(selected, start + 1):
            numbered_lines.append(f"{i:4d}| {line}")

        result = "\n".join(numbered_lines)

        # 字符级截断（硬安全线）
        if len(result) > MAX_TOOL_OUTPUT_CHARS:
            result = result[:MAX_TOOL_OUTPUT_CHARS]
            # 计算实际显示到了第几行
            shown_lines = result.count("\n") + 1
            next_offset = start + shown_lines
            result += (
                f"\n\n... [output truncated at {MAX_TOOL_OUTPUT_CHARS:,} chars]"
                f"\n[File: {total_lines:,} lines, {file_size:,} bytes]"
                f"\n[Shown: lines {start + 1}-{next_offset} of {total_lines:,}]"
                f"\n[To continue: read_file(\"{file_path}\", offset={next_offset})]"
                f"\n[Tip: read each segment, summarize it, then combine summaries]"
            )
        elif end < total_lines:
            result += (
                f"\n... ({total_lines - end:,} more lines)"
                f"\n[To continue: read_file(\"{file_path}\", offset={end})]"
            )

        return result

    except UnicodeDecodeError:
        return f"[Error] Cannot read file (binary or unknown encoding): {file_path}"
    except Exception as e:
        return f"[Error] Failed to read file: {str(e)}"


@tool
def write_file(file_path: str, content: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Write content to a file.

    Use this to:
    - Save generated content
    - Create new files
    - Modify existing files

    Args:
        file_path: Path to the file (absolute or relative to working directory)
        content: Content to write to the file
    """
    path = resolve_path(file_path, runtime.context.working_directory)

    try:
        # 确保父目录存在
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding="utf-8")
        return f"[Success] File written: {path}"

    except Exception as e:
        return f"[Error] Failed to write file: {str(e)}"


@tool
def glob(pattern: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Find files matching a glob pattern.

    Use this to:
    - Find files by name pattern (e.g., "**/*.py" for all Python files)
    - List files in a directory with wildcards
    - Discover project structure

    Args:
        pattern: Glob pattern (e.g., "**/*.py", "src/**/*.ts", "*.md")
    """
    cwd = runtime.context.working_directory

    try:
        # 使用 Path.glob 进行匹配
        matches = sorted(cwd.glob(pattern))

        if not matches:
            return f"No files matching pattern: {pattern}"

        # 限制返回数量
        max_results = 100
        result_lines = []

        for path in matches[:max_results]:
            try:
                rel_path = path.relative_to(cwd)
                result_lines.append(str(rel_path))
            except ValueError:
                result_lines.append(str(path))

        result = "\n".join(result_lines)

        if len(matches) > max_results:
            result += f"\n... and {len(matches) - max_results} more files"

        return f"[OK]\n\n{result}"

    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
def grep(pattern: str, path: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    Search for a pattern in files.

    Use this to:
    - Find code containing specific text or regex
    - Search for function/class definitions
    - Locate usages of variables or imports

    Args:
        pattern: Regular expression pattern to search for
        path: File or directory path to search in (use "." for current directory)
    """
    cwd = runtime.context.working_directory
    search_path = resolve_path(path, cwd)

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"[FAILED] Invalid regex pattern: {e}"

    results = []
    max_results = 50
    files_searched = 0

    try:
        if search_path.is_file():
            files = [search_path]
        else:
            # 搜索所有文本文件，排除常见的二进制/隐藏目录
            files = []
            for p in search_path.rglob("*"):
                if p.is_file():
                    # 排除隐藏文件和常见的非代码目录
                    parts = p.parts
                    if any(part.startswith(".") or part in ("node_modules", "__pycache__", ".git", "venv", ".venv") for part in parts):
                        continue
                    files.append(p)

        for file_path in files:
            if len(results) >= max_results:
                break

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.split("\n")
                files_searched += 1

                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        try:
                            rel_path = file_path.relative_to(cwd)
                        except ValueError:
                            rel_path = file_path
                        results.append(f"{rel_path}:{line_num}: {line.strip()[:100]}")

                        if len(results) >= max_results:
                            break

            except (UnicodeDecodeError, PermissionError, IsADirectoryError):
                continue

        if not results:
            return f"No matches found for pattern: {pattern} (searched {files_searched} files)"

        output = "\n".join(results)
        if len(results) >= max_results:
            output += f"\n... (truncated, showing first {max_results} matches)"

        return f"[OK]\n\n{output}"

    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
def edit(
    file_path: str,
    old_string: str,
    new_string: str,
    runtime: ToolRuntime[SkillAgentContext]
) -> str:
    """
    Edit a file by replacing text.

    Use this to:
    - Modify existing code
    - Fix bugs by replacing incorrect code
    - Update configuration values

    The old_string must match exactly (including whitespace/indentation).
    For safety, the old_string must be unique in the file.

    Args:
        file_path: Path to the file to edit
        old_string: The exact text to find and replace
        new_string: The text to replace it with
    """
    path = resolve_path(file_path, runtime.context.working_directory)

    if not path.exists():
        return f"[FAILED] File not found: {file_path}"

    if not path.is_file():
        return f"[FAILED] Not a file: {file_path}"

    try:
        content = path.read_text(encoding="utf-8")

        # 检查 old_string 是否存在
        count = content.count(old_string)

        if count == 0:
            return f"[FAILED] String not found in file. Make sure the text matches exactly including whitespace."

        if count > 1:
            return f"[FAILED] String appears {count} times in file. Please provide more context to make it unique."

        # 执行替换
        new_content = content.replace(old_string, new_string, 1)
        path.write_text(new_content, encoding="utf-8")

        # 计算变化的行数
        old_lines = len(old_string.split("\n"))
        new_lines = len(new_string.split("\n"))

        return f"[OK]\n\nEdited {path.name}: replaced {old_lines} lines with {new_lines} lines"

    except UnicodeDecodeError:
        return f"[FAILED] Cannot edit file (binary or unknown encoding): {file_path}"
    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
def list_dir(path: str, runtime: ToolRuntime[SkillAgentContext]) -> str:
    """
    List contents of a directory.

    Use this to:
    - Explore directory structure
    - See what files exist in a folder
    - Check if files/folders exist

    Args:
        path: Directory path (use "." for current directory)
    """
    dir_path = resolve_path(path, runtime.context.working_directory)

    if not dir_path.exists():
        return f"[FAILED] Directory not found: {path}"

    if not dir_path.is_dir():
        return f"[FAILED] Not a directory: {path}"

    try:
        entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

        result_lines = []
        for entry in entries[:100]:  # 限制数量
            if entry.is_dir():
                result_lines.append(f"📁 {entry.name}/")
            else:
                # 显示文件大小
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size // 1024}KB"
                else:
                    size_str = f"{size // (1024 * 1024)}MB"
                result_lines.append(f"   {entry.name} ({size_str})")

        if len(entries) > 100:
            result_lines.append(f"... and {len(entries) - 100} more entries")

        return f"[OK]\n\n{chr(10).join(result_lines)}"

    except PermissionError:
        return f"[FAILED] Permission denied: {path}"
    except Exception as e:
        return f"[FAILED] {str(e)}"


ALL_TOOLS = [load_skill, bash, read_file, write_file, glob, grep, edit, list_dir]
