#!/usr/bin/env bash
# =============================================================================
# start.sh — 一键启动 Skills Agent 前后端开发服务器 (Linux / macOS)
#
# 功能：
#   1. 检测并处理端口占用（8000, 5173）
#   2. 安装后端 Python 依赖 (uv sync)
#   3. 安装前端 Node 依赖 (npm install)
#   4. 启动 FastAPI 后端 (SSE 流式接口)
#   5. 启动 Vite 前端开发服务器 (React)
#   6. Ctrl+C 时自动清理两个子进程
#
# 环境变量（均可选，有默认值）：
#   BACKEND_PORT      — 后端端口，默认 8000
#   FRONTEND_PORT     — 前端端口，默认 5173
#   VITE_API_BASE_URL — 前端连接后端的地址，默认 http://127.0.0.1:{BACKEND_PORT}
# =============================================================================

# -e: 任何命令失败立即退出
# -u: 使用未定义变量时报错
# -o pipefail: 管道中任何一个命令失败，整个管道失败
set -euo pipefail

# ── 1. 确定项目根目录和前端目录 ──────────────────────────────────────────────
# ${BASH_SOURCE[0]} 是当前脚本的路径，通过 cd + pwd 获取绝对路径
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$ROOT_DIR/web"

# ── 2. 读取端口配置（使用环境变量或默认值）──────────────────────────────────
BACKEND_PORT="${BACKEND_PORT:-8000}"          # 后端默认 8000
FRONTEND_PORT="${FRONTEND_PORT:-5173}"        # 前端默认 5173（Vite 默认端口）
API_BASE_URL="${VITE_API_BASE_URL:-http://127.0.0.1:${BACKEND_PORT}}"  # 前端请求后端的地址

# ── 3. 初始化子进程 PID 变量（用于清理）───────────────────────────────────────
BACKEND_PID=""
FRONTEND_PID=""

# ── 4. 工具函数：检查必要的命令是否存在 ──────────────────────────────────────
require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[start] Missing required command: $cmd"
    exit 1
  fi
}

# ── 4b. 工具函数：检测并终止占用指定端口的进程 ────────────────────────────────
check_and_kill_port() {
  local port="$1"
  local service="$2"
  local pid

  # lsof 方式（macOS / 大部分 Linux）
  if command -v lsof >/dev/null 2>&1; then
    pid=$(lsof -ti :"$port" 2>/dev/null || true)
  # ss 方式（部分 Linux 无 lsof）
  elif command -v ss >/dev/null 2>&1; then
    pid=$(ss -tlnp "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)
  fi

  if [[ -n "$pid" ]]; then
    echo "[start] Port $port ($service) is occupied by PID $pid, terminating..."
    kill "$pid" 2>/dev/null || true
    sleep 1
    # 如果还没退出，强制终止
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
    echo "[start] Port $port freed."
  fi
}

# 工具函数：检查端口占用（仅提示，不终止）
check_port_info() {
  local port="$1"
  local service="$2"
  local pid

  if command -v lsof >/dev/null 2>&1; then
    pid=$(lsof -ti :"$port" 2>/dev/null || true)
  elif command -v ss >/dev/null 2>&1; then
    pid=$(ss -tlnp "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)
  fi

  if [[ -n "$pid" ]]; then
    echo "[start] Info: Port $port ($service) is in use by PID $pid (expected if already running)."
  fi
}

# ── 5. 清理函数：退出时关闭所有子进程 ────────────────────────────────────────
cleanup() {
  # 先移除 trap，避免递归调用
  trap - EXIT INT TERM

  # 如果前端进程还活着，发送 kill 信号
  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi

  # 如果后端进程还活着，发送 kill 信号
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi

  # 等待两个子进程结束（忽略错误）
  wait "$FRONTEND_PID" "$BACKEND_PID" 2>/dev/null || true
}

# 注册清理函数：脚本退出(EXIT)、Ctrl+C(INT)、终止信号(TERM) 时触发
trap cleanup EXIT INT TERM

# ── 6. 前置检查 ──────────────────────────────────────────────────────────────
require_cmd uv    # Python 包管理器 (必须预先安装)
require_cmd npm   # Node 包管理器 (必须预先安装)

# 检查前端目录是否存在
if [[ ! -d "$WEB_DIR" ]]; then
  echo "[start] Frontend directory not found: $WEB_DIR"
  exit 1
fi

# ── 6b. 端口冲突检测和处理 ─────────────────────────────────────────────────
check_and_kill_port "$BACKEND_PORT" "Backend"
check_and_kill_port "$FRONTEND_PORT" "Frontend"
check_port_info 5432 "PostgreSQL"

# ── 7. 安装后端 Python 依赖 ──────────────────────────────────────────────────
echo "[start] Installing backend dependencies with uv sync..."
cd "$ROOT_DIR"
uv sync  # 根据 pyproject.toml 安装/同步所有依赖到虚拟环境

# ── 8. 安装前端 Node 依赖 ────────────────────────────────────────────────────
echo "[start] Installing frontend dependencies with npm install..."
cd "$WEB_DIR"
npm install  # 根据 package.json 安装前端依赖

# ── 9. 启动后端 FastAPI 服务（后台运行）──────────────────────────────────────
echo "[start] Starting backend on :$BACKEND_PORT ..."
cd "$ROOT_DIR"
SKILLS_WEB_HOST="0.0.0.0" \
SKILLS_WEB_PORT="$BACKEND_PORT" \
SKILLS_WEB_RELOAD="true" \
uv run langchain-skills-web &     # & 放入后台运行
BACKEND_PID=$!                    # 记录后端进程 PID

# ── 10. 启动前端 Vite 开发服务器（后台运行）──────────────────────────────────
echo "[start] Starting frontend on :$FRONTEND_PORT ..."
cd "$WEB_DIR"
VITE_API_BASE_URL="$API_BASE_URL" \
npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" &  # & 放入后台运行
FRONTEND_PID=$!                                           # 记录前端进程 PID

# ── 11. 打印访问地址 ─────────────────────────────────────────────────────────
echo "[start] Backend:  http://127.0.0.1:$BACKEND_PORT"
echo "[start] Frontend: http://127.0.0.1:$FRONTEND_PORT"
echo "[start] Press Ctrl+C to stop both services."

# ── 12. 监控循环：等待任一服务退出，然后关闭另一个 ────────────────────────────
SERVICE_EXIT_CODE=0
while true; do
  # 检查后端是否还活着
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    wait "$BACKEND_PID" || SERVICE_EXIT_CODE=$?
    break
  fi

  # 检查前端是否还活着
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    wait "$FRONTEND_PID" || SERVICE_EXIT_CODE=$?
    break
  fi

  sleep 1  # 每秒检查一次
done

# 任一服务退出后，cleanup 函数会被 EXIT trap 触发，关闭另一个服务
echo "[start] One service exited, shutting down..."
exit "$SERVICE_EXIT_CODE"
