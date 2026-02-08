@echo off
REM =============================================================================
REM start.bat — 一键启动 Skills Agent 前后端开发服务器 (Windows)
REM
REM 功能：
REM   1. 安装后端 Python 依赖 (uv sync)
REM   2. 安装前端 Node 依赖 (npm install)
REM   3. 启动 FastAPI 后端 (SSE 流式接口)
REM   4. 启动 Vite 前端开发服务器 (React)
REM   5. Ctrl+C 时提示用户手动关闭窗口（Windows 无法像 Linux 一样自动清理后台进程）
REM
REM 环境变量（均可选，有默认值）：
REM   BACKEND_PORT      — 后端端口，默认 8000
REM   FRONTEND_PORT     — 前端端口，默认 5173
REM   VITE_API_BASE_URL — 前端连接后端的地址，默认 http://127.0.0.1:{BACKEND_PORT}
REM
REM 使用方法：
REM   双击 start.bat，或在命令行中执行：
REM     start.bat
REM     set BACKEND_PORT=9000 && start.bat
REM =============================================================================

REM ── 启用延迟变量扩展 ───────────────────────────────────────────────────────
REM   默认 cmd 在解析整行时就展开 %VAR%，enabledelayedexpansion 允许用 !VAR!
REM   在运行时动态取值，这对 if/for 块中修改变量是必须的
setlocal enabledelayedexpansion

REM ── 1. 确定项目根目录和前端目录 ────────────────────────────────────────────
REM   %~dp0 展开为当前脚本所在目录的绝对路径（含尾部 \）
set "ROOT_DIR=%~dp0"
REM   去掉尾部的 \ ，保持路径格式统一
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
set "WEB_DIR=%ROOT_DIR%\web"

REM ── 2. 读取端口配置（使用环境变量或默认值）─────────────────────────────────
REM   如果 BACKEND_PORT 未定义，设置为默认值 8000
if not defined BACKEND_PORT set "BACKEND_PORT=8000"
REM   如果 FRONTEND_PORT 未定义，设置为默认值 5173（Vite 默认端口）
if not defined FRONTEND_PORT set "FRONTEND_PORT=5173"
REM   如果 VITE_API_BASE_URL 未定义，根据后端端口拼接默认地址
if not defined VITE_API_BASE_URL set "VITE_API_BASE_URL=http://127.0.0.1:%BACKEND_PORT%"

REM ── 3. 检查必要的命令是否存在 ──────────────────────────────────────────────
REM   where 命令类似 Linux 的 which，检查命令是否在 PATH 中
where uv >nul 2>&1
if errorlevel 1 (
    echo [start] Missing required command: uv
    echo [start] Please install uv first: https://docs.astral.sh/uv/
    pause
    exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
    echo [start] Missing required command: npm
    echo [start] Please install Node.js first: https://nodejs.org/
    pause
    exit /b 1
)

REM ── 4. 检查前端目录是否存在 ────────────────────────────────────────────────
if not exist "%WEB_DIR%" (
    echo [start] Frontend directory not found: %WEB_DIR%
    pause
    exit /b 1
)

REM ── 5. 安装后端 Python 依赖 ────────────────────────────────────────────────
echo [start] Installing backend dependencies with uv sync...
REM   pushd 切换目录并记住原目录（popd 可返回），比 cd 更安全
pushd "%ROOT_DIR%"
REM   uv sync 根据 pyproject.toml 安装/同步所有 Python 依赖到虚拟环境
uv sync
if errorlevel 1 (
    echo [start] Failed to install backend dependencies
    popd
    pause
    exit /b 1
)
popd

REM ── 6. 安装前端 Node 依赖 ──────────────────────────────────────────────────
echo [start] Installing frontend dependencies with npm install...
pushd "%WEB_DIR%"
REM   npm install 根据 package.json 安装前端依赖到 node_modules/
npm install
if errorlevel 1 (
    echo [start] Failed to install frontend dependencies
    popd
    pause
    exit /b 1
)
popd

REM ── 7. 启动后端 FastAPI 服务（新窗口运行）──────────────────────────────────
REM   Windows 没有 Linux 的 & 后台运行机制，改用 start 命令打开新窗口
REM   /B 参数表示不打开新窗口（在同一窗口后台运行），但这里我们需要新窗口
REM   以便前后端日志分开显示
REM
REM   设置环境变量后启动后端：
REM     SKILLS_WEB_HOST=0.0.0.0  — 监听所有网络接口
REM     SKILLS_WEB_PORT          — 监听端口
REM     SKILLS_WEB_RELOAD=true   — 启用热重载（代码修改自动重启）
echo [start] Starting backend on :%BACKEND_PORT% ...
start "[Skills Agent] Backend" cmd /c "cd /d "%ROOT_DIR%" && set SKILLS_WEB_HOST=0.0.0.0 && set SKILLS_WEB_PORT=%BACKEND_PORT% && set SKILLS_WEB_RELOAD=true && uv run langchain-skills-web"

REM ── 8. 启动前端 Vite 开发服务器（新窗口运行）───────────────────────────────
REM   设置 VITE_API_BASE_URL 环境变量让前端知道后端地址
REM   --host 0.0.0.0  — 监听所有网络接口（允许局域网访问）
REM   --port           — 指定前端端口
echo [start] Starting frontend on :%FRONTEND_PORT% ...
start "[Skills Agent] Frontend" cmd /c "cd /d "%WEB_DIR%" && set VITE_API_BASE_URL=%VITE_API_BASE_URL% && npm run dev -- --host 0.0.0.0 --port %FRONTEND_PORT%"

REM ── 9. 打印访问地址和使用说明 ──────────────────────────────────────────────
echo.
echo ============================================================
echo [start] Backend:  http://127.0.0.1:%BACKEND_PORT%
echo [start] Frontend: http://127.0.0.1:%FRONTEND_PORT%
echo ============================================================
echo.
echo [start] Two new windows have been opened for backend and frontend.
echo [start] Close those windows to stop the services.
echo [start] Or press any key in this window to exit the launcher.
echo.

REM   pause 让启动窗口保持打开，用户可以看到上面的信息
REM   前后端运行在各自的窗口中，关闭对应窗口即可停止服务
pause
