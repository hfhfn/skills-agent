@echo off
REM =============================================================================
REM start.bat - One-click launcher for Skills Agent (Windows)
REM
REM Environment variables (all optional, have defaults):
REM   BACKEND_PORT      - Backend port, default 8000
REM   FRONTEND_PORT     - Frontend port, default 5173
REM   VITE_API_BASE_URL - Frontend-to-backend URL, default http://127.0.0.1:{BACKEND_PORT}
REM =============================================================================

REM == Phase 1: Resolve PATH (needs delayed expansion) ==
setlocal enabledelayedexpansion

set "PATH=%USERPROFILE%\.local\bin;!PATH!"
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "PATH=%%B;!PATH!"

where uv >nul 2>&1
if errorlevel 1 (
    where conda >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%B in ('conda info --base 2^>nul') do (
            if exist "%%B\envs" (
                for /d %%E in ("%%B\envs\*") do (
                    if exist "%%E\Scripts\uv.exe" set "PATH=%%E\Scripts;%%E;%%E\Library\bin;!PATH!"
                )
            )
        )
    )
)

REM Export resolved PATH to global scope
endlocal & set "PATH=%PATH%"

REM == Phase 2: Main logic ==

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
set "WEB_DIR=%ROOT_DIR%\web"

if not defined BACKEND_PORT set "BACKEND_PORT=8000"
if not defined FRONTEND_PORT set "FRONTEND_PORT=5173"
if not defined VITE_API_BASE_URL set "VITE_API_BASE_URL=http://127.0.0.1:%BACKEND_PORT%"

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

if not exist "%WEB_DIR%" (
    echo [start] Frontend directory not found: %WEB_DIR%
    pause
    exit /b 1
)

REM -- Port conflict detection
call :check_and_kill_port %BACKEND_PORT% Backend
call :check_and_kill_port %FRONTEND_PORT% Frontend
call :check_port_info 5432 PostgreSQL

REM -- Install backend dependencies
echo [start] Installing backend dependencies...
pushd "%ROOT_DIR%"
uv sync
if errorlevel 1 (
    echo [start] Failed to install backend dependencies
    popd
    pause
    exit /b 1
)
popd

REM -- Install frontend dependencies
REM    IMPORTANT: use "call npm" not "npm" - npm.cmd is a batch file,
REM    calling it without "call" transfers control and never returns.
echo [start] Installing frontend dependencies...
pushd "%WEB_DIR%"
call npm install
if errorlevel 1 (
    echo [start] Failed to install frontend dependencies
    popd
    pause
    exit /b 1
)
popd

REM -- Write launcher scripts to temp (avoids nested-quote issues)
set "BE_LAUNCHER=%TEMP%\skills_backend_start.bat"
set "FE_LAUNCHER=%TEMP%\skills_frontend_start.bat"

echo @echo off> "%BE_LAUNCHER%"
echo cd /d "%ROOT_DIR%">> "%BE_LAUNCHER%"
echo set SKILLS_WEB_HOST=0.0.0.0>> "%BE_LAUNCHER%"
echo set SKILLS_WEB_PORT=%BACKEND_PORT%>> "%BE_LAUNCHER%"
echo set SKILLS_WEB_RELOAD=true>> "%BE_LAUNCHER%"
echo uv run langchain-skills-web>> "%BE_LAUNCHER%"
echo echo.>> "%BE_LAUNCHER%"
echo echo [start] Backend exited. Press any key to close.>> "%BE_LAUNCHER%"
echo pause ^>nul>> "%BE_LAUNCHER%"

echo @echo off> "%FE_LAUNCHER%"
echo cd /d "%WEB_DIR%">> "%FE_LAUNCHER%"
echo set VITE_API_BASE_URL=%VITE_API_BASE_URL%>> "%FE_LAUNCHER%"
echo call npm run dev -- --host 0.0.0.0 --port %FRONTEND_PORT%>> "%FE_LAUNCHER%"
echo echo.>> "%FE_LAUNCHER%"
echo echo [start] Frontend exited. Press any key to close.>> "%FE_LAUNCHER%"
echo pause ^>nul>> "%FE_LAUNCHER%"

REM -- Launch services in new windows
echo [start] Starting backend on :%BACKEND_PORT% ...
start "[Skills Agent] Backend" "%BE_LAUNCHER%"

echo [start] Starting frontend on :%FRONTEND_PORT% ...
start "[Skills Agent] Frontend" "%FE_LAUNCHER%"

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
pause
exit /b 0

REM ============================================================================
:check_and_kill_port
setlocal enabledelayedexpansion
set "PORT=%~1"
set "SERVICE=%~2"
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING" 2^>nul') do (
    set "PID=%%P"
    if !PID! NEQ 0 (
        echo [start] Port %PORT% [%SERVICE%] is occupied by PID !PID!, terminating...
        taskkill /F /PID !PID! >nul 2>&1
        timeout /t 1 /nobreak >nul 2>&1
    )
)
endlocal
goto :eof

REM ============================================================================
:check_port_info
setlocal enabledelayedexpansion
set "PORT=%~1"
set "SERVICE=%~2"
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING" 2^>nul') do (
    set "PID=%%P"
    if !PID! NEQ 0 (
        echo [start] Info: Port %PORT% [%SERVICE%] is in use by PID !PID!.
    )
)
endlocal
goto :eof
