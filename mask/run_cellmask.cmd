@echo off
chcp 65001 >nul
setlocal

REM ============================================================
REM  CellMask — Cellpose 分割 + 人工筛选
REM  请先确保 conda 环境 mem 已安装 cellpose / tifffile 等
REM ============================================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM 尝试激活 mem 环境
where conda >nul 2>&1
if %ERRORLEVEL%==0 (
    call conda activate mem
    if errorlevel 1 (
        echo [警告] conda activate mem 失败，尝试用当前 Python
    ) else (
        echo 已激活 conda 环境: mem
    )
) else (
    echo [提示] 未找到 conda 命令。若已在 mem 环境中可忽略。
)

echo.
echo 用法示例:
echo   run_cellmask.cmd "D:\课题同步\实验结果图\共定位-荧光\3B_7.20"
echo   run_cellmask.cmd "D:\...\3B_7.20" --limit 3 --model cyto3
echo   run_cellmask.cmd "D:\...\3B_7.20" --scan-only
echo.

if "%~1"=="" (
    echo 请把实验目录作为第一个参数传入。
    echo.
    python -m cellmask -h
    exit /b 1
)

python -m cellmask --input-dir %*
set "EC=%ERRORLEVEL%"
echo.
echo 退出码: %EC%
exit /b %EC%
