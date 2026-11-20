@echo off
chcp 65001 >nul
setlocal

set "WORK_DIR=C:\Users\Q\Desktop\stock-predictor-main\stock-predictor-main"
set "GIT_DIR=C:\Users\Q\Desktop\py\stock-predictor(1)\stock-predictor"

if exist "%WORK_DIR%\stock_predictor.py" (
    echo [1/4] 同步最新源码与测试到 Git 仓库...
    copy /Y "%WORK_DIR%\stock_predictor.py" "%GIT_DIR%\stock_predictor.py" >nul
    if exist "%WORK_DIR%\tests\test_temporal_intraday.py" copy /Y "%WORK_DIR%\tests\test_temporal_intraday.py" "%GIT_DIR%\tests\test_temporal_intraday.py" >nul
)

cd /d "%GIT_DIR%"
echo [2/4] 检查 Git 变更...
git status -s
git add stock_predictor.py tests/ .github/ scripts/ README.md 2>nul

for /f "tokens=1-3 delims=/- " %%a in ("%date%") do set "D=%%a-%%b-%%c"
set "MSG=chore: sync and push updates (%D% %time:~0,5%)"

git diff --cached --quiet
if %errorlevel%==0 (
    echo [3/4] 没有新的代码改动需要提交，直接推送远端 main...
) else (
    echo [3/4] 提交改动: %MSG%
    git commit -m "%MSG%"
)

echo [4/4] 推送到 GitHub (origin main)...
git push origin main
if errorlevel 1 (
    echo [失败] git push 出错，请检查网络或认证状态。
    pause
    exit /b 1
)

echo.
echo [完成] 已一键推送至 GitHub origin/main!
echo        若要在 GitHub 云端一键推送手机/查看每日报表，可打开：
echo        https://github.com/Li-Ding-PhDL/stock-predictor/actions
pause
