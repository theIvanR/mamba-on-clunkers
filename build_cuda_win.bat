@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM make sure you have git cloned the repo to source first

REM Set up Paths
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (echo [ERROR] vcvarsall.bat failed & exit /b 1)
set "PY=C:\Users\%USERNAME%\miniconda3\envs\py311_main\python.exe"

REM Set up project paths
set "PROJECT_DIR=C:\Users\%USERNAME%\source\mamba-on-clunkers"
cd /d "%PROJECT_DIR%" || (echo [build] cd to %PROJECT_DIR% failed & exit /b 1)

REM tell distutils/torch we already activated the VC env 
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"


echo Get pytorch info 
echo ============================================================

"%PY%" -c "import sys, torch; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count())"
if errorlevel 1 (echo [ERROR] PyTorch import / CUDA check failed. & exit /b 1)

REM verify the toolchain that setup.py will hard-require 
where cl   >nul 2>nul || (echo [ERROR] cl   not found on PATH. & exit /b 1)
where nvcc >nul 2>nul || (echo [ERROR] nvcc not found on PATH. & exit /b 1)
nvcc --version | findstr /C:"release"
if errorlevel 1 (echo [ERROR] nvcc --version failed. & exit /b 1)

REM detect compiled CUDA archs -> TORCH_CUDA_ARCH_LIST 
set "ARCH_FILE=%TEMP%\torch_arch_detect_%RANDOM%.tmp"

"%PY%" -c "import torch, re, sys; out=[]; [out.append(f'{m.group(1)[:-1]}.{m.group(1)[-1]}{m.group(2)}') if (m := re.match(r'sm_(\d+)([a-z]*)', a)) else sys.exit('bad arch: '+a) for a in torch.cuda.get_arch_list()]; open(r'%ARCH_FILE%','w').write(';'.join(out)+'\n')"
if errorlevel 1 (echo [ERROR] Architecture detection failed. & del "%ARCH_FILE%" 2>nul & exit /b 1)

set "TORCH_CUDA_ARCH_LIST="
set /p "TORCH_CUDA_ARCH_LIST="<"%ARCH_FILE%"
del "%ARCH_FILE%" 2>nul
if not defined TORCH_CUDA_ARCH_LIST (echo [ERROR] Detected architecture list is empty. & exit /b 1)

echo [build] TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%


echo Building
echo ============================================================

REM build in place install 
REM "%PY%" -m pip install -e . --no-build-isolation --no-deps -v
REM if errorlevel 1 (echo [ERROR] pip install failed. & exit /b 1)

REM build a wheel 
"%PY%" -m pip wheel . --no-build-isolation --no-deps --no-cache-dir -w dist -v
if errorlevel 1 (echo [ERROR] Wheel build failed. & exit /b 1)

echo [build] SUCCESS