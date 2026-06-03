@echo off
REM ═══════════════════════════════════════════════════════════════
REM  Automated chunking pipeline — run unattended overnight.
REM  Switches conda environments automatically between stages.
REM ═══════════════════════════════════════════════════════════════

echo ══════════════════════════════════════════════════════════════
echo  Stage 1/3: Classify pages (rag env)
echo ══════════════════════════════════════════════════════════════
call conda activate rag
python chunking.py classify
if errorlevel 1 (
    echo ERROR: Classification failed. Aborting.
    exit /b 1
)

echo.
echo ══════════════════════════════════════════════════════════════
echo  Stage 2/3: Extract table pages with Docling (docling_test env)
echo ══════════════════════════════════════════════════════════════
call conda activate docling_test
python chunking.py extract-tables
if errorlevel 1 (
    echo ERROR: Docling extraction failed. Aborting.
    exit /b 1
)

echo.
echo ══════════════════════════════════════════════════════════════
echo  Stage 3/3: Chunk prose + tables (rag env)
echo ══════════════════════════════════════════════════════════════
call conda activate rag
python chunking.py chunk
if errorlevel 1 (
    echo ERROR: Chunking failed. Aborting.
    exit /b 1
)

echo.
echo ══════════════════════════════════════════════════════════════
echo  Pipeline complete.
echo ══════════════════════════════════════════════════════════════