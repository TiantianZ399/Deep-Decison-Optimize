PYTHON ?= python
PDFLATEX ?= pdflatex

.PHONY: all experiment tex clean

all: tex

experiment:
	$(PYTHON) scripts/run_benchmark_comparison.py

tex:
	$(PDFLATEX) -interaction=nonstopmode main.tex
	$(PDFLATEX) -interaction=nonstopmode main.tex

clean:
	$(PYTHON) -c "from pathlib import Path; patterns = ['*.aux', '*.bbl', '*.bcf', '*.blg', '*.fdb_latexmk', '*.fls', '*.log', '*.out', '*.run.xml', '*.synctex.gz', '*.toc', '*.nav', '*.snm', '*.vrb', 'main.pdf']; [p.unlink() for pattern in patterns for p in Path('.').glob(pattern) if p.is_file()]"
