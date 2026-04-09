PYTHON ?= python

all: benchmark

benchmark:
	$(PYTHON) scripts/run_benchmark_comparison.py --out_dir .

clean:
	rm -f tables/*.csv tables/*.tex figures/*.png summary_zh.md
