.PHONY: install data eval test demo serve all clean
install:
	pip install -r requirements.txt
data:
	python scripts/make_data.py
eval:
	python scripts/run_eval.py
test:
	python -m pytest -q
demo:
	python scripts/make_demo.py
serve:
	ER_DATA=data/demo uvicorn er.api:app --host 127.0.0.1 --port 8000
all: data eval test
clean:
	rm -rf er/__pycache__ tests/__pycache__ .pytest_cache
