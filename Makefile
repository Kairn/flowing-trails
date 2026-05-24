.PHONY: index embed upsert deploy deploy-all corpus corpus-prompts corpus-manifest eval eval-full samples calibrate calibrate-analyze train-env train-prep train-describe train-manifest train-validate train-upload train-poc train-list train-clean train-export train-promote

TRAIN_PYTHON = training/.venv/bin/python
TRAIN_PIP = training/.venv/bin/pip

embed:
	modal run retrieval/embed_corpus.py

upsert:
	modal run retrieval/index_corpus.py

index:
	modal run retrieval/embed_corpus.py && modal run retrieval/index_corpus.py

deploy:
	modal deploy orchestrator/app.py

deploy-all:
	modal deploy musicgen_service/app.py && modal deploy orchestrator/app.py

corpus-prompts:
	.venv/bin/python retrieval/generate_corpus_prompts.py

corpus: corpus-prompts
	modal run retrieval/generate_corpus.py

corpus-manifest: corpus-prompts
	modal run retrieval/update_corpus_manifest.py

eval:
	.venv/bin/pytest eval/golden/test_ci_eval.py -v -x -m ci

eval-full:
ifndef COMPOSE_URL
	$(error COMPOSE_URL is required. Run: COMPOSE_URL=https://... make eval-full)
endif
	.venv/bin/python scripts/run_full_eval.py $(COMPOSE_URL)

samples:
ifndef COMPOSE_URL
	$(error COMPOSE_URL is required. Run: COMPOSE_URL=https://... make samples)
endif
	.venv/bin/python scripts/generate_samples.py $(COMPOSE_URL)

calibrate:
ifndef COMPOSE_URL
	$(error COMPOSE_URL is required. Run: COMPOSE_URL=https://... make calibrate)
endif
	.venv/bin/python scripts/run_calibration.py $(COMPOSE_URL)

calibrate-analyze:
	.venv/bin/python scripts/analyze_thresholds.py

# ── Training ─────────────────────────────────────────────────────────────────

train-env:
	python3.11 -m venv training/.venv
	$(TRAIN_PIP) install --upgrade pip
	$(TRAIN_PIP) install -r training/requirements.txt

train-prep:
	$(TRAIN_PYTHON) training/prep_normalize.py

train-describe:
	$(TRAIN_PYTHON) training/prep_describe.py

train-manifest:
	$(TRAIN_PYTHON) training/prep_manifest.py

train-validate:
	$(TRAIN_PYTHON) training/validate_dataset.py

train-upload:
	modal volume put flowing-trails-training training/prepared/ /data/

train-poc:
	modal run training/app.py::TrainingRunner.train --config-name poc_small

train-list:
	modal run training/app.py::TrainingRunner.list_checkpoints

train-clean:
	modal run training/app.py::TrainingRunner.clean_xps

train-export:
	modal run training/app.py::TrainingRunner.export_and_push --xp-sig $(XP_SIG) --tag $(TAG)

train-promote:
ifndef TAG
	$(error TAG is required. Run: TAG=poc-small-v1 make train-promote)
endif
	modal secret set flowing-trails-secrets MODEL_TAG=$(TAG)
	@echo "MODEL_TAG set to '$(TAG)'. Run 'make deploy-all' to pick it up."

train-push:
	$(TRAIN_PYTHON) training/push_to_hub.py $(ARGS)
