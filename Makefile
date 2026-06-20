.PHONY: index embed upsert deploy deploy-all corpus corpus-prompts corpus-manifest eval eval-full regen-golden samples calibrate calibrate-analyze ab-select ab-reset ab-upload ab-index ab-test ab-analyze train-playlists train-download train-env train-prep train-describe train-manifest train-validate train-upload train-poc train-full train-full-h200 train-list train-clean train-export train-promote checkpoint-eval

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

regen-golden:
ifndef COMPOSE_URL
	$(error COMPOSE_URL is required. Run: COMPOSE_URL=https://... make regen-golden)
endif
	.venv/bin/python scripts/regen_golden.py $(COMPOSE_URL)

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

# ── Retrieval A/B Test ──────────────────────────────────────────────────────

ab-select:
	.venv/bin/python scripts/select_ab_tracks.py

ab-reset:
	.venv/bin/python scripts/reset_qdrant_collection.py

ab-upload:
	modal volume put flowing-trails-corpus eval/ab_tracks/ /ab_tracks/ --force
	modal volume put flowing-trails-corpus eval/ab_corpus_manifest.json /corpus_manifest.json --force

ab-index:
	modal run retrieval/embed_corpus.py && modal run retrieval/index_corpus.py

ab-test:
ifndef COMPOSE_URL
	$(error COMPOSE_URL is required. Run: COMPOSE_URL=https://... make ab-test)
endif
	.venv/bin/python scripts/run_ab_test.py $(COMPOSE_URL)

ab-analyze:
	.venv/bin/python scripts/analyze_ab_test.py

# ── Training ─────────────────────────────────────────────────────────────────

train-playlists:
	.venv/bin/python training/prep_playlists.py

train-download:
	.venv/bin/python training/prep_from_exports.py $(ARGS)

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

train-full:
	modal run training/app.py::TrainingRunner.train --config-name full_large

train-full-h200:
	FT_GPU=h200 modal run training/app.py::TrainingRunner.train --config-name full_large_h200

train-list:
	modal volume ls flowing-trails-training xps/

train-clean:
	modal volume rm flowing-trails-training xps/ -r

train-export:
ifdef EPOCH
	modal run training/app.py::TrainingUtils.export_and_push --xp-sig $(XP_SIG) --tag $(TAG) --epoch $(EPOCH)
else
	modal run training/app.py::TrainingUtils.export_and_push --xp-sig $(XP_SIG) --tag $(TAG)
endif

train-promote:
ifndef TAG
	$(error TAG is required. Run: TAG=poc-small-v1 make train-promote)
endif
	modal secret create --force --from-dotenv .env flowing-trails-secrets MODEL_TAG=$(TAG)
	@echo "MODEL_TAG set to '$(TAG)'. Run 'make deploy-all' to pick it up."

checkpoint-eval:
ifndef XP_SIG
	$(error XP_SIG is required. Run: XP_SIG=48bdbba4 EPOCHS=3,6,10 make checkpoint-eval)
endif
ifndef EPOCHS
	$(error EPOCHS is required. Run: XP_SIG=48bdbba4 EPOCHS=3,6,10 make checkpoint-eval)
endif
	modal run training/checkpoint_eval.py::CheckpointEvaluator.evaluate_all --xp-sig $(XP_SIG) --epochs $(EPOCHS)

train-push:
	$(TRAIN_PYTHON) training/push_to_hub.py $(ARGS)
