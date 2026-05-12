.PHONY: index embed upsert deploy deploy-all corpus corpus-prompts corpus-manifest eval eval-full samples calibrate calibrate-analyze

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
	python retrieval/generate_corpus_prompts.py

corpus: corpus-prompts
	modal run retrieval/generate_corpus.py

corpus-manifest: corpus-prompts
	modal run retrieval/update_corpus_manifest.py

# Not yet implemented (M4)
eval:
	@echo "TODO: pytest eval/ -m ci -x"

eval-full:
	@echo "TODO: pytest eval/ --model haiku"

samples:
ifndef COMPOSE_URL
	$(error COMPOSE_URL is required. Run: COMPOSE_URL=https://... make samples)
endif
	python scripts/generate_samples.py $(COMPOSE_URL)

calibrate:
ifndef COMPOSE_URL
	$(error COMPOSE_URL is required. Run: COMPOSE_URL=https://... make calibrate)
endif
	python scripts/run_calibration.py $(COMPOSE_URL)

calibrate-analyze:
	python scripts/analyze_thresholds.py
