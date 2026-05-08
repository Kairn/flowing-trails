.PHONY: index embed deploy deploy-all corpus corpus-prompts corpus-manifest eval eval-full samples

embed:
	@echo "Run: modal run retrieval/embed_corpus.py"

index:
	@echo "Run: modal run retrieval/index_corpus.py"

deploy:
	@echo "Run: modal deploy orchestrator/app.py"

deploy-all:
	modal deploy musicgen_service/app.py
	modal deploy orchestrator/app.py

corpus-prompts:
	python retrieval/generate_corpus_prompts.py

corpus: corpus-prompts
	modal run retrieval/generate_corpus.py

corpus-manifest: corpus-prompts
	modal run retrieval/update_corpus_manifest.py

eval:
	@echo "Run: pytest eval/ -m ci -x"

eval-full:
	@echo "Run: pytest eval/ --model haiku"

samples:
ifndef COMPOSE_URL
	$(error COMPOSE_URL is required. Run: COMPOSE_URL=https://... make samples)
endif
	python scripts/generate_samples.py $(COMPOSE_URL)
