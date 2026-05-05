.PHONY: index deploy eval eval-full samples

index:
	@echo "Run: modal run retrieval/index_corpus.py"

deploy:
	@echo "Run: modal deploy orchestrator/app.py"

eval:
	@echo "Run: pytest eval/ -m ci -x"

eval-full:
	@echo "Run: pytest eval/ --model haiku"

samples:
ifndef COMPOSE_URL
	$(error COMPOSE_URL is required. Run: COMPOSE_URL=https://... make samples)
endif
	python scripts/generate_samples.py $(COMPOSE_URL)
