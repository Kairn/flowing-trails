.PHONY: index deploy eval eval-full

index:
	@echo "Run: modal run retrieval/index_corpus.py"

deploy:
	@echo "Run: modal deploy orchestrator/app.py"

eval:
	@echo "Run: pytest eval/ -m ci -x"

eval-full:
	@echo "Run: pytest eval/ --model haiku"
