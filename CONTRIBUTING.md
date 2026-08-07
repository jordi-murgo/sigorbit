# Contributing

Thanks for improving SigOrbit.

1. Open an issue before large architecture or preprocessing changes.
2. Create a focused branch and add tests for behavior changes.
3. Install development dependencies with `pip install -e '.[api,dev]'`.
4. Run `ruff check .` and `pytest`.
5. Do not commit signature images, embeddings tied to people, databases, API
   secrets or third-party model artifacts.
6. State the origin and license of copied code, datasets or weights.
7. Never change the model/preprocess contract without a new version identifier
   and migration notes.

By submitting a contribution, you agree that your contribution is licensed
under the repository's MIT License and that you have the right to provide it.
