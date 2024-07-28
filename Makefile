.PHONY: test test_cov
test:
	@PYTHONPATH=. pytest tests

test_cov:
	@PYTHONPATH=. pytest --cov --cov-report term-missing tests/


solver:
	@python3 -m al.solver


vt_solver:
	@python3 -m al.variable_threshold_solver