.PHONY: test test_cov
test:
	@PYTHONPATH=. pytest tests

test_cov:
	@PYTHONPATH=. pytest --cov --cov-report term-missing tests/


basic_solver:
	@python3 -m optim.basic_solver


t_solver:
	@python3 -m optim.threshold_approx_solver


sim:
	@python3 -m optim.similarity_measure_eval