.PHONY: test test_cov
test:
	@PYTHONPATH=. pytest tests

test_cov:
	@PYTHONPATH=. pytest --cov --cov-report term-missing tests/


solver:
	@python3 -m optim.solver


vt_solver:
	@python3 -m optim.variable_threshold_solver


sim:
	@python3 -m optim.sim_measure_eval