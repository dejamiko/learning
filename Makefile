.PHONY: test test_cov
test:
	@PYTHONPATH=. pytest tests

test_cov:
	@PYTHONPATH=. pytest --cov --cov-report term-missing tests/


b_solver:
	@python3 -m optim.basic_solver


t_solver:
	@python3 -m optim.threshold_approx_solver


a_solver:
	@python3 -m optim.affine_approx_solver


sim:
	@python3 -m analysis.similarity_measure_eval


vis:
	@python3 -m analysis.visualiser


run_server:
	@python3 -m analysis.http_server
