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


model_training:
	@python3 -m optim.model_training


sim:
	@python3 -m analysis.similarity_measure_eval


vis_sim:
	@python3 -m analysis.visualiser_sim_eval


vis_proj:
	@python3 -m analysis.visualiser_proj


approx:
	@python3 -m analysis.approximation_calculations


train:
	@python3 -m optim.model_training