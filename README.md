# DINOBot Visual Similarity Measure analysis and Active Imitation Learning Implementation

This repository contains the code for all parts of the project that were not directly related to the DINOBot
reimplementation. That includes the Visual Similarity Measure Evaluation framework, the Active Imitation Learning
algorithms, the different metaheuristics, and much more.

More information about the implemented methods, including a user guide, can be found in the dissertation report.

This repository was private for the duration of the project and made public on the 5th of September 2024.

### Structure

The repository contains the following:

- The `_data` directory which contains all the object images, the precomputed embeddings, and the Siamese models
  training data in the `training_data` subdirectory. It also has the `objects.json` and `transfers.json` files which
  provide object names and transfer success rates as found using the DINOBot experiments, and the `siamese_similarities`
  directory which contains the precomputed similarities for different Siamese models
- The `analysis` module which contains different data analysis components
    - `approximation_calculations.py` - used for estimating the threshold and affine transformations for the entire
      dataset, using different preprocessing steps
    - `approximations.json` - the precomputed approximations
    - `similarity_measure_eval.py` - the main Visual Similarity Measure Evaluation framework
    - `visualiser_proj.py` - the interactive visualisation tool that projects the data into a 2d space
    - `visualiser_sim_eval.py` - the interactive visualisation tool that plots the visual similarity against the
      transfer success rates
    - `initial_data_analysis.ipynb` - a notebook with some initial data analysis
    - The `results` directory which stores the results of the Visual Similarity Measure Evaluation framework
- The `optim` module which contains code related to optimisation problems
    - The `mh` module which houses all the metaheuristics, namely Clustering, Evolutionary Search, Exhaustive Search,
      Greedy Local Search, MealPy heuristics, Random Search, Random Hill Climbing, Simulated Annealing, Binary Particle
      Swarm Optimisation, and Tabu Search. It also has `metaheuristic.py` which contains the common functions used
      across the different algorithms and `simanneal_lib.py` which used a Simulated Annealing library for hyperparameter
      selection
    - The `models` directory stores all Siamese models trained for this project (3 training types by 8 preprocessing
      combinations)
    - `affine_approx_solver.py` - contains the implementation of the Iterative Affine Approximation Algorithm. It also
      has some helper code used for obtaining experiments of experiments
    - `approx_solver.py` - the abstract class for an Iterative Approximation Algorithm
    - `basic_solver.py` - the basic solver which runs a chosen metaheuristic once. It also has some experiment code used
      to compile results
    - `hyperparam.py` - contains the hyperparameter search code used to find parameters for the different metaheuristics
    - `model_training.py` - code related to the training of the Siamese models
    - `solver.py` - the abstract class for a solver
    - `threshold_approximation_solver.py` - contains the implementation of the Iterative Threshold Approximation
      Algorithm. It also includes code used to generate the results for the report
    - `training_res.json` and `training_res.txt` contain the log of model training, with the json file being a parsed
      version of the txt
    - `utils.py` with a NeighbourGenerator
- The `playground` module with the implementation of the Environment in which the Objects reside
    - `basic_object.py` contains the code for an abstract object with a random latent representation
    - `Contour.ipynb` - a notebook containing contour embedding extraction code to be run on Google Colab, due to
      versioning issues when running on the GPU cluster
    - `environment.py` which provides an interface between Objects and Solvers. It provides common functionality for
      dealing with visual similarity vs transfer success rate, and so on
    - `extractor.py` - the code responsible for extracting embeddings from images. It also deals with preprocessing and
      potentially using multiple images. For efficiency, it saves all computed embeddings
    - `object.py` contains the abstract Object class which contains the code for calculating similarity between
      different embeddings
    - `sim_object.py` - the implementation of an Object for the DINOBot data from simulation
    - `storage.py` - an abstraction of the storage of all objects, so Environment can focus on the similarity
      calculations. It can create random, or load objects from data
- `tests` is the module that contains all automated tests. 100% of the `playground` module is covered by those tests.
- `config.py` contains the configuration used throughout the repository
- `helper.py` which contains many helper methods
- `tm_utils.py` which contains globally shared utilities, including enums for different tasks, image embeddings types,
  and so on
- `affine_runner.sh` is the bash script for running the Iterative Affine Approximation Algorithm. Other such scripts
  were created, including `approx_runner.sh` used for `approximation_calculations.py`, `helper_runner.sh` for running
  the `helper.py`, `hyper_runner.sh` for running the metaheuristics hyperparameter search, `sim_measure_runner.sh` for
  running the Visual Similarity Measure Evaluation, `solver_runner.sh` for running
  the `basic_solver.py`, `threshold_runner.sh` for running the Iterative Threshold Approximation Algorithm,
  and `training_runner.sh` for running the `model_training.py` code