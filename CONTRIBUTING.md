# Contributing to Methuselah‑prediction

Thank you for considering a contribution to the Methuselah‑prediction
project!  This guide outlines the steps and expectations for
contributing code, data or documentation.  We aim to foster an open
community that advances ageing research while adhering to high
standards of reproducibility and ethics.

## Reporting issues

If you encounter a bug or have a feature request, please open an
issue on GitHub.  Provide as much detail as possible, including:

* Steps to reproduce the problem.
* Expected behaviour and observed behaviour.
* Environment details (operating system, Python version, dependencies).
* Screenshots or logs if applicable.

Constructive suggestions and discussions are welcome.  Please be
respectful of others’ time and perspectives.

## Adding data

The current demo uses an outdated protein localisation dataset.  To
improve biological relevance, we welcome contributions of open
datasets related to yeast ageing, gene expression under nutrient
perturbations, replicative or chronological lifespan measurements and
other omics profiles.  When adding a dataset:

1. **Check the licence.** Ensure that the data are public and
   licensed for redistribution.  Include a citation and licence link
   in the data folder or the README.
2. **Convert to CSV.** Our pipeline expects a tabular CSV with a
   single target column (e.g., `survival_label`) and any number of
   numeric or categorical features.  Provide a script if special
   preprocessing is required (e.g., RNA‑seq normalisation).
3. **Update `configs/base.yaml`.** Point `paths.raw_dir` to your raw
   data and set `paths.processed` to the desired processed CSV name.
4. **Document the dataset.** In the README, describe what the data
   represent, how they were generated and why they are relevant to
   ageing.

## Coding standards

* **Testing:** All new modules should include unit tests under
  `tests/`.  Tests should be deterministic and run quickly.
* **Documentation:** Public functions and scripts must have
  docstrings explaining their purpose, arguments and return values.
* **Type hints:** Use Python type hints where practical.
* **Formatting:** We use [Black](https://black.readthedocs.io/) and
  [Flake8](https://flake8.pycqa.org/) for code style.  Run
  `black .` and `flake8` before submitting a pull request.
* **Dependencies:** When adding new Python packages, pin their
  versions in `requirements.txt` and justify why they are needed.

## Ethical considerations

Predictive models of ageing carry ethical implications.  Biases in
datasets (e.g., strain selection, experimental conditions) can lead
to misleading conclusions.  We encourage contributors to discuss
potential biases, limitations and ethical considerations in pull
requests and documentation.

## Submitting a pull request

1. Fork the repository and create a feature branch.
2. Make your changes following the guidelines above.
3. Run the test suite locally (`pytest -q`) and ensure it passes.
4. Update documentation (`README.md`, relevant docstrings) and
   requirements if needed.
5. Push your branch and open a pull request.  Describe what you’ve
   done and why.  If your changes implement a suggestion from the
   issue tracker, reference the issue.
6. A project maintainer will review your PR.  They may request
   changes or improvements before merging.  We appreciate your
   patience and cooperation.

Thank you for helping to advance open, reproducible ageing research!