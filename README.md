![header](docs/header.png)

# AF3Complex

Accurate structures of protein complexes are essential for understanding biological pathway function. A previous study showed how downstream modifications to AlphaFold 2 could yield AF2Complex, a model better suited for protein complexes. Here, we introduce AF3Complex, a model equipped with the same improvements as AF2Complex, along with a novel method for excluding ligands, built on AlphaFold 3.

## Software

The following package provides the full implementation of the AF3Complex protein
structure prediction pipeline.

Please also refer to the package documentation for information concerning input, output, 
and performance.

## Obtaining Model Parameters for AF3Complex

This repository contains all the code necessary to run AF3Complex. However, 
for the inference pipeline to run, one must provide the AlphaFold3 model weights. 

To request access to the AlphaFold3 model parameters, please complete
[this form](https://forms.gle/svvpY4u2jsHEwWYS6). Access to the model's weights
is granted solely at Google DeepMind’s discretion.

## Installation and Running Your First Prediction

See the [installation documentation](docs/installation.md).

There are several flags that one must pass into the `run_feature_generation.py` and `run_af3complex_inference.py` programs, to
list them all run the scripts with the `--help` argument. 

To run some example processes, please refer to the [example documentation](examples/example.md).

## AF3Complex Input

See the [input documentation](docs/input.md).

## AF3Complex Output

See the [output documentation](docs/output.md).

## AF3Complex Performance

See the [performance documentation](docs/performance.md), which is based on the performance
information of the parent model of AF3Complex, AlphaFold3.

## Known Issues

Known issues are documented in the
[known issues documentation](docs/known_issues.md). The known issues listed include
those of AlphaFold3 as several of them correspond directly to known issues of AF3Complex. 

## Models and Features for AF3Complex Paper

Please visit this [Zenodo](https://zenodo.org/records/15514353) page to view the 
features and models for the AF3Complex paper's analyses. 


## Citing This Work

Any publication that discloses findings arising from using this source code or outputs produced by this source code
should cite:

```
@article {Feldman2025.02.27.640585,
	author = {Feldman, Jonathan and Skolnick, Jeffrey},
	title = {AF3Complex Yields Improved Structural Predictions of Protein Complexes},
	elocation-id = {2025.02.27.640585},
	year = {2025},
	doi = {10.1101/2025.02.27.640585},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/03/03/2025.02.27.640585},
	eprint = {https://www.biorxiv.org/content/early/2025/03/03/2025.02.27.640585.full.pdf},
	journal = {bioRxiv}
}
```


## Acknowledgements

We would like to thank the phenomenal researchers at Google DeepMind for developing AlphaFold3 and making it 
open-source. Without their work, AF3Complex could not have been developed. 





