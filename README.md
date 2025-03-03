![header](docs/header.png)

# AF3Complex

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

There are several flags that one must pass into the `run_af3complex.py` program, to
list them all run `python run_af3complex.py --help`. 

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

## Citing This Work

Any publication that discloses findings arising from using this source code, the
model parameters or outputs produced by those should cite:

```TBD```

## Accessing dat from the study

Please visit [our Zenodo page](https://zenodo.org/records/14927488?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjNmMTMwYzNmLTNmZjItNGFlZS1iNDUyLWU2Njc5ZDlkMTYyYSIsImRhdGEiOnt9LCJyYW5kb20iOiJhYmQ5NjZiODM2OTdiZjQ2OWZjYjFjZjNlYzFjNTRmMSJ9.F40J_RC-u3dAoBpGpYWaXc8bp3N3_Ynl4et50HLpo2dhBNr58mX0eiu4yFajVB28fPsu3xRg6j2PXaxHEPI-DA) to view the structural predictions from the study 
as well as the input parameters for the CASP16 predictions, which can be input directly
into the AF3Complex model. 


## Acknowledgements

We would like to thank the phenomenal researchers at Google DeepMind for developing AlphaFold3 and making it 
open-source. Without their work, AF3Complex could not have been developed. 





