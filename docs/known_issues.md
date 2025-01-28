# Known Issues

## Numerical performance for different GPU devices

There are numerical performance issues with some GPU types that are under
investigation, see
[this Issue](https://github.com/google-deepmind/alphafold3/issues/59) for
tracking.

### Verified devices

We have run successful large-scale numerical tests for the following devices and
maximum number of tokens:

-   H100 80 GB: up to 5,120 tokens.
-   A100 80 GB: up to 5,120 tokens.
-   A100 40 GB: up to 4,352 tokens with
    [unified memory configuration](https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md#nvidia-a100-40-gb).
-   P100 16 GB: up to 1,024 tokens.

Note that the 80 GB devices can run larger targets using unified memory, but
outputs have only been verified on particular examples rather than a large-scale
test set.

#### CUDA Capability 7.x GPUs: known issues

All CUDA Capability 7.x GPUs (e.g. V100) produce obviously bad output, with lots
of clashing residues (the clashes cause a ranking score of -99 or lower). With a
small fix relating to `bfloat16` conversion to `float32` outputs look normal,
but there are numerical performance regressions for some bucket sizes (tested on
V100 devices).

#### CUDA Capability 6.x GPUs: no known issues

CUDA Capability 6.x GPUs give reasonable output, but large scale numerical
testing has only been done for P100.
