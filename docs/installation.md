# Installation and Running Your First Prediction

You will need a machine running Linux; AF3Complex does not support other
operating systems. Full installation requires up to 1 TB of disk space to keep
genetic databases (SSD storage is recommended) and an NVIDIA GPU with Compute
Capability 8.0 or greater (GPUs with more memory can predict larger protein
structures). We have verified that inputs with up to 5,120 tokens can fit on a
single NVIDIA A100 80 GB, or a single NVIDIA H100 80 GB. We have verified
numerical accuracy on both NVIDIA A100 and H100 GPUs.

Especially for long targets, the genetic search stage can consume a lot of RAM –
we recommend running with at least 64 GB of RAM.

We provide installation instructions for a machine with an NVIDIA A100 80 GB GPU
and a clean Ubuntu 22.04 LTS installation, and expect that these instructions
should aid others with different setups. If you are installing locally outside
of a Docker container, please ensure CUDA, cuDNN, and JAX are correctly
installed; the
[JAX installation documentation](https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu)
is a useful reference for this case.

Proceed only if `nvidia-smi` has a sensible output.


## Obtaining AF3Complex Source Code


Install `git` and download the AF3Complex repository:


```
git clone https://github.com/Jfeldman34/AF3Complex.git
```
Locate and cd into the cloned repository and then run: 

```
pip install .
```

## Obtaining Genetic Databases

This step requires `wget` and `zstd` to be installed on your machine. On
Debian-based systems install them by running `sudo apt install wget zstd`.

AF3Complex needs multiple genetic (sequence) protein and RNA databases to run:

*   [BFD small](https://bfd.mmseqs.com/)
*   [MGnify](https://www.ebi.ac.uk/metagenomics/)
*   [PDB](https://www.rcsb.org/) (structures in the mmCIF format)
*   [PDB seqres](https://www.rcsb.org/)
*   [UniProt](https://www.uniprot.org/uniprot/)
*   [UniRef90](https://www.uniprot.org/help/uniref)
*   [NT](https://www.ncbi.nlm.nih.gov/nucleotide/)
*   [RFam](https://rfam.org/)
*   [RNACentral](https://rnacentral.org/)

We provide a bash script `fetch_databases.sh` that can be used to download and
set up all of these databases. This process takes around 45 minutes when not
installing on local SSD. We recommend running the following in a `screen` or
`tmux` session as downloading and decompressing the databases takes some time.

```sh
cd af3complex  # Navigate to the directory with cloned AF3Complex repository.
./fetch_databases.sh <DB_DIR>
```

This script downloads the databases from a mirror hosted on GCS, with all
versions being the same as used in the AF3Complex paper.

:ledger: **Note: The download directory `<DB_DIR>` should *not* be a
subdirectory in the AF3Complex repository directory.** If it is, the Docker
build will be slow as the large databases will be copied during the image
creation.

:ledger: **Note: The total download size for the full databases is around 252 GB
and the total size when unzipped is 630 GB. Please make sure you have sufficient
hard drive space, bandwidth, and time to download. We recommend using an SSD for
better genetic search performance.**

:ledger: **Note: If the download directory and datasets don't have full read and
write permissions, it can cause errors with the MSA tools, with opaque
(external) error messages. Please ensure the required permissions are applied,
e.g. with the `sudo chmod 755 --recursive <DB_DIR>` command.**

Once the script has finished, you should have the following directory structure:

```sh
mmcif_files/  # Directory containing ~200k PDB mmCIF files.
bfd-first_non_consensus_sequences.fasta
mgy_clusters_2022_05.fa
nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta
pdb_seqres_2022_09_28.fasta
rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta
rnacentral_active_seq_id_90_cov_80_linclust.fasta
uniprot_all_2021_04.fa
uniref90_2022_05.fa
```

Optionally, after the script finishes, you may want copy databases to an SSD.
You can use theses two scripts:

*   `src/scripts/gcp_mount_ssd.sh <SSD_MOUNT_PATH>` Mounts and formats an
    unmounted GCP SSD drive. It will skip the either step if the disk is either
    already formatted or already mounted. The default `<SSD_MOUNT_PATH>` is
    `/mnt/disks/ssd`.
*   `src/scripts/copy_to_ssd.sh <DB_DIR> <SSD_DB_DIR>` this will copy as many
    files that it can fit on to the SSD. The default `<DATABASE_DIR>` is
    `$HOME/public_databases` and the default `<SSD_DB_DIR>` is
    `/mnt/disks/ssd/public_databases`.

## Obtaining Model Parameters

To request access to the AlphaFold3 model parameters, please complete
[this form](https://forms.gle/svvpY4u2jsHEwWYS6).

## Building the molecular databases. 

After cloning and installing the AF3Complex repository, run the 
following command to build the CCD.pickle file, which is used
to process ligands. 

```sh
build_data
```

Also, to avoid a possible CUDA bottleneck, run this command within your workspace:

```sh
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
```

## Completion

You can now run AF3Complex on your device! 

Run AF3Complex repository with the following command, substituting the appropriate paths
and values.  

```
run_af3complex.py --json_file_path=input_json_path --model_dir=model_parameters_path
--db_dir=database_dir_path --output_dir=output_dir_path --input_json_type=either_af3_or_server
```


:ledger: **Note: In the example above the databases have been placed on the
persistent disk, which is slow.** If you want better genetic and template search
performance, make sure all databases are placed on a local SSD.

If you have databases on SSD in `<SSD_DB_DIR>` you can use uses it as the
location to look for databases but allowing for a multiple fallbacks with
`--db_dir` which can be specified multiple times.


