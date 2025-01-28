# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Tests end-to-end running of AlphaFold 3."""

import contextlib
import csv
import difflib
import functools
import hashlib
import json
import os
import pathlib
import pickle
from typing import Any

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from alphafold3 import structure
from alphafold3.common import folding_input
from alphafold3.common import resources
from alphafold3.common.testing import data as testing_data
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.diffusion import model as diffusion_model
from alphafold3.model.scoring import alignment
from alphafold3.structure import test_utils
import jax
import numpy as np

import run_alphafold
import shutil
import datetime



_JACKHMMER_BINARY_PATH = shutil.which('jackhmmer')
_NHMMER_BINARY_PATH = shutil.which('nhmmer')
_HMMALIGN_BINARY_PATH = shutil.which('hmmalign')
_HMMSEARCH_BINARY_PATH = shutil.which('hmmsearch')
_HMMBUILD_BINARY_PATH = shutil.which('hmmbuild')


@contextlib.contextmanager
def _output(name: str):
  with open(result_path := f'{absltest.TEST_TMPDIR.value}/{name}', "wb") as f:
    yield result_path, f


jax.config.update('jax_enable_compilation_cache', False)


def _generate_diff(actual: str, expected: str) -> str:
  return '\n'.join(
      difflib.unified_diff(
          expected.split('\n'),
          actual.split('\n'),
          fromfile='expected',
          tofile='actual',
          lineterm='',
      )
  )


@functools.singledispatch
def _hash_data(x: Any, /) -> str:
  if x is None:
    return '<<None>>'
  return _hash_data(json.dumps(x).encode('utf-8'))


@_hash_data.register
def _(x: bytes, /) -> str:
  return hashlib.sha256(x).hexdigest()


@_hash_data.register
def _(x: jax.Array) -> str:
  return _hash_data(jax.device_get(x))


@_hash_data.register
def _(x: np.ndarray) -> str:
  if x.dtype == object:
    return ';'.join(map(_hash_data, x.ravel().tolist()))
  return _hash_data(x.tobytes())


@_hash_data.register
def _(_: structure.Structure) -> str:
  return '<<structure>>'


@_hash_data.register
def _(_: atom_layout.AtomLayout) -> str:
  return '<<atom-layout>>'


class InferenceTest(test_utils.StructureTestCase):
  """Test AlphaFold 3 inference."""

  def setUp(self):
    super().setUp()
    small_bfd_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/bfd-first_non_consensus_sequences__subsampled_1000.fasta'
    ).path()
    mgnify_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/mgy_clusters__subsampled_1000.fa'
    ).path()
    uniprot_cluster_annot_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/uniprot_all__subsampled_1000.fasta'
    ).path()
    uniref90_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/uniref90__subsampled_1000.fasta'
    ).path()
    ntrna_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq__subsampled_1000.fasta'
    ).path()
    rfam_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/rfam_14_4_clustered_rep_seq__subsampled_1000.fasta'
    ).path()
    rna_central_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/rnacentral_active_seq_id_90_cov_80_linclust__subsampled_1000.fasta'
    ).path()
    pdb_database_path = testing_data.Data(
        resources.ROOT / 'data/testdata/templates_v2/ww_pdb'
    ).path()
    seqres_database_path = testing_data.Data(
        resources.ROOT
        / 'test_data/miniature_databases/pdb_seqres_2022_09_28__subsampled_1000.fasta'
    ).path()

    self._data_pipeline_config = pipeline.DataPipelineConfig(
        jackhmmer_binary_path=_JACKHMMER_BINARY_PATH,
        nhmmer_binary_path=_NHMMER_BINARY_PATH,
        hmmalign_binary_path=_HMMALIGN_BINARY_PATH,
        hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH,
        hmmbuild_binary_path=_HMMBUILD_BINARY_PATH,
        small_bfd_database_path=small_bfd_database_path,
        mgnify_database_path=mgnify_database_path,
        uniprot_cluster_annot_database_path=uniprot_cluster_annot_database_path,
        uniref90_database_path=uniref90_database_path,
        ntrna_database_path=ntrna_database_path,
        rfam_database_path=rfam_database_path,
        rna_central_database_path=rna_central_database_path,
        pdb_database_path=pdb_database_path,
        seqres_database_path=seqres_database_path,
        max_template_date=datetime.date(2021, 9, 30),
    )
    test_input = {
        'name': '5tgy',
        'modelSeeds': [1234],
        'sequences': [
            {
                'protein': {
                    'id': 'A',
                    'sequence': 'SEFEKLRQTGDELVQAFQRLREIFDKGDDDSLEQVLEEIEELIQKHRQLFDNRQEAADTEAAKQGDQWVQLFQRFREAIDKGDKDSLEQLLEELEQALQKIRELAEKKN',
                    'modifications': [],
                    'unpairedMsa': None,
                    'pairedMsa': None,
                }
            },
            {'ligand': {'id': 'B', 'ccdCodes': ['7BU']}},
        ],
        'dialect': folding_input.JSON_DIALECT,
        'version': folding_input.JSON_VERSION,
    }
    self._test_input_json = json.dumps(test_input)
    self._runner = run_alphafold.ModelRunner(
        model_class=run_alphafold.diffusion_model.Diffuser,
        config=run_alphafold.make_model_config(),
        device=jax.local_devices()[0],
        model_dir=run_alphafold.DEFAULT_MODEL_DIR,
    )

  def compare_golden(self, result_path: str) -> None:
    filename = os.path.split(result_path)[1]
    golden_path = testing_data.Data(
        resources.ROOT / f'test_data/{filename}'
    ).path()
    with open(golden_path, 'r') as golden_file:
      golden_text = golden_file.read()
    with open(result_path, 'r') as result_file:
      result_text = result_file.read()

    diff = _generate_diff(result_text, golden_text)

    self.assertEqual(diff, "", f"Result differs from golden:\n{diff}")

  def test_config(self):
    model_config = run_alphafold.make_model_config()
    model_config_as_str = json.dumps(
        model_config.as_dict(), sort_keys=True, indent=2
    )
    with _output('model_config.json') as (result_path, output):
      output.write(model_config_as_str.encode('utf-8'))
    self.compare_golden(result_path)

  def test_featurisation(self):
    """Run featurisation and assert that the output is as expected."""
    fold_input = folding_input.Input.from_json(self._test_input_json)
    data_pipeline = pipeline.DataPipeline(self._data_pipeline_config)
    full_fold_input = data_pipeline.process(fold_input)
    featurised_example = featurisation.featurise_input(
        full_fold_input,
        ccd=chemical_components.cached_ccd(),
        buckets=None,
    )

    with _output('featurised_example.pkl') as (_, output):
      output.write(pickle.dumps(featurised_example))
    featurised_example = jax.tree_util.tree_map(_hash_data, featurised_example)
    with _output('featurised_example.json') as (result_path, output):
      output.write(
          json.dumps(featurised_example, sort_keys=True, indent=2).encode(
              'utf-8'
          )
      )
    self.compare_golden(result_path)

  def test_model_inference(self):
    """Run model inference and assert that the output is as expected."""
    featurised_examples = pickle.loads(
        (resources.ROOT / 'test_data' / 'featurised_example.pkl').read_bytes()
    )

    self.assertLen(featurised_examples, 1)
    featurised_example = featurised_examples[0]
    inference_result = self._runner.run_inference(
        featurised_example, jax.random.PRNGKey(0)
    )
    inference_result = jax.tree_util.tree_map(_hash_data, inference_result)
    self.assertIsNotNone(inference_result)

  def test_write_input_json(self):
    fold_input = folding_input.Input.from_json(self._test_input_json)
    output_dir = self.create_tempdir()
    run_alphafold.write_fold_input_json(fold_input, output_dir)
    with open(
        os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json'),
        'rt',
    ) as f:
      actual_fold_input = folding_input.Input.from_json(f.read())

    self.assertEqual(actual_fold_input, fold_input)

  def test_process_fold_input_runs_only_data_pipeline(self):
    fold_input = folding_input.Input.from_json(self._test_input_json)
    output_dir = self.create_tempdir()
    run_alphafold.process_fold_input(
        fold_input=fold_input,
        data_pipeline_config=self._data_pipeline_config,
        model_runner=None,
        output_dir=output_dir,
    )
    with open(
        os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json'),
        'rt',
    ) as f:
      actual_fold_input = folding_input.Input.from_json(f.read())

    featurisation.validate_fold_input(actual_fold_input)

  def test_process_fold_input_runs_only_inference(self):
    with self.assertRaisesRegex(ValueError, 'missing unpaired MSA.'):
      run_alphafold.process_fold_input(
          fold_input=folding_input.Input.from_json(self._test_input_json),
          # No data pipeline config, so featursation will run first, and fail
          # since the input is missing MSAs.
          data_pipeline_config=None,
          model_runner=self._runner,
          output_dir=self.create_tempdir(),
      )

  def test_no_chains_in_input(self):
    fold_input = folding_input.Input(
        name='empty',
        chains=[],
        rng_seeds=[0],
    )

    with self.assertRaisesRegex(ValueError, 'Fold input has no chains.'):
      run_alphafold.process_fold_input(
          fold_input=fold_input,
          data_pipeline_config=self._data_pipeline_config,
          model_runner=run_alphafold.ModelRunner(
              model_class=diffusion_model.Diffuser,
              config=run_alphafold.make_model_config(),
              device=jax.local_devices(backend='gpu')[0],
              model_dir=pathlib.Path(run_alphafold.DEFAULT_MODEL_DIR),
          ),
          output_dir='unused output dir',
      )

  @parameterized.named_parameters(
      {
          'testcase_name': 'default_bucket',
          'bucket': None,
          'exp_ranking_scores': [0.69, 0.69, 0.72, 0.75, 0.70],
      },
      {
          'testcase_name': 'bucket_1024',
          'bucket': 1024,
          'exp_ranking_scores': [0.69, 0.71, 0.71, 0.69, 0.70],
      },
  )
  def test_inference(self, bucket, exp_ranking_scores):
    """Run AlphaFold 3 inference."""

    ### Prepare inputs.
    fold_input = folding_input.Input.from_json(self._test_input_json)

    output_dir = self.create_tempdir()
    actual = run_alphafold.process_fold_input(
        fold_input,
        self._data_pipeline_config,
        run_alphafold.ModelRunner(
            model_class=diffusion_model.Diffuser,
            config=run_alphafold.make_model_config(),
            device=jax.local_devices(backend='gpu')[0],
            model_dir=pathlib.Path(run_alphafold.DEFAULT_MODEL_DIR),
        ),
        output_dir=output_dir,
        buckets=None if bucket is None else [bucket],
    )
    logging.info('finished get_inference_result')
    expected_model_cif_filename = f'{fold_input.sanitised_name()}_model.cif'
    expected_summary_confidences_filename = (
        f'{fold_input.sanitised_name()}_summary_confidences.json'
    )
    expected_confidences_filename = (
        f'{fold_input.sanitised_name()}_confidences.json'
    )
    expected_data_json_filename = f'{fold_input.sanitised_name()}_data.json'

    self.assertSameElements(
        os.listdir(output_dir),
        [
            # Subdirectories, one for each sample.
            'seed-1234_sample-0',
            'seed-1234_sample-1',
            'seed-1234_sample-2',
            'seed-1234_sample-3',
            'seed-1234_sample-4',
            # Top ranking result.
            expected_confidences_filename,
            expected_model_cif_filename,
            expected_summary_confidences_filename,
            # Ranking scores for all samples.
            'ranking_scores.csv',
            # The input JSON defining the job.
            expected_data_json_filename,
            # The output terms of use.
            'TERMS_OF_USE.md',
        ],
    )

    with open(os.path.join(output_dir, expected_data_json_filename), 'rt') as f:
      actual_input_json = json.load(f)

    self.assertEqual(
        actual_input_json['sequences'][0]['protein']['sequence'],
        fold_input.protein_chains[0].sequence,
    )
    self.assertSequenceEqual(
        actual_input_json['sequences'][1]['ligand']['ccdCodes'],
        fold_input.ligands[0].ccd_ids,
    )
    self.assertNotEmpty(
        actual_input_json['sequences'][0]['protein']['unpairedMsa']
    )
    self.assertNotEmpty(
        actual_input_json['sequences'][0]['protein']['pairedMsa']
    )
    self.assertIsNotNone(
        actual_input_json['sequences'][0]['protein']['templates']
    )

    with open(os.path.join(output_dir, 'ranking_scores.csv'), 'rt') as f:
      actual_ranking_scores = list(csv.DictReader(f))

    self.assertLen(actual_ranking_scores, 5)
    self.assertEqual(
        [int(s['seed']) for s in actual_ranking_scores], [1234] * 5
    )
    self.assertEqual(
        [int(s['sample']) for s in actual_ranking_scores], [0, 1, 2, 3, 4]
    )
    np.testing.assert_array_almost_equal(
        [float(s['ranking_score']) for s in actual_ranking_scores],
        exp_ranking_scores,
        decimal=2,
    )

    with open(os.path.join(output_dir, 'TERMS_OF_USE.md'), 'rt') as f:
      actual_terms_of_use = f.read()
    self.assertStartsWith(
        actual_terms_of_use, '# ALPHAFOLD 3 OUTPUT TERMS OF USE'
    )

    bucket_label = 'default' if bucket is None else bucket
    output_filename = f'run_alphafold_test_output_bucket_{bucket_label}.pkl'

    # Convert to dict to enable simple serialization.
    actual_dict = [
        dict(
            seed=actual_inf.seed,
            inference_results=actual_inf.inference_results,
            full_fold_input=actual_inf.full_fold_input,
        )
        for actual_inf in actual
    ]
    with _output(output_filename) as (_, output):
      output.write(pickle.dumps(actual_dict))

    logging.info('Comparing inference results with expected values.')

    ### Assert that output is as expected.
    expected_dict = pickle.loads(
        (
            resources.ROOT
            / 'test_data'
            / 'alphafold_run_outputs'
            / output_filename
        ).read_bytes()
    )
    expected = [
        run_alphafold.ResultsForSeed(**expected_inf)
        for expected_inf in expected_dict
    ]
    for actual_inf, expected_inf in zip(actual, expected, strict=True):
      for actual_inf, expected_inf in zip(
          actual_inf.inference_results,
          expected_inf.inference_results,
          strict=True,
      ):

        # Check RMSD is within tolerance.
        # 5tgy is very stable, NMR samples were all within 3.0 RMSD.
        actual_rmsd = alignment.rmsd_from_coords(
            actual_inf.predicted_structure.coords,
            expected_inf.predicted_structure.coords,
        )
        self.assertLess(actual_rmsd, 3.0)
        np.testing.assert_array_equal(
            actual_inf.predicted_structure.atom_occupancy,
            [1.0] * actual_inf.predicted_structure.num_atoms,
        )


if __name__ == '__main__':
  absltest.main()
