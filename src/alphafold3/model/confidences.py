# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Functions for extracting and processing confidences from model outputs."""
import warnings

from absl import logging
from alphafold3 import structure
from alphafold3.constants import residue_names
from alphafold3.cpp import mkdssp
import jax.numpy as jnp
import numpy as np
from scipy import spatial
import scipy
from typing import *





# From Sander & Rost 1994 https://doi.org/10.1002/prot.340200303
MAX_ACCESSIBLE_SURFACE_AREA = {
    'ALA': 106.0,
    'ARG': 248.0,
    'ASN': 157.0,
    'ASP': 163.0,
    'CYS': 135.0,
    'GLN': 198.0,
    'GLU': 194.0,
    'GLY': 84.0,
    'HIS': 184.0,
    'ILE': 169.0,
    'LEU': 164.0,
    'LYS': 205.0,
    'MET': 188.0,
    'PHE': 197.0,
    'PRO': 136.0,
    'SER': 130.0,
    'THR': 142.0,
    'TRP': 227.0,
    'TYR': 222.0,
    'VAL': 142.0,
}

# Weights for ranking confidence.
_CLASH_PENALIZATION_WEIGHT = 100.0


def windowed_solvent_accessible_area(cif: str, window: int = 25) -> np.ndarray:
  """Implementation of AlphaFold_RSA.

  AlphaFold_RSA defined in
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9601767/.

  Args:
    cif: raw cif string.
    window: The window over which to average accessible surface area

  Returns:
    An array of size num_res that predicts disorder by using windowed solvent
    accessible surface area.
  """
  result = mkdssp.get_dssp(cif, calculate_surface_accessibility=True)
  parse_row = False
  rasa = []
  for row in result.splitlines():
    if parse_row:
      aa = row[13:14]
      if aa == '!':
        continue
      aa3 = residue_names.PROTEIN_COMMON_ONE_TO_THREE.get(aa, 'ALA')
      max_acc = MAX_ACCESSIBLE_SURFACE_AREA[aa3]
      acc = int(row[34:38])
      norm_acc = acc / max_acc
      if norm_acc > 1.0:
        norm_acc = 1.0
      rasa.append(norm_acc)
    if row.startswith('  #  RESIDUE'):
      parse_row = True

  half_w = (window - 1) // 2
  pad_rasa = np.pad(rasa, (half_w, half_w), 'reflect')
  rasa = np.convolve(pad_rasa, np.ones(window), 'valid') / window
  return rasa


def fraction_disordered(
    struc: structure.Structure, rasa_disorder_cutoff: float = 0.581
) -> float:
  """Compute fraction of protein residues that are disordered.

  Args:
    struc: A structure to compute rASA metrics on.
    rasa_disorder_cutoff: The threshold at which residues are considered
      disordered.  Default value taken from
      https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9601767/.

  Returns:
    The fraction of protein residues that are disordered
    (rasa > rasa_disorder_cutoff).
  """
  struc = struc.filter_to_entity_type(protein=True)
  rasa = []
  seq_rasa = {}
  for chain_id, chain_seq in struc.chain_single_letter_sequence().items():
    if chain_seq in seq_rasa:
      # We assume that identical sequences have approximately similar rasa
      # values to speed up the computation.
      rasa.extend(seq_rasa[chain_seq])
      continue
    chain_struc = struc.filter(chain_id=chain_id)
    try:
      rasa_per_residue = windowed_solvent_accessible_area(
          chain_struc.to_mmcif()
      )
      seq_rasa[chain_seq] = rasa_per_residue
      rasa.extend(rasa_per_residue)
    except (ValueError, RuntimeError):
      logging.warning('%s: rasa calculation failed', struc.name)

  if not rasa:
    return 0.0
  return np.mean(np.array(rasa) > rasa_disorder_cutoff)


def has_clash(
    struc: structure.Structure,
    cutoff_radius: float = 1.1,
    min_clashes_for_overlap: int = 100,
    min_fraction_for_overlap: float = 0.5,
) -> bool:
  """Determine whether the structure has at least one clashing chain.

  A clashing chain is defined as having greater than 100 polymer atoms within
  1.1A of another polymer atom, or having more than 50% of the chain with
  clashing atoms.

  Args:
    struc: A structure to get clash metrics for.
    cutoff_radius: atom distances under this threshold are considered a clash.
    min_clashes_for_overlap: The minimum number of atom-atom clashes for a chain
      to be considered overlapping.
    min_fraction_for_overlap: The minimum fraction of atoms within a chain that
      are clashing for the chain to be considered overlapping.

  Returns:
    True if the structure has at least one clashing chain.
  """
  struc = struc.filter_to_entity_type(protein=True, rna=True, dna=True)
  if not struc.chains:
    return False
  coords = struc.coords
  coord_kdtree = spatial.cKDTree(coords)
  clashes_per_atom = coord_kdtree.query_ball_point(
      coords, p=2.0, r=cutoff_radius
  )
  per_atom_has_clash = np.zeros(len(coords), dtype=np.int32)
  for atom_idx, clashing_indices in enumerate(clashes_per_atom):
    for clashing_idx in clashing_indices:
      if np.abs(struc.res_id[atom_idx] - struc.res_id[clashing_idx]) > 1 or (
          struc.chain_id[atom_idx] != struc.chain_id[clashing_idx]
      ):
        per_atom_has_clash[atom_idx] = True
        break
  for chain_id in struc.chains:
    mask = struc.chain_id == chain_id
    num_atoms = np.sum(mask)
    if num_atoms == 0:
      continue
    num_clashes = np.sum(per_atom_has_clash * mask)
    frac_clashes = num_clashes / num_atoms
    if (
        num_clashes > min_clashes_for_overlap
        or frac_clashes > min_fraction_for_overlap
    ):
      return True
  return False


 
def get_IS_ranking_score(
    ptm: float, IS_score: float, fraction_disordered_: float, has_clash_: bool
) -> float:
  # pTM will be used when there is only one chain.
  if IS_score == 0:
    ranking_value = ptm
  else:
    ranking_value = IS_score
  return (
      ranking_value
      - _CLASH_PENALIZATION_WEIGHT * has_clash_
  )
  
  


def rank_metric(
    full_pde: jnp.ndarray | np.ndarray, contact_probs: jnp.ndarray | np.ndarray
) -> jnp.ndarray | np.ndarray:
  """Compute the metric that will be used to rank predictions, higher is better.

  Args:
    full_pde: A [num_samples, num_tokens,num_tokens] matrix of predicted
      distance errors between pairs of tokens.
    contact_probs: A [num_tokens, num_tokens] matrix consisting of the
      probability of contact (<8A) that is returned from the distogram head.

  Returns:
    A scalar that can be used to rank (higher is better).
  """
  if not isinstance(full_pde, type(contact_probs)):
    raise ValueError('full_pde and contact_probs must be of the same type.')

  if isinstance(full_pde, np.ndarray):
    sum_fn = np.sum
  elif isinstance(full_pde, jnp.ndarray):
    sum_fn = jnp.sum
  else:
    raise ValueError('full_pde must be a numpy array or a jax array.')
  # It was found that taking the contact_map weighted average was better than
  # just the predicted distance error on its own.
  return -sum_fn(full_pde * contact_probs[None, :, :], axis=(-2, -1)) / (
      sum_fn(contact_probs) + 1e-6
  )


def weighted_mean(mask, value, axis):
  return np.mean(mask * value, axis=axis) / (1e-8 + np.mean(mask, axis=axis))


def pde_single(
    num_tokens: int,
    asym_ids: np.ndarray,
    full_pde: np.ndarray,
    contact_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Compute 1D PDE summaries.

  Args:
    num_tokens: The number of tokens (not including padding).
    asym_ids: The asym_ids (array of shape num_tokens).
    full_pde: A [num_samples, num_tokens, num_tokens] matrix of predicted
      distance errors.
    contact_probs: A [num_tokens, num_tokens] matrix consisting of the
      probability of contact (<8A) that is returned from the distogram head.

  Returns:
    A tuple (ichain, xchain, full_chain) where:
      `ichain` is a [num_samples, num_chains] matrix where the
      value assigned to each chain is an average of the full PDE matrix over all
      its within-chain interactions, weighted by `contact_probs`.
      `xchain` is a [num_samples, num_chains] matrix where the
      value assigned to each chain is an average of the full PDE matrix over all
      its cross-chain interactions, weighted by `contact_probs`.
      `full_chain` is a [num_samples, num_tokens] matrix where the
      value assigned to each token is an average of it PDE against all tokens,
      weighted by `contact_probs`.
  """

  full_pde = full_pde[:, :num_tokens, :num_tokens]
  contact_probs = contact_probs[:num_tokens, :num_tokens]
  asym_ids = asym_ids[:num_tokens]
  unique_asym_ids = np.unique(asym_ids)
  num_chains = len(unique_asym_ids)
  num_samples = full_pde.shape[0]

  asym_ids = asym_ids[None]
  contact_probs = contact_probs[None]

  ichain = np.zeros((num_samples, num_chains))
  xchain = np.zeros((num_samples, num_chains))

  for idx, asym_id in enumerate(unique_asym_ids):
    my_asym_id = asym_ids == asym_id
    imask = my_asym_id[:, :, None] * my_asym_id[:, None, :]
    xmask = my_asym_id[:, :, None] * ~my_asym_id[:, None, :]
    imask = imask * contact_probs
    xmask = xmask * contact_probs
    ichain[:, idx] = weighted_mean(mask=imask, value=full_pde, axis=(-2, -1))
    xchain[:, idx] = weighted_mean(mask=xmask, value=full_pde, axis=(-2, -1))

  full_chain = weighted_mean(mask=contact_probs, value=full_pde, axis=(-1,))

  return ichain, xchain, full_chain


def chain_pair_pde(
    num_tokens: int, asym_ids: np.ndarray, full_pde: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Compute predicted distance errors for all pairs of chains.

  Args:
    num_tokens: The number of tokens (not including padding).
    asym_ids: The asym_ids (array of shape num_tokens).
    full_pde: A [num_samples, num_tokens, num_tokens] matrix of predicted
      distance errors.

  Returns:
    chain_pair_pred_err_mean - a [num_chains, num_chains] matrix with average
      per chain-pair predicted distance error.
    chain_pair_pred_err_min - a [num_chains, num_chains] matrix with min
      per chain-pair predicted distance error.
  """
  full_pde = full_pde[:, :num_tokens, :num_tokens]
  asym_ids = asym_ids[:num_tokens]
  unique_asym_ids = np.unique(asym_ids)
  num_chains = len(unique_asym_ids)
  num_samples = full_pde.shape[0]
  chain_pair_pred_err_mean = np.zeros((num_samples, num_chains, num_chains))
  chain_pair_pred_err_min = np.zeros((num_samples, num_chains, num_chains))

  for idx1, asym_id_1 in enumerate(unique_asym_ids):
    subset = full_pde[:, asym_ids == asym_id_1, :]
    for idx2, asym_id_2 in enumerate(unique_asym_ids):
      subsubset = subset[:, :, asym_ids == asym_id_2]
      chain_pair_pred_err_mean[:, idx1, idx2] = np.mean(subsubset, axis=(1, 2))
      chain_pair_pred_err_min[:, idx1, idx2] = np.min(subsubset, axis=(1, 2))
  return chain_pair_pred_err_mean, chain_pair_pred_err_min


def weighted_nanmean(
    value: np.ndarray, mask: np.ndarray, axis: int
) -> np.ndarray:
  """Nan-mean with weighting -- empty slices return NaN."""
  assert mask.shape == value.shape
  assert not np.isnan(mask).all()

  nan_idxs = np.where(np.isnan(value))
  # Need to NaN the mask to get the correct denominator weighting.
  mask_with_nan = mask.copy()
  mask_with_nan[nan_idxs] = np.nan
  with warnings.catch_warnings():
    # Mean of empty slice is ok and should return a NaN.
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    return np.nanmean(value * mask_with_nan, axis=axis) / np.nanmean(
        mask_with_nan, axis=axis
    )


def chain_pair_pae(
    *,
    num_tokens: int,
    asym_ids: np.ndarray,
    full_pae: np.ndarray,
    mask: np.ndarray | None = None,
    contact_probs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Compute predicted errors for all pairs of chains.

  Args:
    num_tokens: The number of tokens (not including padding).
    asym_ids: The asym_ids (array of shape num_tokens).
    full_pae: A [num_samples, num_tokens, num_tokens] matrix of predicted
      errors.
    mask: A [num_tokens, num_tokens] mask matrix.
    contact_probs: A [num_tokens, num_tokens] matrix consisting of the
      probability of contact (<8A) that is returned from the distogram head.

  Returns:
    chain_pair_pred_err_mean - a [num_chains, num_chains] matrix with average
      per chain-pair predicted error.
  """
  if mask is None:
    mask = np.ones(shape=full_pae.shape[1:], dtype=bool)
  if contact_probs is None:
    contact_probs = np.ones(shape=full_pae.shape[1:], dtype=float)
  assert mask.shape == full_pae.shape[1:]

  full_pae = full_pae[:, :num_tokens, :num_tokens]
  mask = mask[:num_tokens, :num_tokens]
  asym_ids = asym_ids[:num_tokens]
  contact_probs = contact_probs[:num_tokens, :num_tokens]
  unique_asym_ids = np.unique(asym_ids)
  num_chains = len(unique_asym_ids)
  num_samples = full_pae.shape[0]
  chain_pair_pred_err_mean = np.zeros((num_samples, num_chains, num_chains))
  chain_pair_pred_err_min = np.zeros((num_samples, num_chains, num_chains))

  for idx1, asym_id_1 in enumerate(unique_asym_ids):
    subset = full_pae[:, asym_ids == asym_id_1, :]
    subset_mask = mask[asym_ids == asym_id_1, :]
    subset_contact_probs = contact_probs[asym_ids == asym_id_1, :]
    for idx2, asym_id_2 in enumerate(unique_asym_ids):
      subsubset = subset[:, :, asym_ids == asym_id_2]
      subsubset_mask = subset_mask[:, asym_ids == asym_id_2]
      subsubset_contact_probs = subset_contact_probs[:, asym_ids == asym_id_2]
      (flat_mask_idxs,) = np.where(subsubset_mask.flatten() > 0)
      flat_subsubset = subsubset.reshape([num_samples, -1])
      flat_contact_probs = subsubset_contact_probs.flatten()
      # A ligand chain will have no valid frames if it contains fewer than
      # three non-colinear atoms (e.g. a sodium ion).
      if not flat_mask_idxs.size:
        chain_pair_pred_err_mean[:, idx1, idx2] = np.nan
        chain_pair_pred_err_min[:, idx1, idx2] = np.nan
      else:
        chain_pair_pred_err_min[:, idx1, idx2] = np.min(
            flat_subsubset[:, flat_mask_idxs], axis=1
        )
        chain_pair_pred_err_mean[:, idx1, idx2] = weighted_mean(
            mask=flat_contact_probs[flat_mask_idxs],
            value=flat_subsubset[:, flat_mask_idxs],
            axis=-1,
        )
  return chain_pair_pred_err_mean, chain_pair_pred_err_min, unique_asym_ids


def reduce_chain_pair(
    *,
    chain_pair_met: np.ndarray,
    num_chain_tokens: np.ndarray,
    agg_over_col: bool,
    agg_type: str,
    weight_method: str,
) -> tuple[np.ndarray, np.ndarray]:
  """Compute 1D summaries from a chain-pair summary.

  Args:
    chain_pair_met: A [num_samples, num_chains, num_chains] aggregate matrix.
    num_chain_tokens: A [num_chains] array of number of tokens for each chain.
      Used for 'per_token' weighting.
    agg_over_col: Whether to aggregate the PAE over rows (i.e. average error
      when aligned to me) or columns (i.e. my average error when aligned to all
      others.)
    agg_type: The type of aggregation to use, 'mean' or 'min'.
    weight_method: The method to use for weighting the PAE, 'per_token' or
      'per_chain'.

  Returns:
    A tuple (ichain, xchain) where:
      `ichain` is a [num_samples, num_chains] matrix where the
      value assigned to each chain is an average of the full PAE matrix over all
      its within-chain interactions, weighted by `contact_probs`.
      `xchain` is a [num_samples, num_chains] matrix where the
      value assigned to each chain is an average of the full PAE matrix over all
      its cross-chain interactions, weighted by `contact_probs`.
  """
  num_samples, num_chains, _ = chain_pair_met.shape

  ichain = chain_pair_met.diagonal(axis1=-2, axis2=-1)

  if weight_method == 'per_chain':
    chain_weight = np.ones((num_chains,), dtype=float)
  elif weight_method == 'per_token':
    chain_weight = num_chain_tokens
  else:
    raise ValueError(f'Unknown weight method: {weight_method}')

  if agg_over_col:
    agg_axis = -1
  else:
    agg_axis = -2

  if agg_type == 'mean':
    weight = np.ones((num_samples, num_chains, num_chains), dtype=float)
    weight -= np.eye(num_chains, dtype=float)
    weight *= chain_weight[None] * chain_weight[:, None]
    xchain = weighted_nanmean(chain_pair_met, mask=weight, axis=agg_axis)
  elif agg_type == 'min':
    is_self = np.eye(num_chains)
    with warnings.catch_warnings():
      # Min over empty slice is ok and should return a NaN.
      warnings.filterwarnings('ignore', message='All-NaN slice encountered')
      xchain = np.nanmin(chain_pair_met + 1e8 * is_self, axis=agg_axis)
  else:
    raise ValueError(f'Unknown aggregation method: {agg_type}')

  return ichain, xchain


def pae_metrics(
    num_tokens: int,
    asym_ids: np.ndarray,
    full_pae: np.ndarray,
    mask: np.ndarray,
    contact_probs: np.ndarray,
    tm_adjusted_pae: np.ndarray,
):
  """PAE aggregate metrics."""
  assert mask.shape == full_pae.shape[1:]
  assert contact_probs.shape == full_pae.shape[1:]

  chain_pair_contact_weighted, _, unique_asym_ids = chain_pair_pae(
      num_tokens=num_tokens,
      asym_ids=asym_ids,
      full_pae=full_pae,
      mask=mask,
      contact_probs=contact_probs,
  )

  ret = {}
  ret['chain_pair_pae_mean'], ret['chain_pair_pae_min'], _ = chain_pair_pae(
      num_tokens=num_tokens,
      asym_ids=asym_ids,
      full_pae=full_pae,
      mask=mask,
  )
  chain_pair_iptm = np.stack(
      [
          chain_pairwise_predicted_tm_scores(
              tm_adjusted_pae=sample_tm_adjusted_pae[:num_tokens],
              asym_id=asym_ids[:num_tokens],
              pair_mask=mask[:num_tokens, :num_tokens],
          )
          for sample_tm_adjusted_pae in tm_adjusted_pae
      ],
      axis=0,
  )

  num_chain_tokens = np.array(
      [sum(asym_ids == asym_id) for asym_id in unique_asym_ids]
  )

  def reduce_chain_pair_fn(chain_pair: np.ndarray):
    def inner(agg_over_col):
      ichain_pae, xchain_pae = reduce_chain_pair(
          num_chain_tokens=num_chain_tokens,
          chain_pair_met=chain_pair,
          agg_over_col=agg_over_col,
          agg_type='mean',
          weight_method='per_chain',
      )
      return ichain_pae, xchain_pae

    ichain, xchain_row_agg = inner(False)
    _, xchain_col_agg = inner(True)
    with warnings.catch_warnings():
      # Mean of empty slice is ok and should return a NaN.
      warnings.filterwarnings(action='ignore', message='Mean of empty slice')
      xchain = np.nanmean(
          np.stack([xchain_row_agg, xchain_col_agg], axis=0), axis=0
      )
    return ichain, xchain

  pae_ichain, pae_xchain = reduce_chain_pair_fn(chain_pair_contact_weighted)
  iptm_ichain, iptm_xchain = reduce_chain_pair_fn(chain_pair_iptm)

  ret.update({
      'chain_pair_iptm': chain_pair_iptm,
      'iptm_ichain': iptm_ichain,
      'iptm_xchain': iptm_xchain,
      'pae_ichain': pae_ichain,
      'pae_xchain': pae_xchain,
  })

  return ret


def get_iptm_xchain(chain_pair_iptm: np.ndarray) -> np.ndarray:
  """Cross chain aggregate ipTM."""
  num_samples, num_chains, _ = chain_pair_iptm.shape
  weight = np.ones((num_samples, num_chains, num_chains), dtype=float)
  weight -= np.eye(num_chains, dtype=float)
  xchain_row_agg = weighted_nanmean(chain_pair_iptm, mask=weight, axis=-2)
  xchain_col_agg = weighted_nanmean(chain_pair_iptm, mask=weight, axis=-1)
  with warnings.catch_warnings():
    # Mean of empty slice is ok and should return a NaN.
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    iptm_xchain = np.nanmean(
        np.stack([xchain_row_agg, xchain_col_agg], axis=0), axis=0
    )
  return iptm_xchain


def predicted_tm_score(
    tm_adjusted_pae: np.ndarray,
    pair_mask: np.ndarray,
    asym_id: np.ndarray,
    interface: bool = False,
) -> float:
  """Computes predicted TM alignment or predicted interface TM alignment score.

  Args:
    tm_adjusted_pae: [num_res, num_res] Relevant tensor for computing TMScore
      values.
    pair_mask: A [num_res, num_res] mask. The TM score will only aggregate over
      masked-on entries.
    asym_id: [num_res] asymmetric unit ID (the chain ID). Only needed for ipTM
      calculation, i.e. when interface=True.
    interface: If True, the interface predicted TM score is computed. If False,
      the predicted TM score without any residue pair restrictions is computed.

  Returns:
   score: pTM or ipTM score.
  """
  num_tokens, _ = tm_adjusted_pae.shape
  if tm_adjusted_pae.shape != (num_tokens, num_tokens):
    raise ValueError(
        f'Bad tm_adjusted_pae shape, expected ({num_tokens, num_tokens}), got '
        f'{tm_adjusted_pae.shape}.'
    )

  if pair_mask.shape != (num_tokens, num_tokens):
    raise ValueError(
        f'Bad pair_mask shape, expected ({num_tokens, num_tokens}), got '
        f'{pair_mask.shape}.'
    )
  if pair_mask.dtype != bool:
    raise TypeError(f'Bad pair mask type, expected bool, got {pair_mask.dtype}')
  if asym_id.shape[0] != num_tokens:
    raise ValueError(
        f'Bad asym_id shape, expected ({num_tokens},), got {asym_id.shape}.'
    )

  # Create pair mask.
  if interface:
    pair_mask = pair_mask * (asym_id[:, None] != asym_id[None, :])

  # Ions and other ligands with colinear atoms have ill-defined frames.
  if pair_mask.sum() == 0:
    return np.nan

  normed_residue_mask = pair_mask / (
      1e-8 + np.sum(pair_mask, axis=-1, keepdims=True)
  )
  per_alignment = np.sum(tm_adjusted_pae * normed_residue_mask, axis=-1)
  return per_alignment.max()


def chain_pairwise_predicted_tm_scores(
    tm_adjusted_pae: np.ndarray,
    pair_mask: np.ndarray,
    asym_id: np.ndarray,
) -> np.ndarray:
  """Compute predicted TM (pTM) between each pair of chains independently.

  Args:
    tm_adjusted_pae: [num_res, num_res] Relevant tensor for computing TMScore
      values.
    pair_mask: A [num_res, num_res] mask specifying which frames are valid.
      Invalid frames can be the result of chains with not enough atoms (e.g.
      ions).
    asym_id: [num_res] asymmetric unit ID (the chain ID).

  Returns:
    A [num_chains, num_chains] matrix, where row i, column j indicates the
    predicted TM-score for the interface between chain i and chain j.
  """
  unique_chains = list(np.unique(asym_id))
  num_chains = len(unique_chains)
  all_pairs_iptms = np.zeros((num_chains, num_chains))
  for i, chain_i in enumerate(unique_chains):
    chain_i_mask = asym_id == chain_i
    for j, chain_j in enumerate(unique_chains[i:]):
      chain_j_mask = asym_id == chain_j
      mask = chain_i_mask | chain_j_mask
      (indices,) = np.where(mask)
      is_interface = chain_i != chain_j
      indices = np.ix_(indices, indices)
      iptm = predicted_tm_score(
          tm_adjusted_pae=tm_adjusted_pae[indices],
          pair_mask=pair_mask[indices],
          asym_id=asym_id[mask],
          interface=is_interface,
      )
      all_pairs_iptms[i, i + j] = iptm
      all_pairs_iptms[i + j, i] = iptm
  return all_pairs_iptms

def predicted_tm_score_for_IS(
    full_pae: np.ndarray,
    bin_centers: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    chain_mask: Optional[np.ndarray] = None,
    inter_chain_mask: Optional[np.ndarray] = None) -> np.ndarray:
  """Computes predicted TM alignment score.

  Args:
    full_pae: [num_res, num_res, num_bins] the full predicted alignment error from the confidence head.
    bin_centers: [num_bins] the normalized centers of the error bins
    residue_weights: [num_res] the per residue weights to use for the
      expectation used only for IS-score calculations. 

  Returns:
    ptm_score: the predicted TM alignment score for IS-score calculations.
  """
  # residue_weights has to be in [0, 1], but can be floating-point, i.e. the
  # exp. resolved head's probability.
  if residue_weights is None:
    residue_weights = np.ones(full_pae.shape[0])


  num_res = np.sum(residue_weights)
  # Clip num_res to avoid negative/undefined d0.
  clipped_num_res = jnp.maximum(num_res, 19)


  # Compute d_0(num_res) as defined by TM-score, eqn. (5) in
  # http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
  # Yang & Skolnick "Scoring function for automated
  # assessment of protein structure template quality" 2004
  d0 = 1.24 * (clipped_num_res - 15) ** (1./3) - 1.8

  # This modification may assist with prediction
  if d0 < 0.5: d0 = 0.02*num_res
  
  bin_centers = bin_centers[0, :]
  bin_centers = bin_centers[None, None, :]

  # Converts the PAE to probabilities
  pae_probs = scipy.special.softmax(full_pae, axis=-1)
  

  # Calculates the TM-Score term for every bin
  tm_per_bin = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0))
     
  predicted_tm_term = jnp.sum(pae_probs * tm_per_bin, axis=-1)

  # Assists in the calculation of the interface-similarity score
  if inter_chain_mask is not None:
      predicted_tm_term = predicted_tm_term * inter_chain_mask

  normed_residue_mask = residue_weights / (1e-8 + residue_weights.sum())

  if chain_mask is not None:
      per_alignment = np.sum(predicted_tm_term * normed_residue_mask * chain_mask, axis=-1)
      return np.asarray(per_alignment[(per_alignment * residue_weights).argmax()])
  else:
      per_alignment = np.sum(predicted_tm_term * normed_residue_mask, axis=-1)
      return np.asarray(per_alignment[(per_alignment * residue_weights).argmax()])



def predicted_interface_tm_score(
    full_pae: np.ndarray,
    bin_centers: np.ndarray,
    # residue_indices: np.ndarray,
    pos: np.ndarray,
    atom_mask: np.ndarray,
    asym_id: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    distance_threshold: Optional[int] = 4.5,
    inter_chain_mask: Optional[np.ndarray] = None) -> Dict[str, Union[np.ndarray, int]]:

  """Computes the predicted interfacial TM-score using only the residues
    that make up the interface between different chains in a protein
    complex (piTM)

  Args:
    full_pae: [num_res, num_res, num_bins] the full predicted alignment error from the confidence head.
    bin_centers: [num_bins] the normalized centers of the error bins
    residue_weights: [num_res] the per residue weights to use for the
      expectation.
    distance_threshold: maximum distance between two residue's heavy atoms
      from different chains to be considered in the interface

  Returns:
    None if target is a single chain, otherwise
    pitm_dict: a dict of np.ndarrays containing
      "score" - piTM score for the sequence
      "num_residues" - number of residues in the interface
      "num_contacts" - number of contacts along the interface of protein complex

  """
  # only calculate piTMS if there is a multi-chain target
  
  if np.all(asym_id == asym_id[0]):
    return {
      'score': np.asarray(0),
      'num_residues': np.asarray(0, dtype=np.int32),
      'num_contacts': np.asarray(0, dtype=np.int32),
    }

  residue_mask, contact_mask = get_residue_and_contact_masks(
      asym_id, pos, atom_mask, distance_threshold)

  # return 0.0 if no inter-chain contacts found
  
  if residue_mask.sum() == 0:
    return {
      'score': np.asarray(0),
      'num_residues': np.asarray(0, dtype=np.int32),
      'num_contacts': np.asarray(0, dtype=np.int32),
    }
    
    
  # Operate only over interfacial residues
  '''print("The residue weights for the piTM are: ")
    get_ranking_score(residue_weights)'''
  if residue_weights is None:
    residue_weights = np.ones(full_pae.shape[0])
  residue_weights = residue_weights * residue_mask

  # Returns piTM score and other relevant data
  return {
    'score': predicted_tm_score_for_IS(full_pae, bin_centers, residue_weights, inter_chain_mask=inter_chain_mask),
    'num_residues': np.asarray( residue_mask.sum() ),
    'num_contacts': np.asarray( contact_mask.sum() ),
  }
  

  
def make_tm_score_masks(asym_id):
  
  # Creates an inter-chain mask where only residues relevant to the TM score are marked as ones.
  # This necessitates that the best local frame from residues of other chains.
  
  full_length = len(asym_id)

  def _make_tm_score_masks(start_a, end_a):
      inter_chain_mask = np.zeros((full_length, full_length))
      chain_mask = np.zeros(full_length, dtype=int)
      chain_mask[start_a:end_a] = 1
      inter_chain_mask[start_a:end_a,:] = 1
      inter_chain_mask[:,start_a:end_a] = 1
      inter_chain_mask[start_a:end_a,start_a:end_a] = 0
      return inter_chain_mask == 1, chain_mask == 1

  tm_masks = []
  for i in range(int(asym_id.max()+1)):
    rangei = np.where(asym_id == i)[0]

    prev_id = np.roll(rangei, 1)
    
    diff = np.abs(rangei - prev_id)
  
    non_contiguous = diff > 1

    if len(non_contiguous) > 0:
        non_contiguous[0] = False
    else:
        continue

    if np.any(non_contiguous):
      # If the range is not contiguous (ie. the superchain crosses chains not
      # next to each other in the sequence) xor the masks of each contiguous sequence
      ranges = []
      start = 0
      for end in np.where(non_contiguous)[0]:
        if end == 0:
          continue
        ranges.append((rangei[start], rangei[end]))
        start = end
      inter_chain_mask = None
      chain_mask = None
      for starti, termini in ranges:
        _inter_mask, _mask  = _make_tm_score_masks(starti, termini)
        if inter_chain_mask is None:
          inter_chain_mask = _inter_mask
          chain_mask = _mask
        else:
          inter_chain_mask = np.bitwise_xor(inter_chain_mask, _inter_mask)
          chain_mask = np.bitwise_xor(chain_mask, _mask)
        tm_masks.append((inter_chain_mask, chain_mask))

    else:
      starti, termini = rangei.min(), rangei.max() + 1
      tm_masks.append(_make_tm_score_masks(starti, termini))

  return tm_masks 



def predicted_interface_score(
    full_pae: np.ndarray,
    bin_centers: np.ndarray,
    pos: np.ndarray,
    atom_mask: np.ndarray,
    asym_id: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    distance_threshold: Optional[float] = 4.5) -> Dict[str, Union[np.ndarray, int]]:
  """ Returns the predicted interface-similarity score, the number of residues in the interface, and
    the number of contacts in the complex model. This is a further tweak from piTM by
    looking the best tm-score estimate from other chains

  Args:
    full_pae: [num_res, num_res, num_bins] the full predicted alignment error from the confidence head.
    bin_centers: [num_bins] the normalized centers of the error bins
    residue_weights: [num_res] the per residue weights to use for the
      expectation.
    pos: [num_res, atom_type_num, 3] the predicted atom positions
    atom_mask: mask for atoms (each residue type has different number of atoms)
    asym_id: [num_res] a unique integer per chain indicating the chain number.
      The ordering of the input chains is arbitrary (As defined by AF-Multimer
      paper)
    residue_weights: [num_res] the per residue weights to use for the
      expectation.
    distance_threshold: maximum distance between two residue's heavy atoms
      from different chains to be considered in the interface
      
  Returns:
    None if target is a single chain, otherwise
    is_score_dict: dict of np.ndarrays containing
      "score" - piTM score for the sequence
      "num_residues" - number of residues in the interface
      "num_contacts" - number of contacts along the interface of protein complex
  """
  # only calculate the score if multi-chain target
  if np.all(asym_id == asym_id[0]):
    return {
      'score': np.asarray(0),
      'num_residues': np.asarray(0, dtype=np.int32),
      'num_contacts': np.asarray(0, dtype=np.int32),
    }

  residue_mask, contact_mask = get_residue_and_contact_masks(
      asym_id, pos, atom_mask, distance_threshold)

  # return 0.0 if no inter-chain contacts found
  if residue_mask.sum() == 0:
    return {
      'score': np.asarray(0),
      'num_residues': np.asarray(0, dtype=np.int32),
      'num_contacts': np.asarray(0, dtype=np.int32),
    }

  score = calculate_interface_score(
    full_pae, bin_centers, asym_id, residue_mask, residue_weights
  )

  return {
      'score': score,
      'num_residues': np.asarray(residue_mask.sum(), dtype=np.int32),
      'num_contacts': np.asarray(contact_mask.sum(), dtype=np.int32),
  }
  
def get_residue_and_contact_masks(
    asym_id: np.array,
    pos: np.ndarray,
    atom_mask: np.ndarray,
    distance_threshold: Optional[float] = 4.5,):
  """Computes the residue and contact masks for the IS-score

  Args:
    asym_id: [num_res] a unique integer per chain indicating the chain number.
      The ordering of the input chains is arbitrary (As defined by AF-Multimer
      paper)
    pos: [num_res, atom_type_num, 3] the predicted atom positions
    atom_mask: mask for atoms (each residue type has different number of atoms)
    distance_threshold: maximum distance between two residue's heavy atoms
      from different chains to be considered in the interface

  Returns:
    residue_mask - [num_res] array indicating if a residue is in the interface
      of the complex.
    contact_mask - [num_res, num_res] 2D array indicating which residues are
      in contact with each other (only for interface residues)
  """

  residue_mask = np.zeros(pos.shape[0]).astype(bool)
  contact_mask = np.zeros((pos.shape[0], pos.shape[0])).astype(bool)

  # calculates the minimum distance between each residue's heavy atoms
  def get_min_pairwise_dist(a, b, mask_a, mask_b):
    a = a[mask_a > 0.5]
    b = b[mask_b > 0.5]
    pairwise_dist = scipy.spatial.distance.cdist(a, b, metric='euclidean')
    return pairwise_dist.min()

  for i in range(pos.shape[0]):
      for j in range(i+1, pos.shape[0]):  # use symmetry
          if asym_id[i] != asym_id[j]:
              if atom_mask[i].sum() == 0 or atom_mask[j].sum() == 0:
                  continue
              dist = get_min_pairwise_dist(pos[i], pos[j], atom_mask[i], atom_mask[j])
              residue_mask[i] = residue_mask[i] or dist < distance_threshold
              residue_mask[j] = residue_mask[j] or dist < distance_threshold
              contact_mask[i, j] = contact_mask[i, j] or dist < distance_threshold
  return residue_mask, contact_mask
  
  
#Modification: added the interface-similarity score calculation 

def calculate_interface_score(
    full_pae: np.ndarray,
    bin_centers: np.ndarray,
    asym_id: np.ndarray,
    residue_mask: np.ndarray,
    residue_weights: Optional[np.ndarray] = None) -> Dict[str, Union[np.ndarray, int]]:
  """Directly omputes the interface-score of a complex model.

  Args:
    full_pae: [num_res, num_res, num_bins] the full_pae output from
      PredictedAlignedErrorHead.
    bin_centers: [num_bins] the centers of the error bins.
    asym_id: [num_res] a unique integer per chain indicating the chain number.
      The ordering of the input chains is arbitrary (As defined by AF-Multimer
      paper)
    residue_mask: [num_res] array indicating if a residue is in the interface
      of the complex.
    residue_weights: [num_res] the per residue weights to use for the
      expectation.
    distance_threshold: maximum distance between two residue's heavy atoms
      from different chains to be considered in the interface

  Returns:
    score - the IS-score for the model
  """
  
  #print(asym_id)

  tm_masks= make_tm_score_masks(asym_id)

  # select only interfacial residues
  if residue_weights is None:
    residue_weights = np.ones(full_pae.shape[0])
  residue_weights = residue_weights * residue_mask

  # calculate the tm-score for interface residues of each chain (or chain set) pair, and sum them.
  score = 0
  for inter_chain_mask, chain_mask in tm_masks:
    sc = predicted_tm_score_for_IS(full_pae, bin_centers, residue_weights,
           chain_mask=chain_mask, inter_chain_mask=inter_chain_mask )
    score += sc

  # return the  interaction score and other interface data
  return score
  
  
