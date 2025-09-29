# jaxtrace/utils/random.py
"""
Random number generation utilities with unified JAX/NumPy interface.

Provides a consistent API that works with JAX's PRNG system when available,
falling back to NumPy's random module otherwise.
"""

from __future__ import annotations
from typing import Union, Tuple, Sequence, Optional, Any
import numpy as np

from .jax_utils import JAX_AVAILABLE

if JAX_AVAILABLE:
    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jrandom
        _jax_key_type = jax.Array
    except Exception:
        JAX_AVAILABLE = False

if not JAX_AVAILABLE:
    # Define dummy type for type hints
    _jax_key_type = Any  # type: ignore

# Type aliases
ArrayLike = Union[np.ndarray, "jax.Array"]  # type: ignore
KeyLike = Union[_jax_key_type, int, np.random.Generator]
Shape = Union[int, Sequence[int]]


def rng_key(seed: int = 42) -> KeyLike:
    """
    Create a random key/generator from a seed.
    
    Parameters
    ----------
    seed : int
        Random seed
        
    Returns
    -------
    KeyLike
        JAX PRNGKey if JAX available, otherwise NumPy Generator
    """
    if JAX_AVAILABLE:
        return jrandom.PRNGKey(seed)
    else:
        return np.random.Generator(np.random.PCG64(seed))


def split_keys(key: KeyLike, num: int = 2) -> Union[Tuple[_jax_key_type, ...], Tuple[KeyLike, ...]]:
    """
    Split a random key into multiple independent keys.
    
    Parameters
    ----------
    key : KeyLike  
        Input key/generator
    num : int
        Number of keys to generate
        
    Returns
    -------
    tuple
        Tuple of independent keys
    """
    if JAX_AVAILABLE and hasattr(key, "shape"):  # JAX key
        return tuple(jrandom.split(key, num))
    else:
        # For NumPy, create new generators with different seeds
        if isinstance(key, np.random.Generator):
            base_seed = key.bit_generator._seed_seq.entropy
        else:
            base_seed = int(key) if isinstance(key, (int, np.integer)) else 42
        
        keys = []
        for i in range(num):
            new_seed = (base_seed + i * 982451653) % (2**32)  # Mix seeds
            keys.append(np.random.Generator(np.random.PCG64(new_seed)))
        return tuple(keys)


def uniform(
    key: KeyLike,
    shape: Shape = (),
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: np.dtype = np.float32,
) -> ArrayLike:
    """
    Sample from uniform distribution.
    
    Parameters
    ----------
    key : KeyLike
        Random key/generator  
    shape : Shape
        Output shape
    minval, maxval : float
        Range bounds
    dtype : np.dtype  
        Output dtype
        
    Returns
    -------
    ArrayLike
        Random samples
    """
    shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
    
    if JAX_AVAILABLE and hasattr(key, "shape"):  # JAX key
        return jrandom.uniform(key, shape_tuple, dtype, minval, maxval)
    else:
        # NumPy generator
        if isinstance(key, np.random.Generator):
            gen = key
        else:
            gen = np.random.Generator(np.random.PCG64(int(key)))
        
        samples = gen.uniform(minval, maxval, shape_tuple)
        return samples.astype(dtype)


def normal(
    key: KeyLike,
    shape: Shape = (),
    mean: float = 0.0,
    std: float = 1.0, 
    dtype: np.dtype = np.float32,
) -> ArrayLike:
    """
    Sample from normal distribution.
    
    Parameters
    ----------
    key : KeyLike
        Random key/generator
    shape : Shape  
        Output shape
    mean, std : float
        Distribution parameters
    dtype : np.dtype
        Output dtype
        
    Returns
    -------  
    ArrayLike
        Random samples
    """
    shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
    
    if JAX_AVAILABLE and hasattr(key, "shape"):  # JAX key
        samples = jrandom.normal(key, shape_tuple, dtype)
        return samples * std + mean
    else:
        # NumPy generator
        if isinstance(key, np.random.Generator):
            gen = key
        else:
            gen = np.random.Generator(np.random.PCG64(int(key)))
            
        samples = gen.normal(mean, std, shape_tuple)
        return samples.astype(dtype)


def shuffle(key: KeyLike, array: ArrayLike, axis: int = 0) -> ArrayLike:
    """
    Shuffle array along given axis.
    
    Parameters
    ----------
    key : KeyLike
        Random key/generator
    array : ArrayLike
        Input array to shuffle
    axis : int
        Axis to shuffle along
        
    Returns
    -------
    ArrayLike  
        Shuffled array
    """
    if JAX_AVAILABLE and hasattr(key, "shape") and hasattr(array, "shape"):  # JAX
        return jrandom.permutation(key, array, axis=axis, independent=False)
    else:
        # NumPy fallback
        arr = np.asarray(array)
        if isinstance(key, np.random.Generator):
            gen = key
        else:
            gen = np.random.Generator(np.random.PCG64(int(key)))
        
        # Create permutation indices
        n = arr.shape[axis]
        indices = gen.permutation(n)
        
        # Apply permutation along specified axis
        return np.take(arr, indices, axis=axis)


# Additional utilities

def random_choice(
    key: KeyLike,
    a: Union[int, ArrayLike],
    shape: Shape = (),
    replace: bool = True,
    p: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Sample from array or range with optional probabilities.
    
    Parameters
    ----------
    key : KeyLike
        Random key/generator
    a : int or ArrayLike
        If int, sample from range(a). If array, sample from array.
    shape : Shape
        Output shape  
    replace : bool
        Whether to sample with replacement
    p : ArrayLike, optional
        Probabilities for each element
        
    Returns
    -------
    ArrayLike
        Random choices
    """
    shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
    
    if isinstance(a, int):
        choices = np.arange(a)
    else:
        choices = np.asarray(a)
    
    if isinstance(key, np.random.Generator):
        gen = key
    else:
        gen = np.random.Generator(np.random.PCG64(int(key)))
    
    # Use NumPy's choice (JAX doesn't have a direct equivalent)
    if p is not None:
        p = np.asarray(p)
    
    if shape_tuple == ():
        return gen.choice(choices, size=None, replace=replace, p=p)
    else:
        return gen.choice(choices, size=shape_tuple, replace=replace, p=p)


def random_integers(
    key: KeyLike,
    low: int,
    high: Optional[int] = None, 
    shape: Shape = (),
    dtype: np.dtype = np.int32,
) -> ArrayLike:
    """
    Sample random integers from [low, high).
    
    Parameters
    ----------
    key : KeyLike
        Random key/generator
    low : int
        Lower bound (inclusive)
    high : int, optional
        Upper bound (exclusive). If None, sample from [0, low)
    shape : Shape
        Output shape
    dtype : np.dtype
        Integer dtype
        
    Returns
    -------
    ArrayLike
        Random integers
    """
    if high is None:
        high = low
        low = 0
        
    shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
    
    if JAX_AVAILABLE and hasattr(key, "shape"):  # JAX key
        return jrandom.randint(key, shape_tuple, low, high, dtype)
    else:
        # NumPy generator  
        if isinstance(key, np.random.Generator):
            gen = key
        else:
            gen = np.random.Generator(np.random.PCG64(int(key)))
            
        samples = gen.integers(low, high, shape_tuple, dtype=dtype)
        return samples