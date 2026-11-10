#!/usr/bin/env python3
"""G3 reference: immutable page-level copy-on-write with atomic KV snapshot publication.

Correctness and update-cost experiment, NOT production lock-free/GPU/DMA safety,
formal numerical proof, accelerator speedup or a novel processor mechanism.
"""
from __future__ import annotations
from dataclasses import dataclass
from threading import Lock
from types import SimpleNamespace
from typing import Mapping, Sequence
import math
import numpy as np

from g3_guarded_mixed_precision_audit import FORMATS, guarded_admission


class WriteConflict(RuntimeError):
    """Another transaction committed before this one."""


def _sealed(arr: np.ndarray) -> np.ndarray:
    # The bytes owner prevents merely toggling numpy WRITEABLE back to True.
    a = np.ascontiguousarray(arr)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, slots=True)
class SealedPage:
    K: np.ndarray
    V: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def create(cls, K, V, *, page_size: int, d: int, vd: int, key_dtype):
        K = np.asarray(K)
        V = np.asarray(V)
        if K.dtype != key_dtype or V.dtype != np.dtype('float64'):
            raise ValueError('keys must have chosen stored dtype; values must be float64')
        if K.shape != (page_size, d) or V.shape != (page_size, vd):
            raise ValueError('incorrect page shape')
        if not np.isfinite(K.astype('float64')).all() or not np.isfinite(V).all():
            raise ValueError('nonfinite data')
        key = _sealed(K)
        val = _sealed(V)
        lo = _sealed(np.min(key, axis=0).astype('float32'))
        hi = _sealed(np.max(key, axis=0).astype('float32'))
        return cls(key, val, lo, hi)


@dataclass(frozen=True, slots=True)
class Root:
    pages: tuple[SealedPage, ...]
    epoch: int
    page_size: int
    d: int
    vd: int
    key_dtype: np.dtype


class TransactionalKV:
    def __init__(self, K, V, page_size):
        K, V = np.asarray(K), np.asarray(V)
        if not isinstance(page_size, int) or page_size <= 0:
            raise ValueError('invalid page size')
        if K.ndim != 2 or V.ndim != 2 or K.shape[0] != V.shape[0] or K.shape[0] <= 0 or K.shape[0] % page_size:
            raise ValueError('invalid key/value dimensions')
        if K.shape[1] == 0 or V.shape[1] == 0 or K.dtype not in tuple(FORMATS.values()):
            raise ValueError('unsupported stored dtype or shape')
        if V.dtype != np.dtype('float64'):
            raise ValueError('value dtype must be float64')
        d,vd=K.shape[1],V.shape[1]
        pages=tuple(SealedPage.create(K[i:i+page_size],V[i:i+page_size],page_size=page_size,d=d,vd=vd,key_dtype=K.dtype)
                    for i in range(0,len(K),page_size))
        self._lock=Lock()
        self._root=Root(pages,0,page_size,d,vd,K.dtype)

    def capture(self) -> Root:
        # Pointer capture under the same lock as publication. Subsequent reads
        # need no lock because every reachable page and root is immutable by API.
        with self._lock:
            return self._root

    def commit(self, expected: Root, *, replacements: Mapping[int,tuple] | None=None,
               appended: Sequence[tuple] = ()) -> Root:
        """Validate and seal changes privately; publish all at once if expected is current.

        O(P) new pointer tuple plus O(changed_pages * page_payload) bytes, not O(total KV bytes).
        A failed transaction changes nothing. Python immutable semantics, not a DMA guarantee.
        """
        if not isinstance(expected,Root):
            raise TypeError('expected must be a captured root')
        changes={} if replacements is None else dict(replacements)
        if not changes and not appended:
            raise ValueError('empty transaction')
        if any(not isinstance(i,int) or i<0 or i>=len(expected.pages) for i in changes):
            raise ValueError('invalid page index')
        create=lambda pair: SealedPage.create(pair[0],pair[1],page_size=expected.page_size,d=expected.d,vd=expected.vd,key_dtype=expected.key_dtype)
        # No state mutation until every replacement is fully validated.
        staged={i:create(pair) for i,pair in changes.items()}
        additions=[create(pair) for pair in appended]
        pages=list(expected.pages)
        for i,page in staged.items():
            pages[i]=page
        pages.extend(additions)
        if expected.epoch >= 2**64-1:
            raise OverflowError('64-bit epoch exhausted')
        proposed=Root(tuple(pages),expected.epoch+1,expected.page_size,expected.d,expected.vd,expected.key_dtype)
        with self._lock:
            if self._root is not expected:
                raise WriteConflict('stale expected snapshot: retry from current root')
            self._root=proposed
        return proposed


def dense_stream(root: Root, q: np.ndarray, *, chosen_page: int | None = None) -> np.ndarray:
    """Online softmax across immutable pages without concatenating entire KV arrays."""
    q=np.asarray(q)
    if q.shape != (root.d,) or q.dtype != root.key_dtype or not np.isfinite(q.astype('float64')).all():
        raise ValueError('query shape/dtype/finite contract violated')
    if chosen_page is not None and not (0<=chosen_page<len(root.pages)):
        raise ValueError('invalid page')
    pages=root.pages if chosen_page is None else (root.pages[chosen_page],)
    q64=q.astype('float64')
    m=-math.inf
    z=0.
    numer=np.zeros(root.vd,dtype=np.float64)
    for page in pages:
        scores=(page.K.astype('float64') @ q64)/math.sqrt(root.d)
        mnew=max(m,float(np.max(scores)))
        rescale=0. if m == -math.inf else math.exp(m-mnew)
        w=np.exp(scores-mnew)
        numer=numer*rescale+w@page.V
        z=z*rescale+float(np.sum(w))
        m=mnew
    return numer/z


def guarded_query(root:Root,q:np.ndarray,*,eps=0.02):
    """Existing conditional certificate; metadata and pages from ONE root.

    Stacking metadata is O(number of pages * dimension) per query and is not
    a high-performance implementation. Mathematical FP endpoint gaps remain.
    """
    q=np.asarray(q)
    if q.shape!=(root.d,) or q.dtype!=root.key_dtype or not np.isfinite(q.astype('float64')).all():
        raise ValueError('query must match stored key dtype, shape and be finite')
    meta=SimpleNamespace(lo=np.stack([p.lo for p in root.pages]),hi=np.stack([p.hi for p in root.pages]),version=root.epoch)
    cache=SimpleNamespace(d=root.d,pages=len(root.pages),version=root.epoch)
    yes,page,bound,reason=guarded_admission(cache,meta,q,eps=eps)
    output=dense_stream(root,q,chosen_page=page if yes else None)
    return output,yes,bound,reason
