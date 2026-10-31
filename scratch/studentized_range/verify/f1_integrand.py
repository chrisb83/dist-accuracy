# Evaluate SciPy's Cython pdf integrand directly at single (z, s) nodes (z = x[0], s = x[1]).
import ctypes, numpy as np
from scipy.stats import _stats
from scipy import LowLevelCallable
k, df, q = 1.5, 10.0, 1.0
lc = _stats._studentized_range_pdf_logconst(k, df)
ud = np.array([q, k, df, lc], float)
llc = LowLevelCallable.from_cython(_stats, '_studentized_range_pdf', ud.ctypes.data_as(ctypes.c_void_p))
cap = llc.function
PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.restype = ctypes.c_void_p
PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
PyCapsule_GetName = ctypes.pythonapi.PyCapsule_GetName
PyCapsule_GetName.restype = ctypes.c_char_p; PyCapsule_GetName.argtypes = [ctypes.py_object]
p = PyCapsule_GetPointer(cap, PyCapsule_GetName(cap))
f = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)(p)
for z, s in [(0.0, 1.0), (5.0, 1.0), (8.0, 1.0), (9.0, 1.0), (-40.0, 1.0), (0.0, 1e-20)]:
    xs = (ctypes.c_double*2)(z, s)
    print(z, s, f(2, xs, ud.ctypes.data))
