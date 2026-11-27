/* CUDA build shim for glibc >= 2.41 (Ubuntu 25.04+).
 *
 * glibc declares the C23 math functions (cospi, sinpi, tanpi, acospi, ...) in this header, and the
 * CUDA toolkit declares the same names in its own crt/math_functions.h.  cudafe++ then rejects the
 * redeclaration ("exception specification is incompatible with that of previous function").  Turning
 * glibc's block off is safe for nvcc: the toolkit's declarations then win, and nothing in CUDA's
 * headers needs glibc's.
 *
 * Used as:  -I<this directory>   (see STRATA_CUDA_FLAGS in the install command)
 */
#undef __GLIBC_USE_IEC_60559_FUNCS_EXT_C23
#define __GLIBC_USE_IEC_60559_FUNCS_EXT_C23 0
#include_next <bits/mathcalls.h>
