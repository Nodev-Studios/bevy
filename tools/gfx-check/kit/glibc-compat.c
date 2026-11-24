// Binds the kit's Linux testbed to the oldest versions of the libm functions that glibc 2.35, 2.43 and 2.44
// re-versioned, so the binary built on a rolling distribution also starts on older ones (Ubuntu 22.04 has 2.35).
// Linked with `-Wl,--wrap=<name>` for each function below (tools/gfx-check/kit/make-kit.sh): every call to `f`
// in the link becomes a call to `__wrap_f`, which calls `f@GLIBC_2.2.5`. Same results to the last bit is not
// promised: the new versions are faster or more accurate implementations of the same functions.

#define COMPAT(type, name, params, args)                                  \
    type name##_compat params;                                          \
    __asm__(".symver " #name "_compat, " #name "@GLIBC_2.2.5");         \
    type __wrap_##name params { return name##_compat args; }

COMPAT(float, acosf, (float x), (x))
COMPAT(float, asinf, (float x), (x))
COMPAT(float, atan2f, (float y, float x), (y, x))
COMPAT(float, log10f, (float x), (x))
COMPAT(float, acoshf, (float x), (x))
COMPAT(float, coshf, (float x), (x))
COMPAT(float, sinhf, (float x), (x))
COMPAT(float, hypotf, (float x, float y), (x, y))
COMPAT(double, cosh, (double x), (x))
COMPAT(double, sinh, (double x), (x))
