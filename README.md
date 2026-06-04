# Linear Algebra
A Basic linear algebra data structure

### Contents
Available data structures:
- Matrix
- Vector
- LU factorization Matrix
- Woodbury cache Matrix
- Partitioned Woodbury cache Matrix

Those data structures are availabe in stack (array-based) & heap version (Box<[f64]>).

### Usage
This library originally used for audio processing, therefore all the implementations on the heap version requires the caller to provide the target matrix.
If you're building for MacOs & Linux, the stack version will probably just fine (and it's faster). The downside is there are more const generics you have to specify.
If you're building for Windows, the heap-based is the safest bet, but it's kinda slower. However I haven't tested it only for pure lu-factorization.

### Neon SIMD
For Aarch64 architecture, if Neon is enabled, I also provided the Neon instruction version (it will be automatically compiled).
