/* Sample C kernel for asv_bench_mpi tests — pure C, no Python. */
#ifdef _WIN32
#  define ASV_EXPORT __declspec(dllexport)
#else
#  define ASV_EXPORT __attribute__((visibility("default")))
#endif

/* double asv_kernel_sum(long n) — O(n) work, returns sum 0..n-1 */
ASV_EXPORT double asv_kernel_sum(long n)
{
    double s = 0.0;
    long i;
    if (n < 0)
        n = 0;
    for (i = 0; i < n; ++i)
        s += (double)i;
    return s;
}

/* empty kernel for timing noise floor */
ASV_EXPORT double asv_kernel_nop(long n)
{
    (void)n;
    return 0.0;
}
