! Sample Fortran kernel with C binding for asv_bench_mpi
function asv_kernel_fsum(n) result(s) bind(C, name="asv_kernel_fsum")
  use iso_c_binding, only: c_long, c_double
  implicit none
  integer(c_long), value :: n
  real(c_double) :: s
  integer(c_long) :: i
  s = 0.0_c_double
  if (n < 0_c_long) n = 0_c_long
  do i = 0_c_long, n - 1_c_long
    s = s + real(i, c_double)
  end do
end function asv_kernel_fsum
