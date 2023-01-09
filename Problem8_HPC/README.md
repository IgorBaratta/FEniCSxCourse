Strong and weak scaling tests

## Poisson
For the Poisson equation, a conjugate gradient (CG) solver with a classical algebraic multigrid (BoomerAMG) 
preconditioner is recommended. 
For a weak scaling test with 8 MPI processes and 500k degrees-of-freedom per process:

```bash
mpirun -np 8 ./dolfinx-scaling-test \
--problem_type poisson \
--scaling_type weak \
--ndofs 500000 \
-log_view \
-ksp_view \
-ksp_type cg \
-ksp_rtol 1.0e-8 \
-pc_type hypre \
-pc_hypre_type boomeramg \
-pc_hypre_boomeramg_strong_threshold 0.7 \
-pc_hypre_boomeramg_agg_nl 4 \
-pc_hypre_boomeramg_agg_num_paths 2 \
-options_left
```