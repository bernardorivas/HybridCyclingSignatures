#!/usr/bin/env julia
#
# Smoke test / regression check for the period_doubling Julia environment.
#
# Synthesizes a noisy unit-tangent lift of a circle (3 loops), builds the
# trajectory space, and exercises the same CyclingSignatures API surface as
# chyll_v2/cycling_signature/julia/run_subsegments.jl:
#
#   utb_trajectory_space_from_trajectory -> betti_1 ->
#   RandomSubsegmentExperiment -> run_experiment -> rank_distribution ->
#   cycspace_length_countmatrix_at_r -> cycspace_inclusion_matrix
#
# Sampling: 1200 points over 3 loops, i.e. 400 samples per loop. Segments of
# length 100 and 300 cover at most 3/4 of a loop and therefore detect no cycle
# (rank-1 count 0); length 450 wraps a full loop and every run detects the
# single cycle. Length 450 is included so that the cycling-space list is
# nonempty and the inclusion-matrix diagonal check is not vacuous.
#
# Expected: betti_1 == 1; rank-1 counts at r=0.4 == [0, 0, 20];
# size(M) == (1, 3) with M == [0 0 20]; inclusion-matrix diagonal all true.
#
# Usage:
#   julia --project=code/period_doubling/julia code/period_doubling/julia/smoke_test.jl

import Pkg
Pkg.activate(@__DIR__)

using CyclingSignatures
using LinearAlgebra
using Random

# --- synthetic noisy circle lift -------------------------------------------
rng = MersenneTwister(42)
t = range(0, 6pi, length=1200)
X = 3 * [cos.(t)'; sin.(t)'] .+ 0.01 .* randn(rng, 2, length(t))
TX = [-sin.(t)'; cos.(t)']
for i in 1:size(TX, 2)
    @views TX[:, i] ./= norm(TX[:, i])
end

# --- trajectory space + comparison Betti number ----------------------------
ts = utb_trajectory_space_from_trajectory(X, TX, 0.5, 1; flt_max_heuristic=0.5)
b1 = betti_1(ts)
println("betti_1 = ", b1)
b1 == 1 || error("smoke test FAILED: expected betti_1 == 1, got $b1")

# --- random-subsegment experiment ------------------------------------------
segment_lengths = [100, 300, 450]
n_runs = 20
experiment = RandomSubsegmentExperiment(ts, segment_lengths, n_runs, 1234)
t_exp = @elapsed result = run_experiment(experiment; threshold=0.5, progress=false)
println("run_experiment wall time: ", round(t_exp, digits=3), " s")

# rank-1 counts (step functions in the radius r), evaluated at r = 0.4
fns = rank_distribution(result, 1)
rank1_counts = [Int(f(0.4)) for f in fns]
println("rank-1 counts at r=0.4 (lengths ", segment_lengths, "): ", rank1_counts)
rank1_counts == [0, 0, 20] ||
    error("smoke test FAILED: expected rank-1 counts [0, 0, 20], got $rank1_counts")

# cycling-space count matrix at r = 0.4
spaces, M = cycspace_length_countmatrix_at_r(result, 1, 0.4)
println("n rank-1 spaces = ", length(spaces), ", size(M) = ", size(M))

# inclusion matrix of the spaces against themselves: diagonal must be true
inc = cycspace_inclusion_matrix(spaces, spaces)
diag_inc = [inc[i, i] for i in 1:size(inc, 1)]
println("diag(inclusion) = ", diag_inc)
isempty(diag_inc) && error("smoke test FAILED: no cycling spaces found (vacuous check)")
all(diag_inc) || error("smoke test FAILED: inclusion-matrix diagonal not all true")

println("SMOKE TEST PASSED")
