#!/usr/bin/env julia
#
# Cycling-signature computation on the compass-gait analytic lift.
#
# Inputs (from data/compass_gait/):
#   continuous_lift_analytic*_positions.csv  (T rows x d columns)
#   continuous_lift_analytic*_tangents.csv   (T rows x d columns)
#
# Output:
#   data/compass_gait/barcode_H1_analytic*.csv  (bar index, birth, [death])
#
# Usage:
#   julia --project="time series/cycling_signature" "time series/cycling_signature/run_cycling_signature.jl"
#   julia --project="time series/cycling_signature" "time series/cycling_signature/run_cycling_signature.jl" --suffix _eta2

import Pkg
Pkg.activate(@__DIR__)

REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
CS_LOCAL = joinpath(REPO_ROOT, "local_docs", "CyclingSignatures.jl-main", "CyclingSignatures.jl-main")

if !isfile(joinpath(@__DIR__, "Manifest.toml"))
    println("First run: installing deps (may download packages; ~1-5 min) ...")
    # StepFunctions.jl is an unregistered dep of CyclingSignatures; must be added by URL FIRST.
    Pkg.add(url="https://github.com/davidhien/StepFunctions.jl")
    # Then develop CyclingSignatures from the local source tree.
    Pkg.develop(path=CS_LOCAL)
    Pkg.instantiate()
end

using CyclingSignatures
using DelimitedFiles
using LinearAlgebra

function parse_args_cli(args)
    suffix = ""
    base = "continuous_lift_analytic"
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--suffix"
            i == length(args) && error("--suffix requires a value")
            suffix = args[i + 1]
            i += 2
        elseif arg == "--base"
            i == length(args) && error("--base requires a value")
            base = args[i + 1]
            i += 2
        else
            error("Unknown argument: $arg")
        end
    end
    return suffix, base
end

suffix, base_prefix = parse_args_cli(ARGS)
DATA_DIR = joinpath(REPO_ROOT, "data", "compass_gait")
base_name = base_prefix * suffix
pos_csv = joinpath(DATA_DIR, base_name * "_positions.csv")
tan_csv = joinpath(DATA_DIR, base_name * "_tangents.csv")

X_rows  = readdlm(pos_csv, ' ', Float64)
TX_rows = readdlm(tan_csv, ' ', Float64)

X  = Matrix(transpose(X_rows))    # (d, N)
TX = Matrix(transpose(TX_rows))   # (d, N)
d, N = size(X)
println("Loaded compass lift: dim=$d, samples=$N")

# Renormalize tangents to unit length in Julia: the CSV roundtrip can drop a
# few ULPs, and David's `utb_trajectory_space_from_trajectory` enforces
# isapprox(norm(t), 1.0) at Julia's default tolerance (~1.5e-8).
nrms = [norm(t) for t in eachcol(TX)]
if !isempty(nrms)
    min_nrm, max_nrm = extrema(nrms)
    println("pre-renormalize tangent norm range: [$min_nrm, $max_nrm]")
    for i in 1:N
        @views TX[:, i] ./= nrms[i]
    end
end
for (i, t) in enumerate(eachcol(TX))
    nrm = norm(t)
    if !isapprox(nrm, 1.0)
        error("Sphere Bundle component $i is not unit (|t|=$nrm)")
    end
end

# Parameter sweep: the first run at boxsize=0.2 gave beta_1(Y)=0 (simply
# connected cover), so scan a range of (boxsize, sb_radius) pairs to locate
# the regime where the comparison space captures the orbit's loop.
param_grid = [
    (0.30, 1), (0.20, 1), (0.10, 1), (0.05, 1),
    (0.30, 2), (0.20, 2), (0.10, 2), (0.05, 2),
    (0.30, 4), (0.20, 4), (0.10, 4), (0.05, 4),
]

results = Vector{NamedTuple}()
for (boxsize, sb_radius) in param_grid
    ts = utb_trajectory_space_from_trajectory(X, TX, boxsize, sb_radius)
    b1 = betti_1(ts)
    sig = cycling_signature(ts, (1, N))
    rank_sig = dimension(sig)
    births = sig.birth_vector
    maxbirth = isempty(births) ? NaN : maximum(births)
    push!(results, (;
        boxsize=boxsize, sb_radius=sb_radius,
        C=boxsize*sb_radius, b1=b1, rank=rank_sig,
        n_births=length(births), max_birth=maxbirth,
    ))
    println(lpad("boxsize=$(boxsize)", 14), "  ",
            lpad("sb_radius=$(sb_radius)", 13), "  ",
            lpad("C=$(boxsize*sb_radius)", 8), "  ",
            lpad("beta1(Y)=$(b1)", 12), "  ",
            lpad("rank=$(rank_sig)", 8), "  ",
            "n_births=$(length(births))  max_birth=$(maxbirth)")
end

# Pick the first (boxsize, sb_radius) where beta_1(Y) > 0 AND rank > 0 for the canonical write.
canonical_idx = findfirst(r -> r.b1 > 0 && r.rank > 0, results)
if canonical_idx === nothing
    # Fall back to the first non-trivial comparison space (even if rank=0)
    canonical_idx = findfirst(r -> r.b1 > 0, results)
end
if canonical_idx === nothing
    println()
    println("WARN: every tested (boxsize, sb_radius) gave beta_1(Y) = 0. ")
    println("The cubical cover never captures the orbit's loop at these scales.")
    canonical_idx = 1  # arbitrary, will record trivial result
end

best = results[canonical_idx]
boxsize, sb_radius = best.boxsize, best.sb_radius
println()
println("Canonical cell: boxsize=$boxsize, sb_radius=$sb_radius, beta_1(Y)=$(best.b1), rank=$(best.rank)")

# Recompute at canonical cell to get the birth vector
ts_final = utb_trajectory_space_from_trajectory(X, TX, boxsize, sb_radius)
sig_final = cycling_signature(ts_final, (1, N))
births_final = sig_final.birth_vector

# Barcode filename mirrors the input base (drop the "continuous_lift_" prefix):
#   continuous_lift_analytic       -> barcode_H1_analytic.csv
#   continuous_lift_analytic_eta2  -> barcode_H1_analytic_eta2.csv
#   continuous_lift_relaxed        -> barcode_H1_relaxed.csv
barcode_stem = replace(base_name, "continuous_lift_" => "barcode_H1_")
barcode_csv = joinpath(DATA_DIR, barcode_stem * ".csv")
open(barcode_csv, "w") do io
    println(io, "# Cycling-signature H_1 births on compass lift (base_name=$(base_name))")
    println(io, "# full grid of (boxsize, sb_radius, C=boxsize*sb_radius, beta_1(Y), rank, n_births, max_birth) shown below,")
    println(io, "# followed by the explicit birth vector at the canonical cell.")
    println(io, "#")
    println(io, "# boxsize,sb_radius,C,beta_1,rank,n_births,max_birth")
    for r in results
        println(io, "# $(r.boxsize),$(r.sb_radius),$(r.C),$(r.b1),$(r.rank),$(r.n_births),$(r.max_birth)")
    end
    println(io, "#")
    println(io, "# Canonical cell (first (boxsize, sb_radius) with beta_1>0 and rank>0):")
    println(io, "# boxsize=$boxsize, sb_radius=$sb_radius")
    println(io, "#")
    println(io, "bar_index,birth")
    for (i, b) in enumerate(births_final)
        println(io, "$i,$b")
    end
end
println("wrote $(barcode_csv)")
