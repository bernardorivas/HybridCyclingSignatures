#!/usr/bin/env julia
#
# Sweep (boxsize, sb_radius) on a CHyLL v2 lift and report beta_1 of the
# UTB comparison space Y. This is a quick scale-sensitivity probe: if
# beta_1(Y) is genuinely the rank of H_1 of the underlying continuous
# space, it should stabilise over a band of (boxsize, sb_radius); if it
# is a Vietoris-Rips artifact, it should fluctuate.
#
# Output CSV columns: boxsize, sb_radius, C, beta1_Y, n_cells, n_pts.
#
# Usage:
#   julia --project="time series/cycling_signature" \
#     chyll_v2/cycling_signature/julia/sweep_beta1_comparison_space.jl \
#     --data-dir chyll_v2/cycling_signature/data/bouncing_ball \
#     --base continuous_lift_chyll_v2_bb_phaseB \
#     --boxsizes "0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.75,1.00" \
#     --sb-radii "1,2,3" \
#     --r-max 0.5 \
#     --out subsegments_chyll_v2_bb_phaseB_beta1_sweep.csv

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "time series", "cycling_signature"))

using CyclingSignatures
using DelimitedFiles
using LinearAlgebra
using Printf


const REPO_ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))


function parse_cli(args)
    opts = Dict{String,String}(
        "data-dir" => "",
        "base" => "",
        "boxsizes" => "0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.75,1.00",
        "sb-radii" => "1,2,3",
        "r-max" => "",
        "out" => "",
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = arg[3:end]
            haskey(opts, key) || error("Unknown argument: --$key")
            i == length(args) && error("--$key requires a value")
            opts[key] = args[i + 1]
            i += 2
        else
            error("Unknown positional argument: $arg")
        end
    end
    isempty(opts["data-dir"]) && error("--data-dir is required")
    isempty(opts["base"]) && error("--base is required")
    return opts
end


function resolve_path(path::AbstractString)
    isabspath(path) ? abspath(path) : abspath(joinpath(REPO_ROOT, path))
end


function load_lift(data_dir::AbstractString, base::AbstractString)
    pos_csv = joinpath(data_dir, base * "_positions.csv")
    tan_csv = joinpath(data_dir, base * "_tangents.csv")
    X_rows = readdlm(pos_csv, ' ', Float64)
    TX_rows = readdlm(tan_csv, ' ', Float64)
    X = Matrix(transpose(X_rows))
    TX = Matrix(transpose(TX_rows))
    for i in 1:size(TX, 2)
        n = norm(TX[:, i])
        if n > 0
            @views TX[:, i] ./= n
        end
    end
    return X, TX
end


function main(args)
    opts = parse_cli(args)
    data_dir = resolve_path(opts["data-dir"])
    base = opts["base"]
    boxsizes = parse.(Float64, split(opts["boxsizes"], ","))
    sb_radii = parse.(Int, split(opts["sb-radii"], ","))
    out_csv = isempty(opts["out"]) ?
        joinpath(data_dir, "$(base)_beta1_sweep.csv") :
        joinpath(data_dir, opts["out"])

    X, TX = load_lift(data_dir, base)
    d, N = size(X)
    println("base=$base dim=$d samples=$N")

    open(out_csv, "w") do io
        println(io, "boxsize,sb_radius,C,beta1_Y")
        for bs in boxsizes
            r_max_local = isempty(opts["r-max"]) ? bs : parse(Float64, opts["r-max"])
            for sb in sb_radii
                ts = utb_trajectory_space_from_trajectory(
                    X, TX, bs, sb;
                    flt_max_heuristic=r_max_local,
                )
                b1 = betti_1(ts)
                C = bs * sb
                @printf("boxsize=%.4f sb_radius=%d C=%.4f beta1_Y=%d\n", bs, sb, C, b1)
                println(io, "$bs,$sb,$C,$b1")
                flush(io)
            end
        end
    end
    println("wrote $out_csv")
end


main(ARGS)
