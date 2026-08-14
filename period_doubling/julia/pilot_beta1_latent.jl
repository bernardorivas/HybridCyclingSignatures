#!/usr/bin/env julia
#
# Pilot: comparison-space beta_1 of the latent compass-gait lifts across a
# boxsize x stride grid. Builds only the trajectory space (no subsegment
# experiments), so the whole sweep runs in one Julia session.
#
# Usage:
#   julia --project=. pilot_beta1_latent.jl \
#     [--data-dir ../data/compass_gait_latent] \
#     [--out ../data/compass_gait_latent/pilot_beta1.csv] \
#     [--boxsizes 0.3,0.45,0.6,0.8,1.0] [--strides 1,2] [--sb-radius 1]

import Pkg
Pkg.activate(@__DIR__)

using CyclingSignatures
using DelimitedFiles
using LinearAlgebra

function getopt(args, key, default)
    i = findfirst(==(key), args)
    i === nothing ? default : args[i + 1]
end

data_dir = abspath(getopt(ARGS, "--data-dir", joinpath(@__DIR__, "..", "data", "compass_gait_latent")))
out_path = abspath(getopt(ARGS, "--out", joinpath(data_dir, "pilot_beta1.csv")))
boxsizes = parse.(Float64, split(getopt(ARGS, "--boxsizes", "0.3,0.45,0.6,0.8,1.0"), ","))
strides = parse.(Int, split(getopt(ARGS, "--strides", "1,2"), ","))
sb_radius = parse(Int, getopt(ARGS, "--sb-radius", "1"))

regimes = ["period1", "period2", "period4", "period8", "chaos"]

open(out_path, "w") do io
    println(io, "regime,stride,boxsize,beta1,n_samples,build_seconds")
    for regime in regimes
        base = "compass_$(regime)"
        X_rows = readdlm(joinpath(data_dir, base * "_positions.csv"), ' ', Float64)
        TX_rows = readdlm(joinpath(data_dir, base * "_tangents.csv"), ' ', Float64)
        X0 = Matrix(transpose(X_rows))
        TX0 = Matrix(transpose(TX_rows))
        for i in 1:size(TX0, 2)
            @views TX0[:, i] ./= norm(TX0[:, i])
        end
        for stride in strides
            X = X0[:, 1:stride:end]
            TX = TX0[:, 1:stride:end]
            for boxsize in boxsizes
                t0 = time()
                local b1
                try
                    ts = utb_trajectory_space_from_trajectory(
                        X, TX, boxsize, sb_radius; flt_max_heuristic=boxsize,
                    )
                    b1 = betti_1(ts)
                catch err
                    @warn "failed" regime stride boxsize err
                    b1 = -1
                end
                dtb = round(time() - t0, digits=2)
                line = "$regime,$stride,$boxsize,$b1,$(size(X, 2)),$dtb"
                println(io, line)
                flush(io)
                println(line)
            end
        end
    end
end
println("wrote $out_path")
