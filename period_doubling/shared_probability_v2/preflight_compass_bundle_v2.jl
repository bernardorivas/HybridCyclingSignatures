#!/usr/bin/env julia
# Read-only beta_1 and curve-bound gate for a hashed Compass analysis stream.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "julia"))

using CyclingSignatures
using DelimitedFiles
using LinearAlgebra


function parse_cli(args)
    values = Dict(
        "positions" => "",
        "tangents" => "",
        "stride" => "1",
        "tangent-normalization" => "linf",
        "boxsize" => "5",
        "sb-radius" => "1",
        "metric-c" => "5",
        "r-max" => "5",
        "expected-beta1" => "",
    )
    index = 1
    while index <= length(args)
        key = args[index]
        startswith(key, "--") || error("unexpected argument: $key")
        name = key[3:end]
        haskey(values, name) || error("unknown option: $key")
        index < length(args) || error("$key needs a value")
        values[name] = args[index + 1]
        index += 2
    end
    isempty(values["positions"]) && error("--positions is required")
    isempty(values["tangents"]) && error("--tangents is required")
    return values
end


function main(args)
    options = parse_cli(args)
    positions = readdlm(realpath(options["positions"]), ' ', Float64)
    tangents = readdlm(realpath(options["tangents"]), ' ', Float64)
    size(positions) == size(tangents) || error("position/tangent shape mismatch")
    stride = parse(Int, options["stride"])
    stride > 0 || error("stride must be positive")
    X = Matrix(transpose(positions))[ :, 1:stride:end]
    TX = Matrix(transpose(tangents))[ :, 1:stride:end]
    normalization = lowercase(options["tangent-normalization"])
    normalization == "linf" || error("this gate requires linf normalization")
    for tangent in eachcol(TX)
        value = norm(tangent, Inf)
        isfinite(value) && value > 0 || error("invalid tangent")
        tangent ./= value
    end
    metric_c = parse(Float64, options["metric-c"])
    bound = 0.0
    for index in 1:size(X, 2)-1
        bound = max(
            bound,
            norm(view(X, :, index + 1) - view(X, :, index)),
            metric_c * norm(view(TX, :, index + 1) - view(TX, :, index)),
        )
    end
    r_max = parse(Float64, options["r-max"])
    bound < r_max || error("curve bound $bound is not below r_max=$r_max")
    lifted = [X; TX]
    comparison = sb_cubical_vr_comparison_space_via_cover(
        lifted,
        parse(Float64, options["boxsize"]),
        parse(Int, options["sb-radius"]),
    )
    beta1 = betti_1(comparison)
    if !isempty(options["expected-beta1"])
        expected = parse(Int, options["expected-beta1"])
        beta1 == expected || error("beta1(Y)=$beta1, expected $expected")
    end
    println("beta1_Y=$beta1")
    println("global_curve_bound=$bound")
    println("analysis_samples=$(size(X, 2))")
    println("dimension=$(size(X, 1))")
    println("PREFLIGHT ONLY: no cycling signatures computed and no files written")
end


main(ARGS)
