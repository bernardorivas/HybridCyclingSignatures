#!/usr/bin/env julia
#
# David-style random-subsegment cycling-signature summaries for CHyLL v2 lifts.
#
# This script reads an exported unit-tangent lift
#
#   {data_dir}/{base}_positions.csv
#   {data_dir}/{base}_tangents.csv
#
# and writes dense CSV summaries suitable for publication plotting:
#
#   subsegments_{base}_rank_heatmap_rank{k}.csv
#   subsegments_{base}_rank_at_radius.csv
#   subsegments_{base}_rank1_spaces_at_radius.csv
#   subsegments_{base}_segment_starts.csv
#   subsegments_{base}_metadata.txt
#
# Usage:
#   julia --project="time series/cycling_signature" \
#     chyll_v2/cycling_signature/julia/run_subsegments.jl \
#     --data-dir chyll_v2/cycling_signature/data/rimless_wheel \
#     --base continuous_lift_chyll_v2 \
#     --boxsize 0.3 --sb-radius 1 \
#     --segment-lengths 20:10:300 --n-runs 200

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "time series", "cycling_signature"))

using CyclingSignatures
using DelimitedFiles
using LinearAlgebra
using Printf


const REPO_ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))


function parse_cli(args)
    opts = Dict{String,String}(
        "data-dir" => joinpath(REPO_ROOT, "chyll_v2", "cycling_signature", "data", "rimless_wheel"),
        "base" => "continuous_lift_chyll_v2",
        "boxsize" => "0.3",
        "sb-radius" => "1",
        "segment-lengths" => "20:10:300",
        "n-runs" => "200",
        "seed" => "20260512",
        "r-max" => "",
        "r-subdivisions" => "101",
        "eval-radius" => "",
        "max-rank" => "1",
        "max-spaces" => "8",
        "parallel-inner" => "false",
        "progress" => "true",
        "out-prefix" => "",
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
    return opts
end


function parse_bool(s::AbstractString)
    t = lowercase(strip(s))
    if t in ("1", "true", "yes", "y")
        return true
    elseif t in ("0", "false", "no", "n")
        return false
    end
    error("Cannot parse Bool from '$s'")
end


function parse_segment_lengths(spec::AbstractString)
    s = strip(spec)
    if occursin(":", s)
        parts = parse.(Int, split(s, ":"))
        if length(parts) == 2
            return collect(parts[1]:parts[2])
        elseif length(parts) == 3
            return collect(parts[1]:parts[2]:parts[3])
        end
        error("segment-lengths range must be start:stop or start:step:stop")
    end
    return parse.(Int, split(s, ","))
end


function resolve_path(path::AbstractString)
    isabspath(path) ? abspath(path) : abspath(joinpath(REPO_ROOT, path))
end


function load_lift(data_dir::AbstractString, base::AbstractString)
    pos_csv = joinpath(data_dir, base * "_positions.csv")
    tan_csv = joinpath(data_dir, base * "_tangents.csv")
    isfile(pos_csv) || error("Missing positions CSV: $pos_csv")
    isfile(tan_csv) || error("Missing tangents CSV: $tan_csv")

    X_rows = readdlm(pos_csv, ' ', Float64)
    TX_rows = readdlm(tan_csv, ' ', Float64)
    X = Matrix(transpose(X_rows))
    TX = Matrix(transpose(TX_rows))
    size(X) == size(TX) || error("positions/tangents shape mismatch: $(size(X)) vs $(size(TX))")

    nrms = [norm(t) for t in eachcol(TX)]
    min_nrm, max_nrm = extrema(nrms)
    for i in 1:size(TX, 2)
        @views TX[:, i] ./= nrms[i]
    end
    return X, TX, (pos_csv=pos_csv, tan_csv=tan_csv, min_nrm=min_nrm, max_nrm=max_nrm)
end


function write_matrix_csv(path::AbstractString, row_label::AbstractString, rows, col_label::AbstractString, cols, M)
    open(path, "w") do io
        print(io, row_label)
        for c in cols
            print(io, ",", c)
        end
        println(io)
        for (i, r) in enumerate(rows)
            print(io, r)
            for j in eachindex(cols)
                print(io, ",", M[i, j])
            end
            println(io)
        end
    end
end


function rank_heatmap(result, rank::Int, radii)
    fns = rank_distribution(result, rank)
    M = zeros(Int, length(radii), length(result.segment_lengths))
    for (i, r) in enumerate(radii)
        for (j, f) in enumerate(fns)
            M[i, j] = Int(f(r))
        end
    end
    return M
end


function write_rank_at_radius(path::AbstractString, result, ranks, eval_radius)
    rank_mats = Dict(k => rank_heatmap(result, k, [eval_radius]) for k in ranks)
    open(path, "w") do io
        print(io, "segment_length")
        for k in ranks
            print(io, ",rank", k)
        end
        println(io)
        for (j, len) in enumerate(result.segment_lengths)
            print(io, len)
            for k in ranks
                print(io, ",", rank_mats[k][1, j])
            end
            println(io)
        end
    end
end


function write_segment_starts(path::AbstractString, result)
    open(path, "w") do io
        println(io, "segment_length,run_index,start_index")
        for (i, len) in enumerate(result.segment_lengths)
            for (j, start) in enumerate(result.segment_starts[i])
                println(io, "$len,$j,$start")
            end
        end
    end
end


function matrix_to_compact_string(M)
    isempty(M) && return "[]"
    rows = String[]
    for i in 1:size(M, 1)
        push!(rows, join(M[i, :], " "))
    end
    return join(rows, ";")
end


function write_cycspace_counts(path::AbstractString, result, k::Int, eval_radius, max_spaces::Int)
    spaces, M = cycspace_length_countmatrix_at_r(result, k, eval_radius)
    n_keep = min(max_spaces, length(spaces))
    spaces = spaces[1:n_keep]
    M = M[1:n_keep, :]
    open(path, "w") do io
        print(io, "space_index,space_matrix")
        for len in result.segment_lengths
            print(io, ",", len)
        end
        println(io)
        for i in 1:size(M, 1)
            print(io, i, ",\"", matrix_to_compact_string(spaces[i]), "\"")
            for j in 1:size(M, 2)
                print(io, ",", M[i, j])
            end
            println(io)
        end
    end
    return length(spaces)
end


function write_birth_summary(path::AbstractString, result)
    open(path, "w") do io
        println(io, "segment_length,run_index,rank,births")
        for (i, len) in enumerate(result.segment_lengths)
            for (j, sig) in enumerate(result.signatures[i])
                rank = length(sig.birth_vector)
                births = join(sig.birth_vector, ";")
                println(io, "$len,$j,$rank,\"$births\"")
            end
        end
    end
end


function main(args)
    opts = parse_cli(args)
    data_dir = resolve_path(opts["data-dir"])
    base = opts["base"]
    boxsize = parse(Float64, opts["boxsize"])
    sb_radius = parse(Int, opts["sb-radius"])
    segment_lengths = parse_segment_lengths(opts["segment-lengths"])
    n_runs = parse(Int, opts["n-runs"])
    seed = parse(Int, opts["seed"])
    r_max = isempty(opts["r-max"]) ? boxsize : parse(Float64, opts["r-max"])
    r_subdivisions = parse(Int, opts["r-subdivisions"])
    eval_radius = isempty(opts["eval-radius"]) ? r_max : parse(Float64, opts["eval-radius"])
    max_rank = parse(Int, opts["max-rank"])
    max_spaces = parse(Int, opts["max-spaces"])
    parallel_inner = parse_bool(opts["parallel-inner"])
    progress = parse_bool(opts["progress"])
    out_prefix = isempty(opts["out-prefix"]) ? "subsegments_" * base : opts["out-prefix"]

    X, TX, files = load_lift(data_dir, base)
    d, N = size(X)
    maximum(segment_lengths) <= N || error("largest segment length exceeds sample count $N")

    println("data_dir: $data_dir")
    println("base: $base")
    println("lift dim=$d samples=$N")
    println("pre-renormalize tangent norm range: [$(files.min_nrm), $(files.max_nrm)]")
    println("boxsize=$boxsize sb_radius=$sb_radius r_max=$r_max eval_radius=$eval_radius")
    println("segment_lengths=$(first(segment_lengths)):...:$(last(segment_lengths)) ($(length(segment_lengths)) values), n_runs=$n_runs")

    ts = utb_trajectory_space_from_trajectory(
        X,
        TX,
        boxsize,
        sb_radius;
        flt_max_heuristic=r_max,
    )
    b1 = betti_1(ts)
    println("comparison beta_1(Y)=$b1")

    exp = RandomSubsegmentExperiment(ts, segment_lengths, n_runs, seed)
    result = run_experiment(
        exp;
        threshold=r_max,
        progress=progress,
        parallel_inner=parallel_inner,
    )

    radii = collect(range(0.0, r_max; length=r_subdivisions))
    ranks = collect(0:max_rank)

    for k in ranks
        M = rank_heatmap(result, k, radii)
        out = joinpath(data_dir, "$(out_prefix)_rank_heatmap_rank$(k).csv")
        write_matrix_csv(out, "radius", radii, "segment_length", segment_lengths, M)
        println("wrote $out")
    end

    rank_at_path = joinpath(data_dir, "$(out_prefix)_rank_at_radius.csv")
    write_rank_at_radius(rank_at_path, result, ranks, eval_radius)
    println("wrote $rank_at_path")

    spaces_path = joinpath(data_dir, "$(out_prefix)_rank1_spaces_at_radius.csv")
    n_spaces = write_cycspace_counts(spaces_path, result, 1, eval_radius, max_spaces)
    println("wrote $spaces_path (rank-1 spaces=$n_spaces)")

    starts_path = joinpath(data_dir, "$(out_prefix)_segment_starts.csv")
    write_segment_starts(starts_path, result)
    println("wrote $starts_path")

    births_path = joinpath(data_dir, "$(out_prefix)_birth_summary.csv")
    write_birth_summary(births_path, result)
    println("wrote $births_path")

    meta_path = joinpath(data_dir, "$(out_prefix)_metadata.txt")
    open(meta_path, "w") do io
        println(io, "CHyLL v2 subsegment cycling-signature experiment")
        println(io, "base=$base")
        println(io, "positions=$(files.pos_csv)")
        println(io, "tangents=$(files.tan_csv)")
        println(io, "dim=$d")
        println(io, "samples=$N")
        println(io, "boxsize=$boxsize")
        println(io, "sb_radius=$sb_radius")
        println(io, "C=$(boxsize * sb_radius)")
        println(io, "beta1_Y=$b1")
        println(io, "r_max=$r_max")
        println(io, "r_subdivisions=$r_subdivisions")
        println(io, "eval_radius=$eval_radius")
        println(io, "segment_lengths=$(join(segment_lengths, ","))")
        println(io, "n_runs=$n_runs")
        println(io, "seed=$seed")
        println(io, "parallel_inner=$parallel_inner")
        println(io, "max_rank=$max_rank")
        println(io, "max_spaces=$max_spaces")
    end
    println("wrote $meta_path")
end


main(ARGS)
