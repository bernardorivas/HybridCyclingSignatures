#!/usr/bin/env julia
#
# Extended cycling-signature driver for Roessler period-doubling study.
# Adds stride subsampling, rank-2 spaces, inclusion matrices, and multi-radius support.
#
# This script reads an exported unit-tangent lift
#
#   {data_dir}/{base}_positions.csv
#   {data_dir}/{base}_tangents.csv
#
# and writes dense CSV summaries suitable for publication plotting.
#
# Usage:
#   julia --project=/path/to/project run_signatures.jl \
#     --data-dir /path/to/data \
#     --base circle \
#     --boxsize 0.5 --sb-radius 1 \
#     --segment-lengths 100,300 \
#     --n-runs 10 \
#     --stride 2 \
#     --max-rank 3 \
#     --max-spaces 8 \
#     --max-spaces-2 8 \
#     --eval-radius 0.3,0.45 \
#     --out-dir /path/to/data/signatures
#
# --eval-radius defaults to r-max; --out-dir defaults to --data-dir.

import Pkg
Pkg.activate(@__DIR__)

using CyclingSignatures
using DelimitedFiles
using LinearAlgebra
using Printf


const REPO_ROOT = abspath(@__DIR__)


function parse_cli(args)
    opts = Dict{String,String}(
        "data-dir" => joinpath(REPO_ROOT, "data"),
        "base" => "circle",
        "boxsize" => "0.5",
        "sb-radius" => "1",
        "segment-lengths" => "100,300",
        "n-runs" => "10",
        "seed" => "20260512",
        "r-max" => "",
        "r-subdivisions" => "101",
        "eval-radius" => "",
        "max-rank" => "3",
        "max-spaces" => "8",
        "max-spaces-2" => "8",
        "stride" => "1",
        "parallel-inner" => "false",
        "progress" => "true",
        "out-dir" => "",
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


function parse_eval_radii(spec::AbstractString)
    return parse.(Float64, split(strip(spec), ","))
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
    return spaces, n_keep
end


function write_inclusion_matrix(path::AbstractString, spaces1::Vector, spaces2::Vector)
    M = cycspace_inclusion_matrix(spaces1, spaces2)
    open(path, "w") do io
        print(io, "rank1_index,rank1_space")
        for space2_idx in 1:length(spaces2)
            print(io, ",", space2_idx)
        end
        println(io)
        for (i, space1) in enumerate(spaces1)
            print(io, i, ",\"", matrix_to_compact_string(space1), "\"")
            for j in 1:length(spaces2)
                print(io, ",", Int(M[i, j]))
            end
            println(io)
        end
    end
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


function radius_suffix(r::Float64)
    s = @sprintf("%.10f", r)
    s = rstrip(s, '0')
    s = rstrip(s, '.')
    return replace(s, "." => "p")
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
    stride = parse(Int, opts["stride"])
    r_max = isempty(opts["r-max"]) ? boxsize : parse(Float64, opts["r-max"])
    r_subdivisions = parse(Int, opts["r-subdivisions"])
    eval_radii = isempty(opts["eval-radius"]) ? [r_max] : parse_eval_radii(opts["eval-radius"])
    max_rank = parse(Int, opts["max-rank"])
    max_spaces = parse(Int, opts["max-spaces"])
    max_spaces_2 = parse(Int, opts["max-spaces-2"])
    parallel_inner = parse_bool(opts["parallel-inner"])
    progress = parse_bool(opts["progress"])
    out_dir = isempty(opts["out-dir"]) ? data_dir : resolve_path(opts["out-dir"])
    out_prefix = isempty(opts["out-prefix"]) ? "subsegments_" * base : opts["out-prefix"]
    mkpath(out_dir)

    println("=== Cycling Signature Analysis ===")
    println("data_dir: $data_dir")
    println("base: $base")
    println("out_dir: $out_dir")
    println("out_prefix: $out_prefix")

    t_load_start = time()
    X, TX, files = load_lift(data_dir, base)
    d, N_orig = size(X)

    # Apply stride subsampling
    X = X[:, 1:stride:end]
    TX = TX[:, 1:stride:end]
    N = size(X, 2)
    t_load_end = time()

    println("lift dim=$d samples=$N_orig (pre-stride), $N (post-stride, stride=$stride)")
    println("pre-renormalize tangent norm range: [$(files.min_nrm), $(files.max_nrm)]")
    println("boxsize=$boxsize sb_radius=$sb_radius r_max=$r_max")
    println("segment_lengths=$(first(segment_lengths)):...:$(last(segment_lengths)) ($(length(segment_lengths)) values), n_runs=$n_runs")
    println("eval_radii: $(join(eval_radii, ", "))")
    println("max_rank=$max_rank max_spaces=$max_spaces max_spaces_2=$max_spaces_2")
    println("Load time: $(round(t_load_end - t_load_start, digits=3))s")

    maximum(segment_lengths) <= N || error("largest segment length exceeds sample count $N")

    t_space_start = time()
    ts = utb_trajectory_space_from_trajectory(
        X,
        TX,
        boxsize,
        sb_radius;
        flt_max_heuristic=r_max,
    )
    b1 = betti_1(ts)
    t_space_end = time()
    println("\nComparison space: beta_1(Y)=$b1")
    println("Trajectory space build time: $(round(t_space_end - t_space_start, digits=3))s")

    t_exp_start = time()
    exp = RandomSubsegmentExperiment(ts, segment_lengths, n_runs, seed)
    result = run_experiment(
        exp;
        threshold=r_max,
        progress=progress,
        parallel_inner=parallel_inner,
    )
    t_exp_end = time()
    println("Experiment time: $(round(t_exp_end - t_exp_start, digits=3))s")

    radii = collect(range(0.0, r_max; length=r_subdivisions))
    ranks = collect(0:max_rank)

    # Write heatmaps (radius-independent, full radius grid)
    t_hm_start = time()
    for k in ranks
        M = rank_heatmap(result, k, radii)
        out = joinpath(out_dir, "$(out_prefix)_rank_heatmap_rank$(k).csv")
        write_matrix_csv(out, "radius", radii, "segment_length", segment_lengths, M)
        println("wrote $out")
    end
    t_hm_end = time()
    println("Heatmap time: $(round(t_hm_end - t_hm_start, digits=3))s")

    # Write per-radius outputs
    single_radius = length(eval_radii) == 1

    for eval_radius in eval_radii
        suffix = single_radius ? "" : "_r$(radius_suffix(eval_radius))"

        rank_at_path = joinpath(out_dir, "$(out_prefix)_rank_at_radius$(suffix).csv")
        write_rank_at_radius(rank_at_path, result, ranks, eval_radius)
        println("wrote $rank_at_path")

        t_rank1_start = time()
        spaces1_path = joinpath(out_dir, "$(out_prefix)_rank1_spaces_at_radius$(suffix).csv")
        spaces1, n_spaces1 = write_cycspace_counts(spaces1_path, result, 1, eval_radius, max_spaces)
        t_rank1_end = time()
        println("wrote $spaces1_path (rank-1 spaces=$n_spaces1, time=$(round(t_rank1_end - t_rank1_start, digits=3))s)")

        t_rank2_start = time()
        spaces2_path = joinpath(out_dir, "$(out_prefix)_rank2_spaces_at_radius$(suffix).csv")
        spaces2, n_spaces2 = write_cycspace_counts(spaces2_path, result, 2, eval_radius, max_spaces_2)
        t_rank2_end = time()
        println("wrote $spaces2_path (rank-2 spaces=$n_spaces2, time=$(round(t_rank2_end - t_rank2_start, digits=3))s)")

        # Write inclusion matrix if we have rank-2 spaces
        if n_spaces2 > 0
            t_incl_start = time()
            incl_path = joinpath(out_dir, "$(out_prefix)_s21_inclusion$(suffix).csv")
            write_inclusion_matrix(incl_path, spaces1, spaces2)
            t_incl_end = time()
            println("wrote $incl_path (time=$(round(t_incl_end - t_incl_start, digits=3))s)")
        else
            # Write header-only file
            incl_path = joinpath(out_dir, "$(out_prefix)_s21_inclusion$(suffix).csv")
            open(incl_path, "w") do io
                println(io, "rank1_index,rank1_space")
            end
            println("wrote $incl_path (empty, no rank-2 spaces)")
        end
    end

    t_segs_start = time()
    starts_path = joinpath(out_dir, "$(out_prefix)_segment_starts.csv")
    write_segment_starts(starts_path, result)
    t_segs_end = time()
    println("wrote $starts_path (time=$(round(t_segs_end - t_segs_start, digits=3))s)")

    t_birth_start = time()
    births_path = joinpath(out_dir, "$(out_prefix)_birth_summary.csv")
    write_birth_summary(births_path, result)
    t_birth_end = time()
    println("wrote $births_path (time=$(round(t_birth_end - t_birth_start, digits=3))s)")

    t_meta_start = time()
    meta_path = joinpath(out_dir, "$(out_prefix)_metadata.txt")
    open(meta_path, "w") do io
        println(io, "Cycling signature analysis -- period doubling study")
        println(io, "base=$base")
        println(io, "positions=$(files.pos_csv)")
        println(io, "tangents=$(files.tan_csv)")
        println(io, "dim=$d")
        println(io, "samples=$N_orig")
        println(io, "samples_post_stride=$N")
        println(io, "stride=$stride")
        println(io, "boxsize=$boxsize")
        println(io, "sb_radius=$sb_radius")
        println(io, "C=$(boxsize * sb_radius)")
        println(io, "beta1_Y=$b1")
        println(io, "r_max=$r_max")
        println(io, "r_subdivisions=$r_subdivisions")
        println(io, "eval_radii=$(join(eval_radii, ","))")
        println(io, "segment_lengths=$(join(segment_lengths, ","))")
        println(io, "n_runs=$n_runs")
        println(io, "seed=$seed")
        println(io, "parallel_inner=$parallel_inner")
        println(io, "max_rank=$max_rank")
        println(io, "max_spaces=$max_spaces")
        println(io, "max_spaces_2=$max_spaces_2")
    end
    t_meta_end = time()
    println("wrote $meta_path (time=$(round(t_meta_end - t_meta_start, digits=3))s)")

    total_time = t_meta_end - t_load_start
    println("\nTotal time: $(round(total_time, digits=3))s")
end


main(ARGS)
