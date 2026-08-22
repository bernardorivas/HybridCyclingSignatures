#!/usr/bin/env julia
#
# Shared random-subsegment cycling-probability kernel.
#
# Both continuous and learned-suspension trajectories enter through the same
# positions/tangents interface.  The defaults reproduce the documented
# Rössler subsampling protocol: lengths 100:20:1200, 20 independently
# sampled starts per length, F_43 coefficients, boxsize 5, sphere-cover
# radius 1, dynamic-distance coefficient C=5, and filtration radii 0:5/200:5.
#
# This file intentionally does not render figures or generate trajectories.
# It writes only trial births, sampled starts, the rank-zero heatmap, and
# provenance metadata.

import Pkg
Pkg.activate(@__DIR__)

using CyclingSignatures
using DelimitedFiles
using LinearAlgebra
using SHA


const CODE_ROOT = realpath(joinpath(@__DIR__, "..", ".."))
const SAFE_OUTPUT_ROOT = joinpath(
    CODE_ROOT,
    "experiments_planned",
    "outputs",
    "shared_coauthor_protocol",
)


function print_help()
    println("""
Usage:
  julia --project=period_doubling/julia \\
    period_doubling/julia/run_shared_probability.jl \\
    --positions PATH --tangents PATH --out-dir PATH [options]

Required:
  --positions PATH              Space-delimited rows=samples position CSV.
  --tangents PATH               Space-delimited rows=samples tangent CSV.
  --out-dir PATH                Output directory below
                                experiments_planned/outputs/
                                shared_coauthor_protocol/.

Coauthor-protocol defaults:
  --stride 1
  --sample-dt 0.01            (raw source cadence, before --stride)
  --segment-lengths 100:20:1200
  --n-runs 20
  --seed 20260820
  --tangent-normalization linf (l2 compatibility also available)
  --boxsize 5
  --sb-radius 1
  --metric-c 5
  --r-max 5
  --r-subdivisions 201
  --field-prime 43
  --require-sample-radius-below-r-max true
  --out-prefix shared_probability

Execution controls:
  --parallel-inner false
  --progress true
  --check-only                 Validate inputs and resolved settings, build
                               no comparison space, and write nothing.
  --help                       Show this message and exit.

The nominal segment-duration coordinate is
segment_length * sample_dt * stride, as in the documented Rössler experiment.
Starts are independently resampled for each length, with replacement, by
RandomSubsegmentExperiment.
""")
end


function parse_cli(args)
    opts = Dict{String,String}(
        "positions" => "",
        "tangents" => "",
        "stride" => "1",
        "sample-dt" => "0.01",
        "segment-lengths" => "100:20:1200",
        "n-runs" => "20",
        "seed" => "20260820",
        "tangent-normalization" => "linf",
        "boxsize" => "5",
        "sb-radius" => "1",
        "metric-c" => "5",
        "r-max" => "5",
        "r-subdivisions" => "201",
        "field-prime" => "43",
        "require-sample-radius-below-r-max" => "true",
        "out-dir" => "",
        "out-prefix" => "shared_probability",
        "parallel-inner" => "false",
        "progress" => "true",
    )
    check_only = false

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help"
            return opts, check_only, true
        elseif arg == "--check-only"
            check_only = true
            i += 1
        elseif startswith(arg, "--")
            key = arg[3:end]
            haskey(opts, key) || error("unknown argument: --$key")
            i == length(args) && error("--$key requires a value")
            startswith(args[i + 1], "--") &&
                error("--$key requires a value, got $(args[i + 1])")
            opts[key] = args[i + 1]
            i += 2
        else
            error("unknown positional argument: $arg")
        end
    end

    for key in ("positions", "tangents", "out-dir")
        isempty(opts[key]) && error("--$key is required")
    end
    return opts, check_only, false
end


function parse_bool(value::AbstractString)
    normalized = lowercase(strip(value))
    normalized in ("1", "true", "yes", "y") && return true
    normalized in ("0", "false", "no", "n") && return false
    error("cannot parse Bool from '$value'")
end


function parse_segment_lengths(spec::AbstractString)
    value = strip(spec)
    lengths = if occursin(":", value)
        parts = parse.(Int, split(value, ":"))
        if length(parts) == 2
            collect(parts[1]:parts[2])
        elseif length(parts) == 3
            parts[2] > 0 || error("segment-length step must be positive")
            collect(parts[1]:parts[2]:parts[3])
        else
            error("segment lengths must be start:stop or start:step:stop")
        end
    else
        parse.(Int, split(value, ","))
    end
    isempty(lengths) && error("segment-length grid must not be empty")
    all(>(1), lengths) || error("every segment length must exceed one")
    issorted(lengths) || error("segment lengths must be sorted")
    allunique(lengths) || error("segment lengths must be unique")
    return lengths
end


function resolved_path(path::AbstractString)
    probe = abspath(path)
    suffix = String[]
    while !ispath(probe)
        parent = dirname(probe)
        parent == probe && error("cannot resolve path: $path")
        pushfirst!(suffix, basename(probe))
        probe = parent
    end
    return normpath(joinpath(realpath(probe), suffix...))
end


function is_subpath(path::AbstractString, parent::AbstractString)
    path_parts = splitpath(resolved_path(path))
    parent_parts = splitpath(resolved_path(parent))
    return length(path_parts) >= length(parent_parts) &&
        path_parts[1:length(parent_parts)] == parent_parts
end


function reject_symlink_components(path::AbstractString)
    probe = abspath(path)
    stop = dirname(CODE_ROOT)
    while true
        islink(probe) && error("output path contains a symlink: $probe")
        probe == stop && break
        parent = dirname(probe)
        parent == probe && break
        probe = parent
    end
end


function file_sha256(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(sha256(io))
    end
end


function validate_real_input(path::AbstractString, label::AbstractString)
    requested = abspath(path)
    islink(requested) && error("$label must not be a symlink: $requested")
    isfile(requested) || error("missing $label: $requested")
    return realpath(requested)
end


function output_paths(out_dir::AbstractString, prefix::AbstractString)
    return (
        births=joinpath(out_dir, prefix * "_births.csv"),
        starts=joinpath(out_dir, prefix * "_segment_starts.csv"),
        rank0=joinpath(out_dir, prefix * "_rank0_heatmap.csv"),
        metadata=joinpath(out_dir, prefix * "_metadata.txt"),
    )
end


function validate_output(
    requested_dir::AbstractString,
    prefix::AbstractString,
)
    occursin(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", prefix) ||
        error("--out-prefix must be a filename-safe basename")
    basename(prefix) == prefix && prefix ∉ (".", "..") ||
        error("--out-prefix must be a filename-safe basename")

    reject_symlink_components(requested_dir)
    out_dir = resolved_path(requested_dir)
    safe_root = resolved_path(SAFE_OUTPUT_ROOT)
    is_subpath(out_dir, safe_root) ||
        error("--out-dir must stay below $safe_root")
    out_dir != safe_root ||
        error("--out-dir must be a named child below $safe_root")
    islink(abspath(requested_dir)) &&
        error("--out-dir must not be a symlink: $(abspath(requested_dir))")

    paths = output_paths(out_dir, prefix)
    for path in values(paths)
        (ispath(path) || islink(path)) &&
            error("refusing to overwrite existing output: $path")
    end
    return out_dir, paths
end


function normalize_tangents!(TX::AbstractMatrix, mode::AbstractString)
    mode in ("l2", "linf") ||
        error("--tangent-normalization must be l2 or linf")
    p = mode == "l2" ? 2 : Inf
    norms = [norm(tangent, p) for tangent in eachcol(TX)]
    all(value -> isfinite(value) && value > 0, norms) ||
        error("tangents must be finite and nonzero")
    for (index, value) in enumerate(norms)
        @views TX[:, index] ./= value
    end
    post_norms = [norm(tangent, p) for tangent in eachcol(TX)]
    all(value -> isapprox(value, 1.0; atol=1e-12, rtol=1e-10), post_norms) ||
        error("failed to normalize tangents in $mode")
    return extrema(norms), extrema(post_norms)
end


function load_lift(
    positions_path::AbstractString,
    tangents_path::AbstractString,
    stride::Int,
    tangent_normalization::AbstractString,
)
    positions = validate_real_input(positions_path, "positions CSV")
    tangents = validate_real_input(tangents_path, "tangents CSV")
    X_rows = readdlm(positions, ' ', Float64)
    TX_rows = readdlm(tangents, ' ', Float64)
    ndims(X_rows) == 2 || error("positions CSV must be a matrix")
    ndims(TX_rows) == 2 || error("tangents CSV must be a matrix")
    size(X_rows) == size(TX_rows) ||
        error("positions/tangents shape mismatch")
    size(X_rows, 1) >= 2 || error("lift must contain at least two samples")
    size(X_rows, 2) >= 1 || error("lift must contain at least one dimension")
    all(isfinite, X_rows) || error("positions contain nonfinite values")
    all(isfinite, TX_rows) || error("tangents contain nonfinite values")

    X_full = Matrix(transpose(X_rows))
    TX_full = Matrix(transpose(TX_rows))
    X = X_full[:, 1:stride:end]
    TX = TX_full[:, 1:stride:end]
    pre_norms, post_norms = normalize_tangents!(TX, tangent_normalization)
    return X, TX, (
        positions=positions,
        tangents=tangents,
        source_samples=size(X_rows, 1),
        dimension=size(X_rows, 2),
        pre_norms=pre_norms,
        post_norms=post_norms,
    )
end


function global_curve_bound(X, TX, metric_c::Float64)
    bound = 0.0
    for index in 1:size(X, 2)-1
        dx = norm(view(X, :, index + 1) - view(X, :, index))
        dv = norm(view(TX, :, index + 1) - view(TX, :, index))
        bound = max(bound, dx, metric_c * dv)
    end
    return bound
end


function build_trajectory_space(X, TX, boxsize, sb_radius, metric_c, r_max)
    lifted = [X; TX]
    trajectory = RefinedEquidistantTrajectory(lifted)
    comparison = sb_cubical_vr_comparison_space_via_cover(
        lifted,
        boxsize,
        sb_radius,
    )
    metric = DynamicDistance(size(X, 1), metric_c)
    return TrajectorySpace(trajectory, comparison, metric, r_max)
end


function write_births(path, result, effective_sample_dt)
    open(path, "w") do io
        println(
            io,
            "segment_length,segment_duration,run_index,start_index," *
            "end_index,rank,births",
        )
        for (length_index, segment_length) in enumerate(result.segment_lengths)
            duration = segment_length * effective_sample_dt
            for run_index in 1:result.n_runs
                start = result.segment_starts[length_index][run_index]
                stop = start + segment_length - 1
                signature = result.signatures[length_index][run_index]
                births = join(signature.birth_vector, ";")
                println(
                    io,
                    "$segment_length,$duration,$run_index,$start,$stop," *
                    "$(length(signature.birth_vector)),\"$births\"",
                )
            end
        end
    end
end


function write_starts(path, result, effective_sample_dt)
    open(path, "w") do io
        println(
            io,
            "segment_length,segment_duration,run_index,start_index,end_index",
        )
        for (length_index, segment_length) in enumerate(result.segment_lengths)
            duration = segment_length * effective_sample_dt
            for run_index in 1:result.n_runs
                start = result.segment_starts[length_index][run_index]
                stop = start + segment_length - 1
                println(
                    io,
                    "$segment_length,$duration,$run_index,$start,$stop",
                )
            end
        end
    end
end


function write_rank0_heatmap(path, result, radii)
    open(path, "w") do io
        print(io, "radius")
        for segment_length in result.segment_lengths
            print(io, ",", segment_length)
        end
        println(io)
        for radius in radii
            print(io, radius)
            for signatures in result.signatures
                rank_zero = count(
                    signature -> dimension(signature, radius) == 0,
                    signatures,
                )
                print(io, ",", rank_zero)
            end
            println(io)
        end
    end
end


function write_metadata(
    path,
    settings,
    files,
    n_analysis_samples,
    beta1,
    curve_bound,
    elapsed_space,
    elapsed_experiment,
    artifact_paths,
)
    open(path, "w") do io
        println(io, "Shared cycling-probability analysis")
        println(io, "protocol=coauthor_roessler_probability_v1")
        println(io, "positions=$(files.positions)")
        println(io, "positions_sha256=$(file_sha256(files.positions))")
        println(io, "tangents=$(files.tangents)")
        println(io, "tangents_sha256=$(file_sha256(files.tangents))")
        println(io, "driver=$(realpath(@__FILE__))")
        println(io, "driver_sha256=$(file_sha256(realpath(@__FILE__)))")
        println(io, "dimension=$(files.dimension)")
        println(io, "source_samples=$(files.source_samples)")
        println(io, "analysis_samples=$n_analysis_samples")
        println(io, "stride=$(settings.stride)")
        println(io, "start_index_space=post_stride_analysis_samples")
        println(io, "raw_sample_dt=$(settings.sample_dt)")
        println(io, "effective_sample_dt=$(settings.effective_sample_dt)")
        println(io, "sample_dt_cli_semantics=raw_source_cadence_before_stride")
        println(io, "duration_convention=segment_length_times_effective_sample_dt")
        println(io, "segment_lengths=$(join(settings.segment_lengths, ','))")
        println(io, "n_runs=$(settings.n_runs)")
        println(io, "seed=$(settings.seed)")
        println(io, "resample_segment_start=true")
        println(io, "sampling_with_replacement=true")
        println(io, "tangent_normalization=$(settings.tangent_normalization)")
        println(io, "normalization_applied_after_stride=true")
        println(io, "pre_normalization_min=$(files.pre_norms[1])")
        println(io, "pre_normalization_max=$(files.pre_norms[2])")
        println(io, "post_normalization_min=$(files.post_norms[1])")
        println(io, "post_normalization_max=$(files.post_norms[2])")
        println(io, "boxsize=$(settings.boxsize)")
        println(io, "sb_radius=$(settings.sb_radius)")
        println(io, "metric_C=$(settings.metric_c)")
        println(io, "r_max=$(settings.r_max)")
        println(io, "r_subdivisions=$(settings.r_subdivisions)")
        println(io, "field_prime=$(settings.field_prime)")
        println(io, "filtration_threshold=closed_leq")
        println(
            io,
            "require_sample_radius_below_r_max=" *
            "$(settings.require_sample_radius_below_r_max)",
        )
        println(io, "beta1_Y=$beta1")
        println(io, "sample_radius=$curve_bound")
        println(io, "global_curve_bound=$curve_bound")
        println(io, "parallel_inner=$(settings.parallel_inner)")
        println(io, "julia_threads=$(Threads.nthreads())")
        println(io, "comparison_space_seconds=$elapsed_space")
        println(io, "experiment_seconds=$elapsed_experiment")
        println(io, "births_sha256=$(file_sha256(artifact_paths.births))")
        println(io, "segment_starts_sha256=$(file_sha256(artifact_paths.starts))")
        println(io, "rank0_heatmap_sha256=$(file_sha256(artifact_paths.rank0))")
    end
end


function publish_outputs(writer, out_dir, paths)
    mkpath(out_dir)
    reject_symlink_components(out_dir)
    islink(out_dir) && error("output directory became a symlink: $out_dir")
    for path in values(paths)
        (ispath(path) || islink(path)) &&
            error("refusing to overwrite existing output: $path")
    end

    mktempdir(out_dir; prefix=".shared-probability-stage-") do stage
        # `writer` owns the staging filenames because the public prefix cannot
        # be reconstructed reliably from a suffixed target path.
        staged = writer(stage)
        for key in (:births, :starts, :rank0, :metadata)
            source = getproperty(staged, key)
            target = getproperty(paths, key)
            (ispath(target) || islink(target)) &&
                error("refusing to overwrite output created concurrently: $target")
            mv(source, target; force=false)
        end
    end
end


function main(args)
    opts, check_only, help_requested = parse_cli(args)
    if help_requested
        print_help()
        return
    end

    stride = parse(Int, opts["stride"])
    stride > 0 || error("--stride must be positive")
    sample_dt = parse(Float64, opts["sample-dt"])
    isfinite(sample_dt) && sample_dt > 0 ||
        error("--sample-dt must be positive and finite")
    effective_sample_dt = sample_dt * stride
    isfinite(effective_sample_dt) && effective_sample_dt > 0 ||
        error("--sample-dt times --stride must be positive and finite")
    segment_lengths = parse_segment_lengths(opts["segment-lengths"])
    n_runs = parse(Int, opts["n-runs"])
    n_runs > 0 || error("--n-runs must be positive")
    seed = parse(Int, opts["seed"])
    seed >= 0 || error("--seed must be nonnegative")
    tangent_normalization = lowercase(opts["tangent-normalization"])
    tangent_normalization in ("l2", "linf") ||
        error("--tangent-normalization must be l2 or linf")
    boxsize = parse(Float64, opts["boxsize"])
    isfinite(boxsize) && boxsize > 0 ||
        error("--boxsize must be positive and finite")
    sb_radius = parse(Int, opts["sb-radius"])
    sb_radius > 0 || error("--sb-radius must be positive")
    metric_c = parse(Float64, opts["metric-c"])
    isfinite(metric_c) && metric_c > 0 ||
        error("--metric-c must be positive and finite")
    r_max = parse(Float64, opts["r-max"])
    isfinite(r_max) && r_max > 0 ||
        error("--r-max must be positive and finite")
    r_subdivisions = parse(Int, opts["r-subdivisions"])
    r_subdivisions >= 2 || error("--r-subdivisions must be at least two")
    field_prime = parse(Int, opts["field-prime"])
    is_prime(field_prime) || error("--field-prime must be prime")
    require_sample_radius_below_r_max = parse_bool(
        opts["require-sample-radius-below-r-max"],
    )
    parallel_inner = parse_bool(opts["parallel-inner"])
    progress = parse_bool(opts["progress"])
    out_dir, paths = validate_output(opts["out-dir"], opts["out-prefix"])

    X, TX, files = load_lift(
        opts["positions"],
        opts["tangents"],
        stride,
        tangent_normalization,
    )
    maximum(segment_lengths) <= size(X, 2) ||
        error("largest segment length exceeds $(size(X, 2)) analysis samples")
    curve_bound = global_curve_bound(X, TX, metric_c)
    if require_sample_radius_below_r_max && !(curve_bound < r_max)
        error(
            "sample radius $curve_bound is not below r_max=$r_max; " *
            "use a wider versioned protocol before computing signatures",
        )
    end

    settings = (
        stride=stride,
        sample_dt=sample_dt,
        effective_sample_dt=effective_sample_dt,
        segment_lengths=segment_lengths,
        n_runs=n_runs,
        seed=seed,
        tangent_normalization=tangent_normalization,
        boxsize=boxsize,
        sb_radius=sb_radius,
        metric_c=metric_c,
        r_max=r_max,
        r_subdivisions=r_subdivisions,
        field_prime=field_prime,
        require_sample_radius_below_r_max=require_sample_radius_below_r_max,
        parallel_inner=parallel_inner,
        progress=progress,
    )

    println("=== Shared cycling-probability kernel ===")
    println("positions: $(files.positions)")
    println("tangents: $(files.tangents)")
    println(
        "source samples=$(files.source_samples), analysis samples=$(size(X, 2)), " *
        "dimension=$(files.dimension), stride=$stride",
    )
    println(
        "lengths=$(first(segment_lengths)):...:$(last(segment_lengths)), " *
        "n_runs=$n_runs, raw_sample_dt=$sample_dt, " *
        "effective_sample_dt=$(settings.effective_sample_dt)",
    )
    println(
        "normalization=$tangent_normalization, boxsize=$boxsize, " *
        "sb_radius=$sb_radius, C=$metric_c, r_max=$r_max, F_$field_prime",
    )
    println("global consecutive dynamic-distance bound: $curve_bound")
    println("planned output directory: $out_dir")

    if check_only
        println("CHECK ONLY: no comparison space built and no files written")
        return
    end

    elapsed_space = @elapsed trajectory_space = build_trajectory_space(
        X,
        TX,
        boxsize,
        sb_radius,
        metric_c,
        r_max,
    )
    beta1 = betti_1(trajectory_space)
    println("comparison-space beta_1=$beta1 ($(round(elapsed_space, digits=3)) s)")

    field = Core.apply_type(FF, field_prime)
    experiment = RandomSubsegmentExperiment(
        trajectory_space,
        segment_lengths,
        n_runs,
        seed,
    )
    elapsed_experiment = @elapsed result = run_experiment(
        experiment;
        field=field,
        threshold=r_max,
        resample_segment_start=true,
        progress=progress,
        parallel_inner=parallel_inner,
    )
    radii = collect(range(0.0, r_max; length=r_subdivisions))

    prefix = opts["out-prefix"]
    publish_outputs(out_dir, paths) do stage
        staged = output_paths(stage, prefix)
        write_births(staged.births, result, settings.effective_sample_dt)
        write_starts(staged.starts, result, settings.effective_sample_dt)
        write_rank0_heatmap(staged.rank0, result, radii)
        write_metadata(
            staged.metadata,
            settings,
            files,
            size(X, 2),
            beta1,
            curve_bound,
            elapsed_space,
            elapsed_experiment,
            staged,
        )
        return staged
    end

    println("experiment seconds: $(round(elapsed_experiment, digits=3))")
    for path in values(paths)
        println("wrote $path")
    end
end


main(ARGS)
