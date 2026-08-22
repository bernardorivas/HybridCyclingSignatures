#!/usr/bin/env julia
#
# Fine-compass C/radius experiment runner.  It consumes a fixed manifest of
# physical-duration windows so every construction is evaluated on exactly the
# same segments.  It has been used for the completed tied metric/tangent-cover
# diagnostic documented in this directory.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "period_doubling", "julia"))

using CyclingSignatures
using DelimitedFiles
using LinearAlgebra
using Printf
using SHA


function print_help()
    println("""
Usage: julia --project=period_doubling/julia \\
  experiments_planned/run_duration_c_radius.jl \\
  --data-dir PATH --base NAME --manifest CSV --boxsize C \\
  --sb-radius 1 --rho-max 1.75 --split tune \
  [--metric-c C] [--tangents PATH] [--tangent-provenance JSON] \
  --out-dir PATH [--out-prefix NAME]

Without --metric-c, DynamicDistance uses the historical default
C=boxsize*sb_radius.  --metric-c changes only that metric coefficient; the
comparison cover remains fixed by --boxsize and --sb-radius.  --tangents
overrides only BASE_tangents.csv; positions still come from --data-dir.
--tangent-provenance records the immutable exporter document for an override.
The output directory must be below experiments_planned/outputs/.
""")
end


function parse_cli(args)
    opts = Dict{String,String}(
        "data-dir" => "",
        "base" => "",
        "manifest" => "",
        "boxsize" => "",
        "sb-radius" => "1",
        "metric-c" => "",
        "tangents" => "",
        "tangent-provenance" => "",
        "rho-max" => "1.75",
        "split" => "all",
        "out-dir" => "",
        "out-prefix" => "",
    )
    i = 1
    while i <= length(args)
        startswith(args[i], "--") || error("unknown positional argument: $(args[i])")
        key = args[i][3:end]
        haskey(opts, key) || error("unknown argument: --$key")
        i == length(args) && error("--$key requires a value")
        opts[key] = args[i + 1]
        i += 2
    end
    for key in ("data-dir", "base", "manifest", "boxsize", "out-dir")
        isempty(opts[key]) && error("--$key is required")
    end
    return opts
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


function file_sha256(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(sha256(io))
    end
end


function load_lift(
    data_dir::AbstractString,
    base::AbstractString,
    tangents_override::AbstractString,
)
    positions_path = joinpath(data_dir, base * "_positions.csv")
    tangents_path = isempty(tangents_override) ?
        joinpath(data_dir, base * "_tangents.csv") : tangents_override
    isfile(positions_path) || error("missing positions: $positions_path")
    isfile(tangents_path) || error("missing tangents: $tangents_path")
    positions_path = realpath(positions_path)
    tangents_path = realpath(tangents_path)
    X = Matrix(transpose(readdlm(positions_path, ' ', Float64)))
    TX = Matrix(transpose(readdlm(tangents_path, ' ', Float64)))
    size(X) == size(TX) || error("positions/tangents shape mismatch")
    for tangent in eachcol(TX)
        tangent_norm = norm(tangent)
        isfinite(tangent_norm) && tangent_norm > 0 ||
            error("tangents must be finite and nonzero")
        tangent ./= tangent_norm
    end
    return X, TX, positions_path, tangents_path
end


function load_manifest(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("empty manifest: $path")
    expected = [
        "target_duration", "split", "run_index", "start_index", "end_index",
        "realized_duration", "duration_error",
    ]
    split(lines[1], ',') == expected || error("unexpected manifest header: $path")
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = split(line, ',')
        length(values) == length(expected) || error("malformed manifest row: $line")
        push!(rows, (
            target_duration=parse(Float64, values[1]),
            split=values[2],
            run_index=parse(Int, values[3]),
            start_index=parse(Int, values[4]),
            end_index=parse(Int, values[5]),
            realized_duration=parse(Float64, values[6]),
            duration_error=parse(Float64, values[7]),
        ))
    end
    isempty(rows) && error("manifest has no windows: $path")
    return rows
end


function curve_bound(X, TX, first_index::Int, last_index::Int, C::Float64)
    last_index > first_index || error("window needs at least two samples")
    h = 0.0
    for index in first_index:last_index-1
        dx = norm(view(X, :, index + 1) - view(X, :, index))
        dv = norm(view(TX, :, index + 1) - view(TX, :, index))
        h = max(h, dx, C * dv)
    end
    return h
end


function main(args)
    if args == ["--help"]
        print_help()
        return
    end
    opts = parse_cli(args)
    code_root = realpath(joinpath(@__DIR__, ".."))
    safe_output_root = resolved_path(joinpath(@__DIR__, "outputs"))
    out_dir = resolved_path(opts["out-dir"])
    is_subpath(out_dir, safe_output_root) ||
        error("out-dir must stay below $safe_output_root")

    data_dir = realpath(opts["data-dir"])
    base = opts["base"]
    manifest_path = realpath(opts["manifest"])
    boxsize = parse(Float64, opts["boxsize"])
    sb_radius = parse(Int, opts["sb-radius"])
    isfinite(boxsize) && boxsize > 0 || error("--boxsize must be positive")
    sb_radius > 0 || error("--sb-radius must be positive")
    cover_default_C = boxsize * sb_radius
    metric_mode = isempty(opts["metric-c"]) ? "cover_default" : "explicit"
    C = metric_mode == "cover_default" ?
        cover_default_C : parse(Float64, opts["metric-c"])
    isfinite(C) && C > 0 || error("--metric-c must be positive and finite")
    rho_max = parse(Float64, opts["rho-max"])
    isfinite(rho_max) && rho_max > 0 || error("--rho-max must be positive")
    r_max = C * rho_max
    out_prefix = isempty(opts["out-prefix"]) ? base : opts["out-prefix"]
    basename(out_prefix) == out_prefix && out_prefix ∉ (".", "..") ||
        error("out-prefix must be a filename-safe basename")
    occursin(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", out_prefix) ||
        error("out-prefix must contain only letters, digits, '.', '_', or '-'")

    output_path = joinpath(out_dir, out_prefix * "_births.csv")
    metadata_path = joinpath(out_dir, out_prefix * "_metadata.txt")
    for path in (output_path, metadata_path)
        (ispath(path) || islink(path)) &&
            error("refusing to overwrite existing output: $path")
    end

    X, TX, positions_path, tangents_path = load_lift(
        data_dir,
        base,
        opts["tangents"],
    )
    tangent_provenance_path = isempty(opts["tangent-provenance"]) ? "" :
        realpath(opts["tangent-provenance"])
    rows = load_manifest(manifest_path)
    requested_split = opts["split"]
    requested_split in ("all", "tune", "validate") ||
        error("--split must be all, tune, or validate")
    if requested_split != "all"
        rows = [row for row in rows if row.split == requested_split]
        isempty(rows) && error("manifest has no $requested_split windows")
    end
    maximum(row.end_index for row in rows) <= size(X, 2) ||
        error("manifest exceeds lift length")

    trajectory_space = if metric_mode == "cover_default"
        utb_trajectory_space_from_trajectory(
            X,
            TX,
            boxsize,
            sb_radius;
            flt_max_heuristic=r_max,
        )
    else
        utb_trajectory_space_from_trajectory(
            X,
            TX,
            boxsize,
            sb_radius;
            metric=DynamicDistance(size(X, 1), C),
            flt_max_heuristic=r_max,
        )
    end
    beta1 = betti_1(trajectory_space)
    mkpath(out_dir)
    open(output_path, "w") do io
        println(io,
            "target_duration,split,run_index,start_index,end_index," *
            "realized_duration,duration_error,curve_bound,rank,births")
        for row in rows
            index_range = row.start_index:row.end_index
            h = curve_bound(X, TX, row.start_index, row.end_index, C)
            signature = cycling_signature(
                Val(:DistanceMatrix), trajectory_space, index_range, r_max
            )
            births = join(signature.birth_vector, ";")
            println(io,
                "$(row.target_duration),$(row.split),$(row.run_index)," *
                "$(row.start_index),$(row.end_index),$(row.realized_duration)," *
                "$(row.duration_error),$h,$(length(signature.birth_vector))," *
                "\"$births\"")
        end
    end

    open(metadata_path, "w") do io
        println(io, "Prepared fine-compass duration/C/radius sweep")
        println(io, "base=$base")
        println(io, "positions=$positions_path")
        println(io, "tangents=$tangents_path")
        println(io, "manifest=$manifest_path")
        println(io, "boxsize=$boxsize")
        println(io, "sb_radius=$sb_radius")
        println(io, "metric_mode=$metric_mode")
        println(io, "cover_default_C=$cover_default_C")
        println(io, "metric_C=$C")
        println(io, "C=$C")
        println(io, "rho_max=$rho_max")
        println(io, "r_max=$r_max")
        println(io, "beta1_Y=$beta1")
        println(io, "n_windows=$(length(rows))")
        println(io, "split=$requested_split")
        println(io, "code_root=$code_root")
        println(io, "positions_sha256=$(file_sha256(positions_path))")
        println(io, "tangents_sha256=$(file_sha256(tangents_path))")
        println(io, "manifest_sha256=$(file_sha256(manifest_path))")
        if !isempty(tangent_provenance_path)
            println(io, "tangent_provenance=$tangent_provenance_path")
            println(io,
                "tangent_provenance_sha256=$(file_sha256(tangent_provenance_path))")
        end
    end
    println("wrote $output_path")
    println("wrote $metadata_path")
end


main(ARGS)
