using CSVFiles, DataFrames
using DataStructures
using Statistics
using ArgParse
using HDF5, JLD
using ProgressBars

module HelperModule
using CSVFiles, DataFrames
"""
helper functions
"""
function Base.parse(::Type{Rational{T}}, x::AbstractString) where {T<:Integer}
    list = split(x, '/'; keepempty=false)
    if length(list) == 1
        return parse(T, list[1]) // 1
    else
        @assert length(list) == 2
        return parse(T, list[1]) // parse(T, list[2])
    end
end


function dir2idx(dir)
    return last(split(dir, "_"))
end


function bounds2uni(bounds)
     num_uni = 1.
    for k in keys(bounds)
        if k != "output"
            num_uni *= bounds[k]
        end
    end
    deno_uni = num_uni * bounds["output"]
    return num_uni, deno_uni
end

function get_test_indices(dir)
    test_indices = DataFrame(load(dir))
    test_indices_dict = Dict(Pair.(test_indices.idx, test_indices.cnt))
    test_indices_dict = Dict(item.first => parse.(Int, split(chop(item.second; head=1, tail=1), ',')) for item in test_indices_dict)
    return test_indices_dict
end

export parse, dir2idx, bounds2uni, get_test_indices
end

using .HelperModule

function get_summary(summary_path, dirs; prediction=false)
    if isfile(summary_path)
        println("reading cached results ...")
        return load(summary_path)
    end

    latte_cmd = `integrate --valuation=integrate --cone-decompose --monomials=polynomial.latte polytope.hrep.latte`
    @time Threads.@threads for dir in dirs
        cmd_dir = Cmd(latte_cmd; dir)
        out = read(ignorestatus(cmd_dir), String)
        open(joinpath(dir, "out.txt"),"w") do f
            println(f, out)
        end
        run(Cmd(`rm -f Check_emp.lp Check_emp.lps Check_emp.out polynomial.latte polytope.hrep.latte numOfLatticePoints`;dir))
    end
    println("done running latte")

    println("start reading results")
    res_summary = Dict()
    if !prediction
        numerator = 0
        denominator = 0
        # @time Threads.@threads for dir in dirs
        for dir in ProgressBar(dirs)
            problem_idx = dir2idx(dir)
            if !haskey(res_summary, problem_idx)
                res_summary[problem_idx] = [0., 0.]
            end
                
            f = open(joinpath(dir, "out.txt"), "r")
            s = read(f, String)
            res = match(r"Answer:.*", s)
            if res != nothing
                vol = parse(Rational{BigInt}, split(res.match, " ")[2])
                vol = float(vol)
                if occursin("numerator", dir)
                    res_summary[problem_idx][1] += vol
                else
                    @assert occursin("denominator", dir)
                    res_summary[problem_idx][2] += vol
                end
            end
            close(f)
        end
    else
        for dir in ProgressBar(dirs)
            problem_idx = dir2idx(dir)
                
            f = open(joinpath(dir, "out.txt"), "r")
            s = read(f, String)
            res = match(r"Answer:.*", s)
            if res != nothing
                vol = parse(Rational{BigInt}, split(res.match, " ")[2])
                vol = float(vol)
                res_summary[problem_idx] = vol
            else
                res_summary[problem_idx] = 0.
            end
            close(f)
        end
    end
    println("done reading results")
    save(summary_path, res_summary)
    return res_summary
end


function evaluate_likelihood(test_indices_dict, tmp_dir, res_summary)
    all_log_lik = Dict()
    for test_index in keys(test_indices_dict)
        p_sum = DefaultDict(Vector{Any})
        for problem_index in test_indices_dict[test_index]
            df = DataFrame(load("$(pwd())/$(tmp_dir)/bounds_$(problem_index).csv"))
            bounds = Dict(Pair.(df.variable, df.diff))
            num_uni, deno_uni = bounds2uni(bounds)
            res_numerator, res_denominator = res_summary[string(problem_index)]
            for epsilon in float_e
                push!(p_sum[epsilon], (epsilon*num_uni + res_numerator) / (epsilon*deno_uni + res_denominator))
            end
        all_log_lik[test_index] = Dict(item.first => mean(item.second) for item in p_sum)
        end
    end

    io = open("$(likelihood_path)/epsilon.txt", "w")
    for epsilon in float_e
        write(io, string(epsilon), "\n")
        write(io, string(mean([log(item.second[epsilon]) for item in all_log_lik])), "\n")
    end
    close(io)
end

function evaluate_prediction(test_indices_dict, tmp_dir, res_summary)
    all_prediction = []
    all_prediction = Dict()
    ground_truth = Dict()
    for test_index in keys(test_indices_dict)
        y_sum = []
        for problem_index in test_indices_dict[test_index]
            df = DataFrame(load("$(pwd())/$(tmp_dir)/bounds_$(problem_index).csv"))
            bounds = Dict(Pair.(df.variable, df.diff))
            num_uni, _ = bounds2uni(bounds)
            pred = res_summary[string(problem_index)] / num_uni
    
            gt = DataFrame(load("$(prediction_path)/gt_$(problem_index).csv"))
            gt = Dict(Pair.(gt.k, gt.v))
            
            push!(y_sum, pred + gt["constant"])
            ground_truth[test_index] = gt["y"]
        end
        all_prediction[test_index] = mean(y_sum)
    end
    
    # unnormalized rmse
    error = []
    for test_index in keys(test_indices_dict)
        push!(error, (ground_truth[test_index] - all_prediction[test_index])^2)
    end
    rmse = sqrt(mean(error))
    
    io = open("$(prediction_path)/unnormalized_rmse.txt", "w")
    write(io, string(rmse), "\n")
    close(io)
end


args = ArgParseSettings()
@add_arg_table args begin
    "--dir"
        help = "latte problem directory"
        arg_type = String
        required = true
    "--e"
        help = "epsilon parameter"
        arg_type = String
        required = true
    "--relu"
        help = "whether last second layer stochastic or not"
        action = :store_true
end

parsed_args = parse_args(ARGS, args)
tmp_dir = parsed_args["dir"]
likelihood_path = "$(pwd())/$(tmp_dir)/likelihood"
prediction_path = "$(pwd())/$(tmp_dir)/prediction"

epsilons = parsed_args["e"]
relu = parsed_args["relu"]
float_e = [parse(Float32, e) for e in split(epsilons)]

likelihood_dirs = filter(isdir, readdir(likelihood_path; join=true))
println("Found $(length(likelihood_dirs)) likelihood latte problems")
if relu
    prediction_dirs = filter(isdir, readdir(prediction_path; join=true))
    println("Found $(length(prediction_dirs)) prediction latte problems")
end

"""
likelihood
"""
summary_path = "$(likelihood_path)/summary.jld"
res_summary = get_summary(summary_path, likelihood_dirs)
test_indices_dict = get_test_indices("$(pwd())/$(tmp_dir)/test_indices.csv")
evaluate_likelihood(test_indices_dict, tmp_dir, res_summary)
run(Cmd(`cp $(likelihood_path)/epsilon.txt $(likelihood_path)/../..`))

if relu
    """
    prediction
    """
    summary_path = "$(prediction_path)/summary.jld"
    res_summary = get_summary(summary_path, prediction_dirs, prediction=true)
    evaluate_prediction(test_indices_dict, tmp_dir, res_summary)
    run(Cmd(`cp $(prediction_path)/unnormalized_rmse.txt $(prediction_path)/../..`))
end

run(Cmd(`rm -rf $(pwd())/$(tmp_dir)`))