using QuadGK
using NPZ
using LogExpFunctions
using ProgressMeter
using ArgParse
using LinearAlgebra

println("number of Threads:")
println(Threads.nthreads())


args = ArgParseSettings()
@add_arg_table args begin
    "--dir"
        help = "wmi problem directory"
        arg_type = String
        required = true
    "--i"
        help = "index of collapsed samples"
        arg_type = Int
        required = true
    "--split"
        help = "ood split"
        arg_type = Int
        required = false
        default = -1
end

parsed_args = parse_args(ARGS, args)
dir = parsed_args["dir"]
collapsed_i = parsed_args["i"]
split = parsed_args["split"]

if split < 0
    output = npzread("$dir/logits_$collapsed_i.npy")
    wmi_stats = npzread("$dir/wmi_stats_$(collapsed_i).npz")
else
    output = npzread("$dir/logits_$(collapsed_i)_split_$split.npy")
    wmi_stats = npzread("$dir/wmi_stats_$(collapsed_i)_split_$split.npz")
end


function f(cons, lower, upper, coef=1)
    integral, _ = quadgk(x -> 1 / (1 + exp(cons-x * coef)), lower, upper, rtol=1e-8)
    integral /= (upper - lower)
end


function compute_prob(logits, zw, w_min, w_max, z)
    num_classes = size(logits)[1]
    a = transpose(reshape(repeat(logits, num_classes), num_classes, :))
    a[diagind(a)] .= -Inf
    c = logsumexp(a; dims=2) - logits + zw
    results = map(f, c, w_min, w_max, z)
    results
end

n_class = size(output[1, :])[1]
println("number of classes: $n_class")
all_prob = zeros(size(output))
p = Progress(size(output)[1])
@time Threads.@threads for i in range(1, size(output)[1])
    logits = output[i, :]
    zw = wmi_stats["zw"][i, :]
    w_min = wmi_stats["w_min"][i, :]
    w_max = wmi_stats["w_max"][i, :]
    z = wmi_stats["z"][i, :]
    all_prob[i, :] = compute_prob(logits, zw, w_min, w_max, z)
    next!(p)
end
finish!(p)

if split < 0
    npzwrite("$dir/all_prob_$collapsed_i.npz", all_prob)
else
    npzwrite("$dir/all_prob_$(collapsed_i)_split_$split.npz", all_prob)
end