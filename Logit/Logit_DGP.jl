
function TrueParameters()
 [1.0, -1.0, 0.5]
end    

function PriorDraw(S)
    randn(Float32, 3, S)
end


# S simulations, each at a different θ, for training the net
# the training uses a second order perturbation, for speed
# θs are returned standardized and normalized, for training
function MakeData(S)
    θs = PriorDraw(S)
    n = 100
    X = zeros(Float32, n, S, 4)
    # solve models and simulate data
    Threads.@threads for s = 1:S
        data = Float32[rand(n) randn(n,2) zeros(n)]
        data[:,4] = rand(n) .< 1.0 ./(1. .+ exp.(-data[:,1:3]*θs[:,s]))
        X[:, s, :] = Float32.(data)
    end    
    tabular2conv(permutedims(Float32.(X), (3, 2, 1))), θs
end


# Want S simulations, all at the same θ
# This does one long simulation, and separates samples using burnin draws
# between them
function MakeData(θ, S)
    n = 100
    X = zeros(Float32, n, S, 4)
    # solve models and simulate data
    Threads.@threads for s = 1:S
        data = Float32[rand(n) randn(n,2) zeros(n)]
        data[:,4] = rand(n) .< 1.0 ./(1. .+ exp.(-data[:,1:3]*θ))
        X[:, s, :] = Float32.(data)
    end    
    tabular2conv(permutedims(Float32.(X), (3, 2, 1)))
end



