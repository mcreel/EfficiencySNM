# defines the net and the DSGE model, and needed functions
include("Setup.jl")
 
function main()

net = load_trained()
Flux.testmode!(net)
#data, θtrue = MakeData(1)
θtrue = TrueParameters()
data = MakeData(θtrue, 100, CKmodel)
θnn = Float64.(UntransformParameters(net(data)))
@info "θtrue: "
display(θtrue)
@info "θnn:"
display(θnn)
e = θnn .- TrueParameters()
s = std(e, dims=2)
b = mean(e, dims=2)
r = sqrt.(mean(e.^2, dims=2))
println("bias: $(b)")
println("std.dev.: $(s)")
println("rmse: $(r)")
return θnn, b, r, s;

#=
# set up proposal
covreps = 1000
_,Σₚ = simmomentscov(net, dgp, covreps, θnn)
δ = 1.0 # tuning

# do MCMC
S = 40 # simulations to compute moments
# initial short chain
chain = mcmc(θnn, θnn, δ, Σₚ, S, net, dgp; burnin=0, chainlength=200)
accept = mean(chain[:,end-1])
# loop to get good tuning
while accept < 0.2 || accept > 0.3
    accept < 0.2 ? δ *= 0.75 : nothing
    accept > 0.3 ? δ *= 1.5 : nothing
    chain = mcmc(θnn, θnn, δ, Σₚ, S, net, dgp; burnin=0, chainlength=200)
    accept = mean(chain[:,end-1])
end
# final long chain
chain = mcmc(θnn, θnn, δ, Σₚ, S, net, dgp; burnin=0, chainlength=2000)
# report results
chn = Chains(chain[:,1:end-2], ["θ₁", "θ₂"])
plot(chn)
savefig("chain.png")
pretty_table([θtrue[:] Float64.(θnn[:]) mean(chain[:,1:end-2],dims=1)[:]], header = (["θtrue", "θnn", "θpos_mean"]))
display(chn)
=#
end
t, b, r, s = main();
