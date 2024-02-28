# defines the net and the DSGE model,  and needed functions
include("Setup.jl")

function main()
    train_net() # comment out if already trained
    net = load_trained()
#=
    Flux.testmode!(net)
    θtrue = TrueParameters()
    data = MakeData(θtrue, 100, CKmodel)
    Float64.(UntransformParameters(θtrue[:]))
    θnn = Float64.(UntransformParameters(net(data)))
    @info "θtrue: "
    display(θtrue)
    @info "θnn:"
    display(θnn)
return θnn

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
main()
