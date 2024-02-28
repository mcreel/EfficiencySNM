using Flux, BSON, Statistics, CUDA
include("Logit_DGP.jl")
include("../Net.jl")

# function to create the net
function make_net(dev)
    tcn = build_tcn(100, 4, 3,
        dilation_factor=2,
        kernel_size=32,
        channels=64,
        summary_size=4)
    tcn |> dev
end

# Create and train the net, and save final state
function train_net()
    net = make_net(cpu)
    train_network!(net;
        validation_size=8192,
        nsamples=1_000,
        batchsize=512,
        epochs=20,
        print_every=100)
    net_state = cpu(Flux.state(net))
    BSON.@save "net_state.bson"  net_state
end

# Recreate trained net on CPU to do Bayesian MSM
function load_trained()
    isfile("net_state.bson") ? @info("loading trained net") : @info("load_trained: can't create net on CPU: net_state has not been saved")
    net = make_net(cpu) # recreate the net
    BSON.@load "net_state.bson"  net_state # get the trained parameters
    Flux.loadmodel!(net, net_state)
    Flux.testmode!(net)
    return net
end
 

