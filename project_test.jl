# ========== NeXtCortex project ==========

using DrWatson
findproject(@__DIR__) |> quickactivate

include("src/network_utils.jl")
using .NetworkUtils

using SpikingNeuralNetworks
using UnPack
using Logging
using Plots
using Statistics
using Random
import SpikingNeuralNetworks: @update

global_logger(ConsoleLogger())
SNN.@load_units

import SpikingNeuralNetworks: IF, PoissonLayer, Stimulus, SpikingSynapse, compose, monitor!, sim!, SingleExpSynapse, IFParameter, Population, PostSpike, STTC

# --------------------------------------------------------------
# Network configuration (short names only)
# --------------------------------------------------------------

TC3inhib_network = (
    Npop = (TE=200, CE=4000, PV=800, SST=100, VIP=100),
    seed = 1234,

    exc = IFParameter(
        τm=200pF / 10nS,
        El=-70mV,
        Vt=-50mV,
        Vr=-70mV,
        R=1/10nS,
    ),

    inh_PV  = IFParameter(τm=100pF/10nS, El=-70mV, Vt=-53mV, Vr=-70mV, R=1/10nS),
    inh_SST = IFParameter(τm=200pF/10nS, El=-70mV, Vt=-53mV, Vr=-70mV, R=1/10nS),
    inh_VIP = IFParameter(τm=150pF/10nS, El=-70mV, Vt=-53mV, Vr=-70mV, R=1/10nS),

    spike     = PostSpike(τabs=5ms),
    spike_PV  = PostSpike(τabs=2ms),
    spike_SST = PostSpike(τabs=10ms),
    spike_VIP = PostSpike(τabs=5ms),

    synapse     = SingleExpSynapse(τi=5ms, τe=5ms, E_i=-80mV, E_e=0mV),
    synapse_PV  = SingleExpSynapse(τi=5ms, τe=5ms, E_i=-80mV, E_e=0mV),
    synapse_SST = SingleExpSynapse(τi=12ms, τe=5ms, E_i=-80mV, E_e=0mV),
    synapse_VIP = SingleExpSynapse(τi=7ms, τe=5ms, E_i=-80mV, E_e=0mV),

    connections = (
        TE_to_CE = (p=0.05, μ=4nS, rule=:Fixed),
        TE_to_PV = (p=0.05, μ=4nS, rule=:Fixed),

        CE_to_CE = (p=0.15, μ=2nS, rule=:Fixed),
        CE_to_PV = (p=0.05, μ=2nS, rule=:Fixed),
        CE_to_TE = (p=0.05, μ=2nS, rule=:Fixed),

        PV_to_CE  = (p=0.05, μ=10nS, rule=:Fixed),
        PV_to_PV  = (p=0.05, μ=10nS, rule=:Fixed),
        PV_to_SST = (p=0.05, μ=10nS, rule=:Fixed),

        SST_to_CE  = (p=0.025, μ=10nS, rule=:Fixed),
        SST_to_PV  = (p=0.025, μ=10nS, rule=:Fixed),
        SST_to_VIP = (p=0.025, μ=10nS, rule=:Fixed),

        VIP_to_SST = (p=0.3, μ=10nS, rule=:Fixed),
    ),

    afferents_to_TE  = (layer=PoissonLayer(rate=1.5Hz, N=1000), conn=(p=0.05, μ=4nS, rule=:Fixed)),
    afferents_to_CE  = (layer=PoissonLayer(rate=1.5Hz, N=1000), conn=(p=0.02, μ=4nS, rule=:Fixed)),
    afferents_to_PV  = (layer=PoissonLayer(rate=1.5Hz, N=1000), conn=(p=0.02, μ=4nS, rule=:Fixed)),
    afferents_to_SST = (layer=PoissonLayer(rate=1.5Hz, N=1000), conn=(p=0.15, μ=2nS, rule=:Fixed)),
    afferents_to_VIP = (layer=PoissonLayer(rate=1.5Hz, N=1000), conn=(p=0.10, μ=2nS, rule=:Fixed)),
)

# --------------------------------------------------------------
# Build and simulate
# --------------------------------------------------------------
model = NetworkUtils.build_network(TC3inhib_network)
monitor!(model.pop, [:v], sr=1kHz)

Random.seed!(TC3inhib_network.seed)
sim!(model, 3s)

# --------------------------------------------------------------
# Raster plot
# --------------------------------------------------------------
SNN.raster(model.pop, every=1,
           title="Raster plot (TE, CE, PV, SST, VIP)")
savefig("raster_full.png")

# --------------------------------------------------------------
# Firing rate dynamics
# --------------------------------------------------------------
t, rates, frplt = NetworkUtils.plot_firing_rates(model)
savefig(frplt, "firing_rates.png")


# --------------------------------------------------------------
# Membrane potential dynamics 
# --------------------------------------------------------------
plt_v = NetworkUtils.plot_membrane_potentials(model)
savefig(plt_v, "membrane_potentials_dynamic.png")

# --------------------------------------------------------------
# STTC
# --------------------------------------------------------------
myspikes = SNN.spiketimes(model.pop)    
sttc_value = mean(STTC(myspikes[1:5:end]  , 50ms)) # Using subsampled myspikes: only 1 every 5
println("STTC = ", sttc_value)


# %%

# Function to change the sample resolution of the plots
pop_spiketimes = SNN.spiketimes(model.pop.CE)
subsample = pop_spiketimes[1:10:end]
SNN.STTC(subsample, 10ms)

#%%

# %% [markdown]
# VIP→SST Modulation Experiment

base = TC3inhib_network.connections

config_test = @update TC3inhib_network begin
    connections = merge(base, (
        VIP_to_SST=NetworkUtils.scaled_connection(base.CortVip_to_CortSst; μ_scale=1.5, p_scale=1.0),
        PV_to_SST=NetworkUtils.scaled_connection(base.CortPv_to_CortSst; μ_scale=2.0, p_scale=1.0),
        PV_to_CE=NetworkUtils.scaled_connection(base.CortPv_to_CortExc; μ_scale=1.5, p_scale=1.0),
        SST_to_CE=NetworkUtils.scaled_connection(base.CortSst_to_CortExc; μ_scale=0.5, p_scale=1.0),
    ))
end


μ_scales = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10]
p_scales = [0.1, 0.2, 0.7, 1.0, 1.3]

results = Dict{NamedTuple,Float64}()

for μ in μ_scales, p in p_scales
    results[(μ, p)] = NetworkUtils.run_condition(
        TC3inhib_network;
        target=:CortVip_to_CortSst,
        μ_scale=μ,
        p_scale=p
    )
end


M = [results[(μ=μ, p=p)] for μ in μ_scales, p in p_scales]

heatmap(
    p_scales,
    μ_scales,
    M,
    xlabel="p scale",
    ylabel="μ scale",
    title="VIP→SST modulation effect on synchrony"
)

# %% [markdown]
# Cortical Feedback to Thalamus Sweep

μ_scales = [0.5, 1.0, 1.5, 2.0, 3.0]
p_scales = [0.05, 0.1, 0.2, 0.3]

# Network with feedback
#results_with = NetworkUtils.sweep_TC_feedback(TC3inhib_network; μ_scales=μ_scales, p_scales=p_scales)


# Measure the onset of epileptic activity (STTC)
myspikes = SNN.spiketimes(model.pop)[1:5:end]      # Subsample: only 1 every 5
sttc_value = mean(SNN.STTC(myspikes, 50ms))

#STTC over time

μ_scales = [0.5, 1.0, 1.5, 2.0]

tvals, result = NetworkUtils.sweep_sttc_time(TC3inhib_network; μ_scales=μ_scales)

# build matrix: rows=time, columns=scaling

H = hcat([result[(μ=μ, p=1.0)] for μ in μ_scales]...)

heatmap(
    tvals ./ 1s,    # convert to seconds
    μ_scales,
    H', #transpose H
    xlabel="Time (s)",
    ylabel="μ scaling",
    title="STTC Over Time for VIP→SST Scaling",
    colorbar_title="STTC"
)

# Visualize the spiking activity of the network with the new parameters.
#
# %%
# Plot raster plot of the new network activity
SNN.raster(model.pop, every=1,
    title="Raster plot of the network for p=0.025, μ=10nS")    # Change the parameters accordingly

# %%

# Plot vector field plot of the new network activity
SNN.vecplot(model.pop.CE, :v, neurons=13,
    title="Vecplot of the network for p=0.025, μ=10nS",        # Change the parameters accordingly
    xlabel="Time (s)",
    ylabel="Potential (mV)",
    lw=2,
    c=:darkcyan)

# %%
# test 
