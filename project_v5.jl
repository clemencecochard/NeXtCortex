# %%
# Import necessary packages and set up environment
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

# Set global logger to display messages in console
global_logger(ConsoleLogger())

# Load units for physical quantities
SNN.@load_units



# %% [markdown]
# ## Network Configuration
#
# Define the network parameters including neuron populations, synaptic properties,
# and connection probabilities.

# %%
# Define network configuration parameters
import SpikingNeuralNetworks: IF, PoissonLayer, Stimulus, SpikingSynapse, compose, monitor!, sim!, firing_rate, @update, SingleExpSynapse, IFParameter, Population, PostSpike, AdExParameter, STTC


TC3inhib_network = (
    # Number of neurons in each population
    Npop = (ThalExc=200, CortExc=4000,
        CortPvInh=800, CortSstInh=100, CortVipInh=100),

    seed = 1234,

    # Parameters for cortical excitatory neurons
    exc = IFParameter(
        τm = 200pF / 10nS,  # Membrane time constant
        El = -70mV,         # Leak reversal potential
        Vt = -50.0mV,       # Spike threshold
        Vr = -70.0f0mV,     # Reset potential
        R = 1 / 10nS,       # Membrane resistance
    ),

    # Parameters for the different populations of inhibitory neurons
    # PV population parameters (FASTER)
    inh_PV = IFParameter(       
        τm = 100pF / 10nS,      # Membrane time constant: PV faster
        El = -70mV,             # Leak reversal potential
        Vt = -53mV,             # Spike threshold
        Vr = -70mV,             # Reset potential
        R = 1 / 10nS,           # Membrane resistance
    ),

    # SST population parameters (SLOWER)
    inh_SST = IFParameter(      
        τm = 200pF / 10nS,      # Membrane time constant: SST slower
        El = -70mV,             # Leak reversal potential
        Vt = -53mV,             # Spike threshold
        Vr = -70mV,             # Reset potential
        R = 1 / 10nS,           # Membrane resistance
    ),

    # VIP population parameters (MEDIUM)
    inh_VIP = IFParameter(      
        τm = 150pF / 10nS,      # Membrane time constant: VIP medium
        El = -70mV,             # Leak reversal potential
        Vt = -53mV,             # Spike threshold
        Vr = -70mV,             # Reset potential
        R = 1 / 10nS,           # Membrane resistance
    ),

    # Spiking threshold properties: absolute refractory period
    spike = PostSpike(τabs=5ms),
    spike_PV  = PostSpike(τabs = 2ms),
    spike_SST = PostSpike(τabs = 10ms),
    spike_VIP = PostSpike(τabs = 5ms),

    # Synaptic properties
    synapse = SingleExpSynapse(
        τi = 5ms,             # Inhibitory synaptic time constant
        τe = 5ms,             # Excitatory synaptic time constant
        E_i = -80mV,          # Inhibitory reversal potential
        E_e = 0mV             # Excitatory reversal potential
    ),
    synapse_PV  = SingleExpSynapse(τi = 3ms,  τe = 5ms, E_i = -80mV, E_e = 0mV),
    synapse_SST = SingleExpSynapse(τi = 12ms, τe = 5ms, E_i = -80mV, E_e = 0mV),
    synapse_VIP = SingleExpSynapse(τi = 7ms,  τe = 5ms, E_i = -80mV, E_e = 0mV),


    # Connection probabilities and synaptic weights
    connections=(
        # from ThalExc
        ThalExc_to_CortExc = (p=0.05, μ=4nS, rule=:Fixed),
        ThalExc_to_CortPv = (p=0.05, μ=4nS, rule=:Fixed),
        # from CortExc
        CortExc_to_CortExc = (p=0.05, μ=2nS, rule=:Fixed),
        CortExc_to_CortPv = (p=0.05, μ=2nS, rule=:Fixed),
        CortExc_to_ThalExc = (p=0.05, μ=2nS, rule=:Fixed),        # CE_to_TE connection added
        # from CortPv
        CortPv_to_CortExc = (p=0.05, μ=10nS, rule=:Fixed),
        CortPv_to_CortPv = (p=0.05, μ=10nS, rule=:Fixed),
        CortPv_to_CortSst = (p=0.05, μ=10nS, rule=:Fixed),        # PV_to_SST connection added
        # from CortSst
        CortSst_to_CortExc = (p=0.025, μ=10nS, rule=:Fixed),
        CortSst_to_CortPv = (p=0.025, μ=10nS, rule=:Fixed),
        CortSst_to_CortVip = (p=0.025, μ=10nS, rule=:Fixed),
        # from CortVip
        CortVip_to_CortSst = (p=0.3, μ=10nS, rule=:Fixed),
    ),

    # Parameters for external Poisson input
    afferents_to_ThalExc=(
        layer = PoissonLayer(rate=1.5Hz, N=1000),           # Poisson input layer
        conn = (p=0.05, μ=4.0nS, rule=:Fixed),              # Connection probability and weight
    ),
    afferents_to_CortExc=(
        layer = PoissonLayer(rate=1.5Hz, N=1000),           # Poisson input layer
        conn = (p=0.02, μ=4.0nS, rule=:Fixed),              # Connection probability and weight
    ),
    afferents_to_CortPv=(
        layer = PoissonLayer(rate=1.5Hz, N=1000),           # Poisson input layer
        conn = (p=0.02, μ=4.0nS, rule=:Fixed),              # Connection probability and weight
    ),
    afferents_to_CortSst=(
        layer = PoissonLayer(rate=1.5Hz, N=1000),           # Poisson input layer
        conn = (p=0.15, μ=2.0nS, rule=:Fixed),              # Connection probability and weight
    ),
    afferents_to_CortVip=(
        layer = PoissonLayer(rate=1.5Hz, N=1000),           # Poisson input layer
        conn = (p=0.10, μ=2.0nS, rule=:Fixed),              # Connection probability and weight
    ),
)

# %% [markdown]
# ## Network Simulation
#
# Create the network and simulate it for a fixed duration.

# %%
# Create and simulate the network
model = NetworkUtils.build_network(TC3inhib_network)
SNN.print_model(model)                      # Print model summary
SNN.monitor!(model.pop, [:v], sr=1kHz)      # Monitor membrane potentials
SNN.sim!(model, duration=3s)                # Simulate for 3 seconds

# %%
# Measure the onset of epileptic activity (STTC)
myspikes = SNN.spiketimes(model.pop)[1:5:end]      # Subsample: only 1 every 5
sttc_value = mean(SNN.STTC(myspikes, 50ms))



# %% [markdown]
# ## Visualization
#
# Visualize the spiking activity of the network.

# %%
# Plot raster plot of network activity
SNN.raster(model.pop, every=1,
    title = "Raster plot of the balanced network")

# %%

# Plot vector field plot of network activity
SNN.vecplot(model.pop.CE, :v, neurons=13,
    title = "Vecplot of the balanced network",
    xlabel = "Time (s)",
    ylabel = "Potential (mV)",
    lw=2,
    c=:darkcyan)

# %%

# Function to change the sample resolution of the plots
pop_spiketimes = SNN.spiketimes(model.pop.CE)
subsample = pop_spiketimes[1:10:end]
SNN.STTC(subsample, 10ms)

#%%

# %% [markdown]
# VIP→SST modulation effect on synchrony

base = TC3inhib_network.connections

config_test = @update TC3inhib_network begin
    connections = merge(base, (
        VIP_to_SST = NetworkUtils.scaled_connection(base.CortVip_to_CortSst; μ_scale=1.5, p_scale=1.0),
        PV_to_SST  = NetworkUtils.scaled_connection(base.CortPv_to_CortSst;  μ_scale=2.0, p_scale=1.0),
        PV_to_CE   = NetworkUtils.scaled_connection(base.CortPv_to_CortExc;  μ_scale=1.5, p_scale=1.0),
        SST_to_CE  = NetworkUtils.scaled_connection(base.CortSst_to_CortExc; μ_scale=0.5, p_scale=1.0),
    ))
end


μ_scales = [0.5, 1.0, 1.5, 2.0]
p_scales = [0.7, 1.0, 1.3]

results = Dict{NamedTuple,Float64}()

for μ in μ_scales, p in p_scales
    key = (μ=μ, p=p)
    results[key] = NetworkUtils.run_condition(
        TC3inhib_network;
        target=:CortVip_to_CortSst,
        μ_scale=μ,
        p_scale=p
    )
end


M = [results[(μ=μ,p=p)] for μ in μ_scales, p in p_scales]

heatmap(
    p_scales, μ_scales,
    M,
    xlabel="p scale",
    ylabel="μ scale",
    title="VIP→SST modulation effect on synchrony"
)

# %% [markdown]
# CortExc_to_ThalExc modulation effect on synchrony

μ_scales = [0.5, 1.0, 1.5, 2.0, 3.0]  # factor of base synaptic weight
p_scales = [0.05, 0.1, 0.2, 0.3]      # connection probability


# Sweep for network WITH feedback
results_with = NetworkUtils.sweep_TC_feedback(TC3inhib_network;
    μ_scales=μ_scales, p_scales=p_scales)

# Sweep for network WITHOUT feedback
config_no_feedback = @update TC3inhib_network begin
    connections = merge(TC3inhib_network.connections,
                        (:CortExc_to_ThalExc => (p=0.0, μ=0nS, rule=:Fixed),))
end

results_without = NetworkUtils.sweep_TC_feedback(config_no_feedback;
    μ_scales=μ_scales, p_scales=p_scales)






# Measure the onset of epileptic activity (STTC)
myspikes = SNN.spiketimes(model.pop)[1:5:end]      # Subsample: only 1 every 5
sttc_value = mean(SNN.STTC(myspikes, 50ms))

# Visualize the spiking activity of the network with the new parameters.
#
# %%
# Plot raster plot of the new network activity
SNN.raster(model.pop, every=1,
    title = "Raster plot of the network for p=0.025, μ=10nS")    # Change the parameters accordingly

# %%

# Plot vector field plot of the new network activity
SNN.vecplot(model.pop.CE, :v, neurons=13,
    title = "Vecplot of the network for p=0.025, μ=10nS",        # Change the parameters accordingly
    xlabel = "Time (s)",
    ylabel = "Potential (mV)",
    lw=2,
    c=:darkcyan)

# %%