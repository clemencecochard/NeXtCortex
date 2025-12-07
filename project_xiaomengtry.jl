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
    Npop=(ThalExc=200, CortExc=4000,
        CortPvInh=800, CortSstInh=100,), seed=1234,

    # Parameters for cortical excitatory neurons
    exc=SNN.AdExParameter(
        C=200pF,
        gl=10nS,
        El=-60mV,
        Vt=-50mV,
        Vr=-65mV,
        ΔT=2mV,
        τw=500ms,
        a=4nS,
        b=130pA
    ),
    # Parameters for thalamic excitatory neurons
    thal=SNN.AdExParameter(
        C=200pF,
        gl=10nS,
        El=-65mV,
        Vt=-50mV,
        Vr=-65mV,
        ΔT=1.0mV,
        τw=50ms,
        a=0nS,
        b=0pA
    ),

    # Parameters for the different populations of inhibitory neurons
    # PV population parameters 
    # FS firing pattern: low input resistance, Brief action potential , Can sustain high firing frequency,Little or no spike frequency adaption
    inh_PV=SNN.AdExParameter(
        C=200pF, # membrane capacitance
        gl=10nS, # leak conductance
        El=-60mV, # leak reversal potential
        Vt=-50mV,# spike threshold
        Vr=-65mV, # reset potential
        ΔT=0.5mV, # slope factor
        τw=500ms, # adaptation time constant
        a=0nS, # subthreshold adaptation
        b=0pA # spike-triggered adaptation
    ),

    # SST population parameters 
    #adapting or bursting: high input resistance, low max frequency firing, may show rebond spike, spike frenquency adaptation.
    inh_SST=SNN.AdExParameter(
        C=200pF,
        gl=10nS,
        El=-55mV,
        Vt=-50mV,
        Vr=-65mV,
        ΔT=1.5mV,
        τw=500ms,
        a=4nS,
        b=25pA
    ),

    # VIP population parameters (MEDIUM)
    # high input resistance, many firing patterns,irregular spiking , bursting, mostly adapting.
    # inh_VIP=SNN.AdExParameter( C=200pF,gl=10nS,El=-60mV,Vt=-50mV, Vr=-65mV,ΔT=2.0mV,τw=500ms,a=3.0nS,b=40.0pA),

    # Spiking threshold properties: absolute refractory period
    spike=PostSpike(τabs=5ms),
    spike_PV=PostSpike(τabs=2ms),
    spike_SST=PostSpike(τabs=10ms),

    # Synaptic properties
    synapse_ce=SingleExpSynapse(
        τi=5ms,             # Inhibitory synaptic time constant
        τe=5ms,             # Excitatory synaptic time constant
        E_i=-80mV,          # Inhibitory reversal potential
        E_e=0mV             # Excitatory reversal potential
    ),
    synapse_thal=SingleExpSynapse(τi=5ms, τe=5ms, E_i=-80mV, E_e=0mV),
    synapse_PV=SingleExpSynapse(τi=5ms, τe=1ms, E_i=-80mV, E_e=0mV),
    synapse_SST=SingleExpSynapse(τi=5ms, τe=2ms, E_i=-80mV, E_e=0mV),
    #synapse_VIP=SingleExpSynapse(τi=5ms, τe=5ms, E_i=-80mV, E_e=0mV),




    # Connection probabilities and synaptic weights
    connections=(
        # ThalExc-> Cortex
        ThalExc_to_CortExc=(p=0.05, μ=4nS, rule=:Fixed),
        ThalExc_to_CortPv=(p=0.05, μ=4nS, rule=:Fixed),
        # from Cortex recurrent
        CortExc_to_CortExc=(p=0.05, μ=2nS, rule=:Fixed),
        CortExc_to_CortPv=(p=0.05, μ=2nS, rule=:Fixed),
        CortExc_to_CortSst=(p=0.05, μ=2nS, rule=:Fixed),
        CortExc_to_ThalExc=(p=0.05, μ=2nS, rule=:Fixed),        # CE_to_TE connection added
        # from CortPv
        CortPv_to_CortExc=(p=0.05, μ=10nS, rule=:Fixed),
        CortPv_to_CortPv=(p=0.05, μ=10nS, rule=:Fixed),
        #CortPv_to_CortSst=(p=0.05, μ=10nS, rule=:Fixed),        # PV_to_SST connection added
        # from CortSst
        CortSst_to_CortExc=(p=0.025, μ=5nS, rule=:Fixed),
        CortSst_to_CortPv=(p=0.025, μ=5nS, rule=:Fixed),
        # CortSst_to_CortVip=(p=0.025, μ=10nS, rule=:Fixed),
        # from CortVip
        # CortVip_to_CortSst=(p=0.3, μ=10nS, rule=:Fixed),
    ),

    # Parameters for external Poisson input
    afferents_to_ThalExc=(
        layer=PoissonLayer(rate=1.5Hz, N=1000), # Poisson input layer
        conn=(p=0.05, μ=4.0nS, rule=:Fixed), # Connection probability and weight
    ),
    afferents_to_CortExc=(
        layer=PoissonLayer(rate=1.5Hz, N=1000), # Poisson input layer
        conn=(p=0.02, μ=4.0nS, rule=:Fixed), # Connection probability and weight
    ),
    afferents_to_CortPv=(
        layer=PoissonLayer(rate=1.5Hz, N=1000), # Poisson input layer
        conn=(p=0.02, μ=4.0nS, rule=:Fixed), # Connection probability and weight
    ),
    afferents_to_CortSst=(
        layer=PoissonLayer(rate=1.5Hz, N=1000), # Poisson input layer
        conn=(p=0.15, μ=2.0nS, rule=:Fixed), # Connection probability and weight
    ),
    #afferents_to_CortVip=(
    #layer=PoissonLayer(rate=1.5Hz, N=1000), # Poisson input layer
    #conn=(p=0.10, μ=2.0nS, rule=:Fixed), # Connection probability and weight
    #),
)

# %% [markdown]
# ## Network Simulation
# %%
# Create and simulate the network
# Define a function to create the network based on the configuration parameters.

# %%
# Function to create the network
function network(config)
    @unpack afferents_to_ThalExc, afferents_to_CortExc, afferents_to_CortPv, afferents_to_CortSst, connections, Npop, spike, spike_PV, exc, thal, inh_PV, inh_SST = config
    @unpack synapse_ce, synapse_thal, synapse_PV, synapse_SST = config

    # Create neuron populations
    TE = Population(thal; synapse=synapse_thal, spike, N=Npop.ThalExc, name="ThalExc")
    CE = Population(exc; synapse=synapse_ce, spike, N=Npop.CortExc, name="CortExc")
    PV = Population(inh_PV; synapse=synapse_PV, spike, N=Npop.CortPvInh, name="CortPvInh")
    SST = Population(inh_SST; synapse=synapse_SST, spike, N=Npop.CortSstInh, name="CortSstInh")
    # VIP = Population(inh; synapse, spike, N=Npop.CortVipInh, name="CortVipInh")

    # Create external Poisson input
    @unpack layer = afferents_to_ThalExc
    afferentTE = Stimulus(layer, TE, :glu, conn=afferents_to_ThalExc.conn, name="bgTE")  # Excitatory input
    @unpack layer = afferents_to_CortExc
    afferentRE = Stimulus(layer, CE, :glu, conn=afferents_to_CortExc.conn, name="bgRE")  # Excitatory input
    @unpack layer = afferents_to_CortPv
    afferentPV = Stimulus(layer, PV, :glu, conn=afferents_to_CortPv.conn, name="bgPV")  # Excitatory input
    @unpack layer = afferents_to_CortSst
    afferentSST = Stimulus(layer, SST, :glu, conn=afferents_to_CortSst.conn, name="bgSST")  # Excitatory input
    #@unpack layer = afferents_to_CortVip
    #afferentVIP = Stimulus(layer, VIP, :glu, conn=afferents_to_CortSst.conn, name="bgSST")  # Excitatory input

    # Create recurrent connections
    synapses = (
        TE_to_CE=SpikingSynapse(TE, CE, :glu, conn=connections.ThalExc_to_CortExc),
        TE_to_PV=SpikingSynapse(TE, PV, :glu, conn=connections.ThalExc_to_CortPv), CE_to_CE=SpikingSynapse(CE, CE, :glu, conn=connections.CortExc_to_CortExc),
        CE_to_PV=SpikingSynapse(CE, PV, :glu, conn=connections.CortExc_to_CortPv),
        CE_to_SST=SpikingSynapse(CE, SST, :glu, conn=connections.CortExc_to_CortSst),
        CE_to_TE=SpikingSynapse(CE, TE, :glu, conn=connections.CortExc_to_ThalExc), PV_to_CE=SpikingSynapse(PV, CE, :gaba, conn=connections.CortPv_to_CortExc),
        PV_to_PV=SpikingSynapse(PV, PV, :gaba, conn=connections.CortPv_to_CortPv), SST_to_CE=SpikingSynapse(SST, CE, :gaba, conn=connections.CortSst_to_CortExc),
        SST_to_PV=SpikingSynapse(SST, PV, :gaba, conn=connections.CortSst_to_CortPv),
        #SST_to_VIP=SpikingSynapse(SST, VIP, :gaba, conn=connections.CortSst_to_CortVip), VIP_to_SST=SpikingSynapse(VIP, SST, :gaba, conn=connections.CortVip_to_CortSst),
    )

    # Compose the model
    model = compose(; TE, CE, PV, SST, #VIP,
        afferentTE, afferentRE,
        afferentPV, afferentSST, synapses...,
        name="thalamo-cortical network")

    # Set up monitoring
    monitor!(model.pop, [:fire])  # Monitor spikes
    monitor!(model.stim, [:fire])  # Monitor input spikes

    return model
end

# %% [markdown]
# ## Network Simulation
model = network(TC3inhib_network)
SNN.print_model(model)  # Print model summary
SNN.monitor!(model.pop, [:v])
SNN.sim!(model, duration=3s)               # Simulate for 3 seconds

# %%
# Measure the onset of epileptic activity (STTC)
myspikes = SNN.spiketimes(model.pop)[1:5:end]      # Subsample: only 1 every 5
sttc_value = mean(SNN.STTC(myspikes, 50ms))

# Simulate for 5 seconds

# %% [markdown]
# ## Visualization
#
# Visualize the spiking activity of the network.

# %%
# Plot raster plot of network activity
SNN.raster(model.pop, every=1,
    title="Raster plot of the balanced network",
    yrotation=0,
)

# %%
SNN.vecplot(model.pop.CE, :v, neurons=13,
    xlabel="Time (s)",
    ylabel="Potential (mV)",
    lw=2,
    c=:darkblue)
# %% [markdown]
# ## Visualization
#
# Visualize the spiking activity of the network.

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
# Plot Membrane potential dynamics of  neuron subtypes in one figure
# %%
p1 = SNN.vecplot(model.pop.PV, :v, neurons=1, title="PV", c=:darkorange)
p2 = SNN.vecplot(model.pop.SST, :v, neurons=1, title="SST", c=:darkgreen)
p3 = SNN.vecplot(model.pop.CE, :v, neurons=1, title="Exc", c=:darkcyan)
p4 = SNN.vecplot(model.pop.TE, :v, neurons=1, title="ThalExc", c=:blue)

plot(p1, p2, p3, p4, layout=(4, 1), link=:x, size=(2000, 1800),
    xlabel="Time (s)", ylabel="Membrane potential (mV)")
# %%
# Plot Firing rates of neuron subtypes in one figure
time_axis = 0:20:3000
# === CE ===
rates_CE = SNN.firing_rate(model.pop.CE, time_axis, sampling=20ms, τ=25ms)
rates_CE_mat = rates_CE[1]
t = collect(rates_CE[2]) ./ 1000   # 秒
pop_CE = vec(mean(rates_CE_mat, dims=1))
# === PV ===
rates_PV = SNN.firing_rate(model.pop.PV, time_axis, sampling=20ms, τ=25ms)
rates_PV_mat = rates_PV[1]
pop_PV = vec(mean(rates_PV_mat, dims=1))

# === SST ===
rates_SST = SNN.firing_rate(model.pop.SST, time_axis, sampling=20ms, τ=25ms)
rates_SST_mat = rates_SST[1]
pop_SST = vec(mean(rates_SST_mat, dims=1))
# === TE (Thalamic Exc) ===
rates_TE = SNN.firing_rate(model.pop.TE, time_axis, sampling=20ms, τ=25ms)
rates_TE_mat = rates_TE[1]
pop_TE = vec(mean(rates_TE_mat, dims=1))
# Plot all populations together

plot(t, pop_CE, lw=2, label="CortExc (CE)")
plot!(t, pop_PV, lw=2, label="CortPV (PV)")
plot!(t, pop_SST, lw=2, label="CortSST (SST)")
plot!(t, pop_TE, lw=2, label="ThalExc (TE)")

xlabel!("Time (s)")
ylabel!("Population firing rate (Hz)")
title!("Population firing rates of all subtypes")


# %%

# Function to change the sample resolution of the plots
pop_spiketimes = SNN.spiketimes(model.pop.CE)
subsample = pop_spiketimes[1:10:end]
SNN.STTC(subsample, 10ms)

#%%


# %%
