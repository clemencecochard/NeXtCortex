# %%
# Import necessary packages and set up environment
using DrWatson
findproject(@__DIR__) |> quickactivate

using SpikingNeuralNetworks
using UnPack
using Logging
using Plots
using Statistics

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
        CortPvInh=800, CortSstInh=100, CortVipInh=100), seed=1234,

    # Parameters for cortical excitatory neurons
    exc=IFParameter(
        τm=200pF / 10nS,  # Membrane time constant
        El=-70mV,         # Leak reversal potential
        Vt=-50.0mV,       # Spike threshold
        Vr=-70.0f0mV,     # Reset potential
        R=1 / 10nS,       # Membrane resistance
    ),

    # Parameters for the different populations of inhibitory neurons
    # PV population parameters (FASTER)
    inh_PV=IFParameter(
        τm=100pF / 10nS,      # Membrane time constant: PV faster
        El=-70mV,             # Leak reversal potential
        Vt=-53mV,             # Spike threshold
        Vr=-70mV,             # Reset potential
        R=1 / 10nS,           # Membrane resistance
    ),

    # SST population parameters (SLOWER)
    inh_SST=IFParameter(
        τm=200pF / 10nS,      # Membrane time constant: SST slower
        El=-70mV,             # Leak reversal potential
        Vt=-53mV,             # Spike threshold
        Vr=-70mV,             # Reset potential
        R=1 / 10nS,           # Membrane resistance
    ),

    # VIP population parameters (MEDIUM)
    inh_VIP=IFParameter(
        τm=150pF / 10nS,      # Membrane time constant: VIP medium
        El=-70mV,             # Leak reversal potential
        Vt=-53mV,             # Spike threshold
        Vr=-70mV,             # Reset potential
        R=1 / 10nS,           # Membrane resistance
    ),

    # Spiking threshold properties: same for all neurons 
    spike=PostSpike(τabs=5ms),         # Absolute refractory period

    # Synaptic properties: same for all neurons
    synapse=SingleExpSynapse(
        τi=5ms,             # Inhibitory synaptic time constant
        τe=5ms,             # Excitatory synaptic time constant
        E_i=-80mV,          # Inhibitory reversal potential
        E_e=0mV             # Excitatory reversal potential
    ),
    synapse_PV=SingleExpSynapse(τi=10ms, τe=5ms, E_i=-80mV, E_e=0mV),
    synapse_SST=SingleExpSynapse(τi=15ms, τe=5ms, E_i=-80mV, E_e=0mV),
    synapse_VIP=SingleExpSynapse(τi=7ms, τe=5ms, E_i=-80mV, E_e=0mV),


    # Connection probabilities and synaptic weights
    connections=(
        # from ThalExc
        ThalExc_to_CortExc=(p=0.05, μ=4nS, rule=:Fixed),
        ThalExc_to_CortPv=(p=0.05, μ=4nS, rule=:Fixed),
        # from CortExc
        CortExc_to_CortExc=(p=0.05, μ=2nS, rule=:Fixed),
        CortExc_to_CortPv=(p=0.05, μ=2nS, rule=:Fixed),
        CortExc_to_ThalExc=(p=0.05, μ=2nS, rule=:Fixed),        # CE_to_TE connection added
        # from CortPv
        CortPv_to_CortExc=(p=0.05, μ=10nS, rule=:Fixed),
        CortPv_to_CortPv=(p=0.05, μ=10nS, rule=:Fixed),
        CortPv_to_CortSst=(p=0.05, μ=10nS, rule=:Fixed),        # PV_to_SST connection added
        # from CortSst
        CortSst_to_CortExc=(p=0.025, μ=10nS, rule=:Fixed),
        CortSst_to_CortPv=(p=0.025, μ=10nS, rule=:Fixed),
        CortSst_to_CortVip=(p=0.025, μ=10nS, rule=:Fixed),
        # from CortVip
        CortVip_to_CortSst=(p=0.3, μ=10nS, rule=:Fixed),
    ),

    # Parameters for external Poisson input
    afferents_to_ThalExc=(
        layer=PoissonLayer(rate=1.5Hz, N=1000),           # Poisson input layer
        conn=(p=0.05, μ=4.0nS, rule=:Fixed),              # Connection probability and weight
    ),
    afferents_to_CortExc=(
        layer=PoissonLayer(rate=1.5Hz, N=1000),           # Poisson input layer
        conn=(p=0.02, μ=4.0nS, rule=:Fixed),              # Connection probability and weight
    ),
    afferents_to_CortPv=(
        layer=PoissonLayer(rate=1.5Hz, N=1000),           # Poisson input layer
        conn=(p=0.02, μ=4.0nS, rule=:Fixed),              # Connection probability and weight
    ),
    afferents_to_CortSst=(
        layer=PoissonLayer(rate=1.5Hz, N=1000),           # Poisson input layer
        conn=(p=0.15, μ=2.0nS, rule=:Fixed),              # Connection probability and weight
    ),
    afferents_to_CortVip=(
        layer=PoissonLayer(rate=1.5Hz, N=1000),           # Poisson input layer
        conn=(p=0.10, μ=2.0nS, rule=:Fixed),              # Connection probability and weight
    ),
)

# %% [markdown]
# ## Network Construction
#
# Define a function to create the network based on the configuration parameters.

# %%
# Function to create the network
function network(config)
    @unpack afferents_to_ThalExc, afferents_to_CortExc, afferents_to_CortPv, afferents_to_CortSst, afferents_to_CortVip, connections, Npop, spike, exc, inh_PV, inh_SST, inh_VIP = config
    @unpack synapse = config

    # Create neuron populations
    TE = Population(exc; synapse, spike, N=Npop.ThalExc, name="ThalExc")                # Excitatory population
    CE = Population(exc; synapse, spike, N=Npop.CortExc, name="CortExc")                # Excitatory population
    PV = Population(inh_PV; synapse_PV, spike, N=Npop.CortPvInh, name="CortPvInh")         # Inhibitory population
    SST = Population(inh_SST; synapse_SST, spike, N=Npop.CortSstInh, name="CortSstInh")     # Inhibitory population
    VIP = Population(inh_VIP; synapse_VIP, spike, N=Npop.CortVipInh, name="CortVipInh")     # Inhibitory population

    # Create external Poisson input
    @unpack layer = afferents_to_ThalExc
    afferentTE = Stimulus(layer, TE, :glu, conn=afferents_to_ThalExc.conn, name="bgTE")     # Excitatory input
    @unpack layer = afferents_to_CortExc
    afferentRE = Stimulus(layer, CE, :glu, conn=afferents_to_CortExc.conn, name="bgRE")     # Excitatory input
    @unpack layer = afferents_to_CortPv
    afferentPV = Stimulus(layer, PV, :glu, conn=afferents_to_CortPv.conn, name="bgPV")      # Excitatory input
    @unpack layer = afferents_to_CortSst
    afferentSST = Stimulus(layer, SST, :glu, conn=afferents_to_CortSst.conn, name="bgSST")  # Excitatory input
    @unpack layer = afferents_to_CortVip
    afferentVIP = Stimulus(layer, VIP, :glu, conn=afferents_to_CortSst.conn, name="bgVIP")  # Excitatory input

    # Create recurrent connections
    synapses = (
        # from ThalExc
        TE_to_CE=SpikingSynapse(TE, CE, :glu, conn=connections.ThalExc_to_CortExc),
        TE_to_PV=SpikingSynapse(TE, PV, :glu, conn=connections.ThalExc_to_CortPv),

        # from CortExc
        CE_to_CE=SpikingSynapse(CE, CE, :glu, conn=connections.CortExc_to_CortExc),
        CE_to_PV=SpikingSynapse(CE, PV, :glu, conn=connections.CortExc_to_CortPv),
        CE_to_TE=SpikingSynapse(CE, TE, :glu, conn=connections.CortExc_to_ThalExc),

        # from CortPv
        PV_to_CE=SpikingSynapse(PV, CE, :gaba, conn=connections.CortPv_to_CortExc),
        PV_to_PV=SpikingSynapse(PV, PV, :gaba, conn=connections.CortPv_to_CortPv),
        PV_to_SST=SpikingSynapse(PV, SST, :gaba, conn=connections.CortPv_to_CortSst),

        # from CortSst
        SST_to_CE=SpikingSynapse(SST, CE, :gaba, conn=connections.CortSst_to_CortExc),
        SST_to_PV=SpikingSynapse(SST, PV, :gaba, conn=connections.CortSst_to_CortPv),
        SST_to_VIP=SpikingSynapse(SST, VIP, :gaba, conn=connections.CortSst_to_CortVip),

        # from CortVip
        VIP_to_SST=SpikingSynapse(VIP, SST, :gaba, conn=connections.CortVip_to_CortSst),
    )

    # Compose the model
    model = compose(; TE, CE, PV, SST, VIP,
        afferentTE, afferentRE,
        afferentPV, afferentSST, afferentVIP, synapses...,
        name="thalamo-cortical network")

    # Set up monitoring
    monitor!(model.pop, [:fire])    # Monitor spikes
    monitor!(model.stim, [:fire])   # Monitor input spikes

    return model
end

# %% [markdown]
# ## Network Simulation
#
# Create the network and simulate it for a fixed duration.

# %%
# Create and simulate the network
model = network(TC3inhib_network)
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
    title="Raster plot of the balanced network")

# %%

# Plot vector field plot of network activity
SNN.vecplot(model.pop.CE, :v, neurons=13,
    title="Vecplot of the balanced network",
    xlabel="Time (s)",
    ylabel="Potential (mV)",
    lw=2,
    c=:darkcyan)

# %%

# Function to change the sample resolution of the plots
pop_spiketimes = SNN.spiketimes(model.pop.CE)
subsample = pop_spiketimes[1:10:end]
SNN.STTC(subsample, 10ms)

#%%

# %% [markdown]
# This section is to test the different values and hypothesis.
#
# Function to test different neuron parameters
config_test_inh_short = SNN.@update TC3inhib_network begin
    ThalExc_to_CortExc = (p=0.05, μ=4nS, rule=:Fixed)
    ThalExc_to_CortPv = (p=0.05, μ=4nS, rule=:Fixed)
    CortExc_to_CortExc = (p=0.05, μ=2nS, rule=:Fixed)
    CortExc_to_CortPv = (p=0.05, μ=2nS, rule=:Fixed)
    CortPv_to_CortExc = (p=0.05, μ=10nS, rule=:Fixed)
    CortPv_to_CortPv = (p=0.05, μ=10nS, rule=:Fixed)
    CortSst_to_CortExc = (p=0.025, μ=10nS, rule=:Fixed)
    CortSst_to_CortPv = (p=0.025, μ=10nS, rule=:Fixed)
    CortSst_to_CortVip = (p=0.025, μ=10nS, rule=:Fixed)
    CortVip_to_CortSst = (p=0.3, μ=10nS, rule=:Fixed)
end

# Measure the onset of epileptic activity (STTC)
myspikes = SNN.spiketimes(model.pop)[1:5:end]      # Subsample: only 1 every 5
sttc_value = mean(SNN.STTC(myspikes, 50ms))

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