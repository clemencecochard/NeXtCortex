# %%
# Import necessary packages and set up environment
using DrWatson
findproject(@__DIR__) |> quickactivate

using SpikingNeuralNetworks
using UnPack
using Logging
using Plots

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
import SpikingNeuralNetworks: IF, PoissonLayer, Stimulus, SpikingSynapse, compose, monitor!, sim!, firing_rate, @update, SingleExpSynapse, IFParameter, Population, PostSpike, AdExParameter


TC3inhib_network = (
    # Number of neurons in each population
    Npop = (ThalExc=200, CortExc=4000,
            CortPvInh=800, CortSstInh=100, CortVipInh=100),

    # Parameters for excitatory neurons
    exc = IFParameter(
                τm = 200pF / 10nS,  # Membrane time constant
                El = -70mV,         # Leak reversal potential
                Vt = -50.0mV,       # Spike threshold
                Vr = -70.0f0mV,     # Reset potential
                R  = 1/10nS,        # Membrane resistance
                ),

    # Parameters for inhibitory neurons
    inh = IFParameter(
                τm = 200pF / 10nS,  # Membrane time constant
                El = -70mV,         # Leak reversal potential
                Vt = -53.0mV,       # Spike threshold
                Vr = -70.0f0mV,     # Reset potential
                R  = 1/10nS,        # Membrane resistance
                ),

    # Spiking threshold properties - same for all neurons 
    spike = PostSpike(τabs = 5ms),         # Absolute refractory period

    # Synaptic properties - same for all neurons
    synapse = SingleExpSynapse(
                τi=5ms,             # Inhibitory synaptic time constant
                τe=5ms,             # Excitatory synaptic time constant
                E_i = -80mV,        # Inhibitory reversal potential
                E_e = 0mV           # Excitatory reversal potential
            ),

    # Connection probabilities and synaptic weights
    connections = (
        # from ThalExc
        ThalExc_to_CortExc = (p = 0.05, μ = 4nS,  rule=:Fixed), 
        ThalExc_to_CortPv = (p = 0.05, μ = 4nS,  rule=:Fixed),  
        # from CortExc
        CortExc_to_CortExc = (p = 0.05, μ = 2nS,  rule=:Fixed), 
        CortExc_to_CortPv = (p = 0.05, μ = 2nS,  rule=:Fixed),  
        # from CortPv
        CortPv_to_CortExc = (p = 0.05, μ = 10nS,  rule=:Fixed), 
        CortPv_to_CortPv = (p = 0.05, μ = 10nS,  rule=:Fixed),  
        # from CortSst
        CortSst_to_CortExc = (p = 0.025, μ = 10nS,  rule=:Fixed),
        CortSst_to_CortPv = (p = 0.025, μ = 10nS,  rule=:Fixed),
        CortSst_to_CortVip = (p = 0.025, μ = 10nS,  rule=:Fixed), 
        # from CortVip
        CortVip_to_CortSst= (p = 0.3, μ = 10nS,  rule=:Fixed),  
        ),

    # Parameters for external Poisson input
    afferents_to_ThalExc = (
        layer = PoissonLayer(rate=1.5Hz, N=1000), # Poisson input layer
        conn = (p = 0.05, μ = 4.0nS, rule=:Fixed), # Connection probability and weight
        ),
    afferents_to_CortExc = (
        layer = PoissonLayer(rate=1.5Hz, N=1000), # Poisson input layer
        conn = (p = 0.02, μ = 4.0nS, rule=:Fixed), # Connection probability and weight
        ),
    afferents_to_CortPv= (
        layer = PoissonLayer(rate=1.5Hz, N=1000), # Poisson input layer
        conn = (p = 0.02, μ = 4.0nS, rule=:Fixed), # Connection probability and weight
        ),
    afferents_to_CortSst= (
        layer = PoissonLayer(rate=1.5Hz, N=1000), # Poisson input layer
        conn = (p = 0.15, μ = 2.0nS, rule=:Fixed), # Connection probability and weight
        ),
    afferents_to_CortVip= (
        layer = PoissonLayer(rate=1.5Hz, N=1000), # Poisson input layer
        conn = (p = 0.10, μ = 2.0nS, rule=:Fixed), # Connection probability and weight
        ),
)

# %% [markdown]
# ## Network Construction
#
# Define a function to create the network based on the configuration parameters.

# %%
# Function to create the network
function network(config)
    @unpack afferents_to_ThalExc, afferents_to_CortExc, afferents_to_CortPv, afferents_to_CortSst, afferents_to_CortVip, connections, Npop, spike, exc, inh = config
    @unpack synapse = config

    # Create neuron populations
    TE = Population(exc; synapse, spike, N=Npop.ThalExc, name="ThalExc")  
    CE = Population(exc; synapse, spike, N=Npop.CortExc, name="CortExc")  
    PV = Population(inh; synapse, spike, N=Npop.CortPvInh, name="CortPvInh")
    SST = Population(inh; synapse, spike, N=Npop.CortSstInh, name="CortSstInh")
    VIP = Population(inh; synapse, spike, N=Npop.CortVipInh, name="CortVipInh")

    # Create external Poisson input
    @unpack layer = afferents_to_ThalExc
    afferentTE = Stimulus(layer, TE, :glu, conn=afferents_to_ThalExc.conn, name="bgTE")  # Excitatory input
    @unpack layer = afferents_to_CortExc
    afferentRE = Stimulus(layer, CE, :glu, conn=afferents_to_CortExc.conn, name="bgRE")  # Excitatory input
    @unpack layer = afferents_to_CortPv
    afferentPV = Stimulus(layer, PV, :glu, conn=afferents_to_CortPv.conn, name="bgPV")  # Excitatory input
    @unpack layer = afferents_to_CortSst
    afferentSST= Stimulus(layer, SST, :glu, conn=afferents_to_CortSst.conn, name="bgSST")  # Excitatory input
    @unpack layer = afferents_to_CortVip
    afferentVIP = Stimulus(layer, VIP, :glu, conn=afferents_to_CortSst.conn, name="bgSST")  # Excitatory input

    # Create recurrent connections
    synapses = (
        TE_to_CE = SpikingSynapse(TE, CE, :glu, conn = connections.ThalExc_to_CortExc),
        TE_to_PV= SpikingSynapse(TE, PV, :glu, conn = connections.ThalExc_to_CortPv),

        CE_to_CE = SpikingSynapse(CE, CE, :glu, conn = connections.CortExc_to_CortExc),
        CE_to_PV = SpikingSynapse(CE, PV, :glu, conn = connections.CortExc_to_CortPv),

        PV_to_CE = SpikingSynapse(PV, CE, :gaba, conn = connections.CortPv_to_CortExc),
        PV_to_PV = SpikingSynapse(PV, PV, :gaba, conn = connections.CortPv_to_CortPv),

        SST_to_CE = SpikingSynapse(SST, CE, :gaba, conn = connections.CortSst_to_CortExc),
        SST_to_PV = SpikingSynapse(SST, PV, :gaba, conn = connections.CortSst_to_CortPv),
        SST_to_VIP = SpikingSynapse(SST, VIP, :gaba, conn = connections.CortSst_to_CortVip),

        VIP_to_SST = SpikingSynapse(VIP, SST, :gaba, conn = connections.CortVip_to_CortSst),
    )

    # Compose the model
    model = compose(; TE, CE, PV, SST, VIP,
                    afferentTE, afferentRE, 
                    afferentPV, afferentSST, afferentVIP, synapses..., 
                    name="thalamo-cortical network")

    # Set up monitoring
    monitor!(model.pop, [:fire])  # Monitor spikes
    monitor!(model.stim, [:fire])  # Monitor input spikes

    return model
end

# %% [markdown]
# ## Network Simulation
#
# Create the network and simulate it for a fixed duration.

# %%
# Create and simulate the network
model = network(TC3inhib_network)
SNN.print_model(model)  # Print model summary
SNN.monitor!(model.pop, [:v])
SNN.sim!(model, duration=3s)  # Simulate for 5 seconds

# %% [markdown]
# ## Visualization
#
# Visualize the spiking activity of the network.

# %%
# Plot raster plot of network activity
SNN.raster(model.pop, every=1, 
            title="Raster plot of the balanced network")

# %%
SNN.vecplot(model.pop.CE, :v, neurons=13,
            xlabel="Time (s)", 
            ylabel="Potential (mV)", 
            lw=2, 
            c=:darkblue)
