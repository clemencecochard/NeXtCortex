# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Julia 1.11.5
#     language: julia
#     name: julia-1.11
# ---

# %% [markdown]
# # Thalamocortical Circuit Simulation for Epileptic Seizure Transitions
#
# This notebook implements the thalamocortical circuit model from:
# "Combined Effects of Feedforward Inhibition and Excitation in Thalamocortical Circuit 
# on the Transitions of Epileptic Seizures" by Denggui Fan et al.
#
# The model includes cortical pyramidal cells, interneurons, thalamic relay cells, 
# and reticular thalamic neurons with feedforward inhibition/excitation pathways.

# %%
# Import necessary packages and set up environment
using DrWatson
findproject(@__DIR__) |> quickactivate

using UnPack
using Logging
using Plots
using LinearAlgebra

using SpikingNeuralNetworks

# Set global logger to display messages in console
global_logger(ConsoleLogger())

# Load units for physical quantities
SNN.@load_units

# %% [markdown]
# ## Network Configuration for Thalamocortical Circuit
#
# Define the network parameters based on the paper's model.

# %%
# Define thalamocortical circuit configuration
import SpikingNeuralNetworks: IF, PoissonLayer, Stimulus, SpikingSynapse, compose, monitor!, sim!, firing_rate, @update, SingleExpSynapse, IFParameter, Population, PostSpike

Thalamocortical_network = (
    # Population sizes :
    Npop = (PC=800,    # Pyramidal Cells (cortical)
        IN=200,    # Interneurons (cortical) 
        TC=600,    # Thalamocortical Relay Cells
        RE=400,    # Reticular Thalamic Neurons
    ),


    # Neuron parameters - Adapted for epileptiform activity
    # Cortical Pyramidal Cells
    PC = IFParameter(
        τm = 20ms,        # Membrane time constant
        El = -65mV,       # Leak reversal potential
        Vt = -50mV,       # Spike threshold
        Vr = -70mV,       # Reset potential
        R  = 1/1nS,       # Membrane resistance
    ),

    # Cortical Interneurons
    IN = IFParameter(
        τm = 10ms,
        El = -65mV,
        Vt = -50mV,
        Vr = -70mV,
        R  = 1/1nS,
    ),

    # Thalamocortical Relay Cells
    TC = IFParameter(
        τm = 15ms,
        El = -65mV,
        Vt = -55mV,       # Lower threshold for burst firing
        Vr = -70mV,
        R  = 1/1nS,
    ),

    # Reticular Thalamic Neurons
    RE = IFParameter(
        τm = 10ms,
        El = -65mV,
        Vt = -55mV,
        Vr = -70mV,
        R  = 1/1nS,
    ),

    # Refractory periods
    spike = PostSpike(τabs = 2ms),

    # Synaptic properties - Critical for seizure transitions
    synapse = SingleExpSynapse(
        τe=2ms,           # Excitatory synaptic time constant
        τi=5ms,           # Inhibitory synaptic time constant  
        E_e = 0mV,        # Excitatory reversal potential
        E_i = -80mV,      # Inhibitory reversal potential
    ),

    # Connection probabilities and weights - Key parameters from paper
    connections = (
        # Cortical connections
        PC_to_PC = (p = 0.02, μ = 1.5nS, rule=:Fixed),   # Recurrent excitation
        PC_to_IN = (p = 0.15, μ = 2.0nS, rule=:Fixed),   # Feedforward to interneurons
        IN_to_PC = (p = 0.20, μ = 4.0nS, rule=:Fixed),   # Feedback inhibition
        
        # Thalamocortical connections  
        TC_to_PC = (p = 0.10, μ = 3.0nS, rule=:Fixed),   # Thalamic drive to cortex
        TC_to_IN = (p = 0.10, μ = 2.5nS, rule=:Fixed),   # Thalamic drive to interneurons
        TC_to_RE = (p = 0.25, μ = 2.0nS, rule=:Fixed),   # Relay to reticular
        
        # Reticular nucleus connections
        RE_to_TC = (p = 0.30, μ = 6.0nS, rule=:Fixed),   # Strong inhibition to relay cells
        RE_to_RE = (p = 0.05, μ = 3.0nS, rule=:Fixed),   # Recurrent inhibition in RTN
        
        # Corticothalamic connections
        PC_to_RE = (p = 0.08, μ = 2.0nS, rule=:Fixed),   # Cortical control of RTN
        PC_to_TC = (p = 0.05, μ = 1.5nS, rule=:Fixed),   # Cortical feedback to thalamus
    ),

    # Parameters for external Poisson input
    afferents = (
        N = 200,                        # Number of input neurons
        rate = 15Hz,                    # Base firing rate
        conn = (p = 0.1, μ = 2.0nS),    # Connection probability and weight
    )
)

# %% [markdown]
# ## Network Construction
#
# Build the thalamocortical circuit with feedforward inhibition/excitation pathways.

# %%
function build_thalamocortical_network(config)
    @unpack afferents, connections, Npop, synapse, spike, PC, IN, TC, RE = config

    # Create cortical populations
    pyramidal = Population(PC; synapse, spike, N=Npop.PC, name="Pyramidal")
    interneuron = Population(IN; synapse, spike, N=Npop.IN, name="Interneuron")

    # Create thalamic populations  
    relay = Population(TC; synapse, spike, N=Npop.TC, name="ThalamicRelay")
    reticular = Population(RE; synapse, spike, N=Npop.RE, name="Reticular")

    # External Poisson inputs to different populations
    AfferentParam = PoissonLayer(rate=afferents.rate, N=afferents.N)
    afferent_PC = Stimulus(AfferentParam, pyramidal, :ge, conn=afferents.conn, name="noise_PC")
    afferent_IN = Stimulus(AfferentParam, interneuron, :ge, conn=afferents.conn, name="noise_IN") 
    afferent_TC = Stimulus(AfferentParam, relay, :ge, conn=afferents.conn, name="noise_TC")
    afferent_RE = Stimulus(AfferentParam, reticular, :ge, conn=afferents.conn, name="noise_RE")

    # Create all synaptic connections based on paper's circuitry
    synapses = (
        # Cortical microcircuit
        PC_to_PC = SpikingSynapse(pyramidal, pyramidal, :ge, conn=connections.PC_to_PC, name="PC_to_PC"),
        PC_to_IN = SpikingSynapse(pyramidal, interneuron, :ge, conn=connections.PC_to_IN, name="PC_to_IN"),
        IN_to_PC = SpikingSynapse(interneuron, pyramidal, :gi, conn=connections.IN_to_PC, name="IN_to_PC"),
        
        # Thalamocortical feedforward pathways
        TC_to_PC = SpikingSynapse(relay, pyramidal, :ge, conn=connections.TC_to_PC, name="TC_to_PC"),
        TC_to_IN = SpikingSynapse(relay, interneuron, :ge, conn=connections.TC_to_IN, name="TC_to_IN"),
        
        # Thalamic interactions
        TC_to_RE = SpikingSynapse(relay, reticular, :ge, conn=connections.TC_to_RE, name="TC_to_RE"),
        RE_to_TC = SpikingSynapse(reticular, relay, :gi, conn=connections.RE_to_TC, name="RE_to_TC"),
        RE_to_RE = SpikingSynapse(reticular, reticular, :gi, conn=connections.RE_to_RE, name="RE_to_RE"),
        
        # Corticothalamic feedback
        PC_to_RE = SpikingSynapse(pyramidal, reticular, :ge, conn=connections.PC_to_RE, name="PC_to_RE"),
        PC_to_TC = SpikingSynapse(pyramidal, relay, :ge, conn=connections.PC_to_TC, name="PC_to_TC"),
    )

    # Compose the model
    model = compose(;
        pyramidal, interneuron, relay, reticular,
        afferent_PC, afferent_IN, afferent_TC, afferent_RE,
        synapses...,
        silent=true,
        name="ThalamocorticalEpilepsyModel"
    )

    # Set up monitoring
    monitor!(model.pop, [:fire, :v])  # Monitor spikes
    monitor!(model.stim, [:fire, :v])  # Monitor input spikes

    return model
end

# %% [markdown]
# ## Parameter Modulation for Seizure Transitions
#
# Define parameters that can induce seizure-like activity.

# %%
# Parameters to modulate for seizure transitions
seizure_params = (
    normal = (
        PC_to_PC_scale = 1.0,    # Recurrent excitation strength
        IN_to_PC_scale = 1.0,    # Inhibition strength  
        RE_to_TC_scale = 1.0,    # Thalamic inhibition
        external_noise = 15Hz,   # Background activity
    ),
    
    pre_seizure = (
        PC_to_PC_scale = 1.8,    # Increased recurrent excitation
        IN_to_PC_scale = 0.7,    # Reduced inhibition
        RE_to_TC_scale = 0.6,    # Reduced thalamic inhibition
        external_noise = 25Hz,   # Increased background
    ),
    
    seizure = (
        PC_to_PC_scale = 2.5,    # Strong recurrent excitation
        IN_to_PC_scale = 0.3,    # Severely reduced inhibition
        RE_to_TC_scale = 0.2,    # Minimal thalamic inhibition
        external_noise = 35Hz,   # High background
    ),
)

# %% [markdown]
# ## Simulation with Dynamic Parameter Changes
#
# Simulate the transition between normal, pre-seizure, and seizure states.

# %%
# Create the network
model = build_thalamocortical_network(Thalamocortical_network)
SNN.print_model(model)  # Print model summary
SNN.sim!(model, duration=5s)  # Simulate for 5 seconds


# Define simulation phases
phases = [
    (state=:normal, duration=2s, params=seizure_params.normal),
    (state=:pre_seizure, duration=1.5s, params=seizure_params.pre_seizure),
    (state=:seizure, duration=2s, params=seizure_params.seizure),
    (state=:recovery, duration=1.5s, params=seizure_params.normal)
]

for phase in phases
    @info "Simulating $(phase.state) for $(phase.duration)"
    @unpack PC_to_PC_scale, IN_to_PC_scale, RE_to_TC_scale, external_noise = phase.params

    # Instead of scaling synapses, adjust external input noise or delay
    model.stim[:afferent_PC].param.rates .= external_noise
    model.stim[:afferent_IN].param.rates .= external_noise
    model.stim[:afferent_TC].param.rates .= external_noise
    model.stim[:afferent_RE].param.rates .= external_noise

    # Optionally adjust synapse time‐constants if those fields exist
    # e.g. syn.param.τe or τi — check with fieldnames call
    # if hasfield(typeof(syn.param), :τe)
    #     syn.param.τe .= syn.param.τe * some_scale
    # end

    SNN.sim!(model, duration=phase.duration)
end



# %% [markdown]
# ## Results Analysis
#
# Analyze the firing rates and membrane potentials of the neurons.

# %%
# Compute population firing rates for all populations
# 'pop_average=true' gives the mean firing rate across neurons
fr, r, labels = SNN.firing_rate(model.pop, interval=0f0:10ms:7s, pop_average=true)

# Plot the population firing rates
plot(r, fr, labels=hcat(labels...),
     xlabel="Time (s)", ylabel="Firing rate (Hz)",
     title="Population firing rates", lw=2, c=[:darkred :darkblue :green :purple])


# %% [markdown]
# ## Raster Plots
#
# Visualize spiking activity of cortical and thalamic populations.

# %%
plot_cortical = SNN.raster((model.pop[:pyramidal], model.pop[:interneuron]), every=5, title="Cortical Raster")
plot_thalamic = SNN.raster((model.pop[:relay], model.pop[:reticular]), every=5, title="Thalamic Raster")

# %% [markdown]
# ## Firing Rate Plots
#
# Compute and plot firing rates for each population.

# %%
fr_pyr, t = SNN.firing_rate(model.pop[:pyramidal], interval=0f0:100ms:7s, pop_average=true)
fr_in, _ = SNN.firing_rate(model.pop[:interneuron], interval=0f0:100ms:7s, pop_average=true)
fr_relay, _ = SNN.firing_rate(model.pop[:relay], interval=0f0:100ms:7s, pop_average=true)
fr_ret, _ = SNN.firing_rate(model.pop[:reticular], interval=0f0:100ms:7s, pop_average=true)

plot_fr_cortical = plot(t, fr_pyr, label="Pyramidal", lw=2, c=:blue)
plot!(t, fr_in, label="Interneuron", lw=2, c=:red, title="Cortical Firing Rates", xlabel="Time (s)", ylabel="Hz")

plot_fr_thalamic = plot(t, fr_relay, label="Relay", lw=2, c=:green)
plot!(t, fr_ret, label="Reticular", lw=2, c=:purple, title="Thalamic Firing Rates", xlabel="Time (s)", ylabel="Hz")

# Combine all plots
plot(plot_cortical, plot_thalamic, plot_fr_cortical, plot_fr_thalamic, layout=(2,2),
     plot_title="Thalamocortical Seizure Dynamics", size=(1200,800))

# %% [markdown]
# ## Membrane Potentials
#
# Plot representative neurons from cortical and thalamic populations.

# %%
v_pyr, t_pyr = SNN.record(model.pop[:pyramidal], :v, range=true)
v_relay, t_relay = SNN.record(model.pop[:relay], :v, range=true)

plot_v_pyr = plot(t_pyr, v_pyr[1,:], label="Pyramidal 1", lw=1.5, c=:blue, ylabel="Membrane (mV)")
plot_v_relay = plot(t_relay, v_relay[1,:], label="Relay 1", lw=1.5, c=:green, ylabel="Membrane (mV)")

plot(plot_v_pyr, plot_v_relay, layout=(2,1), plot_title="Representative Neuron Dynamics",
     xlabel="Time (s)", size=(1000,600))

# %% [markdown]
# ## Feedforward Inhibition/Excitation Balance
#
# Compute the ratio of feedforward inhibition to excitation for each state.

# %%
function calculate_ff_ratio_from_pop(model, state_window)
    t_start, t_end = state_window
    interval = t_start:1ms:t_end

    # Compute firing rates directly
    fr_pyr, t = SNN.firing_rate(model.pop.pyramidal, interval=interval)
    fr_in,  _ = SNN.firing_rate(model.pop.interneuron, interval=interval)
    fr_relay, _ = SNN.firing_rate(model.pop.relay, interval=interval)

    # Total spikes in window = sum over time bins
    ff_excitation = sum(fr_relay) * 0.001   # multiply by dt=1ms to get spikes
    ff_inhibition = sum(fr_in) * 0.001

    return ff_inhibition / (ff_excitation + 1e-9)
end

# Compute for all states
ff_ratios = Dict()
for (state, window) in state_windows
    ff_ratios[state] = calculate_ff_ratio_from_pop(model, window)
end

println("\nFeedforward Inhibition/Excitation Ratios:")
for (state, ratio) in ff_ratios
    println("$state: $(round(ratio,digits=3))")
end
