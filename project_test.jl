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

# Logger
global_logger(ConsoleLogger())
SNN.@load_units

import SpikingNeuralNetworks: IF, PoissonLayer, Stimulus, SpikingSynapse, compose, monitor!, sim!, firing_rate, @update, SingleExpSynapse, IFParameter, Population, PostSpike, AdExParameter, STTC

# --------------------------------------------------------------
# Network configuration
# --------------------------------------------------------------

TC3inhib_network = (
    # Number of neurons in each population
    Npop=(ThalExc=200, CortExc=4000,
        CortPvInh=800, CortSstInh=100, CortVipInh=100),
    seed=1234,

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

    # Spiking threshold properties: absolute refractory period
    spike=PostSpike(τabs=5ms),
    spike_PV=PostSpike(τabs=2ms),
    spike_SST=PostSpike(τabs=10ms),
    spike_VIP=PostSpike(τabs=5ms),

    # Synaptic properties
    synapse=SingleExpSynapse(
        τi=5ms,             # Inhibitory synaptic time constant
        τe=5ms,             # Excitatory synaptic time constant
        E_i=-80mV,          # Inhibitory reversal potential
        E_e=0mV             # Excitatory reversal potential
    ),
    synapse_PV=SingleExpSynapse(τi=5ms, τe=5ms, E_i=-80mV, E_e=0mV),
    synapse_SST=SingleExpSynapse(τi=12ms, τe=5ms, E_i=-80mV, E_e=0mV),
    synapse_VIP=SingleExpSynapse(τi=7ms, τe=5ms, E_i=-80mV, E_e=0mV),


    # Connection probabilities and synaptic weights
    connections=(
        # from ThalExc
        ThalExc_to_CortExc=(p=0.05, μ=4nS, rule=:Fixed),
        ThalExc_to_CortPv=(p=0.05, μ=4nS, rule=:Fixed),
        # from CortExc
        CortExc_to_CortExc=(p=0.15, μ=2nS, rule=:Fixed),
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


# --------------------------------------------------------------
# Build and simulate
# --------------------------------------------------------------
model = NetworkUtils.build_network(TC3inhib_network)
SNN.monitor!(model.pop, [:v], sr=1kHz)
Random.seed!(TC3inhib_network.seed)
SNN.sim!(model, duration=3s)

# --------------------------------------------------------------
# Extract spike times
# --------------------------------------------------------------
pops = (:TE, :CE, :PV, :SST, :VIP)
spk = Dict(p => SNN.spiketimes(getfield(model.pop, p)) for p in pops)

# --------------------------------------------------------------
# Raster plot
# --------------------------------------------------------------
plt_raster = SNN.raster(model.pop, every=1, title="Raster plot of the balanced network")
savefig(plt_raster, "raster_full.png")

# --------------------------------------------------------------
# Firing rate helper
# --------------------------------------------------------------
function firing_rate(spiketimes, dt, T)
    nb = Int(ceil(T/dt))
    edges = collect(0:dt:nb*dt)
    counts = zeros(nb)
    for s in spiketimes
        counts .+= histcounts(s, edges)
    end
    rate = counts ./ (length(spiketimes)*dt)
    t = edges[1:end-1] .+ dt/2
    return t, rate
end

# --------------------------------------------------------------
# Firing rate plot
# --------------------------------------------------------------
frplt = plot(title="Population firing rates", xlabel="Time (s)", ylabel="Hz")
for p in pops
    t, r = firing_rate(spk[p], 0.01, 3.0)
    plot!(frplt, t, r, label=string(p))
end
savefig(frplt, "firing_rates.png")

# --------------------------------------------------------------
# Membrane potential traces (mean + std)
# --------------------------------------------------------------
function pop_vm(model, popsym::Symbol)
    pop = getfield(model.pop, popsym)
    v = pop.v
    if size(v,1) < size(v,2)
        v = v'
    end
    t = collect(0:1/model.monitor_sr:(size(v,1)-1)/model.monitor_sr)
    mv = mean(v, dims=2)[:]
    sv = std(v, dims=2)[:]
    return t, mv, sv
end

vmplt = plot(title="Vm dynamics", xlabel="Time (s)", ylabel="mV")
for p in pops
    try
        t, mv, sv = pop_vm(model, p)
        plot!(vmplt, t, mv, label=string(p))
        band!(vmplt, t, mv.-sv, mv.+sv, alpha=0.1)
    catch
    end
end
savefig(vmplt, "vm_dynamics.png")

# --------------------------------------------------------------
# STTC computation
# --------------------------------------------------------------
function pop_sttc(spikes; npairs=200, dt=0.05)
    ids = findall(s -> !isempty(s), spikes)
    if length(ids) < 2
        return NaN
    end
    pairs = []
    while length(pairs) < npairs
        i, j = rand(ids), rand(ids)
        i == j && continue
        push!(pairs, (min(i,j), max(i,j)))
    end
    vals = Float64[]
    for (i,j) in pairs
        try
            push!(vals, SNN.STTC([spikes[i], spikes[j]], dt))
        catch
        end
    end
    return mean(vals)
end

sttc = Dict(p => pop_sttc(spk[p]) for p in pops)
allspk = vcat([spk[p] for p in pops]...)
global_sttc = pop_sttc(allspk, npairs=500)

println("STTC per population:")
println(sttc)
println("Global STTC = ", global_sttc)

# ================= VIP→SST Modulation Experiment =================

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
