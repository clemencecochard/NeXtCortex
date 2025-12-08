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
using CSV
using DataFrames
import SpikingNeuralNetworks: @update

global_logger(ConsoleLogger())
SNN.@load_units

import SpikingNeuralNetworks: PoissonLayer, monitor!, sim!, SingleExpSynapse, IFParameter, PostSpike, STTC

img_path = "plots_and_images"

# --------------------------------------------------------------
# Network baseline configuration
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

        CE_to_CE = (p=0.05, μ=2nS, rule=:Fixed),
        CE_to_PV = (p=0.05, μ=2nS, rule=:Fixed),
        CE_to_TE = (p=0.05, μ=2nS, rule=:Fixed),
        CE_to_SST = (p=0.05, μ=2nS, rule=:Fixed),
        CE_to_VIP = (p=0.05, μ=2nS, rule=:Fixed),

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


# Build and simulate
model = NetworkUtils.build_network(TC3inhib_network)
monitor!(model.pop, [:v], sr=1kHz)

Random.seed!(TC3inhib_network.seed)
sim!(model, 3s)


function analysis(model; name = "Baseline", figs=true, csv=false, μ=nothing, p=nothing)
    if figs
        # Raster plot
        SNN.raster(model.pop, every=1,
                title="$name raster plot")
        savefig("$img_path/$name raster_full.png")

        plt = SNN.raster(model.pop,
                        every = 1,
                        title = "$name raster plot zoomed")
        xlims!(plt, 2, 2.5) # Zoom x-axis
        ylims!(plt, 3500, 5200) # Zoom y-axis
        savefig(plt, "$img_path/$name raster_zoom.png")

        # Firing rate dynamics
        frplt = NetworkUtils.plot_firing_rates(model, name = name)
        savefig(frplt, "$img_path/$name firing_rates.png")

        # Membrane potential dynamics 
        plt_v = NetworkUtils.plot_membrane_potentials(model, neurons = 1, name = name)
        savefig(plt_v, "$img_path/$name membrane_potentials_dynamic.png")
    end

    # STTC
    myspikes = SNN.spiketimes(model.pop)    
    sttc_value = mean(STTC(myspikes[1:5:end]  , 50ms)) # Using subsampled myspikes: only 1 every 5

    if csv
        csvfile = "$img_path/$name sttc_results.csv"

        open(csvfile, "a") do io
            write(io, string(μ, ",", p, ",", sttc_value, "\n"))
        end

    else
        open("$img_path/$name sttc_value.txt", "w") do io
            write(io, string(sttc_value))
        end
    end
end

analysis(model)

# --------------------------------------------------------------
# Epileptic-like activity (increasing CE_to_CE)
# --------------------------------------------------------------

TC3inhib_network_modified = (; TC3inhib_network..., 
    connections = (; TC3inhib_network.connections..., 
        CE_to_CE = (; TC3inhib_network.connections.CE_to_CE..., p = 0.15)
    )
)

model = NetworkUtils.build_network(TC3inhib_network_modified)
monitor!(model.pop, [:v], sr=1kHz)
Random.seed!(TC3inhib_network_modified.seed)
sim!(model, 3s)

analysis(model, name = "Epileptic-like state")

# --------------------------------------------------------------
# Thalamic increased connection (increasing TE_to_CE)
# --------------------------------------------------------------

p_values = [0.10, 0.15]

for p in p_values
    TC3inhib_network_modified = (; TC3inhib_network..., 
        connections = (; TC3inhib_network.connections..., 
            TE_to_CE = (; TC3inhib_network.connections.TE_to_CE..., p = p)
        )
    )

    model = NetworkUtils.build_network(TC3inhib_network_modified)
    monitor!(model.pop, [:v], sr=1kHz)
    Random.seed!(TC3inhib_network_modified.seed)
    sim!(model, 3s)

    analysis(model, name = "Thalamic increase p=$p")
end

# --------------------------------------------------------------
# Slower inhibition test (increasing PV membrane time constant)
# --------------------------------------------------------------

TC3inhib_network_modified = (; TC3inhib_network..., 
    synapse_PV  = SingleExpSynapse(τi=20ms, τe=5ms, E_i=-80mV, E_e=0mV),
)

model = NetworkUtils.build_network(TC3inhib_network_modified)
monitor!(model.pop, [:v], sr=1kHz)
Random.seed!(TC3inhib_network_modified.seed)
sim!(model, 3s)

analysis(model, name = "Slower inhibition")


# --------------------------------------------------------------
# Modulations Experiments
# --------------------------------------------------------------

pops_to_modify = (:VIP_to_SST, :PV_to_CE, :SST_to_CE, :TE_to_CE)

μ_values = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10]
p_values = [0.1, 0.2, 0.7, 1.0]

for pop in pops_to_modify

    csvfile = "$img_path/Modulation $pop sttc_results.csv"
    open(csvfile, "w") do io
        write(io, "mu,p,sttc\n")
    end

    for μ in μ_values, p in p_values

        modulation = (; TC3inhib_network.connections[pop]..., μ = μ, p = p)
        TC3inhib_network_modified = (; TC3inhib_network...,
            connections = (; TC3inhib_network.connections..., pop => modulation)
        )

        model = NetworkUtils.build_network(TC3inhib_network_modified)
        monitor!(model.pop, [:v], sr=1kHz)
        Random.seed!(TC3inhib_network_modified.seed)
        sim!(model, 3s)

        analysis(model; name = "Modulation $pop", figs=false, csv = true, μ = μ, p = p)
    end

    df = CSV.read("$img_path/Modulation $pop sttc_results.csv", DataFrame)
    M = [ df[(df.mu .== μ) .& (df.p .== p), :sttc][1]
          for μ in μ_values, p in p_values ]

    sttc_heatmap = heatmap(
        p_values, μ_values, M,
        xlabel="p scale",
        ylabel="μ scale",
        title="Modulation $pop effect on synchrony",
        color = :viridis,
        clims = (0, 1) # to get the same color scale every time
    )

    savefig(sttc_heatmap, "$img_path/Modulation $pop sttc_heatmap.png")
end
