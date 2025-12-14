module NetworkUtils

using SpikingNeuralNetworks
using UnPack
using Random
using Statistics
using Unitful
using Plots

SNN.@load_units

import SpikingNeuralNetworks:
    Population, Stimulus, SpikingSynapse,
    compose, monitor!, sim!, @update, STTC


"""
    build_network(config) -> model

Construct and return a spiking neural network model based on a configuration
`NamedTuple`.

The network consists of five populations:
- TE : excitatory thalamic neurons
- CE : excitatory cortical neurons
- PV : parvalbumin inhibitory interneurons
- SST: somatostatin inhibitory interneurons
- VIP: vasoactive intestinal peptide inhibitory interneurons

External afferent inputs and recurrent synaptic connections are defined by
the fields of `config`.

### Parameters
- `config`: `NamedTuple` containing all network parameters:
  - population sizes (`Npop`)
  - neuron models (`exc`, `inh_*`)
  - synapse models (`synapse`, `synapse_*`)
  - connectivity structures (`connections`)
  - afferent inputs
  - random seed

### Returns
- `model`: a `SpikingNeuralNetworks` composed model ready for simulation
"""
function build_network(config)

    # Ensure reproducibility
    @unpack seed = config
    Random.seed!(seed)

    # Unpack configuration fields
    @unpack afferents_to_TE, afferents_to_CE, afferents_to_PV,
            afferents_to_SST, afferents_to_VIP,
            connections, Npop,
            spike, spike_PV, spike_SST, spike_VIP,
            exc, inh_PV, inh_SST, inh_VIP,
            synapse, synapse_PV, synapse_SST, synapse_VIP = config

    # --- Populations ---------------------------------------------------------
    TE  = Population(exc;    synapse, spike, N=Npop.TE,  name="TE")
    CE  = Population(exc;    synapse, spike, N=Npop.CE,  name="CE")
    PV  = Population(inh_PV; synapse=synapse_PV,  spike=spike_PV,  N=Npop.PV,  name="PV")
    SST = Population(inh_SST;synapse=synapse_SST, spike=spike_SST, N=Npop.SST, name="SST")
    VIP = Population(inh_VIP;synapse=synapse_VIP, spike=spike_VIP, N=Npop.VIP, name="VIP")

    # --- External afferent inputs -------------------------------------------
    aff_TE  = Stimulus(afferents_to_TE.layer,  TE,  :glu, conn=afferents_to_TE.conn)
    aff_CE  = Stimulus(afferents_to_CE.layer,  CE,  :glu, conn=afferents_to_CE.conn)
    aff_PV  = Stimulus(afferents_to_PV.layer,  PV,  :glu, conn=afferents_to_PV.conn)
    aff_SST = Stimulus(afferents_to_SST.layer, SST, :glu, conn=afferents_to_SST.conn)
    aff_VIP = Stimulus(afferents_to_VIP.layer, VIP, :glu, conn=afferents_to_VIP.conn)

    # --- Recurrent synaptic connections -------------------------------------
    syns = (
        TE_to_CE = SpikingSynapse(TE, CE, :glu,  conn=connections.TE_to_CE),
        TE_to_PV = SpikingSynapse(TE, PV, :glu,  conn=connections.TE_to_PV),

        CE_to_CE  = SpikingSynapse(CE, CE,  :glu,  conn=connections.CE_to_CE),
        CE_to_PV  = SpikingSynapse(CE, PV,  :glu,  conn=connections.CE_to_PV),
        CE_to_TE  = SpikingSynapse(CE, TE,  :glu,  conn=connections.CE_to_TE),
        CE_to_SST = SpikingSynapse(CE, SST, :glu,  conn=connections.CE_to_SST),
        CE_to_VIP = SpikingSynapse(CE, VIP, :glu,  conn=connections.CE_to_VIP),

        PV_to_CE  = SpikingSynapse(PV, CE,  :gaba, conn=connections.PV_to_CE),
        PV_to_PV  = SpikingSynapse(PV, PV,  :gaba, conn=connections.PV_to_PV),
        PV_to_SST = SpikingSynapse(PV, SST, :gaba, conn=connections.PV_to_SST),

        SST_to_CE  = SpikingSynapse(SST, CE,  :gaba, conn=connections.SST_to_CE),
        SST_to_PV  = SpikingSynapse(SST, PV,  :gaba, conn=connections.SST_to_PV),
        SST_to_VIP = SpikingSynapse(SST, VIP, :gaba, conn=connections.SST_to_VIP),

        VIP_to_SST = SpikingSynapse(VIP, SST, :gaba, conn=connections.VIP_to_SST),
    )

    # Compose full network
    model = compose(; TE, CE, PV, SST, VIP,
        aff_TE, aff_CE, aff_PV, aff_SST, aff_VIP,
        syns..., name="TC3_inhib_network")

    # Monitor spike events for all populations
    monitor!(model.pop, [:fire])

    return model
end


"""
    plot_firing_rates(model; kwargs...) -> plot

Compute and plot population-averaged firing rates over time.

Firing rates are computed using a sliding window with exponential kernel
smoothing.

### Units
- Spike times are in milliseconds (ms)
- Time axis is converted to seconds (s) for plotting

### Keyword arguments
- `pops`: populations to include
- `dt`: sampling interval (default: 20 ms)
- `τ`: smoothing time constant (default: 25 ms)
- `T`: total duration (default: 3 s)
- `name`: title prefix
- `colors`: line colors per population
"""
function plot_firing_rates(model;
        pops = (:TE, :CE, :PV, :SST, :VIP),
        dt = 20ms,
        τ = 25ms,
        T = 3s,
        name = "",
        colors = (:darkorange, :darkgreen, :purple, :darkcyan, :darkblue),
    )

    time_axis = 0ms:dt:T
    rates = Dict{Symbol, Vector{Float64}}()
    t = nothing

    for p in pops
        rates_mat, t_vec = SNN.firing_rate(
            model.pop[p], time_axis;
            sampling=dt, τ=τ
        )

        # Average across neurons
        rates[p] = vec(mean(rates_mat, dims=1))

        # Convert ms to s for plotting
        t = Float64.(t_vec) ./ 1000
    end

    plt = plot(
        title="$name population firing rates",
        xlabel="Time (s)", 
        ylabel="Firing rate (Hz)",
        lw=2, 
        legend=:topright, 
        size=(600, 400)
    )

    for (i, p) in enumerate(pops)
        plot!(plt, t, rates[p], label=String(p), color=colors[i])
    end

    return plt
end


"""
    plot_membrane_potentials(model; kwargs...) -> plot

Plot membrane potential traces for selected neurons in each population.

Spike times are overlaid on the membrane potential traces.

### Keyword arguments
- `pops`: populations to display
- `neurons`: indices of neurons to plot per population
- `legend`: show legend or not
- `name`: title prefix
"""
function plot_membrane_potentials(model;
        pops = (:TE, :CE, :PV, :SST, :VIP),
        neurons = 1:5,
        legend = false,
        name = "",
        colors = (:darkorange, :darkgreen, :purple, :darkcyan, :darkblue))

    plt = plot(
        layout=(length(pops),1),
        size=(600, 150*length(pops)),
        legend=legend,
        ylabel="V (mV)"
    )

    for (i, p) in enumerate(pops)

        vp = SNN.vecplot(model.pop[p], :v,
                         neurons=neurons, add_spikes=true)

        for s in vp.series_list
            plot!(plt[i], s[:x], s[:y],
                  lw=1.5, c=colors[i], label=String(p))
        end

        if i == 1
            title!(plt[i], "$name membrane potentials")
        elseif i == length(pops)
            xlabel!(plt[i], "Time (s)")
        end
    end

    return plt
end


"""
    analysis(model, img_path; kwargs...)

Run a full analysis pipeline on a simulated model.

Includes:
- raster plots
- firing rate dynamics
- membrane potentials
- spike-time tiling coefficient (STTC)

Optionally saves figures and STTC values to disk.

### STTC computation
STTC is computed using a subsample of neurons (1 every 5)
with a coincidence window of 50 ms.
"""
function plot_analysis(model, img_path;
        name="Baseline",
        save_figs=true,
        csv=false,
        μ=nothing,
        p=nothing)

    colors = (:darkorange, :darkgreen, :purple, :darkcyan, :darkblue)

    
    rplt  = SNN.raster(model.pop, every=1, title="$name raster plot")
    zrplt = SNN.raster(model.pop, every=1, title="$name raster plot zoomed")
    xlims!(zrplt, 2, 2.5)
    ylims!(zrplt, 3500, 5200)

    frplt = plot_firing_rates(model, name=name, colors=colors)
    vplt  = plot_membrane_potentials(model, neurons=1, name=name, colors=colors)

    if save_figs
        savefig(zrplt, "$img_path/$name raster_zoom.png")
        savefig(frplt, "$img_path/$name firing_rates.png")
        savefig(rplt,  "$img_path/$name raster_full.png")
        savefig(vplt,  "$img_path/$name membrane_potentials_dynamic.png")
    end

    # --- STTC ---------------------------------------------------------------
    myspikes = SNN.spiketimes(model.pop)
    sttc_value = mean(STTC(myspikes[1:5:end], 50ms))

    if csv
        open("$img_path/$name sttc_results.csv", "a") do io
            write(io, string(μ, ",", p, ",", sttc_value, "\n"))
        end
    else
        open("$img_path/$name sttc_value.txt", "w") do io
            write(io, string(sttc_value))
        end
    end

    return rplt, zrplt, frplt, vplt
end

function network_modifications(network, p_values, µ_values, pops_to_modify, name, img_path; plots=true)
    plts = nothing
    for p in p_values, µ in µ_values
        network_modified = network
        for pop in pops_to_modify
            network_modified = (; network_modified..., 
                connections = (; 
                    network_modified.connections..., 
                    pop => (; network_modified.connections[pop]..., p = p, µ = µ)
                )
            )
        end
        model = build_network(network_modified)
        monitor!(model.pop, [:v], sr=1kHz)
        Random.seed!(network_modified.seed)
        sim!(model, 3s)
        if plots
            rplt, zrplt, frplt, vplt = plot_analysis(model, img_path, name = "$name p=$p µ=$µ");
            plts = plot(rplt, zrplt, frplt, vplt, layout=(2,2), size=(1000,1000))
            savefig(plts, "$img_path/$name tight_layout p=$p µ=$µ.png")
        end
    end
    return plts
end


function transition_time(model;
        pops = (:TE, :CE, :PV, :SST, :VIP),
        dt = 20ms,
        τ = 25ms,
        T = 3s,
        threshold = 15,   # in Hz
    )

    time_axis = 0ms:dt:T
    rates = Dict{Symbol, Vector{Float64}}()
    t = nothing
    ce_crossing_time = nothing

    for p in pops
        rates_mat, t_vec = SNN.firing_rate(
            model.pop[p], time_axis;
            sampling=dt, τ=τ
        )

        # Average across neurons
        rates[p] = vec(mean(rates_mat, dims=1))

        # Convert ms to s
        t = Float64.(t_vec) ./ 1000

        # Check for apparition of epileptic-like activity in CE
        if p == :CE
            idx = findfirst(r -> r ≥ threshold, rates[p])
            ce_crossing_time = idx === nothing ? nothing : t[idx]
        end
    end

    return ce_crossing_time
end

function plot_transition_time_vs_p(network, p_values, pops_to_modify, name, img_path)

    transition_times = Float64[]

    for p in p_values
        network_modified = network
        for pop in pops_to_modify
            network_modified = (; network_modified..., 
                connections = (; 
                    network_modified.connections..., 
                    pop => (; network_modified.connections[pop]..., p = p)
                )
            )
        end
        model = build_network(network_modified)
        monitor!(model.pop, [:v], sr=1kHz)
        Random.seed!(network_modified.seed)
        sim!(model, 3s)

        t_ce = transition_time(model)
        push!(transition_times,
              t_ce === nothing ? NaN : t_ce)
    end

    plt = plot(
        p_values,
        transition_times,
        xlabel = "p",
        ylabel = "Transition time (s)",
        title = "$name transition time plot",
        legend = false,
        size = (600, 400)
    )
    savefig(plt, "$img_path/$name transition_times_plot.png")
    return plt
end


function heatmap_transition_time(
        network,
        p_values,
        μ_values,
        pops_to_modify,
        name,
        img_path;
        T = 3s
    )

    # Matrix: rows = μ, cols = p
    transition_times = fill(NaN, length(μ_values), length(p_values))

    for (iμ, μ) in enumerate(μ_values)
        for (ip, p) in enumerate(p_values)

            # Immutable network update
            network_modified = network
            for pop in pops_to_modify
                network_modified = (;
                    network_modified...,
                    connections = (;
                        network_modified.connections...,
                        pop = (;
                            network_modified.connections[pop]...,
                            p = p,
                            μ = μ
                        )
                    )
                )
            end

            model = build_network(network_modified)
            monitor!(model.pop, [:fire])
            Random.seed!(network_modified.seed)
            sim!(model, T)

            t_ce = transition_time(model)
            display(t_ce)
            transition_times[iμ, ip] =
                t_ce === nothing ? NaN : t_ce
        end
    end

    plt = heatmap(
        p_values,
        μ_values,
        transition_times;
        xlabel = "p",
        ylabel = "μ",
        title = "$name CE transition time",
        colorbar_title = "Transition time (s)",
        size = (700, 500)
    )

    savefig(plt, "$img_path/$name transition_time_heatmap.png")

    return plt
end


end
