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
        neurons = 1,
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
        save_figs=true)

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

    open("$img_path/$name sttc_value.txt", "w") do io
        write(io, string(sttc_value))
    end

    return rplt, zrplt, frplt, vplt
end

"""
    network_modifications(network, p_values, µ_values, pops_to_modify, name, img_path; plots=true)

Systematically modify network connectivity parameters and simulate the resulting models.

For each combination of connection probability `p` and mean synaptic weight `µ`,
the function:
1. Updates the specified populations' connectivity parameters
2. Builds a new network
3. Runs a simulation
4. Optionally performs a full analysis and saves plots

### Parameters
- `network`: base network configuration (`NamedTuple`)
- `p_values`: iterable of connection probabilities to test
- `µ_values`: iterable of mean synaptic weights to test
- `pops_to_modify`: populations whose connections are modified
- `name`: label prefix for plots and files
- `img_path`: directory where outputs are saved

### Keyword arguments
- `plots`: whether to generate and save analysis plots (default: `true`)

### Returns
- `plts`: combined plot object of the last simulation (or `nothing` if `plots=false`)
"""
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

"""
    transition_time(model; kwargs...) -> Union{Float64, Nothing}

Compute the time at which excitatory cortical activity crosses a firing-rate threshold.

Population-averaged firing rates are computed using a smoothed sliding window.
The transition time is defined as the first time point at which the CE population
exceeds the specified firing-rate threshold.

### Keyword arguments
- `pops`: populations to analyze
- `dt`: sampling interval for firing-rate computation
- `τ`: smoothing time constant
- `T`: total simulation duration
- `threshold`: firing-rate threshold in Hz

### Returns
- Transition time in seconds if threshold is crossed
- `nothing` if no transition occurs
"""
function transition_time(model;
        pops = (:TE, :CE, :PV, :SST, :VIP),
        dt = 20ms,
        τ = 25ms,
        T = 3s,
        threshold = 35,   # in Hz
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

"""
    plot_transition_time_vs_p(network, p_values, pops_to_modify, name, img_path; plots=true)

Evaluate and plot transition times as a function of connection probability `p`.

For each value of `p`, the network is rebuilt, simulated, and analyzed to determine
the transition time of CE activity.

### Parameters
- `network`: base network configuration
- `p_values`: iterable of connection probabilities
- `pops_to_modify`: populations whose connections are modified
- `name`: label prefix for plots
- `img_path`: directory for saved figures

### Keyword arguments
- `plots`: whether to generate and save a plot (default: `true`)

### Returns
- Plot object if `plots=true`
- Vector of transition times otherwise
"""
function plot_transition_time_vs_p(network, p_values, pops_to_modify, name, img_path; plots=true)

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

    if plots
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
    else
        return transition_times
    end
end

"""
    plot_mean_transition_time_vs_p(network, p_values, pops_to_modify, name, img_path; kwargs...)

Compute and plot mean transition times across multiple random seeds.

For each value of `p`, the network is simulated multiple times with different seeds.
Transition times are averaged across runs, ignoring failed (NaN) transitions.

This function has never been used completely due to a lack of computer power to run it in less than hours.

### Parameters
- `network`: base network configuration
- `p_values`: iterable of connection probabilities
- `pops_to_modify`: populations whose connections are modified
- `name`: label prefix for plots
- `img_path`: directory for saved figures

### Keyword arguments
- `plots`: whether to generate and save a plot (default: `true`)
- `n_seeds`: number of random seeds per `p` value (default: 10)

### Returns
- Plot object if `plots=true`
- Vector of mean transition times otherwise
"""
function plot_mean_transition_time_vs_p(
    network,
    p_values,
    pops_to_modify,
    name,
    img_path;
    plots = true,
    n_seeds = 10
)

    mean_transition_times = Float64[]

    for p in p_values
        seed_transition_times = Float64[]

        for seed in 1:n_seeds
            # Modify network with new p
            network_modified = network
            for pop in pops_to_modify
                network_modified = (; network_modified...,
                    connections = (;
                        network_modified.connections...,
                        pop => (;
                            network_modified.connections[pop]...,
                            p = p
                        )
                    ),
                    seed = seed
                )
            end

            model = build_network(network_modified)
            monitor!(model.pop, [:v], sr = 1kHz)

            Random.seed!(network_modified.seed)
            sim!(model, 3s)

            t_ce = transition_time(model)
            push!(seed_transition_times,
                  t_ce === nothing ? NaN : t_ce)
        end

        # Mean over seeds (ignore NaNs)
        mean_t = mean(skipmissing(seed_transition_times))
        push!(mean_transition_times, mean_t)
    end

    if plots
        plt = plot(
            p_values,
            mean_transition_times,
            xlabel = "p",
            ylabel = "Mean transition time (s)",
            title = "$name mean transition time (n=$n_seeds)",
            legend = false,
            size = (600, 400),
            lw = 2,
            marker = :circle
        )

        savefig(plt, "$img_path/$name_mean_transition_times_plot.png")
        return plt
    else
        return mean_transition_times
    end
end

"""
    heatmap_transition_time_test(network, p_values, μ_values, pops_to_modify, name, img_path)

Compute transition times over a grid of synaptic parameters.

For each value of mean synaptic weight `µ`, transition times are computed as a
function of connection probability `p`. This output was meant to be used for heatmap
visualization but we never achieved to compute proper heatmap.

### Parameters
- `network`: base network configuration
- `p_values`: iterable of connection probabilities
- `μ_values`: iterable of mean synaptic weights
- `pops_to_modify`: populations whose connections are modified
- `name`: label prefix
- `img_path`: directory for saved outputs

### Returns
- Nested vector of transition times indexed by `(µ, p)`
"""
function heatmap_transition_time_test(network, p_values, μ_values, pops_to_modify, name, img_path)
    transition_times = Vector{Vector{Float64}}()
    for µ in μ_values
        network_modified = network
        for pop in pops_to_modify
            network_modified = (; network_modified..., 
                connections = (; 
                    network_modified.connections..., 
                    pop => (; network_modified.connections[pop]..., µ = µ)
                )
            )
        end
        push!(transition_times,
              plot_transition_time_vs_p(network_modified, p_values, pops_to_modify, name, img_path; plots=false))
    end
    return transition_times
end

end
