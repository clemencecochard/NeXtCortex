module NetworkUtils

using SpikingNeuralNetworks
using UnPack
using Random
using Statistics
using Unitful
using Plots

SNN.@load_units

import SpikingNeuralNetworks: Population, Stimulus, SpikingSynapse, compose, monitor!, sim!, @update

function build_network(config)
    @unpack seed = config
    Random.seed!(seed)

    @unpack afferents_to_TE, afferents_to_CE, afferents_to_PV,
        afferents_to_SST, afferents_to_VIP, connections, Npop,
        spike, spike_PV, spike_SST, spike_VIP,
        exc, inh_PV, inh_SST, inh_VIP,
        synapse, synapse_PV, synapse_SST, synapse_VIP = config

    TE  = Population(exc;        synapse,       spike,       N=Npop.TE,     name="TE")
    CE  = Population(exc;        synapse,       spike,       N=Npop.CE,     name="CE")
    PV  = Population(inh_PV;     synapse=synapse_PV,  spike=spike_PV,  N=Npop.PV, name="PV")
    SST = Population(inh_SST;    synapse=synapse_SST, spike=spike_SST, N=Npop.SST,name="SST")
    VIP = Population(inh_VIP;    synapse=synapse_VIP, spike=spike_VIP, N=Npop.VIP,name="VIP")

    aff_TE  = Stimulus(afferents_to_TE.layer, TE,  :glu, conn=afferents_to_TE.conn)
    aff_CE  = Stimulus(afferents_to_CE.layer, CE,  :glu, conn=afferents_to_CE.conn)
    aff_PV  = Stimulus(afferents_to_PV.layer,  PV,  :glu, conn=afferents_to_PV.conn)
    aff_SST = Stimulus(afferents_to_SST.layer, SST, :glu, conn=afferents_to_SST.conn)
    aff_VIP = Stimulus(afferents_to_VIP.layer, VIP, :glu, conn=afferents_to_VIP.conn)

    syns = (
        TE_to_CE = SpikingSynapse(TE, CE, :glu,  conn=connections.TE_to_CE),
        TE_to_PV = SpikingSynapse(TE, PV, :glu,  conn=connections.TE_to_PV),

        CE_to_CE = SpikingSynapse(CE, CE, :glu, conn=connections.CE_to_CE),
        CE_to_PV = SpikingSynapse(CE, PV, :glu, conn=connections.CE_to_PV),
        CE_to_TE = SpikingSynapse(CE, TE, :glu, conn=connections.CE_to_TE),
        CE_to_SST = SpikingSynapse(CE, SST, :glu, conn=connections.CE_to_SST),
        CE_to_VIP = SpikingSynapse(CE, VIP, :glu, conn=connections.CE_to_VIP),

        PV_to_CE  = SpikingSynapse(PV, CE,  :gaba, conn=connections.PV_to_CE),
        PV_to_PV  = SpikingSynapse(PV, PV,  :gaba, conn=connections.PV_to_PV),
        PV_to_SST = SpikingSynapse(PV, SST, :gaba, conn=connections.PV_to_SST),

        SST_to_CE  = SpikingSynapse(SST, CE,  :gaba, conn=connections.SST_to_CE),
        SST_to_PV  = SpikingSynapse(SST, PV,  :gaba, conn=connections.SST_to_PV),
        SST_to_VIP = SpikingSynapse(SST, VIP, :gaba, conn=connections.SST_to_VIP),

        VIP_to_SST = SpikingSynapse(VIP, SST, :gaba, conn=connections.VIP_to_SST),
    )

    model = compose(; TE, CE, PV, SST, VIP,
        aff_TE, aff_CE, aff_PV, aff_SST, aff_VIP,
        syns..., name="TC3_inhib_network")

    monitor!(model.pop, [:fire])

    return model
end

function plot_firing_rates(model;
        pops = (:TE, :CE, :PV, :SST, :VIP),
        dt = 20ms,
        τ = 25ms,
        T = 3s,
        name = "",
        colors = (:darkorange, :darkgreen, :purple, :darkcyan, :darkblue))

    time_axis = 0ms:dt:T #time axis
    rates = Dict{Symbol, Vector{Float64}}()
    t = nothing

    # Compute firing rate per population
    for p in pops
        rates_mat, t_vec = SNN.firing_rate(model.pop[p], time_axis;
                                           sampling=dt, τ=τ)
        # mean across neurons
        pop_rate = vec(mean(rates_mat, dims=1))
        # convert ms → s
        t = Float64.(t_vec) ./ 1000
        rates[p] = pop_rate
    end

    # Plot all populations
    plt = plot(title="$name population firing rates",
               xlabel="Time (s)", ylabel="Firing rate (Hz)",
               lw=2, legend=:topright,
               size = (600, 400))
    for (i, p) in enumerate(pops)
        plot!(plt, t, rates[p], label=String(p), color=colors[i])
    end
    return plt
end

function plot_membrane_potentials(model;
        pops = (:TE, :CE, :PV, :SST, :VIP),
        neurons = 1:5,
        legend = false,
        name = "",
        colors = (:darkorange, :darkgreen, :purple, :darkcyan, :darkblue))

    plt = plot(layout=(length(pops),1), 
               size=(600, 150*length(pops)), 
               legend=legend,
               bottom_margin=-2Plots.mm,
               top_margin=-2Plots.mm,
               ylabel="V (mV)",
               titlefontsize=11,
               xguidefontsize=8,
               yguidefontsize=8,
               xtickfontsize=7,
               ytickfontsize=7)
    
    for (i,p) in enumerate(pops)

        vp = SNN.vecplot(model.pop[p], :v, neurons=neurons, add_spikes=true)

        for s in vp.series_list
            xs = s[:x]
            ys = s[:y]
            plot!(plt[i], xs, ys, lw=1.5, c=colors[i], label=String(p), legend = i)
        end

        if i == 1
            plot!(plt[i], 
                  title="$name membrane potentials",
                  xaxis=false)
        elseif i == length(pops)
            plot!(plt[i],
                  xlabel="Time (s)")
        else
            plot!(plt[i],
                  xaxis=false)
        end
    end

    return plt
end

function analysis(model; name = "Baseline", figs=true, save_figs=true, csv=false, μ=nothing, p=nothing)
    
    colors = (:darkorange, :darkgreen, :purple, :darkcyan, :darkblue)

    if figs
        # Raster plot
        rplt = SNN.raster(model.pop, every=1, title="$name raster plot")

        zrplt = SNN.raster(model.pop, every = 1, title = "$name raster plot zoomed")
        xlims!(zrplt, 2, 2.5) # Zoom x-axis
        ylims!(zrplt, 3500, 5200) # Zoom y-axis

        # Firing rate dynamics
        frplt = NetworkUtils.plot_firing_rates(model, name = name, colors = colors)

        # Membrane potential dynamics 
        vplt = NetworkUtils.plot_membrane_potentials(model, neurons = 1, name = name, colors = colors)
        
        if save_figs
            savefig(zrplt, "$img_path/$name raster_zoom.png")
            savefig(frplt, "$img_path/$name firing_rates.png")
            savefig(rplt, "$img_path/$name raster_full.png")
            savefig(vplt, "$img_path/$name membrane_potentials_dynamic.png")
        end
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
    if figs
        return rplt, zrplt, frplt, vplt
    else
        return nothing, nothing, nothing, nothing
    end
end

function sttc_timeseries(model; window = 50ms, step = 10ms, sim_time = 3.0)
    myspikes = SNN.spiketimes(model.pop)
    t = 0:step:sim_time-window
    sttc = Float64[]
    for ti in t
        spikes_window = map(myspikes) do s
            s[(s .≥ ti) .& (s .< ti + window)]
        end
        push!(sttc, mean(STTC(spikes_window, window)))
    end
    return t, sttc
end

function transition_time(t, sttc; high = 0.8, sim_time = 3.0)
    idx = findfirst(x -> x ≥ high, sttc)
    return idx === nothing ? sim_time : t[idx]
end

function average_transition_time(network, pop, μ, p; seeds=1:10)
    times = Float64[]
    for s in seeds
        Random.seed!(s)

        modulation = (; network.connections[pop]..., μ = μ, p = p)
        network_modified = (; network...,
            connections = (; network.connections..., pop => modulation)
        )
        model = NetworkUtils.build_network(network_modified)
        monitor!(model.pop, [:v], sr=1kHz)
        sim!(model, 3s)
        
        t, sttc = sttc_timeseries(model)
        tt = transition_time(t, sttc)
        push!(times, tt)
    end
    return mean(times)
end



end