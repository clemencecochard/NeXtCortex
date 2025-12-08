module NetworkUtils

using SpikingNeuralNetworks
using UnPack
using Random
using Statistics
using Unitful
using Plots

SNN.@load_units

import SpikingNeuralNetworks: Population, Stimulus, SpikingSynapse, compose, monitor!, @update

export build_network, plot_firing_rates, plot_membrane_potentials

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
        name = "")
    # Build time axis (Unitful-safe)
    time_axis = 0ms:dt:T
    # Storage
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
               size = (400, 400))
    for p in pops
        plot!(plt, t, rates[p], label=String(p))
    end
    return plt
end

function plot_membrane_potentials(model;
        pops = (:TE, :CE, :PV, :SST, :VIP),
        neurons = 1:5,
        legend = false,
        name = "")

    plt = plot(layout=(length(pops),1), size=(400, 100*length(pops)), legend = legend)
    colors = (:darkorange, :darkgreen, :purple, :darkcyan, :blue)
    for (i,p) in enumerate(pops)

        vp = SNN.vecplot(model.pop[p], :v, neurons=neurons, add_spikes=true)

        for s in vp.series_list
            xs = s[:x]
            ys = s[:y]
            plot!(plt[i], xs, ys, lw=1.5, c=colors[i])
        end

        title!(plt[i], "$name membrane potential - $p")
        xlabel!(plt[i], "Time (s)")
        ylabel!(plt[i], "V (mV)")
    end

    return plt
end

end