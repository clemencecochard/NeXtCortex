module NetworkUtils

using SpikingNeuralNetworks
using UnPack
using Random
using Statistics
using Unitful
using Plots

SNN.@load_units

import SpikingNeuralNetworks: IF, PoissonLayer, Stimulus, SpikingSynapse, compose, monitor!, sim!, @update, SingleExpSynapse, IFParameter, Population, PostSpike, AdExParameter, STTC

export build_network, plot_firing_rates, scaled_connection, run_condition, sweep_parameters, sweep_TC_feedback, sweep_sttc_time, sttc_over_time, firing_rate

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
        T = 3s)
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
    plt = plot(title="Population firing rates",
               xlabel="Time (s)", ylabel="Firing rate (Hz)",
               lw=2, legend=:topright)
    for p in pops
        plot!(plt, t, rates[p], label=String(p))
    end
    return t, rates, plt
end

function plot_membrane_potentials(model;
        pops = (:TE, :CE, :PV, :SST, :VIP),
        neurons = 1:5,
        legend = false)

    plt = plot(layout=(length(pops),1), size=(800, 200*length(pops)), legend = false)
    colors = (:darkorange, :darkgreen, :purple, :darkcyan, :blue)
    for (i,p) in enumerate(pops)

        vp = SNN.vecplot(model.pop[p], :v, neurons=neurons)

        for s in vp.series_list
            xs = s[:x]
            ys = s[:y]
            plot!(plt[i], xs, ys, lw=1.5, c=colors[i])
        end

        title!(plt[i], string(p))
        xlabel!(plt[i], "Time (s)")
        ylabel!(plt[i], "V (mV)")
    end

    return plt
end


function scaled_connection(conn; μ_scale=1.0, p_scale=1.0)
    return (
        p = clamp(conn.p * p_scale, 0, 1.0),
        μ = conn.μ * μ_scale,
        rule = conn.rule
    )
end

function run_condition(config; target, μ_scale=1.0, p_scale=1.0)
    base_conn = config.connections[target]
    newconn = scaled_connection(base_conn; μ_scale, p_scale)

    newconfig = @update config begin
        connections = merge(config.connections, (target => newconn,))
    end

    model = build_network(newconfig)
    sim!(model, 3s)

    spikes = SNN.spiketimes(model.pop.CE)[1:5:end]
    spikes = filter(!isempty, spikes)
    return mean(STTC(spikes, 50ms))
end

function firing_rate(spiketimes, dt, T)
    nb = Int(ceil(T/dt))
    edges = collect(0:dt:nb*dt)
    counts = zeros(nb)
    for s in spiketimes
        for spike in s
            bin = Int(floor(spike/dt)) + 1
            if 1 ≤ bin ≤ nb
                counts[bin] += 1
            end
        end
    end
    rate = counts ./ (length(spiketimes) * dt)
    t = edges[1:end-1] .+ dt/2
    return t, rate
end


function sttc_over_time(spikes; window=100ms, step=100ms, total=3s)
    times = 0:step:(total - window)
    sttcs = Float64[]

    for t in times
        local_spikes = [
            filter(s -> t <= s <= t + window, neuron_spikes)
            for neuron_spikes in spikes
        ]
        local_spikes = filter(!isempty, local_spikes)

        if !isempty(local_spikes)
            push!(sttcs, mean(STTC(local_spikes, 50ms)))
        else
            push!(sttcs, NaN)
        end
    end

    return collect(times), sttcs
end

function sweep_sttc_time(config; μ_scales=[1.0], p_scales=[1.0], 
                         window=100ms, step=100ms, total=3s)

    M = Dict()
    tvals = nothing

    for μ in μ_scales, p in p_scales
        key = (μ=μ, p=p)

        sttc = run_condition(config; 
            target=:VIP_to_SST, μ_scale=μ, p_scale=p)

        model = build_network(config)
        sim!(model, total)

        spikes = SNN.spiketimes(model.pop.CE)[1:5:end]
        spikes = filter(!isempty, spikes)

        times, sttc_vec = sttc_over_time(spikes; window, step, total)

        M[key] = sttc_vec
        tvals = times
    end

    return tvals, M
end

end
