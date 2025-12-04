module NetworkUtils

using SpikingNeuralNetworks
using UnPack
using Random
using Statistics

SNN.@load_units

import SpikingNeuralNetworks: IF, PoissonLayer, Stimulus, SpikingSynapse, compose, monitor!, sim!, firing_rate, @update, SingleExpSynapse, IFParameter, Population, PostSpike, AdExParameter, STTC

export scaled_connection, build_network, run_condition, sweep_parameters, sweep_TC_feedback, sweep_sttc_time, sttc_over_time

# -------------------------------------------------------------------
# Scale a connection tuple (probability & weight)
# -------------------------------------------------------------------
function scaled_connection(conn; μ_scale=1.0, p_scale=1.0)
    return (
        p = clamp(conn.p * p_scale, 0, 1.0),
        μ = conn.μ * μ_scale,
        rule = conn.rule
    )
end

# -------------------------------------------------------------------
# Build the network from a configuration NamedTuple
# -------------------------------------------------------------------
function build_network(config)
    @unpack seed = config
    Random.seed!(seed)

    @unpack afferents_to_ThalExc, afferents_to_CortExc, afferents_to_CortPv,
        afferents_to_CortSst, afferents_to_CortVip, connections, Npop,
        spike, spike_PV, spike_SST, spike_VIP,
        exc, inh_PV, inh_SST, inh_VIP,
        synapse, synapse_PV, synapse_SST, synapse_VIP = config

    # Populations
    TE  = Population(exc;        synapse,       spike,       N=Npop.ThalExc,     name="ThalExc")
    CE  = Population(exc;        synapse,       spike,       N=Npop.CortExc,     name="CortExc")
    PV  = Population(inh_PV;     synapse=synapse_PV,  spike=spike_PV,  N=Npop.CortPvInh, name="CortPvInh")
    SST = Population(inh_SST;    synapse=synapse_SST, spike=spike_SST, N=Npop.CortSstInh,name="CortSstInh")
    VIP = Population(inh_VIP;    synapse=synapse_VIP, spike=spike_VIP, N=Npop.CortVipInh,name="CortVipInh")

    # External inputs
    afferentTE  = Stimulus(afferents_to_ThalExc.layer, TE,  :glu, conn=afferents_to_ThalExc.conn)
    afferentCE  = Stimulus(afferents_to_CortExc.layer, CE,  :glu, conn=afferents_to_CortExc.conn)
    afferentPV  = Stimulus(afferents_to_CortPv.layer,  PV,  :glu, conn=afferents_to_CortPv.conn)
    afferentSST = Stimulus(afferents_to_CortSst.layer, SST, :glu, conn=afferents_to_CortSst.conn)
    afferentVIP = Stimulus(afferents_to_CortVip.layer, VIP, :glu, conn=afferents_to_CortVip.conn)

    # Recurrent synapses
    syns = (
        # Thalamus
        TE_to_CE = SpikingSynapse(TE, CE, :glu,  conn=connections.ThalExc_to_CortExc),
        TE_to_PV = SpikingSynapse(TE, PV, :glu,  conn=connections.ThalExc_to_CortPv),

        # Excitatory
        CE_to_CE = SpikingSynapse(CE, CE, :glu, conn=connections.CortExc_to_CortExc),
        CE_to_PV = SpikingSynapse(CE, PV, :glu, conn=connections.CortExc_to_CortPv),
        CE_to_TE = SpikingSynapse(CE, TE, :glu, conn=connections.CortExc_to_ThalExc),

        # PV interneurons
        PV_to_CE  = SpikingSynapse(PV, CE,  :gaba, conn=connections.CortPv_to_CortExc),
        PV_to_PV  = SpikingSynapse(PV, PV,  :gaba, conn=connections.CortPv_to_CortPv),
        PV_to_SST = SpikingSynapse(PV, SST, :gaba, conn=connections.CortPv_to_CortSst),

        # SST interneurons
        SST_to_CE  = SpikingSynapse(SST, CE,  :gaba, conn=connections.CortSst_to_CortExc),
        SST_to_PV  = SpikingSynapse(SST, PV,  :gaba, conn=connections.CortSst_to_CortPv),
        SST_to_VIP = SpikingSynapse(SST, VIP, :gaba, conn=connections.CortSst_to_CortVip),

        # VIP interneurons
        VIP_to_SST = SpikingSynapse(VIP, SST, :gaba, conn=connections.CortVip_to_CortSst),
    )

    # Assemble model
    model = compose(; TE, CE, PV, SST, VIP,
        afferentTE, afferentCE, afferentPV, afferentSST, afferentVIP,
        syns..., name="TC3_inhib_network")

    # Monitors
    monitor!(model.pop, [:fire])

    return model
end

# -------------------------------------------------------------------
# Run one condition: modify one connection and measure STTC
# -------------------------------------------------------------------
function run_condition(config; target, μ_scale=1.0, p_scale=1.0)
    base_conn = config.connections[target]
    newconn = scaled_connection(base_conn; μ_scale, p_scale)

    newconfig = @update config begin
        connections = merge(config.connections, (target => newconn,))
    end

    model = build_network(newconfig)
    sim!(model, 3s)

    spikes = SNN.spiketimes(model.pop.CE)[1:5:end]  # only CortExc
    spikes = filter(!isempty, spikes)
    return mean(STTC(spikes, 50ms))
end


# -------------------------------------------------------------------
# Average firing rate helper
# -------------------------------------------------------------------
function average_firing_rate(model; pops=[:CE])
    spks_all = []
    for p in pops
        spks = SNN.spiketimes(getfield(model.pop, p))  # sélectionne la population
        append!(spks_all, filter(!isempty, spks))
    end
    return mean(firing_rate(spks_all))
end


# -------------------------------------------------------------------
# Parameter sweep over μ and p for CortExc -> ThalExc feedback
# -------------------------------------------------------------------
function sweep_TC_feedback(config; μ_scales, p_scales)
    results = Dict{NamedTuple, Dict}()

    for μ in μ_scales, p in p_scales
        key = (μ=μ, p=p)
        sttc = run_condition(config;
            target=:CortExc_to_ThalExc,
            μ_scale=μ,
            p_scale=p
        )

        # Build network to measure firing rate separately
        mod = build_network(config)
        sim!(mod, 3s)
        rate = average_firing_rate(mod)

        results[key] = Dict(:STTC => sttc, :firing_rate => rate)
    end

    return results
end



# -------------------------------------------------------------------
# Compute STTC over time
# -------------------------------------------------------------------
function sttc_over_time(spikes; window=100ms, step=100ms, total=3s)
    times = 0:step:(total - window)
    sttcs = Float64[]

    for t in times
        # extract spikes in [t, t+window]
        local_spikes = [
            filter(s -> t <= s <= t + window, neuron_spikes)
            for neuron_spikes in spikes
        ]
        local_spikes = filter(!isempty, local_spikes)

        if !isempty(local_spikes)
            push!(sttcs, mean(STTC(local_spikes, 50ms)))
        else
            push!(sttcs, NaN)  # or 0.0 if you prefer
        end
    end
    
    return collect(times), sttcs
end


function sweep_sttc_time(config; μ_scales=[1.0], p_scales=[1.0], 
                         window=100ms, step=100ms, total=3s)

    M = Dict()   # maps (μ,p) → STTC vector
    tvals = nothing

    for μ in μ_scales, p in p_scales
        key = (μ=μ, p=p)

        sttc = run_condition(config; 
            target=:CortVip_to_CortSst, μ_scale=μ, p_scale=p)

        # Build model AGAIN to extract full spiketimes
        model = build_network(config)
        sim!(model, total)

        spikes = SNN.spiketimes(model.pop.CE)[1:5:end]
        spikes = filter(!isempty, spikes)

        # sliding window STTC
        times, sttc_vec = sttc_over_time(spikes; window, step, total)

        M[key] = sttc_vec
        tvals = times
    end

    return tvals, M
end

end # module
