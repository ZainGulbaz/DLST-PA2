# Question 1 — River Forecasting Proposal

## 1. Architecture diagram

![Diagram Q1. Every historical input is joined as of forecast issuance. Node and feature axes are explicit; gauges are projected to catchments without implying they are river stations. Future schedules use a separate encoder.](figures/Q1_architecture.png)

## 2. Assumptions

A1. At least two complete hydrological years of ten-minute records are available, including some floods. This is an assumption about data volume, not a supplied fact. With less data, shrink the network and report wider uncertainty; flood performance cannot be promised.

A2. Measurement time, arrival time and schedule announcement time can be recovered. A reading is usable only when it arrived by issuance. If historical arrival times are absent, impose a conservative fixed delay and state that backtest realism is limited.

A3. Units, station datums and clock conventions can be reconciled. Standardise each continuous feature using training-period statistics; datum changes require explicit correction, since an apparent jump is not river dynamics.

A4. A 12-hour history (72 bins) captures the main dependencies relevant to a one-hour forecast. This is a tunable operational assumption, validated chronologically; extend it if slow propagation exceeds 12 hours.

A5. Dominant transport follows the supplied downstream river topology, with no strong reverse backwater effect over this horizon. Thus upstream-to-downstream edges are justified. If tides/backwater matter, add documented reverse edges and boundary levels; the present graph would otherwise be misspecified.

A6. Travel times vary but relevant delays can be represented within the 72-bin lag bank. Do not equate station index with distance or impose a single unknown travel speed. Distances and recent levels condition learned lag weights.

A7. Rain gauges provide imperfect catchment proxies; they do not measure future rainfall. Soft catchment weights use gauge locations and learned corrections. Unmonitored inflow remains partly unobservable, especially for S3 and S4.

A8. Future schedules available at issuance have predictive value but are not guarantees of actual releases. A known/unknown flag and lead-time features distinguish genuine zero releases from absent schedules.

A9. Reporting latency is usually below one ten-minute interval and short outages occur. Masks and ages are mandatory; severe station outages activate a conservative persistence fallback and an operator flag. A continuously staffed flood operation and a historical alert threshold register are assumed for deployment.

Supplied facts: levels at five stations, two located rain gauges, past releases and pre-announced schedules, station locations/connectivity/distances, ten-minute issuance, and one-hour predictions at S1–S4. No dataset, precise travel times, unmonitored tributary inflow or future rainfall is supplied.

## 3. Forecasting task and data specification

Let t be the wall-clock issuance on the ten-minute grid. Use bins ending t-710,...,t minutes (H=72); missing latest values remain missing. Predict the instantaneous/station-reported level y_i(t+60), i in {S1,S2,S3,S4}, not an average over the next hour. B is the batch size; historical lag, node, channel and future-lead axes are distinct.

## Input/output specification

| Quantity | Shape / axes | Time and availability |
|---|---|---|
| Water levels | B x 72 x 5 x 3; [value,mask,age] | S1–S4,T1; measurement <= t and arrival <= t |
| Rainfall | B x 72 x 2 x 3; [depth,mask,age] | R1,R2; causal ten-minute accumulation; arrival <= t |
| Actual reservoir release | B x 72 x 1 x 3 | Past effective bins, recorded/arrived by t |
| Future release schedule | B x 6 x 1 x 3; [release,known,lead] | Effective t+10,...,t+60; latest announcement <= t per effective bin |
| Static graph/geography | 6 nodes, E=17 directed/self edges; node coordinates and catchment links | 4 main-river edges, T1->S2, 6 self edges, 6 main-river ancestor shortcuts |
| Node embeddings | B x 72 x 6 x 64 | Causal projection of aligned inputs and static features |
| Target / output | B x 4, station axis S1–S4 | Target level at t+60; observed later, never an input |

Use a canonical time zone and event-time bins. Aggregate rain depth by summation, release by interval mean and level by the latest valid reading within a bin; never interpolate from a later reading. At every historical issuance rebuild the as-of view, including superseded schedule versions. A gap may carry forward the last value only with mask=0 and increasing age. Fit normalisation and imputation constants on training data only. Split chronologically, keep events intact, and purge at least history plus horizon at split boundaries.

## 4. Model computations and design justification

Let v denote one of six graph nodes, i a forecast station, s an observed bin, r a lag and a_uv the fixed downstream/self adjacency. z_v,s = W_v [aligned levels, gauge-weighted rain, actual release, masks, ages, static features] + b_v in R^64. Zero-fill missing normalised values only after adding mask and age; nonlocal station levels are not silently copied into every node. Rainfall at a node is sum_j w_vj rain_j, with nonnegative catchment weights informed by locations and learned from historical data.

Lag-aware block: e_uvrs = phi(z_v,s, z_u,s-r, river_distance_uv, r); alpha_uvrs = softmax over allowed upstream u and causal lags r in 0..71; m_v,s = sum_(u,r) alpha_uvrs W_msg z_u,s-r. Invalid or unavailable lag positions are masked. h_v,s = GRU([z_v,s,m_v,s], h_v,s-1). Repeat three graph/temporal blocks with residual projections. The fixed 17-edge graph contains five direct flow edges, six self edges and six main-river ancestor shortcuts (reservoir->S2,S3,S4; S1->S3,S4; S2->S4). Ancestor edges encode hydrological reachability, not instantaneous transport; learned lags still determine timing. They prevent a four-edge reservoir-to-S4 path from exceeding the three-block spatial reach. T1 can reach S4 through its three direct downstream edges.

The schedule GRU encodes six issued future release entries into q_t in R^64; missing schedule bins retain a known flag. Select h_i,t for S1–S4 and concatenate q_t, producing B x 4 x 128. A station-conditioned head gives delta_i = MLP_i([h_i,t,q_t]); yhat_i(t+60) = last_valid_level_i(t) + delta_i. If there is no valid level in the history, use a training-period datum fallback and flag it. Shared features improve data efficiency, while station heads allow different rating/response relationships. The delta skip reduces the burden of learning near-persistence.

Train with L = (1/(4B)) sum_(b,i) |yhat_bi-y_bi| on available labels, normalising per video/issuance by the number of valid stations. If all four labels are present this is ordinary MAE; batches without a valid station label are excluded. Report level MAE/RMSE by station, lead (here fixed at one hour), flood/nonflood strata and event. Use chronological validation to tune width/history and compare against last level and a linear autoregression. Do not infer hydraulic conservation from a learned graph: graph direction/catchment assignment are physical assumptions, but delay weights, nonlinear stage responses and rain-to-inflow conversion are empirical mappings.

Upstream levels propagate through directed lag paths; T1 contributes downstream of its junction at S2, and then S3/S4. Rain can affect all corresponding catchments; the missing tributary is represented only through indirect level/rain evidence. Reservoir history captures realised forcing, while the separate schedule context provides planned forcing that a purely historical model would miss. No future rainfall or future measured level appears in any equation.

## 5. Operation and exceptional conditions

(i) Missing observations: retain value/mask/age channels, train with realistic block outages, and avoid treating absent rain or release as zero. Forecast from other nodes when possible. If S1–S4 are all stale beyond a validated outage threshold, issue a flagged persistence forecast with expanded empirical error bands rather than an unqualified network value.

(ii) Delayed observations: use arrival-time filtering; an observation stamped before t but arriving after t is unavailable for this forecast. Incorporate it at the next issuance and replay causal history if needed. Backtests must mimic this rule, including the same delayed-data policy.

(iii) Announced future releases: select only schedule versions announced by t. Map their effective periods to the six future bins, using duration-weighted rates for partial bins and preserving unknown flags. A release announced after t is excluded even if effective within the forecast horizon.

(iv) Scheduled versus actual releases: retain separate fields and never overwrite a historical schedule with the realised future release. At the next issuance update actual-release history; learn residual schedule error statistics by lead and operator regime. Simulate plausible deviations for stress scenarios, label them as scenarios and widen conditional residual bands. Schedule forecasts can fail when operations deviate abruptly.

Every ten minutes, fetch the as-of snapshot, validate, update encoders, score four stations and log both input version IDs and forecasts. Six nodes, 72 bins and width 64 give a small bounded graph; even all-node causal lag mixing is at most O(B H^2 E d) and the chosen sparse graph reduces constants. This is plausibly feasible within ten minutes, but no latency benchmark is claimed. Require a deployment latency/load test, monitoring, an input deadline and fallback if ingestion or inference misses the deadline.

## 6. Uncertainty, limitations, and additional information

## Limitations and requested information

| Issue / request | Forecast impact and response | Usable without? |
|---|---|---|
| Unmonitored tributary; request a gauge or inflow estimate | Unseen surges can bias S3/S4 low. Use indirect rain/level evidence and flag wet events; irreducible uncertainty remains. | Yes, with explicit blind spot |
| Changing travel times; request flow, channel cross-sections, roughness | Lag attention can adapt to level regime but may extrapolate poorly in extreme floods. Validate eventwise; distances alone do not identify delays. | Yes; lag bank remains empirical |
| Rainfall coverage; request radar, catchment boundaries, forecast rain | Two gauges miss spatial storms; no available-at-issuance future rain means flood growth can be missed. Add forecasts only with archived issue times. | Yes, degraded storm lead |
| Operational deviations; request schedule revisions and actual release logs | Announcement/effective/arrival histories prevent leakage and quantify schedule reliability; emergency changes remain unpredictable. | Yes; unknown schedules masked |
| Quality and latency; request datum changes, maintenance, arrival logs | Spikes and stale readings distort levels and delay inference. QC, outage augmentation, masks and fallback help. | Conditional; reliable timestamps/units essential |
| Rare extremes and regime shifts; request long records and event labels | Global MAE can hide peak failure. Event-stratified assessment and drift monitoring are required. | Yes for monitored pilot; extreme-flood claims unvalidated |
| Backwater/tides; request downstream controls, boundary levels | Reverse effects violate A5; add edges and forcings if documented. | Conditional on A5 validity |
| Uncertainty calibration; request operational alert thresholds | Use rolling held-out signed residual quantiles per station/regime around point forecast for empirical 80/95% intervals. Coverage under shift is not guaranteed; validate peaks and recalibrate. | Yes; report levels and bands, avoid uncalibrated probabilities |

This is a design proposal, as requested: no river data, river training result, guaranteed flood probability or measured deployment latency is claimed. Empirical residual intervals describe observed errors and do not resolve unseen tributary forcing or guarantee coverage during unprecedented events.
