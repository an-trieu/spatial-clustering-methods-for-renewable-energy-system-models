import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import pypsa
from sklearn.cluster import KMeans, AgglomerativeClustering
from typing import List, Optional

# Helper Function 

# Data Fetching Function 
def fetch_network(size, clustering_method, path):
    file_path = f"{path}/base_s_{size}_elec_{clustering_method}_optimized.nc"
    print(f"Loading network: {file_path}")
    return pypsa.Network(file_path)

# Save
def save_dict_to_json(data_dict, file_path):
    with open(file_path, 'w') as f:
        json.dump(data_dict, f, indent=4)
    print(f"\n Results saved to {file_path}")

# Clustering Methods

# K-means Clustering
from sklearn.cluster import KMeans
def kmeans_clustering(network, n_clusters):
    buses = network.buses
    generators = network.generators
    loads = network.loads
    loads_t = network.loads_t.p_set 

    # Extract features
    bus_coords = buses[['x', 'y']].values
    bus_order = buses.index.tolist()

    # Calculate weights
    # Define conventional carriers 
    conventional_carriers = ['nuclear', 'oil', 'OCGT', 'CCGT', 'coal', 'lignite', 'geothermal', 'biomass']

    # Filter for conventional generators
    conventional_generators = generators[generators['carrier'].isin(conventional_carriers)].copy()

    # Calculate total nominal power capacity across conventionals
    total_conv_capacity = conventional_generators.p_nom.sum()

    # Calculate conventional capacity for each bus
    conventional_gen_capacity_at_bus = conventional_generators.groupby('bus').p_nom.sum()

    # Calculate averaged electricity demand over time for each bus
    averaged_demand_at_bus = loads_t.mean(axis=0)

    # Calculate total average demand 
    total_averaged_demand = averaged_demand_at_bus.sum()

    # Calculate weights for each bus
    bus_weights = {}
    for bus_name in bus_order:
        gen_weight_component = 0.0
        demand_weight_component = 0.0

        # Component from conventional generation
        if total_conv_capacity > 0:
            bus_gen_capacity = conventional_gen_capacity_at_bus.get(bus_name, 0.0)
            gen_weight_component = bus_gen_capacity / total_conv_capacity

        # Component from demand
        if total_averaged_demand > 0:
            bus_avg_demand = averaged_demand_at_bus.get(bus_name, 0.0)
            demand_weight_component = bus_avg_demand / total_averaged_demand

        # Final weight
        bus_weights[bus_name] = gen_weight_component + demand_weight_component

    # Convert weights to array, ensuring the order matches
    weights_array = np.array([bus_weights[bus_name] for bus_name in bus_order])

    # Apply k-means clustering with weights
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Added n_init for robustness
    kmeans_labels = kmeans.fit_predict(bus_coords, sample_weight=weights_array)

    # Create Mapping
    bus_cluster_mapping = dict(zip(bus_order, kmeans_labels))

    return bus_cluster_mapping


# Ward's method
from sklearn.cluster import AgglomerativeClustering
def ward_clustering(network, n_clusters, feature_type):
    buses = network.buses
    generators = network.generators
    bus_order = buses.index.tolist()

    # Define specific carriers and features
    solar_types = ['solar', 'solar-hsat']
    wind_types = ['onwind', 'offwind-ac', 'offwind-dc', 'offwind-float']
    renewable_carriers = solar_types + wind_types
    feature_categories = ['solar', 'wind']
    features = None

    # Ward's method, capacity factor - based on average capacity factor
    if feature_type == "capacity_factor":
        gen_p_max_pu = network.generators_t.p_max_pu
        annual_cf = gen_p_max_pu.mean(axis=0)

        renewable_gens = generators[generators['carrier'].isin(renewable_carriers)].copy()
        renewable_gens['annual_cf'] = annual_cf.reindex(renewable_gens.index)

        # Map specific carriers to generic types for aggregation (e.g., offwind, onwind to wind)
        carrier_map = {carrier: 'solar' for carrier in solar_types}
        carrier_map.update({carrier: 'wind' for carrier in wind_types})
        renewable_gens['feature_category'] = renewable_gens['carrier'].map(carrier_map)

        # Calculate average capacity factor by bus and feature for solar and wind
        agg_data = renewable_gens.groupby(['bus', 'feature_category'])['annual_cf'].mean().unstack(fill_value=0)
        
        # Reindex to ensure all buses are included in the correct order and have all feature columns
        features_df = agg_data.reindex(bus_order, fill_value=0).reindex(columns=feature_categories, fill_value=0)
        features = features_df.values

    # Ward's method, time series - based on the time series
    elif feature_type == "time_series":
        snapshots = network.snapshots
        gen_p_max_pu = network.generators_t.p_max_pu
        
        # Filter generators 
        renewable_gens = generators[generators['carrier'].isin(renewable_carriers)]

        features_list = []
        
        for bus_name in bus_order:
            bus_feature_vector = []
            for category, members in {'solar': solar_types, 'wind': wind_types}.items():
                # Get renewable gens at bus
                gens_at_bus = renewable_gens[
                    (renewable_gens['bus'] == bus_name) & 
                    (renewable_gens['carrier'].isin(members))
                ]
                
                if not gens_at_bus.empty:
                    # Aggregate time series for all gens
                    aggregated_ts = gen_p_max_pu[gens_at_bus.index].mean(axis=1)
                    bus_feature_vector.append(aggregated_ts.values)
                else:
                    # If no generators, append a zero-series
                    bus_feature_vector.append(np.zeros(len(snapshots)))
            
            features_list.append(np.concatenate(bus_feature_vector))
        features = np.array(features_list)

    graph = create_weighted_graph(network)
    adjacency_matrix = nx.to_numpy_array(graph, nodelist=bus_order)

    # Apply Ward's method with feature
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=adjacency_matrix)
    ward_labels = ward.fit_predict(features)
    
    bus_cluster_mapping = dict(zip(bus_order, ward_labels))

    return bus_cluster_mapping


# Clauset-Newman-Moore greedy modularity maximization
from networkx.algorithms.community import greedy_modularity_communities
def create_weighted_graph(network):
    graph = nx.Graph()

    # Add buses as nodes 
    for bus_id in network.buses.index:
        graph.add_node(bus_id)

    # Add lines as edges, with weight attribute (admittance)
    for _, line in network.lines.iterrows():
        bus0 = line.bus0
        bus1 = line.bus1
        r = line.r
        x = line.x

        # Calculate admittance
        impedance = np.sqrt(r**2 + x**2)
        admittance = 0
        if impedance != 0:
            admittance = 1 / impedance

        graph.add_edge(
            bus0,
            bus1,
            weight=admittance,  
        )

    return graph


def modularity_maximization_clustering(network, n_clusters=None):
    # Create the weighted NetworkX graph
    graph = create_weighted_graph(network)

    # Greedy modularity maximization
    communities = nx.community.greedy_modularity_communities(graph, weight='weight', cutoff=n_clusters, best_n=n_clusters)

    # Create dictionary, mapping buses to communities
    bus_to_community = {}
    for community_id, buses in enumerate(communities):
        for bus in buses:
            bus_to_community[bus] = community_id

    return bus_to_community


# Mapping
def mapping_calulation(network, method, n_clusters):
    if method == 'kmeans':
        return kmeans_clustering(network, n_clusters)
    elif method == 'ward_cap':
        return ward_clustering(network, n_clusters, "capacity_factor")
    elif method == 'ward_time':
        return ward_clustering(network, n_clusters, "time_series")
    elif method == 'modularity':
        return modularity_maximization_clustering(network, n_clusters)



# Metric Functions 

# Modularity
def modularity(graph, clusters):
    communities = {}
    
    for node, cluster_id in clusters.items():
        # Check if current cluster_id is in communities
        if cluster_id not in communities:
            communities[cluster_id] = set()
        communities[cluster_id].add(node)
    communities_list = list(communities.values())

    return nx.community.modularity(graph, communities_list, weight='weight')


# Total System Cost
def total_system_cost(network):
    return network.objective


# Capacity Factor
def capacity_factor(
    network,
    carrier_type,
    start = None,
    end = None,
    return_distribution = False
):
    # Filter generators
    gens = network.generators.copy()
    if carrier_type:
        if isinstance(carrier_type, str):
            carrier_type = [carrier_type]
        gens = gens[gens.carrier.isin(carrier_type)].copy()
    if gens.empty:
        return pd.Series(dtype=float) if return_distribution else float("nan")

    # Filter time period
    snapshots = network.snapshots
    if start and end:
        mask = (snapshots >= pd.Timestamp(start)) & (snapshots <= pd.Timestamp(end))
        snapshots = snapshots[mask]
    if len(snapshots) == 0:
        return pd.Series(dtype=float) if return_distribution else float("nan")

    # Get actual power dispatch
    cols = network.generators_t.p.columns.intersection(gens.index)
    if len(cols) == 0:
        return pd.Series(dtype=float) if return_distribution else float("nan")
    dispatch_ts = network.generators_t.p.loc[snapshots, cols]
    if dispatch_ts.empty:
        return pd.Series(dtype=float) if return_distribution else float("nan")

    cap_col = 'p_nom_opt' if 'p_nom_opt' in gens.columns else 'p_nom'
    
    # Zero division
    capacities = gens.loc[cols, cap_col].replace(0, np.nan).astype(float)

    # Calculate average power output for each gen
    avg_dispatch = dispatch_ts.mean(axis=0)
    cf_per_gen = (avg_dispatch.div(capacities)).clip(lower=0.0, upper=1.0).dropna()

    # For quantiles
    if return_distribution:
        return cf_per_gen
    if cf_per_gen.empty:
        return float("nan")

    # Weighted average capacity factor 
    weights = capacities.loc[cf_per_gen.index]
    if weights.isnull().all() or weights.sum() == 0:
        return float("nan")
    return float(np.average(cf_per_gen.values, weights=weights.values))


# Total System Cost per Carrier - Capex and Opex
def capex_opex(network, carrier_type):
    # Filter generators of that carrier
    if carrier_type:
        if isinstance(carrier_type, str):
            carriers_to_filter = [carrier_type]
        else: 
            carriers_to_filter = list(carrier_type)

    gens = network.generators[network.generators.carrier.isin(carriers_to_filter)].copy()
    gen_ids = gens.index

    # CAPEX = capital_cost * nominal power
    capex = (gens.capital_cost * gens.p_nom_opt).sum()

    # OPEX = marginal_cost * dispatch 
    dispatch = network.generators_t.p[gen_ids]
    opex = (dispatch * gens.marginal_cost).sum().sum()  # sum over time and generators

    return capex + opex


# Full Load Hours
def flh(network, carrier):
    # Filter generators of the specified carrier 
    gens = network.generators[network.generators.carrier.isin(carrier)]
    gens = gens[gens.p_nom_opt > 0]

    if gens.empty:
        return np.nan

    # Get actual dispatch data for these generators
    dispatch = network.generators_t.p[gens.index]

    # Sum all energy produced by this group of generators
    total_energy = dispatch.sum().sum()

    # Sum all installed capacity of this group
    total_capacity = gens.p_nom_opt.sum()
    
    if total_capacity == 0:
        return np.nan

    # Calculate the weighted average FLH 
    return total_energy / total_capacity

# Optimal Capacity
def optimal_capacity(n_high, n_low, bus_map, carrier_groups):
    # Define renewable groups
    solar = {"solar", "solar-hsat"}
    wind = {"onwind", "offwind-ac", "offwind-dc", "offwind-float"}

    # This will hold our results
    comparison_data = {}

    # Loop through the requested carrier groups
    for group in carrier_groups:
        if isinstance(group, str):
            members = [group]
        else:
            members = list(group)

        # Decide the final label
        if any(c in solar for c in members):
            label = "solar"
            members = list(solar)
        elif any(c in wind for c in members):
            label = "wind"
            members = list(wind)
        else:
            label = " + ".join(members)

        # Filter high and low networks
        c_high = n_high.generators[n_high.generators.carrier.isin(members)].copy()
        c_low = n_low.generators[n_low.generators.carrier.isin(members)].copy()

        # Map clusters
        c_high["cluster"] = c_high.bus.map(bus_map).astype(str)
        c_low["cluster"] = c_low.bus.astype(str)
        c_high = c_high.dropna(subset=["cluster"])

        # Aggregate capacities
        cap_high = c_high.groupby("cluster")["p_nom_opt"].sum()
        cap_low = c_low.groupby("cluster")["p_nom_opt"].sum()

        # Build DataFrame
        df = pd.DataFrame({"low": cap_low, "high": cap_high}).dropna()
        if df.empty:
            continue

        df["low_norm"] = df["low"] / df["low"].sum()
        df["high_norm"] = df["high"] / df["high"].sum()

        # Store result
        comparison_data[label] = df[["low_norm", "high_norm"]]
        
    return comparison_data

# MAIN 

# Clustering Methods + Sizes
clustering_methods = ['kmeans', 'ward_cap', 'ward_time', 'modularity']
cluster_sizes = [47, 94, 141, 188, 236, 283, 330, 377, 424]

# Carriers
renewable_carriers = ['solar', 'solar-hsat', 'onwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'ror']
solar_carriers = ['solar', 'solar-hsat']
wind_carriers = ['onwind', 'offwind-ac', 'offwind-dc', 'offwind-float']
conventional_carriers = ['oil', 'OCGT', 'CCGT', 'coal', 'lignite', 'geothermal', 'biomass']

# JSON File
results_file = "all_metrics_results.json"

# Original and cleaned network
networks_path = "../networks"
cleaned_optimized_path = f"{networks_path}/base_s_468_elec_optimized.nc"

n_cleaned_optimized = pypsa.Network(cleaned_optimized_path)

# Graph for modularity
graph = create_weighted_graph(n_cleaned_optimized)

# Metrics
metric_functions = [
    modularity,
    total_system_cost,
    capacity_factor,
    capex_opex,
    flh,
    optimal_capacity,
]

flat_results = {}

# Ground truth for cleaned optimized network
print("\n Ground truth for cleaned optimized network")
for metric_func in metric_functions:
    metric_name = metric_func.__name__
    try:
        # Modularity
        if metric_name == "modularity":
            continue

        # Capacity Factor
        elif metric_name == "capacity_factor":
            for carrier, label in [(solar_carriers, "solar"), (wind_carriers, "wind")]:
                dist = capacity_factor(n_cleaned_optimized, carrier, return_distribution=True)
                avg = capacity_factor(n_cleaned_optimized, carrier, return_distribution=False)
                flat_results[f"{metric_name}_{label}_ground_truth"] = avg

                # Quantiles of Capacity Factor
                flat_results[f"quantile_0.7_{metric_name}_{label}_ground_truth"] = dist.quantile(0.7) if not dist.empty else None
                flat_results[f"quantile_0.8_{metric_name}_{label}_ground_truth"] = dist.quantile(0.8) if not dist.empty else None
                flat_results[f"quantile_0.9_{metric_name}_{label}_ground_truth"] = dist.quantile(0.9) if not dist.empty else None
                flat_results[f"quantile_max_{metric_name}_{label}_ground_truth"] = dist.quantile(1.0) if not dist.empty else None

        # Full Load Hours
        elif metric_name == "flh":
            flat_results[f"{metric_name}_solar_ground_truth"] = flh(n_cleaned_optimized, solar_carriers)
            flat_results[f"{metric_name}_wind_ground_truth"] = flh(n_cleaned_optimized, wind_carriers)

        # Total System Cost per Carrier - Capex and Opex
        elif metric_name == "capex_opex":
            for carrier in renewable_carriers + conventional_carriers:
                flat_results[f"{metric_name}_{carrier}_ground_truth"] = capex_opex(n_cleaned_optimized, carrier)

        # Total System Cost
        elif metric_name == "total_system_cost":
            flat_results[f"{metric_name}_ground_truth"] = total_system_cost(n_cleaned_optimized)
        print(f"{metric_name} (ground_truth) computed")
    except Exception as e:
        print(f"Error computing ground_truth {metric_name}: {e}")


for method in clustering_methods:
    for size in cluster_sizes:
        print(f"\n Method: {method}, Clusters: {size}")
        try:
            n_solved = fetch_network(size, method, networks_path)
        except Exception as e:
            print(f"Failed to load network: {e}")
            continue

        mapping = mapping_calulation(n_cleaned_optimized, method, size)

        for metric_func in metric_functions:
            metric_name = metric_func.__name__

            try:
                # Modularity
                if metric_name == "modularity":
                    result = modularity(graph, mapping)
                    key = f"{metric_name}_{method}"
                    flat_results.setdefault(key, {})[size] = result

                # Total System Cost
                elif metric_name == "total_system_cost":
                    result = total_system_cost(n_solved)
                    key = f"{metric_name}_{method}"
                    flat_results.setdefault(key, {})[size] = result

                # Capacity Factor
                elif metric_name == "capacity_factor":
                    for carrier, label in [(solar_carriers, "solar"), (wind_carriers, "wind")]:
                        dist = capacity_factor(n_solved, carrier, return_distribution=True)
                        avg = capacity_factor(n_solved, carrier, return_distribution=False)
                        flat_results.setdefault(f"{metric_name}_{label}_{method}", {})[size] = avg

                        # Quantiles of Capacity Factor
                        flat_results.setdefault(f"quantile_0.7_{metric_name}_{label}_{method}", {})[size] = dist.quantile(0.7) if not dist.empty else None
                        flat_results.setdefault(f"quantile_0.8_{metric_name}_{label}_{method}", {})[size] = dist.quantile(0.8) if not dist.empty else None
                        flat_results.setdefault(f"quantile_0.9_{metric_name}_{label}_{method}", {})[size] = dist.quantile(0.9) if not dist.empty else None
                        flat_results.setdefault(f"quantile_max_{metric_name}_{label}_{method}", {})[size] = dist.quantile(1.0) if not dist.empty else None
                # Total System Cost per Carrier - Capex and Opex
                elif metric_name == "capex_opex":
                    for carrier in renewable_carriers + conventional_carriers:
                        val = capex_opex(n_solved, carrier)
                        key = f"{metric_name}_{carrier}_{method}"
                        flat_results.setdefault(key, {})[size] = val

                # Full Load Hours
                elif metric_name == "flh":
                    flh_results = {
                        "solar": flh(n_solved, solar_carriers),
                        "wind": flh(n_solved, wind_carriers)
                    }
                    for carrier, val in flh_results.items():
                        key = f"{metric_name}_{carrier}_{method}"
                        flat_results.setdefault(key, {})[size] = val

                # Optimal Capacity
                elif metric_name == "optimal_capacity":
                    oc_results = optimal_capacity(
                        n_cleaned_optimized, n_solved, mapping,
                        [solar_carriers, wind_carriers]
                    )
                
                    for group_label, df in oc_results.items():
                        # Clean key for JSON
                        cleaned_label = group_label.replace(" ", "_").replace("-", "_")
                        key = f"{metric_name}_{cleaned_label}_{method}"
                
                        # Convert df to nested dict
                        df_dict = df.to_dict(orient="index")
                        flat_results.setdefault(key, {})[size] = df_dict   

                print(f"{metric_name} computed")

            except Exception as e:
                print(f"Error computing {metric_name} for {method}-{size}: {e}")
                
# Sort all size entries
for k, v in flat_results.items():
    if isinstance(v, dict):
        flat_results[k] = dict(sorted(v.items()))

# Save as JSON
save_dict_to_json(flat_results, results_file)
