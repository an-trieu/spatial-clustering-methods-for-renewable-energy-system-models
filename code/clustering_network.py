import pypsa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

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


# Preprocessing
def preprocessing_network(n):

    # Identify buses where substation_lv == 0
    buses_to_drop_index = n.buses[n.buses.substation_lv == 0].index
    
    # Drop the identified buses from the network
    n.buses = n.buses.drop(buses_to_drop_index)
    existing_buses = n.buses.index

    # Clean up components
    # Define components connected to a single bus
    nodel_components = [
        ('generators', 'Generators', 'bus'),
        ('loads', 'Loads', 'bus'),
        ('storage_units', 'StorageUnits', 'bus'),
        ('stores', 'Stores', 'bus'),
        ('shunt_impedances', 'ShuntImpedances', 'bus'),
    ]

    # Loop through components and clean them
    for attr_name, comp_name, bus_attr in nodel_components:
        # Get df for current component type
        comp_df = getattr(n, attr_name)

        initial_count = len(comp_df)
        # Keep components where bus attribute is NOT NaN
        # AND 
        # bus name exists in existing_buses
        cleaned_df = comp_df[comp_df[bus_attr].notna() & comp_df[bus_attr].isin(existing_buses)].copy()
        # Update df in the network object
        setattr(n, attr_name, cleaned_df)

    # Define components connected bus0 and bus1 attributes
    branch_components_bus01 = [
        ('lines', 'Lines', 'bus0', 'bus1'),
        ('transformers', 'Transformers', 'bus0', 'bus1'),
        ('links', 'Links', 'bus0', 'bus1'),
    ]

    # Loop through components and clean them
    for attr_name, comp_name, bus0_attr, bus1_attr in branch_components_bus01:
        # Get df for the current component type
        comp_df = getattr(n, attr_name)

        initial_count = len(comp_df)
        # Keep when:
        # bus0 is NOT NaN AND bus0 is in existing_buses
        # AND
        # bus1 is NOT NaN AND bus1 is in existing_buses
        cleaned_df = comp_df[
            comp_df[bus0_attr].notna() & comp_df[bus0_attr].isin(existing_buses) &
            comp_df[bus1_attr].notna() & comp_df[bus1_attr].isin(existing_buses)
        ].copy()
        # Update df
        setattr(n, attr_name, cleaned_df)

    # Remove dc lines
    # Identify lines where attribute is NaN
    lines_with_dc_nan = n.lines[n.lines['dc'].isna()]
    
    # Get index of lines
    lines_to_drop_index = lines_with_dc_nan.index
    
    # Drop lines from df
    initial_line_count = len(n.lines)
    n.lines = n.lines.drop(lines_to_drop_index)
    
    # Clean up time-series df
    # Dictionary maps static component attribute name to df
    components_with_timeseries = {
        'generators': getattr(n, 'generators'),
        'loads': getattr(n, 'loads'),
        'storage_units': getattr(n, 'storage_units'),
        'stores': getattr(n, 'stores'),
        'links': getattr(n, 'links'),
    }

    # Clean up time-series data
    for comp_attr_name, static_df in components_with_timeseries.items():
         # Get time-series dictionary from component
         timeseries_dict = getattr(n, f"{comp_attr_name}_t")

         # Get index of components that were kept in the static cleanup
         existing_component_index = static_df.index

         # Iterate through each time-series df within the dictionary
         for ts_attr_name, timeseries_df in list(timeseries_dict.items()):
             initial_cols = len(timeseries_df.columns)
             # Find intersection of time-series columns and existing static component index
             cols_to_keep = timeseries_df.columns.intersection(existing_component_index)

             # Select only these columns from the time-series df
             cleaned_timeseries_df = timeseries_df[cols_to_keep]

             # Update time-series df within the dictionary
             timeseries_dict[ts_attr_name] = cleaned_timeseries_df

             # Report how many components were removed from this time-series attribute
             removed_cols = initial_cols - len(cleaned_timeseries_df.columns)

    # Consistency check 
    try:
        n.consistency_check()
        print("Consistency check passed.")
    except Exception as e:
        print(f"Consistency check failed: {e}")

    return n

# Create a busmap and cluster network
def cluster_by_busmap(network, method, n_clusters):
    if method == 'kmeans':
        res = kmeans_clustering(network, n_clusters)
    elif method == 'ward':
        res = ward_clustering(network, n_clusters)
    elif method == 'ward_cap':
        res = ward_clustering(network, n_clusters, "capacity_factor")
    elif method == 'ward_time':
        res = ward_clustering(network, n_clusters, "time_series")
    elif method == 'modularity':
        res = modularity_maximization_clustering(network, n_clusters)

    # Create busmap 
    busmap = network.cluster.get_clustering_from_busmap(res)
    
    # Extract network
    C = busmap.network
    
    return C

# Runs all clustering methods and return them as dict
def cluster_network(network, n_clusters):
    # Run all clustering methods
    kmeans_cluster = cluster_by_busmap(network, 'kmeans', n_clusters)
    ward_cap_cluster = cluster_by_busmap(network, 'ward_cap', n_clusters)
    ward_time_cluster = cluster_by_busmap(network, 'ward_time', n_clusters)
    modularity_cluster = cluster_by_busmap(network, 'modularity', n_clusters)

    # Return dict with all results
    return {
        'kmeans': kmeans_cluster,
        'ward_cap': ward_cap_cluster,
        'ward_time': ward_time_cluster,
        'modularity': modularity_cluster
    }


def main():
    # Network
    file_path = "../networks/base_s_471_elec_v7_96.nc"
    network = pypsa.Network(file_path)

    # Only AC lines
    n = network[network.buses.carrier == 'AC']

    # Clean network
    n = preprocessing_network(n)

    # Optimize cleaned network, 468 buses
    n_cleaned = n.copy()
    n_cleaned.optimize(solver_name="gurobi", solver_options={"Method":2,"Crossover":0})
    n_cleaned.export_to_netcdf("../networks/base_s_468_elec_optimized.nc")

    # [424,377,330,283,236,188,141,94,47]
    n_clusters = [424,377,330,283,236,188,141,94,47]
    
    # Apply all clustering methods with size and optimize
    for size in n_clusters:
        res = cluster_network(n, size)
        res['kmeans'].optimize(solver_name="gurobi", solver_options={"Method":2,"Crossover":0})
        res['kmeans'].export_to_netcdf(f"../networks/base_s_{size}_elec_kmeans_optimized.nc")
        res['ward_cap'].optimize(solver_name="gurobi", solver_options={"Method":2,"Crossover":0})
        res['ward_cap'].export_to_netcdf(f"../networks/base_s_{size}_elec_ward_cap_optimized.nc")
        res['ward_time'].optimize(solver_name="gurobi", solver_options={"Method":2,"Crossover":0})
        res['ward_time'].export_to_netcdf(f"../networks/base_s_{size}_elec_ward_time_optimized.nc")
        res['modularity'].optimize(solver_name="gurobi", solver_options={"Method":2,"Crossover":0})
        res['modularity'].export_to_netcdf(f"../networks/base_s_{size}_elec_modularity_optimized.nc")

# Run script
if __name__ == "__main__":
    main()
