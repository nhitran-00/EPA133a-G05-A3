from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import ContinuousSpace
from components import Source, Sink, SourceSink, Bridge, Link, Intersection, Vehicle


# ---------------------------------------------------------------
def set_lat_lon_bound(lat_min, lat_max, lon_min, lon_max, edge_ratio=0.02):
    """
    Set the HTML continuous space canvas bounding box (for visualization)
    give the min and max latitudes and Longitudes in Decimal Degrees (DD)

    Add white borders at edges (default 2%) of the bounding box
    """

    lat_edge = (lat_max - lat_min) * edge_ratio
    lon_edge = (lon_max - lon_min) * edge_ratio

    x_max = lon_max + lon_edge
    y_max = lat_min - lat_edge
    x_min = lon_min - lon_edge
    y_min = lat_max + lat_edge
    return y_min, y_max, x_min, x_max


# ---------------------------------------------------------------
class BangladeshModel(Model):
    """
    The main (top-level) simulation model

    One tick represents one minute; this can be changed
    but the distance calculation need to be adapted accordingly

    Class Attributes:
    -----------------
    step_time: int
        step_time = 1 # 1 step is 1 min

    path_ids_dict: defaultdict
        Key: (origin, destination)
        Value: the shortest path (Infra component IDs) from an origin to a destination

        Only straight paths in the Demo are added into the dict;
        when there is a more complex network layout, the paths need to be managed differently

    sources: list
        all sources in the network

    sinks: list
        all sinks in the network

    """


    step_time = 1

    file_name = '../data/cleaned_data.csv'
    #file_name = '../data/demo-4.csv' # debugging

    def __init__(self, seed=None, x_max=500, y_max=500, x_min=0, y_min=0, route_strategy="straight",
                 cat_a_failure=0, cat_b_failure=0, cat_c_failure=0,
                 cat_d_failure=0):

        self.schedule = BaseScheduler(self)
        self.running = True
        self.path_ids_dict = defaultdict(lambda: pd.Series())
        self.space = None
        self.sources = []
        self.sinks = []
        self.removed_vehicles = []
        self.bridge_breakdown = defaultdict(list)
        self.scenario = {'A': cat_a_failure, 'B': cat_b_failure, 'C': cat_c_failure,
                         'D': cat_d_failure}  # Making a dictionary of the given failure rates
        self.graph = nx.Graph()
        # route strategy
        self.route_strategy = route_strategy
        self.route_strategy_dict = {
            'straight': self.get_straight_route,
            'random': self.get_random_route
        }
        self.generate_model()


    def generate_model(self):
        """
        generate the simulation model according to the csv file component information

        Warning: the labels are the same as the csv column labels
        """

        df = pd.read_csv(self.file_name)

        # a list of names of roads to be generated
        # TODO You can also read in the road column to generate this list automatically
        # roads = ['N1', 'N2']
        roads = df['road'].unique()

        df_objects_all = []
        for road in roads:
            # Select all the objects on a particular road in the original order as in the cvs
            df_objects_on_road = df[df['road'] == road]

            if not df_objects_on_road.empty:
                df_objects_all.append(df_objects_on_road)

                """
                Set the path 
                1. get the serie of object IDs on a given road in the cvs in the original order
                2. add the (straight) path to the path_ids_dict
                3. put the path in reversed order and reindex
                4. add the path to the path_ids_dict so that the vehicles can drive backwards too
                """
                path_ids = df_objects_on_road['id']
                path_ids.reset_index(inplace=True, drop=True)
                self.path_ids_dict[path_ids[0], path_ids.iloc[-1]] = path_ids
                self.path_ids_dict[path_ids[0], None] = path_ids
                path_ids = path_ids[::-1]
                path_ids.reset_index(inplace=True, drop=True)
                self.path_ids_dict[path_ids[0], path_ids.iloc[-1]] = path_ids
                self.path_ids_dict[path_ids[0], None] = path_ids

        # put back to df with selected roads so that min and max and be easily calculated
        df = pd.concat(df_objects_all)
        y_min, y_max, x_min, x_max = set_lat_lon_bound(
            df['lat'].min(),
            df['lat'].max(),
            df['lon'].min(),
            df['lon'].max(),
            0.05
        )

        # generate networkx graph
        self.generate_network_from_dataframe(df, edge_attr="length")

        # ContinuousSpace from the Mesa package;
        # not to be confused with the SimpleContinuousModule visualization
        self.space = ContinuousSpace(x_max, y_max, True, x_min, y_min)

        for df in df_objects_all:
            for _, row in df.iterrows():  # index, row in ...

                # create agents according to model_type
                model_type = row['model_type'].strip()
                agent = None

                name = row['name']
                if pd.isna(name):
                    name = ""
                else:
                    name = name.strip()

                if model_type == 'source':
                    agent = Source(row['id'], self, row['length'], name, row['road'])
                    self.sources.append(agent.unique_id)
                elif model_type == 'sink':
                    agent = Sink(row['id'], self, row['length'], name, row['road'])
                    self.sinks.append(agent.unique_id)
                elif model_type == 'sourcesink':
                    agent = SourceSink(row['id'], self, row['length'], name, row['road'])
                    self.sources.append(agent.unique_id)
                    self.sinks.append(agent.unique_id)
                elif model_type == 'bridge':
                    agent = Bridge(row['id'], self, row['length'], name, row['road'], row['condition'])
                elif model_type == 'link':
                    agent = Link(row['id'], self, row['length'], name, row['road'])
                elif model_type == 'intersection':
                    if not row['id'] in self.schedule._agents:
                        agent = Intersection(row['id'], self, row['length'], name, row['road'])

                if agent:
                    self.schedule.add(agent)
                    y = row['lat']
                    x = row['lon']
                    self.space.place_agent(agent, (x, y))
                    agent.pos = (x, y)

    def generate_network_from_dataframe(self, df, source='id', edge_attr=None):
        """
        Set the attribute self.graph from a pandas dataframe
        _________
        Parameters:
            df (pandas.DataFrame): the dataframe containing the network nodes and their attributes. The dataframe must have the following columns: 'road', 'id', and 'length'
            source (string): the column name of the source node
            edge_attr (str, list(string)): the column names containing the edge attributes
        _________
        Return: graph (nx.Graph): a network with the added edges from the pandas dataframe
        """
        df_edgelist = df.copy()
        df_edgelist["target"] = df_edgelist[source].shift(-1)

        # remove incorrect edges (i.e 2 consecutive Sourcesinks of 2 different roads)
        row_drop_indices = []
        roads = df_edgelist['road'].unique()
        for road in roads:
            row_drop_indices.append(df_edgelist[df_edgelist['road']==road].index[-1])
        df_edgelist = df_edgelist.drop(index=row_drop_indices)

        # generate graph
        self.graph = nx.from_pandas_edgelist(df_edgelist, source=source, target="target", edge_attr=edge_attr)

        # add ["id", "road", "model_type"]
        node_attributes = ["road", "model_type"]
        for node_attribute in node_attributes:
            buf_series = pd.Series(data=df[node_attribute].values, index=df['id'])
            nx.set_node_attributes(self.graph, values=buf_series, name=node_attribute)

        # # add id as node attributes
        # id_series = pd.Series(data=df['id'].values, index=df['id'])
        # nx.set_node_attributes(self.graph, values=id_series, name="id")
        # # add road_name as node attributes
        # road_name_series = pd.Series(data=df['road'].values, index=df['id'])
        # nx.set_node_attributes(self.graph, values=road_name_series, name="road")
        # # add model_type as node attributes
        # model_type_series = pd.Series(data=df['model_type'].values, index=df['id'])
        # nx.set_node_attributes(self.graph, values=model_type_series, name="model_type")

    def find_route(self, source, sink):
        """
        Finds a route between the source and the sink, and update the self.path_ids_dict with the route. Returns route.
        :param source: the unique id of the source node
        :type source: int
        :param sink: the unique id of the sink node
        :type sink: int
        :return: list: the ids of the path components
        """
        if not (source, sink) in self.path_ids_dict.keys():
            if nx.has_path(self.graph, source, sink):
                self.path_ids_dict[source, sink] = pd.Series(nx.shortest_path(self.graph,
                                                                              source, sink, weight='length'))
            else:
                sink = None
        return self.path_ids_dict[source, sink]

    def get_random_route(self, source):
        """
        pick up a random route given an origin. If there is no path between source and random sink, pick the self.path_ids_dict[source, None] route
        :param source: the unique id of the source node
        :type source: int
        :return: list: the ids of the path components
        """
        while True:
            # different source and sink
            sink = self.random.choice(self.sinks)
            if sink is not source:
                break
        # if route doesn't exist yet, find route and update self.path_ids_dict
        return self.find_route(source, sink)


    def get_route(self, source):
        return self.route_strategy_dict[self.route_strategy](source)

    def get_straight_route(self, source):
        """
        pick up a straight route given an origin
        """
        return self.path_ids_dict[source, None]

    def step(self):
        """
        Advance the simulation by one step.
        """
        self.schedule.step()

    def add_scenario_info_to_dataframe(self, dataframe):
        """
        Add scenario info to the input dataframe: seed, breakdown probabilities of different bridge conditions
        __________
        Parameters:
            dataframe (pandas.DataFrame): input dataframe
        __________
        Return a dataframe with 5 added columns on seed, breakdown probabilities of different bridge conditions
        """
        seed_series = pd.Series(data=np.zeros(len(dataframe)) + self._seed, name="seed")
        cat_a_series = pd.Series(np.zeros(len(dataframe)) + self.scenario["A"], name="cat_a")
        cat_b_series = pd.Series(np.zeros(len(dataframe)) + self.scenario["B"], name="cat_b")
        cat_c_series = pd.Series(np.zeros(len(dataframe)) + self.scenario["C"], name="cat_c")
        cat_d_series = pd.Series(np.zeros(len(dataframe)) + self.scenario["D"], name="cat_d")
        dataframe = pd.concat([dataframe, seed_series, cat_a_series, cat_b_series, cat_c_series, cat_d_series], axis=1)
        return dataframe

    def get_vehicle_travel_time_dataframe(self, include_en_route=False, scenario_info=False):
        """
        Calculate the total travel time of all vehicles arriving at sinks and return a pandas dataframe
        ____________
        Parameters:
            scenario_info (Boolean): if True, return the dataframe with scenario information (seed, breakdown probabilities of different bridge conditions). False otherwise
            include_en_route (Boolean): if True, include travel time of en-route vehicles (i.e, vehicles hasn't reached the destination)
        ____________
        Return: pandas dataframe with the following columns: road_name, id, travel_time. And seed, breakdown probabilities of different bridge conditions
        """
        travel_time_dict = defaultdict(list)
        # add removed vehicles
        for vehicle in self.removed_vehicles:
            travel_time_dict["id"].append(vehicle.unique_id)
            travel_time_dict["ori_road"].append(vehicle.generated_by.road_name)
            travel_time_dict["des_road"].append(vehicle.get_destination_road_name())
            travel_time_dict["travel_time"].append(vehicle.calculate_travel_time())
            travel_time_dict["travel_distance"].append(vehicle.get_travel_distance())

        # add en-route vehicles
        if include_en_route:
            for agent in self.schedule.agents:
                if isinstance(agent, Vehicle):
                    travel_time_dict["id"].append(agent.unique_id)
                    travel_time_dict["ori_road"].append(agent.generated_by.road_name)
                    travel_time_dict["des_road"].append(agent.get_destination_road_name())
                    travel_time_dict["travel_time"].append(agent.calculate_travel_time())
                    travel_time_dict["travel_distance"].append(agent.get_travel_distance())

        dataframe = pd.DataFrame(travel_time_dict)
        # add scenario info
        if scenario_info:
            dataframe = self.add_scenario_info_to_dataframe(dataframe)
        return dataframe

    def record_bridge_breakdown(self, bridge):
        """
        Add to the bridge breakdown record when a bridge breaks down
        ____________
        Parameters:
            bridge (Bridge): a bridge instance
        ____________
        Return: None
        """
        assert isinstance(bridge, Bridge), "Input must be a Bridge object, but got {}".format(type(bridge))
        self.bridge_breakdown["id"].append(bridge.unique_id)
        self.bridge_breakdown["road_name"].append(bridge.road_name)
        self.bridge_breakdown["condition"].append(bridge.condition)
        self.bridge_breakdown["length"].append(bridge.length)
        self.bridge_breakdown["delay_time"].append(bridge.delay_time)

    def get_bridge_breakdown_dataframe(self, scenario_info=False):
        """
        Return a pandas dataframe with all the bridge breakdown occurrences
        ____________
        Parameters:
            scenario_info (Boolean): if True, return the dataframe with scenario information (seed, breakdown probabilities of different bridge conditions). False otherwise
        ____________
        Return: pandas dataframe with the following columns: road_name, id, delay_time. And seed, breakdown probabilities of different bridge conditions
        """
        dataframe = pd.DataFrame(self.bridge_breakdown)
        if scenario_info:
            dataframe = self.add_scenario_info_to_dataframe(dataframe)
        return dataframe


    def get_network_centrality_dataframe(self, scenario_info=False):
        """
        Return a pandas dataframe with all the bridge centralities
        ____________
        Parameters:
            scenario_info (Boolean): if True, return the dataframe with scenario information (seed). False otherwise
        ____________
        Return: pandas dataframe with the following columns: road_name, type, degree, betweeness, closeness, seed. And seed, breakdown probabilities of different bridge conditions
        """
        # add centrality as node attribute
        degree = nx.degree_centrality(self.graph)
        nx.set_node_attributes(self.graph, values=degree, name="degree")
        betweenness = nx.betweenness_centrality(self.graph)
        nx.set_node_attributes(self.graph, values=betweenness, name="betweenness")
        closeness = nx.closeness_centrality(self.graph)
        nx.set_node_attributes(self.graph, values=closeness, name="closeness")

        node_dict = dict(self.graph.nodes(data=True))
        dataframe = pd.DataFrame.from_dict(node_dict, orient='index').reset_index().rename(columns={'index': 'id'})

        if scenario_info:
            dataframe = self.add_scenario_info_to_dataframe(dataframe)
        return dataframe





# EOF -----------------------------------------------------------
