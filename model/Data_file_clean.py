import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

pd.options.mode.chained_assignment = None

def clean_bridges():
    df_bridges = pd.read_excel('../data/BMMS_overview.xlsx')
    # remove duplicate bridges
    # First remove those right lanes
    df_bridges = df_bridges[df_bridges["name"].str[-3:] != "(R)"]

    # Then combine those with same LRP, as they are seen as the same bridge.
    df_bridge_road_count = pd.DataFrame(df_bridges["LRPName"].value_counts())

    # Firstly, drop those with duplicate names
    df_bridges.drop_duplicates(subset="LRPName", inplace=True)

    # Then drop the ones which are most likely impossible
    # which bridges will be dropped
    bridges_index_drop = []
    df_bridges.sort_values(by="chainage", ascending=True, inplace=True, ignore_index=True)

    # Grab bridge columns needed
    df_bridges = df_bridges[["road", "lat", "lon", "chainage", "length", "condition", "name"]]

    # Add model type
    df_bridges["model_type"] = "bridge"

    # make sure the chaninage is in meters
    df_bridges.chainage = df_bridges.chainage * 1000  # bridges are in meters, chainage in kilometers All should be in meters

    # Will check every bridge if the length added to chainage results in another bridge found
    for index, row in df_bridges.iterrows():
        # Skips bridges already getting dropped
        if index in bridges_index_drop:
            continue
        chainage = df_bridges.loc[index, "chainage"]
        search = df_bridges.loc[index, "length"] + chainage

        # Checks if the bridges are basically in the same area as another bridge. The first bridge stays. The others
        # gets removed
        if df_bridges[df_bridges["chainage"] <= search].loc[index + 1:].any(axis=None):
            bridges_index_drop = bridges_index_drop + (
                df_bridges.index[(df_bridges["chainage"] <= search) & (df_bridges.index > index)].tolist())

    # Now the found bridges will be dropped
    df_bridges.drop(axis="index", index=bridges_index_drop, inplace=True)
    df_bridges.reset_index(drop=True, inplace=True)

    # in case for later
    df_bridge_road_count.rename(columns={"count": "lanes"})

    return df_bridges

def add_intersections():
    df_roads = pd.read_csv('../data/_roads3.csv')
    # For checked roads, those that have been fully found or are not looked at because of certain restrictions.
    roads_checked = []
    # For the intersections that are found.
    intersections_found = pd.DataFrame()
    # This is based on looking at 10 different possible intersections, all values were lower than this.
    intersect = 0.001

    #Check the roads we need and find the possible intersections.
    for road in ["N1", "N2"]:

        df_roads_search = df_roads[df_roads.road == road]

        # Only roads that are smaller than 25km need to be found
        if df_roads_search.chainage.max() <= 25:
            roads_checked = roads_checked + [road]
            continue

        df_roads_compare = df_roads[df_roads.road != road]

        # To makes sure the road is compared to a road close by to save on power.
        search_lat_min = df_roads_search.lat.min() * 0.97
        search_lat_max = df_roads_search.lat.max() * 1.03
        search_lon_min = df_roads_search.lon.min() * 0.97
        search_lon_max = df_roads_search.lon.max() * 1.03

        for road_compare in df_roads_compare.road.unique():

            # Skips if the road has already been compared with every road
            if road_compare in roads_checked:
                continue

            # Make sure the road is skipped if it is less than 25 kilometers, plus adds it to the checked list.
            if df_roads_compare[df_roads_compare.road == road_compare].chainage.max() <= 25:
                roads_checked = roads_checked + [road_compare]
                continue

            # Grab road we want to compare
            df_roads_compare_here = df_roads_compare[df_roads_compare.road == road_compare]

            # Check if it is close to the search road
            if df_roads_compare_here[((df_roads_compare_here.lat <= search_lat_max) & (df_roads_compare_here.lat >= search_lat_min)) \
                                & ((df_roads_compare_here.lon <= search_lon_max) & (df_roads_compare_here.lon >= search_lon_min))].empty:
                continue

            # Calculates the distances between every datapoint.
            distance = cdist(
                df_roads_search[["lat", "lon"]],
                df_roads_compare_here[["lat", "lon"]],
                metric="euclidean"
            )
            #print(df_roads_search.road.unique(), df_roads_compare_here.road.unique(), np.min(distance))
            # ['N1'] ['N101'] 0.04 Not
            # ['N1']['N102'] 2.8e-05 Intersect
            # ['N1'] ['N103'] 0.48 Not
            # ['N1'] ['N104'] 8.8e-05 Intersect
            # ['N1'] ['N105'] 2.8e-05 Intersect
            # ['N1'] ['N2'] 1.9e-04 Intersect
            # ['N1'] ['N3'] 0.1 Not
            if distance.min() < intersect:
                # Grab the indexes of the smallest distance
                search_intersection_index = np.where(distance == distance.min())[0][0]
                compare_intersection_index = np.where(distance == distance.min())[1][0]

                # Grab the pd entry of the possible intersection
                df_search_found = df_roads_search.reset_index(drop=True).loc[search_intersection_index]
                df_compare_found = df_roads_compare_here.reset_index(drop=True).loc[compare_intersection_index]

                # Make it a dataframe
                df_search_found = df_search_found.to_frame()
                df_compare_found = df_compare_found.to_frame()

                # Transpose it
                df_search_found = df_search_found.T
                df_compare_found = df_compare_found.T

                # Then grabbing the necessary information and making sure it is all correct.
                # The lat and lon need to be the same for later
                df_search_found = df_search_found[["road", "chainage", "lat", "lon", "name"]]
                df_search_found["condition"] = "-"
                df_search_found["model_type"] = "intersection"
                df_search_found["length"] = 1

                df_compare_found = df_compare_found[["road", "name","chainage"]]
                df_compare_found["condition"] = "-"
                df_compare_found["model_type"] = "intersection"
                df_compare_found["lat"] = df_search_found.iloc[0].lat
                df_compare_found["lon"] = df_search_found.iloc[0].lon
                df_compare_found["length"] = 1

                # Add it to those found
                intersections_found = pd.concat([intersections_found, df_search_found, df_compare_found],
                                                              ignore_index=True)

        # In order to make sure the same road does not get checked twice.
        roads_checked = roads_checked + [road]
    # Chainage needs to be in meters
    intersections_found.chainage = intersections_found.chainage * 1000
    intersections_found.to_csv('../data/intersections.csv')

    return intersections_found

def add_road_lengths(df_bridges, road, roads_search):
    # This function adds the length of the roads to the bridges dataset

    # First make sure chainage is from low to high
    df_bridges.sort_values(by=["chainage"], inplace=True, ignore_index=True)

    # Now set up the links in between
    df_shift = df_bridges.shift(periods=1, fill_value=0)  # shift values
    df_roads = pd.DataFrame()
    df_roads["chainage"] = df_shift["chainage"] + df_shift["length"]
    df_roads["length"] = (df_bridges.chainage) - (df_shift.chainage + df_shift.length)
    df_roads["lat"] = (df_bridges.lat + df_shift.lat) / 2
    df_roads["lon"] = (df_bridges.lon + df_shift.lon) / 2
    df_roads["model_type"] = "link"
    df_roads["condition"] = "-"
    df_roads["road"] = road
    df_roads["name"] = ""

    # Set the parameters for the source
    df_roads.loc[0, "model_type"] = "sourcesink"
    df_roads.loc[0, "chainage"] = 0
    df_roads.loc[0, "lat"] = df_bridges.loc[0, "lat"]
    df_roads.loc[0, "lon"] = df_bridges.loc[0, "lon"]
    df_roads.loc[0, "name"] = ""

    # set the parameters for the sink
    df_sink = pd.DataFrame()
    df_sink.loc[0, "length"] = 0
    df_sink.loc[0, "model_type"] = "sourcesink"
    df_sink.loc[0, "lat"] = df_bridges["lat"].iloc[-1]
    df_sink.loc[0, "lon"] = df_bridges["lon"].iloc[-1]
    df_sink.loc[0, "condition"] = "-"
    df_sink.loc[0, "road"] = road
    df_sink.loc[0, "chainage"] = df_bridges["chainage"].iloc[-1] + df_bridges["length"].iloc[-1]
    df_sink.loc[0, "name"] = ""

    df = pd.concat([df_roads, df_sink], ignore_index=True)
    df = df[df.length >= 0]
    df.loc[((df.length > 0) & (df.chainage == 0)), "chainage"] = 0
    df.sort_values(by=["chainage"], inplace=True, ignore_index=True)
    return df

def link_intersections(df_unlinked):
    skipped = []
    for index, row in df_unlinked[df_unlinked.model_type == "intersection"].iterrows():
        if index in skipped:
            continue
        index_other = df_unlinked.index[((df_unlinked.lat == row["lat"]) & (df_unlinked.lon == row["lon"]) & (df_unlinked.index != index)
                                         & (df_unlinked.model_type == "intersection"))]
        df_unlinked.loc[index_other, "id"] = df_unlinked.loc[index, "id"]
        skipped = skipped + index_other.tolist()

    return df_unlinked
def make_df_model_ready():
    # road, id,      model_type, name,  lat, lon, length
    # N1,   1000000, source,     source, 0,   0,   4
    # N1,   1000001, link,       link1,  1,   1,   1800
    # df[["road: roadname"
    # , id: unique identifier: counting total length of dataframe.
    # , model_type: bridge/link(road)
    # , name: from the name of the datasets, otherwise empty string
    # , lat: lat
    # , lon: lon
    # , length: bridge length, roads, calculate through chainage and length of the bridges
    # , condition: bridge condition, roads -
    # , model_type: link, bridge, sourcesink
    # , chainage: how far along the road in kilometers

    #First clean the bridges
    df_bridges = clean_bridges()

    # Then add intersections
    df_intersections = add_intersections()

    # Remove roads that are not in both datasets. As some have don't have bridges, which means it's outside of our focus
    # While other roads we don't look at, as they need to be >25km and connected to the N1 or N2
    # (done through add_intersections)
    df_bridges = df_bridges[df_bridges.road.isin(df_intersections.road.unique())]
    df_intersections = df_intersections[df_intersections.road.isin(df_bridges.road.unique())]

    # Then combine the two datasets
    df = pd.concat([df_bridges, df_intersections], ignore_index=True)
    df.loc[df["chainage"] == 0, "chainage"] = 1

    #add road lengths
    df_road_lengths = pd.DataFrame()
    for road in df['road'].unique():
        df_road_lengths = pd.concat([df_road_lengths, add_road_lengths(df[df.road == road], road,
                                             df['road'].unique())], ignore_index=True)

    df = pd.concat([df, df_road_lengths], ignore_index=True)
    df.sort_values(by=["road", "chainage"], inplace=True, ignore_index=True)

    # add id's to the df data
    df["id"] = np.linspace(10000, 9999 + len(df), len(df))
    df = df[['road', 'id', 'model_type', 'condition', 'name', 'lat', 'lon', 'length']]


    df = link_intersections(df)

    return df

df = make_df_model_ready()
df.to_excel("../data/cleaned_data.xlsx", index = False)
df.to_csv('../data/cleaned_data.csv', index = False)
