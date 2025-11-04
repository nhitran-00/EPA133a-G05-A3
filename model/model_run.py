from model import BangladeshModel
import pandas as pd

"""
    Run simulation
    Export simulation outputs as csv files in experiment folder
"""

# ---------------------------------------------------------------

# run time 5 x 24 hours; 1 tick 1 minute
run_length = 5 * 24 * 60

seeds = [12345679, 627920158, 4393959, 87654321, 123459876,
         23456789, 918273645, 1029384756, 123456789, 321654987]

scenarios = [[0, 0, 0, 0], [0, 0, 0, 0.05], [0, 0, 0.05, 0.10],
             [0, 0.05, 0.10, 0.20], [0.05, 0.10, 0.20, 0.40]]


# debgging
# run_length = 1000
#
# seeds = [12345679, 627920158]
#
# scenarios = [[0, 0.05, 0.10, 0.20]]


def experiment(run_length=run_length, seeds=seeds, scenarios=scenarios, include_en_route=True, scenario_info=True):
    """
    Run simulation using the input
    ____________
    Parameters
        run_length int: number of time steps for the model to run
        seeds list: seeds input for the model
        scenarios list(list): list of breakdown probabilities of different bridge category. I.e [[0, 0, 0, 0], [0, 0, 0, 0.05], [0, 0, 0, 0.10]]
        scenario_info bool: if True, include scenario information in the output csv file. False otherwise
    ____________
    Returns None
    """
    for i, scenario in enumerate(scenarios):
        df_car = []
        df_bridge = []
        df_graph = []
        for seed in seeds:
            sim_model = BangladeshModel(seed=seed,
                                        cat_a_failure=scenario[0], cat_b_failure=scenario[1],
                                        cat_c_failure=scenario[2], cat_d_failure=scenario[3])
            for j in range(run_length):
                sim_model.step()

            data_vehicles = sim_model.get_vehicle_travel_time_dataframe(
                include_en_route=include_en_route, scenario_info=scenario_info)
            df_car.append(data_vehicles)

            data_bridges = sim_model.get_bridge_breakdown_dataframe(scenario_info=scenario_info)
            df_bridge.append(data_bridges)

            data_graph = sim_model.get_network_centrality_dataframe(scenario_info=scenario_info)
            df_graph.append(data_graph)

        # # write output file for a scenario
        pd.concat(df_car).to_csv("../experiment/vehicle_scenario_%d.csv" %i, index=False)
        pd.concat(df_bridge).to_csv("../experiment/bridge_scenario_%s.csv" %i, index=False)
        pd.concat(df_graph).to_csv("../experiment/centrality_scenario_%s.csv" % i, index=False)
        print(f"{i+1} out of {len(scenarios)} scenarios are finished simulating. ")


if __name__ == "__main__":
    experiment()


