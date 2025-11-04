Created by: EPA133a Group 05

|          Name           | Student Number |
|:-----------------------:|:---------------|
|       Yashi Punia       | 6045979        |
|        Nhi Trân         | 5914884        |
|     Jeroen van Til      | 5167280        |
|    Koen van den Berg    | 4968530        |
| Gabriella Low Chew Tung | 5973058        |

## Introduction

- This project automatically generates a model of the Bangladesh road and bridge infrastructure based on the data provided in following location [cleaned_data.csv](data/cleaned_data.csv).
- This cleaned data was created by running the [Data_file_clean.py](model/Data_file_clean.py) that cleans and formats the raw data from files "../data/BMMS_overview.xlsx" and "../data/_roads3.csv".
- The model generation process is facilitated by classes defined in [components.py](model/components.py) and [model.py](model/model.py).
- The model is simulated for 5 scenarios using 10 seeds in [model_run.py](model/model_run.py). When model_run.py is executed, relevant data is collected and exported to csv files for each scenario.
- The results of each scenario are recorded in the [experiment folder](experiment).
- The output data is composed of the total travel times of each vehicle agent created as it journeys from Source to Sink. This is found in the files "vehicle-scenario_X.csv".
- Data regarding which broken down bridges vehicles encountered and corresponding delay times are recorded for analysis in the [experiment folder](experiment) folder. This data is found in "bridge_scenario_X.csv".
- In [G05-A3-Bonus-Question.ipynb](notebooks/G05-A3-Bonus-Question.ipynb), an analysis is performed to address the Bonus Question of Lab 3 Assignment on identifying road intersections using the shape files of the Bangladesh road network.
- Data analysis was performed at plots generated in [visualisation.ipynb](notebooks/G05-A3-data-visualisation.ipynb).
## How to Use

To use this project, execute the following:
1. Create and activate a virtual environment.
2. Navigate to and execute the following command in the terminal:
```
    $ pip install-r EPA133a-G05-A2/requirements.txt
```
3. To generate the [cleaned_data.csv](data/cleaned_data.csv) file, open [Data_file_clean.py](model/Data_file_clean.py) and execute it. 
4. To build and simulate the model for the set scenarios over multiple seeds, execute the file [model_run.py](model/model_run.py). Execute the experiment function to simulate the model over all scenarios. 
    Should the IDE request the download of openpxyl, please execute the following in your terminal. Alternatively, install the package using the IDE interface.
```
    $pip install openpxyl 
```
5. See [experiment folder](experiment) folder for the csv files created by the script.
6. See [G05-A3-data-visualisation.ipynb](notebooks/G05-A3-data-visualisation.ipynb) for data analysis and generated plots.
