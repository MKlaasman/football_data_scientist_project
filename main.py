import data_parser
import preprocessing
import feature_engineering
import generate_EPV_values
import analysis


def run_pipeline():
    """
    Runs all functions in the pipeline. Parses tracking and events data from 52 xml files. Preprocesses the DataFrame
    to conform to Metrica Sports format. Then calculates EPV values to get the optimal passes using the
    Friends Of Tracking code, which can be found in the EPV_code folder. Then creates multiple features based on
    tracking and events data. Followed by the analysis, using a Linear Regression and Decision Tree.

    After each step files are saved to the /data folder.
    """
    data_parser.parse_data()
    preprocessing.preprocess()
    generate_EPV_values.generate_epv_files()
    feature_engineering.engineer_features()
    analysis.run_analysis()


if __name__ == '__main__':
    run_pipeline()


