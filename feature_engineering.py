# -*- coding: utf-8 -*-
"""
Created on Sun 26 July 10:35:34 2020

@author: Milan
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import os

DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, 'EPV_code')
DATADIR = os.path.join(DIRNAME, 'data')
os.chdir(PATH)

import Metrica_PitchControl as mpc
import Metrica_IO as mio


def create_filepaths_and_get_dataframes(match_id):
    """
    # creates filepaths and reads csv files of one match: tracking_home, tracking_away, and epv_df
        outputs the corresponding DataFrames

    Parameters
    -----------
        match_id: match id string
    Returns
    -----------
        epv_df: the events dataframe with corresponding Expected Possession Values (EPV) data
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
    """
    # create data paths
    tracking_home_path = DATADIR + '\\preprocessed\\' + match_id + '_tracking_home_processed.csv'
    tracking_away_path = DATADIR + '\\preprocessed\\' + match_id + '_tracking_away_processed.csv'
    epv_df_path = DATADIR + '\\transition_passes\\' + match_id + '_EPV_df.csv'

    # load in data files
    tracking_home = pd.read_csv(tracking_home_path, index_col=0)
    tracking_away = pd.read_csv(tracking_away_path, index_col=0)
    epv_df = pd.read_csv(epv_df_path, index_col=0)

    # add matchId to match, for later tracking_data flip.
    epv_df['match_id'] = base_file

    return epv_df, tracking_home, tracking_away


def get_pass_degrees(start_x, start_y, end_x, end_y):
    """
    # calculates pass direction in degrees ranging from -180 to 180
                - a value of 0 is has horizontal direction from left to right
                - Positive values represent a degree above the horizontal axis,
                - Negative values represent a degree under the horizontal axis

    Parameters
    -----------
        start_x: starting x position of the pass
        start_y: starting y position of the pass
        end_x: ending x position of the pass
        end_y: ending y position of the pass
    Returns
    -----------
        degree: the degree with the horizontal axis.
    """
    radians = math.atan2(end_y - start_y, end_x - start_x)
    degree = math.degrees(radians)
    return degree


def get_playing_direction(row, tracking_home, tracking_away):
    """
    # get the playing direction of the Home and Away team

    Parameters
    -----------
        row: selected row of the pass events to get playin direction
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
    Returns
    -----------
        playing_direction: the playing direction of the passing team of the selected row
    """
    # also you can directly return the playing_direction in the if statement
    if row.Team == 'Home':
        return mio.find_playing_direction(tracking_home, 'Home')
    return mio.find_playing_direction(tracking_away, 'Away')


def calculate_direction_of_pass(events, event_id, playing_direction):
    """
    #  Calculation of the direction of the pass, forwards, backwards or sideways based on the playing direction
    Here we took a value of 3 meters as a distance.

    Parameters
    -----------
        events: all transition pass events of this match
        event_id: index corresponding to the passing event
        playing_direction: the playing direction of the passing team of the selected row

    Returns
    -----------
        events: all transition pass events of this match including columns:
            - backward_pass, with 1 is True and 0 is False
            - forward_pass, with 1 is True and 0 is False
            - sideways_pass, with 1 is True and 0 is False
    """
    # get the values based on the playing direction
    if playing_direction == 1:  # playing direction left>right
        if events.loc[event_id, 'End X'] < (events.loc[event_id, 'Start X'] - 3):
            backward_pass = 1
        else:
            backward_pass = 0

        if events.loc[event_id, 'End X'] > (events.loc[event_id, 'Start X'] + 3):
            forward_pass = 1
        else:
            forward_pass = 0

    else:  # playing direction right>left
        if events.loc[event_id, 'End X'] > (events.loc[event_id, 'Start X'] + 3):
            backward_pass = 1
        else:
            backward_pass = 0

        if events.loc[event_id, 'End X'] < (events.loc[event_id, 'Start X'] - 3):
            forward_pass = 1
        else:
            forward_pass = 0

    if (events.loc[event_id, 'End X'] > events.loc[event_id, 'Start X'] - 3) \
            & (events.loc[event_id, 'End X'] < events.loc[event_id, 'Start X'] + 3):
        sideways_pass = 1
    else:
        sideways_pass = 0

    # assign values
    events.loc[event_id, 'backwards_pass'] = backward_pass
    events.loc[event_id, 'forward_pass'] = forward_pass
    events.loc[event_id, 'sideways_pass'] = sideways_pass
    return events


def feature_engineering_based_on_events(events, tracking_home, tracking_away):
    """
    # create multiple EPV, distance and degree based features using events data

    Parameters
    -----------
        events: all transition pass events of this match
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
    Returns
    -----------
        events: all transition pass events of this match including columns:
            - EEPV_added
            - diff_optimal: expected 'Expected' Expected Possession Value (EEPV) difference
                            between maximum (/optimal) and the actual pass
            - EEPV_perc_of_optimal_pass:  percentage of the EEPV_added in comparison with the optimal pass
            - pass_length: the euclidean distance between start location and end location of the pass
            - pass_length_optimal: the euclidean distance between start location and end location of the optimal pass
            - pass_direction_degrees: the direction of the pass in degrees (-180 to 180),
                            with 0 being horizontal (x-axis) from left to right
            - playing_direction: the playing direction of the passing team, either 1 (left>right) or -1 (right>left)
    """

    # add expected value difference with optimal pass and percentage of optimal pass
    events.loc[:, 'diff_optimal'] = events['max_EEPV_added'] - events['EEPV_added']
    events.loc[:, 'EEPV_perc_of_optimal_pass'] = (
                                                         events['EEPV_added'] / events['max_EEPV_added']) * 100

    for i, row in events.iterrows():
        # calculate the shortest distance of the actual pass and the optimal pass
        events.loc[i, 'pass_length'] = math.sqrt(
            (row['End X'] - row['Start X']) ** 2 + (row['End Y'] - row['Start Y']) ** 2)
        events.loc[i, 'pass_length_optimal'] = math.sqrt(
            (row['EPV_target_x'] - row['Start X']) ** 2 + (row['EPV_target_y'] - row['Start Y']) ** 2)
        events.loc[i, 'pass_direction_degrees'] = get_pass_degrees(row['Start X'], row['Start Y'], row['End X'],
                                                                   row['End Y'])

        # calculate direction of pass: backwards, sideways and forwards. For this we need the playing direction
        playing_direction = get_playing_direction(
            row, tracking_home, tracking_away)
        events.loc[i, 'playing_direction'] = playing_direction
        events = calculate_direction_of_pass(events, i, playing_direction)
    return events


def calculate_pitch_control_towards_goal(frame, team, event_id, events, tracking_home, tracking_away, column_name):
    """
    Calculates the pitch control percentage of the tiles between the passer and the goal

    Parameters
    -----------
        frame: frame of corresponding tracking row
        team: string 'Home' or 'Away'
        event_id: index corresponding to the passing event
        events: all transition pass events of this match
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
        column_name: the passing player's corresponding base column name for the tracking data
    Returns
    -----------
       percentage_pitch_control: percentage of pitch control of the attacking team of the tile between the passer,
        and the goal
    """
    # first get pitch control model parameters
    params = mpc.default_model_params()
    # find goalkeepers for offside calculation
    gk_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]
    # evaluated pitch control surface for pass event
    PPCa, xgrid, ygrid = mpc.generate_pitch_control_for_event(
        event_id, events, tracking_home, tracking_away, params, gk_numbers, field_dimen=(105., 68.,), n_grid_cells_x=50,
        offsides=True)

    # get playing_direction and x position of passing player
    if team == 'Home':
        player_x = tracking_home.loc[frame, column_name + '_x']
        playing_direction = mio.find_playing_direction(tracking_home, team)
        # player_y = tracking_home.loc[frame, column_name + '_y']
    else:
        player_x = tracking_away.loc[frame, column_name + '_x']
        playing_direction = mio.find_playing_direction(tracking_away, team)

    # divide the playing field into grid cells
    n_grid_cells_x = 32
    n_grid_cells_y = 50
    field_dimen = [105, 68]

    # get grid cell of passing player
    dx = field_dimen[0] / n_grid_cells_x
    x_grid = np.arange(n_grid_cells_x) * dx - field_dimen[0] / 2. + dx / 2.

    # calculate the pitch control value of the grids towards goal,
    # and calculate the maximum pitch control for that area:
    if playing_direction == 1:  # direction: left -> right
        # get number of grids closer to goal
        num_grids_to_goal = len([i for i in x_grid if i > player_x])

        # maximum == number of grids to goal and a Pitch Control value of 1,
        # meaning that it is totally controlled by attacker
        max_pitch_control = num_grids_to_goal * n_grid_cells_y
        # get the pitch control using the pitch control values that are created above (PPCa)
        pitch_control_att_team = sum(sum(PPCa[-num_grids_to_goal:]))

    else:  # direction: right -> left
        num_grids_to_goal = len([i for i in x_grid if i < player_x])
        max_pitch_control = num_grids_to_goal * n_grid_cells_y
        pitch_control_att_team = sum(sum(PPCa[:num_grids_to_goal]))

    percentage_pitch_control = round(
        (pitch_control_att_team / max_pitch_control) * 100, 2)
    return percentage_pitch_control


def calculate_team_width(frame, tracking_def, tracking_att):
    """
    # calculates the width of both the attacking and defending team

    Parameters
    -----------
        frame: frame of corresponding tracking row
        tracking_def: tracking DataFrame for the attacking team
        tracking_att: tracking DataFrame for the defending team
    Returns
    -----------
        def_team_width: the team width on the y-axis of the defending team
        att_team_width: the team width on the y-axis of the attacking team
    """

    # list all y columns in the tracking DataFrame of the defensive team
    def_team_columns = [c for c in tracking_def.columns if (
            c[-2:] in ['_y']) & (c not in ['ball_y'])]
    y_def_team = tracking_def.loc[frame, def_team_columns]

    # list all y columns in the tracking DataFrame of the attacking team
    att_team_columns = [c for c in tracking_att.columns if (
            c[-2:] in ['_y']) & (c not in ['ball_y'])]
    y_att_team = tracking_att.loc[frame, att_team_columns]

    # take difference between maximum and minimum to calculate the field length
    def_team_width = max(y_def_team) - min(y_def_team)  # defending team width
    att_team_width = max(y_att_team) - min(y_att_team)  # attacking team width
    return def_team_width, att_team_width


def get_distances_to_defensive_lines(frame, player_id, team, tracking_att, tracking_def):
    """
    # calculates:
            - Distances to defensive lines: both to the passing player's own defensive line,
                and the defending team's defending line
            - the team length (x-axis) of both team, based on the difference between the defensive lines.
            - the team width (y-axis) of both team, based on maximum and minimum values
            - the number of defenders closer to goal
            - ca

    Parameters
    -----------

        frame: frame of corresponding tracking row
        player_id: the player id string of the passing player
        team: string 'Home' or 'Away'
        tracking_att: tracking DataFrame for the attacking team
        tracking_def: tracking DataFrame for the defending team
    Returns
    -----------
    dist_def_line_own: shortest distance to the attackers defensive line
    dist_def_line_opponent: shortest distance to the defenders defensive line
    length_playing_field: length of the playing field, calculated as the difference between the defensive lines
    num_defenders_closer_to_goal: the number of defenders closer to goal than the passing player
    def_team_length: defending team length (x-axis)
    att_team_length: attacking team length (x-axis)
    def_team_width: defending team width (y-axis)
    att_team_width: attacking team width (y-axis)
    """
    # The defensive line is equal to the 3 defenders that are closest to goal, except the goalkeeper
    # So we need the goalkeeper numbers to exclude them from the X positions list we will make.
    x_passing_player = tracking_att.loc[frame, team + '_' + player_id + '_x']
    gk_numbers = [mio.find_goalkeeper(tracking_att), mio.find_goalkeeper(tracking_def)]

    # list all y columns in the tracking DataFrame of the defensive team
    def_team_columns = [c for c in tracking_def.columns if (
            c[-2:] in ['_x']) & (c not in ['ball_x', ]) & (gk_numbers[1] not in c)]
    x_def_team = tracking_def.loc[frame, def_team_columns]

    # list all y columns in the tracking DataFrame of the defensive team
    att_team_columns = [c for c in tracking_att.columns if (
            c[-2:] in ['_x']) & (c not in ['ball_x', ]) & (gk_numbers[0] not in c)]
    x_att_team = tracking_att.loc[frame, att_team_columns]

    # Using X tracking data of both team and the playing direction,
    # we can calculate the average of the three defensive players and,
    # the number of defenders closer to goal
    playing_direction = mio.find_playing_direction(tracking_att, team)
    if playing_direction == -1:
        # Passing team plays right > left, thus defensive line == 3 players with lowest x (except goalkeeper)
        # attacking team
        x_att_team = [x for x in x_att_team if str(x) != 'nan']
        three_defensive_players = sorted(x_att_team)[-3:]
        att_defensive_line = np.mean(three_defensive_players)

        # defenders
        three_defensive_players = sorted(x_def_team)[:3]
        def_defensive_line = np.mean(three_defensive_players)

        # opponents closer to goal:
        num_defenders_closer_to_goal = len(
            [i for i in x_def_team if i < x_passing_player])
    else:
        # Passing team plays left > right, thus defensive line == 3 players with highest x (except goalkeeper)
        # filter out nan values:
        # attacking team
        three_defensive_players = sorted(x_att_team)[:3]
        att_defensive_line = np.mean(three_defensive_players)

        # defenders
        x_def_team = [x for x in x_def_team if str(x) != 'nan']
        three_defensive_players = sorted(x_def_team)[-3:]
        def_defensive_line = np.mean(three_defensive_players)

        # opponents closer to goal:
        num_defenders_closer_to_goal = len(
            [i for i in x_def_team if i > x_passing_player])

    # calculate distance to defensive lines for opponents and own
    dist_def_line_opponent = round(abs(def_defensive_line - x_passing_player), 2)
    dist_def_line_own = round(abs(att_defensive_line - x_passing_player), 2)
    length_playing_field = round(abs(att_defensive_line - def_defensive_line), 2)

    # defending team length without keeper
    def_team_length = max(x_def_team) - min(x_def_team)
    # attacking team length without keeper
    att_team_length = max(x_att_team) - min(x_att_team)

    def_team_width, att_team_width = calculate_team_width(
        frame, tracking_def, tracking_att)
    return dist_def_line_own, dist_def_line_opponent, length_playing_field, num_defenders_closer_to_goal, def_team_length, att_team_length, def_team_width, att_team_width


def get_distance(start_x, end_x, start_y, end_y):
    """
    # Calculates euclidean distance and returns distance value
    """
    distance = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    return distance


def get_distances_to_players(frame, team, player_id, player_tracking, tracking):
    """
    # get distances of passing player to the closest player, the number of players within 5 and 15 meters,
        and the average speed of the selected team

    Parameters
    -----------

        frame: frame of corresponding tracking row
        team: string 'Home' or 'Away'
        player_id: the player id string of the passing player
        player_tracking: list of x and y tracking position of the passing player [x,y]
        tracking: tracking DataFrame of either the attacking or defending team
    Returns
    -----------
        average_distance: average distance between the passing player and all players of the selected team
        dist_closest_player: distance to the closest player of the selected team
        num_within_5m: number of players within 5 meter of the passing player of the selected team
        num_within_15m: number of players within 15 meter of the passing player of the selected team
    """

    # get all x and y columns
    c_x = team + '_' + player_id + '_x'
    c_y = team + '_' + player_id + '_y'

    # get all x and y column names except the passing player's
    columns = [c for c in tracking.columns if (
            c[-2:] in ['_x', '_y']) & (c not in [c_x, c_y, 'ball_x', 'ball_y'])]
    distances = []

    for i in np.arange(0, len(columns), 2):
        # select next player
        temp_x = tracking.loc[frame, columns[i]]
        temp_y = tracking.loc[frame, columns[i + 1]]

        # calculate euclidean distance
        distance = get_distance(
            player_tracking[0], temp_x, player_tracking[1], temp_y)
        distances.append(distance)

    distances = [x for x in distances if str(x) != 'nan']
    # average distance
    average_distance = np.sum(distances) / 10
    if len(distances) == 0:
        dist_closest_player = np.nan
    else:
        dist_closest_player = min(distances)

    # players within radius of 5 or 15m
    within_5m = [i for i in distances if i < 5]
    within_15m = [i for i in distances if i < 15]
    num_within_5m = len(within_5m)
    num_within_15m = len(within_15m)
    return round(average_distance, 2), round(dist_closest_player, 2), num_within_5m, num_within_15m


def order_tracking_files(frame, team, column_name, tracking_home, tracking_away):
    """
    # order the tracking data such that we select the tracking data for the attacking team,
        defending team and the x and y tracking values of the passing player

    Parameters
    -----------
        frame: frame of corresponding tracking row
        team: string 'Home' or 'Away'
        column_name: the passing player's corresponding base column name for the tracking data
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
    Returns
    -----------
        tracking_att: tracking DataFrame for the attacking team
        tracking_def: tracking DataFrame for the defending team
        tracking_player: x and y tracking values of the passing player
    """

    if team == 'Home':
        tracking_att = tracking_home
        tracking_def = tracking_away
    else:
        tracking_att = tracking_away
        tracking_def = tracking_home

    # average distance teammates:
    # get_average_distance_to_teammates(team, player_id, tracking_att, frame)
    column_x = column_name + '_x'
    column_y = column_name + '_y'
    tracking_player = tracking_att.loc[frame, [column_x, column_y]]
    return tracking_att, tracking_def, tracking_player


def get_team_speed_features(frame, tracking_att, tracking_def):
    """
    # get the average speed of both teams individually and combined

    Parameters
    -----------
        frame: frame of corresponding tracking row
        tracking_att: tracking DataFrame for the attacking team
        tracking_def: tracking DataFrame for the defending team
    Returns
    -----------
        speed_att_team: average speed of the attacking team
        speed_def_team: average speed of the attacking team
        speed_average: average speed of both teams
    """
    columns_speed_att = [c for c in tracking_att.columns if (
            c[-5:] in ['speed']) & (c not in ['ball_x', 'ball_y'])]
    speed_att_team = np.mean(tracking_att.loc[frame, columns_speed_att])
    columns_speed_def = [c for c in tracking_def.columns if (
            c[-5:] in ['speed']) & (c not in ['ball_x', 'ball_y'])]
    speed_def_team = np.mean(tracking_def.loc[frame, columns_speed_def])
    speed_average = (speed_att_team + speed_def_team) / 2

    return round(speed_att_team, 2), round(speed_def_team, 2), round(speed_average, 2)


def get_passers_speed_and_orientation_features(frame, tracking, column_name):
    """
    # get the passing player's speed, orientation and distance that he covered in the last 5 seconds

    Parameters
    -----------
        frame: frame of corresponding tracking row
        column_name: the passing player's corresponding base column name for the tracking data
    Returns
    -----------
        speed: the speed of the passing player
        distance_covered_5sec: the distance covered in the last 5 seconds, calculated using numpy.diff
        orientation: the orientation of the passing player in degrees (-180 to 180)
    """
    speed = tracking.loc[frame, column_name + '_speed']

    # calculate trajectory distance:
    points = np.array(
        tracking.loc[(frame - 50):frame, [column_name + '_x', column_name + '_y']])
    d = np.diff(points, axis=0)
    distance_covered_5sec = sum(sum((abs(d))))

    # calculate orientation
    vx = tracking.loc[frame, column_name + '_vx']
    vy = tracking.loc[frame, column_name + '_vy']

    # calculate orientation using the speed vectors.
    myradians = math.atan2(vy - 0, vx - 0)
    if not math.isnan(myradians):
        orientation = int(math.degrees(myradians))
    else:
        orientation = np.nan
    return round(speed, 2), round(distance_covered_5sec, 2), round(orientation, 2)


def get_basic_row_attributes(row):
    """
    # get basic row attributes: frame, team, player id and column name

    Parameters
    -----------
        row: selected row of get basic row attributes
    Returns
    -----------
        frame: frame of corresponding tracking row
        team: string 'Home' or 'Away'
        player_id: the player id string of the passing player
        column_name: the passing player's corresponding base column name for the tracking data
    """
    frame = row['Start Frame']
    team = row.Team
    player_id = str(int(row.From))
    column_name = row.Team + '_' + player_id
    return frame, team, player_id, column_name


def append_features_row(features_df, data):
    """
    # append a single feature row to the dataframe

    Parameters
    -----------
        features_df: DataFrame of all the features that are already generated
        data: the features that are generated for the current passing event
    Returns
    -----------
        features_df: DataFrame of all the features
    """
    columns = ['index', 'player_speed', 'player_distance_covered_5sec', 'player_orientation',
               'dist_def_line_own', 'dist_def_line_opponent',
               'length_playing_field', 'num_defenders_closer_to_goal',
               'def_team_length', 'att_team_length',
               'def_team_width', 'att_team_width',
               'speed_attackers', 'speed_defenders', 'speed_average',
               'average_distance_defenders', 'distance_closest_defender',
               'num_defenders_within_5m', 'num_defenders_within_15m',
               'average_distance_attackers', 'distance_closest_attacker',
               'num_attackers_within_5m', 'num_attackers_within_15m',
               'pitch_control_percentage']
    row = pd.DataFrame(data=data).T
    row.columns = columns

    if isinstance(features_df, pd.DataFrame):
        frames = [features_df, row]
        features_df = pd.concat(frames)
    else:  # if first
        features_df = row
    return features_df


def feature_engineering_based_on_tracking(events, tracking_home, tracking_away):
    """
    # create multiple different type of features using tracking data, such as:
            - passing players attributes
            - distances from passing player
            - teams attributes
            - pitch Control

    Parameters
    -----------
        events: all transition pass events of this match
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
    Returns
    -----------
        features_df:  DataFrame of all the features generated on the tracking data
    """
    features_df = 0  # assign zero, so we can create a new DataFrame object after first loop

    # get speed and acceleration of passing player
    for i, row in tqdm(events.iterrows(), total=len(events)):
        frame, team, player_id, column_name = get_basic_row_attributes(row)
        tracking_att, tracking_def, tracking_player = order_tracking_files(
            frame, team, column_name, tracking_home, tracking_away)
        # get player features
        player_speed, player_distance_covered_5sec, player_orientation = get_passers_speed_and_orientation_features(
            frame, tracking_att, column_name)
        # get team attributes
        speed_attackers, speed_defenders, speed_average = get_team_speed_features(
            frame, tracking_att, tracking_def)

        # get_distance_closest_opponents(frame, team, player_id, tracking_att, tracking_def)
        dist_def_line_own, dist_def_line_opponent, \
        length_playing_field, num_defenders_closer_to_goal, def_team_length, att_team_length, \
        def_team_width, att_team_width = get_distances_to_defensive_lines(
            frame, player_id, team, tracking_att, tracking_def)

        average_distance_defenders, distance_closest_defender, num_defenders_within_5m, num_defenders_within_15m = get_distances_to_players(
            frame, team, player_id, tracking_player, tracking_def)
        average_distance_attackers, distance_closest_attacker, num_attackers_within_5m, num_attackers_within_15m = get_distances_to_players(
            frame, team, player_id, tracking_player, tracking_att)

        pitch_control_percentage = calculate_pitch_control_towards_goal(
            frame, team, i, events, tracking_home, tracking_away, column_name)
        data = [i, player_speed, player_distance_covered_5sec, player_orientation,
                dist_def_line_own, dist_def_line_opponent,
                length_playing_field, num_defenders_closer_to_goal,
                def_team_length, att_team_length,
                def_team_width, att_team_width,
                speed_attackers, speed_defenders, speed_average,
                average_distance_defenders, distance_closest_defender,
                num_defenders_within_5m, num_defenders_within_15m,
                average_distance_attackers, distance_closest_attacker,
                num_attackers_within_5m, num_attackers_within_15m,
                pitch_control_percentage]
        features_df = append_features_row(features_df, data)
    return features_df


def engineer_features():
    """
    first creates file path and then creates multiple features based on tracking and events data for all 52 matches
    in the dataset. Saves ..._EPV_features.csv files to data folder
    """
    features_path = DATADIR + 'EPV_features\\'
    xml_directory = os.fsencode(DATADIR + r'\xml_files')
    files = os.listdir(xml_directory)
    for file in tqdm(files[:1], total=len(files[:1]), position=0):
        filename = os.fsdecode(file)
        base_file = filename.split('.xml')[0]
        epv_df, tracking_home, tracking_away = create_filepaths_and_get_dataframes(base_file, DATADIR)
        epv_df = feature_engineering_based_on_events(
            epv_df, tracking_home, tracking_away)
        features_df = feature_engineering_based_on_tracking(
            epv_df, tracking_home, tracking_away)
        EPV_features_df = pd.merge(
            epv_df, features_df, left_on=epv_df.index, right_on='index')
        EPV_features_df.to_csv(features_path + base_file + '_EPV_features.csv')
