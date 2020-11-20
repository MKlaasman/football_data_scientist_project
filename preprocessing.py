# -*- coding: utf-8 -*-
"""
Created on Fri July 24 14:33:10 2020

@author: Milan
"""
import pandas as pd
import math
from tqdm import tqdm
import os

DIRNAME = os.path.dirname(__file__)
PATH = os.path.join(DIRNAME, 'EPV_code')
os.chdir(PATH)
import Metrica_Velocities as mvel


def change_time_values_tracking(tracking_df):
    """
    Changes the time of the second half tracking, i.e. add the time of the first half to the Time [s]
    Changes the time from milliseconds to seconds
    Parameters
    -----------
        tracking_df: merged tracking DataFrame
    Returns
    -----------
        tracking_df: the processed tracking DataFrame
    """

    # get second half start index based on the change in the Period column
    # and add first half Time to second half tracking data
    index_second_half_tracking = tracking_df[tracking_df['Period'].diff() == 1].index[0]

    # add 100 to the first frame of the second half, since it is 10 frames per seconds and this is still in milliseconds 
    values_second_half = tracking_df.loc[index_second_half_tracking:, 'Time [s]'] + (
                tracking_df.loc[index_second_half_tracking - 1, 'Time [s]'] + 100)
    # change time from milliseconds to seconds
    tracking_df.loc[index_second_half_tracking:, 'Time [s]'] = values_second_half
    tracking_df['Time [s]'] = tracking_df['Time [s]'] / 1000

    return tracking_df


def split_tracking_data(tracking_df):
    """
    Splits tracking dataframe into a home and an away tracking dataframe
    Parameters
    -----------
        tracking_df: tracking DataFrame of both the Home and Away teams
    Returns
    -----------
        home: tracking dataframe of the home team
        away: tracking dataframe of the away team
    """

    # change tracking data from centimeter to meter
    tracking_df[tracking_df.columns[2:]] = tracking_df[tracking_df.columns[2:]] / 100

    # get home and away column names
    home_cols = ['Period', 'Time [s]'] + \
                [col for col in tracking_df.columns if 'home' in col] + ['ball_x', 'ball_y']
    away_cols = ['Period', 'Time [s]'] + \
                [col for col in tracking_df.columns if 'away' in col] + ['ball_x', 'ball_y']

    # create home and away tracking
    home = tracking_df[home_cols]
    home.columns = home.columns.str.replace("home", "Home")
    away = tracking_df[away_cols]
    away.columns = away.columns.str.replace("away", "Away")
    return home, away


def change_to_single_playing_direction(tracking_df):
    """
    Flip coordinates in second half so that each team always shoots in the same direction through the match.

    Parameters
    -----------
        tracking_df: tracking DataFrame of a single team
    Returns
    -----------
        tracking_df: tracking DataFrame of a single team, with flipped positional data for the second half
    """

    # get first index of the second half
    second_half_idx = tracking_df.Period.idxmax(2)
    # select all columns with x and y positions in the tracking data
    columns = [c for c in tracking_df.columns if c[-1].lower() in ['x', 'y']]
    # flip the tracking data by multiplying by -1
    tracking_df.loc[second_half_idx:, columns] = - 1 * tracking_df.loc[second_half_idx:, columns]
    return tracking_df


def preprocessing_tracking(tracking_df):
    """
    Preprocesses the tracking data frame, by changing the time values of the second have,
    splitting the DataFrame up in a home and away DataFrame. Changes the tracking data,
    so that each team plays in one direction the entire game and calculates the velocities of the players.

    Parameters
    -----------
        tracking_df: tracking DataFrame
    Returns
    -----------
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
    """
    # change time values and split the tracking data
    tracking_df = change_time_values_tracking(tracking_df)
    tracking_home, tracking_away = split_tracking_data(tracking_df)

    # reverse direction of play in the second half so that home team is always attacking from right->left
    tracking_home = change_to_single_playing_direction(tracking_home)
    tracking_away = change_to_single_playing_direction(tracking_away)

    # calculate the player velocities
    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)
    return tracking_home, tracking_away


# --- EVENTS SECTION ---

def select_columns_of_interest(events):
    """
    Select the columns that are necessary for the analysis

    Parameters
    -----------
        events: events DataFrame
    Returns
    -----------
        events: events DataFrame with only the selected columns
    """

    columns = ['home_or_away',
               'EventName', 'player_name', 'Time', 'IdActor1', 'IdActor2',
               'LocationX', 'LocationY', 'team_name',
               'BodyPart', 'Behaviour',
               'DuelWinner']
    events = events.loc[:, columns]
    return events


def change_to_metrica_sports_format(events):
    """
    transforms the events dataframe to the Metrica sports format,
    so that we can use it to calculate the EPV values of the transition passes

    Parameters
    -----------
        events: events DataFrame
    Returns
    -----------
        events: events DataFrame in the Metrica sports format
    """

    events.columns = ['Team', 'Type', 'player_name', 'Time', 'From',
                      'IdActor2', 'Start X', 'Start Y', 'team_name', 'BodyPart',
                      'Behaviour', 'DuelWinner', 'Time [s]',
                      'Start Frame', 'Period', 'next_event', 'End X', 'End Y', 'End Frame', 'End Time [s]',
                      'Possession Change', 'Evaluation', 'To']

    cols = ['Team', 'Type', 'Possession Change', 'Period', 'Time', 'Start Frame', 'Time [s]',
            'End Frame', 'End Time [s]',
            'From', 'To', 'Start X', 'Start Y', 'End X', 'End Y', 'BodyPart',
            'Behaviour', 'DuelWinner', 'IdActor2', 'next_event',
            'Evaluation', 'player_name', 'team_name']

    events.loc[events['Team'] == 'home', 'Team'] = 'Home'
    events.loc[events['Team'] == 'away', 'Team'] = 'Away'
    events = events[cols]
    return events


def add_start_time_columns(events):
    """
    Adds start time columns to the events DataFrame and adds a Period column.

    Parameters
    -----------
        events: events DataFrame
    Returns
    -----------
        events: events DataFrame including the extra columns
    """

    # Change time format from milliseconds to seconds
    time_secs = (events.loc[:, 'Time'] / 1000)
    events.loc[:, 'Time [s]'] = time_secs.round(decimals=2)

    # Sync Start Frame with tracking id, thus 10 frames per second for tracking data:
    events.loc[:, 'Start Frame'] = (
            events['Time [s]'] * 10).round().astype(int)

    # add period
    start_time_second_half = events[events['EventName'] == 'HalfTime Start'].iloc[1]['Time [s]']
    events.loc[events['Time [s]'] < start_time_second_half, 'Period'] = 1
    events.loc[events['Time [s]'] >= start_time_second_half, 'Period'] = 2
    return events


def add_values_based_on_next_event(events, event_id):
    """
    Adds values based on the next event values. Event name and end location of for instance passes.

    Parameters
    -----------
        events: events DataFrame
        event_id: event index
    Returns
    -----------
        events: events DataFrame including the extra values
    """

    next_row = events.iloc[event_id + 1]
    # add next event type
    events.loc[event_id, 'next_event'] = next_row['EventName']

    # add end locations
    events.loc[event_id, 'End X'] = next_row['LocationX']
    events.loc[event_id, 'End Y'] = next_row['LocationY']
    return events


def add_possession_change(events, event_id):
    """
    Adds possession change value to the current row. If possession changes than 1, else 0

    Parameters
    -----------
        events: events DataFrame
        event_id: event index
    Returns
    -----------
        events: events DataFrame including the extra values
    """
    # add Possession_Change
    if event_id > 0:
        if (events.loc[event_id - 1, 'home_or_away'] != events.loc[event_id, 'home_or_away']) & (
                isinstance(events.loc[event_id - 1, 'home_or_away'], str)):
            events.loc[event_id, 'Possession Change'] = 1
        else:
            events.loc[event_id, 'Possession Change'] = 0
    return events


def add_values_dependent_on_success_of_pass(events, row, event_id):
    """
    Adds values to the current row based on the success of the pass. 

    Parameters
    -----------
        events: events DataFrame
        row: row corresponding to the passing event
        event_id: index corresponding to the passing event
    Returns
    -----------
        events: events DataFrame including the extra values:
            - Evaluation: 'successful' or 'unsuccessful'
            - To: player id of the receiving player if a successful pass
            - End Frame: Equal to the frame of the next event
            - End Time [s]: Equal to the time in seconds of the next event
    """

    list_next_events_for_successful_pass = ['Pass', 'Reception', 'Running with ball',
                                            'Cross', 'Shot on target', 'Clearance uncontrolled', 'Chance',
                                            'Goal', 'Hold of ball', 'Neutral clearance save',
                                            'Shot not on target']

    # We qualify a pass as a successful pass if:
    # next event is by a player of the same team
    # current event has an end position
    # next event is equal to one in the list above
    next_row = events.iloc[event_id + 1]
    if (row.EventName == 'Pass') | (row.EventName == 'Cross'):
        if row.home_or_away == next_row.home_or_away:
            if not math.isnan(events.loc[event_id, 'End X']):
                if events.loc[event_id + 1, 'EventName'] in list_next_events_for_successful_pass:
                    events.loc[event_id, 'Evaluation'] = 'successful'
                    # add receiving player
                    events.loc[event_id, 'To'] = str(next_row['IdActor1'])
                else:
                    events.loc[event_id, 'Evaluation'] = 'unsuccessful'
        events.loc[event_id, 'End Frame'] = round(next_row['End Time [s]'] * 10)
        events.loc[event_id, 'End Time [s]'] = next_row['Time [s]']
    else:
        events.loc[event_id, 'End Frame'] = round((row['Time [s]'] * 10))
        events.loc[event_id, 'End Time [s]'] = row['Time [s]']
    return events


def preprocessing_events(events):
    """
    Preprocesses the parsed events dataframe, by creating new values:
     - Start Frame: start frame of event
     - End Frame: end frame of event, if there is any
     - Time [s]: start time in seconds of event
     - End Time: end time in seconds of event
     - End X: X position of the end location of the event
     - End Y, Y position of the end location of the event
     - Possession Change: if this event changes the possession the value is 1, else 0
     - To: Receiver of the pass
     - Evaluation: whether a pass is successful or not

    Parameters
    -----------
        events: events DataFrame

    Returns
    -----------
       events: events DataFrame with the above mentioned extra columns
    """

    events = select_columns_of_interest(events)
    events = add_start_time_columns(events)
    events = change_to_single_playing_direction(events)

    # change from cm to meters:
    events.loc[:, 'LocationX'] = events['LocationX'] / 100
    events.loc[:, 'LocationY'] = events['LocationY'] / 100

    events['IdActor1'] = events['IdActor1'].astype(str)
    events['IdActor2'] = events['IdActor2'].astype(str)

    # loop through dataframe to generate / change certain values
    for i, row in events.iterrows():
        if i < (len(events) - 1):
            events = add_values_based_on_next_event(events, i)
            events = add_possession_change(events, i)
            events = add_values_dependent_on_success_of_pass(events, row, i)
    events = change_to_metrica_sports_format(events)
    return events


def get_first_pass_after_transition(events, event_id):
    """
    Gets the first pass event within the next 2 events and within 5 seconds of the transition
    # Aiming to accommodate for duels and dribbles within the following seconds of the transition

    Parameters
    -----------
        events: events DataFrame
        event_id: index corresponding to the event
    Returns
    -----------
        output_row: the events row of the found transition pass | None
        index: index of the events row | None
    """
    for i in range(1, 4):
        index = event_id + i
        if (events.loc[index, 'Type'] == 'Pass') & (events.loc[index, 'Possession Change'] == 0):
            # Check whether row has end location
            if not (math.isnan(events.loc[index, 'End X'])):
                time_difference = events.loc[index,
                                             'Time [s]'] - events.loc[event_id, 'Time [s]']
                is_within_time = time_difference < 5
                if is_within_time:
                    output_row = events.loc[index].copy()
                    output_row['time_after_transition'] = time_difference
                    return output_row, index
        # if the team loses the ball in an event, before we find a pass event, return to main function
        if events.loc[index, 'Possession Change'] != 0:
            return None, None
    return None, None


def get_transition_passes(events):
    """
    Extracts all transition passes.
    A transition pass is defined by the first pass within 5 seconds after a transition in open-play.
    Adds a new column called 'time_after_transition', which is the time in seconds the pass took place after the
    transitioning moment.

    Parameters
    -----------
        events: events DataFrame
    Returns
    -----------
       events: events dataframe of all passes after transitions
    """

    column_list = list(events.columns)
    column_list.append('time_after_transition')
    transition_passes_df = pd.DataFrame(columns=column_list)
    for i, row in events.iterrows():
        if i > 1:
            # Check whether it is a transition moment
            if row['Possession Change'] == 1:
                # Check whether pass is a kick off
                if not ((events.loc[i - 1, 'Type'] == 'HalfTime Start') or (events.loc[i - 1, 'Type'] == 'Goal')):
                    # Check whether a pass
                    if (row['Type'] == 'Pass') | (row['Type'] == 'Cross'):
                        # Check whether row has end location
                        if not (math.isnan(events.loc[i, 'End X'])):
                            events.loc[i, 'time_after_transition'] = 0
                    if row['Type'] in ['Reception', 'Hold of ball', 'Running with ball']:
                        result, index = get_first_pass_after_transition(
                            events, i)
                        if isinstance(result, pd.Series):
                            transition_passes_df.loc[index] = result


def preprocess_and_save_match_data(filename, folder):
    """
    Preprocesses tracking and events DataFrames. Generates a transition passes DataFrame,
    which conforms to the selected characteristics of a transition pass. And saves all four files
    Saves home

    Parameters
    -----------
        filename: the xml filename of the current match
        folder: folder path with data files
    """
    # create paths of filenames
    base_file = filename.split('.xml')[0]
    tracking_file = base_file + '_tracking.csv'
    events_file = base_file + '_events.csv'
    tracking_path = folder + '\\' + tracking_file
    events_path = folder + '\\' + events_file
    preprocessed_tracking_home_path = f'{folder}\\preprocessed\\{base_file}_tracking_home_processed.csv'
    preprocessed_tracking_away_path = f'{folder}\\preprocessed\\{base_file}_tracking_away_processed.csv'
    preprocessed_events_path = f'{folder}\\preprocessed\\{base_file}_events_processed.csv'
    transition_passes_path = f'{folder}\\transition_passes\\{base_file}_transition_passes.csv'

    # read in data:
    tracking_df = pd.read_csv(tracking_path, index_col=0)
    events_df = pd.read_csv(events_path, index_col=0)
    # preprocess data
    events = preprocessing_events(events_df)
    tracking_home, tracking_away = preprocessing_tracking(tracking_df)

    # save preprocessed dataframes to csv
    tracking_home.to_csv(preprocessed_tracking_home_path)
    tracking_away.to_csv(preprocessed_tracking_away_path)
    events.to_csv(preprocessed_events_path)

    # select and save transition_passes
    transition_passes = get_transition_passes(events)
    transition_passes.to_csv(transition_passes_path)


def preprocess():
    """
    Preprocesses the parsed data to conform to Metrica Sports format (also adds multiple columns.
     To allow for EPV calculations. And selects transition passes from the events DataFrame
    """
    folder = os.path.join(DIRNAME, 'data')  # same as comment above
    directory = os.fsencode(folder + r'\xml_files')
    files = os.listdir(directory)
    for file in tqdm(files, total=len(files), position=0):
        filename = os.fsdecode(file)
        preprocess_and_save_match_data(filename, folder)
