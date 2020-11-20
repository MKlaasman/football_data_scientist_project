"""
Created on Wed Jul 22 15:18:32 2020

@author: Milan
"""

import os
import xml.etree.ElementTree as et
import numpy as np
import pandas as pd
from tqdm import tqdm

COLUMN_NAMES = ['Time', 'IdActor1', 'IdActor2', 'EventName', 'BodyPart', 'Behaviour',
                'DuelType', 'DuelBrutality', 'DuelInitiative', 'DuelWinner',
                'LocationX', 'LocationY', 'LocationZ', 'TargetX', 'TargetY', 'TargetZ',
                'HitsPost', 'Blocked', 'PhaseType', 'PhaseStartTime', 'PhaseEndTime',
                'PhaseSubType', 'StartOfPlay', 'BreakThroughBall', 'AssistType',
                'BallCurve', 'TechnicalCharacteristics', 'FoulReason',
                'GKHeightOfIntervention', 'DuelOutcome', 'ScoreHomeTeam',
                'ScoreAwayTeam', 'RedCardsHomeTeam', 'RedCardsAwayTeam']


def get_player_lists_single_team(selected_game_file):
    """
    Gets all player ids and names from the one team.
    -----------
       game_file: game file from ElementTree XML corresponding to one match
    Returns
    -----------
       player_list: list of all player ids
       player_list_names: list of all player names
    """
    player_list = []
    player_list_names = []
    for player in selected_game_file:
        player_list.append(player.attrib.get('IdActor'))
        # player name consists of 'UsualFirstName' (first name) and 'NickName'
        # (last name or nick name)
        player_name = player.attrib.get(
            'UsualFirstName') + ' ' + player.attrib.get('NickName')
        player_list_names.append(player_name)
        return player_list, player_list_names


def get_player_lists(game_file):
    """
    Gets all player ids and names from the game file and saves them in lists
    -----------
       game_file: game file from ElementTree XML corresponding to one match
    Returns
    -----------
       player_list_home: list of all Home player ids
       player_list_away: list of all Away player ids
       player_list_home_names: list of all Home player names
       player_list_away_names: list of all Away player names
    """

    # get the player lists for both teams 
    player_list_home, player_list_home_names = get_player_lists_single_team(game_file[0][0][:])
    player_list_away, player_list_away_names = get_player_lists_single_team(game_file[0][1][:])
    return player_list_home, player_list_away, player_list_home_names, player_list_away_names


def transform_to_int(string):
    """
    Transforms a string to an int and handles empty strings.

    Returns
    -----------
       output: either integer or NaN
    """
    if string == '':
        return np.nan
    return int(string)


def parse_events_data(game_file):
    """
    Parses the first and second half of events data from the xml file.
    Did not include extra time in this analysis.
    Also parsed elements I do not need think are needed for the analysis,
    however they could be of interest for the additional analysis section.

    Parameters
    -----------
       game_file: game file from ElementTree XML corresponding to one match
    Returns
    -----------
       events: events DataFrame
    """

    # chose the same column_names as in the xml file
    Time = []
    IdActor1 = []
    IdActor2 = []
    EventName = []
    BodyPart = []
    Behaviour = []
    DuelType = []
    DuelBrutality = []
    DuelInitiative = []
    DuelWinner = []
    LocationX = []
    LocationY = []
    LocationZ = []
    TargetX = []
    TargetY = []
    TargetZ = []
    HitsPost = []
    Blocked = []
    PhaseType = []
    PhaseStartTime = []
    PhaseEndTime = []
    PhaseSubType = []
    StartOfPlay = []
    BreakThroughBall = []
    AssistType = []
    BallCurve = []
    TechnicalCharacteristics = []
    FoulReason = []
    GKHeightOfIntervention = []
    DuelOutcome = []
    ScoreHomeTeam = []
    ScoreAwayTeam = []
    RedCardsHomeTeam = []
    RedCardsAwayTeam = []
    first_half_end_time = int(game_file[1][0][-1].attrib.get('Time'))
    # loop through the game file and create lists with events data
    for half in range(2):
        for i in range(len(game_file[1][half][:])):
            event_data = game_file[1][half][i].attrib
            # add time to dataframe, if second half than add the first half end
            # time to the time
            if half == 1:
                Time.append(transform_to_int(
                    event_data.get("Time")) +
                            first_half_end_time)
            else:
                Time.append(transform_to_int(event_data.get("Time")))

            EventName.append(event_data.get("EventName"))
            BodyPart.append(event_data.get("BodyPart"))
            Behaviour.append(event_data.get("Behaviour"))
            DuelType.append(event_data.get("DuelType"))
            DuelBrutality.append(event_data.get("DuelBrutality"))
            DuelInitiative.append(event_data.get("DuelInitiative"))
            DuelWinner.append(event_data.get("DuelWinner"))
            HitsPost.append(event_data.get("HitsPost"))
            Blocked.append(event_data.get("Blocked"))
            PhaseType.append(event_data.get("PhaseType"))
            PhaseStartTime.append(transform_to_int(event_data.get("PhaseStartTime")))
            PhaseEndTime.append(transform_to_int(event_data.get("PhaseEndTime")))
            PhaseSubType.append(event_data.get("PhaseSubType"))
            StartOfPlay.append(event_data.get("StartOfPlay"))
            BreakThroughBall.append(event_data.get("BreakThroughBall"))
            AssistType.append(event_data.get("AssistType"))
            BallCurve.append(event_data.get("BallCurve"))
            TechnicalCharacteristics.append(event_data.get("TechnicalCharacteristics"))
            FoulReason.append(event_data.get("FoulReason"))
            GKHeightOfIntervention.append(event_data.get("GKHeightOfIntervention"))
            DuelOutcome.append(event_data.get("DuelOutcome"))
            RedCardsHomeTeam.append(event_data.get("RedCardsHomeTeam"))
            RedCardsAwayTeam.append(event_data.get("RedCardsAwayTeam"))

            # for the following using transform to int function within append function:
            IdActor1.append(str(transform_to_int(event_data.get("IdActor1"))))
            IdActor2.append(str(transform_to_int(event_data.get("IdActor2"))))
            LocationX.append(transform_to_int(event_data.get("LocationX")))
            LocationY.append(transform_to_int(event_data.get("LocationY")))
            LocationZ.append(transform_to_int(event_data.get("LocationZ")))
            TargetX.append(transform_to_int(event_data.get("TargetX")))
            TargetY.append(transform_to_int(event_data.get("TargetY")))
            TargetZ.append(transform_to_int(event_data.get("TargetZ")))
            ScoreHomeTeam.append(transform_to_int(event_data.get("ScoreHomeTeam")))
            ScoreAwayTeam.append(transform_to_int(event_data.get("ScoreAwayTeam")))

    # Create dataframe from lists
    data = [Time, IdActor1, IdActor2, EventName, BodyPart, Behaviour,
            DuelType, DuelBrutality, DuelInitiative, DuelWinner, LocationX, LocationY,
            LocationZ, TargetX, TargetY, TargetZ, HitsPost, Blocked, PhaseType, PhaseStartTime,
            PhaseEndTime, PhaseSubType, StartOfPlay, BreakThroughBall, AssistType, BallCurve,
            TechnicalCharacteristics, FoulReason, GKHeightOfIntervention, DuelOutcome, ScoreHomeTeam,
            ScoreAwayTeam, RedCardsHomeTeam, RedCardsAwayTeam]

    events = pd.DataFrame(data=data, index=COLUMN_NAMES).T
    return events


def parse_one_half_of_tracking(tracking_half, half, player_list_home):
    """
    Parses one half of tracking data to a DataFrame.

    Parameters
    -----------
       tracking_half: game_file data of either half 1 or half 2
       half: integer, either half 1 or half 2
       player_list_home: list of all Home player ids

    Returns
    -----------
       tracking: tracking DataFrame with positional data of both teams
    """

    ball_time = []
    ball_x = []
    ball_y = []
    ball_z = []
    tracking_time = []
    period = []
    is_first = True

    # This loop makes sure that we give each player the correct column name corresponding to their team.
    # for this reason it checks whether the current player is part of the home team (player_list_home),
    # or the away team(player_list_away).
    # If it is the first player we add the Period and Time to the
    # corresponding DataFrame.
    for i in tqdm(range(len(tracking_half)), total=len(tracking_half)):
        # if current player in home team
        current_player = tracking_half[i].attrib.get('IdActor')
        team = 'home' if current_player in player_list_home else 'away'

        # is ball, than append the data to lists
        if tracking_half[i].attrib.get('IsBall') == 'True':
            for j in range(int(tracking_half[i].attrib.get('NbPoints'))):
                ball_time.append(tracking_half[i][j].attrib.get('T'))
                ball_x.append(tracking_half[i][j].attrib.get('X'))
                ball_y.append(tracking_half[i][j].attrib.get('Y'))
                ball_z.append(tracking_half[i][j].attrib.get('Z'))

        # otherwise, we get the x and y data of the player and concatenate to existing DataFrame.
        # If it is the first player we add the Period and Time and create a new
        # DataFrame
        else:
            x = []
            y = []
            for j in range(int(tracking_half[i].attrib.get('NbPoints'))):
                x.append(tracking_half[i][j].attrib.get('X'))
                y.append(tracking_half[i][j].attrib.get('Y'))
                period.append(half)
                if is_first:
                    tracking_time.append(tracking_half[i][j].attrib.get('T'))

            if is_first:
                data = [period, tracking_time, x, y]
                column_names = ['Period', 'Time [s]', f"{team}_{current_player}_x", f"{team}_{current_player}_y"]
                tracking = pd.DataFrame(data=data, index=column_names).T
                is_first = False
            else:
                column_names = [
                    team + '_' + current_player + '_x',
                    team + '_' + current_player + '_y'
                ]
                data = [x, y]
                temp_df = pd.DataFrame(data=data, index=column_names).T
                tracking = pd.concat(
                    [tracking, temp_df], axis=1, sort=False)

    # create a ball DataFrame
    column_names = ['Time [s]', 'ball_x', 'ball_y']
    data = [ball_time, ball_x, ball_y]
    ball_df = pd.DataFrame(data=data, index=column_names).T
    # merge ball and player tracking data
    tracking = pd.concat([tracking, ball_df], axis=1, keys='Time [s]', sort=True)
    tracking = tracking.drop(tracking.columns[-3], axis=1)
    return tracking


def parse_tracking_data(current_game_file):
    """
    Parses the tracking data from the xml file by parsing each half separately and
    then adding them together and changing the timestamp.

    Parameters
    -----------
       current_game_file: game file from ElementTree XML corresponding to one match
    Returns
    -----------
       tracking_df: tracking DataFrame with positional data of both teams
    """
    tracking_half1 = current_game_file[-1][0][:]
    tracking_half2 = current_game_file[-1][1][:]

    player_list_home, a, b, c = get_player_lists(current_game_file)

    tracking_df1 = parse_one_half_of_tracking(tracking_half1, 1, player_list_home)
    tracking_df2 = parse_one_half_of_tracking(tracking_half2, 2, player_list_home)

    # select dataframe for concatenation
    df1 = tracking_df1['T']
    df2 = tracking_df2['T']
    df_ball = tracking_df1['i'].copy()
    # add first half index to second half tracking data
    df2.index = df2.index + len(df1)
    # concatenate dataframes
    output_tracking_df = pd.concat([df1, df2], axis=0, ignore_index=True)
    output_tracking_df = pd.concat([output_tracking_df, df_ball], axis=1)
    return output_tracking_df


def add_team_names_and_player_names_to_events(current_game_file, events):
    """
    Adds team names and player to events DataFrame.

    Parameters
    -----------
        current_game_file: game file from ElementTree XML corresponding to one match
        events: events DataFrame
    Returns
    -----------
        events_df: events DataFrame with extra columns team_name and player_name
    """

    player_list_home, player_list_away, player_list_home_names, player_list_away_names = get_player_lists(
        current_game_file)
    # create main_info
    home_team_name = current_game_file[0][0].attrib.get('Name')
    away_team_name = current_game_file[0][1].attrib.get('Name')

    events['Team'] = np.nan
    events['team_name'] = np.nan
    for i, row in events.iterrows():
        if row.IdActor1 in player_list_home:
            events.loc[i, 'Team'] = 'home'
            events.loc[i, 'team_name'] = home_team_name
            # add player_names
            index = player_list_home.index(row.IdActor1)
            events.loc[i, 'player_name'] = player_list_home_names[index]
        if row.IdActor1 in player_list_away:
            events.loc[i, 'Team'] = 'away'
            events.loc[i, 'team_name'] = away_team_name
            # add player_names
            index = player_list_away.index(row.IdActor1)
            events.loc[i, 'player_name'] = player_list_away_names[index]

    cols = events.columns.tolist()
    # rearrange the columns
    cols = [cols[0]] + cols[-3:] + cols[1:-3]
    events = events[cols]
    return events


def parse_data():
    """
    Parses the xml data files in the same format (naming) as the xml file. Only selects first and second half data for
    each match and saves to tracking and events files in data folder
    """
    directory = os.fsencode(r'data\xml_files')

    for file in os.listdir(directory):
        # get the game_file to parse
        filename = os.fsdecode(file)
        tree = et.ElementTree(file='data/xml_files/' + filename)
        game_file = tree.getroot()

        # generate csv output strings
        tracking_filename = filename.split('.xml')[0] + '_tracking.csv'
        events_filename = filename.split('.xml')[0] + '_events.csv'

        # parse and write events data to csv
        events_df = parse_events_data(game_file)
        events_df = add_team_names_and_player_names_to_events(game_file, events_df)
        events_df.to_csv("data/" + events_filename)

        # write tracking data to csv
        tracking_df = parse_tracking_data(game_file)
        tracking_df.to_csv("data/" + tracking_filename)
