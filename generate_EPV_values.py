#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun July 28 09:10:58 2020

@author: Milan
"""

import pandas as pd
from tqdm import tqdm
import os

DIRNAME = os.path.dirname(__file__)
DATADIR = os.path.join(DIRNAME, 'EPV_code')
os.chdir(DATADIR)
import Metrica_EPV as mepv
import Metrica_PitchControl as mpc
import Metrica_IO as mio


def generate_epv_df(events, tracking_home, tracking_away, gk_numbers, epv, params):
    """
    Calculates the EPV and optimal EPV values for an input DataFrame of passes.

    Parameters
    -----------
        events: transition passes DataFrame
        tracking_home: tracking DataFrame of home team
        tracking_away: tracking DataFrame of away team
        gk_numbers: list of the two goalkeeper player ids string
        epv: EPV grid
        params: pitch control model parameters
    Returns
    -----------
       events: transition passes DataFrame + columns:
       - eepv_added: Expected EPV value-added of pass defined by event_id
       - epv_difference: The raw change in EPV (ignoring pitch control) between end and start points of pass
       - max_eepv_added: Maximum possible Expected EPV value-added
       - epv_target_x: X target location of the maximum possible EPV pass
       - epv_target_y: Y target location of the maximum possible EPV pass
    """
    for row in tqdm(events[:5].itertuples(),
                    total=len(events[:5]), position=0):
        i = row.Index
        eepv_added, epv_diff, eepv_start, eepv_target, epv_target, epv_start = mepv.calculate_epv_added(i, events,
                                                                                                        tracking_home,
                                                                                                        tracking_away,
                                                                                                        gk_numbers, epv,
                                                                                                        params)
        max_eepv_added, target = mepv.find_max_value_added_target(i, events, tracking_home,
                                                                  tracking_away, gk_numbers, epv, params)
        # expected EPV data
        events.loc[i, 'EEPV_start'] = eepv_start
        events.loc[i, 'EEPV_target'] = eepv_target
        events.loc[i, 'max_EEPV_added'] = max_eepv_added
        events.loc[i, 'EEPV_added'] = eepv_added

        # EPV data
        events.loc[i, 'EPV_start'] = epv_start
        events.loc[i, 'EPV_target'] = epv_target
        events.loc[i, 'EPV_diff'] = epv_diff
        events.loc[i, 'EPV_target_x'] = target[0]
        events.loc[i, 'EPV_target_y'] = target[1]

    return events


def calculate_epv_events_per_match(base_file, folder):
    """
    Calculates the EPV and optimal EPV values for an input DataFrame of passes.

    Parameters
    -----------
        base_file: input string corresponding to the match CSVs, from which we read the data
        folder: folder path
    Returns
    -----------
       EPV_df: is the original events DataFrame with only the transition passes
       and includes the outcomes of the EPV calculations
    """
    # make path string
    preprocessed_tracking_home_path = f'{folder}\\preprocessed\\{base_file}_tracking_home_processed.csv'
    preprocessed_tracking_away_path = f'{folder}\\preprocessed\\{base_file}_tracking_away_processed.csv'
    transition_passes_path = f'{folder}\\transition_passes\\{base_file}_transition_passes.csv'

    # load data
    tracking_home = pd.read_csv(preprocessed_tracking_home_path, index_col=0)
    tracking_away = pd.read_csv(preprocessed_tracking_away_path, index_col=0)
    events = pd.read_csv(transition_passes_path, index_col=0)

    # select only transition passes with an origin in the midfield:
    events = events[(events['Start X'] > -17.5) & (events['Start X'] < 17.5)]

    """ *** UPDATES TO THE MODEL: OFFSIDES """
    # first get pitch control model parameters
    params = mpc.default_model_params()
    # find goalkeepers for offside calculation
    gk_numbers = [
        mio.find_goalkeeper(tracking_home),
        mio.find_goalkeeper(tracking_away)]
    # get EPV surface
    epv = mepv.load_EPV_grid(DATADIR + '/EPV_grid.csv')
    # generate EPV values and append to events DataFrame
    epv_df = generate_epv_df(
        events,
        tracking_home,
        tracking_away,
        gk_numbers,
        epv,
        params)
    return epv_df


def generate_epv_files():
    """
    loops through all matches and their corresponding transition passes.
    Calculates the EPV values for the transition passes and saves to a new csv file per match.
    """
    folder = os.path.join(DIRNAME, 'data')
    directory = os.fsencode(folder + r'\xml_files')
    files = os.listdir(directory)
    for file in tqdm(files, total=len(files), position=0):
        filename = os.fsdecode(file)
        base_file = filename.split('.xml')[0]
        epv_df = calculate_epv_events_per_match(base_file, folder)
        epv_df_path = folder + '\\transition_passes\\' + base_file + '_EPV_df.csv'
        epv_df.to_csv(epv_df_path)



