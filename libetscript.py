import os

import pandas as pd
import numpy as np
from collections import defaultdict
import csv


class LibetData():
    """
    LibetData is used to extract the Libet related information per trial
    `Trial` is defined as one Libet clock and a `round` is 10-12 consecutive trials stored together
    """

    def __init__(self, root_directory, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.root_directory = root_directory



    def path_exp_libet(self):
        """
        This returns a dictionary of all the file paths, separated by participant, relating to the Libet experiment
        information (i.e. W-time and W-angle etc.).

        :param str root_directory: The file path to the folder within which all the folders are stored.

        :return: a dictionary with all the `Libet experiment information` files as a list of values, with the
                participant code as the key
        :return type: dict
        """

        if not isinstance(self.root_directory, str):
            raise ValueError("Input to `root_directory` should indicate the file path to the folder where all the "
                             "experiment data files are situated")

        subject_path_list, eye_csv_list = [], []
        dict_libetpaths = defaultdict(list)

        for path in os.listdir(self.root_directory):
            subject_path = os.path.join(self.root_directory, path)
            subject_path_list.append(subject_path)
            if os.path.isdir(subject_path):
                for subject in os.listdir(subject_path):
                    if not subject.endswith("eyedata.csv") and not subject.endswith("triggers.csv"):
                        libet_subject = os.path.join(subject_path, subject)
                        dict_libetpaths[path].append(libet_subject)

        return dict_libetpaths

    def access_data(self, participant_code, round_id):
        """
        This function saves the data to a dataframe

        :param str participant_code: The code for the participant under investigation
        :param str root_directory: The file path to the folder within which all the folders are stored.
        :param str round_id: The experiment was broken up into 10-12 rounds, each of which consisted of 11 trials (one
                            trial = one Libet clock)


        """
        if not isinstance(self.root_directory, str):
            raise ValueError("Input to `root_directory` should indicate the file path to the folder where all the "
                             "experiment data files are situated")
        if not isinstance(participant_code, str):
            raise ValueError("Input to `participant_code` should indicate the participant of interest and be of type "
                             "string")
        if not isinstance(round_id, str):
            raise ValueError("Input to `round_id` should indicate the round of interest and be of type "
                             "string")

        for path in os.listdir(self.root_directory):
            if path == participant_code:
                subject_path = os.path.join(self.root_directory, path)
                for file in os.listdir(subject_path):
                    if file.endswith(participant_code + round_id + '.csv'):
                        libet_df = pd.read_csv(os.path.join(subject_path, file),
                                               delimiter=';', header=0)
                        libet_df["{}{}".format(participant_code, round_id)] = ""

        return libet_df

    # def w_time_round(self, participant_code, round_id):
    #     """
    #     This function loops through a pandas dataframe and calculates the average "W" time (i.e. the average moment of
    #     subjective conscious awareness as reported by the participant
    #
    #     :param pandas.DataFrame libet_df: dataframe containing the libet information for a specific round for a specific
    #                                       participant
    #     """
    #
    #     for path in os.listdir(self.root_directory):
    #         if path == participant_code:
    #             subject_path = os.path.join(self.root_directory, path)
    #             for file in os.listdir(subject_path):
    #                 if file.endswith(participant_code + round_id + '.csv'):
    #                     libet_df = pd.read_csv(os.path.join(subject_path, file),
    #                                            delimiter=';', header=0)
    #     temp_df = libet_df.drop(libet_df[libet_df.loc['userError'] == True.index)
    #     print(temp_df)
