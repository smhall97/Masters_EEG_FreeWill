import os
import csv
import math

import pandas as pd
import numpy as np

from collections import defaultdict


def evaluate_tuple(gaze_values):
    """
    The Tobii device saves the left and right gaze data as a set of co-ordinates in a tuple: (x,y)
    This function evaluates these tuples, removing the brackets and return the two co-ordinate values separately

    :param tuple gaze_values: Tuple extracted from the csv

    :return: two float values corresponding to the x and y values
    :return type: float, float
    """

    x, y = gaze_values.strip('()').split(',')
    return float(x), float(y)


class EyeTrackingData():
    """
    `EyeTrackingData` is used to extract the necessary information from the .csv files recorded during the Libet
    experiment. The code for the Libet experiment, with eye tracking can be found at:
    """

    def __init__(self, root_directory, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.root_directory = root_directory

    def path_exp_eyedata(self):
        """
        This returns a dictionary of all the file paths, separated by participant, relating to the eye data (viz.
        system time; pc time; (L) gaze; gaze co-ordinates [x,y]; (R) gaze co-ordinates [x,y]; (L) pupil dilation)

        :param str root_directory: The file path to the folder within which all the folders are stored. Each folder
                                    should correspond to a collection of each participants' data.

        :return: a dictionary with all the `eye data` files as a list of values, with the participant code as the key
        :return type: dict
        """

        if not isinstance(self.root_directory, str):
            raise ValueError("Input to `root_directory` should indicate the file path to the folder where all the "
                             "experiment data files are situated")

        subject_path_list, eye_csv_list = [], []
        dict_eyedata = defaultdict(list)

        for path in os.listdir(self.root_directory):
            subject_path = os.path.join(self.root_directory, path)
            subject_path_list.append(subject_path)
            if os.path.isdir(subject_path):
                for subject in os.listdir(subject_path):
                    if subject.endswith("eyedata.csv"):
                        eye_subject = os.path.join(subject_path, subject)
                        dict_eyedata[path].append(eye_subject)

        return dict_eyedata

    def path_exp_triggers(self, participant_code):
        """
        This returns a dictionary of all the file paths, separated by participant, relating to the triggers per trial.
        These are timestamped to system time to correlate with the system time as recorded in the eye data files

        :param str root_directory: The file path to the folder within which all the folders are stored.
        :param str participant_code: The code for the participant under investigation

        :return: a dictionary with all the `trigger information` files as a list of values, with the participant code
                as the key
        :return type: dict
        """

        if not isinstance(self.root_directory, str):
            raise ValueError("Input to `root_directory` should indicate the file path to the folder where all the "
                             "experiment data files are situated")
        if not isinstance(participant_code, str):
            raise ValueError("Input to `participant_code` should indicate the participant of interest and be of type "
                             "string")

        subject_path_list, eye_csv_list = [], []
        dict_triggers_path = defaultdict(list)

        for path in os.listdir(self.root_directory):
            if path == participant_code:
                subject_path = os.path.join(self.root_directory, path)
                subject_path_list.append(subject_path)
                if os.path.isdir(subject_path):
                    for subject in os.listdir(subject_path):
                        if subject.endswith("triggers.csv"):
                            eye_subject = os.path.join(subject_path, subject)
                            dict_triggers_path[path].append(eye_subject)

        return dict_triggers_path

    def trigger_trials(self, participant_code, round_id):
        """
        This returns a pandas dataframe of specific trials relating to a participant code and round id
        These are timestamped (system time) to mark the start ("s") and end ("l"/"r") of each trial. These timestamps
        correlate with the system time as recorded in the eye data files

        :param str participant_code: The code for the participant under investigation
        :param str round_id: The experiment was broken up into 10-12 rounds, each of which consisted of 11 trials (one
                            trial = one Libet clock)

        :return: a pandas with all the timestamps signalling the start and end of each trial
        :return type: pandas.DataFrame
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
                print(subject_path)
                if os.path.isdir(subject_path):
                    for file in os.listdir(subject_path):
                        round = file.split("_")
                        if file.endswith("triggers.csv") and round[1] == (path + round_id):
                            trigger_df = pd.read_csv(os.path.join(subject_path, file),
                                                     delimiter=';', header=None)
                            trigger_df.columns = ["system_time", "start_condition", "trial_number"]
                            trigger_df["{}{}".format(participant_code, round_id)] = ""

        return trigger_df

    def trigger_time(self, participant_code, round_id):
        """
        This returns a pandas dataframe of specific trials relating to a participant code and round id
        These are timestamped (system time) to mark the start ("s") and end ("l"/"r") of each trial. These timestamps
        correlate with the system time as recorded in the eye data files

        :param str participant_code: The code for the participant under investigation
        :param str round_id: The experiment was broken up into 10-12 rounds, each of which consisted of 11 trials (one
                            trial = one Libet clock)

        :return: a pandas with all the timestamps signalling the start and end of each trial and list of each indices
                corresponding to the different conditions
        :return type: pandas.DataFrame, list, list, list
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
                print(subject_path)
                if os.path.isdir(subject_path):
                    for file in os.listdir(subject_path):
                        round = file.split("_")
                        if file.endswith("triggers.csv") and round[1] == (path + round_id):
                            trigger_df = pd.read_csv(os.path.join(subject_path, file),
                                                     delimiter=';', header=None)
                            trigger_df.columns = ["system_time", "start_condition", "trial_number"]

        start = np.where(trigger_df["start_condition"] == 's')
        start_index = start[0].tolist()
        left = np.where(trigger_df["start_condition"] == 'l')
        left_index = left[0].tolist()
        right = np.where(trigger_df["start_condition"] == 'r')
        right_index = right[0].tolist()

        return trigger_df, start_index, left_index, right_index

    def trial_data(self, participant_code, round_id):
        """
        This parses the csv file into the different data points of interest and stores them in a data frame.
        The data points of interest are: system time; pc time; (L) gaze co-ordinates [x,y]; (R) gaze co-ordinates [x,y];
        (L) pupil dilation (change over time, with the units specific to Tobii's internal metrics); (R) pupil dilation

        :param str root_directory: The file path to the folder within which all the folders are stored.
        :param str participant_code: The participant of interest
        :param str round_id: The experiment was broken up into 10-12 rounds, each of which consisted of 11 trials (one
                    trial = one Libet clock)

        :return: a pandas with all the timestamps signalling the start and end of each trial
        :return type: pandas.DataFrame
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
                if os.path.isdir(subject_path):
                    for file in os.listdir(subject_path):
                        round = file.split("_")
                        if file.endswith("eyedata.csv") and round[1] == (path + round_id):
                            trial_data_df = pd.read_csv(os.path.join(subject_path, file),
                                                        delimiter=';', header=None)
                            trial_data_df.columns = ['system_time', 'pc_time', 'l_gaze', 'r_gaze', 'l_pupil', 'r_pupil']
                            trial_data_df["{}{}".format(participant_code, round_id)] = ""

        return trial_data_df
