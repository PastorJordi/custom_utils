from scipy.signal import find_peaks
import seaborn as sns
from ast import literal_eval
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
# from concurrent.futures import as_completed, ProcessPoolExecutor
import concurrent.futures as confu
import tqdm
from utilsJ.Behavior.glm2afc import piped_moduleweight


# fixiing com detection / normalizing speed
# TODO: add all default params in __init__() so we can run the whole pipe with a single call (compipe)
# TODO: fix pipe for fixationbreak in trajectories.
# TODO: adapt to _feedback sessions (which may have inverted trials!)
# just dirty labelling missing ?
# TODO: decrease mem usage since working in parallel (7 or 8) can cause OOM | there are some arrays with shape (1e5, 1e6)
# they are boolean. Can be related to this strided rolling window
# TODO: finish: load_available(plot=True); describe_sessions()
# TODO: label my own delay trials
# TODO: add delay col so eventually we an plot various stuff vs delay in stim onset;
# TODO: class needs a through rework so all params/settings are set when instantiating object, then just expand 
#       kwargs on subsequent methods.

# GET ALL TRAJECTORIES from transitions (startsound and previous, to get correct fixation) :s; so we get the goddamn trajectories for all trials

# retrieve fb trajectories as well # apparently done
# add a method to get them like get_fb
# to do:
# when loading available sessions, add option to load sessions w/o vid but trajectories and framestamps
# fix teleports


class chom:
    """
    Class used to preprocess/transform join behavioral data (BPOD csvs) 
    and video coordinates (.hdf5 output from deep_lab_cut).
    """
    # file extensions to generalize across session base_name
    # pose_ext='DeepCut_resnet50_newsetup_general2019Jan07shuffle1_1030000.h5' # old DCNN
    pose_ext = "DeepCut_resnet50_metamix2019Jul03shuffle1_1030000.h5" # newer DCNN
    video_ext = ".avi"
    csv_ext = ".csv"

    # @staticmethod

    def rolling_window(a, size):
        """ unreadable function to speedup transition pattern matching 
        (eg for a correct fixation + response: port2In, tup, port2Out, portXIn)
        a: array like where pattern will be searched
        size: size of the pattern
        """
        shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    # @staticmethod
    # deprec.
    # def get_fpositive_seq(df, seq, idx):  # this will operate with strings
    #     window = len(seq)
    #     sub_seq = (
    #         df.iloc[idx:]
    #         .loc[(df.TYPE == "EVENT") | (df.TYPE == "TRANSITION"), "MSG"]
    #         .iloc[:window]
    #         .values
    #     )  # hell
    #     return np.all(sub_seq == seq)

    def polish_median(coordvec2d):
        """returns median after purging >2sd outliers, 
        coordvec2d: is a 2d coordinate vector (x,y) through time"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")  # , r'All-NaN (slice|axis) encountered')

            sd = coordvec2d.std(axis=0)
            median = np.median(coordvec2d, axis=0)
            to_dismiss = np.logical_or(
                (coordvec2d > median + (sd * 2))[:, 0],
                (coordvec2d > median + (sd * 2))[:, 1],
            )
            # triggers warnigs
            return np.median(
                np.delete(coordvec2d, to_dismiss, axis=0), axis=0
            )  # raises warning
        # FutureWarning: in the future insert will treat boolean arrays and array-likes as boolean index instead of casting it to integer

    def calculate_angle(point_a, point_b, c=(0, 0)):
        """ Calculate angle between two points (assuming vertex=origin"""
        ang_a = np.arctan2(*point_a[::-1])  # if 0,0 = 0
        ang_b = np.arctan2(*point_b[::-1])
        # return np.rad2deg((ang_a - ang_b) % (2 * np.pi))
        return (ang_a - ang_b) % (2 * np.pi)

    def get_bodypart_angle(coords, x1, y1, x2, y2):
        """adapt above to get relative bodyparts angle
        coords: array containing bodypart_i_x, bp_i_y, ...., bpnx, bpny
        next args are corresponding indexes in prev array"""
        try:
            p1 = np.array([coords[x1], coords[y1]])
            p2 = np.array([coords[x2], coords[y2]])
            p12 = p2 - p1
            out = np.rad2deg((-np.arctan2(*p12[::-1])) % (2 * np.pi))
            if out < 180:
                return out
            else:
                return out - 360
        except:
            return np.nan

    def rotate_coords_ccw_vec(points, angle, center_point=(0, 0)):
        """rotate points (2d arr) given angles(rad) around given origin"""
        if len(points.shape) == 1:
            points = points[None, :]
        cos, sin = np.cos(angle), np.sin(angle)
        new_point = points - center_point
        new_point[:, 0], new_point[:, 1] = (
            new_point[:, 0] * cos + new_point[:, 1] * sin,
            -new_point[:, 0] * sin + new_point[:, 1] * cos,
        )
        return new_point + center_point

    def rearrange_fbidx(l):
        """unreadable function to apply and tile fixation break frame indexes"""
        tlen = len(l)
        if not tlen:
            return np.array(l)
        else:
            o = np.array(list(zip(l[: int(tlen / 2)], l[int(tlen / 2) :]))).reshape(
                -1, 2
            )
            return o

    def populate_fb_traj(fb_idxs, posedf, part="rabove-snout"):
        """to apply in fb trajectories, shit happens, adding df(pose) as an arg"""
        if len(fb_idxs):
            o = []
            for r in fb_idxs:
                o += [posedf.iloc[r[0] : r[1]][part, "y"].values]
            return np.array(o)
        else:
            return np.array([])

    @staticmethod
    def sess_info(df):  # is this even used?
        # TODO: add{repetitive cols which coul be replaced by integers: ie sessionnames, subjids, task, rotations/activeports, meanframerate?}
        try:
            sess_name = df.loc[df.MSG == "SESSION-NAME", "+INFO"].values[0]
        except:
            sess_name = "not_found"
        subject_name = literal_eval(
            df.loc[df.MSG == "SUBJECT-NAME", "+INFO"].values[0]
        )[0]
        try:
            task = literal_eval(df.loc[df.MSG == "TASK", "+INFO"].values[0])[0]
        except:
            task = "not_found"
        box = df.loc[df.MSG == "VAR_BOX", "+INFO"].values[0]
        try:
            Variation = df.loc[df.MSG == "Variation", "+INFO"].values[-1]
        except:
            Variation = "not_found"

        try:
            Postop = df.loc[df.MSG == "Postop", "+INFO"].values[-1]
        except:
            Postop = "not_found"

        try:
            experimenter = literal_eval(
                df.loc[df.MSG == "CREATOR-NAME", "+INFO"].values[0]
            )[0]
        except:
            experimenter = "not_found"
        return {
            "sess_id": sess_name,
            "subj": subject_name,
            "task": task,
            "box": box,
            "variation": Variation,
            "postop": Postop,
            "experimenter": experimenter,
        }

    @staticmethod
    def get_rep(df):
        """function to retrieve rep/alt blocks, for those very early sessions (pilot) where it was not
        explicitly stated in the csv."""
        blen = int(
            df.loc[(df.TYPE == "VAL") & (df.MSG == "VAR_BLEN"), "+INFO"].values[0]
        )
        trial_list = literal_eval(
            df.loc[(df.TYPE == "VAL") & (df.MSG == "REWARD_SIDE"), "+INFO"].values[0]
        )
        nblocks = int(round(len(trial_list) / blen, 0))
        transitions = np.arange(nblocks, step=1) * blen
        segmentrepprobs = []
        for item in transitions.tolist():
            segment = trial_list[item : item + blen]
            rep = 0
            for j in range(1, blen):
                if segment[j] == segment[j - 1]:
                    rep += 1
            segmentrepprobs = segmentrepprobs + [rep]
        thereps = np.array(segmentrepprobs) * 100 / (blen - 1)
        first, second = np.arange(0, nblocks, step=2), np.arange(1, nblocks, step=2)
        if thereps[first].mean() > thereps[second].mean():
            probreporder = [80, 20]
        else:
            probreporder = [20, 80]
        if abs(thereps[first].mean() - thereps[second].mean()) < 20:
            # print('beware, not rly different probs among blocks')
            pass
        return np.repeat(probreporder * int(nblocks / 2), blen)

    # rework this f**** cr** | random_eda or glm_regressors notebooks have better stratj.
    # deprecated
    # @staticmethod
    # def extr_listened_frames(soundvec, frames_listened):
    #     """using this you take the risk of assuming that frames weight= throughout its duration in a future"""
    #     if np.isnan(frames_listened):  # delay triaals where sound is not triggered
    #         return np.nan
    #     else:
    #         a = soundvec[: 1 + int(frames_listened)].copy()
    #         if frames_listened >= 20:  # with 20 it crashes
    #             frames_listened = 19.99
    #         normvec = np.concatenate(
    #             (np.ones(int(frames_listened)), np.array([frames_listened % 1]))
    #         )
    #         a = a * normvec
    #         return a.sum() / frames_listened

    # removing from self
    def help(*args):
        """In development. This just accept strings, not methods"""
        helpdict = {
            "filestructure": "files should be organized following this structure:\n"
            + "[...]/data\n"
            + "\t├──/LEXX\n"
            + "\t│\t├──/poses\n"
            + "\t│\t│\t└── LEXX_[...].h5\n"
            + "\t│\t├──/sessions\n"
            + "\t│\t│\t└── LEXX[...].csv\n"
            + "\t│\t└──/videos\n"
            + "\t│\t\t├── LEXX[...].avi\n"
            + "\t│\t\t└── LEXX[...].npy\n"
            + "\t├──/LEYY\n"
            + "\t│\t├──/poses\n"
            + "etc."
        }
        if not len(args):
            print(
                """
                Usual flow is: 
                object = chom('LEXX', parent='../data/', analyze_trajectories) # this last one makes a diff! 
                .load_available(kwargs) # just required if changing defaults or plotting
                .load(target) | list of available sessions stored in .available 
                .process(normcoords=True, interpolate=False)
                .get_trajectories(bodypart='rabove-snout', fixationbreaks=True)
                    bodypart: tracking reference to get trajectories. ['rabove-snout', 'isnout']
                    fixationbreaks: whether to extract fb trajectories or not
                .suggest_coms()
                everything should be stored in those attributes
                    .trial_sess: stuff sorted in trial basis
                    .poses: dlc output
                    .sess: raw bpod output
                    .framestamps: raw npy framestamps
                    .trajectories: dic containing trial trajectories start & end frame idx

                or simply:
                df =  extraction_pipe(subjlist, **kwargs)
                    """
            )
            print(
                f"use .help(topic) for further info among those: {list(helpdict.keys())}"
            )
        else:  # even if args = single arg it is a touple so this should work
            for item in args:
                if item not in helpdict.keys():
                    print(f'did not understand "{item}"')
                    print(
                        f"use .help(topic) for further info among those: {list(helpdict.keys())}"
                    )
                    print(
                        """
                    Usual flow is: 
                    object = chom('LEXX', parent='../data/', analyze_trajectories) # this last one makes a diff! 
                    .load_available(kwargs) # just required if changing defaults or plotting
                    .load(target) | list of available sessions stored in .available 
                    .process(normcoords=True, interpolate=False)
                    .get_trajectories(bodypart='rabove-snout', fixationbreaks=True)
                        bodypart: tracking reference to get trajectories. ['rabove-snout', 'isnout']
                        fixationbreaks: whether to extract fb trajectories or not
                    .suggest_coms()
                    everything should be stored in those attributes
                        .trial_sess: stuff sorted in trial basis
                        .poses: dlc output
                        .sess: raw bpod output
                        .framestamps: raw npy framestamps
                        .trajectories: dic containing trial trajectories start & end frame idx

                    or simply:
                    df =  extraction_pipe(subjlist, **kwargs)
                        """
                    )
                else:
                    # commenting this out idk where it comes from : available
                    print(item)
                    print(helpdict[item])

    def __init__(
        self,
        subject,
        parentpath="/home/jordi/Documents/changes_of_mind/data/",
        analyze_trajectories=True,
        replace_silent=True
    ):
        """
        input subject (LEXX) and optionally datapath to instantiate class
        parentpath: where to look for, this expect some file structure, check help for more info
        analyze_trajectories: whether to analyze trajectories (requires: corresponding .avi, .npy, .hdf5)
        replace_silent: whether to repace content of envelopes in silent trials by [0]*20
        """
        ## ideally it contains all defaults so we can run whole pipe with a single call
        # perhaps we should create a 2nd class which inherits shared stuff from chom(subj)
        # instead of storing non shared stuff in above class (subj)
        self.available = []
        self.CSVS_PATH = f"{parentpath}{subject}/sessions/"
        self.POSES_PATH = f"{parentpath}{subject}/poses/"
        self.VIDEOS_PATH = f"{parentpath}{subject}/videos/"
        self.subject = subject
        self.sess = None  # session based
        self.pose = None  # session based
        self.trial_sess = None  # if not a copy_deep just a ref, ie no more mem req-
        self.processed = False  # session based
        self.target = None  # session id
        self.trajectories = None  # sessuib based
        self.fixed_framerate = None  # session based
        self.dirty_trajectories_trials = None  # session based
        self.info = {}  # session info
        self.framestamps = None  # session based
        self.normcoords = False # rotate coords, silly naming
        self.active_ports = None  # session based
        self.replace_silent = replace_silent
        self.dict_events = dict(
            zip([x for x in range(68, 85, 2)], [f"Port{x+1}" for x in range(8)])
        )
        self.event_to_int = dict(
            zip(
                [
                    f"Port{int(x/2)+1}In" if x % 2 == 0 else f"Port{int(x/2)+1}Out"
                    for x in range(0, 16)
                ],
                [x for x in range(68, 84)],
            )
        )
        self.newSM = False  # new state machine flag aka noenv + feedback sessions
        self.sound_timings = None
        self.analyze_trajectories = analyze_trajectories
        self.switching_idx = []  # 0 based indexing
        self.fair = False  # this is session-based, avoid defining it here because we can use 1 chom instance
        # to process several sessions in parallell which

        doublecheck = [self.POSES_PATH, self.CSVS_PATH, self.VIDEOS_PATH]
        if not analyze_trajectories:
            doublecheck = [self.CSVS_PATH]
        for item in doublecheck:
            if not os.path.isdir(item):
                # raise ValueError(f'could not find path: {item}') # crashes pipes, just warn with a print
                print(f"could not find path: {item}")
        chom.load_available(self, npy=analyze_trajectories)

    # this could be executed when creating instance
    def load_available(self, npy=True, plot=False, plot_kwargs={}):
        """check available sessions in stated dir & subject & files"""
        if self.analyze_trajectories:
            pose_list = [x[: -len(chom.pose_ext)] for x in os.listdir(self.POSES_PATH)]
            csv_list = [x[: -len(chom.csv_ext)] for x in os.listdir(self.CSVS_PATH)]
            video_list = [
                x[: -len(chom.video_ext)]
                for x in os.listdir(self.VIDEOS_PATH)
                if x.endswith(self.video_ext)
            ]
            if npy:
                npy_list = [
                    x[: -len(chom.video_ext)]
                    for x in os.listdir(self.VIDEOS_PATH)
                    if x.endswith(".npy")
                ]
            if not npy:
                self.available = sorted(
                    [x for x in pose_list if ((x in csv_list) and (x in video_list))]
                )
            else:
                self.available = sorted(
                    [
                        x
                        for x in pose_list
                        if ((x in csv_list) and (x in video_list) and (x in npy_list))
                    ]
                )
        else:
            self.available = [
                x[: -len(chom.csv_ext)] for x in os.listdir(self.CSVS_PATH)
            ]  # all csvs

        if plot:
            # plot perc of available, videos etc./ unmatched stuff, mean filesizes, by unique task (p1,p2,p3,p4_a, p4_b, etc.)
            raise NotImplementedError

    # experimenter, box, task, num trials etc.. few descriptives a little more in depth than load available
    def describe_sessions_pre(self, df=None, pattern=None, deep=False, njobs=1):
        """ in development
        pattern should be either a string or a function which operates with list of session rawnames (LEXX_....csv) and returns a list
        deep= whether to look for coms and extra process
        df= df which contain already processed sessions, uses pattern in sessid"""
        if not self.available:  # empty list
            print(
                "no available list yet. Try .load_available() or doublecheck parent / filestructure"
            )
            raise ValueError("empty list of available sessions")
        # few ideas missing sounds, invalids / perfs something like dailyreport /trend
        # RT, tachometric, psychometric
        if pattern is None:
            pattern = self.subject

        if isinstance(pattern, str):
            selection = [x for x in self.available if pattern in x]

        if not selection:  # empty subset
            raise ValueError("empty list of selected sessions")
        raise NotImplementedError

    def describe_sessions_post():  # once analysis is complete
        return None

    def load(self, target):  # target should be one of the available sessions
        """loads csv (and perhaps more)"""
        self.target = target
        self.sess = pd.read_csv(
            self.CSVS_PATH + target + chom.csv_ext,
            sep=";",
            skiprows=6,
            error_bad_lines=False,
        )
        self.sess["PC-TIME"] = pd.to_datetime(self.sess["PC-TIME"])

        self.info = chom.sess_info(self.sess)
        # TODO: if stored as metadata add fullpath location when this is called for ease of use later!

        # sessions with new state machine diagram
        if any([True if x in target else False for x in ["noenv", "feedback"]]):
            self.newSM = True
            # TODO: adapt to options
        # apparently old api issue, this is not used anyway
        if self.info["sess_id"] == "not_found":
            self.info["sess_id"] = self.target
        if self.info["task"] == "not_found":  # in old api this is not printed
            self.info["task"] = "_".join(self.target.split("_")[1:-1])

        self.active_ports = sorted(
            [
                self.dict_events[x]
                for x in self.dict_events.keys()
                if x
                in self.sess.loc[self.sess.TYPE == "EVENT", "MSG"].astype(int).unique()
            ]
        )
        if self.analyze_trajectories:
            try:
                self.pose = pd.read_hdf(self.POSES_PATH + target + chom.pose_ext).xs(
                    chom.pose_ext[:-3], axis=1, drop_level=True
                )
                self.framestamps = np.load(
                    self.VIDEOS_PATH + target + ".npy", allow_pickle=True
                )
                # if self.framestamps.dtype is np.dtype(np.object): # idk if this will work: https://stackoverflow.com/questions/26921836/correct-way-to-test-for-numpy-dtype
                # 'is' ddoes not work, boolean == does.
                if self.framestamps.dtype == np.dtype(np.object):
                    # i hate this: https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64
                    self.framestamps = np.array(
                        self.framestamps.tolist(), dtype="datetime64"
                    )
            except:
                # self.framestamps = np.load(self.VIDEOS_PATH+target+'.npy', allow_pickle=True)
                raise IOError(f"could not load {target} framestamps")

    # get active ports with this #allportevents = a.sess.loc[(a.sess.TYPE=='EVENT') & (a.sess['+INFO'].str.startswith('Port')), '+INFO'].unique()
    # set([x[:5] for x in allportevents])

    # comment this out because not used at all
    # TODO: delete this commented function
    # @staticmethod  # why not simply (self, port)??
    # def get_port_coord(sessdf, posedf, port, bodypart='rabove-snout'):
    #     '''port is a string eg Port1; requires fixed_int and self.sess'''
    #     ## np.unique(seq.reshape(-1,2), axis=0)
    #     seq = sessdf.loc[(sessdf.TYPE == 'EVENT') & ((sessdf['+INFO'] == port+'In')
    #                                                  | (sessdf['+INFO'] == port+'Out')), 'MSG'].astype(int).diff().values
    #     seq = seq[1:]
    #     seq = np.array(seq.tolist()+[1])  # add placeholder in last position
    #     findex = sessdf.loc[(sessdf.TYPE == 'EVENT') & ((sessdf['+INFO'] == port+'In') | (
    #         sessdf['+INFO'] == port+'Out')), 'fixed_int'].iloc[np.where(seq != 0)[0]].astype(int).values
    #     findex = findex.reshape((int(findex.size/2), 2))
    #     #framespace = np.arange(0, 1+fok.sess.loc[(fok.sess.TYPE=='EVENT') & ((fok.sess['+INFO']=='Port1In')|(fok.sess['+INFO']=='Port1Out')), 'fixed_int'].iloc[np.where(seq!=0)[0]].max())
    #     framespace = np.arange(0, 1+findex.flatten().max())
    #     mask = (framespace >= findex[:, 0][:, None]) & (
    #         framespace < findex[:, 1][:, None])  # 1 row per event (with its init - end)
    #     # this can lead to: 'positional indexeers are out-of-bounds'
    #     return posedf.iloc[np.where(mask)[1]].loc[:, bodypart].values

    @staticmethod
    # 2nd function---perhaps I crashed the function. If it works relace first one
    def get_port_coord2(sessdf, posedf, port, bodypart="rabove-snout"):
        """attempts to find port coordinates. i.e. returns median x,y coordinate of
        a given tracked bodypart while rat was in that port (between port-in and port-out)
        port is a string eg Port1; requires fixed_int and self.sess"""
        ## np.unique(seq.reshape(-1,2), axis=0)
        seq = sessdf.loc[
            (sessdf.TYPE == "EVENT")
            & ((sessdf["+INFO"] == port + "In") | (sessdf["+INFO"] == port + "Out")),
            "+INFO",
        ].values
        findex = sessdf.loc[
            (sessdf.TYPE == "EVENT")
            & ((sessdf["+INFO"] == port + "In") | (sessdf["+INFO"] == port + "Out")),
            "fixed_int",
        ].values
        ins = seq == (port + "In")
        # shift so first position is True = Port in
        shift = np.where(ins)[0][0]
        ins = ins[shift:]
        findex = findex[shift:]
        # removing last one which can be a byproduct of roll and we do not want to delete it (yet?)
        seek_and_destroy = np.where((np.roll(ins * 1, -1) - ins * 1)[:-1] == 0)[0]
        for i, item in enumerate(seek_and_destroy.tolist()):
            # False-False repetition (port-out+portout)
            if not ins[seek_and_destroy[i]]:
                # when falses repeated, remove 2nd. If Trues, we keep deleting first of the 2
                seek_and_destroy[i] = 1 + seek_and_destroy[i]
        if len(seek_and_destroy):
            ins = np.delete(ins, seek_and_destroy)
            findex = np.delete(findex, seek_and_destroy)
        if ins.size % 2:
            # we assume last one is unpaired, because we set an offset first and destroyed all repetitioins
            findex = findex[:-1].reshape(-1, 2)
        else:
            findex = findex.reshape(-1, 2)
        # framespace = np.arange(0, 1+fok.sess.loc[(fok.sess.TYPE=='EVENT') & ((fok.sess['+INFO']=='Port1In')|(fok.sess['+INFO']=='Port1Out')), 'fixed_int'].iloc[np.where(seq!=0)[0]].max())
        framespace = np.arange(0, 1 + findex.flatten().max())
        mask = (framespace >= findex[:, 0][:, None]) & (
            framespace < findex[:, 1][:, None]
        )  # 1 row per event (with its init - end)
        # this can lead to: 'positional indexeers are out-of-bounds' # when apparently video is shorter/corrupt [video complains about table index] or posedf is corrupted
        # print('mask',mask.shape)
        # print('npwhere mask', np.where(mask)[1].shape)
        # print('poses', posedf.shape)
        return posedf.iloc[np.where(mask)[1]].loc[:, bodypart].values

    # , target=self.target) # switch default to normcoords?
    def process(self, normcoords=True, interpolate=False):
        """preprocess data so it is organized by trials (1 row = 1 trial)
        resulting df is stored in self.trial_sess
        """
        # check if frame timestamps are available
        # first of all get rid of all last incomplete trial
        realendidx = self.sess[self.sess.MSG == "coherence01"].tail(1).index[0]

        if (
            self.sess.shape[0] > (self.sess.MSG == "coherence01").sum() * 500
        ):  # buggy probably. usually raw csv has around 50 rows / trial
            # else, find out why i get memory errors when processing those buggy sessions
            raise SystemError(
                f"session {self.target} is likely to be buggy because raw csv has >500rows/trial"
            )

        self.sess["cum_initial"] = np.nan  # kek
        self.sess.loc[
            (self.sess.TYPE == "INFO") & (self.sess.MSG == "TRIAL-BPOD-TIME"),
            "cum_initial",
        ] = self.sess.loc[
            (self.sess.TYPE == "INFO") & (self.sess.MSG == "TRIAL-BPOD-TIME"),
            "BPOD-INITIAL-TIME",
        ]
        self.sess["cum_initial"].fillna(method="backfill", inplace=True)
        # forgot to add col 'cum_initial, I assume'
        self.sess.loc[self.sess["BPOD-INITIAL-TIME"].isna(), "cum_initial"] = np.nan
        self.sess["cum_initial"] += self.sess["BPOD-INITIAL-TIME"]
        self.sess["frame_initial"] = self.sess["cum_initial"] * 30.0  # old
        self.sess.loc[
            self.sess["frame_initial"].notna(), "frame_initial"
        ] = self.sess.loc[self.sess["frame_initial"].notna(), "frame_initial"].astype(
            int
        )
        self.sess["trial_idx"] = np.nan

        # newline (2020-04-02); beware malfunction
        self.sess.loc[self.sess.MSG == "coherence01", "trial_idx"] = np.arange(
            1, (self.sess.MSG == "coherence01").sum() + 1
        )
        self.sess.loc[:, "trial_idx"] = self.sess.loc[:, "trial_idx"].fillna(
            method="ffill"
        )
        # self.sess.loc[self.sess.MSG=='84', 'trial_idx'] = np.arange(1,len(self.sess[self.sess.MSG=='84'])+1, step=1)
        # self.sess.loc[self.sess.TYPE=='EVENT', 'trial_idx']=self.sess.loc[self.sess.TYPE=='EVENT', 'trial_idx'].fillna(method='ffill') # why is this one
        # using events? ill replace it since coherebce01(register value) is the first constant found in every trial iteration

        # generate dict with session info (eg. experimenter, taskversion, subject, daytime, box...) ~ check above function 'sess_info()'
        # trial list?

        # process csv, now that we are doing it seriously, take dtypes into account | 1 month later: ayylmao
        # (remove last incomplete trial, hit history,response side,listened frames, resulting stim, frameidx, fixedframe)
        # filter events sequences, block type, p-repeat, trial_idx, coh, .... [check old simple analysis]

        if self.analyze_trajectories:
            # body parts used to infere snout location when likelihood is low
            bodyparts = ["L-eye", "R-eye", "L-ear", "R-ear"]
            coords = {}
            for item in bodyparts:
                coords[item] = self.pose.loc[:, (item, ["x", "y"])].values
            nrows = coords[bodyparts[0]].shape[0]
            dvec_eye = (coords["R-eye"].flatten() - coords["L-eye"].flatten()) / 2
            eyemid = (dvec_eye + coords["L-eye"].flatten()).reshape(nrows, 2)
            dvec_ear = (coords["R-ear"].flatten() - coords["L-ear"].flatten()) / 2
            earmid = (dvec_ear + coords["L-ear"].flatten()).reshape(nrows, 2)
            # this value might vary when rats are young/old
            dvec_midline = (eyemid.flatten() - earmid.flatten()) * 1.7
            synth_snout = (earmid.flatten() + dvec_midline).reshape(nrows, 2)
            self.pose["isnout", "x"] = np.nan
            self.pose["isnout", "y"] = np.nan
            self.pose.loc[:, ("isnout", ["x", "y"])] = self.pose.loc[
                :, ("snout", ["x", "y"])
            ].values

            self.pose["synth_snout", "x"] = synth_snout[:, 0]
            self.pose["synth_snout", "y"] = synth_snout[:, 1]
            self.pose["synth_snout", "likelihood"] = (
                self.pose.loc[:, ("L-ear", "likelihood")]
                * self.pose.loc[:, ("R-ear", "likelihood")]
                * self.pose.loc[:, ("L-eye", "likelihood")]
                * self.pose.loc[:, ("R-eye", "likelihood")]
            )

            # leave empty those slots where the synth_snout is not confident either
            self.pose.loc[
                (self.pose["snout", "likelihood"] < 0.6)
                & (self.pose["synth_snout", "likelihood"] > 0.6),
                ("isnout", ["x", "y"]),
            ] = self.pose.loc[
                (self.pose["snout", "likelihood"] < 0.6)
                & (self.pose["synth_snout", "likelihood"] > 0.6),
                ("synth_snout", ["x", "y"]),
            ].values
            if interpolate:
                # here calc teleports and fix them
                self.pose.loc[
                    (self.pose["snout", "likelihood"] < 0.6)
                    & (self.pose["synth_snout", "likelihood"] < 0.6),
                    ("isnout", ["x", "y"]),
                ] = np.nan
                self.pose["isnout", "x"].interpolate(inplace=True)
                self.pose["isnout", "y"].interpolate(inplace=True)

            # print video frames, frameloss etc. compared to csv.
            cap = cv2.VideoCapture(self.VIDEOS_PATH + self.target + chom.video_ext)
            reported_fps = cap.get(cv2.CAP_PROP_FPS)
            reported_total_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            reported_time = reported_total_f / reported_fps
            cap.release()

            # once processed
            # TODO: review this, mainly crap and cutting in a not really sophisticated manner (use tottrialstoconsider)
            # discard useless/not real last trial.
            self.sess = self.sess.iloc[:realendidx]
            sess_time = self.sess[self.sess.TYPE == "EVENT"]["cum_initial"].values[-1]
            fixed_fps = (reported_total_f) / sess_time
            self.sess["fixed_frames"] = self.sess["cum_initial"] * fixed_fps
            self.info["apparent_framerate"] = fixed_fps
            session_frames = self.sess[self.sess.TYPE == "EVENT"][
                "frame_initial"
            ].values[-1]

            self.fixed_framerate = fixed_fps
            if np.isnan(self.fixed_framerate):
                raise ValueError(
                    f"cannot compute framerate in {self.target}\n either number of frames ({reported_total_f}) or session time ({sess_time}) is null"
                )
            # TILL HERE

            if self.framestamps is not None:
                # ayy albert pipe-destroyer Font # fixing rogue plugin triggered except
                if abs(reported_total_f - self.framestamps.size) < 4:
                    self.sess["fixed_int"] = np.nan
                    # append nans or -1 at the end
                    if reported_total_f == self.framestamps.size:
                        self.sess.loc[
                            self.sess.TYPE == "EVENT", "fixed_int"
                        ] = np.searchsorted(
                            self.framestamps,
                            self.sess.loc[self.sess.TYPE == "EVENT", "PC-TIME"].values,
                        )
                    elif (
                        reported_total_f < self.framestamps.size
                    ):  # discard last stamps
                        self.framestamps = self.framestamps[
                            : int(reported_total_f - self.framestamps.size)
                        ]
                        self.sess.loc[
                            self.sess.TYPE == "EVENT", "fixed_int"
                        ] = np.searchsorted(
                            self.framestamps,
                            self.sess.loc[self.sess.TYPE == "EVENT", "PC-TIME"].values,
                        )
                    # discard last frames [no need to since it wont find corresponding timestamp in the vector]
                    elif reported_total_f > self.framestamps.size:
                        self.sess.loc[
                            self.sess.TYPE == "EVENT", "fixed_int"
                        ] = np.searchsorted(
                            self.framestamps,
                            self.sess.loc[self.sess.TYPE == "EVENT", "PC-TIME"].values,
                        )
                        # legit = np.searchsorted(self.framestamps, self.sess.loc[self.sess.TYPE=='EVENT', 'PC-TIME'].values)
                        # self.sess.loc[self.sess[self.sess.TYPE=='EVENT'][:legit.size].index, 'fixed_int']= legit
                    # else:

                    # self.sess.loc[self.sess.TYPE=='EVENT', 'fixed_int']= np.searchsorted(self.framestamps, self.sess.loc[self.sess.TYPE=='EVENT', 'PC-TIME'].values)
                    # np.searchsorted(self.framestamps, np.concatenate((startf, endf))).reshape((2,startf.size))
                else:  # self.framestamps.size != reported_total_f:
                    self.framestamps = None

            else:
                self.sess.loc[
                    self.sess["fixed_frames"].notna(), "fixed_int"
                ] = self.sess.loc[
                    self.sess["fixed_frames"].notna(), "fixed_frames"
                ].astype(
                    int
                )

            # here correct all isnouts which exceed certain speed (fps corrected)
            if interpolate:
                try:
                    # arbitrary constraint
                    speedlim = int(1500 / self.fixed_framerate)
                except:
                    # some malformed video give a nan here, use biggest thr
                    speedlim = int(50)
                test = self.pose.loc[:, ("isnout", ["x", "y"])].diff()
                test.columns = test.columns.droplevel(0)
                wix = test[test.x.abs() > speedlim].index.values
                wiy = test[test.y.abs() > speedlim].index.values
                # we do not know whether its 1st or 2nd frame which is wrong
                wi = np.unique(np.concatenate((wix, wiy)))
                wi = np.unique(np.concatenate((wi, wi - 1)))
                self.pose.loc[wi, ("isnout", ["x", "y"])] = np.nan
                self.pose["isnout", "x"].interpolate(inplace=True)
                self.pose["isnout", "y"].interpolate(inplace=True)

        # extract trial info from csv regardless of video and trajectories
        df1 = self.sess.copy(deep=True)
        df1["trial_index"] = np.nan

        df1.loc[
            (df1.TYPE == "VAL") & (df1.MSG == "coherence01"), "trial_index"
        ] = np.arange(
            1, 1 + len(df1.loc[(df1.TYPE == "VAL") & (df1["MSG"] == "coherence01")])
        )  # 1 based indexing
        df1["trial_index"].fillna(method="ffill", inplace=True)

        # dismiss trials after global timer end (some buggy sessions out there)
        if (df1["+INFO"] == "GlobalTimer1_End").sum():
            # trial before first instance
            tottrialstoconsider = int(
                df1.loc[df1["+INFO"] == "GlobalTimer1_End", "trial_index"].values[0] - 1
            )
            # global timer ends in waitcpoke hence that trial needs to be discarded
        else:
            # just take into account trials where stim has been played
            tottrialstoconsider = df1[
                (df1.TYPE == "TRANSITION") & (df1.MSG == "WaitResponse")
            ].shape[0]

        df1 = df1[~(df1.trial_index > tottrialstoconsider)]  # discard trash

        # using inverse mask else it trashes initial useful info
        states = (
            df1[df1.TYPE == "STATE"]
            .sort_values(["trial_index", "BPOD-INITIAL-TIME"])
            .reset_index(drop=True)
        )  # new sort_values but should work
        states["+INFO"] = states["+INFO"].astype(float)
        fix = states[states.MSG == "Fixation"]
        # fix.loc[fix.MSG=='Fixation','+INFO'] = fix.loc[fix.MSG=='Fixation','+INFO'].astype(float) ## because added line @-2
        if not self.newSM:  # regular sess, _noenv below because we'll alter dfstates
            fb = (
                fix.groupby(["trial_index", "MSG"])["+INFO"]
                .apply(list)
                .apply(lambda x: np.array(x[:-1]))
                .values
            )

        coh = df1[df1.MSG == "coherence01"]["+INFO"].values
        trialidx = np.arange(1, tottrialstoconsider + 1, step=1)
        rewside = np.array(
            literal_eval(
                df1.loc[
                    df1.loc[df1.MSG == "REWARD_SIDE", "+INFO"].index.values[-1], "+INFO"
                ]
            )
        )[:tottrialstoconsider]

        # hotfix for fairsessions
        if df1[df1.MSG == "Variation"].shape[0]:
            if df1.loc[df1.MSG == "Variation", "+INFO"].values[0] == "fair":
                self.fair = True
                switching_idx = df1.loc[
                    df1.MSG == "fair_sc_switch_rewside", "+INFO"
                ].values.astype(int)
                # reasign resulting value
                rewside[switching_idx] = (rewside[switching_idx] - 1) ** 2
                self.switching_idx = switching_idx  # 0 based
        hithistory = np.where(
            states[states.MSG == "Reward"]["BPOD-FINAL-TIME"].astype(float) > 0,
            1,
            np.nan,
        )
        hithistory[
            np.where(
                states[states.MSG == "Punish"]["BPOD-FINAL-TIME"].astype(float) > 0
            )[0]
        ] = 0
        hithistory[
            np.where(
                states[states.MSG == "Invalid"]["BPOD-FINAL-TIME"].astype(float) > 0
            )[0]
        ] = -1
        if self.fair:
            hithistory[
                np.where(
                    states[states.MSG == "invPunish"][  # this does not exist anymore
                        "BPOD-FINAL-TIME"
                    ].astype(float)
                    > 0
                )[0]
            ] = 0
            hithistory[
                np.where(
                    states[states.MSG == "invReward"][  # this still exists
                        "BPOD-FINAL-TIME"
                    ].astype(float)
                    > 0
                )[0]
            ] = 1
        if self.newSM:
            if len(
                df1.loc[df1.MSG == "enhance_com_switch"]
            ):  # likely altered from original reward_side
                if (
                    float(
                        literal_eval(df1.loc[df1.MSG == "TASK", "+INFO"].values[0])[1]
                    )
                    < 0.3
                ):
                    # task earlier than v0.3 were sub-optimal
                    # whether they are correct is fine, just need to make sure side is fine!
                    # comswitchtrials = df1.loc[df1.MSG=='enhance_com_switch', 'trial_index'].values # perhaps not all of them are switch
                    comswitchtrials = (
                        df1.loc[df1.MSG == "enhance_com_switch", "+INFO"].values.astype(
                            int
                        )
                        + 1
                    )  # this was not working prop, previous on top
                    # it should be 1-based now
                    eventdfwobnc = df1.loc[
                        df1.trial_index.isin(comswitchtrials)
                        & (df1.TYPE == "EVENT")
                        & ~(df1["+INFO"].str.startswith("BNC", na=False))
                    ]

                    pat = np.asarray(["Tup", "Tup", self.active_ports[1] + "Out"])
                    N = pat.size
                    arr = eventdfwobnc["+INFO"].values
                    b = np.all(chom.rolling_window(arr, N) == pat, axis=1)
                    c = np.mgrid[0 : len(b)][b]

                    d = [i for x in c for i in range(x, x + N)]
                    eventdfwobnc["onset"] = np.in1d(np.arange(len(arr)), d)
                    # iterate to avoid weird patterns/coincidences
                    # switched_responses = [] # no need to keep it
                    for i, tr in enumerate(comswitchtrials):
                        try:  ## TODO: debug and rerun trials
                            indx = eventdfwobnc.loc[
                                (eventdfwobnc.trial_index == tr) & eventdfwobnc.onset
                            ].index.max()
                            c_resp = eventdfwobnc.loc[
                                (eventdfwobnc.index > indx + 1)
                                & (eventdfwobnc.trial_index == tr)
                                & ~(
                                    eventdfwobnc["+INFO"].str.startswith(
                                        self.active_ports[1]
                                    )
                                )
                                & (eventdfwobnc.MSG != "104"),
                                "+INFO",
                            ].values[0][:5]
                            if self.active_ports[0] == c_resp:  # rat response was left
                                if hithistory[int(tr) - 1] == 1:  # it was a hit
                                    rewside[int(tr) - 1] = 0
                                elif (
                                    hithistory[int(tr) - 1] == 0
                                ):  # miss # we do not know with invalids
                                    rewside[int(tr) - 1] = 1
                            elif (
                                self.active_ports[2] == c_resp
                            ):  # rat response was right
                                if hithistory[int(tr) - 1] == 1:  # it was a hit
                                    rewside[int(tr) - 1] = 1
                                elif (
                                    hithistory[int(tr) - 1] == 0
                                ):  # miss # we do not know with invalids
                                    rewside[int(tr) - 1] = 0
                        except Exception as e:
                            print(
                                f"could not elucidate which was the reward side in trial {tr}\n{e}"
                            )
                else:  # issue solved in vers >0.3!
                    # this should remain the same in 0.5 (another issue solved)
                    # just invert rewside in those trials
                    # comswitchtrials = df1.loc[df1.MSG=='enhance_com_switch', 'trial_index'].values.astype(int) # 1 based to keep it the same way than previous if
                    comswitchtrials = (
                        df1.loc[df1.MSG == "enhance_com_switch", "+INFO"].values.astype(
                            int
                        )
                        + 1
                    )  # 1 based to keep it the same way than previous if
                    rewside[comswitchtrials - 1] = (
                        rewside[comswitchtrials - 1] - 1
                    ) ** 2
            else:
                comswitchtrials = np.array([])

        # get soundR failures
        sr_play = df1.loc[
            (df1.MSG.str.startswith("SoundR: Play.")) & (df1.TYPE == "stdout"),
            "trial_index",
        ].values
        sr_stop = df1.loc[
            (df1.MSG.str.startswith("SoundR: Stop.")) & (df1.TYPE == "stdout"),
            "trial_index",
        ].values
        soundrfail = sr_stop[
            np.isin(sr_stop, sr_play, assume_unique=True, invert=True)
        ].astype(
            int
        )  # 1 based trial index
        soundrok = sr_stop[
            np.isin(sr_stop, sr_play, assume_unique=True, invert=False)
        ].astype(
            int
        )  # inverse
        startpctime = df1.loc[
            df1.MSG.str.startswith("SoundR: P") & df1.trial_index.isin(soundrok),
            "PC-TIME",
        ].values
        stoppctime = df1.loc[
            df1.MSG.str.startswith("SoundR: S") & df1.trial_index.isin(soundrok),
            "PC-TIME",
        ].values
        soundr_len = (stoppctime - startpctime).astype(float) / 1000000  # from ns to ms

        # TODO: fix feedback if revesing reward(fair)! ~ i would preprocess whole session before puting it through pipe
        if self.newSM:
            # startsound indexes
            id_ss = states[states["MSG"] == "StartSound"].index.values
            id_wcp1 = states.drop_duplicates(
                subset="trial_index"
            ).index.values  # first waitcpoke per trial
            # drop states unrelated to fb
            tmp = states.drop(np.concatenate([id_ss - 1, id_ss - 2, id_wcp1]))
            tmptrial_index = tmp.loc[tmp.MSG == "WaitCPoke", "trial_index"].values
            fb_length = (
                tmp.loc[tmp.MSG == "WaitCPoke", "BPOD-INITIAL-TIME"].values
                - tmp.loc[tmp.MSG == "Fixation_fb", "BPOD-INITIAL-TIME"].values
            )
            fbdf = pd.DataFrame({"len": fb_length, "trial_index": tmptrial_index})
            missingidxmask = np.isin(
                np.arange(id_ss.size).astype(float) + 1, tmptrial_index, invert=True
            )
            missingidx = (np.arange(id_ss.size).astype(float) + 1)[missingidxmask]
            fbdf = fbdf.append(
                pd.DataFrame(
                    {
                        "len": np.repeat(np.nan, missingidxmask.sum()),
                        "trial_index": missingidx,
                    }
                ),
                ignore_index=True,
            )
            fb = (
                fbdf.groupby("trial_index")["len"]
                .apply(list)
                .apply(lambda x: x if (~np.isnan(x)).sum() else [])
                .values
            )

        # get delays from albert's device
        # below

        notinv = np.where(hithistory >= 0)[0]  # 0-based
        RResponse = np.zeros(len(hithistory))

        RResponse[
            np.where(np.logical_and(rewside == 1, hithistory == 1) == True)[0]
        ] = 1  # right and correct
        RResponse[
            np.where(np.logical_and(rewside == 0, hithistory == 0) == True)[0]
        ] = 1  # left and incorrect
        # else, invalids will be considered L_responses
        RResponse[~(hithistory >= 0)] = np.nan

        lenv = df1[df1.MSG == "left_envelope"]["+INFO"]
        renv = df1[df1.MSG == "right_envelope"]["+INFO"]
        lenv = lenv.apply(lambda x: np.array(literal_eval(x)))
        renv = renv.apply(lambda x: np.array(literal_eval(x)))

        
            # if silence task get indexes from silent trials to adapt envelope values to 0
        if "silence" in self.CSVS_PATH + self.target + chom.csv_ext:
            silent_trial_idx = (
                df1.loc[df1.MSG == "silence_trial", "trial_index"].values.astype(int)
                - 1
            )  # 0 indexed
            if self.replace_silent:
                renv.iloc[silent_trial_idx] = [np.zeros(20)] * silent_trial_idx.size
                lenv.iloc[silent_trial_idx] = [np.zeros(20)] * silent_trial_idx.size

        kek = pd.DataFrame(
            data=np.array(
                [trialidx, coh, rewside, hithistory, RResponse]
            ).T,  # why notinv?
            columns=["origidx", "coh", "rewside", "hithistory", "R_response"],
        )
        for i in list(kek.columns)[1:]:
            kek[i] = kek[i].astype(float)
        kek["coh"] = kek.coh.round(decimals=3)
        subj = literal_eval(
            df1.loc[(df1.TYPE == "INFO") & (df1.MSG == "SUBJECT-NAME"), "+INFO"].values[
                0
            ]
        )[0]
        sessid = df1[(df1.TYPE == "INFO") & (df1.MSG == "SESSION-NAME")][
            "+INFO"
        ].values[0]
        kek["subjid"] = subj
        kek["sessid"] = sessid
        trialonset = (
            df1[(df1.TYPE == "INFO") & (df1.MSG == "TRIAL-BPOD-TIME")][
                "BPOD-INITIAL-TIME"
            ]
            .astype(float)
            .values
        )
        soundonset = (
            df1[(df1.TYPE == "STATE") & (df1.MSG == "StartSound")]["BPOD-INITIAL-TIME"]
            .astype(float)
            .values
        )
        kek["resp_len"] = (
            self.sess.loc[
                (self.sess.TYPE == "STATE") & (self.sess.MSG == "WaitResponse"), "+INFO"
            ]
            .values[:tottrialstoconsider]
            .astype(float)
        )
        if not "_noenv" in self.target:  # specific for noenvelope sessions
            kek["lenv"] = lenv.values
            kek["renv"] = renv.values
            kek["res_sound"] = kek["lenv"] + kek["renv"]
        else:  # specific for noenvelope sessions
            kek["renv"] = kek.coh
            kek["lenv"] = kek.coh - 1
            kek["res_sound"] = kek.renv + kek.lenv
            silent_trial_idx = df1.loc[
                df1.MSG == "silence_trial", "trial_index"
            ].values.astype(
                int
            )  # 0 indexed
            if silent_trial_idx.size:
                kek.loc[silent_trial_idx, ["renv", "lenv", "res_sound"]] = 0

        kek["trialonset"] = trialonset
        # what happen in silent trials? fix: should be nan already
        kek["soundonset"] = soundonset
        sound_len = (
            df1[(df1.MSG == "StartSound") & (df1.TYPE == "STATE")]["+INFO"]
            .astype(float)
            .values
        )  # buggy or faulty bpod?
        kek["sound_len"] = sound_len  # buggy ? # also nan for
        kek["sound_len"] = kek["sound_len"].astype(float) * 1000
        if self.newSM:
            if "noenv" in self.target:
                kek["frames_listened"] = np.nan
            elif "feedback" in self.target:
                kek["frames_listened"] = kek["sound_len"] / 25
        else:
            kek["frames_listened"] = kek["sound_len"] / 50

        if "uncorrelated" in self.target:
            # uncorrelated silence new [states 0, but it's 0.5]
            kek["prob_repeat"] = 0.5
        elif "leftright" in self.target:
            # adapt L/R blocks to prob repeat
            LRblockvec = np.ones(len(df1[df1.MSG == "prob_repeat"]["+INFO"]))
            LRblockvec[df1[df1.MSG == "prob_repeat"]["+INFO"] == "L"] = -1
            kek["prob_repeat"] = 0.5 + (
                (kek.rewside.shift(1) * 2 - 1) * 0.3 * LRblockvec
            )
        else:  # regular sessions
            if len(df1[df1.MSG == "prob_repeat"]["+INFO"].astype(float)) > 0:
                kek["prob_repeat"] = (
                    df1[df1.MSG == "prob_repeat"]["+INFO"].astype(float).values
                )
            else:
                # avoid. get rep is not defined
                print(
                    f"somehow could not get prob_repeat in {self.target}; inferring...\nBbeware, this will lead to wrong results if not .2-.8 and rep-alt"
                )
                kek["prob_repeat"] = chom.get_rep(df1)[: len(trialidx)]
        # add withinblock index
        blen = int(
            float(
                df1.loc[(df1.TYPE == "VAL") & (df1.MSG == "VAR_BLEN"), "+INFO"].values[
                    -1
                ]
            )
        )
        bnum = int(
            float(
                df1.loc[(df1.TYPE == "VAL") & (df1.MSG == "VAR_BNUM"), "+INFO"].values[
                    -1
                ]
            )
        )
        # wtf what about invalid trials?
        kek["wibl_idx"] = np.tile(np.arange(1, blen + 1, step=1), bnum)[: len(trialidx)]
        kek["bl_idx"] = np.repeat(np.arange(1, 1 + bnum), blen)[: len(trialidx)]
        kek["aftererror"] = (~(kek["hithistory"].shift(1).astype(bool))) * 1
        kek["fb"] = fb
        kek["soundrfail"] = False
        kek.loc[kek.origidx.isin(soundrfail), "soundrfail"] = True

        # apparently some sessions which crashed or finnished somehow different, soundr_len is 1 trial shorter than kek
        # my 2nd guess is that broken pipes & delays in stdout being written may be messing around (IOW, everytyhing may be shifted at certain point)
        kek["soundr_len"] = 0  # some did not play so length =0
        # remaining ones, they look sorted because of low MAE when comparing theoretical(BPOD) to soundR
        kek.loc[kek.origidx.isin(soundrok), "soundr_len"] = soundr_len

        # TODO: CONTINUE HERE
        # get sound length according to albert's detection board # what about delays.
        # it is because of fair task and reversals --- no
        kek["albert_len"] = np.nan
        # some sessions do not contain this info
        if df1["MSG"].isin(["60", "61", "62", "63"]).sum() > 0:
            test = df1[
                df1["MSG"].isin(["StartSound", "WaitResponse", "60", "61", "62", "63"])
            ]
            test = test[test.TYPE != "STATE"]
            testa = test[test.TYPE == "TRANSITION"]
            testb = test[test.MSG.isin(["60", "62"])].drop_duplicates(
                subset=["+INFO", "trial_index"], keep="first"
            )
            testc = test[test.MSG.isin(["61", "63"])].drop_duplicates(
                subset=["+INFO", "trial_index"], keep="last"
            )
            test = pd.concat([testa, testb, testc]).sort_index()  # .sort_values()
            test["trial_index"] = test["trial_index"].astype(int)
            test = pd.pivot_table(
                test,
                values="BPOD-INITIAL-TIME",
                index=["trial_index"],
                columns=["MSG"],
                fill_value=np.nan,
            )
            test["soundrfail"] = False
            test.loc[soundrfail, "soundrfail"] = True
            test["albert_earliest"] = np.nan
            test.loc[test.soundrfail == False, "albert_earliest"] = (
                test.loc[test.soundrfail == False, ["60", "62"]]
                .fillna(value=np.inf)
                .min(axis=1)
            )
            test["albert_latest"] = np.nan
            test.loc[test.soundrfail == False, "albert_latest"] = (
                test.loc[test.soundrfail == False, ["61", "63"]]
                .fillna(value=-np.inf)
                .max(axis=1)
            )
            test["albert_len"] = test.albert_latest - test.albert_earliest
            # those trials where Albert's device did not detect anything but SoundR was played
            test.loc[test.albert_len == -np.inf, "albert_len"] = np.nan

            self.sound_timings = test
            kek["albert_len"] = test.albert_len.values

        self.trial_sess = kek
        self.trial_sess["streak"] = np.nan
        # we consider invalid trials as breaking streak
        changeidx = self.trial_sess.hithistory.fillna(value=0).diff().values
        self.trial_sess.loc[changeidx != 0, "streak"] = np.arange(
            (changeidx != 0).sum()
        )
        self.trial_sess.streak.fillna(method="ffill", inplace=True)
        heh = (
            self.trial_sess.fillna(value=0).groupby(["streak", "hithistory"]).cumcount()
        )
        # place 0 where hithistory = 0
        heh[(self.trial_sess.fillna(value=0).hithistory == 0).values] = -1
        self.trial_sess["streak"] = (heh + 1).shift(1)
        # for ease of use we'll set first one as 0
        self.trial_sess.streak.iloc[0] = 0  # triggers warning
        self.trial_sess["rep_response"] = False
        self.trial_sess.loc[
            self.trial_sess.R_response.diff().values == False, "rep_response"
        ] = True

        # tag weird trials -1=early, 0 = regular, 1= delay, 2 = silence TODO: review trajectories are fine for weird trials
        self.trial_sess["special_trial"] = 0
        self.trial_sess["delay_len"] = 0

        # TODO: add negative delay for early trials
        if "delay" in self.target:
            self.trial_sess["special_trial"] = (
                self.sess.loc[self.sess.MSG == "delay_trial", "+INFO"]
                .astype(int)
                .values[: kek.shape[0]]
            )
            # here add negative delay for those early trials (labeled as -1 in line above)
            self.trial_sess["delay_len"] = (
                self.sess.loc[
                    (self.sess.TYPE == "STATE") & (self.sess.MSG == "Delay"), "+INFO"
                ]
                .astype(float)
                .values[: kek.shape[0]]
            )  # removing *1000 to keep in secs

            # edit those which were early (1ms delay till here)
            earlytrials = np.where(self.trial_sess["special_trial"].values == -1)[
                0
            ]  # 0 based trial index
            # get those trials last fixation, and -0.3 # for robustnes, some sessions have 50ms early, other 150 and other 250? idk
            # this strat should work for all of them since fixation time is chosen by an ifelse
            # if last trial is discarded but marked as early this can raise issues using this loc.
            # ensure we do nto find index>prunned session length
            earlytrials = earlytrials[earlytrials < kek.shape[0]]
            if earlytrials.size:
                self.trial_sess.loc[earlytrials, "delay_len"] = (
                    df1.loc[
                        df1.trial_index.isin(earlytrials.astype(float) + 1)
                        & (df1.TYPE == "STATE")
                        & (df1.MSG == "Fixation")
                    ].drop_duplicates(subset=["MSG", "trial_index"], keep="last")
                )["+INFO"].values.astype(float) - 0.3

        elif "silence" in self.target:
            # kek.loc[silent_trial_idx, 'special_trial'] = 2  # silence ones # just replaced this, rollback if crash
            self.trial_sess.loc[silent_trial_idx, "special_trial"] = 2
        elif self.newSM:
            delay_trials = df1.loc[df1.MSG == "delayed_trial", "+INFO"].values.astype(
                int
            )
            if delay_trials.size:
                self.trial_sess.loc[delay_trials, "special_trial"] = 1
                self.trial_sess.loc[delay_trials, "delay_len"] = df1.loc[
                    df1.MSG == "expected_delay", "+INFO"
                ].values.astype(
                    float
                )  # in seconds

            if len(comswitchtrials):
                self.trial_sess.loc[
                    comswitchtrials - 1, "special_trial"
                ] = 3  # new mark for com-switch-trials
                # -1 because it is 1-based (buggy before Feb2021)
            silence_trials = df1.loc[df1.MSG == "silence_trial", "+INFO"].values.astype(
                int
            )
            if silence_trials.size:
                self.trial_sess.loc[silence_trials, "special_trial"] = 2

        # get fixation onset timestamp
        fix_onset_state = "Fixation"
        if self.newSM:
            fix_onset_state = "Fixation_fb"

        self.trial_sess["fix_onset_dt"] = (
            df1.loc[(df1.MSG == fix_onset_state) & (df1.TYPE == "TRANSITION")]
            .drop_duplicates(subset=["MSG", "trial_index"], keep="last")["PC-TIME"]
            .values[: kek.shape[0]]
        )

        # smooth here
        # alternatively, only smooth the trajectories (we know that the snout likely wont be occluded)

        # needs to be done before get trajectories, because of Y component of V (speed)
        if normcoords & self.analyze_trajectories:
            # adding temp cols for other rotated bodyparts
            # good learning curve there
            self.pose["rL-eye", "x"] = np.nan
            self.pose["rL-eye", "y"] = np.nan
            self.pose["rR-eye", "x"] = np.nan
            self.pose["rR-eye", "y"] = np.nan
            self.pose["rL-ear", "x"] = np.nan
            self.pose["rL-ear", "y"] = np.nan
            self.pose["rR-ear", "x"] = np.nan
            self.pose["rR-ear", "y"] = np.nan
            self.pose["rneck", "x"] = np.nan
            self.pose["rneck", "y"] = np.nan
            self.pose["rback", "x"] = np.nan
            self.pose["rback", "y"] = np.nan
            self.pose["rtail", "x"] = np.nan
            self.pose["rtail", "y"] = np.nan
            self.pose["rabove-snout", "x"] = np.nan
            self.pose["rabove-snout", "y"] = np.nan

            # should this center C-port to (0,0)?
            p1coords, p2coords, p3coords = (
                chom.get_port_coord2(
                    self.sess, self.pose, self.active_ports[0], bodypart="isnout"
                ),
                chom.get_port_coord2(
                    self.sess, self.pose, self.active_ports[1], bodypart="isnout"
                ),
                chom.get_port_coord2(
                    self.sess, self.pose, self.active_ports[2], bodypart="isnout"
                ),
            )
            # using isnout above because rabove-snout does not exist yet!

            # self.pose['rsnout','x'], self.pose['rsnout','y']= np.nan, np.nan # we will directly place it into isnout_
            p1sd, p2sd, p3sd = (
                p1coords.std(axis=0),
                p2coords.std(axis=0),
                p3coords.std(axis=0),
            )
            op1, op2, op3 = [
                chom.polish_median(x) for x in [p1coords, p2coords, p3coords]
            ]
            p3p1 = op1 - op3  # original
            n1 = np.array([op3[0], op3[1] - np.linalg.norm(p3p1)])
            p3n1 = n1 - op3  # rotate to
            targ_rotation = chom.calculate_angle(p3p1, p3n1)
            p1r, p2r, p3r = chom.rotate_coords_ccw_vec(
                np.concatenate([op1, op2, op3]).reshape(-1, 2),
                targ_rotation,
                center_point=op3,
            )

            self.pose.loc[:, ("isnout", ["x", "y"])] = chom.rotate_coords_ccw_vec(
                self.pose["isnout"].values, targ_rotation, op3
            ) - p2r.reshape(
                1, 2
            )  # i am a newb and i hate multiindex

            # add all other bodyparts
            self.pose.loc[:, ("rL-eye", ["x", "y"])] = chom.rotate_coords_ccw_vec(
                self.pose.loc[:, ("L-eye", ["x", "y"])].values, targ_rotation, op3
            ) - p2r.reshape(1, 2)
            self.pose.loc[:, ("rR-eye", ["x", "y"])] = chom.rotate_coords_ccw_vec(
                self.pose.loc[:, ("R-eye", ["x", "y"])].values, targ_rotation, op3
            ) - p2r.reshape(1, 2)
            self.pose.loc[:, ("rL-ear", ["x", "y"])] = chom.rotate_coords_ccw_vec(
                self.pose.loc[:, ("L-ear", ["x", "y"])].values, targ_rotation, op3
            ) - p2r.reshape(1, 2)
            self.pose.loc[:, ("rR-ear", ["x", "y"])] = chom.rotate_coords_ccw_vec(
                self.pose.loc[:, ("R-ear", ["x", "y"])].values, targ_rotation, op3
            ) - p2r.reshape(1, 2)
            self.pose.loc[:, ("rneck", ["x", "y"])] = chom.rotate_coords_ccw_vec(
                self.pose.loc[:, ("neck", ["x", "y"])].values, targ_rotation, op3
            ) - p2r.reshape(1, 2)
            self.pose.loc[:, ("rback", ["x", "y"])] = chom.rotate_coords_ccw_vec(
                self.pose.loc[:, ("back", ["x", "y"])].values, targ_rotation, op3
            ) - p2r.reshape(1, 2)
            self.pose.loc[:, ("rtail", ["x", "y"])] = chom.rotate_coords_ccw_vec(
                self.pose.loc[:, ("tail", ["x", "y"])].values, targ_rotation, op3
            ) - p2r.reshape(1, 2)
            self.pose.loc[:, ("rabove-snout", ["x", "y"])] = chom.rotate_coords_ccw_vec(
                self.pose.loc[:, ("above-snout", ["x", "y"])].values, targ_rotation, op3
            ) - p2r.reshape(1, 2)

            # change plots according to if normcoords
            self.normcoords = True
            self.info["rotation"] = targ_rotation

        self.processed = True

    # we can filter later by response side
    def get_trajectories(
        self, bodypart="rabove-snout", fixationbreaks=False, noffset_frames=2
    ):
        """ # starts at correct fixation
        requires .process() first
        generates a dictionary of initial_frame and ending frame of trajectories = stored in self.trajectories
        The key for each entry is the frame number.
        isnout = 'inferred snout', check above-snout
        # noffset_frames, leaving default = 2 because of historical reasons,
        trajectory frames to shift
        """

        trans = self.sess.loc[self.sess.TYPE == "TRANSITION"]
        if "delay" not in self.target:
            trans.loc[trans.MSG == "StartSound", "trial_idx"] = np.arange(
                1, len(trans.loc[trans.MSG == "StartSound", "trial_idx"]) + 1
            )  # assigns trial idx
            trans.loc[
                trans.MSG.isin(["StartSound", "keep-led-on", "Punish", "Invalid"]),
                "trial_idx",
            ].fillna(method="ffill", inplace=True)
        else:
            trans.loc[trans.MSG == "Delay", "trial_idx"] = np.arange(
                1, len(trans.loc[trans.MSG == "Delay", "trial_idx"]) + 1
            )  # assigns trial idx
            trans.loc[
                trans.MSG.isin(
                    ["Delay", "StartSound", "keep-led-on", "Punish", "Invalid"]
                ),
                "trial_idx",
            ].fillna(method="ffill", inplace=True)

        trans.loc[:, "trial_idx"].fillna(method="bfill", inplace=True)
        trans.loc[:, "fixed_int"] = np.searchsorted(
            self.framestamps,
            self.sess.loc[self.sess.TYPE == "TRANSITION", "PC-TIME"].values,
        )

        # extract trajectories based on transitions.
        # retrieve fixation frame indexes from here instead
        # get startsound transition frame index (* fixation ends)
        # TODO: needs adendum for fair inverted!!! ~ just debug missing
        if self.newSM and list(self.switching_idx):  #
            # we get the bool mas for transition and then we shift it few positions attending what we are interested in
            regulartrans = trans.loc[~trans.trial_idx.isin(self.switching_idx + 1)]
            critical_trans_regular = (
                regulartrans["MSG"] == "StartSound"
            ).values  # switching because of fair cannot be delayed trials
            # contains frames corresponding to the end of the trajectory ~next transition to StartSound = waitresp; next = response. Trial based
            end_traj_vec_regular = regulartrans.loc[
                np.roll(critical_trans_regular, 2), "fixed_int"
            ].values
            start_traj_vec_regular = regulartrans.loc[
                np.roll(critical_trans_regular, -2), "fixed_int"
            ].values
            # end_traj_vec_regular = trans.loc[np.roll(
            #    critical_trans_regular, 2), 'fixed_int'].values # wth crash here?
            # start_traj_vec_regular = trans.loc[np.roll(
            #    critical_trans_regular, -2), 'fixed_int'].values
            # fixed int is sorted, so there should be no problem about sorting them afterwards
            # we get the bool mas for transition and then we shift it few positions attending what we are interested in
            trans_switched = trans.loc[trans.trial_idx.isin(self.switching_idx + 1)]
            critical_trans_switched = (trans_switched["MSG"] == "StartSound").values
            # critical_trans_switched = (trans.loc[trans.trial_idx.isin(
            #    self.switching_idx+1), 'MSG'] == 'StartSound').values
            # contains frames corresponding to the end of the trajectory ~next transition to StartSound = waitresp; next = response. Trial based
            end_traj_vec_switched = trans_switched.loc[
                np.roll(critical_trans_switched, 3), "fixed_int"
            ].values
            # end_traj_vec_switched = trans.loc[np.roll(
            #    critical_trans_switched, 3), 'fixed_int'].values
            start_traj_vec_switched = trans_switched.loc[
                np.roll(critical_trans_switched, -2), "fixed_int"
            ].values
            # start_traj_vec_switched = trans.loc[np.roll(
            #    critical_trans_switched, -2), 'fixed_int'].values
            # merge and sort them ~ because items are frame indexes they will align naturally with trials
            start_traj_vec = np.sort(
                np.concatenate((start_traj_vec_regular, start_traj_vec_switched))
            )
            end_traj_vec = np.sort(
                np.concatenate((end_traj_vec_regular, end_traj_vec_switched))
            )
            # elif self.newSM # smthing to account to delays
        elif "delay" not in str(self.target):
            if self.newSM:
                rollback = -2  # fixation contains an extra state (feedback)
            else:
                rollback = -1
            # we get the bool mas for transition and then we shift it few positions attending what we are interested in
            critical_trans = (trans.MSG == "StartSound").values
            # contains frames corresponding to the end of the trajectory ~next transition to StartSound = waitresp; next = response. Trial based
            end_traj_vec = trans.loc[np.roll(critical_trans, 2), "fixed_int"].values
            # contains frames corresponding to the beginning of the trajectories ~ fisiation onset, = transition startsound-1. Trial based
            start_traj_vec = trans.loc[
                np.roll(critical_trans, rollback), "fixed_int"
            ].values
        else:  # a bit more complicated because sometimes there's an extra state
            critical_trans = (trans.MSG == "Delay").values
            start_traj_vec = trans.loc[
                np.roll(critical_trans, -1), "fixed_int"
            ].values  # -1 = fixation onset
            # now do combination of OR [for all transitions which lead to the ending of the trajectory]
            critical_trans = np.logical_or.reduce(
                (
                    (trans.MSG == "Reward").values,
                    (trans.MSG == "Punish").values,
                    (trans.MSG == "Invalid").values,
                )
            )
            end_traj_vec = trans.loc[
                critical_trans, "fixed_int"
            ].values  # no need to roll

        # remove this / adapt
        # this 'd be fine if framerate is constant
        if self.framestamps is None:
            self.pose[bodypart + "_v", "x"] = (
                self.pose[bodypart, "x"].shift(-1) - self.pose[bodypart, "x"]
            )
            self.pose[bodypart + "_v", "y"] = (
                self.pose[bodypart, "y"].shift(-1) - self.pose[bodypart, "y"]
            )
        else:
            tvec = self.framestamps.astype(float) / 1000  # in ms
            self.pose[bodypart + "_v", "x"] = np.gradient(
                self.pose[bodypart, "x"].values, tvec
            )
            self.pose[bodypart + "_v", "y"] = np.gradient(
                self.pose[bodypart, "y"].values, tvec
            )

        # drop BNCs from albert device: 60,61,62,63 ~ or following section will not work propperly
        # tag as dirty all trials not following this event pattern [drop BNCs events before] * [delay tasks will differ]
        if "delay" in self.target:
            # print('delay has not been implemented yet') # delay tasks hav different transitions (ie use delay rather than startsound, then look for the next1 or 2?) ~ fix later
            pattern_left_choice_sound = np.array(
                [
                    self.event_to_int[self.active_ports[1] + "In"],
                    104,
                    104,
                    self.event_to_int[self.active_ports[1] + "Out"],
                    self.event_to_int[self.active_ports[0] + "In"],
                ]
            )
            pattern_left_choice_nosound = np.array(
                [
                    self.event_to_int[self.active_ports[1] + "In"],
                    104,
                    self.event_to_int[self.active_ports[1] + "Out"],
                    self.event_to_int[self.active_ports[0] + "In"],
                ]
            )
            pattern_right_choice_sound = np.array(
                [
                    self.event_to_int[self.active_ports[1] + "In"],
                    104,
                    104,
                    self.event_to_int[self.active_ports[1] + "Out"],
                    self.event_to_int[self.active_ports[2] + "In"],
                ]
            )
            pattern_right_choice_nosound = np.array(
                [
                    self.event_to_int[self.active_ports[1] + "In"],
                    104,
                    self.event_to_int[self.active_ports[1] + "Out"],
                    self.event_to_int[self.active_ports[2] + "In"],
                ]
            )
        elif self.newSM:
            # TODO: then fix for feedbacksessions with fair stim
            # this should be main trial bulk, just adapt for reversing trials if they exist
            pattern_left_choice = np.array(
                [
                    self.event_to_int[self.active_ports[1] + "In"],
                    104,
                    104,
                    self.event_to_int[self.active_ports[1] + "Out"],
                    self.event_to_int[self.active_ports[0] + "In"],
                ]
            )
            pattern_right_choice = np.array(
                [
                    self.event_to_int[self.active_ports[1] + "In"],
                    104,
                    104,
                    self.event_to_int[self.active_ports[1] + "Out"],
                    self.event_to_int[self.active_ports[2] + "In"],
                ]
            )
            if list(self.switching_idx):
                pattern_left_choice_sw = np.array(
                    [
                        # central port in
                        self.event_to_int[self.active_ports[1] + "In"],
                        104,  # first part of fixation ends by timeup(feedback one)
                        104,  # 2nd fixation
                        # central part-out
                        self.event_to_int[self.active_ports[1] + "Out"],
                        45,  # softcode to switch
                        # left port port-in (choice)
                        self.event_to_int[self.active_ports[0] + "In"],
                    ]
                )
                pattern_right_choice_sw = np.array(
                    [
                        # central port in
                        self.event_to_int[self.active_ports[1] + "In"],
                        104,  # first part of fixation ends by timeup(feedback one)
                        104,  # 2nd fixation
                        # central part-out
                        self.event_to_int[self.active_ports[1] + "Out"],
                        45,  # softcode to switch
                        # right port port-in (choice)
                        self.event_to_int[self.active_ports[2] + "In"],
                    ]
                )
        else:
            pattern_left_choice = np.array(
                [
                    self.event_to_int[self.active_ports[1] + "In"],
                    104,
                    self.event_to_int[self.active_ports[1] + "Out"],
                    self.event_to_int[self.active_ports[0] + "In"],
                ]
            )
            pattern_right_choice = np.array(
                [
                    self.event_to_int[self.active_ports[1] + "In"],
                    104,
                    self.event_to_int[self.active_ports[1] + "Out"],
                    self.event_to_int[self.active_ports[2] + "In"],
                ]
            )

        # Not being used: delete after few comits (2020/04/08)
        # ma = (self.sess.TYPE == 'EVENT').size
        # mb = (~(self.sess.loc[self.sess.TYPE == 'EVENT',
        #                       '+INFO'].str.startswith('BNC'))).size
        bncmask = (self.sess.TYPE == "EVENT") & ~(
            self.sess.loc[self.sess.TYPE == "EVENT", "+INFO"].str.startswith("BNC")
        )

        seq = self.sess.loc[bncmask, "MSG"].astype(int).values

        if "delay" in self.target:
            L_pattern_mask_sound = np.all(
                chom.rolling_window(seq, 5) == pattern_left_choice_sound, axis=1
            )
            # index event, not real one
            L_choice_idx_sound = np.mgrid[0 : len(L_pattern_mask_sound)][
                L_pattern_mask_sound
            ]
            R_pattern_mask_sound = np.all(
                chom.rolling_window(seq, 5) == pattern_right_choice_sound, axis=1
            )
            R_choice_idx_sound = np.mgrid[0 : len(R_pattern_mask_sound)][
                R_pattern_mask_sound
            ]

            L_pattern_mask_nosound = np.all(
                chom.rolling_window(seq, 4) == pattern_left_choice_nosound, axis=1
            )
            L_choice_idx_nosound = np.mgrid[0 : len(L_pattern_mask_nosound)][
                L_pattern_mask_nosound
            ]  # index event, not real one
            R_pattern_mask_nosound = np.all(
                chom.rolling_window(seq, 4) == pattern_right_choice_nosound, axis=1
            )
            R_choice_idx_nosound = np.mgrid[0 : len(R_pattern_mask_nosound)][
                R_pattern_mask_nosound
            ]

            pool_choices_idx = np.sort(
                np.concatenate(
                    [
                        L_choice_idx_sound,
                        R_choice_idx_sound,
                        L_choice_idx_nosound,
                        R_choice_idx_nosound,
                    ]
                )
            )  # event based, not df_idx
        else:
            L_pattern_mask = np.all(
                chom.rolling_window(seq, pattern_left_choice.size)
                == pattern_left_choice,
                axis=1,
            )
            # index event, not real one
            L_choice_idx = np.mgrid[0 : len(L_pattern_mask)][L_pattern_mask]
            R_pattern_mask = np.all(
                chom.rolling_window(seq, pattern_right_choice.size)
                == pattern_right_choice,
                axis=1,
            )
            R_choice_idx = np.mgrid[0 : len(R_pattern_mask)][R_pattern_mask]

            if list(self.switching_idx):
                # extra patterns still missing
                L_pattern_mask_sw = np.all(
                    chom.rolling_window(seq, pattern_left_choice_sw.size)
                    == pattern_left_choice_sw,
                    axis=1,
                )
                # index event, not real one
                L_choice_idx_sw = np.mgrid[0 : len(L_pattern_mask_sw)][
                    L_pattern_mask_sw
                ]
                R_pattern_mask_sw = np.all(
                    chom.rolling_window(seq, pattern_right_choice_sw.size)
                    == pattern_right_choice_sw,
                    axis=1,
                )
                R_choice_idx_sw = np.mgrid[0 : len(R_pattern_mask_sw)][
                    R_pattern_mask_sw
                ]
                pool_choices_idx = np.sort(
                    np.concatenate(
                        [L_choice_idx, R_choice_idx, L_choice_idx_sw, R_choice_idx_sw]
                    )
                )
            else:
                pool_choices_idx = np.sort(
                    np.concatenate([L_choice_idx, R_choice_idx])
                )  # event based, not df_idx

        triali = self.sess.loc[bncmask, "trial_idx"].iloc[pool_choices_idx].values

        # now dirty trajectories
        to_inspect = [
            int(x)
            for x in np.arange(1, len(self.trial_sess) + 1, 1).tolist()
            if x not in triali.tolist()
        ]
        self.dirty_trajectories_trials = np.array(to_inspect)

        # get frames based on transitions aka trans in startf and endf
        self.trajectories = pd.DataFrame(
            np.array([start_traj_vec, end_traj_vec]).T,
            index=np.arange(1, end_traj_vec.size + 1),
            columns=["startf", "endf"],
        ).T.to_dict()  # all trajectories now, just arange

        # now save fixation body coords and confidences
        if self.framestamps is not None:
            totlen = (self.framestamps[-1] - self.framestamps[0]).astype(int) / 1000000
            totframes = self.framestamps.size
            fps = totframes / totlen
            # around 150ms in fixation
            offsetfixframes = int(round((0.15 * fps)))
            fix_coords = []
            fix_conf = []
        # add trajectories to trial_sess (at least y ones), so we can apply vectorized find_com

        traj_y_list = []
        traj_x_list = []
        traj_vy_list = []
        traj_stamps = []
        init_f = (
            []
        )  # getting initial and last frames might help scalating this to db / slicing / retrieving it from original hdf5 etc.
        last_f = []

        # old stuff, try vectorized and get all coords (even for weird trials, evaluate later the trials you please) # cannot really vectorize it :(
        for trial in range(
            start_traj_vec.size
        ):  # beware offset+2 always!! ~ this should be removed!!!!!!
            try:
                init_f += [start_traj_vec[trial]]
                last_f += [end_traj_vec[trial]]
                traj_y_list += [
                    np.array(
                        self.pose.iloc[
                            start_traj_vec[trial]
                            + noffset_frames : end_traj_vec[trial]
                            + noffset_frames
                            + 6
                        ][bodypart, "y"].values
                    )
                ]  # test, remove?
                # where does this +6 comes from?
                # hoh huge bug solved
                traj_x_list += [
                    np.array(
                        self.pose.iloc[
                            start_traj_vec[trial]
                            + noffset_frames : end_traj_vec[trial]
                            + noffset_frames
                            + 6
                        ][bodypart, "x"].values
                    )
                ]
                traj_vy_list += [
                    np.array(
                        self.pose.iloc[
                            start_traj_vec[trial]
                            + noffset_frames : end_traj_vec[trial]
                            + noffset_frames
                            + 6
                        ][bodypart + "_v", "y"].values
                    )
                ]
                if self.framestamps is not None:
                    traj_stamps += [
                        np.array(
                            self.framestamps[
                                start_traj_vec[trial]
                                + noffset_frames : end_traj_vec[trial]
                                + noffset_frames
                                + 6
                            ]
                        )
                    ]
            except:
                traj_y_list += [np.empty(0)]
                traj_vy_list += [np.empty(0)]
                traj_x_list += [np.empty(0)]
                traj_stamps += [np.empty(0)]
                init_f += [-1]
                last_f += [-1]

        # fix_coords = list()
        if "delay" not in self.target:
            fix_f_ind = trans.loc[trans.MSG == "StartSound", "fixed_int"].values
        else:
            # hard to know in irregular tass. Revisit
            fix_f_ind = trans.loc[trans.MSG == "Delay", "fixed_int"].values

        # alternatively generate a vector for fixation lengths and adjust offsetfixtrames instead of broadcasting a single value
        fix_f_ind = (fix_f_ind - offsetfixframes).astype(int)

        fix_coords = list(
            self.pose.loc[
                fix_f_ind,
                (
                    [
                        "isnout",
                        "rL-eye",
                        "rR-eye",
                        "rL-ear",
                        "rR-ear",
                        "rneck",
                        "rback",
                        "rtail",
                        "rabove-snout",
                    ],
                    ["x", "y"],
                ),
            ].values
        )
        fix_conf = list(
            self.pose.loc[
                fix_f_ind,
                (
                    [
                        "snout",
                        "L-eye",
                        "R-eye",
                        "L-ear",
                        "R-ear",
                        "neck",
                        "back",
                        "tail",
                        "above-snout",
                    ],
                    ["likelihood"],
                ),
            ].values
        )

        # list is not shortenned
        self.trial_sess["trajectory_y"] = traj_y_list[: self.trial_sess.shape[0]]
        self.trial_sess["trajectory_vy"] = traj_vy_list[: self.trial_sess.shape[0]]
        # not centered to 0
        self.trial_sess["trajectory_x"] = traj_x_list[: self.trial_sess.shape[0]]
        # also save frame index
        self.trial_sess["vidfnum_0"] = init_f[: self.trial_sess.shape[0]]
        self.trial_sess["vidfnum_f"] = last_f[: self.trial_sess.shape[0]]

        if self.framestamps is not None:
            self.trial_sess["trajectory_stamps"] = traj_stamps[
                : self.trial_sess.shape[0]
            ]
            self.trial_sess["fix_coords"] = fix_coords[: self.trial_sess.shape[0]]
            self.trial_sess["fix_conf"] = fix_conf[: self.trial_sess.shape[0]]
        else:
            self.trial_sess["trajectory_stamps"] = np.nan
            # wth, are they related to gd timestamps?
            self.trial_sess["fix_coords"] = np.nan
            self.trial_sess["fix_conf"] = np.nan

        # calcbodyangle and head ~ there are a lot of nans in the result
        self.trial_sess["bodyangle"] = (
            self.trial_sess.fix_coords.apply(
                lambda x: chom.get_bodypart_angle(x, -4, -3, -6, -5)
            )
            * -1
        )  # not really easy to understand: hardcoded coordinates indexes (ax=1)
        self.trial_sess["headangle"] = (
            self.trial_sess.fix_coords.apply(
                lambda x: chom.get_bodypart_angle(x, -6, -5, 0, 1)
            )
            * -1
        )  # *-1 because angle was reverted, so towards left now is negative

        tempvec = np.repeat(False, len(self.trial_sess))
        tempvec[(np.array(to_inspect) - 1).astype(int)] = True
        self.trial_sess["dirty"] = tempvec

        if fixationbreaks and not self.newSM:  # ('_noenv' not in self.target)
            # now transitions should have frame assigned
            # trans.loc[:,'fixed_int']=trans.loc[:,'fixed_int'].astype(int)
            # trans.loc[:,'fixed_int'].fillna(method='ffill', inplace=True) # having some issue and weird floats -?!$! # does not solve issue

            # trans.loc[:,'fixed_int']=trans.loc[:,'fixed_int'].astype(int)
            # now get sequences
            # should be the same, remove bnc as well
            fbpat = np.array(["Fixation", "WaitCPoke"])
            seq = trans.loc[:, "MSG"].values
            # seq = trans.loc[bncmask,'MSG'].values
            fbmask = np.all(chom.rolling_window(seq, 2) == fbpat, axis=1)
            # iloc of FB in transition-only frame
            fb_idx = np.mgrid[0 : len(fbmask)][fbmask]
            # i think here's the heavy matrix

            # trans index, not iloc [i.e valid row-index for all self.sess]
            sess_fb_ix_s = trans.loc[:, "MSG"].iloc[fb_idx].index.values
            # sess_fb_ix_s=trans.loc[bncmask, 'MSG'].iloc[fb_idx].index.values # trans index, not iloc [i.e valid row-index for all self.sess]
            # Potential bug here | seems ok to add up 2 because after a fixationbreak transition there's only 1 event ->|| no, the animal can poke somewhere else # sess_fb_ix_e=sess_fb_ix_s + 2
            # sess_fb_ix_e=sess_fb_ix_s + 2 # it's ok because transition +1 is waitcpoke +2= fixation again
            sess_fb_ix_e = trans.loc[:, "MSG"].iloc[fb_idx + 2].index.values
            # sess_fb_ix_e=trans.loc[bncmask, 'MSG'].iloc[fb_idx+2].index.values

            self.trial_sess["fb_traj"] = [np.array([])] * self.trial_sess.shape[0]
            self.trial_sess["fb_fidx"] = [np.array([])] * self.trial_sess.shape[0]

            self.trial_sess.loc[
                self.trial_sess.fb.apply(lambda x: len(x) > 0), "fb_fidx"
            ] = (
                trans.loc[sess_fb_ix_s].groupby(["trial_idx"])["fixed_int"].apply(list)
                + trans.loc[sess_fb_ix_e]
                .groupby(["trial_idx"])["fixed_int"]
                .apply(list)
            ).values

            # display(self.trial_sess.loc[self.trial_sess.fb_fidx.apply(len)>0,['fb_traj', 'fb_fidx']])
            self.trial_sess.fb_fidx = self.trial_sess.fb_fidx.apply(
                lambda x: chom.rearrange_fbidx(x)
            )  # TODO: is this memory efficient?
            self.trial_sess.fb_traj = self.trial_sess.fb_fidx.apply(
                lambda x: chom.populate_fb_traj(x, self.pose, part=bodypart)
            )  # here's the 'looping'
        # elif fixationbreaks and ('_noenv' in self.target):
        #    raise NotImplementedError  # idk whether to notimplemented error or simply not implemented


    def scatter_traj(self, idx, scatter_kwargs={}):
        """deprecated
        plots trajectory of trial(idx, dataframe index, not session) from dataframe(df)"""
        if not self.processed:
            raise ValueError("process data first")

        y = self.trial_sess.loc[idx, "trajectory_y"]
        x = self.trial_sess.loc[idx, "trajectory_stamps"].copy().astype(int) / 1000
        x = x - x[0]
        plt.scatter(x, y, **scatter_kwargs)
        plt.xlabel("ms")
        plt.ylabel("px")

        plt.show()

    # derp-precated

    # trialn = 1-indexed
    def plot_trajectory(
        self, trialn=None, background=None, savpath=None, part="isnout"
    ):
        print("deprecated function, probably won't work as intended/used to")
        # retrieve cum initial * 1000, adapt x-ax. align port2in to 0
        # background is a path to img
        if part == "isnout":
            conf_part = "snout"
        else:
            conf_part = part[1:]
        if trialn is None:
            trialn = np.random.choice(np.arange(1, len(self.trial_sess) + 1, 1))
        # a = self.trajectories[]
        # b = extract_trajectory(a[0], a[1], pose, bodypart='isnout')
        b = self.pose.iloc[
            self.trajectories[trialn]["startf"] : self.trajectories[trialn]["endf"]
        ][part].values
        conf = (
            1
            - self.pose.iloc[
                self.trajectories[trialn]["startf"] : self.trajectories[trialn]["endf"]
            ][conf_part].values[:, 2]
        )
        # dont!!! get frames by integers above (b) but arrange on x axis(ms) based on fixed_frames_ms / cum_initial
        if self.framestamps is None:
            x_frame_offset = (
                self.sess.loc[
                    (self.sess.TYPE == "EVENT")
                    & (self.sess.fixed_int == self.trajectories[trialn]["startf"]),
                    "cum_initial",
                ].values[0]
            ) % 1
            x_ms_offset = x_frame_offset * (1000 / self.fixed_framerate)
        else:  # REVIEW THOSE OFFSETS
            x_frame_offset = (
                self.sess.loc[
                    (self.sess.TYPE == "EVENT")
                    & (self.sess.fixed_int == self.trajectories[trialn]["startf"]),
                    "cum_initial",
                ].values[0]
            ) % 1
            x_ms_offset = x_frame_offset * (1000 / self.fixed_framerate)
        # pending: fix lines below
        bms_i = (
            self.sess.loc[
                (self.sess.TYPE == "EVENT")
                & (self.sess.fixed_int == self.trajectories[trialn]["startf"]),
                "cum_initial",
            ].values[0]
            * 1000
        )
        bms_f = (
            self.sess.loc[
                (self.sess.TYPE == "EVENT")
                & (self.sess.fixed_int == self.trajectories[trialn]["endf"]),
                "cum_initial",
            ].values[0]
            * 1000
        )

        spanms = bms_f - bms_i
        bxf = np.arange(
            self.trajectories[trialn]["startf"], self.trajectories[trialn]["endf"], 5
        )  # frames x
        bxf = bxf - bxf[0]
        bxms = bxf * self.fixed_framerate
        # def extract_trajectory(startf, endf, df, bodypart='isnout'):
        #'''this should return ndarray with shape (3, nframes) containing both, initial and final'''
        # return df.iloc[startf:endf+1][bodypart].values
        fig = plt.figure(0, figsize=(12, 8))
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        ax0 = plt.subplot2grid((4, 2), (0, 0), rowspan=3)
        # if self.normcoords:
        #    ax0.invert_yaxis()
        ax1 = plt.subplot2grid((4, 2), (0, 1), rowspan=3, sharey=ax0)
        ax3 = plt.subplot2grid((4, 2), (3, 1), sharex=ax1)  # sharey=ax1
        ax2 = plt.subplot2grid((4, 2), (3, 0), sharey=ax3)

        # apply slicing by xlim and ylim
        if background is not None:
            ax0.imshow(np.flipud(plt.imread(background)), cmap="gray")
            # ax0.set_xlim([350,580])
            if not self.normcoords:
                ax0.set_xlim([350, 580])
        # ax1[0].plot(b[:,0],480-b[:,1]) # check whether to invert or not

        # tweak here
        # get info from self.trial_sess
        trial_to_plot_trialdf = self.trial_sess[self.trial_sess.origidx == trialn]
        # display(trial_to_plot_trialdf)
        soundonset = trial_to_plot_trialdf["soundonset"].values[0] * 1000
        trialonset = trial_to_plot_trialdf["trialonset"].values * 1000
        fixationonset = (trialonset + soundonset) - 3000  # ?? why

        # trial_to_plot_sessdf = self.sess[self.sess.trial_idx==trialn]
        if self.normcoords:
            ytoplot = b[:, 1]
            # ax0.invert_yaxis()

        else:
            ytoplot = 480 - b[:, 1]

        ax0.plot(b[:, 0], ytoplot, color="gray")
        # ax1[0].axis('off')
        ax0.set_ylabel("y(px)")
        ax0.set_xlabel("x (px)")
        # ax1[0].set_xticklabels(bxf,bxms)

        # prev seaborn
        # sns.scatterplot(b[:,0],480-b[:,1],hue=np.arange(len(b[:,0])),legend='brief',palette='coolwarm',s=80,ax=ax0, cbar=True)
        timeplot = ax0.scatter(
            b[:, 0], ytoplot, c=np.arange(len(b[:, 0])), s=60, cmap="viridis"
        )
        # lt.
        ayyy = self.trajectories[trialn]["startf"]
        # ax0.set_title(f"{self.target}, trial {trialn} f# {ayyy}")
        # ax0.set_xlim([300,580])
        if self.normcoords:
            ax0.set_xlim([-140, 40])
            ax0.set_ylim([-100, 100])
        else:
            ax0.set_xlim([320, 600])
            ax0.set_ylim([130, 410])  # scale video ## definethem elsewhere
        target_env = self.trial_sess.iloc[trialn - 1]  # !!!! it was wrong...
        sound_len = target_env["sound_len"]  # correct?
        lenv = target_env["lenv"].copy()
        renv = target_env["renv"].copy()
        res_sound = target_env["res_sound"].copy()
        frames_listened = target_env["frames_listened"]

        # PENDING, ADD dirty events ~ more or less done (just 1st order dirty traj)
        # get ax1 with ms. (-300 bpod timestamp port2in)
        # changed first pair to get it working on both cond
        ax1.fill([-300, 0, 0, -300], [-130, -130, 410, 410], alpha=0.2, color="blue")
        ax1.fill(
            [0, sound_len, sound_len, 0],
            [-130, -130, 410, 410],
            alpha=0.2,
            color="magenta",
        )
        # dont!!! get frames by integers above (b) but arrange on x axis(ms) based on fixed_frames_ms / cum_initial

        # pending: fix lines below
        # ax1.plot(np.linspace(-300,spanms-300, len(b[:,1])),480-b[:,1], color='black',marker='o', markersize=3)
        xtoplot = x_ms_offset + np.linspace(-300, spanms - 300, len(b[:, 1]))

        ax1.fill_between(
            xtoplot,
            ytoplot + (20 * conf),
            ytoplot - (20 * conf),
            alpha=0.5,
            color="orange",
        )
        ax1.plot(xtoplot, ytoplot, color="black", marker="o", markersize=3)
        # ax1.set_ylim([190,390]) # this one was pwning alignement

        # ax1.axvline on lateral poke in position
        # bms_i = self.sess.loc[(self.sess.TYPE=='EVENT')&(self.sess.fixed_int==self.trajectories[trialn]['startf']), 'cum_initial'].values[0]*1000
        if trialn in self.dirty_trajectories_trials.tolist():  # choice-port in iloc = 5
            lportin = (
                self.sess[
                    ~(
                        self.sess.loc[
                            self.sess.TYPE == "EVENT", "+INFO"
                        ].str.startswith("BNC")
                    )
                    & (self.sess.TYPE == "EVENT")
                    & (self.sess.fixed_int >= self.trajectories[trialn]["startf"])
                ].iloc[5]["cum_initial"]
                * 1000
            )
            dirty1 = (
                self.sess[
                    ~(
                        self.sess.loc[
                            self.sess.TYPE == "EVENT", "+INFO"
                        ].str.startswith("BNC")
                    )
                    & (self.sess.TYPE == "EVENT")
                    & (self.sess.fixed_int >= self.trajectories[trialn]["startf"])
                ].iloc[3]["cum_initial"]
                * 1000
            )
            dirty2 = (
                self.sess[
                    ~(
                        self.sess.loc[
                            self.sess.TYPE == "EVENT", "+INFO"
                        ].str.startswith("BNC")
                    )
                    & (self.sess.TYPE == "EVENT")
                    & (self.sess.fixed_int >= self.trajectories[trialn]["startf"])
                ].iloc[4]["cum_initial"]
                * 1000
            )
            ax1.axvline(dirty1 - bms_i - 300, ls="--")
            ax1.axvline(dirty2 - bms_i - 300, ls="--")
            # ev_loc = 5 # there's a pin and poke.out in mid port
        # elif self.newSM: # has an extra state so it should be 4th position
        #    lportin = self.sess[(self.sess.TYPE=='EVENT')&(self.sess.fixed_int>=self.trajectories[trialn]['startf'])].iloc[3]['cum_initial']*1000
        else:  # choice-port in iloc = 3 | drop bnc events to keep it
            lportin = (
                self.sess[
                    ~(
                        self.sess.loc[
                            self.sess.TYPE == "EVENT", "+INFO"
                        ].str.startswith("BNC")
                    )
                    & (self.sess.TYPE == "EVENT")
                    & (self.sess.fixed_int >= self.trajectories[trialn]["startf"])
                ].iloc[3]["cum_initial"]
                * 1000
            )

        # lportin iloc should be 5 for dirty sequences
        # if self.newSM:
        #     ax1.axvline(lportin-bms_i-300)
        # else:
        ax1.axvline(lportin - bms_i - 300)
        # self.trajectories[trialn]['startf']
        ax1.set_xlabel("time(ms)")
        ax1.set_ylabel("snout position (y px)")
        # ax1.set_xticks(bxf)
        # ax1.set_xticklabels([str(round(x,0)) for x in bxms.tolist()])

        # transform vars to strings
        # no invalid trial should reach this point
        side = "Right" if target_env.R_response else "Left"
        hit = "Hit" if target_env.hithistory == 1 else "Miss"
        rep = "repeated choice" if target_env.rep_response else "alternated choice"
        ax2.annotate(f"Response: {side} | {hit} |  {rep}", (0, -1))  # \n\n"+\
        ax2.annotate(
            f"session align 0 = {int(round(target_env.trialonset+target_env.soundonset,0))}s | stim coh: {target_env.coh*2-1} | stim len: {int(round(target_env.sound_len, 0))} ms",
            (0, -0.5),
        )  # \n\n"+\
        ax2.annotate(
            f"p(rep) = {target_env.prob_repeat} | block idx: {target_env.bl_idx} | within block idx: {target_env.wibl_idx}",
            (0, 0),
        )  # \n\n"+\
        ax2.annotate(
            f"prev streak = {int(target_env.streak)} | prev fb: {len(target_env.fb)}",
            (0, 0.5),
        )
        if target_env.Hesitation or target_env.dirty:
            note = ""
            if target_env.Hesitation:
                note += "Hesitation+ ; "
            if target_env.CoM_sugg:
                note += "suggested change of mind; "
            if target_env.dirty:
                note += "Dirty trajectory"
            ax2.annotate(note, (0, 1))
        ax2.axis("off")

        # Envelope fig
        xvec = np.arange(0, 1000, 50)
        fullframes = int(frames_listened)
        partial_frame = frames_listened % 1
        ax3.axhline(0, color="k", linestyle=":")
        if "noenv" in self.target:  # TODO: add 12.5ms ramp
            ax3.plot([0, sound_len], [lenv] * 2, c="green")
            ax3.plot([0, sound_len], [renv] * 2, c="purple")
        else:
            ax3.axhline(0, color="k", linestyle=":")
            ax3.bar(
                xvec[:fullframes],
                lenv[:fullframes],
                width=50,
                edgecolor="k",
                align="edge",
                color="orange",
            )
            ax3.bar(
                xvec[:fullframes],
                renv[:fullframes],
                width=50,
                edgecolor="k",
                align="edge",
                color="orange",
            )
            shiftpos = 50 / 2
            ax3.bar(
                (xvec + shiftpos)[:fullframes],
                res_sound[:fullframes],
                width=40,
                edgecolor="k",
                align="center",
                color="navy",
            )

            ax3.bar(
                xvec[fullframes],
                lenv[fullframes],
                width=50 * partial_frame,
                edgecolor="k",
                align="edge",
                color="orange",
                label="envelope",
            )
            ax3.bar(
                xvec[fullframes],
                renv[fullframes],
                width=50 * partial_frame,
                edgecolor="k",
                align="edge",
                color="orange",
            )
            shiftpos = (50 * partial_frame) / 2

            ax3.bar(
                (xvec + shiftpos)[fullframes],
                res_sound[fullframes],
                width=40 * partial_frame,
                edgecolor="k",
                align="center",
                color="navy",
                label="mean env",
            )

            # perfect integrator
            ax3.plot(
                np.arange(0, int(sound_len)),
                (np.nancumsum(np.repeat(res_sound, 50)) / np.arange(1, 1001))[
                    : int(sound_len)
                ],
                color="r",
                label="p.i.",
                linewidth=3.0,
            )
            if int(sound_len) > 80:
                ax3.plot(
                    np.arange(80, int(sound_len)),
                    (np.nancumsum(np.repeat(res_sound, 50)) / np.arange(1, 1001))[
                        : int(sound_len) - 80
                    ],
                    color="green",
                    label="p.i.-80ms",
                    linewidth=3.0,
                )

            ax3.set_ylim([-1, 1])
            ax3.set_xlim([-350, None])
            ax3.set_yticks([-1, 0, 1])
            ax3.set_yticklabels(["L", "", "R"])
            # ax3.set_yticks([-1,0,1],['L', ' ', 'R'])
            ax3.legend(loc=2)
        ax3.invert_yaxis()
        if self.normcoords:
            ax1.invert_yaxis()

        plt.suptitle(f"{self.target}, trial {trialn} f# {ayyy}")
        # rect [left, bottom, width, height]
        cax = fig.add_axes([0.07, 0.35, 0.05, 0.55])
        plt.colorbar(timeplot, cax=cax)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        if savpath:
            plt.savefig(f"{savpath}{self.target}_t{int(trialn):04}.png")
        else:
            plt.show()

    @staticmethod
    # beware output changes dramatically attending to framerate
    def speed_inflex(speedvec, soundlen, fps=30, delay=False, hotfix=False):
        if not hotfix:
            # sf1 = -fps*0.085+12.5
            # sf1 = -fps*0.07+10 # works quite ok
            # lower a little bit?
            sf1 = -fps * 0.06 + 8
            # before it was : fixation_offset = int(0.275*fps)

            # can lead to : 'cannot convert float NaN to integer', 'occurred at index 0'
            fixation_offset = int(((300 + soundlen) / 1000) * fps)
            # consec_frames_threshold = int(np.round(fps*0.06))
            consec_frames_threshold = (
                2  # this should change according to framerate ! adjust
            )
            # smooth or not? try it out
            speedvec = pd.Series(speedvec).rolling(window=2).mean().iloc[1:].values
            try:  # ignore first few fixation frames
                condition1 = (
                    np.sign(np.round(speedvec[fixation_offset : -int(0.1 * fps)] / sf1))
                ) < 0
                condition2 = (
                    np.sign(np.round(speedvec[fixation_offset : -int(0.1 * fps)] / sf1))
                ) > 0
                a = (
                    (
                        np.diff(
                            np.where(
                                np.concatenate(
                                    (
                                        [condition1[0]],
                                        condition1[:-1] != condition1[1:],
                                        [True],
                                    )
                                )
                            )[0]
                        )[::2]
                    )
                    >= consec_frames_threshold
                ).sum()
                b = (
                    (
                        np.diff(
                            np.where(
                                np.concatenate(
                                    (
                                        [condition2[0]],
                                        condition2[:-1] != condition2[1:],
                                        [True],
                                    )
                                )
                            )[0]
                        )[::2]
                    )
                    >= consec_frames_threshold
                ).sum()
                return bool(bool(a) & bool(b))
            except:
                return False  # except because it crashes if there is no trajectory or no trues?
        else:
            # we are getting speedvec in px/ms and we'll consider valid accel over abs(0.1)
            raise NotImplementedError("wrote a new function")

    def speed_inflex_delay(speedvec, soundlen, delay, special_trial, fps=30):
        sf1 = -fps * 0.06 + 8
        if special_trial == -1:  # TODO: not true with noenv. Review
            fix_time = 150
        else:
            fix_time = 300
        fixation_offset = int(
            ((np.nansum(np.array([fix_time, soundlen, delay]))) / 1000) * fps
        )
        consec_frames_threshold = 2
        speedvec = pd.Series(speedvec).rolling(window=2).mean().iloc[1:].values
        try:  # ignore first few fixation frames
            condition1 = (
                np.sign(np.round(speedvec[fixation_offset : -int(0.1 * fps)] / sf1))
            ) < 0
            condition2 = (
                np.sign(np.round(speedvec[fixation_offset : -int(0.1 * fps)] / sf1))
            ) > 0
            a = (
                (
                    np.diff(
                        np.where(
                            np.concatenate(
                                (
                                    [condition1[0]],
                                    condition1[:-1] != condition1[1:],
                                    [True],
                                )
                            )
                        )[0]
                    )[::2]
                )
                >= consec_frames_threshold
            ).sum()
            b = (
                (
                    np.diff(
                        np.where(
                            np.concatenate(
                                (
                                    [condition2[0]],
                                    condition2[:-1] != condition2[1:],
                                    [True],
                                )
                            )
                        )[0]
                    )[::2]
                )
                >= consec_frames_threshold
            ).sum()
            return bool(bool(a) & bool(b))
        except:
            return False

    # above functions are suboptimal, newer below (also, works with simulated trajectories)
    @staticmethod
    def did_he_hesitate(
        row,
        thr=0.1,
        positioncol="trajectory_y",
        speedcol="trajectory_vy",
        consec_frames_threshold=2,
        simul=False,
        height=5,
    ):
        """this new version should work with simulations as well,
        row is a df row (so we can .apply() this function)
        requires that speeds were calculated using timestamps (px/ms),
        
        it also returns wether CoM or not (and CoM peak frame)"""
        hesitation = False
        if simul:
            try:
                speedvec = row[speedcol]  # already aligned with movement onset.
                speedvec = speedvec[:-50]  # remove last 50ms
                dist = 100
                onset = 0
                offset = speedvec.size  # - 50
            except:
                return hesitation, False, np.nan
        else:
            fixtime = 300
            delay = 0
            if row["special_trial"] != 0:
                if row.special_trial == -1:  # TODO: not true with noenv. Review
                    fixtime = 150
                delay = int(row.delay_len)
            sp = row[speedcol]  # speedvec
            t = (row.trajectory_stamps - row.fix_onset_dt.to_datetime64()).astype(
                int
            ) / 1000_000
            dist = int(t.size * 100 / (t.max() - t.min()))
            t = (
                t - fixtime - delay - row.sound_len
            )  # timestamps adjusted to movement onset (0)
            # get indices to slice,
            onset = np.argmax(t > 0) - 1
            offset = np.argmax(t > (row.resp_len * 1000 - 50)) - 1
            speedvec = np.sign((sp / thr).astype(int))[
                onset:offset
            ]  # already sliced containing {-1, 1} if above thr, else  0

        try:  # ignore first few fixation frames
            condition1 = speedvec < 0
            condition2 = speedvec > 0
            a = (
                (
                    np.diff(
                        np.where(
                            np.concatenate(
                                (
                                    [condition1[0]],
                                    condition1[:-1] != condition1[1:],
                                    [True],
                                )
                            )
                        )[0]
                    )[::2]
                )
                >= consec_frames_threshold
            ).sum()
            b = (
                (
                    np.diff(
                        np.where(
                            np.concatenate(
                                (
                                    [condition2[0]],
                                    condition2[:-1] != condition2[1:],
                                    [True],
                                )
                            )
                        )[0]
                    )[::2]
                )
                >= consec_frames_threshold
            ).sum()

            hesitation = bool(bool(a) & bool(b))
            if not hesitation:
                return hesitation, False, np.nan
            else:
                # calc CoM and peakframe (copying com_or_not fcuntion)
                traj = row[positioncol]
                if simul:
                    yoffset = traj[0]
                    sliced_traj = traj[onset:offset] - yoffset
                    rolling1 = 0
                else:
                    yoffset = traj[int(onset * 0.5) : onset].mean()
                    sliced_traj = traj[onset:offset] - yoffset
                    sliced_traj = (
                        pd.Series(sliced_traj).rolling(window=2).mean().iloc[1:].values
                    )
                    rolling1 = 1

                # TODO: finnish this so it can return CoM true or false!
                if row.R_response > 0:
                    # get idx for peaks (*-1 because in this sidewe want the more negative values, aka minima)
                    opposite_side_peak = find_peaks(
                        -1 * sliced_traj, distance=dist, height=height
                    )
                    same_side_peak = find_peaks(
                        1 * sliced_traj, distance=dist, height=height
                    )
                else:
                    opposite_side_peak = find_peaks(
                        sliced_traj, distance=dist, height=height
                    )
                    same_side_peak = find_peaks(
                        -1 * sliced_traj, distance=dist, height=height
                    )

                if len(opposite_side_peak[0]) > 0:
                    if len(same_side_peak[0]) > 0:
                        # first choice = last, hence hesitation+ but not com
                        if (
                            same_side_peak[0][0] < opposite_side_peak[0][0]
                            and np.abs(sliced_traj[same_side_peak[0][0]]) > 5
                        ):
                            # last comparison is an arbitrary threshold
                            return hesitation, False, np.nan
                    targ = sliced_traj[
                        np.concatenate(
                            [opposite_side_peak[0].flatten(), np.array([-1])]
                        ).astype(int)
                    ]  # idxes
                    targ = np.sign((targ / 2).astype(int))
                    if np.any(targ < 0) and np.any(targ > 0):
                        try:
                            return (
                                hesitation,
                                True,
                                onset + opposite_side_peak[0] + rolling1,
                            )  # adding an extra idnex if we use rolling (w=2)
                        except Exception as e:
                            # raise e
                            # print(fixsound_framespan+1+opposite_side_peak[0])
                            # pass
                            # return True
                            return (
                                hesitation,
                                False,
                                np.nan,
                            )  # somehow there are still empty simul traj
                    else:
                        return hesitation, False, np.nan
                else:
                    # print('no opposite peaks')
                    return hesitation, False, np.nan
        except Exception as e:
            # raise e
            return hesitation, False, np.nan

    # TODO: complete
    """
    def CoM_or_not(traj,slen,resp_side, fps=30 ): # wont work as it is now whenever normcoord = False
        fixsound_framespan=int((300+slen)/(1000/fps))
        sliced_traj = traj[fixsound_framespan:]
        sliced_traj = pd.Series(sliced_traj).rolling(window=2).mean().iloc[1:].values
        if resp_side>0: # Right response (+pixel values in right port), hence we'll look for peaks in inverse traj
            opposite_side_peak = find_peak(-1*sliced_traj, distance=int(fps/10)) # get idx for peaks (*-1 because in this sidewe want the more negative values, aka minima)
        else:
            oppsite_side_peak = find_peak(sliced_traj, distance=int(fps/10))
        if len(opposite_side_peak[0])>0:

            targ = sliced_traj[np.concatenate([opposite_side_peak[0],sliced_traj[-1]])]
            targ = np.sign((targ/5).astype(int))
            print(targ)
            if (targ.any()<0) and (targ.any()>0):
                return True
            else:
                return False
        else:
            return False
    """

    @staticmethod
    def CoM_or_not(
        traj, slen, resp_side, fps=30
    ):  # wont work as it is now whenever normcoord = False
        # adapt this to work with framestamps
        fixsound_framespan = int((300 + slen) / (1000 / fps))
        yoffset = traj[
            int(fixsound_framespan - (0.15 * fps)) : fixsound_framespan
        ].mean()  # is this better-??
        sliced_traj = traj[fixsound_framespan:] - yoffset
        sliced_traj = pd.Series(sliced_traj).rolling(window=2).mean().iloc[1:].values
        dist = int(fps / 10)
        if dist == 0:
            dist = 1
        # Right response (+pixel values in right port), hence we'll look for peaks in inverse traj
        if resp_side > 0:
            # get idx for peaks (*-1 because in this sidewe want the more negative values, aka minima)
            opposite_side_peak = find_peaks(-1 * sliced_traj, distance=dist)
            same_side_peak = find_peaks(1 * sliced_traj, distance=dist)
        else:
            opposite_side_peak = find_peaks(sliced_traj, distance=dist)
            same_side_peak = find_peaks(-1 * sliced_traj, distance=dist)
        if len(opposite_side_peak[0]) > 0:
            # print(f'sameside peak: {same_side_peak}\noppositeside peak: {opposite_side_peak}')
            # print('opposite_side_peak',opposite_side_peak[0].shape ,opposite_side_peak[0])
            # print(type(sliced_traj[-1]))
            if len(same_side_peak[0]) > 0:
                # first choice = last, hence hesitation+ but not com
                if (
                    same_side_peak[0][0] < opposite_side_peak[0][0]
                    and np.abs(sliced_traj[same_side_peak[0][0]]) > 5
                ):
                    # last comparison is an arbitrary threshold
                    return [False, np.nan]
            # print(f'opp: {opposite_side_peak[0].shape}, sliced_traj_full {len(sliced_traj)}')
            # print(np.array(sliced_traj[-1]).shape)
            # targ = sliced_traj[np.concatenate([opposite_side_peak[0].flatten(),np.array([sliced_traj[-1]])]).astype(int)] # indexes
            targ = sliced_traj[
                np.concatenate(
                    [opposite_side_peak[0].flatten(), np.array([-1])]
                ).astype(int)
            ]  # idxes
            # targ = np.sign((targ/5).astype(int))
            # removing /5 because it has already been filtered when selecting hesitation
            targ = np.sign((targ / 2).astype(int))
            # print(targ)
            # print(targ)
            # print('any < 0',(targ<0).any())
            # print('any > 0',np.any(targ>0))
            if np.any(targ < 0) and np.any(targ > 0):
                try:
                    # return it as whole len trajectory index (+1 missing because of the rolling smooth?)
                    return [True, fixsound_framespan + 1 + opposite_side_peak[0]]
                except:
                    # print(fixsound_framespan+1+opposite_side_peak[0])
                    pass
                # return True
            else:
                return [False, np.nan]
        else:
            return [False, np.nan]

    def CoM_or_not_delay(traj, slen, resp_side, delaylen, ttype, fps=30):
        """pending"""
        if ttype < 0:  # this is for early
            fixtime = 150
        else:
            fixtime = 300

        dist = int(fps / 10)
        if dist == 0:
            dist = 1
        # approxtime within port
        fixsound_framespan = int(np.nansum([fixtime + slen + delaylen]) / (1000 / fps))
        yoffset = traj[
            int(fixsound_framespan - (0.15 * fps)) : fixsound_framespan
        ].mean()  # is this better-??
        sliced_traj = traj[fixsound_framespan:] - yoffset
        sliced_traj = pd.Series(sliced_traj).rolling(window=2).mean().iloc[1:].values
        # Right response (+pixel values in right port), hence we'll look for peaks in inverse traj
        if resp_side > 0:
            # get idx for peaks (*-1 because in this sidewe want the more negative values, aka minima)
            opposite_side_peak = find_peaks(-1 * sliced_traj, distance=dist)
            same_side_peak = find_peaks(1 * sliced_traj, distance=dist)
        else:
            opposite_side_peak = find_peaks(sliced_traj, distance=dist)
            same_side_peak = find_peaks(-1 * sliced_traj, distance=dist)
        if len(opposite_side_peak[0]) > 0:
            if len(same_side_peak[0]) > 0:
                # first choice = last, hence hesitation+ but not com
                if (
                    same_side_peak[0][0] < opposite_side_peak[0][0]
                    and np.abs(sliced_traj[same_side_peak[0][0]]) > 5
                ):
                    # last comparison is an arbitrary threshold
                    return [False, np.nan]
            targ = sliced_traj[
                np.concatenate(
                    [opposite_side_peak[0].flatten(), np.array([-1])]
                ).astype(int)
            ]  # idxes
            # removing /5 because it has already been filtered when selecting hesitation
            targ = np.sign((targ / 2).astype(int))
            if np.any(targ < 0) and np.any(targ > 0):
                try:
                    # return it as whole len trajectory index (+1 missing because of the rolling smooth?)
                    return [True, fixsound_framespan + 1 + opposite_side_peak[0]]
                except:
                    pass
            else:
                return [False, np.nan]
        else:
            return [False, np.nan]

    # this actually suggest any trial where the rat hesitates, make func sematically consistent!
    def suggest_coms(self):
        """adds a col in .trial_sess, req trajectories first. Add hesitation, then CoM"""

        # speed factor calc (30fps ~10, 100 fps ~4 )
        # sf = -self.fixed_framerate*0.085+12.5 ### more or less the line we want
        # self.trial_sess['Hesitation'] = self.trial_sess['trajectory_vy'].apply(lambda x: chom.speed_inflex(x, fps=self.fixed_framerate)) # change to Hesitation
        self.trial_sess["Hesitation"] = False
        self.trial_sess["CoM_sugg"] = False
        self.trial_sess["CoM_peakf"] = np.nan
        # newer robust code
        if self.framestamps is not None:  # asumes speed is in px/ms units
            self.trial_sess[
                ["Hesitation", "CoM_sugg", "CoM_peakf"]
            ] = self.trial_sess.apply(
                lambda x: chom.did_he_hesitate(x), axis=1, result_type="expand"
            )
        # older crap
        elif "delay" not in self.target:
            # trying to smooth it \ add soundlen so we can filter noisy fixation + sound traj
            self.trial_sess["Hesitation"] = self.trial_sess.apply(
                lambda x: chom.speed_inflex(
                    x["trajectory_vy"], x["sound_len"], fps=self.fixed_framerate
                ),
                axis=1,
            )

        else:
            # get something new # def speed_inflex_delay(speedvec, soundlen, delay, special_trial ,fps=30):
            self.trial_sess["Hesitation"] = self.trial_sess.apply(
                lambda x: chom.speed_inflex_delay(
                    x["trajectory_vy"],
                    x["sound_len"],
                    x["delay_len"]
                    * 1000,  # we keep delay in secs now, so multipy it here because function assumes ms input
                    x["special_trial"],
                    fps=self.fixed_framerate,
                ),
                axis=1,
            )

        # Now, just in the ones that Hesitation=True

        # self.trial_sess.loc[self.trial_sess.Hesitation==True, 'CoM_sugg'] = self.trial_sess.loc[self.trial_sess.Hesitation==True, :].apply(lambda x: chom.CoM_or_not(x['trajectory_y'], x['sound_len'], x['R_response'], fps=self.fixed_framerate), axis=1)
        if self.framestamps is None:
            if "delay" not in self.target:
                self.trial_sess.loc[
                    self.trial_sess.Hesitation == True, "CoM_sugg"
                ] = self.trial_sess.loc[self.trial_sess.Hesitation == True, :].apply(
                    lambda x: chom.CoM_or_not(
                        x["trajectory_y"],
                        x["sound_len"],
                        x["R_response"],
                        fps=self.fixed_framerate,
                    ),
                    axis=1,
                )
            else:
                self.trial_sess.loc[
                    self.trial_sess.Hesitation == True, "CoM_sugg"
                ] = self.trial_sess.loc[self.trial_sess.Hesitation == True, :].apply(
                    lambda x: chom.CoM_or_not_delay(
                        x["trajectory_y"],
                        x["sound_len"],
                        x["R_response"],
                        x["delay_len"],
                        x["special_trial"],
                        fps=self.fixed_framerate,
                    ),
                    axis=1,
                )
            self.trial_sess.loc[
                self.trial_sess.Hesitation == True, "CoM_peakf"
            ] = self.trial_sess.loc[
                self.trial_sess.Hesitation == True, "CoM_sugg"
            ].apply(
                lambda x: x[1]
            )  # ??
            self.trial_sess.loc[
                self.trial_sess.Hesitation == True, "CoM_sugg"
            ] = self.trial_sess.loc[
                self.trial_sess.Hesitation == True, "CoM_sugg"
            ].apply(
                lambda x: x[0]
            )
        # kek.apply(lambda x: chom.extr_listened_frames(x['res_sound'], x['frames_listened']), axis=1)
        # fixsound_framespan = (300 + fok.trial_sess.loc[fok.trial_sess.CoM_sugg==True, 'sound_len'].iloc[trial_n])/(1000/fok.fixed_framerate) # get rid of these first frames
        # 2nd filter to get changes of mind from hesitation.

    # def buid_video(name, iterable, codec= 'X264'):
    # here it comes


# import warnings
# warnings.filterwarnings('ignore')


# info regarding outputs
# origidx: original index in the session (1 based) # sometimes is detected as an object
# coh: original stim coherence in bpods csv: hence from 0 to 1 (L-R)
# rewside: reward side; 0=L; 1=R
# hithistory (hit): whether the rat received reward (1) or punish (0, timeout) | invalid trials might be =0
# R_response= rat response was R(1) or L(0) or Invalid (-1)
# subjid: subject identifier typically a string "LE##"
# sessid: session id/csv_filename, string as follows: [SUBJECT_TASKPROTOCOL_TIMESTAMP.csv]
# resp_len: time in secs that the rat takes to respond (ie from poke out central-port to lateral-port)
# lenv: left stim envelope
# renv. right stim envelope
# res_sound: resulting stim, L-R framewise (actually a sum bcuz lenv val span from -1 to 0)
# trialonset: time it takes to start that particular trial since the begining of the session (secs) # confound broken sessions
# soundonset: time it takes to start the sound since the beginning of the trial
# sound_len: theoretical stim duration in ms | soudnt last longer than 1s
# frames_listened: same but/50
# tbc
def threaded_gather(inarg):  # com_instance, session, normcoords=True, #**kwargs):
    #                     bodypart='rabove-snout', skip_errors=False, get_trajectories=True):
    # unpack because of the single argument * concurrent-futures shit
    com_instance, session, kwargs = inarg[0], inarg[1], inarg[2]
    try:
        com_instance.load_available()
        com_instance.load(session)
        com_instance.process(normcoords=kwargs["normcoords"])
        if kwargs["analyze_trajectories"] == True:
            com_instance.get_trajectories(
                bodypart=kwargs.get("bodypart", "rabove-snout"),
                fixationbreaks=kwargs.get("fixationbreaks", False),
                noffset_frames=kwargs.get("noffset_frames", 2),
            )  # history
            com_instance.suggest_coms()
            if com_instance.framestamps is not None:
                com_instance.trial_sess.loc[:, "framestamps"] = True
            else:
                com_instance.trial_sess.loc[:, "framestamps"] = False
            com_instance.trial_sess.loc[:, "framerate"] = com_instance.fixed_framerate
        # return com_instance.trial_sess, com_instance.target
        return com_instance.trial_sess

    except Exception as e:
        if not kwargs["skip_errors"]:
            raise e
        print(f"fail {session}: {e}")
        # return -1, com_instance.target
        return -1


# plenty of strategies:
# https://yuanjiang.space/threadpoolexecutor-map-method-with-multiple-parameters
def extraction_pipe(
    targets,
    nworkers=7,
    bodypart="rabove-snout",
    fixationbreaks=True,
    normcoords=True,
    skip_errors=True,
    analyze_trajectories=True,
    sessions={},
    parentpath="/home/jordi/Documents/changes_of_mind/data/",
    tqdm_notebook=True,
    pat="p4",
    noffset_frames=0,
    GLM=False,
    glm_kws={},
    parallel="ThreadPoolExecutor",
    replace_silent=True

):
    """to avoid copying extraction scrpt all the way around, this function joins method calls
    targets = list of subjects ![even if single]
    sessions  =>  dict[subject] = [list of sessions!] ~ will be compared with available
                (ie needs matching target list)
    pat: pattern to match in session name (overrided by sessions dict) eg p4_repalt
    Poggers bar: https://github.com/tqdm/tqdm/issues/484
    Parallel: should be either 'ThreadPoolExecutor' or 'ProcessPoolExecutor'
    replace_silent: replaces silent envelope values by 0s
    other args are actual kwargs for underlying methods
    """
    glm_def={ # defaults
        "lateralized": True,
        "dual": True,
        "plot": False,
        "plot_kwargs": {},
        "filtermask": None,
        "noenv": False,
        "savdir": "",
        "fixedbias": True,
        "return_coefs": True,
        "subjcol": "subjid"
    }
    glm_def.update(glm_kws)

    assert isinstance(targets, list), 'first arg must be a list of subjects (even for single subj.)'
    kwargs = { # those will be passed to threaded_gather
        "normcoords": normcoords,
        "bodypart": bodypart,
        "skip_errors": skip_errors,
        "analyze_trajectories": analyze_trajectories,
        "fixationbreaks": fixationbreaks,
        "noffset_frames": noffset_frames,
    }

    df = pd.DataFrame([])
    for subj in targets:
        com_instance = chom(
            subj, parentpath=parentpath, analyze_trajectories=analyze_trajectories,
            replace_silent=replace_silent
        )
        com_instance.load_available()  # I think default load available uses analyze_trajectories=True
        subj_sess = [x for x in com_instance.available if pat in x]
        if sessions:
            if subj not in sessions.keys():
                print(f"skipping {subj} because it's not in sessions.keys()")
                continue
            final_sess = [
                x if x in com_instance.available else print(f"{x} not in available")
                for x in sessions[subj]
            ]
            subj_sess = final_sess

        # print(f'processing {len(subj_sess)} sessions from {subj}...')
        parallelizator = getattr(confu, parallel)
        with parallelizator(max_workers=nworkers) as executor:
            # for targ_sess, name in executor.map(threaded_gather,
            #                             tqdm.tqdm([[com_instance, x, kwargs] for x in subj_sess])):
            #     if isinstance(targ_sess, pd.DataFrame):
            #         df = df.append(targ_sess, ignore_index=True)
            jobs = [
                executor.submit(
                    threaded_gather,
                    [
                        chom(
                            subj,
                            parentpath=parentpath,
                            analyze_trajectories=analyze_trajectories,
                            replace_silent=replace_silent
                        ),
                        x,
                        kwargs,
                    ],
                )
                for x in subj_sess
            ]  # attempting to generate a new chom instance per job (less efficient but safer because of
            # shared vars (e.g. self.fair)
            if tqdm_notebook:
                progressfun = tqdm.tqdm_notebook
            else:
                progressfun = tqdm.tqdm
            for job in progressfun(
                confu.as_completed(jobs), total=len(subj_sess), desc=subj
            ):
                if skip_errors:
                    try:
                        if isinstance(job.result(), pd.DataFrame):
                            df = df.append(job.result(), ignore_index=True)
                    except Exception as e:
                        print(f"one job crashed:\n{e}")
                else:
                    if isinstance(job.result(), pd.DataFrame):
                        df = df.append(job.result(), ignore_index=True)
    # because submit nature, sort them by subject and date
    if isinstance(df, pd.DataFrame):
        if "sessid" in df.columns:
            df["date"] = pd.to_datetime(df.sessid.str[-15:])
            df = (
                df.sort_values(["subjid", "date"])
                .reset_index(drop=True)
                .drop(columns="date")
            )

        if GLM: # what about return coefs?
            if glm_kws.get('return_coefs', False):
                df, glm_out = piped_moduleweight(df, **glm_def)
                return df, glm_out
            else:
                df = piped_moduleweight(df, **glm_def)
                return df
        else:
            return df
    else:
        return -1


# def piped_moduleweight(
#     df,lateralized=True, dual=True, plot=False, plot_kwargs={}, filtermask=None,
#     noenv=False, savdir='', fixedbias=True, return_coefs=False, subjcol=None
# ):


def build_videos_single(
    df,
    dfindexes,
    outvidobject,
    pose_ext="DeepCut_resnet50_metamix2019Jul03shuffle1_1030000.h5",
    constant_text=None,
    write_custom=None,
):
    """once we have complete dataframe, we can build videos of interest sections (trajectories)
    df: trial-based pandas dataframe
    dfindex: index of the above dataframe to slice vid frames (integer)
    outvidobject: cv2 videowriter object where the frames will be appended
    constant_text: str to write in those frames; can be str (msg) or touple (str, coord)
    write_custom: custom function to be called in order to write/draw other stuff
    
    
    example in action:
    ----------------------------------------------------------------------------------------
    a = df.loc[~(df.dirty)&(df.CoM_sugg==True), :].sort_values(by='y_com').index # sortwhatever trials you want to compile
    fourcc = cv2.VideoWriter_fourcc(*'X264') # codec
    out_test = 'CoM_noenv_nodirty.avi' # filename
    outvid_all = cv2.VideoWriter(out_test, fourcc, 30.0, (640,480)) # create videowriter object

    for trial in tqdm_notebook(a): # loop
        feed_frames(df,trial, outvid_all)
        
    outvid_all.release()
    """
    if isinstance(dfindexes, pd.Index):
        itreable=dfindexes
    else:
        iterable = [dfindexes]
    
    for dfindex in dfindexes:
        subj = df.loc[dfindex, "subjid"]
        targ = df.loc[dfindex, "sessid"]
        trialno = df.loc[dfindex, "origidx"]
        invideo_dir = f"/home/jordi/Documents/changes_of_mind/data/{subj}/videos/{targ}.avi"
        intraj_dir = f"/home/jordi/Documents/changes_of_mind/data/{subj}/poses/{targ}{pose_ext}"
        print(intraj_dir)
        pose = pd.read_hdf(intraj_dir).xs(pose_ext[:-3], axis=1, drop_level=True)

        startf = df.loc[dfindex, "vidfnum"] - 5
        endf = startf + len(df.loc[dfindex, "trajectory_y"]) + 15
        cap = cv2.VideoCapture(invideo_dir)
        cap.set(cv2.CAP_PROP_POS_FRAMES, startf)
        # ycom = round(df.loc[dfindex, 'y_com'], 1)
        traj_as = list(pose.iloc[startf:endf].loc[:, ("above-snout", ("x", "y"))].values)
        traj_sn = list(pose.iloc[startf:endf].loc[:, ("snout", ("x", "y"))].values)
        sfail = df.loc[dfindex, "soundrfail"]
        for i, f in enumerate(range(startf, endf)):
            ret, frame = cap.read()
            if ret:
                cv2.putText(
                    frame,
                    targ + f"{trialno}; #f: " + str(f),  # + ' y_px(AS) = ' + str(ycom),
                    (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if sfail:
                    cv2.putText(
                        frame,
                        "no sound",
                        (30, 430),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.circle(
                    frame, tuple(traj_as[i].astype(int).tolist()), 3, (0, 255, 0), -1
                )
                cv2.circle(
                    frame, tuple(traj_sn[i].astype(int).tolist()), 3, (0, 0, 255), -1
                )
                outvidobject.write(frame)
        cap.release()
    outvidobject.release()


# def build_videos_threaded():
# """"build lots of them and concatenate them using ffmpeg""""


def build_videos_trial(
    df,
    trial_number,
    show_states=True,
    framerate=None,
    videoobject=None,
    savpath="./",
    coords=True,
):
    """build a video from a particular trial to check whats going on
    df = session dataframe (chom.sess) # substitute by session_name and trialnumber
    trial number to show,
    whether to write state transitions in the video or not!
    framerate of out video (none will pick inferred fps, not taken into account if video
                            objct is provided)
    videoobject = preexisting cv2.videowriter obj
    savpath: where to save the video in case it is not already provided
    coords overlay"""
    raise NotImplementedError("incomplete")
    session_name = df.loc[df.MSG == "SESSION-NAME", "+INFO"].values[0]
    if not savpath.endswith("/"):
        savpath += "/"
    if videoobject is None:  # create videoobject
        fourcc = cv2.VideoWriter_fourcc(*"X264")

        finalname = savpath + session_name + ".avi"


def stack_sessions_by_day(df, normalized=False, consecutive=True):
    """input prototypical df, returns pd.Series of the new col containing either full sessidx
    (normalized=False) or normalized one.
    Consecutive=False, pending to implement"""
    if not consecutive:
        raise NotImplementedError("do it!")
    # pretest and warn
    if (np.diff(df.index.values) != 1).sum():
        warnings.warn(
            "Non-consecutive index detected. Requires consecutive trials else this will malfunction"
        )
    df["trainingdayrat"] = np.nan
    df["tmpdate"] = pd.to_datetime(df.sessid.str[-15:])
    first_day_sess = (
        df.set_index("tmpdate")
        .groupby([pd.Grouper(freq="D"), "subjid"])["sessid"]
        .agg("first")
        .values
    )
    ntrials_day = (
        df.set_index("tmpdate")
        .groupby([pd.Grouper(freq="D"), "subjid"])["subjid"]
        .agg("count")
        .values
    )
    df.loc[
        (df.origidx == 1) & df.sessid.isin(first_day_sess), "trainingdayrat"
    ] = np.arange(first_day_sess.size)
    df.trainingdayrat.fillna(method="ffill", inplace=True)

    if normalized:
        out = pd.Series(
            np.concatenate([np.arange(1, x + 1) / x for x in ntrials_day]),
            index=df.index,
            name="daystacknorm",
        )
    else:
        out = pd.Series(
            np.concatenate([np.arange(1, x + 1) for x in ntrials_day]),
            index=df.index,
            name="daystack",
        )

    df.drop(columns=["tmpdate", "trainingdayrat"], inplace=True)

    return out
