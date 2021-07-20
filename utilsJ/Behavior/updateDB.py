# used to update stored data
# works ok with nb-environment
# TODO: update so it reuses (unless stated old analyzed data!)


from utilsJ.regularimports import *
from utilsJ.Behavior.ComPipe import chom, extraction_pipe
from utilsJ.Behavior import glm2afc

# settings
datapath = "/home/jordi/DATA/Documents/changes_of_mind/data/"
savpath = (
    "/home/jordi/Documents/changes_of_mind/firstanal_com_new_setup/all_data_so_far/"
)
SubjectsToUpdate = ["LE82", "LE84", "LE85", "LE86", "LE87"]
UpdateWholeDataset = True
ReuseData = (
    False  # whether to reuse individual subjects pkl files or reanalyze everything
)
mask = (
    "sound_len <= 400 and soundrfail == False and resp_len <=1 and R_response>= 0"
    + " and hithistory >= 0 and special_trial == 0"
)  # for glm analysis, all trials remain in df
pipe_kwargs = dict(
    parentpath=datapath, fixationbreaks=False, skip_errors=True, noffset_frames=0
)


#####
# update particular subjects data + GLM
print("raw extraction began...")
newcols = [f"dW_{x}" for x in ["stim", "short_s", "lat", "trans", "fixedbias"]]
if ReuseData:
    raise NotImplementedError("not yet bro")
    olddf = pd.read_pickle()
else:
    for subj in tqdm.tqdm(SubjectsToUpdate):
        try:
            df = extraction_pipe([subj], **pipe_kwargs)
            df = df.loc[
                df.sessid.str.contains("leftright|nofixnopain|onlylight") == False
            ]
            for col in newcols:
                df.loc[:, col] = np.nan

            if len(df.loc[(df.subjid == subj) & df.sessid.str.contains("feedback")]):
                for bo in [False, True]:
                    cur_mask = (df.subjid == subj) & (
                        df.sessid.str.contains("feedback") == bo
                    )
                    if bo:
                        lateral_flag = True
                        noenv_flag = False
                    else:
                        lateral_flag = False
                        noenv_flag = True
                    df.loc[cur_mask] = glm2afc.piped_moduleweight(
                        df.loc[cur_mask],
                        filtermask=mask,
                        lateralized=lateral_flag,
                        noenv=noenv_flag,
                    )
            elif subj not in [f"LE{x}" for x in range(82, 88)]:
                df.loc[df.subjid == subj] = glm2afc.piped_moduleweight(
                    df.loc[(df.subjid == subj)],
                    filtermask=mask,
                    noenv=False,
                    lateralized=True,
                )
            else:
                df.loc[df.subjid == subj] = glm2afc.piped_moduleweight(
                    df.loc[(df.subjid == subj)],
                    filtermask=mask,
                    noenv=True,
                    lateralized=False,
                )

            df.to_pickle(f"{savpath}{subj}.pkl")
            del df
        except Exception as e:
            print(f"big crash in {subj}\n{e}")


#####
# update whole dataset (should stop using this at some point)
print("Extraction finnished. Building new whole-Dataset...")
try:
    del df  # get some extra mem
except:
    pass

df = pd.read_pickle(savpath + "all_subjects.pkl")  # load
df = df.loc[~df.subjid.isin(SubjectsToUpdate)]  # remove data to be updated


pbar = tqdm.tqdm(SubjectsToUpdate)
for subj in pbar:
    pbar.set_description(subj)
    try:
        df = pd.concat(
            [df, pd.read_pickle(savpath + subj + ".pkl")], ignore_index=True, sort=True
        )
    except Exception as e:
        print(e)

df.reset_index(inplace=True, drop=True)
df.origidx = df.origidx.astype(int)
df.to_pickle(savpath + "all_subjects.pkl")
