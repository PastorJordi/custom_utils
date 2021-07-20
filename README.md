# custom_utils
few python util functions I use regularly (or used to :suspect:) when working in brain circuits and behavior lab


## I guess the most useful thing here is the extraction pipe. It can be integrated with video coordinates but won't cover it here

I wrote chom class using certain folder structure. Hence if you want to reuse it right away, you need that structure as well. Data is organized like this:
```
parentpath/LEXX/
        |
        |---- electro/
        |        LEXX_date_session_folder/
        |---- poses/
        |        LEXX_task_date_DLCnetwork.h5
        |        LEXX_task_date_DLCnetwork.pickle
        |---- sessions/
        |        LEXX_task_date.csv
        |---- videos/
        |        LEXX_task_date.avi
        |        LEXX_task_date.npy
            
```

### How to use it behavior-only example
```
from utilsJ.Behavior import ComPipe # you can install repo with pip install -e .
# csv sessions must be located under parent/LEXX/sessions/
p = ComPipe.chom( # instantiate class
    'LE85',  # subject (must be the same as the folder under parentpath)
    parentpath='/home/jordi/Documents/changes_of_mind/data/',
    analyze_trajectories=False # requires .avi and .h5 file to work
) # this 
p.load_available() # just in case, refresh
print(p.available[0]) # example target session / filename string is the actual arg 
p.load(p.available[0])
p.process()
p.trial_sess.head() # preprocessed dataframe stored in attr. trial_sess of chom object (p in this example)
# raw bpod session/csv is in p.sess
```
:godmode: To preprocess a bunch of sessions (+ optional GLM), use function `extraction_pipe()` in `Models.ComPipe.py`. Briefly, it uses chom object to get trial-based sessions. Afterwards, it can fit the classical Hermoso-Hyafil GLM. No expectation-maximization lapses :finnadie:


### :rage1: preprocessed dataframe (trial-based) columns
**origidx**: 'original index' in that session/csv (1-based)  
**coh**: 'coherence' (0=left, 1=right)  
**rewside**: reward side (0=left, 1=right)  
**hithistory**: whether hit(1), or miss(0). Current trial. Keeping the name due to historical reasons  
**R_response**: rat's response (0=left, 1=right). NANs or -1 == invalid trials  
**subjid**: subject's name  
**sessid**: name of the session (like filename -'.csv')
**resp_len**: response length, in seconds (from central port out to lateral port in)  
**lenv**: left envelope. (if stimulus lacks envelope corresponding amplitude, negative by convention)
**renv**: right envelope
**res_sound**:  resulting sound (right + left)  
**trialonset**: onset of the trial relative to session onset (in seconds)  
**soundonset**: onset of the stimulus (or it's state when it's lacking) relative to trial onset (secs.)  
**sound_len**: sound length (length of its state in BPOD) in ms  
**frames_listened**: ammount of sound frames listened.  
**prob_repeat**: prob to repeat (next trial, so usually it is block ID). Not sure it is working with L/R blocks  
**wibl_idx**: trial index within block (1based).  
**bl_idx**: block index within session (1based).  
**aftererror**: bool, whether this trial occurred after an error.  
**fb**: list of fixation-breaks right before that trial (row). Usually empty  
**soundrfail**: bool, whether soundRserver outputted playing string or not (True==no sound trial).  
**sound_len**: stimulus length (ms) according to sound server.  
**albert_len**: if available, length of the sound according to sensor boards designed by Mr. Font.  
**streak**: (deprecated, might not work as expected). Streak of correct responses.  
**rep_response**: (bool) whether the choice was a repetition
**special_trial**: -1) early; 0) regular; 1) delayed, 2) silent; 3) fair?  
**delay_len**: length of the (sound) delay (secs)  
**fix_onset_dt**: fixation onset in (PC) datetime     

 
### :rage2: trajectory-related
**trajectory_y**: trajectory y-coordinates  
**trajectory_vy**: same but speed  
**trajectory_x**: trajectory x-coordinates  
**vidfnum**: number of frame (ordinal in the video) where trajectory starts  
**trajectory_stamps**: datestamp of each frame in the trajectory  
**fix_coords**: coordinates during fixation  
**fix_conf**: ?  
**bodyangle**: body angle during fixation (neck to butt). 0 center (negative left, positive right)  
**headangle**: head angle during fixation (snout to neck)  
**dirty**: bool, whether the rat poked in central port after fixating  
**Hesitation**: Bool, whether animal showed y-velocity towards both ports  
**CoM_sugg**: bool, suggested Change-Of-Mind trial  
**CoM_peakf**: com peak frame (index in trajectory), if there was a CoM  
**framestams**: bool whether there were timestamps saved? review  
**framerate**: average video framerate  
**date**: self explanatory  
**task**: task type  
**traj_d1**: (y) trajectory 1st derivative  
**traj_d2**: 2nd  
**traj_d3**: 3rd  
**time_to_thr**: time to reach a certain px thr (30px) towards chosen port since movement onset (ms)  

### :rage3: GLM (dW_ stands for dual fit, after correct and after error)
**dW_stim**: listened stimulus \* corresponding weight. L-R space  
**dW_short_s**: after-effect or short desensitiz. L-R space  
**dW_lat**: sum of lateral module (L+, L- ~up to -10 trials \* W) L-R space  
**dW_trans**: sum of transition module (T++, T+-, T-+, T-- ~up to -10 trials \* W) L-R space  
**dW_fixedbias**: fixed bias / intercept  

### :rage4: PSIAM simulations byproduct and other stuff
**avtrapz**: average intensity stimulus, in L-R space, relative to stim. str. 1 (same length). Calculated using trapezoidal rule  
**zidx**: std. trial index  (z-score)  
**coh2**: stim coherence in -1~1 space (LR)  
**sstr**: stim strength  
**priorZt**:  priors (actually just lat and trans) in a given trial, scaled down by some factor  
**prechoice**: pre-planned choice (might include fixedbias and aftereffects)  
**choice_x_coh**: coherence aligned to choice  
**allpriors**: nan sum (axis0) of trans + lat modules  
**choice_x_allpriors**: allpriors aligned to choice  

from time to time I try reducing entropy using block code formatter
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)