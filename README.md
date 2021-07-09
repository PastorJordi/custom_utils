# custom_utils
few python util functions I use regularly (or used to) when working in brain circuits and behavior lab


## I guess the most useful thing here is the extraction pipe. It can be integrated with video coordinates but won't cover it here

### How to use it behavior-only example
```
from utilsJ.Behavior import ComPipe # you can install repo with pip install -e .
# csv sessions must be located under parent/LEXX/sessions/
p = ComPipe.chom( # instantiate class
    'LE85',  # subject (must be the same as the folder under parentpath)
    parentpath='/home/jordi/Documents/changes_of_mind/data/',
    analyze_trajectories=False
) # this 
p.load_available() # just in case, refresh
print(p.available[0]) # example target session / filename string is the actual arg 
p.load(p.available[0])
p.process()
p.trial_sess.head() # preprocessed dataframe stored in attr. trial_sess of chom object (p in this example)
# raw bpod session/csv is in p.sess
```

### preprocessed dataframe (trial-based) columns
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

... there are more with further preprocessing (tbd)  
  
(trajectories)

(glm)

