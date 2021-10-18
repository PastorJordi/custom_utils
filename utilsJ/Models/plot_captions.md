Generally `tab:blue` color is used for data and `tab:orange` for simulations.  
Some acronyms:  
> RT = reaction time / stimulus length  
> MT = motor time (time to execute a response)  
> CoM = change of mind  
> p(CoM) = probability of CoM to happen
> sstr = stimulus strength (nominal coherence of the stimuli). The higher the clearer the stimulus is  
> EA: evidence accumulator
> AI: action initiator
> reversals: proactive trials (triggered by AI) where the evidence variable ends at the opposite side where it started
> SEM: standard error of the mean  
> CI: confidence interval


(being row,col coordinates)  
- [0,0]: RT distribution1
- [0,1]: MT  vs RT in 10ms bins. Each trace (gray) is a unique stimulus strength. solid traces => data, dotted traces => simulations. Mean and SEM are shown  
- [0,2]: DATA tachometric curves: Accuracy vs listened stimuli (RT). Each bin is 3ms. Each color is a stimulus strength. Shows mean and 95% confidence interval by "beta/clopper pearson" method.  
- [0,3]: Idem but simulations  
- [1,0]: Motor time distribution (time taken from central port-out to chosen port)  
- [1,1]: **p(CoM) vs RT** Shows mean and 95 C.I. by clopper pearson of observed CoMs. This is calculated by observed CoM in bin_i / number of observations in bin_i. Each bin is 10ms RT. Traces are distinct sstr.
    - Blue: p(CoM) in DATA. 
    - Orange: p(CoM) in Simulations.  
    - Purple: p(CoM) in simulated proactive trials.  

- [1,2]: probability to reverse: given that a trial have some offset Z_e (prior), this figure shows the probability to end in the other side of the EA. **Only proactive trials are considered**. Again, each cell is calculated as #events/#observations and this sets the color of each cell (probability is shown in the colorbar). The numbers in each cell of the grid shows the total trials that belong to it. In the x-axis we have prior (from left to right) and in the y-axis the average of the listened stimulus (this was sampled from the data, though!)  
- [1,3]: p(CoM) only in reversals. The main point of this grid is to show the "conversion rate" of reversals -> CoM. This is because some reversals are not detected as CoMs.  
- [2,0]: proportion of proactive trials (those triggered by AI) vs RT (10s bins).  
- [2,1]: p(CoM) vs trial index (Means and C.I.). 10 trial bins. It is expected to decrease, since trial index linearly increase "expected MT" (the time taken to execute the preplanned choice) hence being easier to detect CoMs in early trials (they are farther when they revert).  
- [2,2]: **data p(CoM) matrix**, same structure than [1,2]. Colorscales are different though  
- [2,3]: **simulation p(CoM) matrix**.  
- [3,0]: empty by now.  
- [3,1]: contour plot of CoM in simulated data (detected CoMs). x-axis: motor time in ms, y-axis: prior.  
- [3,2]: median trajectories in simulated CoM trials where the bias was "big-to.moderate" or "moderate-to-zero"  
- [3,3]: cummulative histograms of "effective t_update" in proactive trials per RT bin (10ms each). If only in the last bump  = only trials where EA hits when collapsing. Else some fraction hits at horizontal bounds (after AI sets the RT!).  
- [4,0]: **Time to diverge, summary**: ms it takes trajectories  to diverge sstr 1 (purple) or -1 (green) from sstr 0. Relative to movement onset. Dots = data, crosses = simulations. Each RT bin (x-axis) is 10ms.  
- [4,1]: Time to diverge data: it actually shows median trajectories (y coord vs motor time). Bright colors= rt from 100 to 125ms, darker ones RT {0~25}ms   
- [4,2]: Time to diverge simul: same but synthetic trajectories

