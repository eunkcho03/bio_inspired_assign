# TASK OVERVIEW 
The aim is to explore a 2D aisle-based warehouse picking problem. The agent must collect all required items and return to the depot. To mimic  realistic operations, some special tiles are included which introduces constraints and stochasticity, while keeping the MDP stationary. 

- **State**: Grid planes (agent, depot, picks, special tiles) 
- **Actions**: {Up, Right, Down, Left} 
- **Termination**: 
    - All items collected and agent at depot 
    - Trap entered 
    - Step limit 
- **Base Reward**: Per-step cost with completion bonus at the end  (optional) 

## Special Tiles (Warehouse Interpretation)
- **Blocked Zone (Trap, T)**:
    - *Meaning*: Temporarily closed space. 
    - *Transition*: Entering terminates the episode
    - *Reward*: Large penalty 
- **Congestion Zone (Slide, W)**:
    - *Meaning*: Crowded narrow aisle where you might be nudged sideways.  
    - *Transition*: If you attempt action *a*, it succeeds with probability of *1-rho*; otherwise you slip left or right with probability of *rho/2* each. 
    - *Reward*: Nominal step cost
- **Uncertain Pick Station (Decoy, D)**
    - *Meaning*: Station with uncertain outcome
    - *Transition*: Non-terminal
    - *Reward*: With probability P would give reward otherwise a penalty. 
- **Aisle Shortcut (Portal, P)**
    - *Meaning*: Elevator/conveyor/cross-aisle tunnel connecting two fixed points. 
    - *Transition*: Landing on P teleports to its paird P. 
    - *Reward*: Add transfer cost of using the portal. 
    

## Initial Parameters
- c_{step} = -0.04
- r_{trap} = -1.0
- c_{portal} = -0.10
- Slide probability rho = 0.3
- Decoy: p = 0.1, r_good = 0.8, r_bad = -0.5