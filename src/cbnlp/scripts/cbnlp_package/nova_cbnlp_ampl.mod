model;

param NUM_STATES;
param NUM_ACTIONS;
param NUM_OBSERVATIONS;
param NUM_NODES;
param NUM_BELIEFS;

param gamma default 0.95, >= 0.0, <= 1.0;
param B {i in 1..NUM_BELIEFS, s in 1..NUM_STATES} default 0.0, >= 0.0, <= 1.0;
param lambda default 0.0, >= 0.0, <= 1.0;

param T {s in 1..NUM_STATES, a in 1..NUM_ACTIONS, sp in 1..NUM_STATES} default 0.0, >= 0.0, <= 1.0;
param O {a in 1..NUM_ACTIONS, s in 1..NUM_STATES, o in 1..NUM_OBSERVATIONS} default 0.0, >= 0.0, <= 1.0;
param R {s in 1..NUM_STATES, a in 1..NUM_ACTIONS} default 0.0;

var V {1..(NUM_NODES + NUM_BELIEFS), 1..NUM_STATES};
var psi {1..(NUM_NODES + NUM_BELIEFS), 1..NUM_ACTIONS} >= 0.0;
var eta {1..(NUM_NODES + NUM_BELIEFS), 1..NUM_ACTIONS, 1..NUM_OBSERVATIONS, 1..NUM_NODES} >= 0.0;

maximize Value:
   sum {s in 1..NUM_STATES} B[1, s] * V[1, s];

subject to Bellman_Constraint_V_Nodes {x in 1..NUM_NODES, s in 1..NUM_STATES}:
  V[x, s] = sum {a in 1..NUM_ACTIONS} (psi[x, a] * (R[s, a] + (gamma * (1.0 - lambda)) * sum {sp in 1..NUM_STATES} (T[s, a, sp] * sum {o in 1..NUM_OBSERVATIONS} (O[a, sp, o] * sum {xp in 1..NUM_NODES} (eta[x, a, o, xp] * V[xp, sp]))) + (gamma * lambda / NUM_BELIEFS) * sum {xp in (NUM_NODES + 1)..(NUM_NODES + NUM_BELIEFS), sp in 1..NUM_STATES} (B[xp - NUM_NODES, sp] * V[xp, sp])));

subject to Bellman_Constraint_V_Beliefs {x in (NUM_NODES + 1)..(NUM_NODES + NUM_BELIEFS), s in 1..NUM_STATES}:
  V[x, s] = sum {a in 1..NUM_ACTIONS} (psi[x, a] * (R[s, a] + gamma * sum {sp in 1..NUM_STATES} (T[s, a, sp] * sum {o in 1..NUM_OBSERVATIONS} (O[a, sp, o] * sum {xp in 1..NUM_NODES} (eta[x, a, o, xp] * V[xp, sp])))));

subject to Probability_Constraint_Normalization_Psi {x in 1..(NUM_NODES + NUM_BELIEFS)}:
  sum {a in 1..NUM_ACTIONS} psi[x, a] = 1.0;

subject to Probability_Constraint_Normalization_Eta {x in 1..(NUM_NODES + NUM_BELIEFS), a in 1..NUM_ACTIONS, o in 1..NUM_OBSERVATIONS}:
  sum {xp in 1..NUM_NODES} eta[x, a, o, xp] = 1.0;

