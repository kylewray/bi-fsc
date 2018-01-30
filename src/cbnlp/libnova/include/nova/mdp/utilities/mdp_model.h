/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2016 Kyle Hollins Wray, University of Massachusetts
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of
 *  this software and associated documentation files (the "Software"), to deal in
 *  the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *  the Software, and to permit persons to whom the Software is furnished to do so,
 *  subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *  FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 *  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 *  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 *  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


#ifndef NOVA_MDP_MODEL_H
#define NOVA_MDP_MODEL_H


#include <nova/mdp/mdp.h>

namespace nova {

/**
 *  Allocate memory for *only* the MDP's internal arrays, given the relevant parameters.
 *  @param  mdp         The MDP object. Only arrays within will be freed.
 *  @param  n           The number of states.
 *  @param  ns          The maximum number of successor states.
 *  @param  m           The number of actions.
 *  @param  gamma       The discount factor between 0.0 and 1.0.
 *  @param  horizon     The horizon of the MDP.
 *  @param  epsilon     The convergence criterion for algorithms like LAO*.
 *  @param  s0          The initial state.
 *  @param  ng          The number of goals; optional for SSP MDPs.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int mdp_initialize(MDP *mdp, unsigned int n, unsigned int ns, unsigned int m, float gamma,
    unsigned int horizon, float epsilon, unsigned int s0, unsigned int ng);

/**
 *  Free the memory for *only* the MDP's internal arrays.
 *  @param  mdp     The MDP object. Only arrays within will be freed.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int mdp_uninitialize(MDP *mdp);

};


#endif // NOVA_MDP_MODEL_H

