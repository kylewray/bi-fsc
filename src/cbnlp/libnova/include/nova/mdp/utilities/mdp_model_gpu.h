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


#ifndef NOVA_MDP_MODEL_GPU_H
#define NOVA_MDP_MODEL_GPU_H


#include <nova/mdp/mdp.h>

namespace nova {

/**
 *  Initialize CUDA by transferring all of the constant MDP model information to the device.
 *  @param  mdp   The MDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int mdp_initialize_gpu(MDP *mdp);

/**
 *  Uninitialize CUDA by freeing all of the constant MDP model information to the device.
 *  @param  mdp   The MDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int mdp_uninitialize_gpu(MDP *mdp);

/**
 *  Initialize CUDA successors object.
 *  @param  mdp   The MDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int mdp_initialize_successors_gpu(MDP *mdp);

/**
 *  Uninitialize CUDA successors object.
 *  @param  mdp   The MDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int mdp_uninitialize_successors_gpu(MDP *mdp);

/**
 *  Initialize CUDA state transitions object.
 *  @param  mdp   The MDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int mdp_initialize_state_transitions_gpu(MDP *mdp);

/**
 *  Uninitialize CUDA state transitions object.
 *  @param  mdp   The MDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int mdp_uninitialize_state_transitions_gpu(MDP *mdp);

/**
 *  Initialize CUDA rewards object.
 *  @param  mdp   The MDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int mdp_initialize_rewards_gpu(MDP *mdp);

/**
 *  Uninitialize CUDA rewards object.
 *  @param  mdp   The MDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int mdp_uninitialize_rewards_gpu(MDP *mdp);

/**
 *  Initialize CUDA goals object.
 *  @param  mdp   The MDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int mdp_initialize_goals_gpu(MDP *mdp);

/**
 *  Uninitialize CUDA goals object.
 *  @param  mdp   The MDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int mdp_uninitialize_goals_gpu(MDP *mdp);

};


#endif // NOVA_MDP_MODEL_GPU_H


