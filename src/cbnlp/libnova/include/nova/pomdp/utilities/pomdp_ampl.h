/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2017 Kyle Hollins Wray, University of Massachusetts
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


#ifndef NOVA_POMDP_AMPL_H
#define NOVA_POMDP_AMPL_H


#include <nova/pomdp/pomdp.h>

#include <string>

namespace nova {

/**
 *  Export the provided POMDP as an AMPL data file to the path and filename given.
 *  @param  pomdp       The POMDP to export as an AMPL data file.
 *  @param  k           The number of controller nodes.
 *  @param  r           The number of belief points to add from the POMDP.
 *  @param  path        The desired path of the AMPL data file.
 *  @param  filename    The desired name of the AMPL data file.
 *  @return Return NOVA_SUCCESS on success; otherwise return an error.
 */
extern "C" int pomdp_ampl_save_data_file(const POMDP *pomdp, unsigned int k,
        unsigned int r, const char *path, const char *filename);

/**
 *  Parse the standard format assumed returned by any solver command.
 *
 *  Note: Either policy or <psi, eta> can be specified. Whatever the solver output
 *  provides, in combination with whatever variable exists, will determine
 *  what is assigned.
 *
 *  @param  pomdp   The POMDP used for the solver.
 *  @param  k       The number of controller nodes.
 *  @param  policy  The policy probabilities (k-m-z-k array). This may be modified.
 *  @param  psi     The action probabilities (k-m array). This may be modified.
 *  @param  eta     The successor probabilities (k-m-z-k array). This may be modified.
 *  @param  V       The values of each state (k-n array). This will be modified.
 *  @return Return NOVA_SUCCESS on success; otherwise return an error.
 */
extern "C" int pomdp_ampl_parse_solver_output(const POMDP *pomdp, unsigned int k,
        float *policy, float *psi, float *eta, float *V, std::string &solverOutput);

};


#endif // NOVA_POMDP_AMPL_H



