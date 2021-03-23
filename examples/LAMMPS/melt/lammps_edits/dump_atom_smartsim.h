/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef DUMP_CLASS
// clang-format off
DumpStyle(atom/smartsim, DumpAtomSmartSim)
// clang-format on
#else

#ifndef LMP_DUMP_ATOM_SMARTSIM_H
#define LMP_DUMP_ATOM_SMARTSIM_H

#include "dump_atom.h"
#include "client.h"
#include "dataset.h"

namespace LAMMPS_NS
{

class DumpAtomSmartSim : public DumpAtom
{
public:
    DumpAtomSmartSim(class LAMMPS *, int, char **);
    virtual ~DumpAtomSmartSim();

protected:
    virtual void write();
    virtual void init_style();
private:
  std::string _make_dataset_key();
  template <typename T>
  void _pack_buf_into_array(T* data, int length,
			    int start_pos, int stride);
};
}

#endif
#endif
