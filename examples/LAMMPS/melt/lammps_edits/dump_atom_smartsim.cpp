/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Matthew Ellis (HPE)
------------------------------------------------------------------------- */

#include "dump_atom_smartsim.h"
#include "atom.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "memory.h"
#include "universe.h"
#include "update.h"
#include <cstring>

using namespace LAMMPS_NS;

DumpAtomSmartSim::DumpAtomSmartSim(LAMMPS *lmp, int narg, char **arg)
    : DumpAtom(lmp, narg, arg)
{
}

/* ---------------------------------------------------------------------- */

DumpAtomSmartSim::~DumpAtomSmartSim()
{
}

/* ---------------------------------------------------------------------- */

void DumpAtomSmartSim::write()
{
  SmartSimClient client;

  int rank;
  std::string key;
  int* array_dims = new int[1];
  MPI_Comm_rank(world, &rank);

  double* box_dims;
  int n_dims;
    if (domain->triclinic == 0) {
      n_dims = 6;
      box_dims = new double[n_dims];
      box_dims[0] = domain->boxlo[0];
      box_dims[1] = domain->boxhi[0];
      box_dims[2] = domain->boxlo[1];
      box_dims[3] = domain->boxhi[1];
      box_dims[4] = domain->boxlo[2];
      box_dims[5] = domain->boxhi[2];
    } else {
      n_dims = 9;
      box_dims = new double[n_dims];
      box_dims[0] = domain->boxlo_bound[0];
      box_dims[1] = domain->boxhi_bound[0];
      box_dims[2] = domain->boxlo_bound[1];
      box_dims[3] = domain->boxhi_bound[1];
      box_dims[4] = domain->boxlo_bound[2];
      box_dims[5] = domain->boxhi_bound[2];
      box_dims[6] = domain->xy;
      box_dims[7] = domain->xz;
      box_dims[8] = domain->yz;
    }
    key = this->_make_key("domain", rank);
    array_dims[0] = n_dims;
    client.put_array_double(key.c_str(), box_dims, array_dims, 1);
    delete[] box_dims;

    key = this->_make_key("triclinic", rank);
    client.put_scalar_int64(key.c_str(), domain->triclinic);
    key = this->_make_key("scale_flag", rank);
    client.put_scalar_int64(key.c_str(), scale_flag);
    
    int n_local = atom->nlocal;
    int n_cols = (image_flag == 1) ? 8 : 5;
    
    nme = count();
    if (nme > maxbuf) {
        maxbuf = nme;
        memory->destroy(buf);
        memory->create(buf, (maxbuf * n_cols), "dump:buf");
    }
    if (sort_flag && sortcol == 0 && nme > maxids) {
        maxids = nme;
        memory->destroy(ids);
        memory->create(ids, maxids, "dump:ids");
    }

    if (sort_flag && sortcol == 0)
        pack(ids);
    else
        pack(NULL);
    if (sort_flag)
        sort();

    array_dims[0] = n_local;
    int* data_int = new int[n_local];
    double* data_dbl = new double[n_local];
    int buf_len = n_cols*n_local;
    //Atom ID
    this->_pack_buf_into_array<int>(data_int, buf_len, 0, n_cols);
    key = this->_make_key("atom_id", rank);
    client.put_array_int64(key.c_str(), data_int, array_dims, 1);
    //Atom Type
    this->_pack_buf_into_array<int>(data_int, buf_len, 1, n_cols);
    key = this->_make_key("atom_type", rank);
    client.put_array_int64(key.c_str(), data_int, array_dims, 1);
    //Atom x position
    this->_pack_buf_into_array<double>(data_dbl, buf_len, 2, n_cols);
    key = this->_make_key("atom_x", rank);
    client.put_array_double(key.c_str(), data_dbl, array_dims, 1);
    //Atom y position
    this->_pack_buf_into_array<double>(data_dbl, buf_len, 3, n_cols);
    key = this->_make_key("atom_y", rank);
    client.put_array_double(key.c_str(), data_dbl, array_dims, 1);
    //Atom z position
    this->_pack_buf_into_array<double>(data_dbl, buf_len, 4, n_cols);
    key = this->_make_key("atom_z", rank);
    client.put_array_double(key.c_str(), data_dbl, array_dims, 1);

    if (image_flag == 1) {
      //Atom ix image
      this->_pack_buf_into_array<int>(data_int, buf_len, 5, n_cols);
      key = this->_make_key("atom_ix", rank);
      client.put_array_int64(key.c_str(), data_int, array_dims, 1);
      //Atom iy image
      this->_pack_buf_into_array<int>(data_int, buf_len, 6, n_cols);
      key = this->_make_key("atom_iy", rank);
      client.put_array_int64(key.c_str(), data_int, array_dims, 1);
      //Atom iz image
      this->_pack_buf_into_array<int>(data_int, buf_len, 7, n_cols);
      key = this->_make_key("atom_iz", rank);
      client.put_array_int64(key.c_str(), data_int, array_dims, 1);
    }

    delete[] data_int;
    delete[] data_dbl;
    delete[] array_dims;
}

/* ---------------------------------------------------------------------- */

void DumpAtomSmartSim::init_style()
{
    // setup function ptrs to use defaults from atom dump
    if (scale_flag == 1 && image_flag == 0 && domain->triclinic == 0)
        pack_choice = &DumpAtomSmartSim::pack_scale_noimage;
    else if (scale_flag == 1 && image_flag == 1 && domain->triclinic == 0)
        pack_choice = &DumpAtomSmartSim::pack_scale_image;
    else if (scale_flag == 1 && image_flag == 0 && domain->triclinic == 1)
        pack_choice = &DumpAtomSmartSim::pack_scale_noimage_triclinic;
    else if (scale_flag == 1 && image_flag == 1 && domain->triclinic == 1)
        pack_choice = &DumpAtomSmartSim::pack_scale_image_triclinic;
    else if (scale_flag == 0 && image_flag == 0)
        pack_choice = &DumpAtomSmartSim::pack_noscale_noimage;
    else if (scale_flag == 0 && image_flag == 1)
        pack_choice = &DumpAtomSmartSim::pack_noscale_image;
}

std::string DumpAtomSmartSim::_make_key(std::string var_name, int rank)
{
  // create database key using the var_name
  std::string prefix(filename);
  std::string key = prefix + "_rank_" + std::to_string(rank) +
    "_tstep_" + std::to_string(update->ntimestep) + "_" +
    var_name;
  return key;
}

template <typename T>
void DumpAtomSmartSim::_pack_buf_into_array(T* data, int length,
					    int start_pos, int stride)
{
  // pack the dump buffer into an array to send to the database
  int c = 0;
  for(int i = start_pos; i < length; i+=stride) {
    data[c++] = buf[i];
  }
}

