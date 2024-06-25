/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2024, Hewlett Packard Enterprise
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "client.h"
#include <mpi.h>
#include <vector>
#include <string>

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);


     // Initialize tensor dimensions
    size_t dim1 = 3;
    size_t dim2 = 2;
    size_t dim3 = 5;
    std::vector<size_t> dims = {3, 2, 5};

    // Initialize a tensor to random values.  Note that a dynamically
    // allocated tensor via malloc is also useable with the client
    // API.  The std::vector is used here for brevity.
    size_t n_values = dim1 * dim2 * dim3;
    std::vector<double> input_tensor(n_values, 0);
    for(size_t i=0; i<n_values; i++)
        input_tensor[i] = 2.0*rand()/RAND_MAX - 1.0;

    // Get our rank
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::string logger_name("Client ");
    logger_name += std::to_string(rank);

    // Initialize a SmartRedis client
    SmartRedis::Client client(logger_name);

    // Put the tensor in the database
    std::string key = "3d_tensor_" + std::to_string(rank);
    client.put_tensor(key, input_tensor.data(), dims,
                      SRTensorTypeDouble, SRMemLayoutContiguous);

    // Retrieve the tensor from the database using the unpack feature.
    std::vector<double> unpack_tensor(n_values, 0);
    client.unpack_tensor(key, unpack_tensor.data(), {n_values},
                        SRTensorTypeDouble, SRMemLayoutContiguous);

    // Print the values retrieved with the unpack feature
    std::cout<<"Comparison of the sent and "\
                "retrieved (via unpack) values: "<<std::endl;
    for(size_t i=0; i<n_values; i++)
        std::cout<<"Sent: "<<input_tensor[i]<<" "
                 <<"Received: "<<unpack_tensor[i]<<std::endl;


    // Retrieve the tensor from the database using the get feature.
    SRTensorType get_type;
    std::vector<size_t> get_dims;
    void* get_tensor;
    client.get_tensor(key, get_tensor, get_dims, get_type, SRMemLayoutNested);

    // Print the values retrieved with the unpack feature
    std::cout<<"Comparison of the sent and "\
                "retrieved (via get) values: "<<std::endl;
    for(size_t i=0, c=0; i<dims[0]; i++)
        for(size_t j=0; j<dims[1]; j++)
            for(size_t k=0; k<dims[2]; k++, c++) {
                std::cout<<"Sent: "<<input_tensor[c]<<" "
                         <<"Received: "
                         <<((double***)get_tensor)[i][j][k]<<std::endl;
    }

    MPI_Finalize();

    return 0;
}
