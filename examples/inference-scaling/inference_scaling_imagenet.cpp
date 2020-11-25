#include "client.h"
#include "client_test_utils.h"
#include <mpi.h>

void load_mnist_image_to_array(float*& img)
{
  std::string image_file = "./cat.raw";
  std::ifstream fin(image_file, std::ios::binary);
  std::ostringstream ostream;
  ostream << fin.rdbuf();
  fin.close();
  const std::string tmp = ostream.str();
  const char *image_buf = tmp.data();
  int image_buf_length = tmp.length();
  std::memcpy(img, image_buf,
              image_buf_length*sizeof(char));
}

void run_mnist(const std::string& model_name,
               const std::string& script_name,
               std::ofstream& timing_file)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(!rank)
    std::cout<<"Connecting clients"<<std::endl<<std::flush;

  double constructor_start = MPI_Wtime();
  SmartSimClient client(true);
  double constructor_end = MPI_Wtime();
  double delta_t = constructor_end - constructor_start;
  timing_file << rank << "," << "client()" << ","
              << delta_t << std::endl << std::flush;

  MPI_Barrier(MPI_COMM_WORLD);

  //Allocate a continugous memory to make bcast easier
  float* p = (float*)malloc(224*224*3*sizeof(float));
  if(rank == 0)
    load_mnist_image_to_array(p);
  MPI_Bcast(&(p[0]), 224*224*3, MPI_FLOAT, 0, MPI_COMM_WORLD);
  float*** array = allocate_3D_array<float>(224, 224, 3);
  int c = 0;
  for(int i=0; i<224; i++)
    for(int j=0; j<224; j++)
      for(int k=0; k<3; k++) {
        array[i][j][k] = p[c];
        c++;
      }

  //float**** array = allocate_4D_array<float>(1,1,28,28);
  float** result = allocate_2D_array<float>(1, 1000);

  MPI_Barrier(MPI_COMM_WORLD);
  if(!rank)
    std::cout<<"All ranks have Resnet image"<<std::endl;

  double loop_start = MPI_Wtime();
  for (int i=0; i<10; i++) {

    std::string in_key = "resnet_input_rank_" + std::to_string(rank) + "_" + std::to_string(i);
    std::string script_out_key = "resnet_processed_input_rank_" + std::to_string(rank) + "_" + std::to_string(i);
    std::string out_key = "resnet_output_rank_" + std::to_string(rank) + "_" + std::to_string(i);

    double put_tensor_start = MPI_Wtime();
    client.put_tensor(in_key, "FLOAT", array, {224, 224, 3});
    double put_tensor_end = MPI_Wtime();
    delta_t = put_tensor_end - put_tensor_start;
    timing_file << rank << "," << "put_tensor" << ","
                << delta_t << std::endl << std::flush;

    double run_script_start = MPI_Wtime();
    client.run_script(script_name, "pre_process_3ch", {in_key}, {script_out_key});
    double run_script_end = MPI_Wtime();
    delta_t = run_script_end - run_script_start;
    timing_file << rank << "," << "run_script" << ","
                << delta_t << std::endl << std::flush;

    double run_model_start = MPI_Wtime();
    client.run_model(model_name, {script_out_key}, {out_key});
    double run_model_end = MPI_Wtime();
    delta_t = run_model_end - run_model_start;
    timing_file << rank << "," << "run_model" << ","
                << delta_t << std::endl << std::flush;

    double get_tensor_start = MPI_Wtime();
    client.get_tensor(out_key, "FLOAT", result, {1,1000});
    double get_tensor_end = MPI_Wtime();
    delta_t = get_tensor_end - get_tensor_start;
    timing_file << rank << "," << "get_tensor" << ","
                << delta_t << std::endl << std::flush;
  }
  double loop_end = MPI_Wtime();
  delta_t = loop_end - loop_start;
  timing_file << rank << "," << "loop_time" << ","
                << delta_t << std::endl << std::flush;

  free_3D_array(array, 3, 224);
  free_2D_array(result, 1);
  return;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  double main_start = MPI_Wtime();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //Open Timing file
  std::ofstream timing_file;
  timing_file.open("rank_"+std::to_string(rank)+"_timing.csv");
  MPI_Barrier(MPI_COMM_WORLD);

  if(rank==0) {

    double constructor_start = MPI_Wtime();
    SmartSimClient client(true);
    double constructor_end = MPI_Wtime();
    double delta_t = constructor_end - constructor_start;
    timing_file << rank << "," << "client()" << ","
                << delta_t << std::endl << std::flush;


    std::string model_key = "resnet_model";
    std::string model_file = "./resnet50.pt";
    double model_set_start = MPI_Wtime();
    client.set_model_from_file(model_key, model_file, "TORCH", "GPU", 25);
    double model_set_end = MPI_Wtime();
    delta_t = model_set_end - model_set_start;
    timing_file << rank << "," << "model_set" << ","
                << delta_t << std::endl << std::flush;

    std::string script_key = "resnet_script";
    std::string script_file = "./data_processing_script.txt";

    double script_set_start = MPI_Wtime();
    client.set_script_from_file(script_key, "GPU", script_file);
    double script_set_end = MPI_Wtime();
    delta_t = script_set_end - script_set_start;
    timing_file << rank << "," << "script_set" << ","
                << delta_t << std::endl << std::flush;

    double model_get_start = MPI_Wtime();
    std::string_view model = client.get_model(model_key);
    double model_get_end = MPI_Wtime();
    delta_t = model_get_end - model_get_start;
    timing_file << rank << "," << "model_get" << ","
                << delta_t << std::endl << std::flush;

    double script_get_start = MPI_Wtime();
    std::string_view script = client.get_script(script_key);
    double script_get_end = MPI_Wtime();
    delta_t = script_get_end - script_get_start;
    timing_file << rank << "," << "script_get" << ","
                << delta_t << std::endl << std::flush;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  run_mnist("resnet_model", "resnet_script", timing_file);

  if(rank==0)
    std::cout<<"Finished Resnet test."<<std::endl;

  double main_end = MPI_Wtime();
  double delta_t = main_end - main_start;
  timing_file << rank << "," << "main()" << ","
                << delta_t << std::endl << std::flush;

  MPI_Finalize();

  return 0;
}
