#include <iostream>
#include <sstream>
#include <string>
#include "params.hpp"
#include "tests_slab_random_default.hpp"
#include "tests_slab_random_z_then_yx.hpp"
#include "tests_slab_random_y_then_zx.hpp"

void printHelp() {
   printf("Usage: mpirun -n P [mpi args] slab [options] \n");
   printf("Options (required):\n");
   printf(" --input-dim-x [-nx]: \tDefines the size of the input data in x-direction.\n");
   printf(" --input-dim-y [-ny]: \tDefines the size of the input data in y-direction.\n");
   printf(" --input-dim-z [-nz]: \tDefines the size of the input data in z-direction.\n");
   printf("Options (optional):\n");
   printf(" --sequence [-s]: \tDefines the sequence of dimensions in which the FFT is computed. Available selections are \"ZY_Then_X\" (default), \"Z_Then_YX\" and \"Y_Then_ZX\"\n");
   printf(" --comm-method [-comm]: Specifies whether to use \"Peer2Peer\" or \"All2All\" MPI communication.\n");
   printf(" --send-method [-snd]: \tThere are 3 available selections:\n");
   printf("\t1. Sync: \tThis is the default option. Here, we use cudaDeviceSync before calling MPI_Isend for each receiving rank.\n");
   printf("\t2. Streams: \tUses cudaStreams for cudaMemcpyAsync along with cudaCallHostFunc to notify a second thread to call MPI_Isend. This option requires MPI_THREAD_MULTIPLE.\n");
   printf("\t3. MPI_Type: \tUses MPI_Datatype to avoid using cudaMemcpy2D. If MPI is not CUDA-aware, the sender still has to perform a cudaMemcpy1D (D->H).\n");
   printf(" --testcase [-t]: \tSpecifies which test should be executed.\n");
   printf("   Available selections are:\n");
   printf("\t--testcase 0:\tEach rank generates a random input of size (Nx/P) x Ny x Nz (P specified by mpirun).\n");
   printf("\t--testcase 1:\tRank 0 generates the global input and distributes the slabs while computing the complete 3D FFT. Afterwards rank 0 compares its local result with the distributed result.\n");
   printf(" --opt [-o]: \t\tSpecifies which option to use.\n");
   printf("   Available selections are:\n");
   printf("\t--opt 0:\tDefault selection, where no coordinate transformation is performed.\n");
   printf("\t--opt 1:\tDepending on the selected sequence (via \"-s\"), the algorithm performs a coordinate transform. In general, this enables the sending rank to avoid a cudaMemcpy2D, while requiring it from the receiving rank.\n");
   printf(" --iterations [-i]: \tSpecifies how often the given testcase should be repeated.\n");
   printf(" --warmup-rounds [-w]: \tThis value is added to the number of iterations. For a warmup round, the performance metrics are not stored.\n");
   printf(" --cuda_aware [-c]: \tIf set and available, device memory pointer are used to call MPI routines.\n");
   printf(" --double_prec [-d]: \tIf set, double precision is used.\n");
   printf(" --benchmark_dir [-b]: \tSets the prefix for the benchmark director (default is ../benchmarks).\n");
   printf("\n");
   printf("Example: \n");
   printf("\"mpirun -n 4 slab -nx 256 -ny 256 -nz 256 -s Z_Then_YX -snd Streams -o 1 -i 10 -c -b ../new_benchmarks\"\n");
   printf("Here, four MPI processes are started which execute the default testcase using Z_Then_YX as the sequence along with option 1. A sending rank uses the \"Streams\" method.");
   printf(" CUDA-aware MPI is enabled, the algorithm performs 10 iterations of the testcase, and the benchmark results are saved under ../new_benchmarks (relative to the build dir).");
}

struct SlabParams {
   size_t Nx, Ny, Nz;
   std::string sequence;
   int testcase;
   int opt;
   int iterations;
   int warmup_rounds;
   bool cuda_aware;
   bool double_prec;
   std::string benchmark_dir;
   CommunicationMethod comm_method;
   SendMethod send_method;
};

std::string getValueOfParam(int argc, char *argv[], std::string longdesc, std::string shortdesc) {
   for (int i = 0; i < argc; i++) {
      if (std::string(argv[i]).compare(longdesc) == 0 || std::string(argv[i]).compare(shortdesc) == 0)
         return std::string(argv[i+1]);
   }
   return "";
}

bool checkFlag(int argc, char *argv[], std::string longdesc, std::string shortdesc) {
   for (int i = 0; i < argc; i++) {
      if (std::string(argv[i]).compare(longdesc) == 0 || std::string(argv[i]).compare(shortdesc) == 0)
         return true;
   }
   return false;
}

size_t StringToSize_t(std::string str, bool req=false, std::string error="") {
   if (str.compare("") == 0) {
      if (req)
         throw std::runtime_error(error);
      else 
         return 0;
   }
   std::stringstream sstream;
   sstream << str;
   size_t val;
   sstream >> val;
   return val;
}

int StringToInt(std::string str, bool req=false, std::string error="") {
   if (str.compare("") == 0) {
      if (req)
         throw std::runtime_error(error);
      else 
         return 0;
   }
   std::stringstream sstream;
   sstream << str;
   int val;
   sstream >> val;
   return val;
} 

SlabParams parseParams(int argc, char *argv[]) {
   SlabParams params;
   // Get Nx, Ny, and Nz
   params.Nx = StringToSize_t(getValueOfParam(argc, argv, "--input-dim-x", "-nx"), true, "Input parameter Nx is required.");
   params.Ny = StringToSize_t(getValueOfParam(argc, argv, "--input-dim-y", "-ny"), true, "Input parameter Ny is required.");
   params.Nz = StringToSize_t(getValueOfParam(argc, argv, "--input-dim-z", "-nz"), true, "Input parameter Nz is required.");

   params.iterations = StringToInt(getValueOfParam(argc, argv, "--iterations", "-i"));
   params.warmup_rounds = StringToInt(getValueOfParam(argc, argv, "--warmup-rounds", "-w"));
   if (params.iterations == 0 && params.warmup_rounds == 0)
      params.iterations = 1;
   params.iterations += params.warmup_rounds;
   params.cuda_aware = checkFlag(argc, argv, "--cuda_aware", "-c");
   params.double_prec = checkFlag(argc, argv, "--double_prec", "-d");
   params.benchmark_dir = getValueOfParam(argc, argv, "--benchmark_dir", "-b");
   if (params.benchmark_dir.compare("") == 0)
      params.benchmark_dir = "../benchmarks";

   params.sequence = getValueOfParam(argc, argv, "--sequence", "-s");
   if (params.sequence.compare("") != 0 && params.sequence.compare("ZY_Then_X") != 0 && params.sequence.compare("Z_Then_YX") != 0 && params.sequence.compare("Y_Then_ZX") != 0)
      throw std::runtime_error("Invalid sequence.");

   std::string comm_method_str = getValueOfParam(argc, argv, "--comm-method", "-comm");
   if (comm_method_str.compare("Peer2Peer") == 0 || comm_method_str.compare("") == 0)
      params.comm_method = Peer2Peer;
   else if (comm_method_str.compare("All2All") == 0)
      params.comm_method = All2All;
   else 
      throw std::runtime_error("Invalid communication method.");

   std::string send_method_str = getValueOfParam(argc, argv, "--send-method", "-snd");
   if (send_method_str.compare("Sync") == 0 || send_method_str.compare("") == 0)
      params.send_method = Sync;
   else if (send_method_str.compare("Streams") == 0)
      params.send_method = Streams;
   else if (send_method_str.compare("MPI_Type") == 0)
      params.send_method = MPI_Type;
   else 
      throw std::runtime_error("Invalid send method.");

   // Check selected testcase and option
   params.testcase = StringToInt(getValueOfParam(argc, argv, "--testcase", "-t"));
   if (params.testcase < 0 || params.testcase > 4)
      throw std::runtime_error("Invalid testcase.");
   params.opt = StringToInt(getValueOfParam(argc, argv, "--opt", "-o"));
   if (params.opt < 0 || params.opt > 1)
      throw std::runtime_error("Invalid option.");

   return params;
}

int main(int argc, char *argv[]) 
{ 
   if (argc == 1 || (argc == 2 && (std::string(argv[1]).compare("--help") == 0 || std::string(argv[1]).compare("-h") == 0))) {
      printHelp();
      return 0;
   }
   try {
      SlabParams params = parseParams(argc, argv);
      params.cuda_aware = params.cuda_aware * MPIX_Query_cuda_support();

      Configurations config = {params.cuda_aware, params.warmup_rounds, params.comm_method, params.send_method, params.benchmark_dir};

      if (params.sequence.compare("ZY_Then_X") == 0 || params.sequence.compare("") == 0) {
         if (params.double_prec) {
            Tests_Slab_Random_Default<double> test;
            test.setParams(params.Nx, params.Ny, params.Nz, config);
            test.run(params.testcase, params.opt, params.iterations);
         } else {
            Tests_Slab_Random_Default<float> test;
            test.setParams(params.Nx, params.Ny, params.Nz, config);
            test.run(params.testcase, params.opt, params.iterations);
         }
      } else if (params.sequence.compare("Z_Then_YX") == 0) {
         if (params.double_prec) {
            Tests_Slab_Random_Z_Then_YX<double> test;
            test.setParams(params.Nx, params.Ny, params.Nz, config);
            test.run(params.testcase, params.opt, params.iterations);
         } else {
            Tests_Slab_Random_Z_Then_YX<float> test;
            test.setParams(params.Nx, params.Ny, params.Nz, config);
            test.run(params.testcase, params.opt, params.iterations);
         }
      } else if (params.sequence.compare("Y_Then_ZX") == 0) {
         if (params.double_prec) {
            Tests_Slab_Random_Y_Then_ZX<double> test;
            test.setParams(params.Nx, params.Ny, params.Nz, config);
            test.run(params.testcase, params.opt, params.iterations);
         } else {
            Tests_Slab_Random_Y_Then_ZX<float> test;
            test.setParams(params.Nx, params.Ny, params.Nz, config);
            test.run(params.testcase, params.opt, params.iterations);
         }
      }

   } catch (std::runtime_error& e) {
      printf("%s\n\n", e.what());
      printf("Use \"--help\" or \"-h\" to display the help menu.\n");
    }
   
   return 0; 
} 