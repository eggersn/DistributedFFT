#include <iostream>
#include <sstream>
#include <string>
#include "tests_reference.hpp"


void printHelp() {
   printf("Usage: mpirun -n X [mpi args] reference [options] \n");
   printf("Options (required):\n");
   printf(" --input-dim-x [-nx]: \tDefines the size of the input data in x-direction.\n");
   printf(" --input-dim-y [-ny]: \tDefines the size of the input data in y-direction.\n");
   printf(" --input-dim-z [-nz]: \tDefines the size of the input data in z-direction.\n");
   printf("Options (optional):\n");
   printf(" --testcase [-t]: \tSpecifies which test should be executed.\n");
   printf("   Available values are:\n");
   printf("\t --testcase 0:\tEach rank sends its input to rank 0, where the complete 3D FFT is computed.\n");
   printf("\t --testcase 1:\tCompares the bandwidth of different MPI communication methods.\n");
   printf("\t   To select the communcation method, use \"--opt\" (or \"-o\"). For this testcase, available methods are:\n");
   printf("\t\t --opt 0: MPI Peer2Peer communcation.\n");
   printf("\t\t --opt 0: MPI All2All communcation.\n");
   printf("\t --testcase 2:\tCompares the bandwidth of different MPI sending methods, where the sender has to perform a cudaMemcpy2D (relevant for slab decomposition).\n");
   printf("\t   For this testcase, the arguments \"--partition1\" (or \"-p1\") and \"--partition2\" (or \"-p2\") are required!\n");
   printf("\t   To select the sending method, use \"--opt\" (or \"-o\"). For this testcase, available methods are:\n");
   printf("\t\t --opt 0: The sending rank performs a cudaMemcpy2D for each receiver, using cudaDeviceSync before MPI_Isend.\n");
   printf("\t\t --opt 1: The sending rank performs a cudaMemcpy2D for each receiver, using cudaStreams and cudaCallHostFunc to notify a second thread to call MPI_Isend. Here, MPI has to support MPI_THREAD_MULTIPLE.\n");
   printf("\t\t --opt 2: The sending rank uses a custom MPI_Datatype to avoid using cudaMemcpy2D. If MPI is not CUDA-aware, the sender still has to perform a cudaMemcpy1D (D->H).\n");
   printf("\t --testcase 3:\tCompares the bandwidth of different MPI sending methods, where the sender and receiver have to perform a cudaMemcpy2D (relevant for pencil decomposition).\n");
   printf("\t   For this testcase, the arguments \"--partition1\" (or \"-p1\") and \"--partition2\" (or \"-p2\") are required!\n");
   printf("\t   To select the sending method, use \"--opt\" (or \"-o\"). For this testcase, available methods are:\n");
   printf("\t\t --opt 0: The sending rank performs a cudaMemcpy2D for each receiver, using cudaDeviceSync before MPI_Isend.\n");
   printf("\t\t --opt 1: The sending rank performs a cudaMemcpy2D for each receiver, using cudaStreams and cudaCallHostFunc to notify a second thread to call MPI_Isend. Here, MPI has to support MPI_THREAD_MULTIPLE.\n");
   printf("\t\t --opt 2: The sending rank uses a custom MPI_Datatype to avoid using cudaMemcpy2D. If MPI is not CUDA-aware, the sender still has to perform a cudaMemcpy1D (D->H).\n");
   printf(" --opt [-o]: \t\tSpecifies which option should be used (depending on the testcase).\n");
   printf(" --partition1 [-p1]: \tSpecifies the number of partitions in x-direction.\n");
   printf(" --partition2 [-p2]: \tSpecifies the number of partitions in y-direction.\n");
   printf(" --iterations [-i]: \tSpecifies how often the given testcase should be repeated. For testcases 1-3, the bandwidth is computed as the average across the number of iterations. Default value is 1.\n");
   printf(" --warmup-rounds [-w]: \tThis value is added to the number of iterations. For a warmup round, the performance metrics are not stored.\n");
   printf(" --cuda_aware [-c]: \tIf set and available, device memory pointer are used to call MPI routines.\n");
   printf(" --double_prec [-d]: \tIf set, double precision is used.\n");
   printf(" --benchmark_dir [-b]: \tSets the prefix for the benchmark director (default is ../benchmarks).\n");
   printf("\n");
   printf("Example: \n");
   printf("\"mpirun -n 4 reference -nx 256 -ny 256 -nz 256 -t 2 -o 1 -p1 2 -p2 2 -i 10 -c -b ../new_benchmarks\"\n");
   printf("Here, four MPI processes are started which execute the testcase 2 using option 1 (see above).");
   printf(" The input data is of size 256^3, where both x- and y-direction are partitioned (thus, each rank starts with input size 128x128x256).");
   printf(" The bandwidht is computed as the average across 10 iterations while CUDA-aware MPI is used. The results are stored in ../new_benchmarks (relative to the build dir).");
}

struct ReferenceParams {
   size_t Nx, Ny, Nz;
   size_t P1, P2;
   int testcase;
   int opt;
   int iterations;
   int warmup_rounds;
   bool cuda_aware;
   bool double_prec;
   std::string benchmark_dir;
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

ReferenceParams parseParams(int argc, char *argv[]) {
   ReferenceParams params;
   // Get Nx, Ny, and Nz
   params.Nx = StringToSize_t(getValueOfParam(argc, argv, "--input-dim-x", "-nx"), true, "Input parameter Nx is required.");
   params.Ny = StringToSize_t(getValueOfParam(argc, argv, "--input-dim-y", "-ny"), true, "Input parameter Ny is required.");
   params.Nz = StringToSize_t(getValueOfParam(argc, argv, "--input-dim-z", "-nz"), true, "Input parameter Nz is required.");

   params.iterations = StringToInt(getValueOfParam(argc, argv, "--iterations", "-i"));
   if (params.iterations == 0)
      params.iterations == 1;
   params.warmup_rounds = StringToInt(getValueOfParam(argc, argv, "--warmup-rounds", "-w"));
   params.iterations += params.warmup_rounds;
   params.cuda_aware = checkFlag(argc, argv, "--cuda-aware", "-c");
   params.double_prec = checkFlag(argc, argv, "--double_prec", "-d");
   params.benchmark_dir = getValueOfParam(argc, argv, "--benchmark-dir", "-b");
   if (params.benchmark_dir.compare("") == 0)
      params.benchmark_dir = "../benchmarks";

   // Check selected testcase and option
   params.testcase = StringToInt(getValueOfParam(argc, argv, "--testcase", "-t"));
   params.opt = StringToInt(getValueOfParam(argc, argv, "--opt", "-o"));
   if (params.testcase == 0) {
      if (params.opt != 0)
         throw std::runtime_error("Invalid option for selected testcase.");
      params.P1 = StringToSize_t(getValueOfParam(argc, argv, "--partition1", "-p1"), true, "P1 is required for this testcase.");
      params.P2 = StringToSize_t(getValueOfParam(argc, argv, "--partition2", "-p2"), true, "P2 is required for this testcase.");
   } else if (params.testcase == 1) {
      if (params.opt < 0 || params.opt > 2)
         throw std::runtime_error("Invalid option for selected testcase.");
   } else if (params.testcase == 2) {
      if (params.opt < 0 || params.opt > 3)
         throw std::runtime_error("Invalid option for selected testcase.");
      params.P1 = StringToSize_t(getValueOfParam(argc, argv, "--partition1", "-p1"), true, "P1 is required for this testcase.");
      params.P2 = StringToSize_t(getValueOfParam(argc, argv, "--partition2", "-p2"), true, "P2 is required for this testcase.");
   } else if (params.testcase == 3) {
      if (params.opt < 0 || params.opt > 3)
         throw std::runtime_error("Invalid option for selected testcase.");
      params.P1 = StringToSize_t(getValueOfParam(argc, argv, "--partition1", "-p1"), true, "P1 is required for this testcase.");
      params.P2 = StringToSize_t(getValueOfParam(argc, argv, "--partition2", "-p2"), true, "P2 is required for this testcase.");
   } else {
      throw std::runtime_error("Invalid testcase.");
   }

   return params;
}

int main(int argc, char *argv[]) 
{ 
   if (argc == 1 || (argc == 2 && (std::string(argv[1]).compare("--help") == 0 || std::string(argv[1]).compare("-h") == 0))) {
      printHelp();
      return 0;
   }
   try {
      ReferenceParams params = parseParams(argc, argv);
      params.cuda_aware = params.cuda_aware * MPIX_Query_cuda_support();
      if (params.double_prec) {
         Tests_Reference<double> test(params.Nx, params.Ny, params.Nz, params.cuda_aware, params.warmup_rounds, params.benchmark_dir, params.P1, params.P2);
         test.run(params.testcase, params.opt, params.iterations);
      } else {
         Tests_Reference<float> test(params.Nx, params.Ny, params.Nz, params.cuda_aware, params.warmup_rounds, params.benchmark_dir, params.P1, params.P2);
         test.run(params.testcase, params.opt, params.iterations);
      }
   } catch (std::runtime_error& e) {
      printf("%s\n\n", e.what());
      printf("Use \"--help\" or \"-h\" to display the help menu.\n");
    }
   
   return 0; 
} 