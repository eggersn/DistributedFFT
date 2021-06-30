#include <iostream>
#include <sstream>
#include <string>
#include "params.hpp"
#include "tests_pencil_random_1d.hpp"
#include "tests_pencil_random_2d.hpp"
#include "tests_pencil_random_3d.hpp"

void printHelp() {
   printf("Usage: mpirun -n P [mpi args] pencil [options] \n");
   printf("Options (required):\n");
   printf(" --input-dim-x [-nx]: \tDefines the size of the input data in x-direction.\n");
   printf(" --input-dim-y [-ny]: \tDefines the size of the input data in y-direction.\n");
   printf(" --input-dim-z [-nz]: \tDefines the size of the input data in z-direction.\n");
   printf(" --partition1 [-p1]: \tSpecifies the number of partitions in x-direction.\n");
   printf(" --partition2 [-p2]: \tSpecifies the number of partitions in y-direction.\n");
   printf("Options (optional):\n");
   printf(" --comm-method [-comm]: Specifies whether to use \"Peer2Peer\" or \"All2All\" MPI communication.\n");
   printf(" --send-method [-snd]: \tThere are 3 available selections:\n");
   printf("\t1. Sync: \tThis is the default option. Here, we use cudaDeviceSync before calling MPI_Isend for each receiving rank.\n");
   printf("\t2. Streams: \tUses cudaStreams for cudaMemcpyAsync along with cudaCallHostFunc to notify a second thread to call MPI_Isend. This option requires MPI_THREAD_MULTIPLE.\n");
   printf("\t3. MPI_Type: \tUses MPI_Datatype to avoid using cudaMemcpy2D. If MPI is not CUDA-aware, the sender still has to perform a cudaMemcpy1D (D->H).\n");
   printf(" --testcase [-t]: \tSpecifies which test should be executed.\n");
   printf("   Available selections are:\n");
   printf("\t--testcase 0:\tEach rank generates a random input of size (Nx/P1) x (Ny/P1) x Nz. Here, P1*P2 = P must hold.\n");
   printf("\t--testcase 1:\tRank 0 generates the global input and distributes the pencils while computing the complete 3D FFT. Afterwards rank 0 compares its local result with the distributed result. Here, P1*P2+1 = P must hold.\n");
   printf(" --opt [-o]: \t\tSpecifies which option to use.\n");
   printf("   Available selections are:\n");
   printf("\t--opt 0:\tDefault selection, where no coordinate transformation is performed. This option requires multiple plans for the 1D-FFT in y-direction.\n");
   printf("\t--opt 1:\tThe algorithm performs a coordinate transform. Starting from the default data alignment [x][y][z] (z continuous), the 1D-FFT in z-direction transforms the coordinate system into [z][x][y]. Analogously, the 1D-FFT in y-direction into [y][z][x] and finally the 1D-FFT in x-direction into [x][y][z] again.\n");
   printf(" --fft-dim [-f]: \tSpecifies the number of dimension computed by the algorithm. Available selections are 1, 2, and 3 (default).");
   printf(" --iterations [-i]: \tSpecifies how often the given testcase should be repeated.\n");
   printf(" --warmup-rounds [-w]: \tThis value is added to the number of iterations. For a warmup round, the performance metrics are not stored.\n");
   printf(" --cuda_aware [-c]: \tIf set and available, device memory pointer are used to call MPI routines.\n");
   printf(" --double_prec [-d]: \tIf set, double precision is used.\n");
   printf(" --benchmark_dir [-b]: \tSets the prefix for the benchmark director (default is ../benchmarks).\n");
   printf("\n");
   printf("Example: \n");
   printf("\"mpirun -n 4 pencil -nx 256 -ny 256 -nz 256 -p1 2 -p2 2 -snd Streams -o 1 -i 10 -c -b ../new_benchmarks\"\n");
   printf("Here, four MPI processes are started which execute the default testcase using option 1. Each rank start with input size 128x128x256. A sending rank uses the \"Streams\" method.");
   printf(" CUDA-aware MPI is enabled, the algorithm performs 10 iterations of the testcase, and the benchmark results are saved under ../new_benchmarks (relative to the build dir).");
}

struct PencilParams {
   size_t Nx, Ny, Nz;
   size_t P1, P2;
   int fft_dim;
   int testcase = 0;
   int opt = 0;
   int iterations = 1;
   int warmup_rounds = 0;
   bool cuda_aware = false;
   bool double_prec = false;
   std::string benchmark_dir = "../benchmarks";
   CommunicationMethod comm_method = Peer2Peer;
   SendMethod send_method = Sync;
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

PencilParams parseParams(int argc, char *argv[]) {
   PencilParams params;
   // Get Nx, Ny, and Nz
   params.Nx = StringToSize_t(getValueOfParam(argc, argv, "--input-dim-x", "-nx"), true, "Input parameter Nx is required.");
   params.Ny = StringToSize_t(getValueOfParam(argc, argv, "--input-dim-y", "-ny"), true, "Input parameter Ny is required.");
   params.Nz = StringToSize_t(getValueOfParam(argc, argv, "--input-dim-z", "-nz"), true, "Input parameter Nz is required.");
   params.P1 = StringToSize_t(getValueOfParam(argc, argv, "--partition1", "-p1"), true, "Input parameter P1 is required.");
   params.P2 = StringToSize_t(getValueOfParam(argc, argv, "--partition2", "-p2"), true, "Input parameter P2 is required.");

   params.fft_dim = StringToInt(getValueOfParam(argc, argv, "--fft-dim", "-f"));
   if (params.fft_dim == 0)
      params.fft_dim = 3;
   else if (params.fft_dim < 0 || params.fft_dim > 3)
      throw std::runtime_error("Invalid FFT dimension.");

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

   std::string comm_method_str = getValueOfParam(argc, argv, "--comm-method", "-comm");
   if (comm_method_str.compare("Peer2Peer"))
      params.comm_method = Peer2Peer;
   else if (comm_method_str.compare("All2All"))
      params.comm_method = All2All;
   else 
      throw std::runtime_error("Invalid communication method.");

   std::string send_method_str = getValueOfParam(argc, argv, "--send-method", "-snd");
   if (send_method_str.compare("Sync"))
      params.send_method = Sync;
   else if (send_method_str.compare("Streams"))
      params.send_method = Streams;
   else if (send_method_str.compare("MPI_Type"))
      params.send_method = MPI_Type;
   else 
      throw std::runtime_error("Invalid send method.");

   // Check selected testcase and option
   params.testcase = StringToInt(getValueOfParam(argc, argv, "--testcase", "-t"));
   if (params.testcase < 0 || params.testcase > 1)
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
      PencilParams params = parseParams(argc, argv);
      params.cuda_aware = params.cuda_aware * MPIX_Query_cuda_support();

      Configurations config = {params.cuda_aware, params.warmup_rounds, params.comm_method, params.send_method, params.benchmark_dir};

      if (params.double_prec) {
         Tests_Pencil_Random<double> *test;
         if (params.fft_dim == 1) {
            test = new Tests_Pencil_Random_1D<double>();
         } else if (params.fft_dim == 2) {
            test = new Tests_Pencil_Random_2D<double>();
         } else if (params.fft_dim == 3) {
            test = new Tests_Pencil_Random_3D<double>();
         }
         test->setParams(params.Nx, params.Ny, params.Nz, config, params.P1, params.P2);
         test->run(params.testcase, params.opt, params.iterations);
      } else {
         Tests_Pencil_Random<float> *test;
         if (params.fft_dim == 1) {
            test = new Tests_Pencil_Random_1D<float>();
         } else if (params.fft_dim == 2) {
            test = new Tests_Pencil_Random_2D<float>();
         } else if (params.fft_dim == 3) {
            test = new Tests_Pencil_Random_3D<float>();
         }
         test->setParams(params.Nx, params.Ny, params.Nz, config, params.P1, params.P2);
         test->run(params.testcase, params.opt, params.iterations);
      }

   } catch (std::runtime_error& e) {
      printf("%s\n\n", e.what());
      printf("Use \"--help\" or \"-h\" to display the help menu.\n");
    }
   
   return 0; 
} 