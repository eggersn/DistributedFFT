#include "tests_slab_random_default.hpp"
#include "tests_slab_random_z_then_yx.hpp"
#include "tests_slab_random_y_then_zx.hpp"
#include "tests_pencil_random_1d.hpp"
#include "tests_pencil_random_2d.hpp"
#include "tests_pencil_random_3d.hpp"
#include "tests_reference.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>

void printInvalidArgHelp() {
    std::cout << "Invalid Arguments!" << std::endl;
    std::cout << "Option 1: \"Slab <testcase> <opt> <runs> <Nx> <Ny> <Nz> <allow_cuda_aware> <precision>\"" << std::endl;
    std::cout << "Option 2: \"Pencil <testcase> <opt> <runs> <Nx> <Ny> <Nz> <allow_cuda_aware> <precision> <P1> <P2>\"" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 1) {
        printInvalidArgHelp();
        return 1;
    }

    int testcase, opt, runs;
    int Nx, Ny, Nz;
    bool allow_cuda_aware;

    try {
        std::string option(argv[1]);
        std::stringstream sstream;
        sstream << std::string(argv[2]) << ' ' << std::string(argv[3]) << ' ' << std::string(argv[4]) << ' ' << std::string(argv[5]) << ' ' << std::string(argv[6]) << ' ' << std::string(argv[7]) ;
        sstream >> testcase >> opt >> runs >> Nx >> Ny >> Nz;

        allow_cuda_aware = (std::string(argv[8]).compare("true")==0);

        if (option.compare(0, 4, "Slab") == 0) {
            if (option.compare(0, 12, "Slab_Default")==0) {
                if (std::string(argv[9]).compare("double")==0) {
                    Tests_Slab_Random_Default<double> test;
                    test.setParams(Nx, Ny, Nz, allow_cuda_aware);
                    test.run(testcase, opt, runs);
                } else {
                    Tests_Slab_Random_Default<float> test;
                    test.setParams(Nx, Ny, Nz, allow_cuda_aware);
                    test.run(testcase, opt, runs);
                }
            } else if (option.compare(0, 14, "Slab_Y_Then_ZX")==0) {
                if (std::string(argv[9]).compare("double")==0) {
                    Tests_Slab_Random_Y_Then_ZX<double> test;
                    test.setParams(Nx, Ny, Nz, allow_cuda_aware);
                    test.run(testcase, opt, runs);
                } else {
                    Tests_Slab_Random_Y_Then_ZX<float> test;
                    test.setParams(Nx, Ny, Nz, allow_cuda_aware);
                    test.run(testcase, opt, runs);
                }                
            } else if (option.compare(0, 14, "Slab_Z_Then_YX")==0) {                   
                if (std::string(argv[9]).compare("double")==0) {
                    Tests_Slab_Random_Z_Then_YX<double> test;
                    test.setParams(Nx, Ny, Nz, allow_cuda_aware);
                    test.run(testcase, opt, runs);
                } else {
                    Tests_Slab_Random_Z_Then_YX<float> test;
                    test.setParams(Nx, Ny, Nz, allow_cuda_aware);
                    test.run(testcase, opt, runs);
                }                    
            } else {
                throw std::runtime_error("Invalid Testcase!");
            }
        } else if (option.compare(0, 6, "Pencil") == 0) {
            size_t P1, P2;
            sstream.str("");
            sstream.clear();
            sstream << std::string(argv[10]) << ' ' << std::string(argv[11]);
            sstream >> P1 >> P2;

            if (std::string(argv[9]).compare("double")==0) {
                Tests_Pencil_Random<double> *test;

                if (option.compare("Pencil_1D") == 0)
                    test = new Tests_Pencil_Random_1D<double>();
                else if (option.compare("Pencil_2D") == 0)
                    test = new Tests_Pencil_Random_2D<double>();
                else 
                    test = new Tests_Pencil_Random_3D<double>();

                test->setParams(Nx, Ny, Nz, allow_cuda_aware, P1, P2);
                test->run(testcase, opt, runs);
                delete test;
            } else {
                Tests_Pencil_Random<float> *test;

                if (option.compare("Pencil_1D") == 0)
                    test = new Tests_Pencil_Random_1D<float>();
                else if (option.compare("Pencil_2D") == 0)
                    test = new Tests_Pencil_Random_2D<float>();
                else 
                    test = new Tests_Pencil_Random_3D<float>();
                    
                test->setParams(Nx, Ny, Nz, allow_cuda_aware, P1, P2);
                test->run(testcase, opt, runs);
                delete test;
            }   
        } else if (option.compare("Reference") == 0) {
            size_t P1, P2;
            sstream.str("");
            sstream.clear();
            sstream << std::string(argv[10]) << ' ' << std::string(argv[11]);
            sstream >> P1 >> P2;

            if (std::string(argv[9]).compare("double")==0) {
                Tests_Reference<double> test(Nx, Ny, Nz, allow_cuda_aware, P1, P2);
                test.run(testcase, opt, runs);
            } else {
                Tests_Reference<float> test(Nx, Ny, Nz, allow_cuda_aware, P1, P2);
                test.run(testcase, opt, runs);
            }
        } else {
            printInvalidArgHelp();
            return 1;
        }

    } catch (int e) {
        printf("error %d\n", e);
        // printInvalidArgHelp();
        return e;
    }

    return 0;
}