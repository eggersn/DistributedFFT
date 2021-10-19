/* 
* Copyright (C) 2021 Simon Egger
* 
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "tests_slab_random.hpp"

template<typename T> 
class Tests_Slab_Random_Z_Then_YX : public Tests_Slab_Random<T> {
public: 
    int run(const int test_case, const int opt, const int runs);
protected:
    int testcase0(const int opt, const int runs);
    int testcase1(const int opt, const int runs);
    int testcase2(const int opt, const int runs);
    int testcase3(const int opt, const int runs);
    int testcase4(const int opt, const int runs);
    int compute(const int rank, const int world_size, const int opt, const int runs);
    int coordinate(const int world_size, const int runs);

    using Tests_Slab_Random<T>::Nx;
    using Tests_Slab_Random<T>::Ny;
    using Tests_Slab_Random<T>::Nz;
    using Tests_Slab_Random<T>::config;
};