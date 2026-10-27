// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

/*! minimal host-only replacements for the CUDA vector types and
    function qualifiers that deepCompositing.h uses; only included in
    the CPU build (DC_CPU), where neither cuda_runtime.h nor nvcc is
    available. The layout matches the CUDA types, so code and data
    written against the CUDA version of the API stays compatible. */

#if !defined(__CUDACC__)

#ifndef __device__
# define __device__ /* host build: ignore */
#endif
#ifndef __host__
# define __host__   /* host build: ignore */
#endif
#ifndef __global__
# define __global__ /* host build: ignore */
#endif

struct int2   { int x, y; };
struct uint2  { unsigned int x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };

inline int2   make_int2(int x, int y)
{ return int2{x,y}; }
inline uint2  make_uint2(unsigned int x, unsigned int y)
{ return uint2{x,y}; }
inline float3 make_float3(float x, float y, float z)
{ return float3{x,y,z}; }
inline float4 make_float4(float x, float y, float z, float w)
{ return float4{x,y,z,w}; }

#endif
