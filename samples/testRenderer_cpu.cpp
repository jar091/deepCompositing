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

/* host (DC_CPU) version of testRenderer.cu: the same semi-transparent
   cube test scene, with the CUDA kernel replaced by an OpenMP loop
   over pixels that writes into the host-side DeviceInterface */

#include "testRenderer.h"
#include "owl/common/math/vec.h"
#include "owl/common/math/box.h"

namespace dctest {
  using namespace owl::common;

  inline uint32_t scramble(vec3i idx)
  {
    enum { FNV_PRIME = 0x01000193 };
    uint32_t hash = 0x1234567;
    hash = hash * FNV_PRIME ^ idx.x;
    hash = hash * FNV_PRIME ^ idx.y;
    hash = hash * FNV_PRIME ^ idx.z;
    return hash;
  }

  inline bool isMine(vec3i idx,
                     int mpiRank,
                     int mpiSize)
  {
    return (scramble(idx)%mpiSize) == mpiRank;
  }

  inline bool intersects(const box3f &box,
                         const vec3f &org,
                         const vec3f &dir,
                         float &t)
  {
    vec3f lo = (box.lower - org) * rcp(dir);
    vec3f hi = (box.upper - org) * rcp(dir);
    vec3f nr = min(lo,hi);
    vec3f fr = max(lo,hi);
    t = max(0.f,reduce_max(nr));
    return reduce_min(fr) >= t;
  }

  void renderFragments(int testCubeRes,
                       const dc::DeviceInterface &deepFB,
                       const Camera &camera,
                       int mpiRank, int mpiSize)
  {
    assert(deepFB.fbSize.x > 0);
    assert(deepFB.fbSize.y > 0);

#pragma omp parallel for collapse(2) schedule(dynamic,16)
    for (int pixelY=0;pixelY<deepFB.fbSize.y;pixelY++)
      for (int pixelX=0;pixelX<deepFB.fbSize.x;pixelX++) {
        const vec3f org = camera.org;
        const vec3f dir
          = camera.dir_00
          + float(pixelX) * camera.dir_du
          + float(pixelY) * camera.dir_dv;

        if ((pixelY%mpiSize) == mpiRank) {
          const float t = pixelY / (float)(deepFB.getSize().y);
          const vec3f bgColor
            = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f)
            +         t *vec3f(0.5f, 0.7f, 1.0f);
          deepFB.write(pixelX,pixelY,
                       dc::Fragment(1e10f,make_float3(bgColor.x,bgColor.y,bgColor.z),1.f));
        }

        for (int iz=0;iz<testCubeRes;iz++)
          for (int iy=0;iy<testCubeRes;iy++)
            for (int ix=0;ix<testCubeRes;ix++) {
              if (!isMine({ix,iy,iz},mpiRank,mpiSize))
                continue;

              box3f box;
              box.lower = (vec3f(ix,iy,iz)+.1f) * 1.f/testCubeRes;
              box.upper = (vec3f(ix,iy,iz)+.9f) * 1.f/testCubeRes;
              float t;
              if (intersects(box,org,dir,t)) {
                vec3f color = owl::randomColor(12+mpiRank);
                deepFB.write(pixelX,pixelY,
                             dc::Fragment(t,make_float3(color.x,color.y,color.z),.8f));
              }
            }
      }
  }

}
