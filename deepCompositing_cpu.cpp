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

/* Host (CPU) implementation of the deepCompositing library: the same
   algorithm and the same data layout as deepCompositing.cu, but all
   buffers live in host memory, the CUDA kernels are OpenMP loops, the
   cub prefix sums are serial scans, and MPI works on host buffers (no
   CUDA-aware MPI needed). Selected with the CMake option DC_CPU. */

#include "deepCompositing.h"
#include "mpi_helper.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
# include <omp.h>
#endif

namespace dc {

  std::string Compositor::dumpFileName = "";

  namespace {

    template <typename T>
    T *hostAlloc(size_t numItems)
    {
      if (numItems == 0) numItems = 1;
      T *ptr = (T *)malloc(numItems*sizeof(T));
      if (!ptr)
        throw std::runtime_error("deepCompositing: out of host memory");
      return ptr;
    }

    template <typename T>
    void hostFree(T *&ptr)
    {
      free(ptr);
      ptr = nullptr;
    }

    inline bool operator==(const int2 &a, const int2 &b)
    { return a.x == b.x && a.y == b.y; }

    inline uint32_t make_8bit(const float f)
    {
      return std::min(255,std::max(0,int(f*256.f)));
    }

    /*! unlike the CUDA version (which always writes alpha=255), the
        host version stores the accumulated coverage in the alpha
        byte, so that a background can be blended in afterwards; for
        fully covered pixels both versions produce the same value */
    inline uint32_t make_rgba(const float3 color, const float alpha)
    {
      return
        (make_8bit(color.x) << 0) +
        (make_8bit(color.y) << 8) +
        (make_8bit(color.z) << 16) +
        (make_8bit(alpha) << 24);
    }

    inline int computeCountersPerByte(uint32_t maxFragsPerPixel)
    {
      assert(maxFragsPerPixel <= 255);
      if (maxFragsPerPixel < 4) return 4;
      else if (maxFragsPerPixel < 16) return 2;
      else return 1;
    }

    /*! in-place inclusive prefix sum over 'n' uint32 values (the
        equivalent of the cub::DeviceScan::InclusiveSum calls in the
        CUDA version); two-pass parallel scan for large arrays */
    void inclusiveScan(uint32_t *values, size_t n)
    {
#ifdef _OPENMP
      const int numThreads = omp_get_max_threads();
      if (n < (size_t(1) << 16) || numThreads == 1) {
        std::inclusive_scan(values,values+n,values);
        return;
      }
      std::vector<uint32_t> blockSums(numThreads+1,0);
#pragma omp parallel num_threads(numThreads)
      {
        const int t = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const size_t begin = (n*t)/nt;
        const size_t end   = (n*(t+1))/nt;
        uint32_t sum = 0;
        for (size_t i=begin;i<end;i++) { sum += values[i]; values[i] = sum; }
        blockSums[t+1] = sum;
#pragma omp barrier
#pragma omp single
        for (int i=1;i<=nt;i++) blockSums[i] += blockSums[i-1];
        const uint32_t ofs = blockSums[t];
        if (ofs)
          for (size_t i=begin;i<end;i++) values[i] += ofs;
      }
#else
      std::inclusive_scan(values,values+n,values);
#endif
    }

  } // ::<anonymous>

  /*! resize the (host) frame buffer to given size */
  void Compositor::resize(const int2 &newSize)
  {
    if (fbSize == newSize) /* already same size - just skip */ return;

    fbSize = newSize;
    deviceData.fbSize = newSize;

    hostFree(compactedFragmentLists);
    hostFree(fixedSizeFragmentLists);
    hostFree(lowBitCounters);
    hostFree(fullIntCounters);

    this->numPixelsOrg = fbSize.x*fbSize.y;
    this->numPixelsPadded = nextMultipleOf(numPixelsOrg,numCountersPerByte);
    uint32_t compactCounterBytes = numPixelsPadded / numCountersPerByte;

    /* same multi-purpose use of the counters[] array as in the CUDA
       version: per-pixel counters during rendering, prefix-sum offsets
       during compaction, and received (decoded) counters/offsets of
       all ranks for this rank's pixel range during compositing */
    size_t numIntsForCountersArray
      = std::max<size_t>(numPixelsPadded,
                         size_t(numRanks())*computeNumPixelsOnThisRank());
    fullIntCounters = hostAlloc<uint32_t>(numIntsForCountersArray);
    memset(fullIntCounters,0,numIntsForCountersArray*sizeof(uint32_t));
    lowBitCounters = hostAlloc<uint8_t>(compactCounterBytes);

    /*! same for fragments arrays: used both for our own fragments
      during rendering, and for receiving compact fragment lists */
    size_t maxNumFragments
      = std::max<size_t>(/* how many we could possibly produce ourselves: */
                         size_t(numPixelsPadded)*maxFragsPerPixel,
                         /* how many we could possibly receive: */
                         size_t(numRanks())*computeNumPixelsOnThisRank()*maxFragsPerPixel);
    fixedSizeFragmentLists = hostAlloc<Fragment>(maxNumFragments);
    compactedFragmentLists = hostAlloc<Fragment>(maxNumFragments);
  }

  /*! on the host there is no GPU to pick; kept for API compatibility
      with the CUDA version (always returns -1 and does nothing) */
  int Compositor::affinitizeGPU()
  {
    affinitizedGPU = -1;
    return -1;
  }

  /*! for debugging - save current state to disk (same file format as
      the CUDA version) */
  void Compositor::save(const std::string &fileName)
  {
    std::ofstream out(fileName,std::ios::binary);
    uint32_t numCompactedFragments = fullIntCounters[numPixelsOrg-1];
    out.write((char *)&fbSize,sizeof(fbSize));
    out.write((char *)fullIntCounters,numPixelsOrg*sizeof(fullIntCounters[0]));
    out.write((char *)&numCompactedFragments,sizeof(numCompactedFragments));
    out.write((char *)compactedFragmentLists,
              numCompactedFragments*sizeof(compactedFragmentLists[0]));
    std::cout << "SUCCESSFULLY DUMPED FRAME BUFFER, fbsize=" << fbSize.x << "x" << fbSize.y
              << " numFrags=" << numCompactedFragments << std::endl;
  }

  /*! for debugging - load output from prev 'save' back in (host memory) */
  SavedFrags SavedFrags::load(const std::string &fileName)
  {
    SavedFrags result;
    std::ifstream in(fileName,std::ios::binary);
    if (!in.good())
      throw std::runtime_error("cloud not load deep frame buffer from "+fileName);
    in.read((char *)&result.fbSize,sizeof(result.fbSize));
    const size_t numPixels = size_t(result.fbSize.x)*result.fbSize.y;
    result.counters = hostAlloc<uint32_t>(numPixels);
    in.read((char*)result.counters,numPixels*sizeof(*result.counters));
    int numFrags;
    in.read((char *)&numFrags,sizeof(numFrags));
    result.fragments = hostAlloc<Fragment>(numFrags);
    in.read((char*)result.fragments,numFrags*sizeof(*result.fragments));
    return result;
  }

  Compositor::Compositor(int maxFragsPerPixel,
                         MPI_Comm comm)
    : comm(comm),
      maxFragsPerPixel(maxFragsPerPixel),
      numCountersPerByte(computeCountersPerByte(maxFragsPerPixel))
  {
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
  }

  DeviceInterface Compositor::prepare()
  {
    if (fbSize.x <= 0 || fbSize.y <= 0)
      throw std::runtime_error("invalid or un-specified frame buffer size in dc library..."
                               "did you forget a 'DeepCompositor::resize()'?");

    memset(fullIntCounters,0,numPixelsPadded*sizeof(uint32_t));

    deviceData.counters = fullIntCounters;
    deviceData.fbSize = fbSize;
    deviceData.maxFragsPerPixel = maxFragsPerPixel;
    deviceData.fragments = fixedSizeFragmentLists;
    return deviceData;
  }

  void Compositor::finish(uint32_t *whereToWriteFinalPixels)
  {
    if (rank == 0 && whereToWriteFinalPixels == nullptr)
      throw std::runtime_error("dc::Compositor::finish(): rank 0 needs a "
                               "pointer to write the final pixels to");

    size_t stat_numBytesCountersOut = 0;
    size_t stat_numBytesCountersIn = 0;
    size_t stat_numBytesFragsOut = 0;
    size_t stat_numBytesFragsIn = 0;

    const uint32_t numPixelsOnThisRank
      = pixelEnd(rank) - pixelBegin(rank);

    // ------------------------------------------------------------------
    // compact counters: 32-bit counters -> 2/4/8-bit values
    // ------------------------------------------------------------------
    const int64_t numCCBytes = numPixelsPadded / numCountersPerByte;
    {
      const int bitsPerCounter = 8 / numCountersPerByte;
#pragma omp parallel for schedule(static)
      for (int64_t i=0;i<numCCBytes;i++) {
        uint8_t loBitCounter = 0;
        for (uint32_t j=0;j<numCountersPerByte;j++)
          loBitCounter += (fullIntCounters[numCountersPerByte*i+j] << (j*bitsPerCounter));
        lowBitCounters[i] = loBitCounter;
      }
    }

    // ------------------------------------------------------------------
    // compact fragments - eliminate unused fragments by compacting
    // fixed-length lists into compact sequential lists; this also
    // turns the counters[] array into an offsets[] array
    // ------------------------------------------------------------------
    memset(fullIntCounters+numPixelsOrg,0,
           (numPixelsPadded-numPixelsOrg)*sizeof(uint32_t));
    inclusiveScan(fullIntCounters,numPixelsPadded);
#pragma omp parallel for schedule(static)
    for (int64_t pixelID=0;pixelID<int64_t(numPixelsPadded);pixelID++) {
      const uint32_t end = fullIntCounters[pixelID];
      const uint32_t begin = pixelID ? fullIntCounters[pixelID-1] : 0;
      const Fragment *in = fixedSizeFragmentLists+pixelID*maxFragsPerPixel;
      std::copy(in,in+(end-begin),compactedFragmentLists+begin);
    }

    if (dumpFileName != "")
      save(dumpFileName);

    // ------------------------------------------------------------------
    // number of fragments (and counter bytes) sent to / received from
    // each other rank
    // ------------------------------------------------------------------
    std::vector<int> numBytesTo(size), ofsBytesTo(size);
    std::vector<int> numBytesFrom(size), ofsBytesFrom(size);
    std::vector<int> numCounterBytesTo(size), ofsCounterBytesTo(size);
    std::vector<int> numCounterBytesFrom(size), ofsCounterBytesFrom(size);

    size_t sumBytesTo = 0;
    uint32_t ofsTo = 0;
    for (int node=0;node<size;node++) {
      const uint32_t node_pixel_begin = pixelBegin(node);
      const uint32_t node_pixel_end   = pixelEnd(node);
      const uint32_t numFrags
        = (node_pixel_end  ?fullIntCounters[node_pixel_end  -1]:0)
        - (node_pixel_begin?fullIntCounters[node_pixel_begin-1]:0);
      const size_t numBytes = size_t(numFrags)*sizeof(Fragment);
      if (numBytes > size_t(INT32_MAX) || sumBytesTo > size_t(INT32_MAX))
        throw std::runtime_error("deepCompositing: fragment exchange exceeds 2GB per rank");
      numBytesTo[node] = int(numBytes);
      ofsBytesTo[node] = int(sumBytesTo);
      sumBytesTo += numBytes;
      if (node != rank) stat_numBytesFragsOut += numBytes;

      numCounterBytesTo[node] = (node_pixel_end - node_pixel_begin)/numCountersPerByte;
      ofsCounterBytesTo[node] = ofsTo;
      ofsTo += numCounterBytesTo[node];
      if (node != rank) stat_numBytesCountersOut += numCounterBytesTo[node];
    }

    uint32_t sumCounterBytesReceiving = 0;
    for (int node=0;node<size;node++) {
      numCounterBytesFrom[node] = numPixelsOnThisRank / numCountersPerByte;
      ofsCounterBytesFrom[node] = sumCounterBytesReceiving;
      sumCounterBytesReceiving += numCounterBytesFrom[node];
      if (node != rank) stat_numBytesCountersIn += numCounterBytesFrom[node];
    }

    // ------------------------------------------------------------------
    // exchange compressed counters
    // ------------------------------------------------------------------
    std::vector<uint8_t> recvCompactCounters(std::max(1u,sumCounterBytesReceiving));
    MPI_CALL(Alltoallv(lowBitCounters,numCounterBytesTo.data(),
                       ofsCounterBytesTo.data(),MPI_BYTE,
                       recvCompactCounters.data(),numCounterBytesFrom.data(),
                       ofsCounterBytesFrom.data(),MPI_BYTE,
                       comm));

    // ------------------------------------------------------------------
    // uncompress the received counters - this overwrites the fullint
    // array that we so far used as offset array
    // ------------------------------------------------------------------
    {
      const int bitsPerCounter = 8 / numCountersPerByte;
      const uint32_t mask = (1<<bitsPerCounter)-1;
#pragma omp parallel for schedule(static)
      for (int64_t i=0;i<int64_t(sumCounterBytesReceiving);i++) {
        const uint8_t bits = recvCompactCounters[i];
        for (uint32_t j=0;j<numCountersPerByte;j++)
          fullIntCounters[numCountersPerByte*i+j] = (bits >> (j*bitsPerCounter)) & mask;
      }
    }

    // prefix sum so we can look up each pixel's fragment range
    const size_t numCountersReceived = size_t(numRanks())*numPixelsOnThisRank;
    inclusiveScan(fullIntCounters,numCountersReceived);

    size_t sumBytesFrom = 0;
    for (int node=0;node<size;node++) {
      const size_t node_comp_begin = size_t(node+0)*numPixelsOnThisRank;
      const size_t node_comp_end   = size_t(node+1)*numPixelsOnThisRank;
      const uint32_t numFrags
        = (node_comp_end  ?fullIntCounters[node_comp_end  -1]:0)
        - (node_comp_begin?fullIntCounters[node_comp_begin-1]:0);
      const size_t numBytes = size_t(numFrags)*sizeof(Fragment);
      if (numBytes > size_t(INT32_MAX) || sumBytesFrom > size_t(INT32_MAX))
        throw std::runtime_error("deepCompositing: fragment exchange exceeds 2GB per rank");
      numBytesFrom[node] = int(numBytes);
      ofsBytesFrom[node] = int(sumBytesFrom);
      sumBytesFrom += numBytes;
      if (node != rank) stat_numBytesFragsIn += numBytes;
    }

    // ------------------------------------------------------------------
    // exchange the fragments themselves
    // ------------------------------------------------------------------
    Fragment *incomingFragments = fixedSizeFragmentLists;
    MPI_CALL(Alltoallv(compactedFragmentLists,numBytesTo.data(),
                       ofsBytesTo.data(),MPI_BYTE,
                       incomingFragments,numBytesFrom.data(),
                       ofsBytesFrom.data(),MPI_BYTE,
                       comm));

    // ==================================================================
    // composite: per pixel, collect the (per-rank z-sorted) fragment
    // lists of all ranks, sort them by z, and alpha-blend front to
    // back. Fragment colors are premultiplied (associated) alpha, as
    // in the CUDA compositeKernel.
    // ==================================================================
    std::vector<uint32_t> compositedColor(std::max(1u,numPixelsOnThisRank));
    const uint32_t *compOffsets = fullIntCounters;
#pragma omp parallel
    {
      std::vector<Fragment> frags;
      frags.reserve(size_t(size)*maxFragsPerPixel);
#pragma omp for schedule(dynamic,1024)
      for (int64_t pixelIdx=0;pixelIdx<int64_t(numPixelsOnThisRank);pixelIdx++) {
        frags.clear();
        for (int fromRank=0;fromRank<size;fromRank++) {
          const size_t idx = size_t(fromRank)*numPixelsOnThisRank+pixelIdx;
          const uint32_t begin = idx ? compOffsets[idx-1] : 0;
          const uint32_t end   = compOffsets[idx];
          for (uint32_t i=begin;i<end;i++) {
            Fragment f = incomingFragments[i];
            if (std::isnan(f.z)) f.z = 1e19f;
            frags.push_back(f);
          }
        }
        std::stable_sort(frags.begin(),frags.end(),
                         [](const Fragment &a, const Fragment &b)
                         { return a.z < b.z; });
        float alpha = 0.f;
        float3 color = make_float3(0.f,0.f,0.f);
        for (const Fragment &f : frags) {
          if (f.z >= 1e20f) break;
          const float4 fragColor = f.getRGBA();
          color.x += (1.f-alpha)*fragColor.x;
          color.y += (1.f-alpha)*fragColor.y;
          color.z += (1.f-alpha)*fragColor.z;
          alpha += (1.f-alpha)*fragColor.w;
          if (alpha >= 1.f) break;
        }
        compositedColor[pixelIdx] = make_rgba(color,alpha);
      }
    }

    // ==================================================================
    // gather the composited pixel ranges at rank 0
    // ==================================================================
    if (rank == 0) {
      std::vector<MPI_Request> requests(size,MPI_REQUEST_NULL);
      memcpy(whereToWriteFinalPixels,
             compositedColor.data(),
             std::min(numPixelsOnThisRank,numPixelsOrg)*sizeof(uint32_t));
      for (int node=1;node<size;node++) {
        int begin = pixelBegin(node);
        int end   = std::min(numPixelsOrg,pixelEnd(node));
        if (begin < end)
          MPI_CALL(Irecv(whereToWriteFinalPixels+begin,
                         (end-begin)*sizeof(*whereToWriteFinalPixels),
                         MPI_BYTE,node,0,comm,&requests[node]));
      }
      MPI_CALL(Waitall(size,requests.data(),MPI_STATUSES_IGNORE));
    } else {
      int begin = pixelBegin(rank);
      int end   = std::min(numPixelsOrg,pixelEnd(rank));
      if (begin < end)
        MPI_CALL(Send(compositedColor.data(),
                      (end-begin)*sizeof(uint32_t),
                      MPI_BYTE,0,0,comm));
    }

    static int g_pass = 0;
    if (g_pass++ == 0 && getenv("DC_VERBOSE"))
      printf("rank %i: bytes in/out for counters: %.2fMB/out %.2fMB, frags: %.2fMB/%.2fMB, total: %.2fMB/%.2fMB \n",
             rank,
             float(stat_numBytesCountersIn)/(1024*1024),
             float(stat_numBytesCountersOut)/(1024*1024),
             float(stat_numBytesFragsIn)/(1024*1024),
             float(stat_numBytesFragsOut)/(1024*1024),
             float(stat_numBytesCountersIn+stat_numBytesFragsIn)/(1024*1024),
             float(stat_numBytesCountersOut+stat_numBytesFragsOut)/(1024*1024)
             );
  }

}
