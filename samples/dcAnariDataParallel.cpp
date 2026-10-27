// ======================================================================== //
// Copyright 2018-2024 Ingo Wald                                            //
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

/* Data-parallel ANARI rendering + deep compositing test (host build).

   The unit cube is decomposed into one slab (along x) per MPI rank, like
   a 'simple' OpenFOAM domain decomposition. Every rank owns
     - a brick of a structured-regular scalar field (a few Gaussian blobs),
       rendered as a transferFunction1D volume, and
     - the part of a triangulated sphere whose triangles have their
       centroid inside the slab (a stand-in for a local isosurface),
   renders only its own part with an ANARI device on a transparent
   background, and the parts are z/alpha-composited into one image on
   rank 0. Two paths:

     --mode direct : the application renders with any ANARI library
                     (--lib helide|cycles|barney_cpu|...), maps the
                     FLOAT32_VEC4 color + FLOAT32 depth channels, and
                     hands them to dc::Compositor (this is the path an
                     OpenFOAM function object rank would use);
     --mode ptc    : the application loads the 'ptc' ANARI library (the
                     deepCompositing wrapper device), which renders with
                     the wrapped library and composites inside
                     anariRenderFrame(); rank 0 maps the final image.

   Usage: mpirun -n 4 dcAnariDataParallel [--lib helide] [--mode direct|ptc]
            [--scene both|surface|volume] [--size 1024 768] [--spp 16]
            [--grid 128] [--depth auto|device|brick] [--dump-ranks] -o out.png
*/

#include <mpi.h>
#include "deepCompositing.h"
// anari
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/std.h>
// std
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "stb/stb_image_write.h"

namespace {

  using vec2u = std::array<uint32_t, 2>;
  using vec3 = std::array<float, 3>;
  using vec4 = std::array<float, 4>;
  using vec3u = std::array<uint32_t, 3>;

  int g_rank = 0;

  void statusFunc(const void *,
                  ANARIDevice,
                  ANARIObject,
                  ANARIDataType,
                  ANARIStatusSeverity severity,
                  ANARIStatusCode,
                  const char *message)
  {
    if (severity == ANARI_SEVERITY_FATAL_ERROR)
      fprintf(stderr, "[rank %i][FATAL] %s\n", g_rank, message);
    else if (severity == ANARI_SEVERITY_ERROR)
      fprintf(stderr, "[rank %i][ERROR] %s\n", g_rank, message);
    else if (severity == ANARI_SEVERITY_WARNING)
      fprintf(stderr, "[rank %i][WARN ] %s\n", g_rank, message);
  }

  /*! analytic test field: a few Gaussian blobs in the unit cube */
  float blobField(float x, float y, float z)
  {
    static const float blobs[][4] = {
      {0.25f, 0.30f, 0.30f, 0.12f},
      {0.70f, 0.65f, 0.35f, 0.15f},
      {0.45f, 0.75f, 0.70f, 0.10f},
      {0.80f, 0.25f, 0.75f, 0.12f},
      {0.15f, 0.70f, 0.80f, 0.09f},
    };
    float v = 0.f;
    for (const auto &b : blobs) {
      const float dx = x - b[0], dy = y - b[1], dz = z - b[2];
      v += std::exp(-(dx * dx + dy * dy + dz * dz) / (2.f * b[3] * b[3]));
    }
    return v;
  }

  /*! slab [x0,x1] of the unit cube owned by 'rank' */
  void slabOf(int rank, int size, float &x0, float &x1)
  {
    x0 = float(rank) / size;
    x1 = float(rank + 1) / size;
  }

  vec3 rankColor(int rank)
  {
    static const vec3 colors[] = {{0.90f, 0.35f, 0.25f},
                                  {0.30f, 0.70f, 0.35f},
                                  {0.30f, 0.45f, 0.90f},
                                  {0.90f, 0.80f, 0.25f},
                                  {0.70f, 0.35f, 0.85f},
                                  {0.25f, 0.80f, 0.85f},
                                  {0.95f, 0.55f, 0.15f},
                                  {0.60f, 0.60f, 0.60f}};
    return colors[rank % 8];
  }

  /*! the triangles of a UV sphere whose centroid lies in [x0,x1) */
  void localSphereMesh(float x0, float x1, bool lastSlab,
                       std::vector<vec3> &positions,
                       std::vector<vec3> &normals,
                       std::vector<vec3u> &indices)
  {
    const vec3 c = {0.5f, 0.5f, 0.5f};
    const float r = 0.22f;
    const int nTheta = 96, nPhi = 192;
    std::vector<vec3> P, N;
    for (int i = 0; i <= nTheta; i++)
      for (int j = 0; j <= nPhi; j++) {
        const float th = float(M_PI) * i / nTheta;
        const float ph = 2.f * float(M_PI) * j / nPhi;
        const vec3 n = {std::sin(th) * std::cos(ph), std::cos(th),
                        std::sin(th) * std::sin(ph)};
        N.push_back(n);
        P.push_back({c[0] + r * n[0], c[1] + r * n[1], c[2] + r * n[2]});
      }
    std::vector<int> remap(P.size(), -1);
    auto vid = [&](int i, int j) { return i * (nPhi + 1) + j; };
    auto use = [&](int v) {
      if (remap[v] < 0) {
        remap[v] = int(positions.size());
        positions.push_back(P[v]);
        normals.push_back(N[v]);
      }
      return uint32_t(remap[v]);
    };
    for (int i = 0; i < nTheta; i++)
      for (int j = 0; j < nPhi; j++) {
        const int q[4] = {vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)};
        const int tris[2][3] = {{q[0], q[1], q[2]}, {q[0], q[2], q[3]}};
        for (const auto &t : tris) {
          const float cx = (P[t[0]][0] + P[t[1]][0] + P[t[2]][0]) / 3.f;
          if (!(cx >= x0 && (cx < x1 || lastSlab)))
            continue;
          // counter-clockwise seen from outside (geometric normal agrees
          // with the outward vertex normals)
          const vec3 &a = P[t[0]], &b = P[t[1]], &e = P[t[2]];
          const vec3 u = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
          const vec3 v = {e[0] - a[0], e[1] - a[1], e[2] - a[2]};
          const vec3 ng = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                           u[0] * v[1] - u[1] * v[0]};
          const vec3 &n = N[t[0]];
          if (ng[0] * n[0] + ng[1] * n[1] + ng[2] * n[2] >= 0.f)
            indices.push_back({use(t[0]), use(t[1]), use(t[2])});
          else
            indices.push_back({use(t[0]), use(t[2]), use(t[1])});
        }
      }
  }

  struct Options
  {
    std::string lib = "helide";
    std::string mode = "direct";
    std::string scene = "both";
    std::string outFile = "dcAnariDataParallel.png";
    vec2u size = {1024, 768};
    int spp = 16;
    int frames = 1;
    int grid = 128;
    bool dumpRanks = false;
    /*! fragment depth in direct mode: "device" = the device's depth
        channel, "brick" = distance at which the camera ray enters this
        rank's (convex) slab, "auto" = brick if the scene has a volume */
    std::string depthMode = "auto";
  };

  /*! ANARI perspective camera, enough to regenerate its primary rays */
  struct PinholeCamera
  {
    vec3 pos, fwd, right, up; // normalized axes
    float tanHalfFovy, aspect;

    PinholeCamera(const vec3 &p, const vec3 &dir, const vec3 &upHint,
                  float fovy, float aspect_)
      : pos(p), tanHalfFovy(std::tan(0.5f * fovy)), aspect(aspect_)
    {
      fwd = normalize(dir);
      right = normalize(cross(fwd, upHint));
      up = cross(right, fwd);
    }

    static vec3 cross(const vec3 &a, const vec3 &b)
    {
      return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
              a[0] * b[1] - a[1] * b[0]};
    }
    static vec3 normalize(const vec3 &a)
    {
      const float l = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
      return {a[0] / l, a[1] / l, a[2] / l};
    }

    /*! normalized direction of the ray through pixel (x,y); y = 0 is the
        bottom row, like ANARI frame buffers */
    vec3 rayDir(int x, int y, int w, int h) const
    {
      const float sx = (2.f * (x + .5f) / w - 1.f) * tanHalfFovy * aspect;
      const float sy = (2.f * (y + .5f) / h - 1.f) * tanHalfFovy;
      return normalize({fwd[0] + sx * right[0] + sy * up[0],
                        fwd[1] + sx * right[1] + sy * up[1],
                        fwd[2] + sx * right[2] + sy * up[2]});
    }
  };

  /*! distance along 'dir' at which the ray enters box [lo,hi] (+inf: miss) */
  float boxEntry(const vec3 &org, const vec3 &dir, const vec3 &lo, const vec3 &hi)
  {
    float t0 = 0.f, t1 = std::numeric_limits<float>::infinity();
    for (int k = 0; k < 3; k++) {
      const float inv = 1.f / dir[k];
      float a = (lo[k] - org[k]) * inv, b = (hi[k] - org[k]) * inv;
      if (a > b) std::swap(a, b);
      t0 = std::max(t0, a);
      t1 = std::min(t1, b);
    }
    return t0 <= t1 ? t0 : std::numeric_limits<float>::infinity();
  }

  /*! build this rank's part of the scene into a new world */
  anari::World buildLocalWorld(anari::Device d, const Options &o, int rank, int size)
  {
    float x0, x1;
    slabOf(rank, size, x0, x1);
    auto world = anari::newObject<anari::World>(d);

    // --- local surface: part of the sphere -------------------------------
    if (o.scene == "both" || o.scene == "surface") {
      std::vector<vec3> positions, normals;
      std::vector<vec3u> indices;
      localSphereMesh(x0, x1, rank == size - 1, positions, normals, indices);
      if (!indices.empty()) {
        auto geom = anari::newObject<anari::Geometry>(d, "triangle");
        anari::setAndReleaseParameter(d, geom, "vertex.position",
            anari::newArray1D(d, positions.data(), positions.size()));
        anari::setAndReleaseParameter(d, geom, "vertex.normal",
            anari::newArray1D(d, normals.data(), normals.size()));
        anari::setAndReleaseParameter(d, geom, "primitive.index",
            anari::newArray1D(d, indices.data(), indices.size()));
        anari::commitParameters(d, geom);

        auto mat = anari::newObject<anari::Material>(d, "matte");
        anari::setParameter(d, mat, "color", rankColor(rank));
        anari::commitParameters(d, mat);

        auto surf = anari::newObject<anari::Surface>(d);
        anari::setAndReleaseParameter(d, surf, "geometry", geom);
        anari::setAndReleaseParameter(d, surf, "material", mat);
        anari::commitParameters(d, surf);
        anari::setAndReleaseParameter(
            d, world, "surface", anari::newArray1D(d, &surf));
        anari::release(d, surf);
      }
    }

    // --- local volume brick ---------------------------------------------
    if (o.scene == "both" || o.scene == "volume") {
      // global grid of o.grid^3 points on [0,1]^3; neighboring bricks share
      // their boundary point plane so trilinear interpolation is seamless
      const int G = o.grid;
      const float h = 1.f / (G - 1);
      const int i0 = (rank * (G - 1)) / size;
      const int i1 = ((rank + 1) * (G - 1)) / size;
      const int nx = i1 - i0 + 1;
      std::vector<float> voxels(size_t(nx) * G * G);
      for (int k = 0; k < G; k++)
        for (int j = 0; j < G; j++)
          for (int i = 0; i < nx; i++)
            voxels[(size_t(k) * G + j) * nx + i] =
                blobField((i0 + i) * h, j * h, k * h);

      auto field = anari::newObject<anari::SpatialField>(d, "structuredRegular");
      anari::setAndReleaseParameter(d, field, "data",
          anari::newArray3D(d, voxels.data(), nx, G, G));
      anari::setParameter(d, field, "origin", vec3{i0 * h, 0.f, 0.f});
      anari::setParameter(d, field, "spacing", vec3{h, h, h});
      anari::commitParameters(d, field);

      const std::vector<vec3> tfColor = {{0.23f, 0.30f, 0.75f},
                                         {0.55f, 0.69f, 0.99f},
                                         {0.87f, 0.87f, 0.87f},
                                         {0.96f, 0.60f, 0.48f},
                                         {0.71f, 0.02f, 0.15f}};
      const std::vector<float> tfOpacity = {0.f, 0.f, 0.25f, 0.6f, 0.9f};
      auto vol = anari::newObject<anari::Volume>(d, "transferFunction1D");
      anari::setAndReleaseParameter(d, vol, "value", field);
      const float valueRange[2] = {0.f, 1.f};
      anari::setParameter(d, vol, "valueRange", ANARI_FLOAT32_BOX1, valueRange);
      anari::setAndReleaseParameter(d, vol, "color",
          anari::newArray1D(d, tfColor.data(), tfColor.size()));
      anari::setAndReleaseParameter(d, vol, "opacity",
          anari::newArray1D(d, tfOpacity.data(), tfOpacity.size()));
      anari::setParameter(d, vol, "unitDistance", 0.08f);
      anari::commitParameters(d, vol);
      anari::setAndReleaseParameter(d, world, "volume", anari::newArray1D(d, &vol));
      anari::release(d, vol);
    }

    // --- light (identical on all ranks) -----------------------------------
    auto light = anari::newObject<anari::Light>(d, "directional");
    anari::setParameter(d, light, "direction", vec3{-0.4f, -1.f, -0.6f});
    anari::setParameter(d, light, "irradiance", 3.f);
    anari::commitParameters(d, light);
    anari::setAndReleaseParameter(d, world, "light", anari::newArray1D(d, &light));
    anari::release(d, light);

    anari::commitParameters(d, world);
    return world;
  }

  void writePNG(const std::string &fileName, const vec2u &size, const uint32_t *rgba)
  {
    stbi_flip_vertically_on_write(1);
    stbi_write_png(fileName.c_str(), int(size[0]), int(size[1]), 4, rgba,
                   4 * int(size[0]));
  }

  /*! blend 'bg' under premultiplied RGBA8 pixels (rank 0, after compositing) */
  void blendBackground(std::vector<uint32_t> &pixels, const vec3 &bg)
  {
    for (auto &p : pixels) {
      const float a = ((p >> 24) & 0xff) / 255.f;
      uint32_t out = 0xff000000u;
      for (int k = 0; k < 3; k++) {
        const float c = ((p >> (8 * k)) & 0xff) / 255.f + (1.f - a) * bg[k];
        out |= uint32_t(std::clamp(c, 0.f, 1.f) * 255.f + .5f) << (8 * k);
      }
      p = out;
    }
  }

  /*! linear float RGBA -> sRGB-encoded RGBA8 for display */
  uint32_t toSrgb8(uint32_t linear)
  {
    uint32_t out = linear & 0xff000000u;
    for (int k = 0; k < 3; k++) {
      const float l = ((linear >> (8 * k)) & 0xff) / 255.f;
      const float s = l <= 0.0031308f ? 12.92f * l : 1.055f * std::pow(l, 1.f / 2.4f) - 0.055f;
      out |= uint32_t(std::clamp(s, 0.f, 1.f) * 255.f + .5f) << (8 * k);
    }
    return out;
  }

} // namespace

int main(int ac, char **av)
{
  int provided = 0;
  MPI_Init_thread(&ac, &av, MPI_THREAD_FUNNELED, &provided);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  g_rank = rank;

  Options o;
  for (int i = 1; i < ac; i++) {
    const std::string arg = av[i];
    if (arg == "--lib") o.lib = av[++i];
    else if (arg == "--mode") o.mode = av[++i];
    else if (arg == "--scene") o.scene = av[++i];
    else if (arg == "-o") o.outFile = av[++i];
    else if (arg == "--size") { o.size[0] = std::stoi(av[++i]); o.size[1] = std::stoi(av[++i]); }
    else if (arg == "--spp") o.spp = std::stoi(av[++i]);
    else if (arg == "--frames") o.frames = std::stoi(av[++i]);
    else if (arg == "--grid") o.grid = std::stoi(av[++i]);
    else if (arg == "--dump-ranks") o.dumpRanks = true;
    else if (arg == "--depth") o.depthMode = av[++i];
    else {
      if (rank == 0) fprintf(stderr, "unknown argument '%s'\n", arg.c_str());
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  const bool ptc = (o.mode == "ptc");
  const vec3 bgColor = {0.12f, 0.12f, 0.15f};

  // --- device -------------------------------------------------------------
  anari::Library lib = anari::loadLibrary(ptc ? "ptc" : o.lib.c_str(), statusFunc);
  if (!lib) {
    fprintf(stderr, "[rank %i] could not load ANARI library '%s'\n", rank,
            ptc ? "ptc" : o.lib.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  anari::Device d = anari::newDevice(lib, "default");
  if (ptc) {
    anari::setParameter(d, d, "wrappedLibrary", o.lib.c_str());
    // VOID_POINTER parameter: 'mem' is the pointer value itself (here: the
    // address of the communicator, dereferenced at commit time)
    MPI_Comm comm = MPI_COMM_WORLD;
    anari::setParameter(d, d, "mpiCommunicator", ANARI_VOID_POINTER, &comm);
    anari::commitParameters(d, d);
  }

  auto t0 = std::chrono::steady_clock::now();
  anari::World world = buildLocalWorld(d, o, rank, size);

  // --- camera, renderer, frame ------------------------------------------------
  auto camera = anari::newObject<anari::Camera>(d, "perspective");
  const vec3 eye = {1.75f, 1.25f, 2.05f};
  const vec3 at = {0.5f, 0.45f, 0.5f};
  vec3 dir = {at[0] - eye[0], at[1] - eye[1], at[2] - eye[2]};
  const vec3 upHint = {0.f, 1.f, 0.f};
  const float fovy = float(M_PI / 4.5);
  const float aspect = float(o.size[0]) / o.size[1];
  anari::setParameter(d, camera, "aspect", aspect);
  anari::setParameter(d, camera, "position", eye);
  anari::setParameter(d, camera, "direction", dir);
  anari::setParameter(d, camera, "up", upHint);
  anari::setParameter(d, camera, "fovy", fovy);
  anari::commitParameters(d, camera);

  auto renderer = anari::newObject<anari::Renderer>(d, "default");
  // transparent background on every rank: required for correct compositing
  anari::setParameter(d, renderer, "background", vec4{0.f, 0.f, 0.f, 0.f});
  anari::setParameter(d, renderer, "ambientRadiance", 0.6f);
  anari::setParameter(d, renderer, "pixelSamples", o.spp);
  if (o.lib == "cycles") {
    // offline frames: no reduced-resolution first frame (CYCLES_RENDERER_
    // INTERACTIVE_SCALING), and wait for the full sample count
    anari::setParameter(d, renderer, "interactiveScaling", false);
    anari::setParameter(d, renderer, "runAsync", false);
  }
  anari::commitParameters(d, renderer);

  auto frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, frame, "size", o.size);
  anari::setParameter(d, frame, "channel.color",
                      ptc ? ANARI_UFIXED8_VEC4 : ANARI_FLOAT32_VEC4);
  anari::setParameter(d, frame, "channel.depth", ANARI_FLOAT32);
  if (ptc)
    anari::setParameter(d, frame, "ptc.background", vec4{bgColor[0], bgColor[1], bgColor[2], 1.f});
  anari::setAndReleaseParameter(d, frame, "renderer", renderer);
  anari::setAndReleaseParameter(d, frame, "camera", camera);
  anari::setAndReleaseParameter(d, frame, "world", world);
  anari::commitParameters(d, frame);
  auto t1 = std::chrono::steady_clock::now();

  const size_t numPixels = size_t(o.size[0]) * o.size[1];
  std::vector<uint32_t> finalImage(rank == 0 ? numPixels : 0);
  double tRender = 0., tComposite = 0.;

  if (ptc) {
    // render (+ composite, inside the wrapper) ---------------------------------
    for (int f = 0; f < o.frames; f++) {
      anari::render(d, frame);
      anari::wait(d, frame);
    }
    auto t2 = std::chrono::steady_clock::now();
    tRender = std::chrono::duration<double>(t2 - t1).count();
    if (rank == 0) {
      auto fb = anari::map<uint32_t>(d, frame, "channel.color");
      std::copy(fb.data, fb.data + numPixels, finalImage.begin());
      anari::unmap(d, frame, "channel.color");
    }
  } else {
    // render locally ----------------------------------------------------------
    for (int f = 0; f < o.frames; f++) {
      anari::render(d, frame);
      anari::wait(d, frame);
    }
    auto t2 = std::chrono::steady_clock::now();
    tRender = std::chrono::duration<double>(t2 - t1).count();

    // hand color + depth to the deep compositor --------------------------------
    dc::Compositor compositor(1, MPI_COMM_WORLD);
    compositor.resize(o.size[0], o.size[1]);
    dc::DeviceInterface di = compositor.prepare();

    auto color = anari::map<vec4>(d, frame, "channel.color");
    auto depth = anari::map<float>(d, frame, "channel.depth");
    if (!color.data || !depth.data) {
      fprintf(stderr, "[rank %i] device did not provide color/depth\n", rank);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // fragment depth: the device's depth channel only works for opaque
    // surfaces (e.g. cycles writes 0 for volume-only pixels); with volumes,
    // use the entry distance of each pixel's ray into this rank's convex
    // slab, which orders disjoint bricks correctly for every ray
    const bool brickDepth = o.depthMode == "brick"
        || (o.depthMode == "auto" && o.scene != "surface");
    std::vector<float> fragDepth(depth.data, depth.data + numPixels);
    if (brickDepth) {
      float x0, x1;
      slabOf(rank, size, x0, x1);
      const PinholeCamera cam(eye, dir, upHint, fovy, aspect);
#pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < int64_t(numPixels); i++) {
        const int x = int(i % o.size[0]), y = int(i / o.size[0]);
        fragDepth[i] = boxEntry(eye, cam.rayDir(x, y, o.size[0], o.size[1]),
                                {x0, 0.f, 0.f}, {x1, 1.f, 1.f});
      }
    }
    dc::writeImageFragments(di, (const float *)color.data, fragDepth.data());
    if (o.dumpRanks) {
      std::vector<uint32_t> local(numPixels);
      const float *c = (const float *)color.data;
      for (size_t i = 0; i < numPixels; i++) {
        float a = c[4 * i + 3];
        uint32_t p = 0xff000000u;
        for (int k = 0; k < 3; k++) {
          const float v = c[4 * i + k] + (1.f - a) * bgColor[k];
          p |= uint32_t(std::clamp(v, 0.f, 1.f) * 255.f + .5f) << (8 * k);
        }
        local[i] = toSrgb8(p);
      }
      std::string base = o.outFile.substr(0, o.outFile.rfind('.'));
      writePNG(base + "_rank" + std::to_string(rank) + ".png", o.size, local.data());
    }
    anari::unmap(d, frame, "channel.color");
    anari::unmap(d, frame, "channel.depth");

    compositor.finish(rank == 0 ? finalImage.data() : nullptr);
    auto t3 = std::chrono::steady_clock::now();
    tComposite = std::chrono::duration<double>(t3 - t2).count();
    if (rank == 0)
      blendBackground(finalImage, bgColor);
  }

  double tSetup = std::chrono::duration<double>(t1 - t0).count();
  double times[3] = {tSetup, tRender, tComposite}, maxTimes[3];
  MPI_Reduce(times, maxTimes, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (auto &p : finalImage)
      p = toSrgb8(p);
    writePNG(o.outFile, o.size, finalImage.data());
    printf("dcAnariDataParallel: %i ranks, lib=%s mode=%s scene=%s %ux%u spp=%i: "
           "setup %.2fs render %.2fs composite %.3fs (max over ranks) -> %s\n",
           size, o.lib.c_str(), (o.mode + (ptc ? "" : "/depth=" + o.depthMode)).c_str(), o.scene.c_str(), o.size[0],
           o.size[1], o.spp, maxTimes[0], maxTimes[1], maxTimes[2],
           o.outFile.c_str());
  }

  anari::release(d, frame);
  anari::release(d, d);
  anari::unloadLibrary(lib);
  MPI_Finalize();
  return 0;
}
