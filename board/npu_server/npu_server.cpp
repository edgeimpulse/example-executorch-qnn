// npu_server: keeps a QNN (Hexagon HTP) ExecuTorch program loaded and serves
// inference over a Unix domain socket, so a Python app can use the NPU at
// camera frame rate without reloading the model.
//
// Protocol (little-endian, one request/response at a time per connection):
//   request : u32 n_bytes, then n_bytes of float32 NCHW input
//             (n_bytes == 0xFFFFFFFF asks for INFO instead of inference)
//   response: u32 status (0 ok, 1 bad input size, 2 inference failed),
//             u32 n_floats, u32 infer_us, then n_floats float32 logits
//   INFO    : u32 json_len, then json_len bytes of JSON
//
// Usage: npu_server --model model_qnn.pte [--socket /run/rps-npu/npu.sock]
//                   [--selftest] [--warmup 10]
//   --selftest : load, run 20 random inferences, print latency, exit 0/1
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

using executorch::extension::Module;
using executorch::extension::make_tensor_ptr;
using executorch::runtime::Error;

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop = true; }

static constexpr uint32_t kInfoRequest = 0xFFFFFFFFu;

static bool read_full(int fd, void* buf, size_t n) {
  auto* p = static_cast<uint8_t*>(buf);
  while (n > 0) {
    ssize_t r = ::read(fd, p, n);
    if (r == 0) return false;  // peer closed
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

static bool write_full(int fd, const void* buf, size_t n) {
  auto* p = static_cast<const uint8_t*>(buf);
  while (n > 0) {
    ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
    if (w < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += w;
    n -= static_cast<size_t>(w);
  }
  return true;
}

struct Engine {
  std::unique_ptr<Module> module;
  std::vector<int32_t> in_shape;
  size_t in_numel = 1;
  size_t out_numel = 0;
  std::string model_path;

  bool load(const std::string& path) {
    model_path = path;
    module = std::make_unique<Module>(path);
    Error err = module->load_forward();
    if (err != Error::Ok) {
      std::fprintf(stderr, "ERROR: load_forward failed (error 0x%x). "
                           "Is the QNN backend registered and the DSP reachable?\n",
                   static_cast<unsigned>(err));
      return false;
    }
    auto meta = module->method_meta("forward");
    if (!meta.ok()) {
      std::fprintf(stderr, "ERROR: method_meta(forward) failed\n");
      return false;
    }
    auto info = meta->input_tensor_meta(0);
    if (!info.ok()) {
      std::fprintf(stderr, "ERROR: input_tensor_meta(0) failed\n");
      return false;
    }
    in_shape.clear();
    in_numel = 1;
    for (auto s : info->sizes()) {
      in_shape.push_back(static_cast<int32_t>(s));
      in_numel *= static_cast<size_t>(s);
    }
    return true;
  }

  // Runs one inference. Returns false on failure. logits is resized as needed.
  bool infer(std::vector<float>&& input, std::vector<float>& logits, uint32_t& us) {
    auto t0 = std::chrono::steady_clock::now();
    auto tensor = make_tensor_ptr(in_shape, std::move(input));
    auto result = module->forward(tensor);
    auto t1 = std::chrono::steady_clock::now();
    us = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    if (!result.ok()) return false;
    const auto& out = result.get();
    if (out.empty() || !out[0].isTensor()) return false;
    const auto& t = out[0].toTensor();
    const float* d = t.const_data_ptr<float>();
    size_t n = static_cast<size_t>(t.numel());
    logits.assign(d, d + n);
    out_numel = n;
    return true;
  }

  std::string info_json() const {
    std::string s = "{\"backend\":\"qnn-htp\",\"model\":\"" + model_path +
                    "\",\"input_dtype\":\"float32\",\"input_layout\":\"NCHW\",\"input_shape\":[";
    for (size_t i = 0; i < in_shape.size(); ++i) {
      if (i) s += ",";
      s += std::to_string(in_shape[i]);
    }
    s += "],\"num_outputs\":" + std::to_string(out_numel) + "}";
    return s;
  }
};

static std::vector<float> random_input(size_t n, std::mt19937& rng) {
  std::uniform_real_distribution<float> d(0.f, 1.f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

static int run_selftest(Engine& e, int warmup) {
  std::mt19937 rng(1);
  std::vector<float> logits;
  uint32_t us = 0;
  for (int i = 0; i < warmup; ++i) {
    if (!e.infer(random_input(e.in_numel, rng), logits, us)) {
      std::fprintf(stderr, "selftest: warm-up inference %d failed\n", i);
      return 1;
    }
  }
  std::vector<uint32_t> times;
  for (int i = 0; i < 20; ++i) {
    if (!e.infer(random_input(e.in_numel, rng), logits, us)) {
      std::fprintf(stderr, "selftest: inference %d failed\n", i);
      return 1;
    }
    times.push_back(us);
    for (float v : logits)
      if (!std::isfinite(v)) {
        std::fprintf(stderr, "selftest: non-finite output\n");
        return 1;
      }
  }
  std::sort(times.begin(), times.end());
  std::printf("selftest OK: %zu outputs, median %u us, p95 %u us\n", logits.size(),
              times[times.size() / 2], times[times.size() * 95 / 100]);
  return 0;
}

static int serve(Engine& e, const std::string& sock_path) {
  int srv = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (srv < 0) { std::perror("socket"); return 1; }
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (sock_path.size() >= sizeof(addr.sun_path)) {
    std::fprintf(stderr, "socket path too long\n");
    return 1;
  }
  std::strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);
  ::unlink(sock_path.c_str());
  if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    std::perror("bind");
    return 1;
  }
  ::chmod(sock_path.c_str(), 0660);
  if (::listen(srv, 4) != 0) { std::perror("listen"); return 1; }
  std::fprintf(stderr, "npu_server: listening on %s (input %zu floats)\n",
               sock_path.c_str(), e.in_numel);

  uint64_t frames = 0, total_us = 0;
  while (!g_stop) {
    int c = ::accept(srv, nullptr, nullptr);
    if (c < 0) {
      if (errno == EINTR) continue;
      std::perror("accept");
      break;
    }
    for (;;) {
      uint32_t n_bytes = 0;
      if (!read_full(c, &n_bytes, 4)) break;
      if (n_bytes == kInfoRequest) {
        std::string j = e.info_json();
        uint32_t len = static_cast<uint32_t>(j.size());
        if (!write_full(c, &len, 4) || !write_full(c, j.data(), j.size())) break;
        continue;
      }
      const size_t want = e.in_numel * sizeof(float);
      if (n_bytes != want) {
        // drain what the client sent, then report a size error
        std::vector<uint8_t> junk(std::min<size_t>(n_bytes, 1u << 24));
        size_t left = n_bytes;
        while (left > 0) {
          size_t k = std::min(left, junk.size());
          if (!read_full(c, junk.data(), k)) { left = 0; break; }
          left -= k;
        }
        uint32_t hdr[3] = {1, 0, 0};
        if (!write_full(c, hdr, sizeof(hdr))) break;
        continue;
      }
      std::vector<float> in(e.in_numel);
      if (!read_full(c, in.data(), want)) break;
      std::vector<float> logits;
      uint32_t us = 0;
      bool ok = e.infer(std::move(in), logits, us);
      uint32_t hdr[3] = {ok ? 0u : 2u, ok ? static_cast<uint32_t>(logits.size()) : 0u, us};
      if (!write_full(c, hdr, sizeof(hdr))) break;
      if (ok && !write_full(c, logits.data(), logits.size() * sizeof(float))) break;
      if (ok) {
        ++frames;
        total_us += us;
        if (frames % 500 == 0)
          std::fprintf(stderr, "npu_server: %llu frames, mean %.1f us\n",
                       static_cast<unsigned long long>(frames),
                       static_cast<double>(total_us) / static_cast<double>(frames));
      }
    }
    ::close(c);
  }
  ::close(srv);
  ::unlink(sock_path.c_str());
  return 0;
}

int main(int argc, char** argv) {
  std::string model, sock = "/run/rps-npu/npu.sock";
  bool selftest = false;
  int warmup = 10;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model" && i + 1 < argc) model = argv[++i];
    else if (a == "--socket" && i + 1 < argc) sock = argv[++i];
    else if (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (a == "--selftest") selftest = true;
    else {
      std::fprintf(stderr, "usage: %s --model m.pte [--socket path] [--warmup N] [--selftest]\n", argv[0]);
      return 2;
    }
  }
  if (model.empty()) { std::fprintf(stderr, "--model is required\n"); return 2; }

  struct sigaction sa{};
  sa.sa_handler = on_signal;  // no SA_RESTART: accept() must return on SIGTERM
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
  std::signal(SIGPIPE, SIG_IGN);

  Engine e;
  if (!e.load(model)) return 1;

  // Warm up (also discovers the output size for INFO).
  std::mt19937 rng(7);
  std::vector<float> logits;
  uint32_t us = 0;
  for (int i = 0; i < std::max(1, warmup); ++i) {
    if (!e.infer(random_input(e.in_numel, rng), logits, us)) {
      std::fprintf(stderr, "ERROR: warm-up inference failed\n");
      return 1;
    }
  }
  std::fprintf(stderr, "npu_server: model loaded, warm-up last %u us, %zu outputs\n", us,
               e.out_numel);

  if (selftest) return run_selftest(e, warmup);
  return serve(e, sock);
}
