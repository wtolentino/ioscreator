// staged_copy.cu — transferência GPU→GPU entre sockets sem P2P cross-QPI.
//
// O caminho P2P direto entre root complexes de sockets diferentes não é
// encaminhado com concorrência pelo QPI (~4-5 requisições em voo, ~0,3 GB/s).
// Este código evita esse caminho: cada hop PCIe fica dentro do seu próprio
// socket, e a travessia entre sockets vira um memcpy de CPU — que o QPI
// encaminha bem, em streaming.
//
//   GPU_src --D2H--> pinned[nó A]  --memcpy NT--> pinned[nó B] --H2D--> GPU_dst
//            local                    QPI                       local
//
// Os três estágios rodam em pipeline sobre um anel de slots.

#include <cuda_runtime.h>
#include <numa.h>
#include <pthread.h>
#include <immintrin.h>

#include <atomic>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <chrono>

// ---------------------------------------------------------------- parâmetros
static size_t CHUNK  = 8ull << 20;  // 8 MB por slot
static int    NSLOTS = 8;
static int    NCOPY  = 6;           // threads de memcpy

#define CUDA_CHECK(x) do {                                                    \
    cudaError_t _e = (x);                                                     \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,           \
                cudaGetErrorString(_e));                                      \
        exit(1);                                                              \
    }                                                                         \
} while (0)

// ------------------------------------------------------------------- helpers

// O nó NUMA de uma GPU sai do sysfs. cudaDeviceGetPCIBusId devolve o BDF em
// maiúsculas; o sysfs usa minúsculas.
static int gpu_numa_node(int dev)
{
    char bdf[32];
    CUDA_CHECK(cudaDeviceGetPCIBusId(bdf, sizeof(bdf), dev));
    for (char *p = bdf; *p; ++p) *p = (char)tolower((unsigned char)*p);

    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", bdf);

    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int node = -1;
    if (fscanf(f, "%d", &node) != 1) node = -1;
    fclose(f);
    return node < 0 ? 0 : node;   // -1 = plataforma sem NUMA exposto
}

static void pin_thread_to_node(int node)
{
    struct bitmask *bm = numa_allocate_cpumask();
    numa_node_to_cpus(node, bm);

    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < numa_num_configured_cpus(); ++i)
        if (numa_bitmask_isbitset(bm, i)) CPU_SET(i, &set);

    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    numa_free_cpumask(bm);
}

// Buffer pinned num nó NUMA específico. numa_alloc_onnode já devolve alinhado
// a página, que é o que cudaHostRegister exige.
static void *alloc_pinned_on_node(size_t bytes, int node)
{
    void *p = numa_alloc_onnode(bytes, node);
    if (!p) { fprintf(stderr, "numa_alloc_onnode(%zu, %d) falhou\n", bytes, node); exit(1); }
    memset(p, 0, bytes);   // first-touch pela thread chamadora, já no nó certo
    CUDA_CHECK(cudaHostRegister(p, bytes, cudaHostRegisterPortable));
    return p;
}

static void free_pinned_on_node(void *p, size_t bytes)
{
    cudaHostUnregister(p);
    numa_free(p, bytes);
}

// memcpy com non-temporal stores: escreve sem RFO e sem poluir a LLC.
// As threads ficam no nó de ORIGEM — leitura local, escrita remota. Store NT
// é fire-and-forget; read remoto pagaria round-trip completo pelo QPI.
static inline void nt_memcpy(void *__restrict dst, const void *__restrict src, size_t n)
{
    char       *d = (char *)dst;
    const char *s = (const char *)src;
    size_t i = 0;

    for (; i + 128 <= n; i += 128) {
        __m256i a = _mm256_stream_load_si256((const __m256i *)(s + i));
        __m256i b = _mm256_stream_load_si256((const __m256i *)(s + i + 32));
        __m256i c = _mm256_stream_load_si256((const __m256i *)(s + i + 64));
        __m256i e = _mm256_stream_load_si256((const __m256i *)(s + i + 96));
        _mm256_stream_si256((__m256i *)(d + i),      a);
        _mm256_stream_si256((__m256i *)(d + i + 32), b);
        _mm256_stream_si256((__m256i *)(d + i + 64), c);
        _mm256_stream_si256((__m256i *)(d + i + 96), e);
    }
    if (i < n) memcpy(d + i, s + i, n - i);
    _mm_sfence();
}

// -------------------------------------------------------------------- slots

enum SlotState {
    EMPTY = 0,      // livre
    D2H_ISSUED,     // D2H submetido, evento pendente
    COPYING,        // um copier pegou o slot
    COPIED,         // pronto no nó B
    H2D_ISSUED      // H2D submetido, evento pendente
};

struct Slot {
    void       *hA = nullptr;       // pinned no nó da GPU origem
    void       *hB = nullptr;       // pinned no nó da GPU destino
    size_t      offset = 0;
    size_t      bytes  = 0;
    cudaEvent_t evD2H{};
    cudaEvent_t evH2D{};
    std::atomic<int> state{EMPTY};
};

struct Pipeline {
    int    devSrc, devDst;
    int    nodeSrc, nodeDst;
    std::vector<Slot> slots;
    cudaStream_t streamD2H{}, streamH2D{};

    std::atomic<size_t> chunksDone{0};
    std::atomic<bool>   stop{false};
    size_t totalChunks = 0;
};

// ------------------------------------------------------------------ estágios

static void producer(Pipeline *P, const void *src, size_t bytes)
{
    pin_thread_to_node(P->nodeSrc);
    CUDA_CHECK(cudaSetDevice(P->devSrc));

    size_t off = 0;
    int    idx = 0;

    while (off < bytes) {
        Slot &s = P->slots[idx];

        // espera o slot liberar
        while (s.state.load(std::memory_order_acquire) != EMPTY) {
            if (P->stop.load(std::memory_order_relaxed)) return;
            _mm_pause();
        }

        size_t n = (bytes - off < CHUNK) ? (bytes - off) : CHUNK;
        s.offset = off;
        s.bytes  = n;

        // Um stream só para todos os D2H: a copy engine é serial de qualquer
        // forma, e isso preserva a ordem de emissão.
        CUDA_CHECK(cudaMemcpyAsync(s.hA, (const char *)src + off, n,
                                   cudaMemcpyDeviceToHost, P->streamD2H));
        CUDA_CHECK(cudaEventRecord(s.evD2H, P->streamD2H));

        s.state.store(D2H_ISSUED, std::memory_order_release);

        off += n;
        idx = (idx + 1) % NSLOTS;
    }
}

static void copier(Pipeline *P)
{
    // Nó de origem: leitura local, escrita remota com NT stores.
    pin_thread_to_node(P->nodeSrc);
    CUDA_CHECK(cudaSetDevice(P->devSrc));

    int idx = 0;
    while (!P->stop.load(std::memory_order_relaxed)) {
        Slot &s = P->slots[idx];
        idx = (idx + 1) % NSLOTS;

        int expected = D2H_ISSUED;
        if (!s.state.compare_exchange_strong(expected, COPYING,
                                             std::memory_order_acq_rel)) {
            _mm_pause();
            continue;
        }

        CUDA_CHECK(cudaEventSynchronize(s.evD2H));
        nt_memcpy(s.hB, s.hA, s.bytes);

        s.state.store(COPIED, std::memory_order_release);
    }
}

static void consumer(Pipeline *P, void *dst)
{
    pin_thread_to_node(P->nodeDst);
    CUDA_CHECK(cudaSetDevice(P->devDst));

    while (P->chunksDone.load(std::memory_order_relaxed) < P->totalChunks) {
        if (P->stop.load(std::memory_order_relaxed)) return;

        bool progressed = false;

        for (int i = 0; i < NSLOTS; ++i) {
            Slot &s = P->slots[i];
            int st = s.state.load(std::memory_order_acquire);

            if (st == COPIED) {
                CUDA_CHECK(cudaMemcpyAsync((char *)dst + s.offset, s.hB, s.bytes,
                                           cudaMemcpyHostToDevice, P->streamH2D));
                CUDA_CHECK(cudaEventRecord(s.evH2D, P->streamH2D));
                s.state.store(H2D_ISSUED, std::memory_order_release);
                progressed = true;

            } else if (st == H2D_ISSUED) {
                // colheita não-bloqueante: libera o slot assim que o H2D fecha
                cudaError_t q = cudaEventQuery(s.evH2D);
                if (q == cudaSuccess) {
                    s.state.store(EMPTY, std::memory_order_release);
                    P->chunksDone.fetch_add(1, std::memory_order_relaxed);
                    progressed = true;
                } else if (q != cudaErrorNotReady) {
                    CUDA_CHECK(q);
                }
            }
        }
        if (!progressed) _mm_pause();
    }
}

// --------------------------------------------------------------- API pública

// Copia `bytes` de dsrc (na GPU devSrc) para ddst (na GPU devDst) pelo
// caminho de 3 hops. Aceita qualquer tamanho; o chunking é interno.
static double staged_copy(int devSrc, int devDst,
                          const void *dsrc, void *ddst, size_t bytes)
{
    Pipeline P;
    P.devSrc  = devSrc;
    P.devDst  = devDst;
    P.nodeSrc = gpu_numa_node(devSrc);
    P.nodeDst = gpu_numa_node(devDst);
    P.totalChunks = (bytes + CHUNK - 1) / CHUNK;
    P.slots.resize(NSLOTS);

    CUDA_CHECK(cudaSetDevice(devSrc));
    CUDA_CHECK(cudaStreamCreate(&P.streamD2H));
    CUDA_CHECK(cudaSetDevice(devDst));
    CUDA_CHECK(cudaStreamCreate(&P.streamH2D));

    for (int i = 0; i < NSLOTS; ++i) {
        P.slots[i].hA = alloc_pinned_on_node(CHUNK, P.nodeSrc);
        P.slots[i].hB = alloc_pinned_on_node(CHUNK, P.nodeDst);
    }

    // Um evento fica preso ao device corrente na criação. evD2H é gravado num
    // stream de devSrc e evH2D num de devDst — cada um precisa nascer sob o
    // seu device, ou cudaEventRecord devolve InvalidResourceHandle.
    CUDA_CHECK(cudaSetDevice(devSrc));
    for (int i = 0; i < NSLOTS; ++i)
        CUDA_CHECK(cudaEventCreateWithFlags(&P.slots[i].evD2H, cudaEventDisableTiming));

    CUDA_CHECK(cudaSetDevice(devDst));
    for (int i = 0; i < NSLOTS; ++i)
        CUDA_CHECK(cudaEventCreateWithFlags(&P.slots[i].evH2D, cudaEventDisableTiming));

    auto t0 = std::chrono::steady_clock::now();

    std::thread tp(producer, &P, dsrc, bytes);
    std::vector<std::thread> tc;
    for (int i = 0; i < NCOPY; ++i) tc.emplace_back(copier, &P);
    std::thread tk(consumer, &P, ddst);

    tp.join();
    tk.join();
    P.stop.store(true, std::memory_order_relaxed);
    for (auto &t : tc) t.join();

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    cudaSetDevice(devSrc);
    for (int i = 0; i < NSLOTS; ++i) cudaEventDestroy(P.slots[i].evD2H);
    cudaSetDevice(devDst);
    for (int i = 0; i < NSLOTS; ++i) cudaEventDestroy(P.slots[i].evH2D);

    for (int i = 0; i < NSLOTS; ++i) {
        free_pinned_on_node(P.slots[i].hA, CHUNK);
        free_pinned_on_node(P.slots[i].hB, CHUNK);
    }
    cudaStreamDestroy(P.streamD2H);
    cudaStreamDestroy(P.streamH2D);

    return bytes / secs / 1e9;   // GB/s
}

// ------------------------------------------------------------- baselines

// P2P direto. Com `push`, o contexto ativo é o da ORIGEM: as transações viram
// writes (posted). Com pull, viram reads (non-posted) e pagam round-trip.
static double p2p_copy(int devSrc, int devDst,
                       const void *dsrc, void *ddst, size_t bytes, bool push)
{
    int actor = push ? devSrc : devDst;
    CUDA_CHECK(cudaSetDevice(actor));

    int peer = push ? devDst : devSrc;
    cudaError_t e = cudaDeviceEnablePeerAccess(peer, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
        fprintf(stderr, "  (peer access indisponível: %s)\n", cudaGetErrorString(e));
        return -1.0;
    }

    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreate(&st));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyPeerAsync(ddst, devDst, dsrc, devSrc, bytes, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    auto t1 = std::chrono::steady_clock::now();

    cudaStreamDestroy(st);
    return bytes / std::chrono::duration<double>(t1 - t0).count() / 1e9;
}

// Staging ingênuo: um buffer só, D2H e H2D serializados. É o padrão que
// tipicamente rende ~5 GB/s — está aqui para comparação.
static double naive_staged_copy(int devSrc, int devDst,
                                const void *dsrc, void *ddst, size_t bytes)
{
    void *h = nullptr;
    CUDA_CHECK(cudaSetDevice(devSrc));
    CUDA_CHECK(cudaHostAlloc(&h, CHUNK, cudaHostAllocPortable));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::steady_clock::now();
    for (size_t off = 0; off < bytes; off += CHUNK) {
        size_t n = (bytes - off < CHUNK) ? (bytes - off) : CHUNK;
        CUDA_CHECK(cudaSetDevice(devSrc));
        CUDA_CHECK(cudaMemcpy(h, (const char *)dsrc + off, n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaSetDevice(devDst));
        CUDA_CHECK(cudaMemcpy((char *)ddst + off, h, n, cudaMemcpyHostToDevice));
    }
    auto t1 = std::chrono::steady_clock::now();

    cudaFreeHost(h);
    return bytes / std::chrono::duration<double>(t1 - t0).count() / 1e9;
}

// ------------------------------------------------------------------ main

int main(int argc, char **argv)
{
    int    devSrc = (argc > 1) ? atoi(argv[1]) : 0;
    int    devDst = (argc > 2) ? atoi(argv[2]) : 2;
    size_t MB     = (argc > 3) ? (size_t)atoll(argv[3]) : 1024;
    if (argc > 4) CHUNK  = (size_t)atoll(argv[4]) << 20;
    if (argc > 5) NSLOTS = atoi(argv[5]);
    if (argc > 6) NCOPY  = atoi(argv[6]);

    if (numa_available() < 0) { fprintf(stderr, "libnuma indisponível\n"); return 1; }

    size_t bytes = MB << 20;

    printf("GPU %d (nó NUMA %d)  ->  GPU %d (nó NUMA %d)\n",
           devSrc, gpu_numa_node(devSrc), devDst, gpu_numa_node(devDst));
    printf("%zu MB, chunk %zu MB, %d slots, %d threads de cópia\n\n",
           MB, CHUNK >> 20, NSLOTS, NCOPY);

    void *dsrc = nullptr, *ddst = nullptr;
    CUDA_CHECK(cudaSetDevice(devSrc)); CUDA_CHECK(cudaMalloc(&dsrc, bytes));
    CUDA_CHECK(cudaMemset(dsrc, 0xA5, bytes));
    CUDA_CHECK(cudaSetDevice(devDst)); CUDA_CHECK(cudaMalloc(&ddst, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    struct { const char *name; double gbs; } r[4];

    r[0] = { "P2P direto (push)", p2p_copy(devSrc, devDst, dsrc, ddst, bytes, true)  };
    r[1] = { "P2P direto (pull)", p2p_copy(devSrc, devDst, dsrc, ddst, bytes, false) };
    r[2] = { "staging 1 buffer",  naive_staged_copy(devSrc, devDst, dsrc, ddst, bytes) };
    r[3] = { "staging 3-hop",     staged_copy(devSrc, devDst, dsrc, ddst, bytes) };

    printf("%-22s %10s\n", "caminho", "GB/s");
    printf("%-22s %10s\n", "----------------------", "----------");
    for (auto &x : r) {
        if (x.gbs < 0) printf("%-22s %10s\n", x.name, "n/d");
        else           printf("%-22s %10.2f\n", x.name, x.gbs);
    }

    cudaSetDevice(devSrc); cudaFree(dsrc);
    cudaSetDevice(devDst); cudaFree(ddst);
    return 0;
}
