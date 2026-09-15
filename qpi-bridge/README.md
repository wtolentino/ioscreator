# staged_copy — GPU↔GPU entre sockets sem P2P cross-QPI

## O problema

Num servidor dual-socket com QPI (Xeon v1–v4), o P2P PCIe entre root complexes
de sockets diferentes não é encaminhado com concorrência. O sintoma medido:

| Caminho | Medido |
|---|---|
| GPU↔GPU mesma ilha (NVLink) | ~112 GB/s |
| GPU↔GPU cross-socket, P2P direto | **0,28 GB/s** |
| GPU↔GPU cross-socket, via RAM (1 buffer) | ~5 GB/s |

0,28 GB/s com transações de 128 B e ~2 µs de round-trip dá ~4-5 requisições em
voo. Um caminho PCIe saudável mantém centenas. Não é falta de banda — o QPI
está a <1% de ocupação. É limitação estrutural: a Intel não suporta P2P PCIe
entre root complexes de sockets diferentes.

**Não tem conserto por driver ou BIOS.** O patch do geohot faz o P2P existir;
não faz o QPI encaminhá-lo bem.

## A solução

Evitar o caminho. Cada hop PCIe fica dentro do seu próprio socket, e a
travessia entre sockets vira um `memcpy` de CPU — que o QPI encaminha bem, em
streaming, com prefetcher e sem round-trips.

```
GPU_src --D2H--> pinned[nó A] --memcpy NT--> pinned[nó B] --H2D--> GPU_dst
         local                    QPI                      local
        ~13 GB/s                ~15 GB/s                  ~13 GB/s
```

Os três estágios rodam em pipeline sobre um anel de slots, então o teto é o
estágio mais lento: **~12–13 GB/s** numa plataforma Gen3.

Detalhes que importam:

- Buffers alocados com `numa_alloc_onnode` no nó certo, depois `cudaHostRegister`
- Threads de cópia fixadas no nó de **origem**: leitura local, escrita remota.
  NT store é fire-and-forget; um read remoto pagaria round-trip pelo QPI —
  mesma lógica de posted vs. non-posted do PCIe
- `_mm256_stream_si256` evita RFO e não polui a LLC
- Nó NUMA de cada GPU lido de `/sys/bus/pci/devices/<bdf>/numa_node`

## Build

```bash
sudo apt install libnuma-dev          # ou: dnf install numactl-devel
make                                   # ajuste ARCH se não for sm_86
```

## Uso

```bash
./staged_copy [devSrc] [devDst] [MB] [chunkMB] [nslots] [ncopy]
./staged_copy 0 2 4096                 # 4 GB da GPU0 para a GPU2
./staged_copy 0 2 4096 16 12 8         # chunk 16 MB, 12 slots, 8 threads
```

Saída: os quatro caminhos medidos lado a lado.

```
caminho                      GB/s
----------------------  ----------
P2P direto (push)             0.28
P2P direto (pull)             0.19
staging 1 buffer              5.10
staging 3-hop                12.80
```

## Tuning

- **chunk**: comece em 8 MB. Menor aumenta overhead por transação; maior
  aumenta a latência de preenchimento do pipeline.
- **ncopy**: uma thread sozinha faz ~5–8 GB/s pelo QPI. 6–8 threads saturam.
- **nslots**: ≥ `ncopy + 2`, para os copiers nunca ficarem sem trabalho.

Antes de medir:

```bash
cpupower frequency-set -g performance
# BIOS: uncore frequency fixa no máximo, package C-state em C0/C1,
#       QPI link power management desabilitado
```

## Teto da plataforma

~12–13 GB/s é o limite de um slot Gen3 x16 — e Xeon v4 (QPI) só tem Gen3.
Um switch PCIe Gen3 e uma ponte FPGA customizada batem no mesmo teto.

Só um **switch PCIe Gen4** com todas as GPUs atrás dele passa disso (~25 GB/s):
o P2P entre portas downstream é comutado dentro do chip e negocia Gen4 com as
GPUs, sem nunca tocar o slot Gen3 da placa-mãe.

## Estado

**Não compilado nem testado** — escrito num ambiente sem CUDA toolkit, libnuma
ou GPUs. Revisado à mão; um bug de binding de evento CUDA a device foi corrigido
nessa revisão. Espere ajustes no primeiro build.
