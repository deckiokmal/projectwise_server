> **Apakah host Docker Anda menggunakan arsitektur CPU standar x86\_64, atau ARM64 (misalnya Raspberry Pi / Apple M1), atau yang lain?**
> Ini penting karena exit code 132 (SIGILL) sering kali muncul kalau biner di dalam container menjalankan instruksi yang tidak didukung oleh CPU host.

---

## 1. Kenapa container selalu restart

Pada `docker inspect` Anda terlihat:

```json
"State": {
  "ExitCode": 132,
  "Restarting": true,
  …
}
```

* **ExitCode 132 = 128 + 4**, di mana **4** adalah sinyal SIGILL (Illegal Instruction).
* Artinya, proses utama di dalam container (di sini: `uv run main.py`) mengalami **illegal CPU instruction** dan langsung crash.
* Karena Anda pakai `restart: unless-stopped`, Docker akan otomatis mencoba memulai ulang—menciptakan loop restart terus-menerus.

**Penyebab umum SIGILL / Exit 132**

1. **Mismatch arsitektur**: image yang Anda pull build untuk x86\_64, tapi host Anda ARM64 (atau sebaliknya).
2. **Binary “uv” / extension C** yang dikompilasi dengan instruksi CPU khusus (misal AVX, SSE) yang tidak didukung host.
3. **Kerusakan file** atau library di dalam image.

**Langkah troubleshooting**

1. Cek arsitektur host:

   ```bash
   uname -m
   ```
2. Cek arsitektur image:

   ```bash
   docker image inspect deckiokmal/projectwise_mcpserver:latest \
     --format '{{.Architecture}}/{{.Os}}'
   ```
3. Jika mismatch, rebuild atau pull image multi-arch / yang sesuai CPU Anda.

---

## Actionable Summary

1. **Periksa arsitektur**

   * Pastikan host dan image cocok (`uname -m` vs. `docker image inspect`).

2. **Perbaiki crash illegal instruction**

   * Gunakan image yang sesuai arsitektur, atau rebuild dengan base image yang kompatibel (mis. `python:3.13-slim-buster` multi-arch).

3. **Verifikasi network**

   * Setelah container stay up, cek IP dengan `docker inspect` atau `docker network inspect projectwise_projectwise-net`.

Dengan container stabil (tidak crash), Anda akan melihat `IPAddress` terisi, dan inter-container ping / HTTP internal akan berjalan normal.

Berikut langkah‐langkah terstruktur untuk memeriksa arsitektur CPU pada host Docker Anda, baik lewat sistem operasi maupun lewat Docker daemon:

---

## **Solusi**:

   * Build ulang image multi‐arch (via `docker buildx`) atau
   * Pull tag khusus ARM64 (jika image upstream sudah multi‐arch)

Contoh build multi-arch dengan Buildx:

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t deckiokmal/projectwise_mcpserver:latest --push .
```

Dengan cara ini Anda jadi tahu persis arsitektur host dan image, sehingga bisa menyesuaikan build atau memilih image yang kompatibel.

---
Dari gejala–gejalanya (ExitCode 132 → illegal instruction) di *host* Proxmox Anda, padahal arsitektur image (`amd64/linux`) sama dengan host (`x86_64`), besar kemungkinannya bukan arsitektur prosesor total mismatch, melainkan **CPU feature set** yang tidak dipaparkan ke VM. Sebagian besar wheel binari (misalnya **pyarrow**, atau bahkan *binary* Astral–UV di `uv`) di‐packaging dengan asumsi minimal dukungan instruksi SSE4.2 atau AVX, sementara VM default Proxmox (`qemu64`/`kvm64`) sering kali hanya memaparkan instruksi dasar (no SSE4.2/AVX), sehingga saat kode C/C++ (dalam pyarrow atau uv) mengeksekusi instruksi SIMD tersebut, kernel akan kirim SIGILL → proses langsung crash.

---

## 1. Kenali sinyal SIGILL (ExitCode 132)

* `132 = 128 + 4` → 4 = **SIGILL**: illegal machine instruction.
* Artinya *binary* mencoba mengeksekusi opcode CPU yang tidak dikenal oleh prosesor.

---

## 2. Verifikasi feature set di VM

Jalankan di dalam VM (atau container yang sempat hidup) sebelum crash:

```bash
grep -m1 flags /proc/cpuinfo
# atau
lscpu | grep Flags
```
hasil:
```bash
flags		: fpu de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx lm constant_tsc nopl xtopology cpuid tsc_known_freq pni ssse3 cx16 sse4_1 sse4_2 x2apic popcnt aes hypervisor lahf_lm cpuid_fault pti
```

Perhatikan apakah ada `sse4_2`, `avx`, `avx2`, dsb. Jika tidak muncul, berarti VM tidak expose instruksi tersebut.

---

## 3. Dua solusi utama

### A. Passthrough fitur CPU ke VM

1. Di Proxmox GUI → VM → Hardware → CPU →
   – **Type**: ubah ke `host` (ketimbang `kvm64`/`qemu64`).
   – Ini akan me-“forward” semua fitur CPU fisik (SSE4.2, AVX, …) ke VM.

2. Restart VM → jalankan ulang container → seharusnya **tidak** lagi crash.

### B. Gunakan wheel/library yang dikompilasi untuk baseline lebih rendah

Jika Anda tidak bisa ubah setting VM:

1. **Rebuild pyarrow dari source** tanpa SIMD:

   ```dockerfile
   RUN pip install --no-binary pyarrow pyarrow
   ```
2. **Ganti “uv” binari** dari Astral ke pure-Python Uvicorn:

   ```dockerfile
   RUN pip install uvicorn
   # lalu di CMD: ["uvicorn","main:app","--host","0.0.0.0","--port","5000"]
   ```

   Karena uvicorn via pip tidak membawa instruksi SIMD yang berat.

---

## 4. Ringkasan actionable

1. **Cek flags CPU** di VM untuk konfirmasi SSE/AVX tidak muncul.
2. **Preferred**: ubah CPU type VM ke `host` agar container bisa jalankan binary SIMD‐enabled tanpa crash.
3. **Alternatif**: rebuild dari source atau gunakan package pip–only untuk menghindari illegal instruction.

Dengan menerapkan salah satu dari dua pendekatan di atas, container Anda seharusnya **tetap hidup** (tidak restart loop) dan kemudian baru akan mendapatkan IP internal Docker seperti biasa.
