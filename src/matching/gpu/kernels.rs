pub(crate) const MAX_STR: usize = 64; // truncate names for GPU DP to keep registers/local mem bounded

// CUDA kernel source for per-pair Levenshtein (two-row DP; lengths capped to MAX_STR)
pub(crate) const LEV_KERNEL_SRC: &str = r#"
    __device__ __forceinline__ int max_i(int a, int b) { return a > b ? a : b; }
    __device__ __forceinline__ int min_i(int a, int b) { return a < b ? a : b; }

    extern "C" __global__ void lev_kernel(
        const char* a_buf, const int* a_off, const int* a_len,
        const char* b_buf, const int* b_off, const int* b_len,
        float* out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const int off_a = a_off[i]; int la = a_len[i]; if (la > (int)64) la = 64;
        const int off_b = b_off[i]; int lb = b_len[i]; if (lb > (int)64) lb = 64;
        const char* A = a_buf + off_a;
        const char* B = b_buf + off_b;
        int prev[65]; int curr[65];
        for (int j=0;j<=lb;++j) prev[j] = j;
        for (int ia=1; ia<=la; ++ia) {
            curr[0] = ia;
            char ca = A[ia-1];
            for (int jb=1; jb<=lb; ++jb) {
                int cost = (ca == B[jb-1]) ? 0 : 1;
                int del = prev[jb] + 1;
                int ins = curr[jb-1] + 1;
                int sub = prev[jb-1] + cost;
                int v = del < ins ? del : ins;
                curr[jb] = v < sub ? v : sub;
            }
            for (int jb=0; jb<=lb; ++jb) prev[jb] = curr[jb];
        }
        int dist = prev[lb];
        int ml = la > lb ? la : lb;
        float score = ml > 0 ? (1.0f - ((float)dist / (float)ml)) * 100.0f : 100.0f;
        out[i] = score;
    }

    extern "C" __global__ void lev_dist_kernel(
        const char* a_buf, const int* a_off, const int* a_len,
        const char* b_buf, const int* b_off, const int* b_len,
        int* out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const int off_a = a_off[i]; int la = a_len[i]; if (la > (int)64) la = 64;
        const int off_b = b_off[i]; int lb = b_len[i]; if (lb > (int)64) lb = 64;
        const char* A = a_buf + off_a;
        const char* B = b_buf + off_b;
        int prev[65]; int curr[65];
        for (int j=0;j<=lb;++j) prev[j] = j;
        for (int ia=1; ia<=la; ++ia) {
            curr[0] = ia;
            char ca = A[ia-1];
            for (int jb=1; jb<=lb; ++jb) {
                int cost = (ca == B[jb-1]) ? 0 : 1;
                int del = prev[jb] + 1;
                int ins = curr[jb-1] + 1;
                int sub = prev[jb-1] + cost;
                int v = del < ins ? del : ins;
                curr[jb] = v < sub ? v : sub;
            }
            for (int jb=0; jb<=lb; ++jb) prev[jb] = curr[jb];
        }
        out[i] = prev[lb];
    }

    __device__ float jaro_core(const char* A, int la, const char* B, int lb) {
        if (la == 0 && lb == 0) return 1.0f;
        int match_dist = max_i(0, max_i(la, lb) / 2 - 1);
        bool a_match[64]; bool b_match[64];
        for (int i=0;i<64;++i) { a_match[i]=false; b_match[i]=false; }
        int matches = 0;
        for (int i=0;i<la; ++i) {
            int start = max_i(0, i - match_dist);
            int end = min_i(i + match_dist + 1, lb);
            for (int j=start; j<end; ++j) {
                if (b_match[j]) continue;
                if (A[i] != B[j]) continue;
                a_match[i] = true; b_match[j] = true; ++matches; break;
            }
        }
        if (matches == 0) return 0.0f;
        int k = 0; int trans = 0;
        for (int i=0;i<la; ++i) {
            if (!a_match[i]) continue;
            while (k < lb && !b_match[k]) ++k;
            if (k < lb && A[i] != B[k]) ++trans;
            ++k;
        }
        float m = (float)matches;
        float j1 = m / la;
        float j2 = m / lb;
        float j3 = (m - trans/2.0f) / m;
        return (j1 + j2 + j3) / 3.0f;
    }



    extern "C" __global__ void jaro_kernel(
        const char* a_buf, const int* a_off, const int* a_len,
        const char* b_buf, const int* b_off, const int* b_len,
        float* out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        int la = a_len[i]; if (la > 64) la = 64;
        int lb = b_len[i]; if (lb > 64) lb = 64;
        const char* A = a_buf + a_off[i];
        const char* B = b_buf + b_off[i];
        float j = jaro_core(A, la, B, lb);
        out[i] = j * 100.0f;
    }

    extern "C" __global__ void jw_kernel(
        const char* a_buf, const int* a_off, const int* a_len,
        const char* b_buf, const int* b_off, const int* b_len,
        float* out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        int la = a_len[i]; if (la > 64) la = 64;
        int lb = b_len[i]; if (lb > 64) lb = 64;
        const char* A = a_buf + a_off[i];
        const char* B = b_buf + b_off[i];
        float j = jaro_core(A, la, B, lb);
        int l = 0; int maxp = 4;
        for (int k=0; k<min_i(min_i(la, lb), maxp); ++k) { if (A[k] == B[k]) ++l; else break; }
        float p = 0.1f;
        float jw = j + l * p * (1.0f - j);
        out[i] = jw * 100.0f;
    }

    extern "C" __global__ void max3_kernel(const float* a, const float* b, const float* c, float* out, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            float m = a[i];
            if (b[i] > m) m = b[i];
            if (c[i] > m) m = c[i];
            out[i] = m;
        }
    }

    extern "C" __global__ void fuzzy_gate_kernel(
        const char* a_buf, const int* a_off, const int* a_len,
        const char* b_buf, const int* b_off, const int* b_len,
        const unsigned char* mp_eq,
        unsigned char* keep_out,
        int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const int off_a = a_off[i]; int la = a_len[i]; if (la > (int)64) la = 64;
        const int off_b = b_off[i]; int lb = b_len[i]; if (lb > (int)64) lb = 64;
        const char* A = a_buf + off_a;
        const char* B = b_buf + off_b;

        int prev[65]; int curr[65];
        for (int j=0;j<=lb;++j) prev[j] = j;
        for (int ia=1; ia<=la; ++ia) {
            curr[0] = ia;
            char ca = A[ia-1];
            for (int jb=1; jb<=lb; ++jb) {
                int cost = (ca == B[jb-1]) ? 0 : 1;
                int del = prev[jb] + 1;
                int ins = curr[jb-1] + 1;
                int sub = prev[jb-1] + cost;
                int v = del < ins ? del : ins;
                curr[jb] = v < sub ? v : sub;
            }
            for (int jb=0; jb<=lb; ++jb) prev[jb] = curr[jb];
        }
        int dist = prev[lb];
        int ml = la > lb ? la : lb;
        float lev = ml > 0 ? (1.0f - ((float)dist / (float)ml)) * 100.0f : 100.0f;

        float j = jaro_core(A, la, B, lb);
        int l = 0; int maxp = 4;
        for (int k=0; k<min_i(min_i(la, lb), maxp); ++k) { if (A[k] == B[k]) ++l; else break; }
        float jw = (j + l * 0.1f * (1.0f - j)) * 100.0f;

        unsigned char keep;
        if (mp_eq[i]) {
            keep = (lev >= 84.0f || jw >= 84.0f) ? 1u : 0u;
        } else {
            keep = (lev >= 84.0f && jw >= 84.0f) ? 1u : 0u;
        }
        keep_out[i] = keep;
    }

    extern "C" __global__ void fuzzy_gate_kernel_resident(
        const char* pool1_bytes, const int* pool1_off, const int* pool1_len,
        const char* pool2_bytes, const int* pool2_off, const int* pool2_len,
        const int* pair_outer_idx, const int* pair_inner_idx,
        const unsigned char* mp_eq,
        unsigned char* keep_out,
        int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        int oi = pair_outer_idx[i];
        int ii = pair_inner_idx[i];
        const int off_a = pool1_off[oi]; int la = pool1_len[oi]; if (la > (int)64) la = 64;
        const int off_b = pool2_off[ii]; int lb = pool2_len[ii]; if (lb > (int)64) lb = 64;
        const char* A = pool1_bytes + off_a;
        const char* B = pool2_bytes + off_b;

        int prev[65]; int curr[65];
        for (int j=0;j<=lb;++j) prev[j] = j;
        for (int ia=1; ia<=la; ++ia) {
            curr[0] = ia;
            char ca = A[ia-1];
            for (int jb=1; jb<=lb; ++jb) {
                int cost = (ca == B[jb-1]) ? 0 : 1;
                int del = prev[jb] + 1;
                int ins = curr[jb-1] + 1;
                int sub = prev[jb-1] + cost;
                int v = del < ins ? del : ins;
                curr[jb] = v < sub ? v : sub;
            }
            for (int jb=0; jb<=lb; ++jb) prev[jb] = curr[jb];
        }
        int dist = prev[lb];
        int ml = la > lb ? la : lb;
        float lev = ml > 0 ? (1.0f - ((float)dist / (float)ml)) * 100.0f : 100.0f;

        float j = jaro_core(A, la, B, lb);
        int l = 0; int maxp = 4;
        for (int k=0; k<min_i(min_i(la, lb), maxp); ++k) { if (A[k] == B[k]) ++l; else break; }
        float jw = (j + l * 0.1f * (1.0f - j)) * 100.0f;

        unsigned char keep;
        if (mp_eq[i]) {
            keep = (lev >= 84.0f || jw >= 84.0f) ? 1u : 0u;
        } else {
            keep = (lev >= 84.0f && jw >= 84.0f) ? 1u : 0u;
        }
        keep_out[i] = keep;
    }
    "#;

// --- GPU FNV-1a 64-bit hash kernel and hashing helpers (module scope) ---
pub(crate) const FNV_KERNEL_SRC: &str = r#"
    extern "C" __global__ void fnv1a64_kernel(
        const char* buf, const int* off, const int* len,
        unsigned long long* out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        unsigned long long hash = 0xcbf29ce484222325ULL;
        const unsigned long long prime = 0x100000001b3ULL;
        const char* s = buf + off[i];
        int L = len[i];
        #pragma unroll 1
        for (int j = 0; j < L; ++j) {
            hash ^= (unsigned long long)(unsigned char)s[j];
            hash *= prime;
        }
        out[i] = hash;
    }
    "#;
